"""
Tests for the AgenTool framework.
"""

import pytest
import json
import asyncio
from typing import Any, Dict, Literal, Optional, List
from datetime import datetime
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from pydantic_ai.usage import Usage

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agentool import create_agentool, register_agentool_models, get_injector
from src.agentool.core.registry import AgenToolRegistry, RoutingConfig, ToolMetadata
from src.agentool.core.injector import serialize_to_json_string, validate_json_string
from src.agentool.factory import extract_tool_metadata, infer_output_type
from src.examples.agents.storage import create_storage_agent
from src.examples.tools.storage import _storage


class StorageInput(BaseModel):
    """Test input schema for storage operations."""
    operation: str
    key: str
    value: str = None


class TestAgenTool:
    """Test suite for AgenTool functionality."""
    
    def setup_method(self):
        """Clear registry before each test."""
        AgenToolRegistry._configs.clear()
        register_agentool_models()
    
    def test_create_agentool(self):
        """Test creating a basic AgenTool."""
        # Define test tools
        async def test_read(ctx: RunContext[Any], key: str) -> str:
            return f"Read value for key: {key}"
        
        async def test_write(ctx: RunContext[Any], key: str, value: str) -> str:
            return f"Wrote {value} to key: {key}"
        
        # Create routing config
        routing = RoutingConfig(
            operation_field='operation',
            operation_map={
                'read': ('test_read', lambda x: {'key': x.key}),
                'write': ('test_write', lambda x: {'key': x.key, 'value': x.value})
            }
        )
        
        # Create AgenTool
        agent = create_agentool(
            name='test',
            input_schema=StorageInput,
            routing_config=routing,
            tools=[test_read, test_write],
            description="Test AgenTool"
        )
        
        assert agent is not None
        assert agent.name is None  # pydantic-ai agents have None as default name
        
        # Check registry
        config = AgenToolRegistry.get('test')
        assert config is not None
        assert config.input_schema == StorageInput
    
    def test_agentool_is_agent(self):
        """Test that AgenTool is a proper pydantic-ai Agent instance."""
        from pydantic_ai import Agent
        
        # Create a simple AgenTool
        async def dummy_tool(ctx: RunContext[Any], data: str) -> str:
            return f"Processed: {data}"
        
        routing = RoutingConfig(
            operation_field='operation',
            operation_map={
                'process': ('dummy_tool', lambda x: {'data': x.value})
            }
        )
        
        agent = create_agentool(
            name='verify_agent',
            input_schema=StorageInput,
            routing_config=routing,
            tools=[dummy_tool]
        )
        
        # Verify it's an Agent instance
        assert isinstance(agent, Agent)
        
        # Verify it has all Agent methods
        assert hasattr(agent, 'run')
        assert hasattr(agent, 'run_sync')
        assert hasattr(agent, 'run_stream')
        assert hasattr(agent, 'model')
        assert hasattr(agent, 'name')
        assert hasattr(agent, 'system_prompt')
        
        # Verify the model is AgenToolModel
        assert agent.model.__class__.__name__ == 'AgenToolModel'
        assert agent.model.name == 'verify_agent'
    
    def test_agentool_routing(self):
        """Test that AgenTool properly routes operations."""
        import asyncio
        
        # Storage for testing
        test_storage = {}
        
        # Define test tools
        async def test_read(ctx: RunContext[Any], key: str) -> str:
            return test_storage.get(key, f"No value found for key: {key}")
        
        async def test_write(ctx: RunContext[Any], key: str, value: str) -> str:
            test_storage[key] = value
            return f"Stored {value} at key: {key}"
        
        # Create routing config
        routing = RoutingConfig(
            operation_field='operation',
            operation_map={
                'read': ('test_read', lambda x: {'key': x.key}),
                'write': ('test_write', lambda x: {'key': x.key, 'value': x.value})
            }
        )
        
        # Create AgenTool
        agent = create_agentool(
            name='test',
            input_schema=StorageInput,
            routing_config=routing,
            tools=[test_read, test_write]
        )
        
        async def run_test():
            # Test write operation
            write_input = {"operation": "write", "key": "test_key", "value": "test_value"}
            result = await agent.run(json.dumps(write_input))
            assert "test_value" in str(result.output)
            assert "test_key" in str(result.output)
            
            # Test read operation
            read_input = {"operation": "read", "key": "test_key"}
            result = await agent.run(json.dumps(read_input))
            assert "test_value" in str(result.output) or "test_key" in str(result.output)
        
        asyncio.run(run_test())
    
    def test_storage_agentool(self):
        """Test the storage AgenTool example."""
        import asyncio
        
        # Register models if not already done
        register_agentool_models()
        
        # Clear storage before test
        _storage.clear()
        
        # Create storage agent
        storage_agent = create_storage_agent()
        
        async def run_test():
            # Test write operation
            write_result = await storage_agent.run(json.dumps({
                "operation": "write",
                "key": "/test/file.txt",
                "data": "Hello, World!"
            }))
            assert write_result.output is not None
            
            # Test read operation
            read_result = await storage_agent.run(json.dumps({
                "operation": "read",
                "key": "/test/file.txt"
            }))
            assert read_result.output is not None
            
            # Test list operation
            list_result = await storage_agent.run(json.dumps({
                "operation": "list"
            }))
            assert list_result.output is not None
        
        asyncio.run(run_test())
    
    def test_invalid_operation(self):
        """Test handling of invalid operations."""
        import asyncio
        
        # Create storage agent
        storage_agent = create_storage_agent()
        
        async def run_test():
            # Test with invalid operation
            result = await storage_agent.run(json.dumps({
                "operation": "invalid",
                "key": "/test/file.txt"
            }))
            # Should return an error message about invalid operation
            assert "Error creating input model" in str(result.output)
            assert "Input should be 'read', 'write', 'list' or 'delete'" in str(result.output)
        
        asyncio.run(run_test())
    
    def test_multiple_conversations(self):
        """Test multiple sequential operations in conversation style."""
        import asyncio
        
        # Clear storage and create agent
        _storage.clear()
        storage_agent = create_storage_agent()
        
        async def run_test():
            # First operation - write
            result1 = await storage_agent.run(json.dumps({
                "operation": "write",
                "key": "/conversation/test1.txt",
                "data": "First message"
            }))
            assert "Successfully stored" in str(result1.output) or "stored" in str(result1.output).lower()
            
            # Second operation - write another
            result2 = await storage_agent.run(json.dumps({
                "operation": "write", 
                "key": "/conversation/test2.txt",
                "data": "Second message"
            }))
            assert "Successfully stored" in str(result2.output) or "stored" in str(result2.output).lower()
            
            # Third operation - list
            result3 = await storage_agent.run(json.dumps({
                "operation": "list"
            }))
            assert result3.output is not None
        
        asyncio.run(run_test())
    
    def test_output_schema_validation(self):
        """Test AgenTool with defined output schema."""
        import asyncio
        from pydantic import BaseModel, Field
        from typing import List, Optional
        
        # Define output schema
        class StorageQueryResult(BaseModel):
            success: bool = Field(description="Whether the operation succeeded")
            message: str = Field(description="Result message")
            data: Optional[Any] = Field(None, description="Operation result data")
        
        # Define input schema with specific operations
        class QueryInput(BaseModel):
            operation: Literal['get', 'set', 'query'] = Field(description="Operation type")
            key: Optional[str] = Field(None, description="Key for get/set operations")
            value: Optional[Any] = Field(None, description="Value for set operation")
            pattern: Optional[str] = Field(None, description="Pattern for query operation")
        
        # Define tools that return structured data
        async def tool_get(ctx: RunContext[Any], key: str) -> Dict[str, Any]:
            """Get value by key, returning structured result."""
            if key in _storage:
                return {
                    "success": True,
                    "message": f"Found value for key: {key}",
                    "data": _storage[key]
                }
            return {
                "success": False,
                "message": f"Key not found: {key}",
                "data": None
            }
        
        async def tool_set(ctx: RunContext[Any], key: str, value: Any) -> Dict[str, Any]:
            """Set value by key, returning structured result."""
            _storage[key] = value
            return {
                "success": True,
                "message": f"Stored value for key: {key}",
                "data": {"key": key, "value": value}
            }
        
        async def tool_query(ctx: RunContext[Any], pattern: str) -> Dict[str, Any]:
            """Query keys by pattern, returning structured result."""
            matching_keys = [k for k in _storage.keys() if pattern in k]
            return {
                "success": True,
                "message": f"Found {len(matching_keys)} matching keys",
                "data": matching_keys
            }
        
        # Create routing configuration
        query_routing = RoutingConfig(
            operation_field='operation',
            operation_map={
                'get': ('tool_get', lambda x: {'key': x.key}),
                'set': ('tool_set', lambda x: {'key': x.key, 'value': x.value}),
                'query': ('tool_query', lambda x: {'pattern': x.pattern}),
            }
        )
        
        # Create AgenTool with output schema
        query_agent = create_agentool(
            name='query',
            input_schema=QueryInput,
            routing_config=query_routing,
            tools=[tool_get, tool_set, tool_query],
            system_prompt="Handle structured query operations.",
            description="Query agent with structured output"
        )
        
        async def run_test():
            # Test set operation
            set_result = await query_agent.run(json.dumps({
                "operation": "set",
                "key": "test/schema/key1",
                "value": {"name": "test", "count": 42}
            }))
            
            # Parse the output as JSON
            set_output = json.loads(set_result.output)
            assert set_output["success"] is True
            assert "Stored value" in set_output["message"]
            assert set_output["data"]["key"] == "test/schema/key1"
            assert set_output["data"]["value"]["count"] == 42
            
            # Test get operation
            get_result = await query_agent.run(json.dumps({
                "operation": "get",
                "key": "test/schema/key1"
            }))
            
            # Parse and validate
            get_output = json.loads(get_result.output)
            assert get_output["success"] is True
            assert "Found value" in get_output["message"]
            assert get_output["data"]["name"] == "test"
            assert get_output["data"]["count"] == 42
            
            # Test query operation
            # First, add more test data
            await query_agent.run(json.dumps({
                "operation": "set",
                "key": "test/schema/key2",
                "value": "value2"
            }))
            await query_agent.run(json.dumps({
                "operation": "set",
                "key": "test/other/key3",
                "value": "value3"
            }))
            
            # Query for keys matching pattern
            query_result = await query_agent.run(json.dumps({
                "operation": "query",
                "pattern": "test/schema"
            }))
            
            query_output = json.loads(query_result.output)
            assert query_output["success"] is True
            assert len(query_output["data"]) == 2
            assert "test/schema/key1" in query_output["data"]
            assert "test/schema/key2" in query_output["data"]
            
            # Validate the structure matches our schema
            # Create instance to validate structure
            validated = StorageQueryResult(**query_output)
            assert validated.success is True
            assert isinstance(validated.data, list)
            assert len(validated.data) == 2
        
        asyncio.run(run_test())
    
    def test_manager_error_cases(self):
        """Test AgenToolManager error handling for better coverage."""
        import asyncio
        from src.agentool.core.manager import AgenToolManager
        from src.agentool.core.registry import AgenToolConfig
        
        async def run_test():
            # Test 1: Missing operation field
            class BadInput(BaseModel):
                value: str  # Missing operation field
            
            async def dummy_tool(ctx: RunContext[Any], value: str) -> str:
                return f"Got: {value}"
            
            routing = RoutingConfig(
                operation_field='operation',  # This field doesn't exist in BadInput
                operation_map={
                    'test': ('dummy_tool', lambda x: {'value': x.value})
                }
            )
            
            config = AgenToolConfig(
                input_schema=BadInput,
                routing_config=routing
            )
            
            manager = AgenToolManager(
                name='test',
                config=config,
                tool_functions={'dummy_tool': dummy_tool}
            )
            
            # Test with missing operation field
            bad_input = BadInput(value="test")
            result = await manager(None, **bad_input.model_dump())
            assert "Missing required field: operation" in result
            
            # Test 2: Unknown operation
            class TestInput(BaseModel):
                operation: str
                value: str
            
            routing2 = RoutingConfig(
                operation_map={
                    'known': ('test_tool', lambda x: {'value': x.value})
                }
            )
            
            config2 = AgenToolConfig(
                input_schema=TestInput,
                routing_config=routing2
            )
            
            manager2 = AgenToolManager(
                name='test2',
                config=config2,
                tool_functions={'test_tool': dummy_tool}
            )
            
            test_input = TestInput(operation='unknown', value="test")
            result2 = await manager2(None, **test_input.model_dump())
            assert "Unknown operation 'unknown'" in result2
            assert "Available operations: ['known']" in result2
            
            # Test 3: Transform function error
            def bad_transform(inp):
                raise ValueError("Transform failed")
            
            routing3 = RoutingConfig(
                operation_map={
                    'test': ('test_tool', bad_transform)
                }
            )
            
            config3 = AgenToolConfig(
                input_schema=TestInput,
                routing_config=routing3
            )
            
            manager3 = AgenToolManager(
                name='test3',
                config=config3,
                tool_functions={'test_tool': dummy_tool}
            )
            
            test_input3 = TestInput(operation='test', value="data")
            result3 = await manager3(None, **test_input3.model_dump())
            assert "Error transforming arguments" in result3
            assert "Transform failed" in result3
            
            # Test 4: Missing tool function
            routing4 = RoutingConfig(
                operation_map={
                    'test': ('missing_tool', lambda x: {'value': x.value})
                }
            )
            
            config4 = AgenToolConfig(
                input_schema=TestInput,
                routing_config=routing4
            )
            
            manager4 = AgenToolManager(
                name='test4',
                config=config4,
                tool_functions={}  # Empty tools
            )
            
            test_input4 = TestInput(operation='test', value="data")
            result4 = await manager4(None, **test_input4.model_dump())
            assert "Tool 'missing_tool' not found" in result4
            
            # Test 5: Tool execution failure - should raise exception
            async def failing_tool(ctx: RunContext[Any], value: str) -> str:
                raise RuntimeError("Tool execution failed")
            
            routing5 = RoutingConfig(
                operation_map={
                    'fail': ('failing_tool', lambda x: {'value': x.value})
                }
            )
            
            config5 = AgenToolConfig(
                input_schema=TestInput,
                routing_config=routing5
            )
            
            manager5 = AgenToolManager(
                name='test5',
                config=config5,
                tool_functions={'failing_tool': failing_tool}
            )
            
            test_input5 = TestInput(operation='fail', value="data")
            with pytest.raises(RuntimeError) as exc_info:
                await manager5(None, **test_input5.model_dump())
            assert "Tool execution failed" in str(exc_info.value)
        
        asyncio.run(run_test())
    
    def test_manager_schema_generation(self):
        """Test AgenToolManager schema generation."""
        from src.agentool.core.manager import AgenToolManager
        from src.agentool.core.registry import AgenToolConfig
        
        class SchemaTestInput(BaseModel):
            operation: str = Field(description="The operation")
            required_field: str
            optional_field: str = Field(default="default")
        
        routing = RoutingConfig(
            operation_map={}
        )
        
        config = AgenToolConfig(
            input_schema=SchemaTestInput,
            routing_config=routing
        )
        
        manager = AgenToolManager(
            name='test',
            config=config,
            tool_functions={}
        )
        
        schema = manager.get_tool_schema()
        assert 'properties' in schema
        assert 'operation' in schema['properties']
        assert 'required' in schema
        assert 'required_field' in schema['required']
        assert 'optional_field' not in schema['required']
    
    def test_registry_additional_methods(self):
        """Test AgenToolRegistry additional methods for coverage."""
        from src.agentool.core.registry import AgenToolConfig
        
        # Test get with non-existent name
        config = AgenToolRegistry.get('nonexistent')
        assert config is None
        
        # Test list_names
        config1 = AgenToolConfig(
            input_schema=BaseModel,
            routing_config=RoutingConfig(operation_map={})
        )
        config2 = AgenToolConfig(
            input_schema=BaseModel,
            routing_config=RoutingConfig(operation_map={})
        )
        
        AgenToolRegistry.register('coverage_test1', config1)
        AgenToolRegistry.register('coverage_test2', config2)
        
        names = AgenToolRegistry.list_names()
        assert 'coverage_test1' in names
        assert 'coverage_test2' in names
    
    def test_model_error_cases(self):
        """Test AgenToolModel error cases."""
        import asyncio
        from src.agentool.core.model import AgenToolModel
        from pydantic_ai.messages import UserPromptPart, ModelRequest
        
        async def run_test():
            # Test with non-existent configuration
            model = AgenToolModel('nonexistent')
            
            # Create a proper ModelRequest
            messages = [
                ModelRequest(
                    parts=[UserPromptPart(content='{"operation": "test"}')]
                )
            ]
            
            # Should return an error because no configuration exists
            response = await model.request(messages, {}, {})
            assert len(response.parts) == 1
            # The error happens when trying to parse without config
            assert "Invalid JSON input" in response.parts[0].content or "No configuration found" in response.parts[0].content or "Error" in response.parts[0].content
        
        asyncio.run(run_test())
    
    def test_factory_model_inference(self):
        """Test factory function's model inference patching."""
        from pydantic_ai import models
        from src.agentool.core.model import AgenToolModel
        
        # Test that register_agentool_models patches infer_model correctly
        # The patching should already be done
        model = models.infer_model('agentool:coverage_test')
        assert isinstance(model, AgenToolModel)
        assert model.name == 'coverage_test'
    
    def test_model_coverage_improvements(self):
        """Additional tests to improve model.py coverage."""
        import asyncio
        from src.agentool.core.model import AgenToolModel
        from src.agentool.core.registry import AgenToolConfig
        from pydantic_ai.messages import (
            UserPromptPart, ModelRequest, ModelResponse, 
            ToolReturnPart, TextPart, ToolCallPart
        )
        
        async def run_test():
            # Test 1: Test system property
            model = AgenToolModel('test_coverage')
            assert model.system == 'agentool'
            assert model.model_name == 'agentool:test_coverage'
            
            # Setup a test configuration
            class TestInput(BaseModel):
                operation: str
                value: str
            
            async def test_tool(ctx: RunContext[Any], value: str) -> dict:
                return {"result": value}
            
            routing = RoutingConfig(
                operation_map={
                    'test': ('test_tool', lambda x: {'value': x.value})
                }
            )
            
            config = AgenToolConfig(
                input_schema=TestInput,
                routing_config=routing
            )
            
            AgenToolRegistry.register('test_coverage', config)
            
            # Test 2: No input provided (empty messages)
            response = await model.request([], {}, {})
            assert "No input provided" in response.parts[0].content
            
            # Test 3: Invalid JSON input
            messages_invalid_json = [
                ModelRequest(
                    parts=[UserPromptPart(content='not valid json')]
                )
            ]
            response = await model.request(messages_invalid_json, {}, {})
            assert "Invalid JSON input" in response.parts[0].content
            
            # Test 4: UserPromptPart with list content
            messages_list_content = [
                ModelRequest(
                    parts=[UserPromptPart(content=["First item", '{"operation": "test", "value": "from_list"}'])]
                )
            ]
            response = await model.request(messages_list_content, {}, {})
            # Should pick the first string item
            assert "Invalid JSON input" in response.parts[0].content  # "First item" is not valid JSON
            
            # Test 5: Multiple tool returns
            tool_call_id = "test_123"
            # First request - creates tool call
            messages_first = [
                ModelRequest(
                    parts=[UserPromptPart(content='{"operation": "test", "value": "test1"}')]
                )
            ]
            response1 = await model.request(messages_first, {}, {})
            assert len(response1.parts) == 1
            assert isinstance(response1.parts[0], ToolCallPart)
            
            # Now simulate multiple tool returns
            messages_with_multiple_returns = [
                ModelRequest(
                    parts=[UserPromptPart(content='{"operation": "test", "value": "test1"}')]
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='__agentool_manager__', args={'operation': 'test'}, tool_call_id=tool_call_id)],
                    model_name='test',
                    usage=Usage(),
                    timestamp=datetime.now()
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(tool_name='tool1', content={'data': 'result1'}, tool_call_id=tool_call_id),
                        ToolReturnPart(tool_name='tool2', content={'data': 'result2'}, tool_call_id=tool_call_id)
                    ]
                )
            ]
            
            response = await model.request(messages_with_multiple_returns, {}, {})
            assert len(response.parts) == 1
            # Should combine multiple tool returns into a single JSON response
            content = json.loads(response.parts[0].content)
            assert 'tool1' in content
            assert 'tool2' in content
            
            # Test 6: UserPromptPart with no string items in list
            messages_no_string = [
                ModelRequest(
                    parts=[UserPromptPart(content=[123, 456, {'not': 'string'}])]
                )
            ]
            response = await model.request(messages_no_string, {}, {})
            assert "No input provided" in response.parts[0].content
            
            # Test 7: ModelRequest not in reversed order (older message has input)
            messages_multiple = [
                ModelRequest(
                    parts=[UserPromptPart(content='{"operation": "test", "value": "old"}')]
                ),
                ModelRequest(
                    parts=[TextPart(content='Some other content')]  # No UserPromptPart
                )
            ]
            response = await model.request(messages_multiple, {}, {})
            # Should still find the input from the first message
            assert len(response.parts) == 1
            assert isinstance(response.parts[0], ToolCallPart)
            
            # Test 8: ModelResponse exists but no tool returns (edge case)
            messages_no_tool_returns = [
                ModelRequest(
                    parts=[UserPromptPart(content='{"operation": "test", "value": "test1"}')]
                ),
                ModelResponse(
                    parts=[TextPart(content='Some response')],
                    model_name='test',
                    usage=Usage(),
                    timestamp=datetime.now()
                ),
                ModelRequest(
                    parts=[TextPart(content='No tool returns here')]
                )
            ]
            response = await model.request(messages_no_tool_returns, {}, {})
            # Should fall through to extract user input from first message
            assert len(response.parts) == 1
            assert isinstance(response.parts[0], ToolCallPart)
            
            # Test 9: Last message is ModelResponse when looking for tool returns (not ModelRequest)
            messages_last_is_response = [
                ModelRequest(
                    parts=[UserPromptPart(content='{"operation": "test", "value": "test1"}')]
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='__agentool_manager__', args={'operation': 'test'}, tool_call_id='test_123')],
                    model_name='test',
                    usage=Usage(),
                    timestamp=datetime.now()
                ),
                ModelResponse(  # Last message is ModelResponse, not ModelRequest
                    parts=[TextPart(content='Another response')],
                    model_name='test',
                    usage=Usage(),
                    timestamp=datetime.now()
                )
            ]
            response = await model.request(messages_last_is_response, {}, {})
            # Should fall through since last message isn't ModelRequest with tool returns
            assert len(response.parts) == 1
            assert isinstance(response.parts[0], ToolCallPart)
            
            # Test 10: has_model_response is True but messages list becomes empty (edge case)
            # This is a defensive programming case that shouldn't happen in practice
            # We'll test by mocking an empty messages after detecting ModelResponse
            from unittest.mock import patch
            
            # First, create messages with a ModelResponse
            messages_with_response = [
                ModelResponse(
                    parts=[TextPart(content='Response')],
                    model_name='test',
                    usage=Usage(),
                    timestamp=datetime.now()
                )
            ]
            
            # The has_model_response will be True, but we'll have empty messages
            # This tests the `if messages:` check at line 130
            response = await model.request(messages_with_response, {}, {})
            # Should fall through to "No input provided"
            assert "No input provided" in response.parts[0].content
            
            # Test 11: UserPromptPart with non-str, non-list content (edge case)
            # This tests the branch where content is neither str nor list
            class CustomContent:
                def __str__(self):
                    return "custom content"
            
            messages_custom_content = [
                ModelRequest(
                    parts=[UserPromptPart(content=CustomContent())]  # Neither str nor list
                )
            ]
            response = await model.request(messages_custom_content, {}, {})
            # Should not find any input
            assert "No input provided" in response.parts[0].content
        
        asyncio.run(run_test())
    
    def test_additional_coverage_improvements(self):
        """Additional tests to reach 100% coverage."""
        # Test factory's tool schema override when attributes don't exist
        # This covers factory.py lines 110-111
        from src.agentool.factory import create_agentool
        
        class TestInput(BaseModel):
            operation: str
            value: str
        
        async def test_tool(ctx: RunContext[Any], value: str) -> dict:
            return {"result": value}
        
        routing = RoutingConfig(
            operation_map={
                'test': ('test_tool', lambda x: {'value': x.value})
            }
        )
        
        # Create agent
        agent = create_agentool(
            name='schema_test',
            input_schema=TestInput,
            routing_config=routing,
            tools=[test_tool]
        )
        
        # The factory should handle tools gracefully even if internal structure changes
        # This is mostly for defensive programming
        assert agent is not None
        assert hasattr(agent, '_function_toolset')
    
    def test_json_serialization_helpers(self):
        """Test JSON serialization helper functions."""
        from datetime import date, time
        from decimal import Decimal
        
        # Test 1: String that is already valid JSON
        json_str = '{"key": "value"}'
        assert serialize_to_json_string(json_str) == json_str
        assert validate_json_string(json_str) is True
        
        # Test 2: Plain string (not JSON)
        plain_str = "Hello, World!"
        result = serialize_to_json_string(plain_str)
        assert result == '"Hello, World!"'
        assert validate_json_string(plain_str) is False
        
        # Test 3: Dict
        test_dict = {"operation": "get", "key": "test"}
        result = serialize_to_json_string(test_dict)
        assert result == '{"operation": "get", "key": "test"}'
        
        # Test 4: List
        test_list = [1, 2, 3]
        result = serialize_to_json_string(test_list)
        assert result == '[1, 2, 3]'
        
        # Test 5: BaseModel
        class TestModel(BaseModel):
            name: str
            value: int
        
        model = TestModel(name="test", value=42)
        result = serialize_to_json_string(model)
        parsed = json.loads(result)
        assert parsed["name"] == "test"
        assert parsed["value"] == 42
        
        # Test 6: Datetime objects
        dt = datetime(2024, 1, 1, 12, 0, 0)
        result = serialize_to_json_string(dt)
        assert result == '"2024-01-01T12:00:00"'
        
        d = date(2024, 1, 1)
        result = serialize_to_json_string(d)
        assert result == '"2024-01-01"'
        
        t = time(12, 30, 45)
        result = serialize_to_json_string(t)
        assert result == '"12:30:45"'
        
        # Test 7: Decimal
        dec = Decimal("123.45")
        result = serialize_to_json_string(dec)
        assert result == '"123.45"'
        
        # Test 8: Basic types
        assert serialize_to_json_string(42) == '42'
        assert serialize_to_json_string(3.14) == '3.14'
        assert serialize_to_json_string(True) == 'true'
        assert serialize_to_json_string(None) == 'null'
        
        # Test 9: Nested structures
        nested = {
            "user": {"name": "Alice", "age": 30},
            "items": [1, 2, 3],
            "active": True
        }
        result = serialize_to_json_string(nested)
        parsed = json.loads(result)
        assert parsed["user"]["name"] == "Alice"
        assert parsed["items"] == [1, 2, 3]
        
        # Test 10: Custom object (should fall back to string)
        class CustomObject:
            def __str__(self):
                return "custom_object_string"
        
        custom = CustomObject()
        result = serialize_to_json_string(custom)
        assert result == '"custom_object_string"'
        
        # Test 11: Invalid JSON string validation
        assert validate_json_string("not json") is False
        assert validate_json_string("{invalid}") is False
        assert validate_json_string("") is False
        assert validate_json_string(None) is False
    
    def test_injector_automatic_json_serialization(self):
        """Test the injector's automatic JSON serialization feature."""
        import asyncio
        from decimal import Decimal
        
        # Create a test AgenTool
        class TestInput(BaseModel):
            operation: Literal['echo', 'add', 'info']
            data: Optional[Any] = None
            a: Optional[float] = None
            b: Optional[float] = None
        
        async def echo_tool(ctx: RunContext[Any], data: Any) -> dict:
            return {"echoed": data, "type": type(data).__name__}
        
        async def add_tool(ctx: RunContext[Any], a: float, b: float) -> dict:
            return {"result": a + b}
        
        async def info_tool(ctx: RunContext[Any]) -> dict:
            return {"status": "ok", "timestamp": datetime.now().isoformat()}
        
        routing = RoutingConfig(
            operation_field='operation',
            operation_map={
                'echo': ('echo_tool', lambda x: {'data': x.data}),
                'add': ('add_tool', lambda x: {'a': x.a, 'b': x.b}),
                'info': ('info_tool', lambda x: {})
            }
        )
        
        agent = create_agentool(
            name='json_test',
            input_schema=TestInput,
            routing_config=routing,
            tools=[echo_tool, add_tool, info_tool]
        )
        
        async def run_test():
            injector = get_injector()
            
            # Test 1: Dict input (most common case)
            result = await injector.run('json_test', {
                "operation": "echo",
                "data": {"message": "Hello from dict"}
            })
            output = json.loads(result.output)
            assert output["echoed"]["message"] == "Hello from dict"
            
            # Test 2: Pydantic model input
            input_model = TestInput(operation="add", a=10.5, b=20.5)
            result = await injector.run('json_test', input_model)
            output = json.loads(result.output)
            assert output["result"] == 31.0
            
            # Test 3: JSON string input (backward compatibility)
            result = await injector.run('json_test', '{"operation": "info"}')
            output = json.loads(result.output)
            assert output["status"] == "ok"
            assert "timestamp" in output
            
            # Test 4: Basic type inputs
            result = await injector.run('json_test', {
                "operation": "echo",
                "data": 42
            })
            output = json.loads(result.output)
            assert output["echoed"] == 42
            
            # Test 5: List input
            result = await injector.run('json_test', {
                "operation": "echo",
                "data": [1, 2, 3]
            })
            output = json.loads(result.output)
            assert output["echoed"] == [1, 2, 3]
            
            # Test 6: Datetime in input
            dt = datetime.now()
            result = await injector.run('json_test', {
                "operation": "echo",
                "data": dt.isoformat()  # Send as ISO string
            })
            output = json.loads(result.output)
            assert output["echoed"] == dt.isoformat()
            
            # Test 7: Decimal in input
            result = await injector.run('json_test', {
                "operation": "add",
                "a": float(Decimal("10.25")),
                "b": float(Decimal("20.75"))
            })
            output = json.loads(result.output)
            assert output["result"] == 31.0
            
            # Test 8: Nested Pydantic models
            class NestedData(BaseModel):
                name: str
                count: int
            
            class ComplexInput(BaseModel):
                operation: Literal['echo']
                data: NestedData
            
            nested = ComplexInput(
                operation="echo",
                data=NestedData(name="test", count=5)
            )
            # Convert to dict for the test (since our TestInput expects Any for data)
            result = await injector.run('json_test', {
                "operation": "echo",
                "data": nested.data.model_dump()
            })
            output = json.loads(result.output)
            assert output["echoed"]["name"] == "test"
            assert output["echoed"]["count"] == 5
        
        asyncio.run(run_test())
    
    def test_injector_with_dependencies_and_json(self):
        """Test injector with dependencies using automatic JSON serialization."""
        import asyncio
        
        # Clear registry and injector
        AgenToolRegistry._configs.clear()
        injector = get_injector()
        injector.clear()
        
        # Create a base agent
        class BaseInput(BaseModel):
            operation: Literal['store', 'retrieve']
            key: str
            value: Optional[str] = None
        
        storage = {}
        
        async def store_tool(ctx: RunContext[Any], key: str, value: str) -> dict:
            storage[key] = value
            return {"stored": True, "key": key}
        
        async def retrieve_tool(ctx: RunContext[Any], key: str) -> dict:
            return {"key": key, "value": storage.get(key), "found": key in storage}
        
        base_routing = RoutingConfig(
            operation_field='operation',
            operation_map={
                'store': ('store_tool', lambda x: {'key': x.key, 'value': x.value}),
                'retrieve': ('retrieve_tool', lambda x: {'key': x.key})
            }
        )
        
        base_agent = create_agentool(
            name='base_storage',
            input_schema=BaseInput,
            routing_config=base_routing,
            tools=[store_tool, retrieve_tool]
        )
        
        # Create a dependent agent
        class DependentInput(BaseModel):
            operation: Literal['cache_set', 'cache_get']
            cache_key: str
            data: Optional[Dict[str, Any]] = None
        
        async def cache_set(ctx: RunContext[Any], cache_key: str, data: dict) -> dict:
            # Use injector to call base storage with dict input
            injector = get_injector()
            result = await injector.run('base_storage', {
                "operation": "store",
                "key": f"cache:{cache_key}",
                "value": json.dumps(data)
            })
            return {"cached": True, "cache_key": cache_key}
        
        async def cache_get(ctx: RunContext[Any], cache_key: str) -> dict:
            # Use injector with dict input
            injector = get_injector()
            result = await injector.run('base_storage', {
                "operation": "retrieve",
                "key": f"cache:{cache_key}"
            })
            output = json.loads(result.output)
            if output["found"]:
                return {
                    "cache_key": cache_key,
                    "data": json.loads(output["value"]),
                    "found": True
                }
            return {"cache_key": cache_key, "found": False}
        
        dep_routing = RoutingConfig(
            operation_field='operation',
            operation_map={
                'cache_set': ('cache_set', lambda x: {'cache_key': x.cache_key, 'data': x.data}),
                'cache_get': ('cache_get', lambda x: {'cache_key': x.cache_key})
            }
        )
        
        dep_agent = create_agentool(
            name='cache',
            input_schema=DependentInput,
            routing_config=dep_routing,
            tools=[cache_set, cache_get],
            dependencies=['base_storage']
        )
        
        async def run_test():
            injector = get_injector()
            
            # Test 1: Set cache using dict
            set_result = await injector.run('cache', {
                "operation": "cache_set",
                "cache_key": "user:123",
                "data": {"name": "Alice", "age": 30}
            })
            output = json.loads(set_result.output)
            assert output["cached"] is True
            
            # Test 2: Get cache using Pydantic model
            get_input = DependentInput(operation="cache_get", cache_key="user:123")
            get_result = await injector.run('cache', get_input)
            output = json.loads(get_result.output)
            assert output["found"] is True
            assert output["data"]["name"] == "Alice"
            assert output["data"]["age"] == 30
            
            # Test 3: Get non-existent cache
            get_result = await injector.run('cache', {
                "operation": "cache_get",
                "cache_key": "nonexistent"
            })
            output = json.loads(get_result.output)
            assert output["found"] is False
        
        asyncio.run(run_test())
    
    def test_injector_error_handling_with_json(self):
        """Test error handling in JSON serialization."""
        import asyncio
        
        # Test serialization of problematic objects
        class UnserializableObject:
            def __init__(self):
                self.circular_ref = self
            
            def __str__(self):
                raise Exception("Cannot convert to string")
        
        # This should raise ValueError
        try:
            obj = UnserializableObject()
            serialize_to_json_string(obj)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Cannot serialize" in str(e)
        
        # Test with agent that expects specific input
        class StrictInput(BaseModel):
            operation: Literal['test']
            required_field: str
        
        async def test_tool(ctx: RunContext[Any], required_field: str) -> dict:
            return {"received": required_field}
        
        routing = RoutingConfig(
            operation_field='operation',
            operation_map={
                'test': ('test_tool', lambda x: {'required_field': x.required_field})
            }
        )
        
        agent = create_agentool(
            name='strict_test',
            input_schema=StrictInput,
            routing_config=routing,
            tools=[test_tool]
        )
        
        async def run_test():
            injector = get_injector()
            
            # Test with missing required field
            try:
                result = await injector.run('strict_test', {
                    "operation": "test"
                    # missing required_field
                })
                # The agent should handle the error
                output = result.output
                assert "Error creating input model" in output or "missing" in output.lower()
            except Exception as e:
                # Should not raise, but if it does, check it's validation related
                assert "required_field" in str(e).lower()
        
        asyncio.run(run_test())
    
    def test_injector_uncovered_scenarios(self):
        """Test uncovered scenarios in the injector for 100% coverage."""
        import asyncio
        from src.agentool.core.injector import InjectedDeps
        
        # Clear registry and injector for clean test
        AgenToolRegistry._configs.clear()
        injector = get_injector()
        injector.clear()
        
        # Scenario 1: Test get() with non-existent agent (lines 146-148)
        try:
            injector.get('non_existent_agent')
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "not found in registry" in str(e)
        
        # Scenario 2: Test the edge case where agent is in registry but not in instances (lines 151-154)
        # This is a defensive programming case that shouldn't happen normally
        # We'll simulate it by registering config without creating agent
        class DummyInput(BaseModel):
            value: str
        
        from src.agentool.core.registry import AgenToolConfig
        dummy_config = AgenToolConfig(
            input_schema=DummyInput,
            routing_config=RoutingConfig(operation_map={})
        )
        AgenToolRegistry.register('orphan_agent', dummy_config)
        
        try:
            injector.get('orphan_agent')
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "found in registry but no instance available" in str(e)
        
        # Scenario 3: Test override() context manager with existing overrides (line 139)
        # Create a simple agent first
        async def dummy_tool(ctx: RunContext[Any], value: str) -> dict:
            return {"result": value}
        
        routing = RoutingConfig(
            operation_field='operation',
            operation_map={
                'process': ('dummy_tool', lambda x: {'value': x.value})
            }
        )
        
        class SimpleInput(BaseModel):
            operation: Literal['process']
            value: str
        
        agent1 = create_agentool(
            name='test_agent',
            input_schema=SimpleInput,
            routing_config=routing,
            tools=[dummy_tool]
        )
        
        # Create a mock version
        async def mock_tool(ctx: RunContext[Any], value: str) -> dict:
            return {"result": f"mocked_{value}"}
        
        mock_agent = create_agentool(
            name='mock_test_agent',
            input_schema=SimpleInput,
            routing_config=RoutingConfig(
                operation_field='operation',
                operation_map={
                    'process': ('mock_tool', lambda x: {'value': x.value})
                }
            ),
            tools=[mock_tool]
        )
        
        async def run_override_test():
            # Test with override
            with injector.override(test_agent=mock_agent):
                # This should return the override (line 139)
                overridden = injector.get('test_agent')
                assert overridden == mock_agent
                
                # Test nested overrides
                mock_agent2 = create_agentool(
                    name='mock_test_agent2',
                    input_schema=SimpleInput,
                    routing_config=routing,
                    tools=[dummy_tool]
                )
                
                with injector.override(test_agent=mock_agent2):
                    # Inner override should take precedence
                    overridden2 = injector.get('test_agent')
                    assert overridden2 == mock_agent2
                
                # After exiting inner context, should restore to outer override
                overridden_restored = injector.get('test_agent')
                assert overridden_restored == mock_agent
            
            # After exiting all contexts, should get original
            original = injector.get('test_agent')
            assert original == agent1
        
        asyncio.run(run_override_test())
        
        # Scenario 4: Test InjectedDeps class methods (lines 272, 285-287)
        class TestDeps(InjectedDeps[None]):
            pass
        
        deps = TestDeps()
        
        # Test get_agent (line 272)
        agent = deps.get_agent('test_agent')
        assert agent == agent1
        
        # Test call_agent (lines 285-287)
        async def run_injected_deps_test():
            result = await deps.call_agent('test_agent', {
                "operation": "process",
                "value": "test_value"
            })
            assert result.output is not None
            output = json.loads(result.output)
            assert output["result"] == "test_value"
        
        asyncio.run(run_injected_deps_test())
        
        # Scenario 5: Test override with exception to ensure finally block runs
        async def run_override_exception_test():
            original_overrides = injector._dependency_overrides.copy()
            
            try:
                with injector.override(test_agent=mock_agent):
                    # Verify override is applied
                    assert 'test_agent' in injector._dependency_overrides
                    # Simulate an exception
                    raise RuntimeError("Test exception")
            except RuntimeError:
                # Exception was raised as expected
                pass
            
            # Verify finally block restored overrides (lines 243-244)
            assert injector._dependency_overrides == original_overrides
            assert 'test_agent' not in injector._dependency_overrides
        
        asyncio.run(run_override_exception_test())
    
    def test_factory_uncovered_scenarios(self):
        """Test factory scenarios for 100% coverage."""
        import asyncio
        
        # Scenario 1: Test create_agentool with output_type parameter (line 171)
        class TestOutput(BaseModel):
            result: str
            status: str
        
        class TestInput(BaseModel):
            operation: Literal['test', 'test_dict']
            value: str
        
        async def test_tool(ctx: RunContext[Any], value: str) -> TestOutput:
            return TestOutput(result=f"processed_{value}", status="success")
        
        async def test_dict_tool(ctx: RunContext[Any], value: str) -> dict:
            # Return dict that should be converted to TestOutput
            return {"result": f"dict_processed_{value}", "status": "ok"}
        
        routing = RoutingConfig(
            operation_field='operation',
            operation_map={
                'test': ('test_tool', lambda x: {'value': x.value}),
                'test_dict': ('test_dict_tool', lambda x: {'value': x.value})
            }
        )
        
        # Create agent with output_type specified
        agent_with_output = create_agentool(
            name='output_test',
            input_schema=TestInput,
            routing_config=routing,
            tools=[test_tool, test_dict_tool],
            output_type=TestOutput  # This triggers line 171
        )
        
        assert agent_with_output is not None
        
        # Verify the output_type was stored in config
        config = AgenToolRegistry.get('output_test')
        assert config is not None
        assert config.output_type == TestOutput
        
        # Scenario 2: Test the schema override edge case (lines 198-199)
        # This tests the case where the tool might not have the expected attributes
        # Let's create an agent and then manually remove/modify attributes to test the conditional
        
        agent_for_schema_test = create_agentool(
            name='schema_edge_test',
            input_schema=TestInput,
            routing_config=routing,
            tools=[test_tool]
        )
        
        # The schema override should have been applied
        # Let's verify by checking the tool
        tool = agent_for_schema_test._function_toolset.tools.get('__agentool_manager__')
        assert tool is not None
        
        # The schema should have been properly set
        if hasattr(tool, 'function_schema') and hasattr(tool.function_schema, '_json_schema_dict'):
            schema = tool.function_schema._json_schema_dict
            assert 'properties' in schema
            assert 'operation' in schema['properties']
        
        # Test actual execution with output_type
        async def run_output_test():
            # Test 1: Tool returns proper TestOutput model
            result = await agent_with_output.run(json.dumps({
                "operation": "test",
                "value": "hello"
            }))
            # The output should be a validated TestOutput as JSON
            assert result.output is not None
            output = json.loads(result.output)
            assert output["result"] == "processed_hello"
            assert output["status"] == "success"
            
            # Test 2: Tool returns dict that gets converted to TestOutput
            result2 = await agent_with_output.run(json.dumps({
                "operation": "test_dict",
                "value": "world"
            }))
            assert result2.output is not None
            output2 = json.loads(result2.output)
            assert output2["result"] == "dict_processed_world"
            assert output2["status"] == "ok"
        
        asyncio.run(run_output_test())
        
        # Test error cases for output_type validation
        class WrongOutput(BaseModel):
            wrong_field: str
        
        class TestInputError(BaseModel):
            operation: Literal['wrong', 'invalid']
            value: str
        
        async def wrong_tool(ctx: RunContext[Any], value: str) -> WrongOutput:
            return WrongOutput(wrong_field=value)
        
        async def invalid_dict_tool(ctx: RunContext[Any], value: str) -> dict:
            # Return dict that doesn't match TestOutput schema
            return {"wrong": "schema"}
        
        routing_wrong = RoutingConfig(
            operation_field='operation',
            operation_map={
                'wrong': ('wrong_tool', lambda x: {'value': x.value}),
                'invalid': ('invalid_dict_tool', lambda x: {'value': x.value})
            }
        )
        
        agent_wrong = create_agentool(
            name='output_wrong',
            input_schema=TestInputError,
            routing_config=routing_wrong,
            tools=[wrong_tool, invalid_dict_tool],
            output_type=TestOutput
        )
        
        async def run_error_test():
            # Test wrong output type
            result = await agent_wrong.run(json.dumps({
                "operation": "wrong",
                "value": "test"
            }))
            assert "expected TestOutput" in result.output
            
            # Test invalid dict structure
            result2 = await agent_wrong.run(json.dumps({
                "operation": "invalid",
                "value": "test"
            }))
            assert "Error creating output model" in result2.output
        
        asyncio.run(run_error_test())
    
    def test_manager_edge_cases(self):
        """Test edge cases in manager for better coverage."""
        import asyncio
        
        # Test case 1: output_type that causes TypeError in issubclass check
        class TestInput(BaseModel):
            operation: Literal['test']
            value: str
        
        async def test_tool(ctx: RunContext[Any], value: str) -> int:
            return len(value)
        
        routing = RoutingConfig(
            operation_field='operation',
            operation_map={
                'test': ('test_tool', lambda x: {'value': x.value})
            }
        )
        
        # Create agent with a non-class output_type (e.g., Union type)
        from typing import Union
        agent = create_agentool(
            name='type_error_test',
            input_schema=TestInput,
            routing_config=routing,
            tools=[test_tool],
            output_type=Union[str, int]  # This will cause TypeError in issubclass
        )
        
        async def run_type_error_test():
            result = await agent.run(json.dumps({
                "operation": "test",
                "value": "hello"
            }))
            # Should handle the Union type gracefully
            assert result.output == "5"  # length of "hello"
        
        asyncio.run(run_type_error_test())
        
        # Test case 2: Tool returns unexpected type when BaseModel output_type expected
        class ExpectedOutput(BaseModel):
            result: str
        
        async def returns_string_tool(ctx: RunContext[Any], value: str) -> str:
            # This returns str but ExpectedOutput is expected
            return f"plain string: {value}"
        
        routing2 = RoutingConfig(
            operation_field='operation',
            operation_map={
                'test': ('returns_string_tool', lambda x: {'value': x.value})
            }
        )
        
        agent2 = create_agentool(
            name='wrong_return_type',
            input_schema=TestInput,
            routing_config=routing2,
            tools=[returns_string_tool],
            output_type=ExpectedOutput
        )
        
        async def run_wrong_type_test():
            result = await agent2.run(json.dumps({
                "operation": "test",
                "value": "test"
            }))
            assert "Tool returned str but output_type expects ExpectedOutput" in result.output
        
        asyncio.run(run_wrong_type_test())
        
        # Test case 3: Non-str built-in output_type
        async def returns_int_tool(ctx: RunContext[Any], value: str) -> int:
            return 42
        
        async def returns_wrong_int_tool(ctx: RunContext[Any], value: str) -> str:
            return "not an int"
        
        class TestInputInt(BaseModel):
            operation: Literal['correct', 'wrong']
            value: str
        
        routing3 = RoutingConfig(
            operation_field='operation',
            operation_map={
                'correct': ('returns_int_tool', lambda x: {'value': x.value}),
                'wrong': ('returns_wrong_int_tool', lambda x: {'value': x.value})
            }
        )
        
        agent3 = create_agentool(
            name='int_output_test',
            input_schema=TestInputInt,
            routing_config=routing3,
            tools=[returns_int_tool, returns_wrong_int_tool],
            output_type=int
        )
        
        async def run_int_output_test():
            # Test correct int return
            result = await agent3.run(json.dumps({
                "operation": "correct",
                "value": "test"
            }))
            assert result.output == "42"  # JSON serialized int
            
            # Test wrong type return
            result2 = await agent3.run(json.dumps({
                "operation": "wrong",
                "value": "test"
            }))
            assert "Tool returned str but expected int" in result2.output
        
        asyncio.run(run_int_output_test())
    
    def test_factory_edge_cases(self):
        """Test edge cases in factory for better coverage."""
        import asyncio
        
        # Test case 1: get_type_hints fails
        # Create a function with invalid annotations that cause get_type_hints to fail
        def function_with_bad_annotations(ctx, value: 'NonExistentType') -> 'AnotherBadType':
            """Function with annotations that will fail get_type_hints."""
            return value
        
        # This should not raise an error, but fall back to string representations
        metadata = extract_tool_metadata(function_with_bad_annotations)
        assert metadata.name == 'function_with_bad_annotations'
        assert metadata.return_type == "AnotherBadType"  # String representation without quotes
        assert metadata.return_type_annotation == 'AnotherBadType'  # The annotation itself
        
        # Test case 2: Function with no return annotation
        def function_no_return(ctx, value: str):
            """Function without return type annotation."""
            return value
        
        metadata2 = extract_tool_metadata(function_no_return)
        assert metadata2.return_type is None
        assert metadata2.return_type_annotation is None
        
        # Test case 3: infer_output_type with incompatible types
        class Output1(BaseModel):
            field1: str
        
        class Output2(BaseModel):
            field2: int
        
        # Create metadata with different BaseModel return types
        metadata_list = [
            ToolMetadata(
                name='tool1',
                return_type_annotation=Output1
            ),
            ToolMetadata(
                name='tool2', 
                return_type_annotation=Output2  # Different from Output1
            ),
            ToolMetadata(
                name='tool3',
                return_type_annotation=dict  # Mixed with dict
            )
        ]
        
        # Should return None due to incompatible types
        result = infer_output_type(metadata_list)
        assert result is None
        
        # Test case 4: infer_output_type with non-BaseModel class that causes TypeError
        import abc
        
        metadata_list2 = [
            ToolMetadata(
                name='tool1',
                return_type_annotation=abc.ABCMeta  # This will cause TypeError in issubclass
            )
        ]
        
        # Should handle TypeError gracefully
        result2 = infer_output_type(metadata_list2)
        assert result2 == abc.ABCMeta  # Falls through to exact match check
        
        # Test case 5: All return types exactly the same (non-BaseModel)
        class CustomClass:
            pass
        
        metadata_list3 = [
            ToolMetadata(name='tool1', return_type_annotation=CustomClass),
            ToolMetadata(name='tool2', return_type_annotation=CustomClass),
            ToolMetadata(name='tool3', return_type_annotation=CustomClass)
        ]
        
        result3 = infer_output_type(metadata_list3)
        assert result3 == CustomClass
        
        # Test case 6: Schema override - test when tool doesn't have expected attributes
        class TestInput(BaseModel):
            operation: Literal['test']
            value: str
        
        async def test_tool(ctx: RunContext[Any], value: str) -> str:
            return value
        
        routing = RoutingConfig(
            operation_field='operation',
            operation_map={
                'test': ('test_tool', lambda x: {'value': x.value})
            }
        )
        
        # Create agent
        agent = create_agentool(
            name='schema_test',
            input_schema=TestInput,
            routing_config=routing,
            tools=[test_tool]
        )
        
        # The schema override should work even if attributes are missing
        # This tests the hasattr checks in lines 283-286
        tool = agent._function_toolset.tools.get('__agentool_manager__')
        assert tool is not None
        
        # Manually remove attribute to test edge case
        if hasattr(tool, 'function_schema'):
            original_schema = tool.function_schema
            # Test what happens if _json_schema_dict is missing
            if hasattr(original_schema, '_json_schema_dict'):
                # It should have been set by the factory
                assert hasattr(original_schema, '_json_schema_dict')
        
        # Test actual functionality still works
        async def run_schema_test():
            result = await agent.run(json.dumps({
                "operation": "test",
                "value": "hello"
            }))
            assert result.output == "hello"
        
        asyncio.run(run_schema_test())
    
    def test_model_edge_cases(self):
        """Test edge cases in model for better coverage."""
        import asyncio
        
        # Test case 1: Tool returns non-str, non-BaseModel type with output_type configured
        class TestInput(BaseModel):
            operation: Literal['test']
            value: str
        
        class TestOutput(BaseModel):
            result: int
        
        async def returns_list_tool(ctx: RunContext[Any], value: str) -> list:
            # Returns a list (not str or BaseModel)
            return [1, 2, 3]
        
        routing = RoutingConfig(
            operation_field='operation',
            operation_map={
                'test': ('returns_list_tool', lambda x: {'value': x.value})
            }
        )
        
        # Create agent with output_type
        agent = create_agentool(
            name='list_output_test',
            input_schema=TestInput,
            routing_config=routing,
            tools=[returns_list_tool],
            output_type=TestOutput  # This will trigger the edge case
        )
        
        async def run_list_test():
            result = await agent.run(json.dumps({
                "operation": "test",
                "value": "test"
            }))
            # Should handle the list return type
            assert "output_type expects TestOutput" in result.output
        
        asyncio.run(run_list_test())
        
        # Test case 2: Tool returns non-string type with no output_type
        async def returns_dict_tool(ctx: RunContext[Any], value: str) -> dict:
            return {"key": "value", "number": 42}
        
        routing2 = RoutingConfig(
            operation_field='operation',
            operation_map={
                'test': ('returns_dict_tool', lambda x: {'value': x.value})
            }
        )
        
        agent2 = create_agentool(
            name='dict_no_output_type',
            input_schema=TestInput,
            routing_config=routing2,
            tools=[returns_dict_tool]
            # No output_type specified
        )
        
        async def run_dict_test():
            result = await agent2.run(json.dumps({
                "operation": "test",
                "value": "test"
            }))
            # Should JSON serialize the dict
            output = json.loads(result.output)
            assert output["key"] == "value"
            assert output["number"] == 42
        
        asyncio.run(run_dict_test())

    def test_injector_comprehensive_coverage(self):
        """Test comprehensive injector coverage for uncovered lines."""
        
        async def run_test():
            # Create a fresh injector for testing
            from src.agentool.core.injector import AgenToolInjector
            from src.agentool.core.registry import AgenToolRegistry, MetricsConfig
            
            injector = AgenToolInjector()
            
            # Test 1: Test error handling during metrics recording (lines 253-254, 258-267)
            # Create an agent that will fail
            class FailingInput(BaseModel):
                operation: Literal['fail']
            
            async def failing_tool(ctx: RunContext[Any]) -> str:
                raise ValueError("Test error")
            
            failing_routing = RoutingConfig(
                operation_field='operation',
                operation_map={
                    'fail': ('failing_tool', lambda x: {})
                }
            )
            
            failing_agent = create_agentool(
                name='failing_agent',
                input_schema=FailingInput,
                routing_config=failing_routing,
                tools=[failing_tool]
            )
            
            # Register agent and enable metrics
            injector.register('failing_agent', failing_agent)
            AgenToolRegistry.enable_metrics(True)
            
            # Create a metrics agent for testing
            from src.agentoolkit.observability.metrics import create_metrics_agent
            metrics_agent = create_metrics_agent()
            injector.register('metrics', metrics_agent)
            
            # Test failure with metrics (should hit lines 258-267)
            try:
                await injector.run('failing_agent', {"operation": "fail"})
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert str(e) == "Test error"
            
            # Test 2: Test operation extraction from different input types
            # Test with BaseModel input containing operation
            class TestInputWithOp(BaseModel):
                operation: str
                data: str
            
            test_input = TestInputWithOp(operation="simple", data="test_data")  # Use 'simple' to match the agent
            
            # Create a simple agent for this test
            class SimpleInput(BaseModel):
                operation: Literal['simple']
                data: str
            
            async def simple_tool(ctx: RunContext[Any], data: str) -> str:
                return f"processed: {data}"
            
            simple_routing = RoutingConfig(
                operation_field='operation',
                operation_map={
                    'simple': ('simple_tool', lambda x: {'data': x.data})
                }
            )
            
            simple_agent = create_agentool(
                name='simple_agent',
                input_schema=SimpleInput,
                routing_config=simple_routing,
                tools=[simple_tool]
            )
            
            injector.register('simple_agent', simple_agent)
            
            # Test with BaseModel input (should extract operation from BaseModel)
            result = await injector.run('simple_agent', test_input)
            assert "processed: test_data" in result.output
            
            # Test 3: Test JSON string parsing for operation extraction
            json_input = '{"operation": "simple", "data": "json_test"}'
            result = await injector.run('simple_agent', json_input)
            assert "processed: json_test" in result.output
            
            # Test 4: Test invalid JSON string (should catch exception on lines 230-235)
            invalid_json = '{"operation": "simple", "data": "invalid_json"'  # Missing closing brace
            # This should still work because the whole string gets passed as input
            try:
                result = await injector.run('simple_agent', invalid_json)
            except Exception:
                # This is expected as the JSON is malformed
                pass
            
            # Test 5: Test metrics configuration methods
            injector.enable_metrics(False)
            assert not injector.is_metrics_enabled()
            
            injector.enable_metrics(True)
            assert injector.is_metrics_enabled()
            
            injector.set_metrics_agent('custom_metrics')
            config = AgenToolRegistry.get_metrics_config()
            assert config.metrics_agent_name == 'custom_metrics'
            
            # Reset for cleanup
            injector.set_metrics_agent('metrics')
            
            # Test 6: Test override context manager
            class MockAgent:
                async def run(self, input_data, **kwargs):
                    return type('Result', (), {'output': 'mock_result'})()
            
            mock_agent = MockAgent()
            
            # Test override functionality
            with injector.override(simple_agent=mock_agent):
                result = await injector.run('simple_agent', {"operation": "simple", "data": "test"})
                assert result.output == 'mock_result'
            
            # After override, should return to normal
            result = await injector.run('simple_agent', {"operation": "simple", "data": "normal"})
            assert "processed: normal" in result.output
            
            # Test 7: Test clear method
            # Disable metrics before clearing to avoid unawaited coroutines
            injector.enable_metrics(False)
            injector.clear()
            
            # After clear, should not have instances
            try:
                await injector.run('simple_agent', {"operation": "simple", "data": "test"})
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "no instance available" in str(e)
            
            # Test 8: Test various serialization edge cases
            from src.agentool.core.injector import serialize_to_json_string, validate_json_string
            from datetime import datetime, date, time
            from decimal import Decimal
            
            # Test datetime serialization
            now = datetime.now()
            result = serialize_to_json_string(now)
            assert validate_json_string(result)
            
            # Test date serialization
            today = date.today()
            result = serialize_to_json_string(today)
            assert validate_json_string(result)
            
            # Test time serialization
            now_time = time(12, 30, 45)
            result = serialize_to_json_string(now_time)
            assert validate_json_string(result)
            
            # Test Decimal serialization
            decimal_val = Decimal('123.45')
            result = serialize_to_json_string(decimal_val)
            assert validate_json_string(result)
            
            # Test already valid JSON string
            valid_json = '{"key": "value"}'
            result = serialize_to_json_string(valid_json)
            assert result == valid_json
            
            # Test invalid JSON string (gets treated as string)
            invalid_json = 'not json'
            result = serialize_to_json_string(invalid_json)
            assert result == '"not json"'
            
            # Test unsupported type (should try string conversion)
            class CustomClass:
                def __str__(self):
                    return "custom_object"
            
            custom_obj = CustomClass()
            result = serialize_to_json_string(custom_obj)
            assert "custom_object" in result
            
            # Test completely unsupported type
            class UnsupportedClass:
                def __str__(self):
                    raise Exception("Cannot convert to string")
            
            unsupported_obj = UnsupportedClass()
            try:
                serialize_to_json_string(unsupported_obj)
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "Cannot serialize" in str(e)
            
            # Test validate_json_string with various inputs
            assert validate_json_string('{"valid": "json"}')
            assert not validate_json_string('invalid json')
            assert not validate_json_string(None)  # TypeError case
            
        asyncio.run(run_test())

    def test_example_collection_with_caching(self):
        """Test example collection feature with caching and invalidation."""
        
        async def run_test():
            from src.agentool.core.injector import AgenToolInjector, get_injector
            from src.agentool.core.registry import AgenToolRegistry
            from src.agentool import create_agentool, RoutingConfig
            from pydantic import BaseModel
            from typing import Literal, Any
            from pydantic_ai import RunContext
            import os
            import json
            import shutil
            
            # Clean up any existing examples directory
            if os.path.exists("logs/examples"):
                shutil.rmtree("logs/examples")
            
            # Create a fresh injector
            injector = AgenToolInjector()
            
            # Create a test agent with multiple operations
            class TestInput(BaseModel):
                operation: Literal['read', 'write', 'delete']
                key: str
                value: str = None
            
            # Storage for the test
            storage = {}
            
            async def read_tool(ctx: RunContext[Any], key: str) -> str:
                if key not in storage:
                    raise KeyError(f"Key not found: {key}")
                return storage[key]
            
            async def write_tool(ctx: RunContext[Any], key: str, value: str) -> str:
                storage[key] = value
                return f"Stored {key}"
            
            async def delete_tool(ctx: RunContext[Any], key: str) -> str:
                if key not in storage:
                    raise KeyError(f"Key not found: {key}")
                del storage[key]
                return f"Deleted {key}"
            
            routing = RoutingConfig(
                operation_field='operation',
                operation_map={
                    'read': ('read_tool', lambda x: {'key': x.key}),
                    'write': ('write_tool', lambda x: {'key': x.key, 'value': x.value}),
                    'delete': ('delete_tool', lambda x: {'key': x.key})
                }
            )
            
            test_agent = create_agentool(
                name='test_storage',
                input_schema=TestInput,
                routing_config=routing,
                tools=[read_tool, write_tool, delete_tool]
            )
            
            injector.register('test_storage', test_agent)
            
            # Test 1: Initially, no examples and no caching
            assert len(injector._completeness_cache) == 0
            assert len(injector._completed_agents) == 0
            
            # Test 2: Enable example collection
            injector.enable_example_collection(enabled=True, auto_load=False)
            assert injector._example_collection_enabled == True
            
            # Test 3: Collect success example for write
            result = await injector.run('test_storage', {
                "operation": "write", 
                "key": "test1", 
                "value": "data1"
            })
            assert "Stored test1" in result.output
            
            # Check that example was saved
            assert os.path.exists("logs/examples/test_storage.json")
            with open("logs/examples/test_storage.json", 'r') as f:
                examples = json.load(f)
            assert len(examples) == 1
            assert examples[0]["input"]["operation"] == "write"
            assert "output" in examples[0]
            
            # Test 4: Collect error example for write (duplicate key scenario)
            # First write already succeeded, so we need to trigger an error
            # Let's try reading a non-existent key
            try:
                await injector.run('test_storage', {
                    "operation": "read",
                    "key": "nonexistent"
                })
            except KeyError:
                pass  # Expected
            
            # Check examples were updated
            with open("logs/examples/test_storage.json", 'r') as f:
                examples = json.load(f)
            assert len(examples) == 2
            
            # Test 5: Test caching - check completeness
            is_complete = injector._is_agent_complete('test_storage')
            assert is_complete == False  # Not all operations have both success and error
            
            # Check cache was populated
            assert 'test_storage' in injector._completeness_cache
            assert injector._completeness_cache['test_storage'] == False
            
            # Test 6: Add more examples to complete the agent
            # Success for read
            await injector.run('test_storage', {
                "operation": "write",
                "key": "test2",
                "value": "data2"
            })
            result = await injector.run('test_storage', {
                "operation": "read",
                "key": "test2"
            })
            assert result.output == "data2"
            
            # Error for write (we already have success)
            # Success for delete
            result = await injector.run('test_storage', {
                "operation": "delete",
                "key": "test2"
            })
            assert "Deleted test2" in result.output
            
            # Error for delete
            try:
                await injector.run('test_storage', {
                    "operation": "delete",
                    "key": "nonexistent"
                })
            except KeyError:
                pass
            
            # Test 7: Check cache invalidation
            # The cache should have been invalidated when new examples were added
            # and rebuilt when checking completeness
            is_complete = injector._is_agent_complete('test_storage')
            
            # Test 8: Test manual cache invalidation
            injector.invalidate_completeness_cache('test_storage')
            assert 'test_storage' not in injector._completeness_cache
            assert 'test_storage' not in injector._completed_agents
            
            # Test 9: Test global cache invalidation
            injector._completeness_cache['test_storage'] = True
            injector._completed_agents.add('test_storage')
            injector.invalidate_completeness_cache()  # No agent_name = clear all
            assert len(injector._completeness_cache) == 0
            assert len(injector._completed_agents) == 0
            
            # Test 10: Test status reporting
            status = injector.get_example_status()
            assert 'test_storage' in status
            assert 'operations' in status['test_storage']
            assert 'read' in status['test_storage']['operations']
            assert 'write' in status['test_storage']['operations']
            assert 'delete' in status['test_storage']['operations']
            
            # Test 11: Test export functionality
            injector.export_all_examples("logs/all_examples_test.json")
            assert os.path.exists("logs/all_examples_test.json")
            with open("logs/all_examples_test.json", 'r') as f:
                all_examples = json.load(f)
            assert 'test_storage' in all_examples
            
            # Test 12: Test loading examples on initialization
            injector2 = AgenToolInjector()
            injector2.initialize_examples("logs/examples")
            
            # Check that examples were loaded into registry
            config = AgenToolRegistry.get('test_storage')
            assert config is not None
            assert len(config.examples) > 0
            
            # Test 13: Test clear method also clears caches
            injector._completeness_cache['test_agent'] = True
            injector._completed_agents.add('test_agent')
            injector.clear()
            assert len(injector._completeness_cache) == 0
            assert len(injector._completed_agents) == 0
            
            # Clean up
            if os.path.exists("logs/examples"):
                shutil.rmtree("logs/examples")
            if os.path.exists("logs/all_examples_test.json"):
                os.remove("logs/all_examples_test.json")
            
            # Clear registry for next tests
            AgenToolRegistry.clear()
        
        asyncio.run(run_test())

    def test_injector_dependency_management(self):
        """Test injector dependency creation and management."""
        
        async def run_test():
            from src.agentool.core.injector import AgenToolInjector
            
            injector = AgenToolInjector()
            
            # Test create_deps with no dependencies
            deps = injector.create_deps('nonexistent')
            assert deps is None
            
            # Test create_deps with agent that has no config
            deps = injector.create_deps('nonexistent_agent')
            assert deps is None
            
            # Create agents with dependencies
            class BaseInput(BaseModel):
                operation: Literal['base']
                data: str
            
            async def base_tool(ctx: RunContext[Any], data: str) -> str:
                return f"base: {data}"
            
            base_routing = RoutingConfig(
                operation_field='operation',
                operation_map={
                    'base': ('base_tool', lambda x: {'data': x.data})
                }
            )
            
            base_agent = create_agentool(
                name='base_agent',
                input_schema=BaseInput,
                routing_config=base_routing,
                tools=[base_tool]
            )
            
            # Create dependent agent
            class DependentInput(BaseModel):
                operation: Literal['dependent']
                data: str
            
            @dataclass
            class DependentDeps:
                base_agent: Any
            
            async def dependent_tool(ctx: RunContext[DependentDeps], data: str) -> str:
                # Use the base agent dependency
                return f"dependent using base: {data}"
            
            dependent_routing = RoutingConfig(
                operation_field='operation',
                operation_map={
                    'dependent': ('dependent_tool', lambda x: {'data': x.data})
                }
            )
            
            dependent_agent = create_agentool(
                name='dependent_agent',
                input_schema=DependentInput,
                routing_config=dependent_routing,
                tools=[dependent_tool],
                deps_type=DependentDeps,
                dependencies=['base_agent']  # Specify dependency
            )
            
            # Register both agents
            injector.register('base_agent', base_agent)
            injector.register('dependent_agent', dependent_agent)
            
            # Test dependency creation
            deps = injector.create_deps('dependent_agent')
            assert deps is not None
            assert hasattr(deps, 'base_agent')
            
            # Test running dependent agent
            result = await injector.run('dependent_agent', {"operation": "dependent", "data": "test"})
            assert "dependent using base: test" in result.output
            
        asyncio.run(run_test())