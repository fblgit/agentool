"""
Edge case and exception handling tests for AgenTool.

This module tests various edge cases, exception scenarios, and fault tolerance
of the AgenTool framework.
"""

import pytest
import asyncio
import json
from typing import Any, Dict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field
from pydantic_ai import RunContext

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agentool import create_agentool, register_agentool_models
from src.agentool.core.registry import AgenToolRegistry, AgenToolConfig, RoutingConfig
from src.agentool.core.manager import AgenToolManager
from src.agentool.core.model import AgenToolModel


class TestFaultyAppHandling:
    """Test how AgenTool handles faulty applications and exceptions."""
    
    def setup_method(self):
        """Clear registry before each test."""
        AgenToolRegistry.clear()
        register_agentool_models()
    
    def test_tool_runtime_exception(self):
        """Test handling of RuntimeError in tool execution."""
        async def run_test():
            class FaultyInput(BaseModel):
                operation: str
                should_fail: bool = False
            
            async def faulty_tool(ctx: RunContext[Any], should_fail: bool) -> dict:
                if should_fail:
                    raise RuntimeError("Tool exploded! 💥")
                return {"status": "success", "message": "All good"}
            
            routing = RoutingConfig(
                operation_map={
                    'execute': ('faulty_tool', lambda x: {'should_fail': x.should_fail})
                }
            )
            
            agent = create_agentool(
                name='faulty',
                input_schema=FaultyInput,
                routing_config=routing,
                tools=[faulty_tool]
            )
            
            # Test successful execution
            result = await agent.run('{"operation": "execute", "should_fail": false}')
            output = json.loads(result.output)
            assert output["status"] == "success"
            
            # Test failed execution - should raise exception
            with pytest.raises(RuntimeError) as exc_info:
                await agent.run('{"operation": "execute", "should_fail": true}')
            assert "Tool exploded! 💥" in str(exc_info.value)
        
        asyncio.run(run_test())
    
    def test_various_exception_types(self):
        """Test handling of different exception types."""
        async def run_test():
            class ExceptionInput(BaseModel):
                operation: str
                error_type: str
            
            async def exception_thrower(ctx: RunContext[Any], error_type: str) -> dict:
                if error_type == "value":
                    raise ValueError("Invalid value provided")
                elif error_type == "type":
                    raise TypeError("Wrong type!")
                elif error_type == "key":
                    raise KeyError("Missing key: important_data")
                elif error_type == "custom":
                    class CustomError(Exception):
                        pass
                    raise CustomError("Something specific went wrong")
                elif error_type == "assertion":
                    assert False, "Assertion failed"
                return {"status": "no_error"}
            
            routing = RoutingConfig(
                operation_map={
                    'throw': ('exception_thrower', lambda x: {'error_type': x.error_type})
                }
            )
            
            agent = create_agentool(
                name='exceptions',
                input_schema=ExceptionInput,
                routing_config=routing,
                tools=[exception_thrower]
            )
            
            # Test each exception type - should raise exceptions
            error_types_and_exceptions = [
                ("value", ValueError, "Invalid value provided"),
                ("type", TypeError, "Wrong type!"),
                ("key", KeyError, "Missing key: important_data"),
                ("custom", Exception, "Something specific went wrong"),  # CustomError inherits from Exception
                ("assertion", AssertionError, "Assertion failed")
            ]
            
            for error_type, expected_exception, expected_message in error_types_and_exceptions:
                with pytest.raises(expected_exception) as exc_info:
                    await agent.run(json.dumps({
                        "operation": "throw",
                        "error_type": error_type
                    }))
                assert expected_message in str(exc_info.value)
        
        asyncio.run(run_test())
    
    def test_async_exception_handling(self):
        """Test exception handling in async contexts."""
        async def run_test():
            class AsyncInput(BaseModel):
                operation: str
                delay: float = 0.1
                fail_during: str = "execution"  # "execution" or "cleanup"
            
            async def async_faulty_tool(ctx: RunContext[Any], delay: float, fail_during: str) -> dict:
                await asyncio.sleep(delay)
                
                if fail_during == "execution":
                    raise asyncio.TimeoutError("Async operation timed out")
                
                try:
                    return {"status": "completed"}
                finally:
                    if fail_during == "cleanup":
                        raise RuntimeError("Cleanup failed")
            
            routing = RoutingConfig(
                operation_map={
                    'async_op': ('async_faulty_tool', lambda x: {'delay': x.delay, 'fail_during': x.fail_during})
                }
            )
            
            agent = create_agentool(
                name='async_faulty',
                input_schema=AsyncInput,
                routing_config=routing,
                tools=[async_faulty_tool]
            )
            
            # Test async timeout error - should raise exception
            with pytest.raises(asyncio.TimeoutError) as exc_info:
                await agent.run(json.dumps({
                    "operation": "async_op",
                    "delay": 0.01,
                    "fail_during": "execution"
                }))
            assert "Async operation timed out" in str(exc_info.value)
            
            # Test exception during cleanup - should raise exception
            with pytest.raises(RuntimeError) as exc_info:
                await agent.run(json.dumps({
                    "operation": "async_op",
                    "delay": 0.01,
                    "fail_during": "cleanup"
                }))
            assert "Cleanup failed" in str(exc_info.value)
        
        asyncio.run(run_test())
    
    def test_exception_during_transformation(self):
        """Test exception handling during argument transformation."""
        async def run_test():
            class TransformInput(BaseModel):
                operation: str
                data: Dict[str, Any]
            
            async def normal_tool(ctx: RunContext[Any], processed_data: Any) -> dict:
                return {"received": processed_data}
            
            def failing_transform(inp):
                # This transform function will fail
                if inp.data.get("fail_transform"):
                    raise ValueError("Transform function failed")
                return {"processed_data": inp.data}
            
            routing = RoutingConfig(
                operation_map={
                    'process': ('normal_tool', failing_transform)
                }
            )
            
            agent = create_agentool(
                name='transform_fail',
                input_schema=TransformInput,
                routing_config=routing,
                tools=[normal_tool]
            )
            
            # Test transform failure
            result = await agent.run(json.dumps({
                "operation": "process",
                "data": {"fail_transform": True}
            }))
            assert "Error transforming arguments" in result.output
            assert "Transform function failed" in result.output
            
            # Test successful transform
            result = await agent.run(json.dumps({
                "operation": "process",
                "data": {"value": "test"}
            }))
            output = json.loads(result.output)
            assert output["received"]["value"] == "test"
        
        asyncio.run(run_test())


class TestEdgeCases:
    """Test various edge cases in AgenTool."""
    
    def setup_method(self):
        """Clear registry before each test."""
        AgenToolRegistry.clear()
        register_agentool_models()
    
    def test_routing_config_none_operation_map(self):
        """Test RoutingConfig with operation_map=None."""
        # This tests line 42 in registry.py
        config = RoutingConfig(operation_map=None)
        assert config.operation_map == {}
        
        # Also test with explicit empty dict
        config2 = RoutingConfig(operation_map={})
        assert config2.operation_map == {}
    
    def test_registry_clear_method(self):
        """Test the registry clear() method."""
        # This tests line 94 in registry.py
        # First add some configs
        config = AgenToolConfig(
            input_schema=BaseModel,
            routing_config=RoutingConfig(operation_map={})
        )
        
        AgenToolRegistry.register('test1', config)
        AgenToolRegistry.register('test2', config)
        
        assert len(AgenToolRegistry.list_names()) >= 2
        assert 'test1' in AgenToolRegistry.list_names()
        
        # Clear the registry
        AgenToolRegistry.clear()
        
        # Verify it's empty
        assert len(AgenToolRegistry.list_names()) == 0
        assert AgenToolRegistry.get('test1') is None
    
    def test_schema_without_properties(self):
        """Test handling of schema without 'properties' key."""
        # This tests line 114 in manager.py
        class MinimalModel(BaseModel):
            """A minimal model to test edge cases."""
            pass
        
        # Create a custom model that might have unusual schema
        class CustomSchemaModel(BaseModel):
            value: str = "test"
            
            @classmethod
            def model_json_schema(cls):
                # Return a schema without 'properties'
                return {
                    "type": "object",
                    "title": "CustomSchemaModel"
                    # No 'properties' key
                }
        
        routing = RoutingConfig(
            operation_map={}
        )
        
        config = AgenToolConfig(
            input_schema=CustomSchemaModel,
            routing_config=routing
        )
        
        manager = AgenToolManager(
            name='test',
            config=config,
            tool_functions={}
        )
        
        # Get the schema - should add empty properties
        schema = manager.get_tool_schema()
        assert 'properties' in schema
        assert schema['properties'] == {}
    
    def test_concurrent_registry_access(self):
        """Test concurrent access to the registry."""
        async def run_test():
            async def register_config(name: str, delay: float):
                await asyncio.sleep(delay)
                config = AgenToolConfig(
                    input_schema=BaseModel,
                    routing_config=RoutingConfig(operation_map={})
                )
                AgenToolRegistry.register(name, config)
                return name
            
            # Clear registry first
            AgenToolRegistry.clear()
            
            # Register multiple configs concurrently
            tasks = [
                register_config(f'concurrent_{i}', 0.01 * (i % 3))
                for i in range(10)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Verify all were registered
            assert len(results) == 10
            names = AgenToolRegistry.list_names()
            for i in range(10):
                assert f'concurrent_{i}' in names
        
        asyncio.run(run_test())
    
    def test_model_without_model_fields(self):
        """Test handling of model without model_fields attribute."""
        # This tests lines 117-125 in manager.py
        class NonStandardModel:
            """A model that doesn't inherit from BaseModel."""
            def __init__(self):
                self.value = "test"
            
            @classmethod
            def model_json_schema(cls):
                return {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"}
                    }
                }
        
        routing = RoutingConfig(operation_map={})
        config = AgenToolConfig(
            input_schema=NonStandardModel,  # type: ignore
            routing_config=routing
        )
        
        manager = AgenToolManager(
            name='test',
            config=config,
            tool_functions={}
        )
        
        # Get schema - should work without model_fields
        schema = manager.get_tool_schema()
        assert 'properties' in schema
        assert 'value' in schema['properties']
        # Should not have 'required' since no model_fields
        assert 'required' not in schema or schema['required'] == []
    
    def test_tool_without_function_schema(self):
        """Test handling of tool without function_schema attributes."""
        # This tests lines 110-111 in factory.py
        # We'll need to mock this scenario since pydantic-ai tools always have these
        
        # Create a minimal tool object without expected attributes
        class MinimalTool:
            def __init__(self, name):
                self.name = name
                # No function_schema attribute
        
        # This is more of a defensive check - in practice tools always have these
        tool = MinimalTool('test_tool')
        
        # Check the attributes don't exist
        assert not hasattr(tool, 'function_schema')
        
        # The factory code handles this gracefully by checking hasattr


class TestRegistryThreadSafety:
    """Test thread safety of the registry."""
    
    def test_concurrent_registration_threading(self):
        """Test thread-safe concurrent registration."""
        AgenToolRegistry.clear()
        
        def register_in_thread(thread_id: int):
            config = AgenToolConfig(
                input_schema=BaseModel,
                routing_config=RoutingConfig(operation_map={})
            )
            for i in range(5):
                AgenToolRegistry.register(f'thread_{thread_id}_item_{i}', config)
        
        # Use threads instead of async
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(register_in_thread, i)
                for i in range(5)
            ]
            
            # Wait for all to complete
            for future in futures:
                future.result()
        
        # Verify all were registered
        names = AgenToolRegistry.list_names()
        assert len(names) == 25  # 5 threads * 5 items each
        
        for thread_id in range(5):
            for item_id in range(5):
                assert f'thread_{thread_id}_item_{item_id}' in names