"""
Tests for AgenTool async and sync execution methods.

This is a separate file to avoid conflicts with pytest-asyncio fixtures.
"""

import asyncio
import json
from typing import Any
from datetime import datetime
from pydantic import BaseModel
from pydantic_ai import RunContext

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agentool import create_agentool, register_agentool_models
from src.agentool.core.registry import AgenToolRegistry, RoutingConfig


class AsyncSyncInput(BaseModel):
    """Input schema for tests."""
    operation: str
    key: str
    value: str = None


class TestAgenToolAsyncSync:
    """Test async and sync methods of AgenTool."""
    
    def setup_method(self):
        """Clear registry before each test."""
        AgenToolRegistry._configs.clear()
        register_agentool_models()
    
    def test_sync_execution(self):
        """Test synchronous execution with run_sync."""
        # Create AgenTool
        async def echo_tool(ctx: RunContext[Any], data: str) -> dict:
            return {"echo": data, "method": "sync_call"}
        
        routing = RoutingConfig(
            operation_field='operation',
            operation_map={
                'echo': ('echo_tool', lambda x: {'data': x.value})
            }
        )
        
        agent = create_agentool(
            name='sync_test',
            input_schema=AsyncSyncInput,
            routing_config=routing,
            tools=[echo_tool]
        )
        
        # Test sync execution
        result = agent.run_sync(json.dumps({
            "operation": "echo",
            "key": "test",
            "value": "hello sync"
        }))
        
        output = json.loads(result.output)
        assert output["echo"] == "hello sync"
        assert output["method"] == "sync_call"
    
    def test_async_execution_in_sync_context(self):
        """Test running async method in sync context using asyncio.run."""
        # Create AgenTool with async tool
        async def process_tool(ctx: RunContext[Any], text: str) -> dict:
            # Simulate async processing
            await asyncio.sleep(0.01)
            return {"processed": text.upper(), "length": len(text)}
        
        routing = RoutingConfig(
            operation_field='operation',
            operation_map={
                'process': ('process_tool', lambda x: {'text': x.value})
            }
        )
        
        agent = create_agentool(
            name='async_in_sync',
            input_schema=AsyncSyncInput,
            routing_config=routing,
            tools=[process_tool]
        )
        
        # Run async method in sync context
        async def run_async():
            return await agent.run(json.dumps({
                "operation": "process",
                "key": "test",
                "value": "hello async"
            }))
        
        result = asyncio.run(run_async())
        output = json.loads(result.output)
        assert output["processed"] == "HELLO ASYNC"
        assert output["length"] == 11
    
    def test_multiple_sync_calls(self):
        """Test multiple synchronous calls in sequence."""
        counter = {'value': 0}
        
        async def counter_tool(ctx: RunContext[Any], increment: int) -> dict:
            counter['value'] += increment
            return {"new_value": counter['value'], "increment": increment}
        
        routing = RoutingConfig(
            operation_field='operation',
            operation_map={
                'count': ('counter_tool', lambda x: {'increment': int(x.value) if x.value else 1})
            }
        )
        
        agent = create_agentool(
            name='counter',
            input_schema=AsyncSyncInput,
            routing_config=routing,
            tools=[counter_tool]
        )
        
        # Make multiple sync calls
        results = []
        for i in range(1, 4):
            result = agent.run_sync(json.dumps({
                "operation": "count",
                "key": "counter",
                "value": str(i)
            }))
            results.append(json.loads(result.output))
        
        # Verify counter incremented correctly
        assert results[0]["new_value"] == 1
        assert results[1]["new_value"] == 3  # 1 + 2
        assert results[2]["new_value"] == 6  # 3 + 3
        assert counter['value'] == 6
    
    def test_sync_with_real_io_simulation(self):
        """Test sync execution with simulated I/O operations."""
        async def io_tool(ctx: RunContext[Any], filename: str) -> dict:
            # Simulate file I/O
            await asyncio.sleep(0.02)  # Simulate read delay
            mock_content = f"Content of {filename}"
            return {
                "filename": filename,
                "content": mock_content,
                "size": len(mock_content)
            }
        
        routing = RoutingConfig(
            operation_field='operation',
            operation_map={
                'read': ('io_tool', lambda x: {'filename': x.value})
            }
        )
        
        agent = create_agentool(
            name='io_test',
            input_schema=AsyncSyncInput,
            routing_config=routing,
            tools=[io_tool]
        )
        
        # Test sync I/O operation
        result = agent.run_sync(json.dumps({
            "operation": "read",
            "key": "file",
            "value": "test.txt"
        }))
        
        output = json.loads(result.output)
        assert output["filename"] == "test.txt"
        assert output["content"] == "Content of test.txt"
        assert output["size"] == 19
    
