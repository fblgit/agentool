"""
Integration tests for AgenTool with standard pydantic-ai agents.

This module tests how AgenTools integrate with regular agents and
verifies they behave consistently from an external perspective.
"""

import pytest
import json
import asyncio
from typing import Any, Dict, List
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agentool import create_agentool, register_agentool_models
from src.agentool.core.registry import AgenToolRegistry, RoutingConfig


class TestAgentIntegration:
    """Test integration between normal agents and AgenTools."""
    
    def setup_method(self):
        """Clear registry and register models before each test."""
        AgenToolRegistry.clear()
        register_agentool_models()
    
    def test_agentool_basic_functionality(self):
        """Test AgenTool basic functionality."""
        import asyncio
        
        async def run_test():
            # Create an AgenTool
            class WeatherInput(BaseModel):
                operation: str
                city: str
            
            async def weather_tool(ctx: RunContext[Any], city: str) -> dict:
                return {"weather": f"Weather in {city}: 25°C, clear skies"}
            
            routing = RoutingConfig(
                operation_map={
                    'get': ('weather_tool', lambda x: {'city': x.city})
                }
            )
            
            agentool = create_agentool(
                name='weather_service',
                input_schema=WeatherInput,
                routing_config=routing,
                tools=[weather_tool]
            )
            
            # Test AgenTool
            agentool_result = await agentool.run(json.dumps({
                "operation": "get",
                "city": "Paris"
            }))
            output = json.loads(agentool_result.output)
            assert output["weather"] == "Weather in Paris: 25°C, clear skies"
        
        asyncio.run(run_test())
    
    def test_agentool_external_behavior(self):
        """Test that AgenTools behave like normal agents from the outside."""
        import asyncio
        
        async def run_test():
            # AgenTool calculator
            class CalcInput(BaseModel):
                operation: str
                a: float
                b: float
            
            async def add_tool(ctx: RunContext[Any], a: float, b: float) -> dict:
                return {"result": a + b, "operation": f"{a} + {b}"}
            
            async def multiply_tool(ctx: RunContext[Any], a: float, b: float) -> dict:
                return {"result": a * b, "operation": f"{a} * {b}"}
            
            routing = RoutingConfig(
                operation_map={
                    'add': ('add_tool', lambda x: {'a': x.a, 'b': x.b}),
                    'multiply': ('multiply_tool', lambda x: {'a': x.a, 'b': x.b})
                }
            )
            
            agentool_calc = create_agentool(
                name='calculator',
                input_schema=CalcInput,
                routing_config=routing,
                tools=[add_tool, multiply_tool]
            )
            
            # Test AgenTool has agent interface
            assert hasattr(agentool_calc, 'run')
            assert hasattr(agentool_calc, 'run_sync')
            assert hasattr(agentool_calc, 'run_stream')
            
            # Test AgenTool returns results with .output (when use_typed_output is not specified)
            agentool_res = await agentool_calc.run(json.dumps({
                "operation": "add",
                "a": 5,
                "b": 3
            }))
            assert hasattr(agentool_res, 'output')  # Expected behavior when use_typed_output is not specified
            
            # Verify output
            agentool_output = json.loads(agentool_res.output)
            assert agentool_output["result"] == 8
            assert agentool_output["operation"] == "5.0 + 3.0"  # Float formatting
            
            # Test multiply operation
            multiply_res = await agentool_calc.run(json.dumps({
                "operation": "multiply",
                "a": 4,
                "b": 7
            }))
            multiply_output = json.loads(multiply_res.output)
            assert multiply_output["result"] == 28
            assert multiply_output["operation"] == "4.0 * 7.0"  # Float formatting
        
        asyncio.run(run_test())
    
    def test_multiple_agentools_coexistence(self):
        """Test multiple AgenTools can coexist without conflicts."""
        import asyncio
        
        async def run_test():
            # Create multiple AgenTools with different purposes
            
            # Storage AgenTool
            class StorageInput(BaseModel):
                operation: str
                key: str
                value: str = None
            
            storage_data = {}
            
            async def store_write(ctx: RunContext[Any], key: str, value: str) -> dict:
                storage_data[key] = value
                return {"status": "stored", "key": key}
            
            async def store_read(ctx: RunContext[Any], key: str) -> dict:
                return {"value": storage_data.get(key, "not found"), "key": key}
            
            storage_routing = RoutingConfig(
                operation_map={
                    'write': ('store_write', lambda x: {'key': x.key, 'value': x.value}),
                    'read': ('store_read', lambda x: {'key': x.key})
                }
            )
            
            storage_agent = create_agentool(
                name='storage',
                input_schema=StorageInput,
                routing_config=storage_routing,
                tools=[store_write, store_read]
            )
            
            # Math AgenTool
            class MathInput(BaseModel):
                operation: str
                numbers: List[float]
            
            async def sum_tool(ctx: RunContext[Any], numbers: List[float]) -> dict:
                return {"result": sum(numbers), "count": len(numbers)}
            
            async def avg_tool(ctx: RunContext[Any], numbers: List[float]) -> dict:
                avg = sum(numbers) / len(numbers) if numbers else 0
                return {"result": avg, "count": len(numbers)}
            
            math_routing = RoutingConfig(
                operation_map={
                    'sum': ('sum_tool', lambda x: {'numbers': x.numbers}),
                    'average': ('avg_tool', lambda x: {'numbers': x.numbers})
                }
            )
            
            math_agent = create_agentool(
                name='math',
                input_schema=MathInput,
                routing_config=math_routing,
                tools=[sum_tool, avg_tool]
            )
            
            # Test both work independently
            # Storage operations
            await storage_agent.run(json.dumps({
                "operation": "write",
                "key": "test_key",
                "value": "test_value"
            }))
            
            read_result = await storage_agent.run(json.dumps({
                "operation": "read",
                "key": "test_key"
            }))
            read_output = json.loads(read_result.output)
            assert read_output["value"] == "test_value"
            
            # Math operations
            sum_result = await math_agent.run(json.dumps({
                "operation": "sum",
                "numbers": [1, 2, 3, 4, 5]
            }))
            sum_output = json.loads(sum_result.output)
            assert sum_output["result"] == 15
            assert sum_output["count"] == 5
            
            avg_result = await math_agent.run(json.dumps({
                "operation": "average",
                "numbers": [10, 20, 30]
            }))
            avg_output = json.loads(avg_result.output)
            assert avg_output["result"] == 20
            assert avg_output["count"] == 3
            
            # Verify they're registered separately
            assert 'storage' in AgenToolRegistry.list_names()
            assert 'math' in AgenToolRegistry.list_names()
            assert AgenToolRegistry.get('storage') != AgenToolRegistry.get('math')
        
        asyncio.run(run_test())
    
    def test_exception_handling(self):
        """Test exception handling in AgenTools."""
        import asyncio
        
        async def run_test():
            # AgenTool with failing tool
            class FailInput(BaseModel):
                operation: str
                should_fail: bool = False
            
            async def agentool_failing(ctx: RunContext[Any], should_fail: bool) -> dict:
                if should_fail:
                    raise ValueError("AgenTool failed as requested")
                return {"status": "success"}
            
            routing = RoutingConfig(
                operation_map={
                    'test': ('agentool_failing', lambda x: {'should_fail': x.should_fail})
                }
            )
            
            failing_agentool = create_agentool(
                name='failing',
                input_schema=FailInput,
                routing_config=routing,
                tools=[agentool_failing]
            )
            
            # Test successful execution
            agentool_success = await failing_agentool.run(json.dumps({
                "operation": "test",
                "should_fail": False
            }))
            success_output = json.loads(agentool_success.output)
            assert success_output["status"] == "success"
            
            # Test failure handling - AgenTool should raise exception
            with pytest.raises(ValueError) as exc_info:
                await failing_agentool.run(json.dumps({
                    "operation": "test",
                    "should_fail": True
                }))
            assert "AgenTool failed as requested" in str(exc_info.value)
        
        asyncio.run(run_test())
    
    def test_concurrent_agentool_execution(self):
        """Test concurrent execution of multiple AgenTools."""
        import asyncio
        
        async def run_test():
            # Create a counter AgenTool
            counter = {'value': 0}
            
            class CounterInput(BaseModel):
                operation: str
                amount: int = 1
            
            async def increment(ctx: RunContext[Any], amount: int) -> dict:
                # Simulate some async work
                await asyncio.sleep(0.01)
                counter['value'] += amount
                return {"new_value": counter['value'], "added": amount}
            
            async def get_count(ctx: RunContext[Any]) -> dict:
                return {"current_value": counter['value']}
            
            routing = RoutingConfig(
                operation_map={
                    'increment': ('increment', lambda x: {'amount': x.amount}),
                    'get': ('get_count', lambda x: {})
                }
            )
            
            counter_agent = create_agentool(
                name='counter',
                input_schema=CounterInput,
                routing_config=routing,
                tools=[increment, get_count]
            )
            
            # Run multiple increments concurrently
            tasks = []
            for i in range(10):
                task = counter_agent.run(json.dumps({
                    "operation": "increment",
                    "amount": 1
                }))
                tasks.append(task)
            
            # Wait for all to complete
            results = await asyncio.gather(*tasks)
            
            # Check final count
            final_result = await counter_agent.run(json.dumps({
                "operation": "get"
            }))
            final_output = json.loads(final_result.output)
            assert final_output["current_value"] == 10
            
            # Verify all increments were processed
            for result in results:
                output = json.loads(result.output)
                assert "new_value" in output
                assert output["added"] == 1
        
        asyncio.run(run_test())
    
    def test_agentool_with_dependencies(self):
        """Test AgenTool with dependency injection."""
        import asyncio
        
        async def run_test():
            # Create a dependency class
            class ServiceDeps:
                def __init__(self):
                    self.data_store = {"default": "value"}
                    self.call_count = 0
            
            # Create agent with dependencies
            deps = ServiceDeps()
            
            # For AgenTools, we pass deps through the agent
            class ServiceInput(BaseModel):
                operation: str
                key: str
                value: str = None
            
            async def service_get(ctx: RunContext[ServiceDeps], key: str) -> dict:
                ctx.deps.call_count += 1
                return {
                    "value": ctx.deps.data_store.get(key, "not found"),
                    "call_count": ctx.deps.call_count
                }
            
            async def service_set(ctx: RunContext[ServiceDeps], key: str, value: str) -> dict:
                ctx.deps.call_count += 1
                ctx.deps.data_store[key] = value
                return {
                    "stored": True,
                    "call_count": ctx.deps.call_count
                }
            
            routing = RoutingConfig(
                operation_map={
                    'get': ('service_get', lambda x: {'key': x.key}),
                    'set': ('service_set', lambda x: {'key': x.key, 'value': x.value})
                }
            )
            
            # Create agent with deps_type
            service_agent = create_agentool(
                name='service',
                input_schema=ServiceInput,
                routing_config=routing,
                tools=[service_get, service_set],
                deps_type=ServiceDeps
            )
            
            # Run with dependencies
            get_result = await service_agent.run(
                json.dumps({"operation": "get", "key": "default"}),
                deps=deps
            )
            get_output = json.loads(get_result.output)
            assert get_output["value"] == "value"
            assert get_output["call_count"] == 1
            
            # Set a new value
            set_result = await service_agent.run(
                json.dumps({"operation": "set", "key": "new_key", "value": "new_value"}),
                deps=deps
            )
            set_output = json.loads(set_result.output)
            assert set_output["stored"] is True
            assert set_output["call_count"] == 2
            
            # Verify the value was stored
            verify_result = await service_agent.run(
                json.dumps({"operation": "get", "key": "new_key"}),
                deps=deps
            )
            verify_output = json.loads(verify_result.output)
            assert verify_output["value"] == "new_value"
            assert verify_output["call_count"] == 3
        
        asyncio.run(run_test())


class TestAgentComparison:
    """Test AgenTool behavior and attributes."""
    
    def setup_method(self):
        """Setup for tests."""
        AgenToolRegistry.clear()
        register_agentool_models()
    
    def test_agentool_attributes(self):
        """Test AgenTool has expected agent attributes."""
        # Create AgenTool
        class DummyInput(BaseModel):
            value: str
        
        agentool = create_agentool(
            name='dummy',
            input_schema=DummyInput,
            routing_config=RoutingConfig(operation_map={}),
            tools=[]
        )
        
        # Should be Agent instance
        assert isinstance(agentool, Agent)
        
        # Common methods
        common_methods = ['run', 'run_sync', 'run_stream']
        for method in common_methods:
            assert hasattr(agentool, method)
        
        # Should have model attribute
        assert hasattr(agentool, 'model')
        
        # Model type
        assert agentool.model.__class__.__name__ == 'AgenToolModel'
    
    def test_streaming_capability(self):
        """Test if AgenTools support streaming interface."""
        # AgenTool
        class StreamInput(BaseModel):
            message: str
        
        async def echo_tool(ctx: RunContext[Any], message: str) -> dict:
            return {"echo": message}
        
        routing = RoutingConfig(
            operation_map={
                'echo': ('echo_tool', lambda x: {'message': x.message})
            },
            operation_field='message'  # Use message as operation
        )
        
        agentool = create_agentool(
            name='streamer',
            input_schema=StreamInput,
            routing_config=routing,
            tools=[echo_tool]
        )
        
        # Should support run_stream (even if not implemented)
        assert hasattr(agentool, 'run_stream')
        
        # AgenTool can run synchronously
        result = agentool.run_sync(json.dumps({"message": "echo"}))
        assert json.loads(result.output)["echo"] == "echo"
