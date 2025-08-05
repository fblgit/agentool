# AgenTool Testing Guide

Comprehensive guide for testing AgenTool-based applications, from unit tests to integration and performance testing.

## Table of Contents

- [Testing Philosophy](#testing-philosophy)
- [Unit Testing](#unit-testing)
- [Integration Testing](#integration-testing)
- [Mocking and Fixtures](#mocking-and-fixtures)
- [Performance Testing](#performance-testing)
- [Test Organization](#test-organization)
- [CI/CD Integration](#cicd-integration)

## Testing Philosophy

AgenTool's deterministic nature makes it highly testable. Unlike LLM-based agents where responses can vary, AgenTool agents provide predictable, reproducible behavior that's ideal for comprehensive testing.

### Key Testing Principles

1. **Deterministic Behavior**: AgenTools always produce the same output for the same input
2. **Schema Validation**: Test both valid and invalid inputs against Pydantic schemas
3. **Error Handling**: Verify graceful error handling for edge cases
4. **Dependency Injection**: Test component interactions and dependency resolution
5. **Performance Characteristics**: Ensure consistent performance under load

## Unit Testing

### Basic Agent Testing

```python
import pytest
import json
from agentool import create_agentool, RoutingConfig, BaseOperationInput
from typing import Literal, Any
from pydantic_ai import RunContext

# Test subject: Calculator agent
class CalculatorInput(BaseOperationInput):
    operation: Literal['add', 'subtract', 'multiply', 'divide']
    a: float
    b: float

async def add(ctx: RunContext[Any], a: float, b: float) -> float:
    return a + b

async def subtract(ctx: RunContext[Any], a: float, b: float) -> float:
    return a - b

async def multiply(ctx: RunContext[Any], a: float, b: float) -> float:
    return a * b

async def divide(ctx: RunContext[Any], a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# Test fixtures
@pytest.fixture
def calculator_agent():
    """Create calculator agent for testing."""
    routing = RoutingConfig(
        operation_map={
            'add': ('add', lambda x: {'a': x.a, 'b': x.b}),
            'subtract': ('subtract', lambda x: {'a': x.a, 'b': x.b}),
            'multiply': ('multiply', lambda x: {'a': x.a, 'b': x.b}),
            'divide': ('divide', lambda x: {'a': x.a, 'b': x.b})
        }
    )
    
    return create_agentool(
        name='test_calculator',
        input_schema=CalculatorInput,
        routing_config=routing,
        tools=[add, subtract, multiply, divide]
    )

class TestCalculatorAgent:
    """Comprehensive test suite for calculator agent."""
    
    @pytest.mark.asyncio
    async def test_addition(self, calculator_agent):
        """Test addition operation."""
        result = await calculator_agent.run('{"operation": "add", "a": 5, "b": 3}')
        assert float(result.output) == 8.0
    
    @pytest.mark.asyncio
    async def test_subtraction(self, calculator_agent):
        """Test subtraction operation."""
        result = await calculator_agent.run('{"operation": "subtract", "a": 10, "b": 4}')
        assert float(result.output) == 6.0
    
    @pytest.mark.asyncio
    async def test_multiplication(self, calculator_agent):
        """Test multiplication operation."""
        result = await calculator_agent.run('{"operation": "multiply", "a": 6, "b": 7}')
        assert float(result.output) == 42.0
    
    @pytest.mark.asyncio
    async def test_division(self, calculator_agent):
        """Test division operation."""
        result = await calculator_agent.run('{"operation": "divide", "a": 15, "b": 3}')
        assert float(result.output) == 5.0
    
    @pytest.mark.asyncio
    async def test_division_by_zero(self, calculator_agent):
        """Test division by zero error handling."""
        result = await calculator_agent.run('{"operation": "divide", "a": 10, "b": 0}')
        assert "Cannot divide by zero" in result.output
    
    @pytest.mark.asyncio
    async def test_invalid_operation(self, calculator_agent):
        """Test invalid operation handling."""
        result = await calculator_agent.run('{"operation": "invalid", "a": 1, "b": 2}')
        assert "Unknown operation" in result.output
        assert "Available operations" in result.output
    
    @pytest.mark.asyncio
    async def test_invalid_json(self, calculator_agent):
        """Test invalid JSON input."""
        result = await calculator_agent.run('invalid json string')
        assert "Invalid JSON" in result.output
    
    @pytest.mark.asyncio
    async def test_missing_required_fields(self, calculator_agent):
        """Test missing required fields."""
        result = await calculator_agent.run('{"operation": "add", "a": 5}')
        assert "Error creating input model" in result.output
    
    @pytest.mark.asyncio
    async def test_invalid_field_types(self, calculator_agent):
        """Test invalid field types."""
        result = await calculator_agent.run('{"operation": "add", "a": "not_a_number", "b": 3}')
        assert "Error creating input model" in result.output
    
    @pytest.mark.asyncio
    async def test_extra_fields_rejected(self, calculator_agent):
        """Test that extra fields are rejected."""
        result = await calculator_agent.run('{"operation": "add", "a": 5, "b": 3, "extra": "field"}')
        assert "Error creating input model" in result.output
    
    @pytest.mark.parametrize("operation,a,b,expected", [
        ("add", 1, 2, 3),
        ("add", -1, 1, 0),
        ("add", 0.5, 0.5, 1.0),
        ("subtract", 5, 3, 2),
        ("subtract", 0, 5, -5),
        ("multiply", 3, 4, 12),
        ("multiply", -2, 3, -6),
        ("divide", 8, 2, 4),
        ("divide", -10, 2, -5),
    ])
    @pytest.mark.asyncio
    async def test_parametrized_operations(self, calculator_agent, operation, a, b, expected):
        """Test multiple operation scenarios."""
        input_data = json.dumps({"operation": operation, "a": a, "b": b})
        result = await calculator_agent.run(input_data)
        assert float(result.output) == expected
```

### Schema Validation Testing

```python
import pytest
from pydantic import ValidationError
from agentool import BaseOperationInput
from typing import Literal, Optional

class TestSchemaValidation:
    """Test Pydantic schema validation directly."""
    
    def test_valid_calculator_input(self):
        """Test valid input creation."""
        input_data = CalculatorInput(operation="add", a=5.0, b=3.0)
        assert input_data.operation == "add"
        assert input_data.a == 5.0
        assert input_data.b == 3.0
    
    def test_invalid_operation(self):
        """Test invalid operation value."""
        with pytest.raises(ValidationError) as exc_info:
            CalculatorInput(operation="invalid", a=5.0, b=3.0)
        
        assert "Input should be" in str(exc_info.value)
    
    def test_missing_required_fields(self):
        """Test missing required fields."""
        with pytest.raises(ValidationError) as exc_info:
            CalculatorInput(operation="add", a=5.0)  # Missing 'b'
        
        assert "Field required" in str(exc_info.value)
    
    def test_type_conversion(self):
        """Test automatic type conversion."""
        input_data = CalculatorInput(operation="add", a="5", b="3")  # Strings that can be converted
        assert isinstance(input_data.a, float)
        assert isinstance(input_data.b, float)
        assert input_data.a == 5.0
        assert input_data.b == 3.0
    
    def test_invalid_type_conversion(self):
        """Test invalid type conversion."""
        with pytest.raises(ValidationError):
            CalculatorInput(operation="add", a="not_a_number", b=3.0)

class TestAdvancedSchemas:
    """Test more complex schema patterns."""
    
    def test_optional_fields(self):
        """Test schemas with optional fields."""
        class OptionalFieldInput(BaseOperationInput):
            operation: Literal['process']
            required_field: str
            optional_field: Optional[str] = None
            default_field: str = "default_value"
        
        # Valid with just required fields
        input1 = OptionalFieldInput(operation="process", required_field="value")
        assert input1.optional_field is None
        assert input1.default_field == "default_value"
        
        # Valid with all fields
        input2 = OptionalFieldInput(
            operation="process", 
            required_field="value",
            optional_field="optional",
            default_field="custom"
        )
        assert input2.optional_field == "optional"
        assert input2.default_field == "custom"
    
    def test_nested_schemas(self):
        """Test schemas with nested objects."""
        from pydantic import BaseModel
        
        class NestedData(BaseModel):
            id: int
            name: str
        
        class NestedInput(BaseOperationInput):
            operation: Literal['nested']
            data: NestedData
        
        input_data = NestedInput(
            operation="nested",
            data=NestedData(id=1, name="test")
        )
        assert input_data.data.id == 1
        assert input_data.data.name == "test"
    
    def test_list_fields(self):
        """Test schemas with list fields."""
        from typing import List
        
        class ListInput(BaseOperationInput):
            operation: Literal['batch']
            items: List[str]
            numbers: List[int]
        
        input_data = ListInput(
            operation="batch",
            items=["item1", "item2"],
            numbers=[1, 2, 3]
        )
        assert len(input_data.items) == 2
        assert len(input_data.numbers) == 3
```

### Tool Function Testing

```python
class TestToolFunctions:
    """Test individual tool functions in isolation."""
    
    @pytest.mark.asyncio
    async def test_add_function_directly(self):
        """Test add function directly."""
        from unittest.mock import Mock
        
        ctx = Mock()
        result = await add(ctx, 5.0, 3.0)
        assert result == 8.0
    
    @pytest.mark.asyncio
    async def test_divide_function_error(self):
        """Test divide function error handling."""
        from unittest.mock import Mock
        
        ctx = Mock()
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            await divide(ctx, 10.0, 0.0)
    
    @pytest.mark.asyncio
    async def test_tool_with_complex_logic(self):
        """Test tool with more complex logic."""
        async def complex_calculation(ctx: RunContext[Any], operation: str, values: list[float]) -> dict:
            """Tool with complex logic."""
            if operation == "sum":
                result = sum(values)
            elif operation == "average":
                result = sum(values) / len(values) if values else 0
            elif operation == "max":
                result = max(values) if values else 0
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            return {
                "operation": operation,
                "input_count": len(values),
                "result": result
            }
        
        from unittest.mock import Mock
        ctx = Mock()
        
        # Test sum
        result = await complex_calculation(ctx, "sum", [1, 2, 3, 4])
        assert result["result"] == 10
        assert result["input_count"] == 4
        
        # Test average
        result = await complex_calculation(ctx, "average", [2, 4, 6])
        assert result["result"] == 4.0
        
        # Test error
        with pytest.raises(ValueError, match="Unknown operation"):
            await complex_calculation(ctx, "invalid", [1, 2, 3])
```

## Integration Testing

### Multi-Agent Testing

```python
from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry

class TestMultiAgentIntegration:
    """Test integration between multiple agents."""
    
    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        """Setup and cleanup for each test."""
        # Clear registry before test
        AgenToolRegistry.clear()
        get_injector().clear()
        yield
        # Clean up after test
        AgenToolRegistry.clear()
        get_injector().clear()
    
    @pytest.fixture
    def validation_agent(self):
        """Create validation agent for testing."""
        class ValidationInput(BaseOperationInput):
            operation: Literal['validate_number', 'validate_string']
            value: Any
        
        async def validate_number(ctx: RunContext[Any], value: Any) -> dict:
            try:
                num = float(value)
                return {"valid": True, "value": num, "type": "number"}
            except (ValueError, TypeError):
                return {"valid": False, "error": "Not a valid number"}
        
        async def validate_string(ctx: RunContext[Any], value: Any) -> dict:
            if isinstance(value, str) and len(value) > 0:
                return {"valid": True, "value": value, "type": "string"}
            else:
                return {"valid": False, "error": "Not a valid string"}
        
        routing = RoutingConfig(
            operation_map={
                'validate_number': ('validate_number', lambda x: {'value': x.value}),
                'validate_string': ('validate_string', lambda x: {'value': x.value})
            }
        )
        
        return create_agentool(
            name='validator',
            input_schema=ValidationInput,
            routing_config=routing,
            tools=[validate_number, validate_string]
        )
    
    @pytest.fixture
    def processing_agent(self):
        """Create processing agent that depends on validator."""
        from agentool.core.injector import InjectedDeps
        from dataclasses import dataclass
        
        @dataclass
        class ProcessingDeps(InjectedDeps):
            async def validate_and_process(self, operation: str, value: Any) -> dict:
                # First validate
                validation_type = "validate_number" if operation in ["add", "multiply"] else "validate_string"
                validation_result = await self.call_agent('validator', {
                    'operation': validation_type,
                    'value': value
                })
                
                validation_data = json.loads(validation_result.output)
                if not validation_data['valid']:
                    return {
                        "success": False,
                        "error": f"Validation failed: {validation_data['error']}"
                    }
                
                # Then process
                if operation == "add":
                    result = validation_data['value'] + 10
                elif operation == "multiply":
                    result = validation_data['value'] * 2
                elif operation == "uppercase":
                    result = validation_data['value'].upper()
                else:
                    result = validation_data['value']
                
                return {
                    "success": True,
                    "original": value,
                    "validated": validation_data['value'],
                    "processed": result
                }
        
        class ProcessingInput(BaseOperationInput):
            operation: Literal['add', 'multiply', 'uppercase']
            value: Any
        
        async def process_value(ctx: RunContext[ProcessingDeps], operation: str, value: Any) -> dict:
            return await ctx.deps.validate_and_process(operation, value)
        
        routing = RoutingConfig(
            operation_map={
                'add': ('process_value', lambda x: {'operation': x.operation, 'value': x.value}),
                'multiply': ('process_value', lambda x: {'operation': x.operation, 'value': x.value}),
                'uppercase': ('process_value', lambda x: {'operation': x.operation, 'value': x.value})
            }
        )
        
        return create_agentool(
            name='processor',
            input_schema=ProcessingInput,
            routing_config=routing,
            tools=[process_value],
            deps_type=ProcessingDeps,
            dependencies=['validator']
        )
    
    @pytest.mark.asyncio
    async def test_successful_processing_chain(self, validation_agent, processing_agent):
        """Test successful processing chain."""
        injector = get_injector()
        
        # Test number processing
        result = await injector.run('processor', {
            'operation': 'add',
            'value': '5'
        })
        
        result_data = json.loads(result.output)
        assert result_data['success'] is True
        assert result_data['validated'] == 5.0
        assert result_data['processed'] == 15.0
    
    @pytest.mark.asyncio
    async def test_validation_failure(self, validation_agent, processing_agent):
        """Test processing chain with validation failure."""
        injector = get_injector()
        
        # Test with invalid number
        result = await injector.run('processor', {
            'operation': 'add',
            'value': 'not_a_number'
        })
        
        result_data = json.loads(result.output)
        assert result_data['success'] is False
        assert 'Validation failed' in result_data['error']
    
    @pytest.mark.asyncio
    async def test_string_processing(self, validation_agent, processing_agent):
        """Test string processing."""
        injector = get_injector()
        
        result = await injector.run('processor', {
            'operation': 'uppercase',
            'value': 'hello world'
        })
        
        result_data = json.loads(result.output)
        assert result_data['success'] is True
        assert result_data['processed'] == 'HELLO WORLD'
```

### End-to-End Testing

```python
class TestEndToEndWorkflow:
    """Test complete workflows end-to-end."""
    
    @pytest.fixture(autouse=True)
    def setup_complete_system(self):
        """Set up a complete multi-agent system."""
        # Clear state
        AgenToolRegistry.clear()
        get_injector().clear()
        
        # Create storage agent
        storage_backend = {}
        
        class StorageInput(BaseOperationInput):
            operation: Literal['get', 'set', 'delete', 'list']
            key: Optional[str] = None
            value: Optional[Any] = None
        
        async def storage_get(ctx, key: str):
            return storage_backend.get(key, f"Key '{key}' not found")
        
        async def storage_set(ctx, key: str, value: Any):
            storage_backend[key] = value
            return f"Stored '{key}'"
        
        async def storage_delete(ctx, key: str):
            if key in storage_backend:
                del storage_backend[key]
                return f"Deleted '{key}'"
            return f"Key '{key}' not found"
        
        async def storage_list(ctx):
            return list(storage_backend.keys())
        
        storage_routing = RoutingConfig(
            operation_map={
                'get': ('storage_get', lambda x: {'key': x.key}),
                'set': ('storage_set', lambda x: {'key': x.key, 'value': x.value}),
                'delete': ('storage_delete', lambda x: {'key': x.key}),
                'list': ('storage_list', lambda x: {})
            }
        )
        
        storage_agent = create_agentool(
            name='storage',
            input_schema=StorageInput,
            routing_config=storage_routing,
            tools=[storage_get, storage_set, storage_delete, storage_list]
        )
        
        # Store reference to backend for testing
        self.storage_backend = storage_backend
        
        yield
        
        # Cleanup
        AgenToolRegistry.clear()
        get_injector().clear()
    
    @pytest.mark.asyncio
    async def test_complete_crud_workflow(self):
        """Test complete CRUD workflow."""
        injector = get_injector()
        
        # Initially empty
        result = await injector.run('storage', {'operation': 'list'})
        assert json.loads(result.output) == []
        
        # Store some values
        await injector.run('storage', {
            'operation': 'set',
            'key': 'user:1',
            'value': {'name': 'Alice', 'age': 30}
        })
        
        await injector.run('storage', {
            'operation': 'set',
            'key': 'user:2',
            'value': {'name': 'Bob', 'age': 25}
        })
        
        # List keys
        result = await injector.run('storage', {'operation': 'list'})
        keys = json.loads(result.output)
        assert sorted(keys) == ['user:1', 'user:2']
        
        # Get specific values
        result = await injector.run('storage', {'operation': 'get', 'key': 'user:1'})
        user_data = json.loads(result.output)
        assert user_data['name'] == 'Alice'
        assert user_data['age'] == 30
        
        # Delete a key
        result = await injector.run('storage', {'operation': 'delete', 'key': 'user:2'})
        assert 'Deleted' in result.output
        
        # Verify deletion
        result = await injector.run('storage', {'operation': 'list'})
        keys = json.loads(result.output)
        assert keys == ['user:1']
        
        # Try to get deleted key
        result = await injector.run('storage', {'operation': 'get', 'key': 'user:2'})
        assert 'not found' in result.output
    
    @pytest.mark.asyncio
    async def test_workflow_with_different_data_types(self):
        """Test workflow with various data types."""
        injector = get_injector()
        
        test_data = [
            ('string_key', 'Simple string'),
            ('number_key', 42),
            ('float_key', 3.14),
            ('bool_key', True),
            ('list_key', [1, 2, 3, 'four']),
            ('dict_key', {'nested': {'data': 'value'}}),
            ('null_key', None)
        ]
        
        # Store all data types
        for key, value in test_data:
            await injector.run('storage', {
                'operation': 'set',
                'key': key,
                'value': value
            })
        
        # Verify all data types
        for key, expected_value in test_data:
            result = await injector.run('storage', {'operation': 'get', 'key': key})
            stored_value = json.loads(result.output)
            assert stored_value == expected_value
```

## Mocking and Fixtures

### Mock Agents

```python
from unittest.mock import AsyncMock, Mock
import pytest

class MockAgenTool:
    """Mock AgenTool for testing."""
    
    def __init__(self, responses: dict = None):
        """Initialize with predefined responses."""
        self.responses = responses or {}
        self.call_history = []
    
    async def run(self, input_data: str) -> Mock:
        """Mock run method."""
        # Parse input to determine response
        try:
            parsed_input = json.loads(input_data)
            operation = parsed_input.get('operation', 'unknown')
            
            # Record call
            self.call_history.append({
                'input': input_data,
                'parsed': parsed_input,
                'operation': operation
            })
            
            # Return predefined response or default
            if operation in self.responses:
                output = self.responses[operation]
            else:
                output = f"Mock response for {operation}"
            
            # Create mock result
            result = Mock()
            result.output = output if isinstance(output, str) else json.dumps(output)
            return result
            
        except json.JSONDecodeError:
            result = Mock()
            result.output = "Mock response for invalid JSON"
            return result

@pytest.fixture
def mock_calculator():
    """Create mock calculator agent."""
    return MockAgenTool({
        'add': '8.0',
        'subtract': '2.0',
        'multiply': '15.0',
        'divide': '3.0'
    })

@pytest.fixture
def mock_storage():
    """Create mock storage agent."""
    return MockAgenTool({
        'get': {'name': 'Mock User', 'age': 25},
        'set': 'Stored successfully',
        'delete': 'Deleted successfully',
        'list': ['key1', 'key2', 'key3']
    })

class TestWithMocks:
    """Test using mock agents."""
    
    @pytest.mark.asyncio
    async def test_with_mock_calculator(self, mock_calculator):
        """Test using mock calculator."""
        result = await mock_calculator.run('{"operation": "add", "a": 5, "b": 3}')
        assert result.output == '8.0'
        
        # Verify call was recorded
        assert len(mock_calculator.call_history) == 1
        assert mock_calculator.call_history[0]['operation'] == 'add'
    
    @pytest.mark.asyncio
    async def test_dependency_injection_with_mocks(self):
        """Test dependency injection using mocks."""
        from agentool.core.injector import get_injector
        
        # Create real agent that depends on mock
        @dataclass
        class TestDeps(InjectedDeps):
            async def process_with_calculator(self, a: float, b: float) -> dict:
                calc_result = await self.call_agent('calculator', {
                    'operation': 'add',
                    'a': a,
                    'b': b
                })
                return {
                    'input': {'a': a, 'b': b},
                    'result': float(calc_result.output)
                }
        
        class TestInput(BaseOperationInput):
            operation: Literal['process']
            a: float
            b: float
        
        async def process_numbers(ctx: RunContext[TestDeps], a: float, b: float) -> dict:
            return await ctx.deps.process_with_calculator(a, b)
        
        test_agent = create_agentool(
            name='test_processor',
            input_schema=TestInput,
            routing_config=RoutingConfig(
                operation_map={
                    'process': ('process_numbers', lambda x: {'a': x.a, 'b': x.b})
                }
            ),
            tools=[process_numbers],
            deps_type=TestDeps,
            dependencies=['calculator']
        )
        
        # Use mock in test
        injector = get_injector()
        mock_calc = MockAgenTool({'add': '10.0'})
        
        with injector.override(calculator=mock_calc):
            result = await injector.run('test_processor', {
                'operation': 'process',
                'a': 3,
                'b': 7
            })
            
            result_data = json.loads(result.output)
            assert result_data['result'] == 10.0
            assert len(mock_calc.call_history) == 1
```

### Advanced Fixtures

```python
import tempfile
import shutil
from pathlib import Path

@pytest.fixture(scope="session")
def temp_directory():
    """Create temporary directory for test session."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def isolated_registry():
    """Provide isolated registry for testing."""
    # Save current state
    from agentool.core.registry import AgenToolRegistry
    original_configs = AgenToolRegistry._configs.copy()
    
    # Clear for test
    AgenToolRegistry.clear()
    
    yield AgenToolRegistry
    
    # Restore original state
    AgenToolRegistry._configs = original_configs

@pytest.fixture
def performance_timer():
    """Fixture for timing operations."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.elapsed = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            if self.start_time:
                self.elapsed = time.time() - self.start_time
                return self.elapsed
            return None
        
        def __enter__(self):
            self.start()
            return self
        
        def __exit__(self, *args):
            self.stop()
    
    return Timer()

class TestWithAdvancedFixtures:
    """Test using advanced fixtures."""
    
    def test_isolated_registry(self, isolated_registry):
        """Test with isolated registry."""
        # Registry should be empty
        assert len(isolated_registry.list_names()) == 0
        
        # Register something
        config = AgenToolConfig(
            input_schema=CalculatorInput,
            routing_config=RoutingConfig()
        )
        isolated_registry.register('test_agent', config)
        
        # Should be available in this test
        assert 'test_agent' in isolated_registry.list_names()
        
        # But won't affect other tests
    
    @pytest.mark.asyncio
    async def test_performance_measurement(self, calculator_agent, performance_timer):
        """Test with performance measurement."""
        with performance_timer:
            result = await calculator_agent.run('{"operation": "add", "a": 5, "b": 3}')
        
        assert result.output == '8.0'
        assert performance_timer.elapsed < 0.1  # Should be fast
        print(f"Operation took {performance_timer.elapsed:.4f} seconds")
```

## Performance Testing

### Load Testing

```python
import asyncio
import time
from statistics import mean, median, stdev
from concurrent.futures import ThreadPoolExecutor

class TestPerformance:
    """Performance testing for AgenTool agents."""
    
    @pytest.mark.asyncio
    async def test_single_operation_performance(self, calculator_agent):
        """Test performance of single operations."""
        iterations = 100
        times = []
        
        for i in range(iterations):
            start_time = time.time()
            result = await calculator_agent.run(f'{{"operation": "add", "a": {i}, "b": {i+1}}}')
            end_time = time.time()
            
            times.append(end_time - start_time)
            assert float(result.output) == i + (i + 1)
        
        # Calculate statistics
        avg_time = mean(times)
        median_time = median(times)
        std_time = stdev(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"Performance Statistics for {iterations} operations:")
        print(f"  Average: {avg_time:.4f}s")
        print(f"  Median:  {median_time:.4f}s")
        print(f"  Std Dev: {std_time:.4f}s")
        print(f"  Min:     {min_time:.4f}s")
        print(f"  Max:     {max_time:.4f}s")
        
        # Performance assertions
        assert avg_time < 0.01, f"Average response time too high: {avg_time:.4f}s"
        assert median_time < 0.01, f"Median response time too high: {median_time:.4f}s"
        assert max_time < 0.05, f"Maximum response time too high: {max_time:.4f}s"
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, calculator_agent):
        """Test performance under concurrent load."""
        async def single_operation(i: int):
            return await calculator_agent.run(f'{{"operation": "multiply", "a": {i}, "b": 2}}')
        
        # Test with different concurrency levels
        for concurrency in [10, 50, 100]:
            start_time = time.time()
            tasks = [single_operation(i) for i in range(concurrency)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Check for errors
            errors = [r for r in results if isinstance(r, Exception)]
            successful = [r for r in results if not isinstance(r, Exception)]
            
            total_time = end_time - start_time
            throughput = len(successful) / total_time
            
            print(f"Concurrency {concurrency}:")
            print(f"  Total time: {total_time:.4f}s")
            print(f"  Successful: {len(successful)}")
            print(f"  Errors: {len(errors)}")
            print(f"  Throughput: {throughput:.2f} ops/sec")
            
            # Performance assertions
            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert total_time < concurrency * 0.01, f"Concurrent execution too slow"
            
            # Verify results
            for i, result in enumerate(successful):
                assert float(result.output) == i * 2
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, calculator_agent):
        """Test memory usage under sustained load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform sustained operations
        batch_size = 100
        num_batches = 10
        
        for batch in range(num_batches):
            tasks = []
            for i in range(batch_size):
                tasks.append(calculator_agent.run(f'{{"operation": "add", "a": {i}, "b": {batch}}}'))
            
            results = await asyncio.gather(*tasks)
            
            # Verify results
            for i, result in enumerate(results):
                assert float(result.output) == i + batch
            
            # Check memory usage periodically
            if batch % 3 == 0:
                current_memory = process.memory_info().rss
                memory_increase = current_memory - initial_memory
                print(f"Batch {batch}: Memory increase: {memory_increase / 1024 / 1024:.2f} MB")
        
        final_memory = process.memory_info().rss
        total_memory_increase = final_memory - initial_memory
        
        print(f"Total memory increase after {num_batches * batch_size} operations: {total_memory_increase / 1024 / 1024:.2f} MB")
        
        # Memory should not increase excessively
        assert total_memory_increase < 100 * 1024 * 1024, f"Memory increase too high: {total_memory_increase / 1024 / 1024:.2f} MB"
    
    @pytest.mark.asyncio
    async def test_stress_testing(self, calculator_agent):
        """Stress test with high load."""
        # Extended stress test
        duration = 30  # seconds
        max_concurrent = 20
        
        start_time = time.time()
        completed_operations = 0
        errors = []
        
        async def stress_worker(worker_id: int):
            """Individual stress test worker."""
            nonlocal completed_operations
            local_operations = 0
            
            while time.time() - start_time < duration:
                try:
                    operation_id = f"{worker_id}_{local_operations}"
                    result = await calculator_agent.run(f'{{"operation": "add", "a": {local_operations}, "b": {worker_id}}}')
                    expected = local_operations + worker_id
                    
                    if float(result.output) != expected:
                        errors.append(f"Worker {worker_id}, Op {local_operations}: Expected {expected}, got {result.output}")
                    
                    local_operations += 1
                    
                    # Small delay to prevent overwhelming
                    await asyncio.sleep(0.001)
                    
                except Exception as e:
                    errors.append(f"Worker {worker_id}: {str(e)}")
            
            return local_operations
        
        # Start workers
        workers = [stress_worker(i) for i in range(max_concurrent)]
        worker_results = await asyncio.gather(*workers, return_exceptions=True)
        
        total_time = time.time() - start_time
        completed_operations = sum(r for r in worker_results if isinstance(r, int))
        throughput = completed_operations / total_time
        
        print(f"Stress Test Results:")
        print(f"  Duration: {total_time:.2f}s")
        print(f"  Operations: {completed_operations}")
        print(f"  Throughput: {throughput:.2f} ops/sec")
        print(f"  Errors: {len(errors)}")
        
        # Stress test assertions
        assert len(errors) < completed_operations * 0.01, f"Error rate too high: {len(errors)}/{completed_operations}"
        assert throughput > 50, f"Throughput too low: {throughput:.2f} ops/sec"
        
        if errors:
            print("Sample errors:")
            for error in errors[:5]:
                print(f"  {error}")
```

### Benchmarking

```python
class TestBenchmarks:
    """Benchmark different AgenTool configurations."""
    
    @pytest.mark.asyncio
    async def test_benchmark_simple_vs_complex_schemas(self):
        """Compare performance of simple vs complex schemas."""
        
        # Simple schema
        class SimpleInput(BaseOperationInput):
            operation: Literal['add']
            a: float
            b: float
        
        # Complex schema
        from typing import Dict, List
        from pydantic import Field
        
        class ComplexInput(BaseOperationInput):
            operation: Literal['process']
            data: Dict[str, Any] = Field(description="Complex data structure")
            metadata: List[str] = Field(default_factory=list)
            config: Optional[Dict[str, str]] = None
            flags: Dict[str, bool] = Field(default_factory=dict)
        
        # Create agents
        async def simple_add(ctx, a: float, b: float) -> float:
            return a + b
        
        async def complex_process(ctx, data: Dict[str, Any], metadata: List[str], 
                                config: Dict[str, str], flags: Dict[str, bool]) -> Dict[str, Any]:
            return {
                "processed": True,
                "data_keys": len(data),
                "metadata_count": len(metadata),
                "config_items": len(config or {}),
                "flags": flags
            }
        
        simple_agent = create_agentool(
            name='simple_benchmark',
            input_schema=SimpleInput,
            routing_config=RoutingConfig(
                operation_map={'add': ('simple_add', lambda x: {'a': x.a, 'b': x.b})}
            ),
            tools=[simple_add]
        )
        
        complex_agent = create_agentool(
            name='complex_benchmark',
            input_schema=ComplexInput,
            routing_config=RoutingConfig(
                operation_map={'process': ('complex_process', lambda x: {
                    'data': x.data,
                    'metadata': x.metadata,
                    'config': x.config or {},
                    'flags': x.flags
                })}
            ),
            tools=[complex_process]
        )
        
        # Benchmark simple operations
        simple_times = []
        for _ in range(100):
            start = time.time()
            await simple_agent.run('{"operation": "add", "a": 1, "b": 2}')
            simple_times.append(time.time() - start)
        
        # Benchmark complex operations
        complex_input = {
            "operation": "process",
            "data": {"key1": "value1", "key2": "value2", "key3": 123},
            "metadata": ["meta1", "meta2"],
            "config": {"setting1": "value1"},
            "flags": {"enabled": True, "debug": False}
        }
        
        complex_times = []
        for _ in range(100):
            start = time.time()
            await complex_agent.run(json.dumps(complex_input))
            complex_times.append(time.time() - start)
        
        # Compare results
        simple_avg = mean(simple_times)
        complex_avg = mean(complex_times)
        overhead = complex_avg - simple_avg
        overhead_percent = (overhead / simple_avg) * 100
        
        print(f"Schema Complexity Benchmark:")
        print(f"  Simple schema avg:  {simple_avg:.6f}s")
        print(f"  Complex schema avg: {complex_avg:.6f}s")
        print(f"  Overhead:           {overhead:.6f}s ({overhead_percent:.1f}%)")
        
        # Overhead should be reasonable
        assert overhead_percent < 50, f"Schema complexity overhead too high: {overhead_percent:.1f}%"
```

## Test Organization

### Test Structure

```
tests/
├── unit/
│   ├── test_core_model.py
│   ├── test_core_manager.py
│   ├── test_core_registry.py
│   ├── test_core_injector.py
│   └── test_factory.py
├── integration/
│   ├── test_multi_agent.py
│   ├── test_dependency_injection.py
│   └── test_end_to_end.py
├── performance/
│   ├── test_load.py
│   ├── test_stress.py
│   └── test_benchmarks.py
├── fixtures/
│   ├── agents.py
│   ├── data.py
│   └── mocks.py
└── conftest.py
```

### Configuration Files

```python
# conftest.py
import pytest
import asyncio
from agentool.core.registry import AgenToolRegistry
from agentool.core.injector import get_injector

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def clean_state():
    """Clean AgenTool state between tests."""
    # Save current state
    original_configs = AgenToolRegistry._configs.copy()
    original_instances = get_injector()._instances.copy()
    
    yield
    
    # Restore clean state
    AgenToolRegistry._configs = original_configs
    get_injector()._instances = original_instances
    get_injector()._dependency_overrides.clear()

# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --strict-config
    --cov=agentool
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=90
asyncio_mode = auto
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    performance: marks tests as performance tests
    unit: marks tests as unit tests
```

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Test AgenTool

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
        pip install -e .
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=agentool --cov-report=xml
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
    
    - name: Run performance tests
      run: |
        pytest tests/performance/ -v -m "not slow"
    
    - name: Upload coverage to Codecov
      if: matrix.python-version == '3.11'
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  performance:
    runs-on: ubuntu-latest
    needs: test
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
        pip install -e .
    
    - name: Run full performance tests
      run: |
        pytest tests/performance/ -v --benchmark-json=benchmark.json
    
    - name: Store benchmark result
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: benchmark.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
```

This comprehensive testing guide provides everything needed to thoroughly test AgenTool-based applications, from simple unit tests to complex performance benchmarks and CI/CD integration.