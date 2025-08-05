#!/usr/bin/env python3
"""
Condensed single-file AgenTool example: Calculator.

This example demonstrates:
- Using @agent.tool decorators for automatic schema generation
- Google-style docstrings with parameter descriptions
- Both @agent.tool (with RunContext) and @agent.tool_plain decorators
- Single parameter tool with simplified schema
- Complete working example using AgenToolModel

Run this example:
    python src/examples/demos/calculator_single_file.py
"""

import asyncio
import json
from typing import Any, Literal
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# Add parent directory to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from src.agentool import AgenToolModel, register_agentool_models
from src.agentool.core.registry import AgenToolRegistry, AgenToolConfig, RoutingConfig


# Register AgenTool models with pydantic-ai
register_agentool_models()


# Define input schema for calculator operations
class CalculatorInput(BaseModel):
    """Calculator operation input.
    
    This model defines the structure for calculator operations,
    demonstrating how a single parameter model gets simplified schema.
    """
    operation: Literal['add', 'subtract', 'multiply', 'divide', 'power', 'sqrt'] = Field(
        description="The mathematical operation to perform"
    )
    a: float = Field(description="First operand")
    b: float = Field(None, description="Second operand (not needed for sqrt)")


# First, we need to register the calculator configuration
# This will be done after we define all the tools

# Create the calculator agent with AgenTool model
calculator = Agent(
    model=AgenToolModel('calculator'),
    system_prompt="You are a calculator that performs mathematical operations.",
)


# Define tools using decorators with Google-style docstrings
@calculator.tool_plain(docstring_format='google', require_parameter_descriptions=True)
def add(a: float, b: float) -> dict[str, Any]:
    """Add two numbers together.
    
    Args:
        a: The first number to add
        b: The second number to add
    
    Returns:
        Dictionary with the operation result
    """
    result = a + b
    return {
        "operation": "add",
        "a": a,
        "b": b,
        "result": result,
        "expression": f"{a} + {b} = {result}"
    }


@calculator.tool_plain(docstring_format='google', require_parameter_descriptions=True)
def subtract(a: float, b: float) -> dict[str, Any]:
    """Subtract one number from another.
    
    Args:
        a: The number to subtract from (minuend)
        b: The number to subtract (subtrahend)
    
    Returns:
        Dictionary with the operation result
    """
    result = a - b
    return {
        "operation": "subtract",
        "a": a,
        "b": b,
        "result": result,
        "expression": f"{a} - {b} = {result}"
    }


@calculator.tool_plain(docstring_format='google', require_parameter_descriptions=True)
def multiply(a: float, b: float) -> dict[str, Any]:
    """Multiply two numbers.
    
    Args:
        a: The first factor
        b: The second factor
    
    Returns:
        Dictionary with the operation result
    """
    result = a * b
    return {
        "operation": "multiply",
        "a": a,
        "b": b,
        "result": result,
        "expression": f"{a} × {b} = {result}"
    }


@calculator.tool(docstring_format='google', require_parameter_descriptions=True)
async def divide(ctx: RunContext[None], a: float, b: float) -> dict[str, Any]:
    """Divide one number by another.
    
    This tool uses @agent.tool to demonstrate access to RunContext,
    though it's not strictly needed for this operation.
    
    Args:
        a: The dividend (number to be divided)
        b: The divisor (number to divide by)
    
    Returns:
        Dictionary with the operation result or error
    """
    if b == 0:
        return {
            "operation": "divide",
            "a": a,
            "b": b,
            "error": "Division by zero",
            "expression": f"{a} ÷ 0 = undefined"
        }
    
    result = a / b
    return {
        "operation": "divide",
        "a": a,
        "b": b,
        "result": result,
        "expression": f"{a} ÷ {b} = {result}"
    }


@calculator.tool_plain(docstring_format='google', require_parameter_descriptions=True)
def power(a: float, b: float) -> dict[str, Any]:
    """Raise a number to a power.
    
    Args:
        a: The base number
        b: The exponent
    
    Returns:
        Dictionary with the operation result
    """
    result = a ** b
    return {
        "operation": "power",
        "a": a,
        "b": b,
        "result": result,
        "expression": f"{a}^{b} = {result}"
    }


# Single parameter tool example - demonstrates simplified schema
class SqrtInput(BaseModel):
    """Square root operation input."""
    value: float = Field(description="The number to find the square root of")


@calculator.tool_plain
def sqrt(input_data: SqrtInput) -> dict[str, Any]:
    """Calculate the square root of a number.
    
    This demonstrates a single parameter tool where the schema
    is simplified to just the parameter's schema.
    """
    if input_data.value < 0:
        return {
            "operation": "sqrt",
            "value": input_data.value,
            "error": "Cannot calculate square root of negative number",
            "expression": f"√{input_data.value} = undefined (in real numbers)"
        }
    
    result = input_data.value ** 0.5
    return {
        "operation": "sqrt",
        "value": input_data.value,
        "result": result,
        "expression": f"√{input_data.value} = {result}"
    }


# Create a proper routing system for the calculator
# Since we're using decorators, we need to manually create the routing

# First, let's create a manager tool that routes based on the operation
@calculator.tool(name='__agentool_manager__')
async def route_operation(ctx: RunContext[None], **kwargs) -> Any:
    """Route calculator operations to appropriate tools."""
    # Extract operation from kwargs
    operation = kwargs.get('operation')
    a = kwargs.get('a')
    b = kwargs.get('b')
    
    # Route to appropriate tool
    if operation == 'add':
        return add(a, b)
    elif operation == 'subtract':
        return subtract(a, b)
    elif operation == 'multiply':
        return multiply(a, b)
    elif operation == 'divide':
        return await divide(ctx, a, b)
    elif operation == 'power':
        return power(a, b)
    elif operation == 'sqrt':
        return sqrt(SqrtInput(value=a))
    else:
        return {
            "error": f"Unknown operation: {operation}",
            "available_operations": ['add', 'subtract', 'multiply', 'divide', 'power', 'sqrt']
        }


# Override the tool's schema to accept CalculatorInput
tool = calculator._function_toolset.tools['__agentool_manager__']
if hasattr(tool, 'function_schema') and hasattr(tool.function_schema, '_json_schema_dict'):
    # Update the schema to use our input schema
    tool.function_schema._json_schema_dict = {
        'type': 'object',
        'properties': {
            'operation': {
                'type': 'string',
                'enum': ['add', 'subtract', 'multiply', 'divide', 'power', 'sqrt'],
                'description': 'The mathematical operation to perform'
            },
            'a': {'type': 'number', 'description': 'First operand'},
            'b': {'type': 'number', 'description': 'Second operand (not needed for sqrt)'}
        },
        'required': ['operation', 'a'],
        'additionalProperties': False
    }

# Now register the calculator configuration in the registry
routing_config = RoutingConfig(
    operation_field='operation',
    operation_map={
        'add': ('__agentool_manager__', lambda x: {'operation': 'add', 'a': x.a, 'b': x.b}),
        'subtract': ('__agentool_manager__', lambda x: {'operation': 'subtract', 'a': x.a, 'b': x.b}),
        'multiply': ('__agentool_manager__', lambda x: {'operation': 'multiply', 'a': x.a, 'b': x.b}),
        'divide': ('__agentool_manager__', lambda x: {'operation': 'divide', 'a': x.a, 'b': x.b}),
        'power': ('__agentool_manager__', lambda x: {'operation': 'power', 'a': x.a, 'b': x.b}),
        'sqrt': ('__agentool_manager__', lambda x: {'operation': 'sqrt', 'a': x.a}),
    }
)

calculator_config = AgenToolConfig(
    input_schema=CalculatorInput,
    routing_config=routing_config,
    description="A calculator that performs basic mathematical operations"
)

AgenToolRegistry.register('calculator', calculator_config)


async def main():
    """Demonstrate the calculator AgenTool."""
    print("=== Calculator AgenTool Example ===\n")
    
    # Example 1: Addition
    print("1. Addition:")
    input1 = {"operation": "add", "a": 10, "b": 5}
    result1 = await calculator.run(json.dumps(input1))
    print(f"Input: {json.dumps(input1)}")
    print(f"Output: {result1.output}\n")
    
    # Example 2: Division (with potential error)
    print("2. Division:")
    input2 = {"operation": "divide", "a": 20, "b": 4}
    result2 = await calculator.run(json.dumps(input2))
    print(f"Input: {json.dumps(input2)}")
    print(f"Output: {result2.output}\n")
    
    # Example 3: Division by zero
    print("3. Division by zero:")
    input3 = {"operation": "divide", "a": 10, "b": 0}
    result3 = await calculator.run(json.dumps(input3))
    print(f"Input: {json.dumps(input3)}")
    print(f"Output: {result3.output}\n")
    
    # Example 4: Square root (single parameter tool)
    print("4. Square root:")
    input4 = {"operation": "sqrt", "a": 25}
    result4 = await calculator.run(json.dumps(input4))
    print(f"Input: {json.dumps(input4)}")
    print(f"Output: {result4.output}\n")
    
    # Example 5: Power
    print("5. Power:")
    input5 = {"operation": "power", "a": 2, "b": 8}
    result5 = await calculator.run(json.dumps(input5))
    print(f"Input: {json.dumps(input5)}")
    print(f"Output: {result5.output}\n")
    
    # Show tool schemas
    print("=== Tool Schemas ===")
    print("\nThe tools are automatically registered with proper schemas")
    print("extracted from function signatures and docstrings.")
    print("\nTools registered:")
    for tool_name in calculator._function_toolset.tools:
        print(f"  - {tool_name}")


if __name__ == "__main__":
    asyncio.run(main())