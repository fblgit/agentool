#!/usr/bin/env python3
"""
Proper AgenTool example: Calculator using create_agentool factory.

This example demonstrates the correct way to create an AgenTool:
- Using the create_agentool factory
- Proper routing configuration
- Clean tool definitions
- Automatic schema generation from input model

Run this example:
    python src/examples/demos/calculator_agentool.py
"""

import asyncio
import json
from typing import Any, Literal
from pydantic import BaseModel, Field
from pydantic_ai import RunContext

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from src.agentool import create_agentool, register_agentool_models
from src.agentool.core.registry import RoutingConfig


# Register AgenTool models with pydantic-ai
register_agentool_models()


# Define input schema for calculator operations
class CalculatorInput(BaseModel):
    """Calculator operation input."""
    operation: Literal['add', 'subtract', 'multiply', 'divide', 'power', 'sqrt'] = Field(
        description="The mathematical operation to perform"
    )
    a: float = Field(description="First operand")
    b: float = Field(None, description="Second operand (not needed for sqrt)")


# Define calculator tools
async def calc_add(ctx: RunContext[Any], a: float, b: float) -> dict[str, Any]:
    """Add two numbers."""
    result = a + b
    return {
        "operation": "add",
        "a": a,
        "b": b,
        "result": result,
        "expression": f"{a} + {b} = {result}"
    }


async def calc_subtract(ctx: RunContext[Any], a: float, b: float) -> dict[str, Any]:
    """Subtract one number from another."""
    result = a - b
    return {
        "operation": "subtract",
        "a": a,
        "b": b,
        "result": result,
        "expression": f"{a} - {b} = {result}"
    }


async def calc_multiply(ctx: RunContext[Any], a: float, b: float) -> dict[str, Any]:
    """Multiply two numbers."""
    result = a * b
    return {
        "operation": "multiply",
        "a": a,
        "b": b,
        "result": result,
        "expression": f"{a} × {b} = {result}"
    }


async def calc_divide(ctx: RunContext[Any], a: float, b: float) -> dict[str, Any]:
    """Divide one number by another."""
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


async def calc_power(ctx: RunContext[Any], a: float, b: float) -> dict[str, Any]:
    """Raise a number to a power."""
    result = a ** b
    return {
        "operation": "power",
        "a": a,
        "b": b,
        "result": result,
        "expression": f"{a}^{b} = {result}"
    }


async def calc_sqrt(ctx: RunContext[Any], a: float) -> dict[str, Any]:
    """Calculate the square root of a number."""
    if a < 0:
        return {
            "operation": "sqrt",
            "value": a,
            "error": "Cannot calculate square root of negative number",
            "expression": f"√{a} = undefined (in real numbers)"
        }
    
    result = a ** 0.5
    return {
        "operation": "sqrt",
        "value": a,
        "result": result,
        "expression": f"√{a} = {result}"
    }


# Create routing configuration
routing_config = RoutingConfig(
    operation_field='operation',
    operation_map={
        'add': ('calc_add', lambda x: {'a': x.a, 'b': x.b}),
        'subtract': ('calc_subtract', lambda x: {'a': x.a, 'b': x.b}),
        'multiply': ('calc_multiply', lambda x: {'a': x.a, 'b': x.b}),
        'divide': ('calc_divide', lambda x: {'a': x.a, 'b': x.b}),
        'power': ('calc_power', lambda x: {'a': x.a, 'b': x.b}),
        'sqrt': ('calc_sqrt', lambda x: {'a': x.a}),
    }
)


# Create the calculator agent using the factory
calculator = create_agentool(
    name='calculator',
    input_schema=CalculatorInput,
    routing_config=routing_config,
    tools=[calc_add, calc_subtract, calc_multiply, calc_divide, calc_power, calc_sqrt],
    system_prompt="You are a calculator that performs mathematical operations.",
    description="A calculator supporting basic arithmetic operations",
    version="1.0.0",
    tags=["math", "calculator", "arithmetic"],
    examples=[
        {
            "description": "Addition example",
            "input": {"operation": "add", "a": 10, "b": 5},
            "output": {"operation": "add", "a": 10, "b": 5, "result": 15, "expression": "10 + 5 = 15"}
        },
        {
            "description": "Square root example",
            "input": {"operation": "sqrt", "a": 25},
            "output": {"operation": "sqrt", "value": 25, "result": 5.0, "expression": "√25 = 5.0"}
        }
    ]
)


async def main():
    """Demonstrate the calculator AgenTool."""
    print("=== Calculator AgenTool Example (Proper Implementation) ===\n")
    
    # Example calculations
    examples = [
        {"operation": "add", "a": 10, "b": 5},
        {"operation": "subtract", "a": 20, "b": 8},
        {"operation": "multiply", "a": 7, "b": 6},
        {"operation": "divide", "a": 100, "b": 4},
        {"operation": "divide", "a": 10, "b": 0},  # Division by zero
        {"operation": "power", "a": 2, "b": 8},
        {"operation": "sqrt", "a": 144},
        {"operation": "sqrt", "a": -4},  # Negative square root
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['operation'].capitalize()}:")
        print(f"   Input: {json.dumps(example)}")
        result = await calculator.run(json.dumps(example))
        print(f"   Output: {result.output}\n")
    
    # Show registered information
    print("=== Registry Information ===")
    from src.agentool.core.registry import AgenToolRegistry
    
    config = AgenToolRegistry.get('calculator')
    if config:
        print(f"Description: {config.description}")
        print(f"Version: {config.version}")
        print(f"Tags: {', '.join(config.tags)}")
        print(f"Operations: {', '.join(config.routing_config.operation_map.keys())}")
        print(f"Tools: {', '.join(t.name for t in config.tools_metadata)}")
    
    # Show detailed registry record
    print("\n=== Detailed Registry Record ===")
    detailed = AgenToolRegistry.list_detailed()
    for agentool in detailed:
        if agentool['name'] == 'calculator':
            print(json.dumps(agentool, indent=2))
    
    # Show tools information
    print("\n=== Detailed Tools Information ===")
    tools_info = AgenToolRegistry.get_tools_info('calculator')
    if tools_info:
        print(json.dumps(tools_info, indent=2))
    
    # Show operations mapping
    print("\n=== Operations Mapping ===")
    operations = AgenToolRegistry.get_operations('calculator')
    if operations:
        print(json.dumps(operations, indent=2))
    
    # Show input schema
    print("\n=== Input Schema ===")
    schema = AgenToolRegistry.get_schema('calculator')
    if schema:
        print(json.dumps(schema, indent=2))
    
    # Show dependency graph (if any dependencies)
    print("\n=== Dependency Graph ===")
    dep_graph = AgenToolRegistry.generate_dependency_graph()
    print(json.dumps(dep_graph, indent=2))
    
    # Export full catalog entry
    print("\n=== Full Catalog Export (Calculator Entry) ===")
    catalog = AgenToolRegistry.export_catalog()
    for agentool in catalog['agentools']:
        if agentool['name'] == 'calculator':
            print(json.dumps({
                "version": catalog['version'],
                "generated_at": catalog['generated_at'],
                "calculator_entry": agentool
            }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())