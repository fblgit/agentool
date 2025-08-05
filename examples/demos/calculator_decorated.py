#!/usr/bin/env python3
"""
Decorator-based AgenTool example: Math Assistant.

This example demonstrates a cleaner approach using pydantic-ai's
decorator pattern with AgenTools, showcasing:
- Automatic schema extraction from docstrings
- Type hints for parameter validation
- Single-model input with @agent.tool decorators
- Google-style docstrings with griffe extraction

Run this example:
    python src/examples/demos/calculator_decorated.py
"""

import asyncio
import json
from typing import Any
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# Add parent directory to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from src.agentool import AgenToolModel, register_agentool_models


# Register AgenTool models
register_agentool_models()


# Create the math assistant agent
math_agent = Agent(
    model=AgenToolModel('math'),
    system_prompt="You are a mathematical assistant that performs calculations with detailed explanations.",
)


# Define a single model for all math operations
class MathOperation(BaseModel):
    """A mathematical operation request with detailed documentation."""
    
    expression: str = Field(
        description="The mathematical expression to evaluate (e.g., '2 + 3', '10 / 5', 'sqrt(16)')"
    )
    show_steps: bool = Field(
        default=False,
        description="Whether to show step-by-step calculation"
    )


@math_agent.tool(docstring_format='google', require_parameter_descriptions=True)
async def calculate(ctx: RunContext[None], operation: MathOperation) -> dict[str, Any]:
    """Evaluate a mathematical expression and return the result.
    
    This tool parses and evaluates basic mathematical expressions,
    supporting addition, subtraction, multiplication, division,
    power operations, and square roots.
    
    Args:
        operation: The mathematical operation to perform, containing
            the expression and whether to show steps.
    
    Returns:
        A dictionary containing:
        - expression: The original expression
        - result: The calculated result
        - steps: Step-by-step calculation (if requested)
        - type: The type of operation performed
    
    Examples:
        >>> calculate(MathOperation(expression="2 + 3"))
        {"expression": "2 + 3", "result": 5, "type": "addition"}
        
        >>> calculate(MathOperation(expression="sqrt(16)", show_steps=True))
        {"expression": "sqrt(16)", "result": 4, "steps": [...], "type": "sqrt"}
    """
    expr = operation.expression.strip()
    steps = []
    
    try:
        # Simple expression parser (in production, use a proper parser)
        # Check sqrt first since it's a special format
        if 'sqrt(' in expr:
            # Extract number from sqrt(number)
            import re
            match = re.search(r'sqrt\(([-+]?\d*\.?\d+)\)', expr)
            if not match:
                raise ValueError("Invalid sqrt format")
            value = float(match.group(1))
            if value < 0:
                return {
                    "expression": expr,
                    "error": "Square root of negative number",
                    "type": "sqrt"
                }
            result = value ** 0.5
            op_type = "sqrt"
            if operation.show_steps:
                steps = [
                    f"Parse expression: {expr}",
                    f"Identify operation: square root",
                    f"Extract value: {value}",
                    f"Check for negative: {value} ≥ 0 ✓",
                    f"Calculate: √{value} = {result}"
                ]
        
        elif '+' in expr:
            parts = expr.split('+')
            a, b = float(parts[0]), float(parts[1])
            result = a + b
            op_type = "addition"
            if operation.show_steps:
                steps = [
                    f"Parse expression: {expr}",
                    f"Identify operation: addition",
                    f"Extract operands: a={a}, b={b}",
                    f"Calculate: {a} + {b} = {result}"
                ]
        
        elif '-' in expr and not expr.startswith('-'):
            parts = expr.split('-')
            a, b = float(parts[0]), float(parts[1])
            result = a - b
            op_type = "subtraction"
            if operation.show_steps:
                steps = [
                    f"Parse expression: {expr}",
                    f"Identify operation: subtraction",
                    f"Extract operands: a={a}, b={b}",
                    f"Calculate: {a} - {b} = {result}"
                ]
        
        elif '*' in expr:
            parts = expr.split('*')
            a, b = float(parts[0]), float(parts[1])
            result = a * b
            op_type = "multiplication"
            if operation.show_steps:
                steps = [
                    f"Parse expression: {expr}",
                    f"Identify operation: multiplication",
                    f"Extract operands: a={a}, b={b}",
                    f"Calculate: {a} × {b} = {result}"
                ]
        
        elif '/' in expr:
            parts = expr.split('/')
            a, b = float(parts[0]), float(parts[1])
            if b == 0:
                return {
                    "expression": expr,
                    "error": "Division by zero",
                    "type": "division"
                }
            result = a / b
            op_type = "division"
            if operation.show_steps:
                steps = [
                    f"Parse expression: {expr}",
                    f"Identify operation: division",
                    f"Extract operands: a={a}, b={b}",
                    f"Check for division by zero: b={b} ✓",
                    f"Calculate: {a} ÷ {b} = {result}"
                ]
        
        elif '^' in expr or '**' in expr:
            separator = '**' if '**' in expr else '^'
            parts = expr.split(separator)
            a, b = float(parts[0]), float(parts[1])
            result = a ** b
            op_type = "power"
            if operation.show_steps:
                steps = [
                    f"Parse expression: {expr}",
                    f"Identify operation: exponentiation",
                    f"Extract operands: base={a}, exponent={b}",
                    f"Calculate: {a}^{b} = {result}"
                ]
        
        
        else:
            # Try to evaluate as a simple number
            result = float(expr)
            op_type = "number"
            if operation.show_steps:
                steps = [f"Parse as number: {result}"]
        
        response = {
            "expression": expr,
            "result": result,
            "type": op_type
        }
        
        if operation.show_steps and steps:
            response["steps"] = steps
            
        return response
        
    except (ValueError, IndexError) as e:
        return {
            "expression": expr,
            "error": f"Invalid expression: {str(e)}",
            "hint": "Supported formats: 'a + b', 'a - b', 'a * b', 'a / b', 'a ^ b', 'sqrt(x)'"
        }


# Create the routing manager for AgenTool
@math_agent.tool(name='__agentool_manager__')
async def math_router(ctx: RunContext[None], **kwargs) -> Any:
    """Route math operations through the calculate tool."""
    # For this simple example, we just forward to calculate
    # In a more complex scenario, you might route to different tools
    return await calculate(ctx, MathOperation(**kwargs))


# Override the manager tool's schema
tool = math_agent._function_toolset.tools['__agentool_manager__']
if hasattr(tool, 'function_schema') and hasattr(tool.function_schema, '_json_schema_dict'):
    # Use the MathOperation schema
    tool.function_schema._json_schema_dict = MathOperation.model_json_schema()


async def main():
    """Demonstrate the math assistant AgenTool."""
    print("=== Math Assistant AgenTool ===\n")
    print("This example shows automatic schema extraction from docstrings")
    print("and type hints using pydantic-ai's decorator pattern.\n")
    
    examples = [
        {"expression": "10 + 5"},
        {"expression": "20 - 8"},
        {"expression": "7 * 6"},
        {"expression": "100 / 4", "show_steps": True},
        {"expression": "2 ^ 8"},
        {"expression": "sqrt(144)", "show_steps": True},
        {"expression": "10 / 0"},  # Error case
        {"expression": "sqrt(-4)"},  # Error case
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. Expression: {example['expression']}")
        if example.get('show_steps'):
            print("   (with steps)")
        
        result = await math_agent.run(json.dumps(example))
        output = json.loads(result.output) if isinstance(result.output, str) else result.output
        
        if 'error' in output:
            print(f"   Error: {output['error']}")
        else:
            print(f"   Result: {output['result']}")
            if 'steps' in output:
                print("   Steps:")
                for step in output['steps']:
                    print(f"     - {step}")
        print()
    
    # Show the tool schema
    print("=== Tool Schema ===")
    print("The calculate tool has the following auto-generated schema:")
    print(json.dumps(MathOperation.model_json_schema(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())