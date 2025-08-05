"""
AgenTool - Deterministic Tool Execution Framework for pydantic-ai.

AgenTool provides a synthetic LLM model provider that enables deterministic,
schema-driven tool execution within the pydantic-ai framework. It bridges
the gap between LLM-powered agents and programmatic tool execution.

Key Features:
    - Schema-driven input validation using Pydantic models
    - Deterministic routing based on operation fields
    - Seamless integration with pydantic-ai's model registry
    - Cost-effective alternative to LLM calls for structured operations
    - Type-safe tool execution with proper error handling

Quick Start:
    >>> from agentool import create_agentool, RoutingConfig
    >>> from pydantic import BaseModel
    >>> 
    >>> class MyInput(BaseModel):
    ...     operation: str
    ...     data: str
    >>> 
    >>> routing = RoutingConfig(
    ...     operation_map={
    ...         'process': ('process_tool', lambda x: {'data': x.data})
    ...     }
    ... )
    >>> 
    >>> agent = create_agentool(
    ...     name='processor',
    ...     input_schema=MyInput,
    ...     routing_config=routing,
    ...     tools=[process_tool]
    ... )
    >>> 
    >>> # Use like any pydantic-ai agent
    >>> result = await agent.run('{"operation": "process", "data": "test"}')

For more examples and documentation, see the README.md file.
"""

from .core import (
    AgenToolModel,
    AgenToolManager, 
    AgenToolRegistry,
    AgenToolConfig,
    RoutingConfig,
)
from .core.injector import get_injector, InjectedDeps
from .base import BaseOperationInput
from .factory import create_agentool, register_agentool_models

__version__ = "1.0.0"

__all__ = [
    # Core components
    'AgenToolModel',
    'AgenToolManager',
    'AgenToolRegistry',
    'AgenToolConfig',
    'RoutingConfig',
    
    # Dependency injection
    'get_injector',
    'InjectedDeps',
    
    # Base schemas and factory
    'BaseOperationInput',
    'create_agentool',
    'register_agentool_models',
]


def setup() -> None:
    """Set up the AgenTool framework by registering model providers."""
    register_agentool_models()


# Auto-setup when imported
setup()