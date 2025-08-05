"""
Factory functions for creating AgenTools.

This module provides the main factory function for creating AgenTool agents
and registering the model provider with pydantic-ai.
"""

from __future__ import annotations

import inspect
from typing import Type, List, Callable, Any, Optional, Dict, get_type_hints
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from .core.model import AgenToolModel
from .core.manager import AgenToolManager
from .core.registry import AgenToolRegistry, AgenToolConfig, RoutingConfig, ToolMetadata


def extract_tool_metadata(tool_func: Callable) -> ToolMetadata:
    """Extract metadata from a tool function.
    
    Args:
        tool_func: The tool function to analyze
        
    Returns:
        ToolMetadata with extracted information
    """
    # Get function name
    name = getattr(tool_func, '__name__', str(tool_func))
    
    # Get docstring
    doc = inspect.getdoc(tool_func)
    description = doc.split('\n')[0] if doc else None
    
    # Check if async
    is_async = inspect.iscoroutinefunction(tool_func)
    
    # Get signature
    sig = inspect.signature(tool_func)
    parameters = []
    parameter_types = {}
    
    # Get type hints for the function
    try:
        type_hints = get_type_hints(tool_func)
    except Exception:
        # If get_type_hints fails, fall back to annotations
        type_hints = {}
    
    for param_name, param in sig.parameters.items():
        # Skip 'ctx' parameter as it's always RunContext
        if param_name == 'ctx':
            continue
        
        parameters.append(param_name)
        
        # Try to get type annotation
        if param.annotation != inspect.Parameter.empty:
            type_str = str(param.annotation)
            # Clean up type string
            type_str = type_str.replace('typing.', '')
            parameter_types[param_name] = type_str
    
    # Get return type
    return_type = None
    return_type_annotation = None
    
    if sig.return_annotation != inspect.Signature.empty:
        return_type = str(sig.return_annotation).replace('typing.', '')
        # Try to get the actual type object
        return_type_annotation = type_hints.get('return', sig.return_annotation)
        # If it's still empty, set to None
        if return_type_annotation == inspect.Signature.empty:
            return_type_annotation = None
    
    return ToolMetadata(
        name=name,
        description=description,
        is_async=is_async,
        parameters=parameters,
        parameter_types=parameter_types,
        return_type=return_type,
        return_type_annotation=return_type_annotation
    )


def infer_output_type(tools_metadata: List[ToolMetadata]) -> Optional[Type[Any]]:
    """Infer the output type from tool return annotations.
    
    This function analyzes the return type annotations of all tools and:
    - Returns the common BaseModel type if all tools return the same type
    - Returns the BaseModel type if some tools return dict and others return that BaseModel
    - Returns None if tools have incompatible return types or no return types
    
    Args:
        tools_metadata: List of ToolMetadata from the tools
        
    Returns:
        The inferred output type or None if cannot be inferred
    """
    if not tools_metadata:
        return None
    
    # Collect all non-None return type annotations
    return_types = []
    for tool_meta in tools_metadata:
        if tool_meta.return_type_annotation is not None:
            # Skip 'Any' type
            if tool_meta.return_type_annotation != Any:
                # Check if it's a generic type (like Dict[str, Any])
                # Generic types have __origin__ attribute
                if hasattr(tool_meta.return_type_annotation, '__origin__'):
                    # Skip generic types as they can't be used with isinstance
                    continue
                return_types.append(tool_meta.return_type_annotation)
    
    if not return_types:
        return None
    
    # Check if all return types are the same
    first_type = return_types[0]
    
    # Skip built-in types (str, int, float, bool, etc.) as they don't need special handling
    # Only infer BaseModel types
    if first_type in (str, int, float, bool, dict, list, tuple, set):
        return None
    
    # Check if first_type is a BaseModel subclass
    try:
        if issubclass(first_type, BaseModel):
            # Check if all other types are either the same BaseModel or dict
            for return_type in return_types[1:]:
                if return_type != first_type and return_type != dict:
                    # Incompatible types
                    return None
            # All types are compatible
            return first_type
    except TypeError:
        # first_type is not a class (might be a generic, union, etc.)
        pass
    
    # Check if all types are exactly the same
    if all(rt == first_type for rt in return_types):
        return first_type
    
    # Types are not compatible for automatic inference
    return None


def create_agentool(
    name: str,
    input_schema: Type[BaseModel],
    routing_config: RoutingConfig,
    tools: List[Callable],
    system_prompt: Optional[str] = None,
    description: Optional[str] = None,
    deps_type: Optional[Type[Any]] = None,
    output_type: Optional[Type[Any]] = None,
    version: str = "1.0.0",
    tags: Optional[List[str]] = None,
    dependencies: Optional[List[str]] = None,
    examples: Optional[List[Dict[str, Any]]] = None,
    **agent_kwargs,
) -> Agent:
    """
    Factory function to create an AgenTool.
    
    This creates a pydantic-ai Agent configured with:
    - AgenToolModel as the LLM provider
    - AgenToolManager registered as the special routing tool
    - All provided tools registered on the agent
    - Configuration stored in the registry
    
    Args:
        name: The name of the AgenTool (e.g., 'storage', 'compute')
        input_schema: The Pydantic model for input validation
        routing_config: Configuration for routing operations to tools
        tools: List of tool functions to register
        system_prompt: Optional system prompt for the agent
        description: Optional description of the AgenTool
        deps_type: Optional type for dependency injection
        output_type: Optional type for output validation
        version: Version of the AgenTool
        tags: List of tags for categorization
        dependencies: List of required dependencies
        examples: List of example inputs/outputs
        **agent_kwargs: Additional keyword arguments passed to Agent
        
    Returns:
        A configured pydantic-ai Agent ready to use
        
    Example:
        >>> from pydantic import BaseModel
        >>> from agentool import create_agentool, RoutingConfig
        >>> 
        >>> class MyInput(BaseModel):
        ...     operation: str
        ...     data: str
        >>> 
        >>> async def process_data(ctx, data: str) -> dict:
        ...     return {"processed": data}
        >>> 
        >>> routing = RoutingConfig(
        ...     operation_map={
        ...         'process': ('process_data', lambda x: {'data': x.data})
        ...     }
        ... )
        >>> 
        >>> agent = create_agentool(
        ...     name='processor',
        ...     input_schema=MyInput,
        ...     routing_config=routing,
        ...     tools=[process_data]
        ... )
    """
    # Extract metadata from tools
    tools_metadata = [extract_tool_metadata(tool) for tool in tools]
    
    # If output_type is not provided, try to infer it from tool return annotations
    if output_type is None:
        output_type = infer_output_type(tools_metadata)
    
    # Create and register the configuration
    config = AgenToolConfig(
        input_schema=input_schema,
        routing_config=routing_config,
        output_type=output_type,
        description=description,
        version=version,
        tags=tags or [],
        tools_metadata=tools_metadata,
        dependencies=dependencies or [],
        examples=examples or []
    )
    AgenToolRegistry.register(name, config)
    
    # Create the agent with our custom model
    model = AgenToolModel(name)
    
    if system_prompt is None:
        system_prompt = f"Process {name} operations based on JSON input."
    
    # Create agent with all supported features
    agent_params = {
        'model': model,
        'system_prompt': system_prompt,
    }
    
    # Add optional parameters if provided
    if deps_type is not None:
        agent_params['deps_type'] = deps_type
    # Note: We don't pass output_type to Agent constructor because AgenTools
    # return JSON strings, not structured output. The output_type is used
    # internally by the AgenTool framework for validation.
    
    # Add any additional kwargs
    agent_params.update(agent_kwargs)
    
    agent = Agent(**agent_params)
    
    # Create a mapping of tool names to functions
    tool_functions = {}
    for tool_func in tools:
        tool_name = getattr(tool_func, '__name__', str(tool_func))
        tool_functions[tool_name] = tool_func
    
    # Create and register the manager as a special tool
    manager = AgenToolManager(name, config, tool_functions)
    
    # Create a wrapper that provides the schema
    @agent.tool(name='__agentool_manager__')
    async def agentool_manager_tool(ctx: RunContext[Any], **kwargs: Any) -> Any:
        """Special routing tool for AgenTool operations."""
        return await manager(ctx, **kwargs)
    
    # Keep reference to avoid unused warning
    _ = agentool_manager_tool
    
    # Override the tool's schema with the input schema
    # This ensures pydantic-ai validates against our schema
    tool = agent._function_toolset.tools['__agentool_manager__']
    if hasattr(tool, 'function_schema') and hasattr(tool.function_schema, '_json_schema_dict'):
        # Update the schema to use our input schema
        schema = manager.get_tool_schema()
        tool.function_schema._json_schema_dict = {
            'type': 'object',
            'properties': schema.get('properties', {}),
            'required': schema.get('required', []),
            'additionalProperties': False
        }
    
    # Register the actual tools
    for tool_func in tools:
        agent.tool(tool_func)
    
    # Register the agent with the global injector
    from .core.injector import get_injector
    injector = get_injector()
    injector.register(name, agent)
    
    return agent


def register_agentool_models() -> None:
    """
    Register the AgenTool model provider with pydantic-ai.
    
    This patches pydantic-ai's model inference to recognize
    'agentool:*' patterns and create appropriate model instances.
    
    Note:
        This function is called automatically when importing agentool.
        You typically don't need to call it manually.
        
    Example:
        >>> from agentool import register_agentool_models
        >>> register_agentool_models()  # Usually not needed
        >>> 
        >>> # Now you can use agentool:* model strings
        >>> from pydantic_ai import Agent
        >>> agent = Agent('agentool:storage')
    """
    from pydantic_ai.models import infer_model as original_infer_model
    from pydantic_ai import models
    
    def patched_infer_model(model):
        """Patched version that recognizes 'agentool:*' patterns."""
        if isinstance(model, str) and model.startswith('agentool:'):
            _, name = model.split(':', maxsplit=1)
            return AgenToolModel(name)
        return original_infer_model(model)
    
    # Monkey patch the infer_model function
    models.infer_model = patched_infer_model