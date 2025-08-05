"""
AgenTool Manager for handling routing and payload transformation.

This module provides the AgenToolManager class, which is registered as a
special tool on agents to handle routing based on input schemas.

The manager is responsible for:
- Validating input against the AgenTool's schema
- Routing operations to the appropriate tool based on configuration
- Transforming input data to match tool signatures
- Calling the actual tool functions
- Handling errors gracefully

Example:
    >>> from agentool import AgenToolManager, AgenToolConfig
    >>> 
    >>> config = AgenToolConfig(input_schema=MySchema, routing_config=routing)
    >>> manager = AgenToolManager('my_tool', config, tool_functions)
    >>> result = await manager(ctx, operation='read', key='test')
"""

from __future__ import annotations

import json
from typing import Any, Dict, Callable, get_type_hints, Type, Optional
from pydantic import BaseModel, ValidationError
from pydantic_ai import RunContext

from .registry import AgenToolRegistry, AgenToolConfig


class AgenToolManager:
    """
    Handles routing and payload transformation for AgenTools.
    
    This is registered as a special tool ('__agentool_manager__') on agents
    and handles:
    1. Receiving the validated input from the model
    2. Determining which tool to call based on the routing config
    3. Transforming the payload for the target tool
    4. Calling the actual tool and returning the result
    """
    
    def __init__(self, name: str, config: AgenToolConfig, tool_functions: Dict[str, Callable] = None):
        """Initialize the manager.
        
        Args:
            name: The name of the AgenTool
            config: The AgenTool configuration
            tool_functions: Dictionary mapping tool names to their callable functions
        """
        self.name = name
        self.config = config
        self.input_schema = config.input_schema
        self.routing_config = config.routing_config
        self.tool_functions = tool_functions or {}
    
    async def __call__(self, ctx: RunContext[Any], **kwargs) -> Any:
        """
        Handle the routing logic when called as a tool.
        
        The kwargs contain the validated input data from the model.
        Pydantic-ai has already validated it against the tool's schema.
        """
        # Convert kwargs to input model instance for easier access
        try:
            input_data = self.input_schema(**kwargs)
        except Exception as e:
            return f"Error creating input model: {e}"
        
        # Get the operation value
        operation_field = self.routing_config.operation_field
        operation = getattr(input_data, operation_field, None)
        
        if operation is None:
            return f"Missing required field: {operation_field}"
        
        # Look up the routing
        route_info = self.routing_config.operation_map.get(operation)
        if not route_info:
            available_ops = list(self.routing_config.operation_map.keys())
            return f"Unknown operation '{operation}'. Available operations: {available_ops}"
        
        tool_name, transform_func = route_info
        
        # Transform the input to tool arguments
        try:
            tool_args = transform_func(input_data)
        except Exception as e:
            return f"Error transforming arguments for {tool_name}: {e}"
        
        # Call the actual tool function
        if tool_name not in self.tool_functions:
            return f"Tool '{tool_name}' not found in AgenTool manager. Available tools: {list(self.tool_functions.keys())}"
        
        try:
            tool_func = self.tool_functions[tool_name]
            # Call the tool function with the transformed arguments and context
            result = await tool_func(ctx, **tool_args)
            
            # If output_type is specified, validate and return the typed object
            if self.config.output_type is not None:
                # Check if output_type is a BaseModel subclass
                try:
                    is_basemodel_type = issubclass(self.config.output_type, BaseModel)
                except TypeError:
                    # output_type is not a class or is a built-in type
                    is_basemodel_type = False
                
                if is_basemodel_type:
                    # Handle Pydantic models
                    if isinstance(result, BaseModel):
                        # Validate it's the correct type
                        if not isinstance(result, self.config.output_type):
                            return f"Tool returned {type(result).__name__} but expected {self.config.output_type.__name__}"
                        # Return the typed object directly
                        return result
                    elif isinstance(result, dict):
                        # Try to create the output model from dict
                        try:
                            output_instance = self.config.output_type(**result)
                            return output_instance
                        except Exception as e:
                            return f"Error creating output model {self.config.output_type.__name__}: {e}"
                    else:
                        return f"Tool returned {type(result).__name__} but output_type expects {self.config.output_type.__name__}"
                else:
                    # For non-BaseModel types (str, int, etc.), validate type and return
                    if isinstance(result, self.config.output_type):
                        # For built-in types, return as JSON string
                        if isinstance(result, str):
                            return result
                        else:
                            return json.dumps(result)
                    else:
                        return f"Tool returned {type(result).__name__} but expected {self.config.output_type.__name__}"
            
            # Default behavior for no output_type
            # For AgenTools, we expect dict results that should be JSON-serialized
            # This maintains consistency with the JSON input/output pattern
            if isinstance(result, dict):
                return json.dumps(result)
            elif isinstance(result, BaseModel):
                return result.model_dump_json()
            return result
        except Exception as e:
            # Re-raise the exception so it propagates properly
            # This allows injector to track failures and clients to handle errors
            raise
    
    def get_tool_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for this manager tool.
        
        This is used by pydantic-ai to validate inputs.
        """
        # Get the schema from the input model
        schema = self.input_schema.model_json_schema()
        
        # Ensure we have properties
        if 'properties' not in schema:
            schema['properties'] = {}
        
        # Add required fields if specified
        if hasattr(self.input_schema, 'model_fields'):
            required = []
            for field_name, field_info in self.input_schema.model_fields.items():
                if field_info.is_required():
                    required.append(field_name)
            if required:
                schema['required'] = required
        
        return schema