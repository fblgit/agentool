"""
Configuration Management AgenTool - Provides structured configuration management.

This toolkit builds on top of storage_kv to provide a high-level configuration
management interface with environment variable support, type validation,
and hierarchical configuration patterns.

Features:
- Environment variable integration
- JSON, YAML, and environment file format support
- Hierarchical configuration (namespace.key patterns)
- Type validation and conversion
- Configuration reloading and validation
- Default value support
- Configuration schema validation

Example Usage:
    >>> from agentoolkit.system.config import create_config_agent
    >>> from agentool.core.injector import get_injector
    >>> 
    >>> # Create and register the agent
    >>> agent = create_config_agent()
    >>> 
    >>> # Use through injector
    >>> injector = get_injector()
    >>> result = await injector.run('config', {
    ...     "operation": "set",
    ...     "key": "database.host",
    ...     "value": "localhost",
    ...     "namespace": "app"
    ... })
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List, Literal, Union
from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_ai import RunContext

from agentool.base import BaseOperationInput
from agentool import create_agentool
from agentool.core.registry import RoutingConfig
from agentool.core.injector import get_injector


class ConfigInput(BaseOperationInput):
    """Input schema for configuration management operations."""
    operation: Literal['get', 'set', 'delete', 'list', 'reload', 'validate', 'load_file', 'save_file'] = Field(
        description="The configuration operation to perform"
    )
    key: Optional[str] = Field(None, description="Configuration key (supports dot notation like 'db.host')")
    value: Optional[Any] = Field(None, description="Configuration value (JSON serializable)")
    namespace: str = Field(default="app", description="Configuration namespace")
    format: Literal['json', 'yaml', 'env'] = Field(default="json", description="Configuration format")
    file_path: Optional[str] = Field(None, description="File path for load_file/save_file operations")
    default: Optional[Any] = Field(None, description="Default value if key not found")
    env_prefix: Optional[str] = Field(None, description="Environment variable prefix for loading")
    validation_schema: Optional[Dict[str, Any]] = Field(None, description="JSON schema for validation")
    
    @field_validator('key')
    def validate_key(cls, v, info):
        """Validate that key is provided for operations that require it."""
        operation = info.data.get('operation')
        if operation in ['get', 'set', 'delete'] and not v:
            raise ValueError(f"key is required for {operation} operation")
        return v
    
    @field_validator('value')
    def validate_value(cls, v, info):
        """Validate that value is provided for set operation."""
        operation = info.data.get('operation')
        # Note: None is a valid configuration value
        return v
    
    @field_validator('validation_schema')
    def validate_schema(cls, v, info):
        """Validate that schema is provided for validate operation."""
        operation = info.data.get('operation')
        if operation == 'validate' and not v:
            raise ValueError("validation_schema is required for validate operation")
        return v


class ConfigOutput(BaseModel):
    """Structured output for configuration operations."""
    success: bool = Field(default=True, description="Whether the operation succeeded")
    operation: str = Field(description="The operation that was performed")
    key: Optional[str] = Field(None, description="The key that was operated on")
    namespace: str = Field(description="The namespace used")
    message: str = Field(description="Human-readable result message")
    data: Optional[Any] = Field(None, description="Operation-specific data")


def _parse_hierarchical_key(key: str) -> List[str]:
    """
    Parse a hierarchical key like 'database.host.port' into parts.
    
    Args:
        key: The hierarchical key
        
    Returns:
        List of key parts
    """
    return key.split('.')


def _get_nested_value(data: Dict[str, Any], key_parts: List[str], default: Any = None) -> Any:
    """
    Get a value from nested dictionary using key parts.
    
    Args:
        data: The dictionary to search
        key_parts: List of key parts
        default: Default value if not found
        
    Returns:
        The found value or default
    """
    current = data
    for part in key_parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def _set_nested_value(data: Dict[str, Any], key_parts: List[str], value: Any) -> None:
    """
    Set a value in nested dictionary using key parts.
    
    Args:
        data: The dictionary to modify
        key_parts: List of key parts
        value: The value to set
    """
    current = data
    for part in key_parts[:-1]:
        if part not in current:
            current[part] = {}
        elif not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[key_parts[-1]] = value


def _delete_nested_value(data: Dict[str, Any], key_parts: List[str]) -> bool:
    """
    Delete a value from nested dictionary using key parts.
    
    Args:
        data: The dictionary to modify
        key_parts: List of key parts
        
    Returns:
        True if the key was found and deleted, False otherwise
    """
    current = data
    for part in key_parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    
    if isinstance(current, dict) and key_parts[-1] in current:
        del current[key_parts[-1]]
        return True
    return False


async def config_get(ctx: RunContext[Any], key: str, namespace: str, default: Any) -> ConfigOutput:
    """
    Get a configuration value.
    
    Args:
        ctx: Runtime context
        key: The configuration key (supports hierarchical keys)
        namespace: The namespace to use
        default: Default value if key not found
        
    Returns:
        ConfigOutput with the configuration value
    """
    try:
        injector = get_injector()
        
        # Get the raw configuration data from storage_kv
        try:
            storage_result = await injector.run('storage_kv', {
                "operation": "get",
                "key": f"config:{namespace}",
                "namespace": "config"
            })
            
            # storage_kv now returns typed output
            if storage_result.success:
                config_data = storage_result.data["value"]
            else:
                # Key not found in storage_kv - no config namespace exists
                # Check if we have a default to return
                if default is not None:
                    return ConfigOutput(
                        success=True,  # Success - we have a default to return
                        operation="get",
                        key=key,
                        namespace=namespace,
                        message=f"Configuration key '{key}' not found, returning default",
                        data={
                            "value": default,
                            "exists": False,
                            "used_default": True
                        }
                    )
                else:
                    return ConfigOutput(
                        success=False,  # Discovery operation - key not found, no default
                        operation="get",
                        key=key,
                        namespace=namespace,
                        message=f"Configuration key '{key}' not found",
                        data={}  # Empty data when not found
                    )
        except Exception:
            # No configuration exists
            # Check if we have a default to return
            if default is not None:
                return ConfigOutput(
                    success=True,  # Success - we have a default to return
                    operation="get",
                    key=key,
                    namespace=namespace,
                    message=f"Configuration key '{key}' not found, returning default",
                    data={
                        "value": default,
                        "exists": False,
                        "used_default": True
                    }
                )
            else:
                return ConfigOutput(
                    success=False,  # Discovery operation - key not found, no default
                    operation="get",
                    key=key,
                    namespace=namespace,
                    message=f"Configuration key '{key}' not found",
                    data={}  # Empty data when not found
                )
        if not isinstance(config_data, dict):
            config_data = {}
        
        # Handle hierarchical keys
        key_parts = _parse_hierarchical_key(key)
        value = _get_nested_value(config_data, key_parts, default)
        exists = _get_nested_value(config_data, key_parts, "__NOT_FOUND__") != "__NOT_FOUND__"
        
        # Record business metrics
        try:
            await injector.run('metrics', {
                "operation": "increment",
                "name": "agentool.config.reads.total",
                "value": 1
            })
            if exists:
                await injector.run('metrics', {
                    "operation": "increment",
                    "name": "agentool.config.reads.found",
                    "value": 1
                })
            else:
                await injector.run('metrics', {
                    "operation": "increment",
                    "name": "agentool.config.reads.not_found",
                    "value": 1
                })
        except:
            pass  # Ignore metrics errors
        
        # Return appropriate response when key doesn't exist
        if not exists:
            # If we have a non-None default, this is successful - we're returning the requested default
            if default is not None:
                return ConfigOutput(
                    success=True,  # Success - we have a default to return
                    operation="get",
                    key=key,
                    namespace=namespace,
                    message=f"Configuration key '{key}' not found, returning default",
                    data={
                        "value": default,
                        "exists": False,
                        "used_default": True
                    }
                )
            else:
                # No default or None default - this is a failed discovery
                return ConfigOutput(
                    success=False,  # Discovery operation - key not found, no default
                    operation="get",
                    key=key,
                    namespace=namespace,
                    message=f"Configuration key '{key}' not found",
                    data={}  # Empty data when not found and no default
                )
        
        return ConfigOutput(
            success=True,
            operation="get",
            key=key,
            namespace=namespace,
            message=f"Successfully retrieved configuration key '{key}'",
            data={
                "value": value,
                "exists": True,
                "used_default": False
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Error retrieving configuration key '{key}': {str(e)}") from e


async def config_set(ctx: RunContext[Any], key: str, value: Any, namespace: str) -> ConfigOutput:
    """
    Set a configuration value.
    
    Args:
        ctx: Runtime context
        key: The configuration key (supports hierarchical keys)
        value: The value to set
        namespace: The namespace to use
        
    Returns:
        ConfigOutput with the set operation result
    """
    try:
        injector = get_injector()
        
        # Get existing configuration data
        try:
            storage_result = await injector.run('storage_kv', {
                "operation": "get",
                "key": f"config:{namespace}",
                "namespace": "config"
            })
            
            # storage_kv now returns typed output
            if storage_result.success:
                config_data = storage_result.data["value"]
                if not isinstance(config_data, dict):
                    config_data = {}
            else:
                # No existing config, create new
                config_data = {}
        except Exception:
            # No existing config, create new
            config_data = {}
        
        # Set the value using hierarchical key
        key_parts = _parse_hierarchical_key(key)
        _set_nested_value(config_data, key_parts, value)
        
        # Store the updated configuration
        store_result = await injector.run('storage_kv', {
            "operation": "set",
            "key": f"config:{namespace}",
            "value": config_data,
            "namespace": "config"
        })
        
        # Record business metrics
        try:
            await injector.run('metrics', {
                "operation": "increment",
                "name": "agentool.config.writes.total",
                "value": 1
            })
            await injector.run('metrics', {
                "operation": "set",
                "name": "agentool.config.keys.total",
                "value": len(str(config_data).split('"'))  # Rough estimate of config keys
            })
        except:
            pass  # Ignore metrics errors
        
        return ConfigOutput(
            success=True,
            operation="set",
            key=key,
            namespace=namespace,
            message=f"Successfully set configuration key '{key}'",
            data={
                "value": value,
                "key_parts": key_parts,
                "namespace_size": len(str(config_data))
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Error setting configuration key '{key}': {str(e)}") from e


async def config_delete(ctx: RunContext[Any], key: str, namespace: str) -> ConfigOutput:
    """
    Delete a configuration value.
    
    Args:
        ctx: Runtime context
        key: The configuration key to delete
        namespace: The namespace to use
        
    Returns:
        ConfigOutput with the delete operation result
    """
    try:
        injector = get_injector()
        
        # Get existing configuration data
        try:
            storage_result = await injector.run('storage_kv', {
                "operation": "get",
                "key": f"config:{namespace}",
                "namespace": "config"
            })
            
            # storage_kv now returns typed output
            if storage_result.success:
                config_data = storage_result.data["value"]
            else:
                # No configuration exists
                return ConfigOutput(
                    success=True,  # Delete is idempotent
                    operation="delete",
                    key=key,
                    namespace=namespace,
                    message=f"Configuration key '{key}' already does not exist",
                    data={"deleted": False, "existed": False}
                )
        except Exception:
            # No configuration exists
            return ConfigOutput(
                success=True,  # Delete is idempotent
                operation="delete",
                key=key,
                namespace=namespace,
                message=f"Configuration key '{key}' already does not exist",
                data={"deleted": False, "existed": False}
            )
        if not isinstance(config_data, dict):
            config_data = {}
        
        # Delete the value using hierarchical key
        key_parts = _parse_hierarchical_key(key)
        deleted = _delete_nested_value(config_data, key_parts)
        
        if deleted:
            # Store the updated configuration
            await injector.run('storage_kv', {
                "operation": "set",
                "key": f"config:{namespace}",
                "value": config_data,
                "namespace": "config"
            })
        
        return ConfigOutput(
            success=True,
            operation="delete",
            key=key,
            namespace=namespace,
            message=f"Configuration key '{key}' {'deleted' if deleted else 'did not exist'}",
            data={"deleted": deleted, "existed": deleted}
        )
        
    except Exception as e:
        raise RuntimeError(f"Error deleting configuration key '{key}': {str(e)}") from e


async def config_list(ctx: RunContext[Any], namespace: str, key: Optional[str]) -> ConfigOutput:
    """
    List configuration keys or get all configuration.
    
    Args:
        ctx: Runtime context
        namespace: The namespace to list
        key: Optional key prefix to filter by
        
    Returns:
        ConfigOutput with the configuration listing
    """
    try:
        injector = get_injector()
        
        # Get the configuration data
        try:
            storage_result = await injector.run('storage_kv', {
                "operation": "get",
                "key": f"config:{namespace}",
                "namespace": "config"
            })
            
            # storage_kv now returns typed output
            if storage_result.success:
                config_data = storage_result.data["value"]
            else:
                # No configuration exists
                return ConfigOutput(
                    success=True,  # List always succeeds even if empty
                    operation="list",
                    key=key,
                    namespace=namespace,
                    message=f"No configuration found in namespace '{namespace}'",
                    data={"config": {}, "keys": [], "count": 0}
                )
        except Exception:
            # No configuration exists
            return ConfigOutput(
                success=True,  # List always succeeds even if empty
                operation="list",
                key=key,
                namespace=namespace,
                message=f"No configuration found in namespace '{namespace}'",
                data={"config": {}, "keys": [], "count": 0}
            )
        if not isinstance(config_data, dict):
            config_data = {}
        
        # Flatten the configuration to get all keys
        def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        flattened = flatten_dict(config_data)
        
        # Filter by key prefix if provided
        if key:
            filtered_keys = [k for k in flattened.keys() if k.startswith(key)]
            filtered_config = {k: flattened[k] for k in filtered_keys}
        else:
            filtered_keys = list(flattened.keys())
            filtered_config = flattened
        
        return ConfigOutput(
            success=True,
            operation="list",
            key=key,
            namespace=namespace,
            message=f"Found {len(filtered_keys)} configuration keys",
            data={
                "config": config_data,
                "flattened": filtered_config,
                "keys": sorted(filtered_keys),
                "count": len(filtered_keys)
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Error listing configuration: {str(e)}") from e


async def config_reload(ctx: RunContext[Any], namespace: str, env_prefix: Optional[str]) -> ConfigOutput:
    """
    Reload configuration from environment variables.
    
    Args:
        ctx: Runtime context
        namespace: The namespace to reload
        env_prefix: Optional prefix for environment variables
        
    Returns:
        ConfigOutput with the reload operation result
    """
    try:
        injector = get_injector()
        
        # Get environment variables
        env_vars = {}
        prefix = env_prefix or f"{namespace.upper()}_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase with dots
                config_key = key[len(prefix):].lower().replace('_', '.')
                
                # Try to parse as JSON, fall back to string
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    parsed_value = value
                
                env_vars[config_key] = parsed_value
        
        # Get existing configuration
        try:
            storage_result = await injector.run('storage_kv', {
                "operation": "get",
                "key": f"config:{namespace}",
                "namespace": "config"
            })
            
            # storage_kv now returns typed output
            if storage_result.success:
                config_data = storage_result.data["value"]
                if not isinstance(config_data, dict):
                    config_data = {}
            else:
                # No existing config, create new
                config_data = {}
        except Exception:
            # No existing config, create new
            config_data = {}
        
        # Merge environment variables
        for key, value in env_vars.items():
            key_parts = _parse_hierarchical_key(key)
            _set_nested_value(config_data, key_parts, value)
        
        # Store the updated configuration
        await injector.run('storage_kv', {
            "operation": "set",
            "key": f"config:{namespace}",
            "value": config_data,
            "namespace": "config"
        })
        
        # Record business metrics
        try:
            await injector.run('metrics', {
                "operation": "increment",
                "name": "agentool.config.reloads.total",
                "value": 1
            })
            await injector.run('metrics', {
                "operation": "observe",
                "name": "agentool.config.env_vars.loaded",
                "value": len(env_vars)
            })
        except:
            pass  # Ignore metrics errors
        
        return ConfigOutput(
            success=True,
            operation="reload",
            key=None,
            namespace=namespace,
            message=f"Reloaded {len(env_vars)} environment variables into configuration",
            data={
                "env_vars_loaded": env_vars,
                "count": len(env_vars),
                "prefix": prefix
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Error reloading configuration from environment: {str(e)}") from e


async def config_validate(ctx: RunContext[Any], namespace: str, validation_schema: Dict[str, Any]) -> ConfigOutput:
    """
    Validate configuration against a JSON schema.
    
    Args:
        ctx: Runtime context
        namespace: The namespace to validate
        validation_schema: JSON schema for validation
        
    Returns:
        ConfigOutput with the validation result
    """
    try:
        injector = get_injector()
        
        # Get the configuration data
        try:
            storage_result = await injector.run('storage_kv', {
                "operation": "get",
                "key": f"config:{namespace}",
                "namespace": "config"
            })
            
            # storage_kv now returns typed output
            if storage_result.success:
                config_data = storage_result.data["value"]
            else:
                raise ValueError(f"No configuration found to validate in namespace '{namespace}'")
        except Exception:
            raise ValueError(f"No configuration found to validate in namespace '{namespace}'")
        
        # Validate using jsonschema if available
        try:
            import jsonschema
            jsonschema.validate(config_data, validation_schema)
            
            # Record successful validation
            try:
                await injector.run('metrics', {
                    "operation": "increment",
                    "name": "agentool.config.validations.success",
                    "value": 1
                })
            except:
                pass  # Ignore metrics errors
            
            return ConfigOutput(
                success=True,
                operation="validate",
                key=None,
                namespace=namespace,
                message=f"Configuration validation passed",
                data={
                    "valid": True,
                    "schema": validation_schema,
                    "config_size": len(str(config_data))
                }
            )
            
        except ImportError:
            raise ImportError("jsonschema library not available for validation")
        
        except jsonschema.ValidationError as e:
            # Record failed validation
            try:
                await injector.run('metrics', {
                    "operation": "increment",
                    "name": "agentool.config.validations.failure",
                    "value": 1
                })
            except:
                pass  # Ignore metrics errors
            
            raise ValueError(f"Configuration validation failed: {e.message}") from e
        
    except Exception as e:
        raise RuntimeError(f"Error validating configuration: {str(e)}") from e


def create_config_agent():
    """
    Create and return the configuration management AgenTool.
    
    Returns:
        Agent configured for configuration management operations
    """
    config_routing = RoutingConfig(
        operation_field='operation',
        operation_map={
            'get': ('config_get', lambda x: {
                'key': x.key, 'namespace': x.namespace, 'default': x.default
            }),
            'set': ('config_set', lambda x: {
                'key': x.key, 'value': x.value, 'namespace': x.namespace
            }),
            'delete': ('config_delete', lambda x: {'key': x.key, 'namespace': x.namespace}),
            'list': ('config_list', lambda x: {'namespace': x.namespace, 'key': x.key}),
            'reload': ('config_reload', lambda x: {
                'namespace': x.namespace, 'env_prefix': x.env_prefix
            }),
            'validate': ('config_validate', lambda x: {
                'namespace': x.namespace, 'validation_schema': x.validation_schema
            }),
        }
    )
    
    return create_agentool(
        name='config',
        input_schema=ConfigInput,
        routing_config=config_routing,
        tools=[config_get, config_set, config_delete, config_list, config_reload, config_validate],
        output_type=ConfigOutput,
        use_typed_output=True,  # Enable typed output for config (Tier 2 - depends on storage_kv)
        system_prompt="Handle configuration management with hierarchical keys and environment variable support.",
        description="Configuration management with environment variable integration and validation",
        version="1.0.0",
        tags=["config", "configuration", "environment", "settings"],
        dependencies=["storage_kv"],
        examples=[
            {
                "description": "Set a hierarchical configuration value",
                "input": {
                    "operation": "set",
                    "key": "database.host",
                    "value": "localhost",
                    "namespace": "app"
                },
                "output": {
                    "operation": "set",
                    "key": "database.host",
                    "namespace": "app",
                    "message": "Successfully set configuration key 'database.host'"
                }
            },
            {
                "description": "Get a configuration value with default",
                "input": {
                    "operation": "get",
                    "key": "database.port",
                    "default": 5432,
                    "namespace": "app"
                },
                "output": {
                    "operation": "get",
                    "key": "database.port",
                    "namespace": "app",
                    "message": "Configuration key 'database.port' not found, returning default",
                    "data": {"value": 5432, "exists": False, "used_default": True}
                }
            },
            {
                "description": "List all configuration keys",
                "input": {"operation": "list", "namespace": "app"},
                "output": {
                    "operation": "list",
                    "namespace": "app",
                    "message": "Found 1 configuration keys",
                    "data": {
                        "keys": ["database.host"],
                        "count": 1,
                        "flattened": {"database.host": "localhost"}
                    }
                }
            }
        ]
    )


# Create the agent instance
agent = create_config_agent()