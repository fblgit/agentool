"""
Key-Value Storage AgenTool - Provides in-memory key-value storage with TTL support.

This toolkit provides a Redis-compatible key-value storage interface with
time-to-live (TTL) support, namespacing, and pattern matching capabilities.

Features:
- Full CRUD operations for key-value pairs
- TTL (Time To Live) support with automatic expiration
- Namespace support for data isolation
- Pattern-based key matching
- Atomic operations
- Memory-efficient implementation
- Redis-compatible interface

Example Usage:
    >>> from agentoolkit.storage.kv import create_storage_kv_agent
    >>> from agentool.core.injector import get_injector
    >>> 
    >>> # Create and register the agent
    >>> agent = create_storage_kv_agent()
    >>> 
    >>> # Use through injector
    >>> injector = get_injector()
    >>> result = await injector.run('storage_kv', {
    ...     "operation": "set",
    ...     "key": "user:123",
    ...     "value": {"name": "Alice", "email": "alice@example.com"},
    ...     "ttl": 3600
    ... })
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Literal, Union
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import RunContext

from agentool.base import BaseOperationInput
from agentool import create_agentool
from agentool.core.registry import RoutingConfig


# Global storage backend (in production, this could be Redis, etc.)
_kv_storage: Dict[str, Dict[str, Any]] = {}  # namespace -> {key -> value}
_kv_expiry: Dict[str, Dict[str, float]] = {}  # namespace -> {key -> expiry_timestamp}


class StorageKvInput(BaseOperationInput):
    """Input schema for key-value storage operations."""
    operation: Literal['get', 'set', 'delete', 'exists', 'keys', 'clear', 'expire', 'ttl', 'get_metric'] = Field(
        description="The key-value operation to perform"
    )
    key: Optional[str] = Field(None, description="Key for the operation")
    value: Optional[Any] = Field(None, description="Value to store (JSON serializable)")
    ttl: Optional[int] = Field(None, description="Time to live in seconds")
    pattern: Optional[str] = Field(None, description="Pattern for keys operation (supports * wildcard)")
    namespace: str = Field(default="default", description="Key namespace for data isolation")
    
    @field_validator('key')
    def validate_key(cls, v, info):
        """Validate that key is provided for operations that require it."""
        operation = info.data.get('operation')
        if operation in ['get', 'set', 'delete', 'exists', 'expire', 'ttl', 'get_metric'] and not v:
            raise ValueError(f"key is required for {operation} operation")
        return v
    
    @field_validator('value')
    def validate_value(cls, v, info):
        """Validate that value is provided for set operation."""
        operation = info.data.get('operation')
        # Note: None is a valid value to store, so we don't reject it
        # The 'value' field is Optional[Any] which includes None
        return v
    
    @field_validator('ttl')
    def validate_ttl(cls, v, info):
        """Validate TTL for operations that require it."""
        operation = info.data.get('operation')
        if operation == 'expire':
            if v is None:
                raise ValueError("ttl is required for expire operation")
            if v <= 0:
                raise ValueError("ttl must be positive for expire operation")
        elif operation == 'set' and v is not None and v <= 0:
            raise ValueError("ttl must be positive when provided")
        return v


class StorageKvOutput(BaseModel):
    """Structured output for key-value storage operations."""
    success: bool = Field(default=True, description="Whether the operation succeeded")
    operation: str = Field(description="The operation that was performed")
    key: Optional[str] = Field(None, description="The key that was operated on")
    namespace: str = Field(description="The namespace used")
    message: str = Field(description="Human-readable result message")
    data: Optional[Any] = Field(None, description="Operation-specific data")


def _cleanup_expired_keys(namespace: str) -> int:
    """
    Clean up expired keys in a namespace.
    
    Args:
        namespace: The namespace to clean up
        
    Returns:
        Number of keys that were expired and removed
    """
    if namespace not in _kv_expiry:
        return 0
    
    current_time = time.time()
    expired_keys = []
    
    for key, expiry_time in _kv_expiry[namespace].items():
        if current_time >= expiry_time:
            expired_keys.append(key)
    
    # Remove expired keys
    for key in expired_keys:
        if namespace in _kv_storage and key in _kv_storage[namespace]:
            del _kv_storage[namespace][key]
        del _kv_expiry[namespace][key]
    
    return len(expired_keys)


def _is_key_expired(namespace: str, key: str) -> bool:
    """
    Check if a key has expired.
    
    Args:
        namespace: The namespace to check
        key: The key to check
        
    Returns:
        True if the key has expired, False otherwise
    """
    if namespace not in _kv_expiry or key not in _kv_expiry[namespace]:
        return False
    
    return time.time() >= _kv_expiry[namespace][key]


async def kv_get(ctx: RunContext[Any], key: str, namespace: str) -> StorageKvOutput:
    """
    Get a value by key.
    
    Args:
        ctx: Runtime context
        key: The key to retrieve
        namespace: The namespace to use
        
    Returns:
        StorageKvOutput with the value or indication that key doesn't exist
    """
    try:
        # Clean up expired keys in namespace
        expired_count = _cleanup_expired_keys(namespace)
        
        # Check if key exists and is not expired
        if (namespace in _kv_storage and 
            key in _kv_storage[namespace] and 
            not _is_key_expired(namespace, key)):
            
            value = _kv_storage[namespace][key]
            
            # Get TTL if it exists
            ttl_remaining = None
            if namespace in _kv_expiry and key in _kv_expiry[namespace]:
                ttl_remaining = max(0, int(_kv_expiry[namespace][key] - time.time()))
            
            return StorageKvOutput(
                success=True,
                operation="get",
                key=key,
                namespace=namespace,
                message=f"Successfully retrieved value for key '{key}'",
                data={
                    "value": value,
                    "exists": True,
                    "ttl_remaining": ttl_remaining,
                    "expired_keys_cleaned": expired_count
                }
            )
        else:
            # Discovery operation - return success=False instead of raising
            return StorageKvOutput(
                success=False,
                operation="get",
                key=key,
                namespace=namespace,
                message=f"Key '{key}' not found in namespace '{namespace}' or has expired",
                data=None
            )
            
    except Exception as e:
        raise RuntimeError(f"Error retrieving key '{key}': {str(e)}") from e


async def kv_set(ctx: RunContext[Any], key: str, value: Any, namespace: str, ttl: Optional[int]) -> StorageKvOutput:
    """
    Set a key-value pair with optional TTL.
    
    Args:
        ctx: Runtime context
        key: The key to set
        value: The value to store
        namespace: The namespace to use
        ttl: Optional time to live in seconds
        
    Returns:
        StorageKvOutput with the set operation result
    """
    try:
        # Initialize namespace if it doesn't exist
        if namespace not in _kv_storage:
            _kv_storage[namespace] = {}
        if namespace not in _kv_expiry:
            _kv_expiry[namespace] = {}
        
        # Store the value
        _kv_storage[namespace][key] = value
        
        # Set TTL if provided
        if ttl is not None:
            _kv_expiry[namespace][key] = time.time() + ttl
        elif key in _kv_expiry[namespace]:
            # Remove existing TTL if no new TTL provided
            del _kv_expiry[namespace][key]
        
        return StorageKvOutput(
            success=True,
            operation="set",
            key=key,
            namespace=namespace,
            message=f"Successfully set key '{key}'" + (f" with TTL {ttl}s" if ttl else ""),
            data={
                "stored": True,
                "ttl": ttl,
                "namespace_size": len(_kv_storage[namespace])
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Error setting key '{key}': {str(e)}") from e


async def kv_delete(ctx: RunContext[Any], key: str, namespace: str) -> StorageKvOutput:
    """
    Delete a key.
    
    Args:
        ctx: Runtime context
        key: The key to delete
        namespace: The namespace to use
        
    Returns:
        StorageKvOutput with the delete operation result
    """
    try:
        existed = False
        
        # Check and delete from storage
        if namespace in _kv_storage and key in _kv_storage[namespace]:
            del _kv_storage[namespace][key]
            existed = True
        
        # Remove TTL if it exists
        if namespace in _kv_expiry and key in _kv_expiry[namespace]:
            del _kv_expiry[namespace][key]
        
        return StorageKvOutput(
            success=True,
            operation="delete",
            key=key,
            namespace=namespace,
            message=f"Key '{key}' {'deleted' if existed else 'did not exist'}",
            data={
                "deleted": existed,
                "existed": existed
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Error deleting key '{key}': {str(e)}") from e


async def kv_exists(ctx: RunContext[Any], key: str, namespace: str) -> StorageKvOutput:
    """
    Check if a key exists.
    
    Args:
        ctx: Runtime context
        key: The key to check
        namespace: The namespace to use
        
    Returns:
        StorageKvOutput with the existence check result
    """
    try:
        exists = (namespace in _kv_storage and 
                 key in _kv_storage[namespace] and 
                 not _is_key_expired(namespace, key))
        
        return StorageKvOutput(
            success=True,  # Always success - it's a query
            operation="exists",
            key=key,
            namespace=namespace,
            message=f"Key '{key}' {'exists' if exists else 'does not exist'}",
            data={"exists": exists}
        )
        
    except Exception as e:
        raise RuntimeError(f"Error checking existence of key '{key}': {str(e)}") from e


async def kv_keys(ctx: RunContext[Any], namespace: str, pattern: Optional[str]) -> StorageKvOutput:
    """
    List keys matching a pattern.
    
    Args:
        ctx: Runtime context
        namespace: The namespace to search
        pattern: Optional pattern with * wildcard support
        
    Returns:
        StorageKvOutput with the matching keys
    """
    try:
        # Clean up expired keys
        expired_count = _cleanup_expired_keys(namespace)
        
        # Get all keys in namespace
        if namespace not in _kv_storage:
            keys = []
        else:
            keys = list(_kv_storage[namespace].keys())
        
        # Apply pattern filtering if provided
        if pattern:
            import fnmatch
            keys = [k for k in keys if fnmatch.fnmatch(k, pattern)]
        
        # Sort keys for consistent output
        keys.sort()
        
        return StorageKvOutput(
            success=True,  # Always success even if no keys match
            operation="keys",
            key=None,
            namespace=namespace,
            message=f"Found {len(keys)} keys" + (f" matching pattern '{pattern}'" if pattern else ""),
            data={
                "keys": keys,
                "count": len(keys),
                "pattern": pattern,
                "expired_keys_cleaned": expired_count
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Error listing keys: {str(e)}") from e


async def kv_clear(ctx: RunContext[Any], namespace: str) -> StorageKvOutput:
    """
    Clear all keys in a namespace.
    
    Args:
        ctx: Runtime context
        namespace: The namespace to clear
        
    Returns:
        StorageKvOutput with the clear operation result
    """
    try:
        keys_count = 0
        
        if namespace in _kv_storage:
            keys_count = len(_kv_storage[namespace])
            del _kv_storage[namespace]
        
        if namespace in _kv_expiry:
            del _kv_expiry[namespace]
        
        return StorageKvOutput(
            success=True,
            operation="clear",
            key=None,
            namespace=namespace,
            message=f"Cleared {keys_count} keys from namespace '{namespace}'",
            data={
                "cleared_count": keys_count,
                "namespace_existed": keys_count > 0
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Error clearing namespace '{namespace}': {str(e)}") from e


async def kv_expire(ctx: RunContext[Any], key: str, namespace: str, ttl: int) -> StorageKvOutput:
    """
    Set TTL for an existing key.
    
    Args:
        ctx: Runtime context
        key: The key to set TTL for
        namespace: The namespace to use
        ttl: Time to live in seconds
        
    Returns:
        StorageKvOutput with the expire operation result
    """
    try:
        # Check if key exists
        if (namespace not in _kv_storage or 
            key not in _kv_storage[namespace] or 
            _is_key_expired(namespace, key)):
            
            raise KeyError(f"Key '{key}' does not exist")
        
        # Initialize expiry namespace if needed
        if namespace not in _kv_expiry:
            _kv_expiry[namespace] = {}
        
        # Set the expiry time
        _kv_expiry[namespace][key] = time.time() + ttl
        
        return StorageKvOutput(
            success=True,
            operation="expire",
            key=key,
            namespace=namespace,
            message=f"Set TTL of {ttl}s for key '{key}'",
            data={
                "ttl_set": ttl,
                "expires_at": _kv_expiry[namespace][key]
            }
        )
        
    except KeyError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error setting TTL for key '{key}': {str(e)}") from e


async def kv_ttl(ctx: RunContext[Any], key: str, namespace: str) -> StorageKvOutput:
    """
    Get the remaining TTL for a key.
    
    Args:
        ctx: Runtime context
        key: The key to check TTL for
        namespace: The namespace to use
        
    Returns:
        StorageKvOutput with the TTL information
    """
    try:
        # Check if key exists
        if (namespace not in _kv_storage or 
            key not in _kv_storage[namespace] or 
            _is_key_expired(namespace, key)):
            
            return StorageKvOutput(
                success=True,  # Always success - it's a query
                operation="ttl",
                key=key,
                namespace=namespace,
                message=f"Key '{key}' does not exist",
                data={
                    "ttl": -2,  # Redis convention: -2 means key doesn't exist
                    "exists": False
                }
            )
        
        # Check if key has TTL
        if namespace not in _kv_expiry or key not in _kv_expiry[namespace]:
            return StorageKvOutput(
                success=True,  # Always success - it's a query
                operation="ttl",
                key=key,
                namespace=namespace,
                message=f"Key '{key}' has no expiry",
                data={
                    "ttl": -1,  # Redis convention: -1 means no expiry
                    "exists": True,
                    "has_expiry": False
                }
            )
        
        # Calculate remaining TTL
        remaining_ttl = max(0, int(_kv_expiry[namespace][key] - time.time()))
        
        return StorageKvOutput(
            success=True,  # Always success - it's a query
            operation="ttl",
            key=key,
            namespace=namespace,
            message=f"Key '{key}' expires in {remaining_ttl}s",
            data={
                "ttl": remaining_ttl,
                "exists": True,
                "has_expiry": True,
                "expires_at": _kv_expiry[namespace][key]
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Error getting TTL for key '{key}': {str(e)}") from e


async def kv_get_metric(ctx: RunContext[Any], key: str, namespace: str) -> StorageKvOutput:
    """
    Get a metric value - special operation for metrics that returns None instead of raising KeyError.
    
    This is specifically designed for the metrics system to check if a metric exists
    without raising an exception, allowing the metrics system to create metrics on-demand.
    
    Args:
        ctx: Runtime context
        key: The metric key to retrieve
        namespace: The namespace to use (typically 'metrics')
        
    Returns:
        StorageKvOutput with the value or None if not found
    """
    try:
        # Clean up expired keys in this namespace
        expired_count = _cleanup_expired_keys(namespace)
        
        # Check if key exists and is not expired
        if (namespace in _kv_storage and 
            key in _kv_storage[namespace] and 
            not _is_key_expired(namespace, key)):
            
            value = _kv_storage[namespace][key]
            
            # Calculate remaining TTL if applicable
            ttl_remaining = None
            if namespace in _kv_expiry and key in _kv_expiry[namespace]:
                ttl_remaining = max(0, int(_kv_expiry[namespace][key] - time.time()))
            
            return StorageKvOutput(
                success=True,
                operation="get_metric",
                key=key,
                namespace=namespace,
                message=f"Successfully retrieved metric '{key}'",
                data={
                    "value": value,
                    "exists": True,
                    "ttl_remaining": ttl_remaining,
                    "expired_keys_cleaned": expired_count
                }
            )
        else:
            # Key not found - return success=False for metrics compatibility
            return StorageKvOutput(
                success=False,  # Special case for metrics system
                operation="get_metric",
                key=key,
                namespace=namespace,
                message=f"Metric '{key}' not found in namespace '{namespace}'",
                data=None  # This is the key difference - return None instead of raising
            )
            
    except Exception as e:
        raise RuntimeError(f"Error retrieving metric '{key}': {str(e)}") from e


def create_storage_kv_agent():
    """
    Create and return the key-value storage AgenTool.
    
    Returns:
        Agent configured for key-value storage operations
    """
    kv_routing = RoutingConfig(
        operation_field='operation',
        operation_map={
            'get': ('kv_get', lambda x: {'key': x.key, 'namespace': x.namespace}),
            'set': ('kv_set', lambda x: {
                'key': x.key, 'value': x.value, 'namespace': x.namespace, 'ttl': x.ttl
            }),
            'delete': ('kv_delete', lambda x: {'key': x.key, 'namespace': x.namespace}),
            'exists': ('kv_exists', lambda x: {'key': x.key, 'namespace': x.namespace}),
            'keys': ('kv_keys', lambda x: {'namespace': x.namespace, 'pattern': x.pattern}),
            'clear': ('kv_clear', lambda x: {'namespace': x.namespace}),
            'expire': ('kv_expire', lambda x: {
                'key': x.key, 'namespace': x.namespace, 'ttl': x.ttl
            }),
            'ttl': ('kv_ttl', lambda x: {'key': x.key, 'namespace': x.namespace}),
            'get_metric': ('kv_get_metric', lambda x: {'key': x.key, 'namespace': x.namespace}),
        }
    )
    
    return create_agentool(
        name='storage_kv',
        input_schema=StorageKvInput,
        routing_config=kv_routing,
        tools=[kv_get, kv_set, kv_delete, kv_exists, kv_keys, kv_clear, kv_expire, kv_ttl, kv_get_metric],
        output_type=StorageKvOutput,
        system_prompt="Handle key-value storage operations with TTL support efficiently.",
        description="Key-value storage with TTL support, namespaces, and Redis-compatible interface",
        version="1.0.0",
        tags=["storage", "key-value", "cache", "ttl", "redis"],
        examples=[
            {
                "description": "Set a value with TTL",
                "input": {
                    "operation": "set",
                    "key": "user:123",
                    "value": {"name": "Alice", "email": "alice@example.com"},
                    "ttl": 3600,
                    "namespace": "users"
                },
                "output": {
                    "operation": "set",
                    "key": "user:123",
                    "namespace": "users",
                    "message": "Successfully set key 'user:123' with TTL 3600s"
                }
            },
            {
                "description": "Get a value",
                "input": {"operation": "get", "key": "user:123", "namespace": "users"},
                "output": {
                    "operation": "get",
                    "key": "user:123", 
                    "namespace": "users",
                    "message": "Successfully retrieved value for key 'user:123'",
                    "data": {
                        "value": {"name": "Alice", "email": "alice@example.com"},
                        "exists": True,
                        "ttl_remaining": 3599
                    }
                }
            },
            {
                "description": "List keys with pattern",
                "input": {"operation": "keys", "pattern": "user:*", "namespace": "users"},
                "output": {
                    "operation": "keys",
                    "namespace": "users",
                    "message": "Found 1 keys matching pattern 'user:*'",
                    "data": {"keys": ["user:123"], "count": 1, "pattern": "user:*"}
                }
            }
        ]
    )


def clear_all_storage():
    """
    Clear all storage data across all namespaces.
    This is a convenience function for testing purposes.
    """
    _kv_storage.clear()
    _kv_expiry.clear()


# Create the agent instance when imported (auto-registers with the registry)
agent = create_storage_kv_agent()