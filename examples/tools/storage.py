"""
Storage tools implementation for AgenTools.

This module provides storage operation tools that demonstrate
best practices for JSON-based I/O with structured responses.
"""

from typing import Dict, Any, Optional
from pydantic_ai import RunContext

from src.examples.schemas.storage import (
    StorageReadResponse,
    StorageWriteResponse,
    StorageListResponse,
    StorageDeleteResponse
)


# In-memory storage for demonstration
_storage: Dict[str, Any] = {}


async def storage_read(ctx: RunContext[Any], key: str) -> dict:
    """
    Read data from storage.
    
    Args:
        ctx: The pydantic-ai run context
        key: The storage key to read
        
    Returns:
        Dict with the data if found
    """
    data = _storage.get(key)
    found = data is not None
    
    return StorageReadResponse(
        success=True,
        operation='read',
        message=f"Read operation completed for key '{key}'",
        key=key,
        data=data,
        found=found
    ).model_dump()


async def storage_write(ctx: RunContext[Any], key: str, data: Any) -> dict:
    """
    Write data to storage.
    
    Args:
        ctx: The pydantic-ai run context
        key: The storage key to write
        data: The data to store
        
    Returns:
        Dict indicating success and whether data was overwritten
    """
    overwritten = key in _storage
    _storage[key] = data
    
    return StorageWriteResponse(
        success=True,
        operation='write',
        message=f"Successfully stored data at key '{key}'",
        key=key,
        overwritten=overwritten
    ).model_dump()


async def storage_list(ctx: RunContext[Any]) -> dict:
    """
    List all keys in storage.
    
    Args:
        ctx: The pydantic-ai run context
        
    Returns:
        Dict with all storage keys
    """
    keys = list(_storage.keys())
    
    return StorageListResponse(
        success=True,
        operation='list',
        message=f"Found {len(keys)} keys in storage",
        keys=keys,
        count=len(keys)
    ).model_dump()


async def storage_delete(ctx: RunContext[Any], key: str) -> dict:
    """
    Delete a key from storage.
    
    Args:
        ctx: The pydantic-ai run context
        key: The storage key to delete
        
    Returns:
        Dict indicating whether the key existed
    """
    found = key in _storage
    if found:
        del _storage[key]
    
    return StorageDeleteResponse(
        success=True,
        operation='delete',
        message=f"Delete operation completed for key '{key}'",
        key=key,
        found=found
    ).model_dump()


# Export all tools
__all__ = ['storage_read', 'storage_write', 'storage_list', 'storage_delete']