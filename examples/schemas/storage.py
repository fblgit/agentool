"""
Storage operation schemas for AgenTools.

This module provides input and output schemas for storage operations,
demonstrating best practices for structured JSON I/O.
"""

from typing import Literal, Optional, Any, List, Dict
from pydantic import BaseModel, Field, field_validator

from src.agentool.base import BaseOperationInput


# Input Schemas
# -------------

class StorageOperationInput(BaseOperationInput):
    """
    Schema for storage-related operations.
    
    This schema validates input for storage operations including
    read, write, list, and delete operations.
    
    Attributes:
        operation: The storage operation to perform
        key: The storage key (required for read, write, delete)
        data: The data to store (required for write)
    
    Example:
        >>> input_data = StorageOperationInput(
        ...     operation='write',
        ...     key='user:123',
        ...     data={'name': 'Alice', 'age': 30}
        ... )
    """
    operation: Literal['read', 'write', 'list', 'delete'] = Field(
        description="The storage operation to perform"
    )
    key: Optional[str] = Field(None, description="The storage key")
    data: Optional[Any] = Field(None, description="The data to store")
    
    @field_validator('key')
    def validate_key_for_operation(cls, v, info):
        """Validate that key is provided for operations that need it."""
        operation = info.data.get('operation')
        if operation in ('read', 'write', 'delete') and not v:
            raise ValueError(f"'key' is required for {operation} operation")
        return v
    
    @field_validator('data')
    def validate_data_for_write(cls, v, info):
        """Validate that data is provided for write operations."""
        if info.data.get('operation') == 'write' and v is None:
            raise ValueError("'data' is required for write operation")
        return v


# Output Schemas
# --------------

class StorageResponse(BaseModel):
    """
    Base response schema for storage operations.
    
    All storage operations return this structured response format
    to ensure consistent JSON output.
    
    Attributes:
        success: Whether the operation succeeded
        operation: The operation that was performed
        message: Human-readable message about the result
    """
    success: bool = Field(description="Whether the operation succeeded")
    operation: str = Field(description="The operation that was performed")
    message: str = Field(description="Human-readable result message")


class StorageReadResponse(StorageResponse):
    """
    Response schema for read operations.
    
    Attributes:
        key: The key that was read
        data: The data retrieved (None if not found)
        found: Whether the key was found
    """
    key: str = Field(description="The key that was read")
    data: Optional[Any] = Field(None, description="The retrieved data")
    found: bool = Field(description="Whether the key was found")


class StorageWriteResponse(StorageResponse):
    """
    Response schema for write operations.
    
    Attributes:
        key: The key that was written
        overwritten: Whether an existing value was overwritten
    """
    key: str = Field(description="The key that was written")
    overwritten: bool = Field(description="Whether an existing value was overwritten")


class StorageListResponse(StorageResponse):
    """
    Response schema for list operations.
    
    Attributes:
        keys: List of all storage keys
        count: Number of keys in storage
    """
    keys: List[str] = Field(description="All keys in storage")
    count: int = Field(description="Number of keys")


class StorageDeleteResponse(StorageResponse):
    """
    Response schema for delete operations.
    
    Attributes:
        key: The key that was deleted
        found: Whether the key existed before deletion
    """
    key: str = Field(description="The key that was deleted")
    found: bool = Field(description="Whether the key existed")


# Type alias for any storage response
StorageResponseType = StorageReadResponse | StorageWriteResponse | StorageListResponse | StorageDeleteResponse