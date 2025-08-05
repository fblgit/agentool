"""
AgenTool example schemas.

This module contains example input/output schemas for various
AgenTool implementations, demonstrating best practices for
structured JSON I/O.
"""

from .storage import (
    StorageOperationInput,
    StorageResponse,
    StorageReadResponse,
    StorageWriteResponse,
    StorageListResponse,
    StorageDeleteResponse,
    StorageResponseType
)

__all__ = [
    'StorageOperationInput',
    'StorageResponse', 
    'StorageReadResponse',
    'StorageWriteResponse',
    'StorageListResponse',
    'StorageDeleteResponse',
    'StorageResponseType'
]