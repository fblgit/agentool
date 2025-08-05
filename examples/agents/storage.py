"""
Storage AgenTool implementation.

This module demonstrates how to create a complete AgenTool for storage
operations with proper JSON I/O and structured routing.
"""

from pydantic_ai import Agent

from src.agentool.factory import create_agentool
from src.agentool.core.registry import RoutingConfig
from src.examples.schemas.storage import StorageOperationInput
from src.examples.tools.storage import storage_read, storage_write, storage_list, storage_delete


def create_storage_agent() -> Agent:
    """
    Create a storage AgenTool.
    
    This agent provides storage operations with JSON-based input/output:
    - read: Retrieve data by key
    - write: Store data with a key
    - list: List all storage keys
    - delete: Remove data by key
    
    Returns:
        A configured storage Agent
        
    Example:
        >>> agent = create_storage_agent()
        >>> result = await agent.run('{"operation": "write", "key": "user:123", "data": {"name": "Alice"}}')
        >>> print(result.output)
        {"success": true, "operation": "write", "message": "Successfully stored data at key 'user:123'", "key": "user:123", "overwritten": false}
    """
    # Define routing configuration
    routing_config = RoutingConfig(
        operation_map={
            'read': ('storage_read', lambda inp: {'key': inp.key}),
            'write': ('storage_write', lambda inp: {'key': inp.key, 'data': inp.data}),
            'list': ('storage_list', lambda inp: {}),
            'delete': ('storage_delete', lambda inp: {'key': inp.key})
        }
    )
    
    # Create the agent using the factory with enhanced metadata
    agent = create_agentool(
        name='storage',
        input_schema=StorageOperationInput,
        routing_config=routing_config,
        tools=[storage_read, storage_write, storage_list, storage_delete],
        system_prompt="Process storage operations with JSON input/output.",
        description="A storage system supporting read, write, list, and delete operations",
        version="1.2.0",
        tags=["storage", "database", "async", "json", "crud"],
        dependencies=["pydantic", "pydantic-ai"],
        examples=[
            {
                "description": "Write data to storage",
                "input": {"operation": "write", "key": "user:123", "data": {"name": "Alice", "age": 30}},
                "output": {"success": True, "operation": "write", "message": "Successfully stored data at key 'user:123'", "key": "user:123", "overwritten": False}
            },
            {
                "description": "Read data from storage",
                "input": {"operation": "read", "key": "user:123"},
                "output": {"success": True, "operation": "read", "key": "user:123", "data": {"name": "Alice", "age": 30}, "exists": True}
            },
            {
                "description": "List all storage keys",
                "input": {"operation": "list"},
                "output": {"success": True, "operation": "list", "keys": ["user:123", "user:456"], "count": 2}
            },
            {
                "description": "Delete a key",
                "input": {"operation": "delete", "key": "user:456"},
                "output": {"success": True, "operation": "delete", "key": "user:456", "existed": True}
            }
        ]
    )
    
    return agent


# Export the factory function
__all__ = ['create_storage_agent']