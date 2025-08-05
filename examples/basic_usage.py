"""
Basic usage examples for AgenTools.

This module demonstrates how to use the AgenTool framework
with simple examples covering:

- Basic CRUD operations with the storage agent
- Error handling and validation
- JSON input/output patterns
- Working with structured data
- Conversation-style usage patterns

These examples show the fundamental patterns that can be applied
to any AgenTool implementation.

Run this example:
    cd /path/to/pydantic-ai
    python src/examples/basic_usage.py
"""

import asyncio
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from src.examples.agents.storage import create_storage_agent
from src.examples.tools.storage import _storage


async def basic_storage_example():
    """
    Demonstrate basic storage operations.
    
    This example shows:
    - Writing structured data (user objects)
    - Reading data by key
    - Listing all stored keys
    - Deleting data
    - Inspecting storage state
    """
    print("=== Basic Storage AgenTool Example ===\n")
    
    # Clear any existing data
    _storage.clear()
    
    # Create storage agent
    storage_agent = create_storage_agent()
    
    # Test write operation
    print("1. Writing data...")
    write_input = json.dumps({
        "operation": "write",
        "key": "user_123",
        "data": {"name": "John Doe", "age": 30}
    })
    result = await storage_agent.run(write_input)
    print(f"Input: {write_input}")
    print(f"Result: {result.output}")
    print(f"Storage: {_storage}\n")
    
    # Test read operation
    print("2. Reading data...")
    read_input = json.dumps({
        "operation": "read",
        "key": "user_123"
    })
    result = await storage_agent.run(read_input)
    print(f"Input: {read_input}")
    print(f"Result: {result.output}\n")
    
    # Test list operation
    print("3. Listing keys...")
    list_input = json.dumps({
        "operation": "list"
    })
    result = await storage_agent.run(list_input)
    print(f"Input: {list_input}")
    print(f"Result: {result.output}\n")
    
    # Test another write
    print("4. Writing more data...")
    write_input2 = json.dumps({
        "operation": "write",
        "key": "config",
        "data": {"theme": "dark", "language": "en"}
    })
    result = await storage_agent.run(write_input2)
    print(f"Input: {write_input2}")
    print(f"Result: {result.output}")
    print(f"Storage: {_storage}\n")
    
    # Test list again
    print("5. Listing all keys...")
    result = await storage_agent.run(list_input)
    print(f"Input: {list_input}")
    print(f"Result: {result.output}\n")
    
    # Test delete operation
    print("6. Deleting data...")
    delete_input = json.dumps({
        "operation": "delete",
        "key": "config"
    })
    result = await storage_agent.run(delete_input)
    print(f"Input: {delete_input}")
    print(f"Result: {result.output}")
    print(f"Storage: {_storage}\n")


async def error_handling_example():
    """
    Demonstrate error handling in AgenTools.
    
    This example shows how AgenTools handle:
    - Invalid JSON input
    - Missing required fields
    - Unknown operations
    - Validation errors
    
    Note: AgenTools return error messages as strings rather than
    raising exceptions to maintain conversation flow.
    """
    print("=== Error Handling Examples ===\n")
    
    # Create storage agent
    storage_agent = create_storage_agent()
    
    # Test invalid JSON
    print("1. Invalid JSON...")
    invalid_json = '{"operation": "read", "key":}'  # Missing value
    result = await storage_agent.run(invalid_json)
    print(f"Input: {invalid_json}")
    print(f"Result: {result.output}")
    
    # Test missing required field
    print("\n2. Missing required field...")
    invalid_input = json.dumps({
        "operation": "read"
        # Missing 'key' field
    })
    result = await storage_agent.run(invalid_input)
    print(f"Input: {invalid_input}")
    print(f"Result: {result.output}")
    
    # Test unknown operation
    print("\n3. Unknown operation...")
    unknown_op = json.dumps({
        "operation": "backup",
        "key": "test"
    })
    result = await storage_agent.run(unknown_op)
    print(f"Input: {unknown_op}")
    print(f"Result: {result.output}")


async def conversation_example():
    """
    Demonstrate conversation-style usage of AgenTools.
    
    This example shows how AgenTools can be used in a conversational
    context, similar to how they might be integrated with an LLM agent.
    """
    print("=== Conversation-Style Usage ===\n")
    
    # Clear storage for fresh start
    _storage.clear()
    
    # Create storage agent
    storage_agent = create_storage_agent()
    
    # Simulate a conversation flow
    conversations = [
        {
            "description": "User wants to store their preferences",
            "input": {
                "operation": "write",
                "key": "user_prefs",
                "data": {
                    "theme": "dark",
                    "notifications": True,
                    "language": "en"
                }
            }
        },
        {
            "description": "User wants to check what preferences are stored",
            "input": {
                "operation": "read",
                "key": "user_prefs"
            }
        },
        {
            "description": "User wants to store session data",
            "input": {
                "operation": "write",
                "key": "session_123",
                "data": {
                    "user_id": "user_456",
                    "started_at": "2025-01-01T10:00:00",
                    "last_active": "2025-01-01T10:15:00"
                }
            }
        },
        {
            "description": "User wants to see all stored items",
            "input": {
                "operation": "list"
            }
        },
        {
            "description": "User wants to clean up old session",
            "input": {
                "operation": "delete",
                "key": "session_123"
            }
        }
    ]
    
    for i, conv in enumerate(conversations, 1):
        print(f"{i}. {conv['description']}")
        input_json = json.dumps(conv['input'], indent=2)
        print(f"Input:\n{input_json}")
        
        result = await storage_agent.run(json.dumps(conv['input']))
        print(f"Output: {result.output}\n")
    
    print(f"Final storage state: {_storage}\n")


async def main():
    """Run all examples."""
    await basic_storage_example()
    await error_handling_example()
    await conversation_example()
    
    print("✅ All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())