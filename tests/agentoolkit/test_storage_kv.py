"""
Tests for storage_kv toolkit.

This module tests all functionality of the key-value storage toolkit
including CRUD operations, TTL support, namespacing, pattern matching, and expiration.
"""

import json
import asyncio
import time

from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry


class TestStorageKv:
    """Test suite for storage_kv toolkit."""
    
    def setup_method(self):
        """Clear registry and injector before each test."""
        AgenToolRegistry.clear()
        get_injector().clear()
        
        # Import and create the agent, clear global storage
        from agentoolkit.storage.kv import create_storage_kv_agent, _kv_storage, _kv_expiry
        
        # Clear global storage
        _kv_storage.clear()
        _kv_expiry.clear()
        
        agent = create_storage_kv_agent()
    
    def test_kv_set_and_get(self):
        """Test basic set and get operations."""
        
        async def run_test():
            injector = get_injector()
            key = "test_key"
            value = {"name": "Alice", "age": 30}
            namespace = "test"
            
            # Test set
            set_result = await injector.run('storage_kv', {
                "operation": "set",
                "key": key,
                "value": value,
                "namespace": namespace
            })
            
            # storage_kv returns typed StorageKvOutput
            assert set_result.success is True
            assert set_result.operation == "set"
            assert set_result.key == key
            assert set_result.namespace == namespace
            assert set_result.data["stored"] is True
            
            # Test get
            get_result = await injector.run('storage_kv', {
                "operation": "get",
                "key": key,
                "namespace": namespace
            })
            
            # storage_kv returns typed StorageKvOutput
            assert get_result.success is True
            assert get_result.operation == "get"
            assert get_result.data["exists"] is True
            assert get_result.data["value"] == value
        
        asyncio.run(run_test())
    
    def test_kv_ttl_functionality(self):
        """Test TTL (Time To Live) functionality."""
        
        async def run_test():
            injector = get_injector()
            key = "ttl_test"
            value = "expires soon"
            namespace = "test"
            ttl = 1  # 1 second
            
            # Set with TTL
            set_result = await injector.run('storage_kv', {
                "operation": "set",
                "key": key,
                "value": value,
                "namespace": namespace,
                "ttl": ttl
            })
            
            # storage_kv returns typed StorageKvOutput
            assert set_result.success is True
            assert set_result.data["ttl"] == ttl
            
            # Get immediately (should exist)
            get_result = await injector.run('storage_kv', {
                "operation": "get",
                "key": key,
                "namespace": namespace
            })
            
            # storage_kv returns typed StorageKvOutput
            assert get_result.success is True
            assert get_result.data["exists"] is True
            assert get_result.data["ttl_remaining"] is not None
            assert get_result.data["ttl_remaining"] <= ttl
            
            # Wait for expiration
            await asyncio.sleep(1.1)
            
            # Get after expiration (should return success=False)
            get_expired_result = await injector.run('storage_kv', {
                "operation": "get",
                "key": key,
                "namespace": namespace
            })
            # storage_kv returns typed StorageKvOutput with success=False
            assert get_expired_result.success is False
            assert key in get_expired_result.message
            assert "expired" in get_expired_result.message
        
        asyncio.run(run_test())
    
    def test_kv_delete(self):
        """Test delete operation."""
        
        async def run_test():
            injector = get_injector()
            key = "delete_test"
            value = "to be deleted"
            namespace = "test"
            
            # Set a value
            await injector.run('storage_kv', {
                "operation": "set",
                "key": key,
                "value": value,
                "namespace": namespace
            })
            
            # Verify it exists
            exists_result = await injector.run('storage_kv', {
                "operation": "exists",
                "key": key,
                "namespace": namespace
            })
            
            # storage_kv returns typed StorageKvOutput
            assert exists_result.success is True
            assert exists_result.data["exists"] is True
            
            # Delete the key
            delete_result = await injector.run('storage_kv', {
                "operation": "delete",
                "key": key,
                "namespace": namespace
            })
            
            # storage_kv returns typed StorageKvOutput
            assert delete_result.success is True
            assert delete_result.operation == "delete"
            assert delete_result.data["deleted"] is True
            assert delete_result.data["existed"] is True
            
            # Verify it no longer exists
            exists_result = await injector.run('storage_kv', {
                "operation": "exists",
                "key": key,
                "namespace": namespace
            })
            
            # storage_kv returns typed StorageKvOutput
            assert exists_result.success is True
            assert exists_result.data["exists"] is False
        
        asyncio.run(run_test())
    
    def test_kv_keys_operation(self):
        """Test keys listing and pattern matching."""
        
        async def run_test():
            injector = get_injector()
            namespace = "test"
            
            # Set multiple keys
            test_keys = {
                "user:1": {"name": "Alice"},
                "user:2": {"name": "Bob"},
                "config:timeout": 30,
                "config:retries": 3,
                "session:abc123": {"user_id": "1"}
            }
            
            for key, value in test_keys.items():
                await injector.run('storage_kv', {
                    "operation": "set",
                    "key": key,
                    "value": value,
                    "namespace": namespace
                })
            
            # List all keys
            keys_result = await injector.run('storage_kv', {
                "operation": "keys",
                "namespace": namespace
            })
            
            # storage_kv returns typed StorageKvOutput
            assert keys_result.success is True
            assert keys_result.operation == "keys"
            assert keys_result.data["count"] == len(test_keys)
            assert set(keys_result.data["keys"]) == set(test_keys.keys())
            
            # Test pattern matching for user keys
            user_keys_result = await injector.run('storage_kv', {
                "operation": "keys",
                "namespace": namespace,
                "pattern": "user:*"
            })
            
            # storage_kv returns typed StorageKvOutput
            assert user_keys_result.success is True
            assert user_keys_result.data["count"] == 2
            assert "user:1" in user_keys_result.data["keys"]
            assert "user:2" in user_keys_result.data["keys"]
            assert "config:timeout" not in user_keys_result.data["keys"]
        
        asyncio.run(run_test())
    
    def test_kv_clear_operation(self):
        """Test clearing all keys in a namespace."""
        
        async def run_test():
            injector = get_injector()
            namespace = "test"
            
            # Set some keys
            test_data = {
                "key1": "value1",
                "key2": "value2",
                "key3": "value3"
            }
            
            for key, value in test_data.items():
                await injector.run('storage_kv', {
                    "operation": "set",
                    "key": key,
                    "value": value,
                    "namespace": namespace
                })
            
            # Verify keys exist
            keys_result = await injector.run('storage_kv', {
                "operation": "keys",
                "namespace": namespace
            })
            
            # storage_kv returns typed StorageKvOutput
            assert keys_result.success is True
            assert keys_result.data["count"] == len(test_data)
            
            # Clear the namespace
            clear_result = await injector.run('storage_kv', {
                "operation": "clear",
                "namespace": namespace
            })
            
            # storage_kv returns typed StorageKvOutput
            assert clear_result.success is True
            assert clear_result.operation == "clear"
            assert clear_result.data["cleared_count"] == len(test_data)
            
            # Verify namespace is empty
            keys_result = await injector.run('storage_kv', {
                "operation": "keys",
                "namespace": namespace
            })
            
            # storage_kv returns typed StorageKvOutput
            assert keys_result.success is True
            assert keys_result.data["count"] == 0
        
        asyncio.run(run_test())
    
    def test_kv_expire_operation(self):
        """Test setting TTL for existing keys."""
        
        async def run_test():
            injector = get_injector()
            key = "expire_test"
            value = "test value"
            namespace = "test"
            
            # Set key without TTL
            await injector.run('storage_kv', {
                "operation": "set",
                "key": key,
                "value": value,
                "namespace": namespace
            })
            
            # Set TTL using expire operation
            expire_result = await injector.run('storage_kv', {
                "operation": "expire",
                "key": key,
                "namespace": namespace,
                "ttl": 1
            })
            
            # storage_kv returns typed StorageKvOutput
            assert expire_result.success is True
            assert expire_result.operation == "expire"
            assert expire_result.data["ttl_set"] == 1
            
            # Check TTL
            ttl_result = await injector.run('storage_kv', {
                "operation": "ttl",
                "key": key,
                "namespace": namespace
            })
            
            # storage_kv returns typed StorageKvOutput
            assert ttl_result.success is True
            assert ttl_result.operation == "ttl"
            assert ttl_result.data["has_expiry"] is True
            assert ttl_result.data["ttl"] <= 1
            
            # Wait for expiration
            await asyncio.sleep(1.1)
            
            # Key should be expired (should return success=False)
            get_result = await injector.run('storage_kv', {
                "operation": "get",
                "key": key,
                "namespace": namespace
            })
            # storage_kv returns typed StorageKvOutput with success=False
            assert get_result.success is False
            assert key in get_result.message
            assert "expired" in get_result.message
        
        asyncio.run(run_test())
    
    def test_kv_ttl_operation(self):
        """Test TTL query operation."""
        
        async def run_test():
            injector = get_injector()
            key_with_ttl = "ttl_key"
            key_without_ttl = "no_ttl_key"
            nonexistent_key = "missing_key"
            namespace = "test"
            
            # Set key with TTL
            await injector.run('storage_kv', {
                "operation": "set",
                "key": key_with_ttl,
                "value": "value",
                "namespace": namespace,
                "ttl": 60
            })
            
            # Set key without TTL
            await injector.run('storage_kv', {
                "operation": "set",
                "key": key_without_ttl,
                "value": "value",
                "namespace": namespace
            })
            
            # Check TTL for key with TTL
            ttl_result = await injector.run('storage_kv', {
                "operation": "ttl",
                "key": key_with_ttl,
                "namespace": namespace
            })
            
            # storage_kv returns typed StorageKvOutput
            assert ttl_result.success is True
            assert ttl_result.operation == "ttl"
            assert ttl_result.data["ttl"] > 0
            assert ttl_result.data["has_expiry"] is True
            
            # Check TTL for key without TTL
            no_ttl_result = await injector.run('storage_kv', {
                "operation": "ttl",
                "key": key_without_ttl,
                "namespace": namespace
            })
            
            # storage_kv returns typed StorageKvOutput
            assert no_ttl_result.success is True
            assert no_ttl_result.operation == "ttl"
            assert no_ttl_result.data["ttl"] == -1  # No expiry
            assert no_ttl_result.data["has_expiry"] is False
            
            # Check TTL for nonexistent key
            missing_ttl_result = await injector.run('storage_kv', {
                "operation": "ttl",
                "key": nonexistent_key,
                "namespace": namespace
            })
            
            # storage_kv returns typed StorageKvOutput
            assert missing_ttl_result.success is True  # TTL query always succeeds
            assert missing_ttl_result.operation == "ttl"
            assert missing_ttl_result.data["ttl"] == -2  # Key doesn't exist
            assert missing_ttl_result.data["exists"] is False
        
        asyncio.run(run_test())
    
    def test_kv_namespace_isolation(self):
        """Test that namespaces properly isolate data."""
        
        async def run_test():
            injector = get_injector()
            key = "shared_key"
            namespace1 = "namespace1"
            namespace2 = "namespace2"
            value1 = "value from namespace1"
            value2 = "value from namespace2"
            
            # Set same key in different namespaces
            await injector.run('storage_kv', {
                "operation": "set",
                "key": key,
                "value": value1,
                "namespace": namespace1
            })
            
            await injector.run('storage_kv', {
                "operation": "set",
                "key": key,
                "value": value2,
                "namespace": namespace2
            })
            
            # Get from namespace1
            get1_result = await injector.run('storage_kv', {
                "operation": "get",
                "key": key,
                "namespace": namespace1
            })
            
            # storage_kv returns typed StorageKvOutput
            assert get1_result.success is True
            assert get1_result.data["value"] == value1
            
            # Get from namespace2
            get2_result = await injector.run('storage_kv', {
                "operation": "get",
                "key": key,
                "namespace": namespace2
            })
            
            # storage_kv returns typed StorageKvOutput
            assert get2_result.success is True
            assert get2_result.data["value"] == value2
            
            # Delete from namespace1
            await injector.run('storage_kv', {
                "operation": "delete",
                "key": key,
                "namespace": namespace1
            })
            
            # Verify still exists in namespace2
            get2_after_delete = await injector.run('storage_kv', {
                "operation": "get",
                "key": key,
                "namespace": namespace2
            })
            
            # storage_kv returns typed StorageKvOutput
            assert get2_after_delete.success is True
            assert get2_after_delete.data["exists"] is True
            assert get2_after_delete.data["value"] == value2
        
        asyncio.run(run_test())
    
    def test_kv_data_types(self):
        """Test storage of various data types."""
        
        async def run_test():
            injector = get_injector()
            namespace = "test"
            
            test_data = {
                "string": "hello world",
                "integer": 42,
                "float": 3.14159,
                "boolean": True,
                "null": None,
                "list": [1, 2, 3, "four"],
                "dict": {"nested": {"deep": "value"}},
                "complex": {
                    "users": [
                        {"id": 1, "name": "Alice", "active": True},
                        {"id": 2, "name": "Bob", "active": False}
                    ],
                    "config": {
                        "timeout": 30,
                        "retries": 3,
                        "endpoints": ["api1.com", "api2.com"]
                    }
                }
            }
            
            # Store all data types
            for key, value in test_data.items():
                set_result = await injector.run('storage_kv', {
                    "operation": "set",
                    "key": key,
                    "value": value,
                    "namespace": namespace
                })
                
                # storage_kv returns typed StorageKvOutput
                assert set_result.success is True
                assert set_result.operation == "set"
            
            # Retrieve and verify all data types
            for key, expected_value in test_data.items():
                get_result = await injector.run('storage_kv', {
                    "operation": "get",
                    "key": key,
                    "namespace": namespace
                })
                
                # storage_kv returns typed StorageKvOutput
                assert get_result.success is True
                assert get_result.operation == "get"
                assert get_result.data["exists"] is True
                assert get_result.data["value"] == expected_value
        
        asyncio.run(run_test())
    
    def test_kv_error_handling(self):
        """Test error handling for edge cases."""
        
        async def run_test():
            injector = get_injector()
            namespace = "test"
            
            # Test getting non-existent key (should return success=False)
            get_result = await injector.run('storage_kv', {
                "operation": "get",
                "key": "nonexistent",
                "namespace": namespace
            })
            # storage_kv returns typed StorageKvOutput with success=False
            assert get_result.success is False
            assert "nonexistent" in get_result.message
            assert "not found" in get_result.message
            
            # Test deleting non-existent key (should be idempotent)
            delete_result = await injector.run('storage_kv', {
                "operation": "delete",
                "key": "nonexistent",
                "namespace": namespace
            })
            
            # storage_kv returns typed StorageKvOutput
            assert delete_result.success is True  # Idempotent
            assert delete_result.operation == "delete"
            assert delete_result.data["existed"] is False
            
            # Test expire on non-existent key (should still raise KeyError - not a discovery operation)
            try:
                expire_result = await injector.run('storage_kv', {
                    "operation": "expire",
                    "key": "nonexistent",
                    "namespace": namespace,
                    "ttl": 60
                })
                # Should not reach here
                assert False, "Expected KeyError for non-existent key"
            except KeyError as e:
                assert "does not exist" in str(e)
                print(f"\n   Expected exception caught: {e}")
        
        asyncio.run(run_test())
    
    def test_kv_expiry_cleanup(self):
        """Test that expired keys are properly cleaned up."""
        
        async def run_test():
            injector = get_injector()
            namespace = "test"
            
            # Set multiple keys with short TTL
            keys_with_ttl = ["expire1", "expire2", "expire3"]
            persistent_key = "persistent"
            
            for key in keys_with_ttl:
                await injector.run('storage_kv', {
                    "operation": "set",
                    "key": key,
                    "value": f"value_{key}",
                    "namespace": namespace,
                    "ttl": 1
                })
            
            # Set one persistent key
            await injector.run('storage_kv', {
                "operation": "set",
                "key": persistent_key,
                "value": "persistent_value",
                "namespace": namespace
            })
            
            # Verify all keys exist
            keys_result = await injector.run('storage_kv', {
                "operation": "keys",
                "namespace": namespace
            })
            
            # storage_kv returns typed StorageKvOutput
            assert keys_result.success is True
            assert keys_result.data["count"] == 4
            
            # Wait for expiration
            await asyncio.sleep(1.1)
            
            # Trigger cleanup by doing a get operation
            get_result = await injector.run('storage_kv', {
                "operation": "get",
                "key": persistent_key,
                "namespace": namespace
            })
            
            # storage_kv returns typed StorageKvOutput
            assert get_result.success is True
            # Should report expired keys were cleaned
            assert get_result.data["expired_keys_cleaned"] == 3
            
            # Verify only persistent key remains
            keys_result = await injector.run('storage_kv', {
                "operation": "keys",
                "namespace": namespace
            })
            
            # storage_kv returns typed StorageKvOutput
            assert keys_result.success is True
            assert keys_result.data["count"] == 1
            assert persistent_key in keys_result.data["keys"]
        
        asyncio.run(run_test())