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
            
            if hasattr(set_result, 'output'):
                set_data = json.loads(set_result.output)
            else:
                set_data = set_result
            
            assert "operation" in set_data
            assert set_data["operation"] == "set"
            assert set_data["key"] == key
            assert set_data["namespace"] == namespace
            assert set_data["data"]["stored"] is True
            
            # Test get
            get_result = await injector.run('storage_kv', {
                "operation": "get",
                "key": key,
                "namespace": namespace
            })
            
            if hasattr(get_result, 'output'):
                get_data = json.loads(get_result.output)
            else:
                get_data = get_result
            
            assert "operation" in get_data
            assert get_data["operation"] == "get"
            assert get_data["data"]["exists"] is True
            assert get_data["data"]["value"] == value
        
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
            
            if hasattr(set_result, 'output'):
                set_data = json.loads(set_result.output)
            else:
                set_data = set_result
            
            assert "operation" in set_data
            assert set_data["data"]["ttl"] == ttl
            
            # Get immediately (should exist)
            get_result = await injector.run('storage_kv', {
                "operation": "get",
                "key": key,
                "namespace": namespace
            })
            
            if hasattr(get_result, 'output'):
                get_data = json.loads(get_result.output)
            else:
                get_data = get_result
            
            assert "operation" in get_data
            assert get_data["data"]["exists"] is True
            assert get_data["data"]["ttl_remaining"] is not None
            assert get_data["data"]["ttl_remaining"] <= ttl
            
            # Wait for expiration
            await asyncio.sleep(1.1)
            
            # Get after expiration (should raise KeyError)
            try:
                get_expired_result = await injector.run('storage_kv', {
                    "operation": "get",
                    "key": key,
                    "namespace": namespace
                })
                assert False, "Expected KeyError for expired key"
            except KeyError as e:
                assert key in str(e)
                assert "expired" in str(e)
        
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
            
            if hasattr(exists_result, 'output'):
                exists_data = json.loads(exists_result.output)
            else:
                exists_data = exists_result
            
            assert exists_data["data"]["exists"] is True
            
            # Delete the key
            delete_result = await injector.run('storage_kv', {
                "operation": "delete",
                "key": key,
                "namespace": namespace
            })
            
            if hasattr(delete_result, 'output'):
                delete_data = json.loads(delete_result.output)
            else:
                delete_data = delete_result
            
            assert "operation" in delete_data
            assert delete_data["data"]["deleted"] is True
            assert delete_data["data"]["existed"] is True
            
            # Verify it no longer exists
            exists_result = await injector.run('storage_kv', {
                "operation": "exists",
                "key": key,
                "namespace": namespace
            })
            
            if hasattr(exists_result, 'output'):
                exists_data = json.loads(exists_result.output)
            else:
                exists_data = exists_result
            
            assert exists_data["data"]["exists"] is False
        
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
            
            if hasattr(keys_result, 'output'):
                keys_data = json.loads(keys_result.output)
            else:
                keys_data = keys_result
            
            assert "operation" in keys_data
            assert keys_data["data"]["count"] == len(test_keys)
            assert set(keys_data["data"]["keys"]) == set(test_keys.keys())
            
            # Test pattern matching for user keys
            user_keys_result = await injector.run('storage_kv', {
                "operation": "keys",
                "namespace": namespace,
                "pattern": "user:*"
            })
            
            if hasattr(user_keys_result, 'output'):
                user_keys_data = json.loads(user_keys_result.output)
            else:
                user_keys_data = user_keys_result
            
            assert "operation" in user_keys_data
            assert user_keys_data["data"]["count"] == 2
            assert "user:1" in user_keys_data["data"]["keys"]
            assert "user:2" in user_keys_data["data"]["keys"]
            assert "config:timeout" not in user_keys_data["data"]["keys"]
        
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
            
            if hasattr(keys_result, 'output'):
                keys_data = json.loads(keys_result.output)
            else:
                keys_data = keys_result
            
            assert keys_data["data"]["count"] == len(test_data)
            
            # Clear the namespace
            clear_result = await injector.run('storage_kv', {
                "operation": "clear",
                "namespace": namespace
            })
            
            if hasattr(clear_result, 'output'):
                clear_data = json.loads(clear_result.output)
            else:
                clear_data = clear_result
            
            assert "operation" in clear_data
            assert clear_data["data"]["cleared_count"] == len(test_data)
            
            # Verify namespace is empty
            keys_result = await injector.run('storage_kv', {
                "operation": "keys",
                "namespace": namespace
            })
            
            if hasattr(keys_result, 'output'):
                keys_data = json.loads(keys_result.output)
            else:
                keys_data = keys_result
            
            assert keys_data["data"]["count"] == 0
        
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
            
            if hasattr(expire_result, 'output'):
                expire_data = json.loads(expire_result.output)
            else:
                expire_data = expire_result
            
            assert "operation" in expire_data
            assert expire_data["data"]["ttl_set"] == 1
            
            # Check TTL
            ttl_result = await injector.run('storage_kv', {
                "operation": "ttl",
                "key": key,
                "namespace": namespace
            })
            
            if hasattr(ttl_result, 'output'):
                ttl_data = json.loads(ttl_result.output)
            else:
                ttl_data = ttl_result
            
            assert "operation" in ttl_data
            assert ttl_data["data"]["has_expiry"] is True
            assert ttl_data["data"]["ttl"] <= 1
            
            # Wait for expiration
            await asyncio.sleep(1.1)
            
            # Key should be expired (should raise KeyError)
            try:
                get_result = await injector.run('storage_kv', {
                    "operation": "get",
                    "key": key,
                    "namespace": namespace
                })
                assert False, "Expected KeyError for expired key"
            except KeyError as e:
                assert key in str(e)
                assert "expired" in str(e)
        
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
            
            if hasattr(ttl_result, 'output'):
                ttl_data = json.loads(ttl_result.output)
            else:
                ttl_data = ttl_result
            
            assert "operation" in ttl_data
            assert ttl_data["data"]["ttl"] > 0
            assert ttl_data["data"]["has_expiry"] is True
            
            # Check TTL for key without TTL
            no_ttl_result = await injector.run('storage_kv', {
                "operation": "ttl",
                "key": key_without_ttl,
                "namespace": namespace
            })
            
            if hasattr(no_ttl_result, 'output'):
                no_ttl_data = json.loads(no_ttl_result.output)
            else:
                no_ttl_data = no_ttl_result
            
            assert "operation" in no_ttl_data
            assert no_ttl_data["data"]["ttl"] == -1  # No expiry
            assert no_ttl_data["data"]["has_expiry"] is False
            
            # Check TTL for nonexistent key
            missing_ttl_result = await injector.run('storage_kv', {
                "operation": "ttl",
                "key": nonexistent_key,
                "namespace": namespace
            })
            
            if hasattr(missing_ttl_result, 'output'):
                missing_ttl_data = json.loads(missing_ttl_result.output)
            else:
                missing_ttl_data = missing_ttl_result
            
            assert "operation" in missing_ttl_data
            assert missing_ttl_data["data"]["ttl"] == -2  # Key doesn't exist
            assert missing_ttl_data["data"]["exists"] is False
        
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
            
            if hasattr(get1_result, 'output'):
                get1_data = json.loads(get1_result.output)
            else:
                get1_data = get1_result
            
            assert get1_data["data"]["value"] == value1
            
            # Get from namespace2
            get2_result = await injector.run('storage_kv', {
                "operation": "get",
                "key": key,
                "namespace": namespace2
            })
            
            if hasattr(get2_result, 'output'):
                get2_data = json.loads(get2_result.output)
            else:
                get2_data = get2_result
            
            assert get2_data["data"]["value"] == value2
            
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
            
            if hasattr(get2_after_delete, 'output'):
                get2_after_data = json.loads(get2_after_delete.output)
            else:
                get2_after_data = get2_after_delete
            
            assert get2_after_data["data"]["exists"] is True
            assert get2_after_data["data"]["value"] == value2
        
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
                
                if hasattr(set_result, 'output'):
                    set_data = json.loads(set_result.output)
                else:
                    set_data = set_result
                
                assert "operation" in set_data
            
            # Retrieve and verify all data types
            for key, expected_value in test_data.items():
                get_result = await injector.run('storage_kv', {
                    "operation": "get",
                    "key": key,
                    "namespace": namespace
                })
                
                if hasattr(get_result, 'output'):
                    get_data = json.loads(get_result.output)
                else:
                    get_data = get_result
                
                assert "operation" in get_data
                assert get_data["data"]["exists"] is True
                assert get_data["data"]["value"] == expected_value
        
        asyncio.run(run_test())
    
    def test_kv_error_handling(self):
        """Test error handling for edge cases."""
        
        async def run_test():
            injector = get_injector()
            namespace = "test"
            
            # Test getting non-existent key (should raise KeyError)
            try:
                get_result = await injector.run('storage_kv', {
                    "operation": "get",
                    "key": "nonexistent",
                    "namespace": namespace
                })
                assert False, "Expected KeyError for non-existent key"
            except KeyError as e:
                assert "nonexistent" in str(e)
                assert "not found" in str(e)
            
            # Test deleting non-existent key (should be idempotent)
            delete_result = await injector.run('storage_kv', {
                "operation": "delete",
                "key": "nonexistent",
                "namespace": namespace
            })
            
            if hasattr(delete_result, 'output'):
                delete_data = json.loads(delete_result.output)
            else:
                delete_data = delete_result
            
            assert "operation" in delete_data  # Idempotent
            assert delete_data["data"]["existed"] is False
            
            # Test expire on non-existent key (should raise KeyError)
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
            
            if hasattr(keys_result, 'output'):
                keys_data = json.loads(keys_result.output)
            else:
                keys_data = keys_result
            
            assert keys_data["data"]["count"] == 4
            
            # Wait for expiration
            await asyncio.sleep(1.1)
            
            # Trigger cleanup by doing a get operation
            get_result = await injector.run('storage_kv', {
                "operation": "get",
                "key": persistent_key,
                "namespace": namespace
            })
            
            if hasattr(get_result, 'output'):
                get_data = json.loads(get_result.output)
            else:
                get_data = get_result
            
            # Should report expired keys were cleaned
            assert get_data["data"]["expired_keys_cleaned"] == 3
            
            # Verify only persistent key remains
            keys_result = await injector.run('storage_kv', {
                "operation": "keys",
                "namespace": namespace
            })
            
            if hasattr(keys_result, 'output'):
                keys_data = json.loads(keys_result.output)
            else:
                keys_data = keys_result
            
            assert keys_data["data"]["count"] == 1
            assert persistent_key in keys_data["data"]["keys"]
        
        asyncio.run(run_test())