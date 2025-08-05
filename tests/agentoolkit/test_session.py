"""
Tests for session toolkit.

This module tests all functionality of the session management toolkit
including creation, validation, renewal, and integration with storage_kv.
"""

import json
import asyncio
import time
import pytest

from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry


class TestSession:
    """Test suite for session toolkit."""
    
    def setup_method(self):
        """Clear registry and injector before each test."""
        AgenToolRegistry.clear()
        get_injector().clear()
        
        # Import and create the agents
        from agentoolkit.storage.kv import create_storage_kv_agent, _kv_storage, _kv_expiry
        from agentoolkit.auth.session import create_session_agent, _sessions
        
        # Clear global storage
        _kv_storage.clear()
        _kv_expiry.clear()
        _sessions.clear()
        
        # Create agents (storage_kv first as session depends on it)
        storage_agent = create_storage_kv_agent()
        session_agent = create_session_agent()
    
    def test_session_create(self):
        """Test session creation."""
        
        async def run_test():
            injector = get_injector()
            
            # Create a session
            create_result = await injector.run('session', {
                "operation": "create",
                "user_id": "user123",
                "metadata": {
                    "ip": "192.168.1.1",
                    "user_agent": "Mozilla/5.0"
                },
                "ttl": 3600  # 1 hour
            })
            
            if hasattr(create_result, 'output'):
                create_data = json.loads(create_result.output)
            else:
                create_data = create_result
            
            assert "success" not in create_data
            assert create_data["operation"] == "create"
            assert "session_id" in create_data["data"]
            assert create_data["data"]["user_id"] == "user123"
            assert create_data["data"]["ttl"] == 3600
            
            session_id = create_data["data"]["session_id"]
            assert len(session_id) > 0
            
            # Verify session exists
            get_result = await injector.run('session', {
                "operation": "get",
                "session_id": session_id
            })
            
            if hasattr(get_result, 'output'):
                get_data = json.loads(get_result.output)
            else:
                get_data = get_result
            
            assert "success" not in get_data
            assert get_data["data"]["user_id"] == "user123"
            assert get_data["data"]["metadata"]["ip"] == "192.168.1.1"
        
        asyncio.run(run_test())
    
    def test_session_validation(self):
        """Test session validation."""
        
        async def run_test():
            injector = get_injector()
            
            # Create a session
            create_result = await injector.run('session', {
                "operation": "create",
                "user_id": "user456",
                "ttl": 3600
            })
            
            if hasattr(create_result, 'output'):
                create_data = json.loads(create_result.output)
            else:
                create_data = create_result
            
            session_id = create_data["data"]["session_id"]
            
            # Validate the session
            validate_result = await injector.run('session', {
                "operation": "validate",
                "session_id": session_id
            })
            
            if hasattr(validate_result, 'output'):
                validate_data = json.loads(validate_result.output)
            else:
                validate_data = validate_result
            
            assert "success" not in validate_data
            assert validate_data["data"]["valid"] is True
            assert validate_data["data"]["user_id"] == "user456"
            
            # Validate non-existent session
            invalid_result = await injector.run('session', {
                "operation": "validate",
                "session_id": "invalid_session_id"
            })
            
            if hasattr(invalid_result, 'output'):
                invalid_data = json.loads(invalid_result.output)
            else:
                invalid_data = invalid_result
            
            assert "success" not in invalid_data
            assert invalid_data["data"]["valid"] is False
        
        asyncio.run(run_test())
    
    def test_session_update(self):
        """Test session data update."""
        
        async def run_test():
            injector = get_injector()
            
            # Create a session
            create_result = await injector.run('session', {
                "operation": "create",
                "user_id": "user789"
            })
            
            if hasattr(create_result, 'output'):
                create_data = json.loads(create_result.output)
            else:
                create_data = create_result
            
            session_id = create_data["data"]["session_id"]
            
            # Update session data
            update_result = await injector.run('session', {
                "operation": "update",
                "session_id": session_id,
                "data": {
                    "cart_items": 3,
                    "last_page": "/products"
                }
            })
            
            if hasattr(update_result, 'output'):
                update_data = json.loads(update_result.output)
            else:
                update_data = update_result
            
            assert "success" not in update_data
            assert update_data["data"]["updated_fields"] == ["cart_items", "last_page"]
            
            # Get session to verify update
            get_result = await injector.run('session', {
                "operation": "get",
                "session_id": session_id
            })
            
            if hasattr(get_result, 'output'):
                get_data = json.loads(get_result.output)
            else:
                get_data = get_result
            
            assert get_data["data"]["data"]["cart_items"] == 3
            assert get_data["data"]["data"]["last_page"] == "/products"
        
        asyncio.run(run_test())
    
    def test_session_renewal(self):
        """Test session renewal/extension."""
        
        async def run_test():
            injector = get_injector()
            
            # Create a session with short TTL
            create_result = await injector.run('session', {
                "operation": "create",
                "user_id": "user_renew",
                "ttl": 60  # 1 minute
            })
            
            if hasattr(create_result, 'output'):
                create_data = json.loads(create_result.output)
            else:
                create_data = create_result
            
            session_id = create_data["data"]["session_id"]
            original_expires = create_data["data"]["expires_at"]
            
            # Renew the session
            renew_result = await injector.run('session', {
                "operation": "renew",
                "session_id": session_id,
                "ttl": 7200  # 2 hours
            })
            
            if hasattr(renew_result, 'output'):
                renew_data = json.loads(renew_result.output)
            else:
                renew_data = renew_result
            
            assert "success" not in renew_data
            assert renew_data["data"]["ttl"] == 7200
            
            new_expires = renew_data["data"]["new_expires_at"]
            assert new_expires != original_expires
        
        asyncio.run(run_test())
    
    def test_session_deletion(self):
        """Test session deletion."""
        
        async def run_test():
            injector = get_injector()
            
            # Create a session
            create_result = await injector.run('session', {
                "operation": "create",
                "user_id": "user_delete"
            })
            
            if hasattr(create_result, 'output'):
                create_data = json.loads(create_result.output)
            else:
                create_data = create_result
            
            session_id = create_data["data"]["session_id"]
            
            # Delete the session
            delete_result = await injector.run('session', {
                "operation": "delete",
                "session_id": session_id
            })
            
            if hasattr(delete_result, 'output'):
                delete_data = json.loads(delete_result.output)
            else:
                delete_data = delete_result
            
            assert "success" not in delete_data
            assert delete_data["data"]["existed"] is True
            
            # Verify session is gone - should raise KeyError
            with pytest.raises(Exception) as exc_info:
                await injector.run('session', {
                    "operation": "get",
                    "session_id": session_id
                })
            
            # Check that it's a KeyError about session not existing
            assert "does not exist" in str(exc_info.value)
        
        asyncio.run(run_test())
    
    def test_session_list_by_user(self):
        """Test listing sessions for a user."""
        
        async def run_test():
            injector = get_injector()
            user_id = "user_with_multiple"
            
            # Create multiple sessions for the same user
            session_ids = []
            for i in range(3):
                create_result = await injector.run('session', {
                    "operation": "create",
                    "user_id": user_id,
                    "metadata": {"session_num": i}
                })
                
                if hasattr(create_result, 'output'):
                    create_data = json.loads(create_result.output)
                else:
                    create_data = create_result
                
                session_ids.append(create_data["data"]["session_id"])
            
            # Create a session for different user
            await injector.run('session', {
                "operation": "create",
                "user_id": "other_user"
            })
            
            # List sessions for our user
            list_result = await injector.run('session', {
                "operation": "list",
                "user_id": user_id
            })
            
            if hasattr(list_result, 'output'):
                list_data = json.loads(list_result.output)
            else:
                list_data = list_result
            
            assert "success" not in list_data
            assert list_data["data"]["count"] == 3
            
            # Verify all sessions belong to the user
            for session in list_data["data"]["sessions"]:
                assert session["user_id"] == user_id
                assert session["session_id"] in session_ids
        
        asyncio.run(run_test())
    
    def test_session_invalidate_all(self):
        """Test invalidating all sessions for a user."""
        
        async def run_test():
            injector = get_injector()
            user_id = "user_invalidate"
            
            # Create multiple sessions
            session_ids = []
            for i in range(3):
                create_result = await injector.run('session', {
                    "operation": "create",
                    "user_id": user_id
                })
                
                if hasattr(create_result, 'output'):
                    create_data = json.loads(create_result.output)
                else:
                    create_data = create_result
                
                session_ids.append(create_data["data"]["session_id"])
            
            # Invalidate all sessions
            invalidate_result = await injector.run('session', {
                "operation": "invalidate_all",
                "user_id": user_id
            })
            
            if hasattr(invalidate_result, 'output'):
                invalidate_data = json.loads(invalidate_result.output)
            else:
                invalidate_data = invalidate_result
            
            assert "success" not in invalidate_data
            assert invalidate_data["data"]["invalidated_count"] == 3
            
            # Verify all sessions are gone
            for session_id in session_ids:
                with pytest.raises(Exception):
                    await injector.run('session', {
                        "operation": "get",
                        "session_id": session_id
                    })
        
        asyncio.run(run_test())
    
    def test_session_get_active(self):
        """Test getting all active sessions."""
        
        async def run_test():
            injector = get_injector()
            
            # Create sessions for different users
            users = ["user1", "user2", "user3"]
            for user in users:
                for i in range(2):
                    await injector.run('session', {
                        "operation": "create",
                        "user_id": user
                    })
            
            # Get all active sessions
            active_result = await injector.run('session', {
                "operation": "get_active"
            })
            
            if hasattr(active_result, 'output'):
                active_data = json.loads(active_result.output)
            else:
                active_data = active_result
            
            assert "success" not in active_data
            assert active_data["data"]["total_active"] == 6
            assert active_data["data"]["unique_users"] == 3
            
            # Verify user session counts
            users_data = active_data["data"]["users"]
            for user in users:
                assert users_data[user] == 2
        
        asyncio.run(run_test())
    
    def test_session_expiration(self):
        """Test session expiration handling."""
        
        async def run_test():
            injector = get_injector()
            
            # Create a session with very short TTL
            create_result = await injector.run('session', {
                "operation": "create",
                "user_id": "user_expire",
                "ttl": 1  # 1 second
            })
            
            if hasattr(create_result, 'output'):
                create_data = json.loads(create_result.output)
            else:
                create_data = create_result
            
            session_id = create_data["data"]["session_id"]
            
            # Wait for expiration
            time.sleep(2)
            
            # Try to get expired session - should raise RuntimeError
            with pytest.raises(Exception) as exc_info:
                await injector.run('session', {
                    "operation": "get",
                    "session_id": session_id
                })
            
            assert "expired" in str(exc_info.value)
        
        asyncio.run(run_test())
    
    def test_session_error_handling(self):
        """Test error handling for edge cases."""
        
        async def run_test():
            injector = get_injector()
            
            # Update non-existent session - should raise KeyError
            with pytest.raises(Exception) as exc_info:
                await injector.run('session', {
                    "operation": "update",
                    "session_id": "non_existent",
                    "data": {"test": "value"}
                })
            assert "does not exist" in str(exc_info.value)
            
            # Renew non-existent session - should raise KeyError
            with pytest.raises(Exception) as exc_info:
                await injector.run('session', {
                    "operation": "renew",
                    "session_id": "non_existent"
                })
            assert "does not exist" in str(exc_info.value)
            
            # Delete non-existent session (should be idempotent)
            delete_result = await injector.run('session', {
                "operation": "delete",
                "session_id": "non_existent"
            })
            
            if hasattr(delete_result, 'output'):
                delete_data = json.loads(delete_result.output)
            else:
                delete_data = delete_result
            
            assert "success" not in delete_data
            assert delete_data["data"]["existed"] is False
        
        asyncio.run(run_test())