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
        from agentoolkit.observability.metrics import create_metrics_agent
        from agentoolkit.system.logging import create_logging_agent
        from agentoolkit.storage.fs import create_storage_fs_agent
        
        # Clear global storage
        _kv_storage.clear()
        _kv_expiry.clear()
        _sessions.clear()
        
        # Create agents in dependency order
        storage_kv_agent = create_storage_kv_agent()  # No dependencies
        storage_fs_agent = create_storage_fs_agent()  # No dependencies (needed by logging)
        metrics_agent = create_metrics_agent()  # Depends on storage_kv
        logging_agent = create_logging_agent()  # Depends on storage_fs
        session_agent = create_session_agent()  # Depends on storage_kv, uses metrics and logging
    
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
            
            # session returns typed SessionOutput
            assert create_result.success is True
            assert create_result.operation == "create"
            assert "session_id" in create_result.data
            assert create_result.data["user_id"] == "user123"
            assert create_result.data["ttl"] == 3600
            
            session_id = create_result.data["session_id"]
            assert len(session_id) > 0
            
            # Verify session exists
            get_result = await injector.run('session', {
                "operation": "get",
                "session_id": session_id
            })
            
            # session returns typed SessionOutput
            assert get_result.success is True
            assert get_result.data["user_id"] == "user123"
            assert get_result.data["metadata"]["ip"] == "192.168.1.1"
        
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
            
            # session returns typed SessionOutput
            session_id = create_result.data["session_id"]
            
            # Validate the session
            validate_result = await injector.run('session', {
                "operation": "validate",
                "session_id": session_id
            })
            
            # session returns typed SessionOutput
            assert validate_result.success is True
            assert validate_result.data["valid"] is True
            assert validate_result.data["user_id"] == "user456"
            
            # Validate non-existent session
            invalid_result = await injector.run('session', {
                "operation": "validate",
                "session_id": "invalid_session_id"
            })
            
            # session returns typed SessionOutput
            assert invalid_result.success is False  # Discovery operation - session not found
            assert invalid_result.data["valid"] is False
        
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
            
            # session returns typed SessionOutput
            session_id = create_result.data["session_id"]
            
            # Update session data
            update_result = await injector.run('session', {
                "operation": "update",
                "session_id": session_id,
                "data": {
                    "cart_items": 3,
                    "last_page": "/products"
                }
            })
            
            # session returns typed SessionOutput
            assert update_result.success is True
            assert update_result.data["updated_fields"] == ["cart_items", "last_page"]
            
            # Get session to verify update
            get_result = await injector.run('session', {
                "operation": "get",
                "session_id": session_id
            })
            
            # session returns typed SessionOutput
            assert get_result.success is True
            assert get_result.data["data"]["cart_items"] == 3
            assert get_result.data["data"]["last_page"] == "/products"
        
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
            
            # session returns typed SessionOutput
            session_id = create_result.data["session_id"]
            original_expires = create_result.data["expires_at"]
            
            # Renew the session
            renew_result = await injector.run('session', {
                "operation": "renew",
                "session_id": session_id,
                "ttl": 7200  # 2 hours
            })
            
            # session returns typed SessionOutput
            assert renew_result.success is True
            assert renew_result.data["ttl"] == 7200
            
            new_expires = renew_result.data["new_expires_at"]
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
            
            # session returns typed SessionOutput
            session_id = create_result.data["session_id"]
            
            # Delete the session
            delete_result = await injector.run('session', {
                "operation": "delete",
                "session_id": session_id
            })
            
            # session returns typed SessionOutput
            assert delete_result.success is True
            assert delete_result.data["existed"] is True
            
            # Verify session is gone - should return success=False
            get_result = await injector.run('session', {
                "operation": "get",
                "session_id": session_id
            })
            
            # Check that session was not found (discovery operation)
            assert get_result.success is False
            assert "does not exist" in get_result.message
        
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
                
                # Session agent has use_typed_output=True, so we get typed result directly
                session_ids.append(create_result.data["session_id"])
            
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
            
            # session returns typed SessionOutput
            assert list_result.success is True
            assert list_result.data["count"] == 3
            
            # Verify all sessions belong to the user
            for session in list_result.data["sessions"]:
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
                
                # Session agent has use_typed_output=True, so we get typed result directly
                session_ids.append(create_result.data["session_id"])
            
            # Invalidate all sessions
            invalidate_result = await injector.run('session', {
                "operation": "invalidate_all",
                "user_id": user_id
            })
            
            # session returns typed SessionOutput
            assert invalidate_result.success is True
            assert invalidate_result.data["invalidated_count"] == 3
            
            # Verify all sessions are gone
            for session_id in session_ids:
                get_result = await injector.run('session', {
                    "operation": "get",
                    "session_id": session_id
                })
                # Should return success=False for not found
                assert get_result.success is False
        
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
            
            # session returns typed SessionOutput
            assert active_result.success is True
            assert active_result.data["total_active"] == 6
            assert active_result.data["unique_users"] == 3
            
            # Verify user session counts
            users_data = active_result.data["users"]
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
            
            # session returns typed SessionOutput
            session_id = create_result.data["session_id"]
            
            # Wait for expiration
            time.sleep(2)
            
            # Try to get expired session - should return success=False
            get_result = await injector.run('session', {
                "operation": "get",
                "session_id": session_id
            })
            
            # Check that session is expired (discovery operation)
            assert get_result.success is False
            assert "expired" in get_result.message
        
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
            
            # session returns typed SessionOutput
            assert delete_result.success is True  # Delete is idempotent
            assert delete_result.data["existed"] is False
        
        asyncio.run(run_test())