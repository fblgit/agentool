"""
Tests for auth toolkit.

This module tests all functionality of the authentication and authorization toolkit
including user registration, login/logout, password management, token generation,
and role-based access control.
"""

import json
import asyncio
import time
import pytest
from datetime import datetime, timezone

from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry


class TestAuth:
    """Test suite for auth toolkit."""
    
    def setup_method(self):
        """Clear registry and injector before each test."""
        AgenToolRegistry.clear()
        get_injector().clear()
        
        # Import and create the agents (in dependency order)
        from agentoolkit.storage.kv import create_storage_kv_agent, _kv_storage, _kv_expiry
        from agentoolkit.storage.fs import create_storage_fs_agent
        from agentoolkit.observability.metrics import create_metrics_agent
        from agentoolkit.system.logging import create_logging_agent, _logging_config
        from agentoolkit.security.crypto import create_crypto_agent
        from agentoolkit.auth.session import create_session_agent, _sessions
        from agentoolkit.auth.auth import create_auth_agent, _users, _role_permissions
        
        # Clear global storage
        _kv_storage.clear()
        _kv_expiry.clear()
        _sessions.clear()
        _users.clear()
        _logging_config.clear()
        
        # Create agents in dependency order
        storage_fs_agent = create_storage_fs_agent()  # No dependencies
        storage_agent = create_storage_kv_agent()     # No dependencies
        crypto_agent = create_crypto_agent()          # No dependencies
        metrics_agent = create_metrics_agent()        # Depends on storage_kv
        logging_agent = create_logging_agent()        # Depends on storage_fs and metrics
        session_agent = create_session_agent()        # Depends on storage_kv, logging, metrics
        auth_agent = create_auth_agent()             # Depends on crypto, storage_kv, session
    
    def test_user_registration(self):
        """Test user registration."""
        
        async def run_test():
            injector = get_injector()
            
            # Register a new user
            register_result = await injector.run('auth', {
                "operation": "register",
                "username": "john_doe",
                "password": "secure_password123",
                "email": "john@example.com",
                "metadata": {
                    "full_name": "John Doe",
                    "country": "USA"
                }
            })
            
            # auth returns typed AuthOutput
            assert register_result.success is True
            assert register_result.operation == "register"
            assert register_result.data["username"] == "john_doe"
            assert register_result.data["email"] == "john@example.com"
            assert "user" in register_result.data["roles"]
            
            # Try registering same user again - should throw exception
            try:
                duplicate_result = await injector.run('auth', {
                    "operation": "register",
                    "username": "john_doe",
                    "password": "another_password"
                })
                assert False, "Should have thrown exception for duplicate user"
            except Exception as e:
                assert "already exists" in str(e)
        
        asyncio.run(run_test())
    
    def test_user_login_logout(self):
        """Test user login and logout."""
        
        async def run_test():
            injector = get_injector()
            
            # Register user first
            await injector.run('auth', {
                "operation": "register",
                "username": "login_test",
                "password": "test_password"
            })
            
            # Login
            login_result = await injector.run('auth', {
                "operation": "login",
                "username": "login_test",
                "password": "test_password"
            })
            
            # auth returns typed AuthOutput
            assert login_result.success is True
            assert "session_id" in login_result.data
            assert login_result.data["username"] == "login_test"
            
            session_id = login_result.data["session_id"]
            
            # Verify session exists
            session_result = await injector.run('session', {
                "operation": "get",
                "session_id": session_id
            })
            
            # session returns typed SessionOutput
            
            # No longer checking success field - function now throws exceptions on failure
            
            # Logout
            logout_result = await injector.run('auth', {
                "operation": "logout",
                "session_id": session_id
            })
            
            # auth returns typed AuthOutput
            assert logout_result.success is True
            assert logout_result.data["session_deleted"] is True
            
            # Verify session is gone - should return success=False
            get_deleted = await injector.run('session', {
                "operation": "get",
                "session_id": session_id
            })
            # session returns typed SessionOutput with success=False for not found
            assert get_deleted.success is False
        
        asyncio.run(run_test())
    
    def test_password_verification(self):
        """Test password verification."""
        
        async def run_test():
            injector = get_injector()
            
            # Register user
            await injector.run('auth', {
                "operation": "register",
                "username": "verify_test",
                "password": "correct_password"
            })
            
            # Verify correct password
            verify_result = await injector.run('auth', {
                "operation": "verify_password",
                "username": "verify_test",
                "password": "correct_password"
            })
            
            # auth returns typed AuthOutput
            assert verify_result.success is True
            assert verify_result.data["valid"] is True
            
            # Verify wrong password
            wrong_result = await injector.run('auth', {
                "operation": "verify_password",
                "username": "verify_test",
                "password": "wrong_password"
            })
            
            # auth returns typed AuthOutput
            assert wrong_result.success is True
            assert wrong_result.data["valid"] is False
            
            # Verify non-existent user
            nouser_result = await injector.run('auth', {
                "operation": "verify_password",
                "username": "non_existent",
                "password": "any_password"
            })
            
            # auth returns typed AuthOutput - should return success=False for discovery operation
            assert nouser_result.success is False
            assert nouser_result.data["valid"] is False
        
        asyncio.run(run_test())
    
    def test_password_change(self):
        """Test password change functionality."""
        
        async def run_test():
            injector = get_injector()
            
            # Register user
            await injector.run('auth', {
                "operation": "register",
                "username": "change_pwd_test",
                "password": "old_password"
            })
            
            # Login to get session
            login_result = await injector.run('auth', {
                "operation": "login",
                "username": "change_pwd_test",
                "password": "old_password"
            })
            
            # auth returns typed AuthOutput
            
            session_id = login_result.data["session_id"]
            
            # Change password
            change_result = await injector.run('auth', {
                "operation": "change_password",
                "username": "change_pwd_test",
                "password": "old_password",
                "new_password": "new_password"
            })
            
            # auth returns typed AuthOutput
            assert change_result.success is True
            assert change_result.data["sessions_invalidated"] is True
            
            # Verify old session is invalid - should return success=False
            get_deleted = await injector.run('session', {
                "operation": "get",
                "session_id": session_id
            })
            # session returns typed SessionOutput with success=False for not found
            assert get_deleted.success is False
            
            # Verify old password doesn't work - should throw exception
            try:
                old_login = await injector.run('auth', {
                    "operation": "login",
                    "username": "change_pwd_test",
                    "password": "old_password"
                })
                assert False, "Should have thrown exception for invalid password"
            except Exception as e:
                assert "Invalid username or password" in str(e) or "password" in str(e).lower()
            
            # Verify new password works
            new_login = await injector.run('auth', {
                "operation": "login",
                "username": "change_pwd_test",
                "password": "new_password"
            })
            
            # auth returns typed AuthOutput
            assert new_login.success is True
        
        asyncio.run(run_test())
    
    def test_password_reset(self):
        """Test password reset (admin functionality)."""
        
        async def run_test():
            injector = get_injector()
            
            # Register user
            await injector.run('auth', {
                "operation": "register",
                "username": "reset_test",
                "password": "original_password"
            })
            
            # Reset password
            reset_result = await injector.run('auth', {
                "operation": "reset_password",
                "username": "reset_test",
                "new_password": "reset_password"
            })
            
            # auth returns typed AuthOutput
            assert reset_result.success is True
            assert reset_result.data["sessions_invalidated"] is True
            
            # Verify new password works
            login_result = await injector.run('auth', {
                "operation": "login",
                "username": "reset_test",
                "password": "reset_password"
            })
            
            # auth returns typed AuthOutput
            assert login_result.success is True
        
        asyncio.run(run_test())
    
    def test_token_generation_verification(self):
        """Test JWT token generation and verification."""
        
        async def run_test():
            injector = get_injector()
            
            # Register user
            await injector.run('auth', {
                "operation": "register",
                "username": "token_test",
                "password": "password"
            })
            
            # Generate token
            token_result = await injector.run('auth', {
                "operation": "generate_token",
                "user_id": "token_test",
                "metadata": {
                    "purpose": "api_access",
                    "scope": "read_write"
                }
            })
            
            # auth returns typed AuthOutput
            assert token_result.success is True
            assert "token" in token_result.data
            assert token_result.data["expires_in"] == 3600
            
            token = token_result.data["token"]
            
            # Verify token
            verify_result = await injector.run('auth', {
                "operation": "verify_token",
                "token": token
            })
            
            # auth returns typed AuthOutput
            assert verify_result.success is True
            assert verify_result.data["valid"] is True
            assert verify_result.data["user_id"] == "token_test"
            assert verify_result.data["metadata"]["purpose"] == "api_access"
            
            # Verify invalid token
            invalid_result = await injector.run('auth', {
                "operation": "verify_token",
                "token": "invalid.token.here"
            })
            
            # auth returns typed AuthOutput - should return success=False for invalid token
            assert invalid_result.success is False
            assert invalid_result.data["valid"] is False
        
        asyncio.run(run_test())
    
    def test_role_management(self):
        """Test role assignment and revocation."""
        
        async def run_test():
            injector = get_injector()
            
            # Register user
            await injector.run('auth', {
                "operation": "register",
                "username": "role_test",
                "password": "password"
            })
            
            # Check initial permissions
            perm_result = await injector.run('auth', {
                "operation": "check_permission",
                "user_id": "role_test",
                "permission": "admin"
            })
            
            # auth returns typed AuthOutput
            assert perm_result.success is True
            assert perm_result.data["has_permission"] is False
            
            # Assign admin role
            assign_result = await injector.run('auth', {
                "operation": "assign_role",
                "user_id": "role_test",
                "role": "admin"
            })
            
            # auth returns typed AuthOutput
            assert assign_result.success is True
            assert "admin" in assign_result.data["current_roles"]
            
            # Check permission again
            admin_perm = await injector.run('auth', {
                "operation": "check_permission",
                "user_id": "role_test",
                "permission": "admin"
            })
            
            # auth returns typed AuthOutput
            assert admin_perm.success is True
            assert admin_perm.data["has_permission"] is True
            
            # Revoke admin role
            revoke_result = await injector.run('auth', {
                "operation": "revoke_role",
                "user_id": "role_test",
                "role": "admin"
            })
            
            # auth returns typed AuthOutput
            assert revoke_result.success is True
            assert "admin" not in revoke_result.data["current_roles"]
            
            # Verify permission revoked
            final_perm = await injector.run('auth', {
                "operation": "check_permission",
                "user_id": "role_test",
                "permission": "admin"
            })
            
            # auth returns typed AuthOutput
            assert final_perm.success is True
            assert final_perm.data["has_permission"] is False
        
        asyncio.run(run_test())
    
    def test_permission_checking(self):
        """Test permission checking across different roles."""
        
        async def run_test():
            injector = get_injector()
            
            # Register users with different roles
            users = [
                ("guest_user", "guest"),
                ("normal_user", "user"),
                ("admin_user", "admin")
            ]
            
            for username, role in users:
                # Register
                await injector.run('auth', {
                    "operation": "register",
                    "username": username,
                    "password": "password"
                })
                
                # Assign role if not default user
                if role != "user":
                    if role == "guest":
                        # First revoke user role
                        await injector.run('auth', {
                            "operation": "revoke_role",
                            "user_id": username,
                            "role": "user"
                        })
                    
                    await injector.run('auth', {
                        "operation": "assign_role",
                        "user_id": username,
                        "role": role
                    })
            
            # Test permissions
            permissions = ["read", "write", "delete", "admin"]
            expected = {
                "guest_user": {"read": True, "write": False, "delete": False, "admin": False},
                "normal_user": {"read": True, "write": True, "delete": False, "admin": False},
                "admin_user": {"read": True, "write": True, "delete": True, "admin": True}
            }
            
            for username, _ in users:
                for permission in permissions:
                    result = await injector.run('auth', {
                        "operation": "check_permission",
                        "user_id": username,
                        "permission": permission
                    })
                    
                    # auth returns typed AuthOutput
                    assert result.success is True
                    assert result.data["has_permission"] == expected[username][permission], \
                        f"{username} should {'have' if expected[username][permission] else 'not have'} {permission} permission"
        
        asyncio.run(run_test())
    
    def test_user_management(self):
        """Test user get, update, and delete operations."""
        
        async def run_test():
            injector = get_injector()
            
            # Register user
            await injector.run('auth', {
                "operation": "register",
                "username": "manage_test",
                "password": "password",
                "metadata": {
                    "age": 25,
                    "city": "New York"
                }
            })
            
            # Get user
            get_result = await injector.run('auth', {
                "operation": "get_user",
                "user_id": "manage_test"
            })
            
            # auth returns typed AuthOutput
            assert get_result.success is True
            user_info = get_result.data
            assert user_info["username"] == "manage_test"
            assert user_info["metadata"]["age"] == 25
            assert "password_hash" not in user_info  # Sensitive data removed
            assert "salt" not in user_info
            
            # Update user
            update_result = await injector.run('auth', {
                "operation": "update_user",
                "user_id": "manage_test",
                "metadata": {
                    "age": 26,
                    "occupation": "Engineer"
                }
            })
            
            # auth returns typed AuthOutput
            assert update_result.success is True
            assert set(update_result.data["updated_fields"]) == {"age", "occupation"}
            
            # Verify update
            get_updated = await injector.run('auth', {
                "operation": "get_user",
                "user_id": "manage_test"
            })
            
            # auth returns typed AuthOutput
            assert get_updated.success is True
            user_info = get_updated.data
            assert user_info["metadata"]["age"] == 26
            assert user_info["metadata"]["occupation"] == "Engineer"
            assert user_info["metadata"]["city"] == "New York"  # Original preserved
            
            # Delete user
            delete_result = await injector.run('auth', {
                "operation": "delete_user",
                "user_id": "manage_test"
            })
            
            # auth returns typed AuthOutput
            assert delete_result.success is True
            assert delete_result.data["sessions_invalidated"] is True
            
            # Verify deletion - should throw exception for non-existent user
            try:
                get_deleted = await injector.run('auth', {
                    "operation": "get_user",
                    "user_id": "manage_test"
                })
                assert False, "Should have thrown exception for non-existent user"
            except Exception as e:
                assert "does not exist" in str(e)
        
        asyncio.run(run_test())
    
    def test_inactive_user_handling(self):
        """Test handling of inactive users."""
        
        async def run_test():
            injector = get_injector()
            
            # Register user
            await injector.run('auth', {
                "operation": "register",
                "username": "inactive_test",
                "password": "password"
            })
            
            # Manually deactivate user (in real scenario, admin would do this)
            from agentoolkit.auth.auth import _users
            _users["inactive_test"]["active"] = False
            
            # Try to login - should throw exception for deactivated user
            try:
                login_result = await injector.run('auth', {
                    "operation": "login",
                    "username": "inactive_test",
                    "password": "password"
                })
                assert False, "Should have thrown exception for deactivated user"
            except Exception as e:
                assert "deactivated" in str(e)
        
        asyncio.run(run_test())
    
    def test_error_handling(self):
        """Test error handling for edge cases."""
        
        async def run_test():
            injector = get_injector()
            
            # Login with non-existent user - should throw exception
            try:
                login_result = await injector.run('auth', {
                    "operation": "login",
                    "username": "non_existent",
                    "password": "password"
                })
                assert False, "Should have thrown exception for non-existent user"
            except Exception as e:
                assert "Invalid username or password" in str(e) or "password" in str(e).lower()
            
            # Change password with wrong current password
            await injector.run('auth', {
                "operation": "register",
                "username": "error_test",
                "password": "correct_password"
            })
            
            # Should throw exception for wrong password
            try:
                change_result = await injector.run('auth', {
                    "operation": "change_password",
                    "username": "error_test",
                    "password": "wrong_password",
                    "new_password": "new_password"
                })
                assert False, "Should have thrown exception for wrong password"
            except Exception as e:
                assert "incorrect" in str(e)
            
            # Generate token for non-existent user - should throw exception
            try:
                token_result = await injector.run('auth', {
                    "operation": "generate_token",
                    "user_id": "non_existent"
                })
                assert False, "Should have thrown exception for non-existent user"
            except Exception as e:
                assert "not found" in str(e)
            
            # Assign non-existent role - should throw exception
            try:
                role_result = await injector.run('auth', {
                    "operation": "assign_role",
                    "user_id": "error_test",
                    "role": "super_admin"
                })
                assert False, "Should have thrown exception for non-existent role"
            except Exception as e:
                assert "does not exist" in str(e)
        
        asyncio.run(run_test())