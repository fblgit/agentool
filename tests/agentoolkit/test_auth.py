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
        from agentoolkit.security.crypto import create_crypto_agent
        from agentoolkit.auth.session import create_session_agent, _sessions
        from agentoolkit.auth.auth import create_auth_agent, _users, _role_permissions
        
        # Clear global storage
        _kv_storage.clear()
        _kv_expiry.clear()
        _sessions.clear()
        _users.clear()
        
        # Create agents in dependency order
        storage_agent = create_storage_kv_agent()
        crypto_agent = create_crypto_agent()
        session_agent = create_session_agent()
        auth_agent = create_auth_agent()
    
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
            
            if hasattr(register_result, 'output'):
                register_data = json.loads(register_result.output)
            else:
                register_data = register_result
            
            # No longer checking success field - function now throws exceptions on failure
            assert register_data["operation"] == "register"
            assert register_data["data"]["username"] == "john_doe"
            assert register_data["data"]["email"] == "john@example.com"
            assert "user" in register_data["data"]["roles"]
            
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
            
            if hasattr(login_result, 'output'):
                login_data = json.loads(login_result.output)
            else:
                login_data = login_result
            
            # No longer checking success field - function now throws exceptions on failure
            assert "session_id" in login_data["data"]
            assert login_data["data"]["username"] == "login_test"
            
            session_id = login_data["data"]["session_id"]
            
            # Verify session exists
            session_result = await injector.run('session', {
                "operation": "get",
                "session_id": session_id
            })
            
            if hasattr(session_result, 'output'):
                session_data = json.loads(session_result.output)
            else:
                session_data = session_result
            
            # No longer checking success field - function now throws exceptions on failure
            
            # Logout
            logout_result = await injector.run('auth', {
                "operation": "logout",
                "session_id": session_id
            })
            
            if hasattr(logout_result, 'output'):
                logout_data = json.loads(logout_result.output)
            else:
                logout_data = logout_result
            
            # No longer checking success field - function now throws exceptions on failure
            assert logout_data["data"]["session_deleted"] is True
            
            # Verify session is gone - should raise KeyError
            with pytest.raises(KeyError) as exc_info:
                await injector.run('session', {
                    "operation": "get",
                    "session_id": session_id
                })
            
            # Verify the error message contains expected text
            assert "does not exist" in str(exc_info.value)
        
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
            
            if hasattr(verify_result, 'output'):
                verify_data = json.loads(verify_result.output)
            else:
                verify_data = verify_result
            
            # No longer checking success field - function now throws exceptions on failure
            assert verify_data["data"]["valid"] is True
            
            # Verify wrong password
            wrong_result = await injector.run('auth', {
                "operation": "verify_password",
                "username": "verify_test",
                "password": "wrong_password"
            })
            
            if hasattr(wrong_result, 'output'):
                wrong_data = json.loads(wrong_result.output)
            else:
                wrong_data = wrong_result
            
            # No longer checking success field - function now throws exceptions on failure
            assert wrong_data["data"]["valid"] is False
            
            # Verify non-existent user
            nouser_result = await injector.run('auth', {
                "operation": "verify_password",
                "username": "non_existent",
                "password": "any_password"
            })
            
            if hasattr(nouser_result, 'output'):
                nouser_data = json.loads(nouser_result.output)
            else:
                nouser_data = nouser_result
            
            # No longer checking success field - function now throws exceptions on failure
            assert nouser_data["data"]["valid"] is False
        
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
            
            if hasattr(login_result, 'output'):
                login_data = json.loads(login_result.output)
            else:
                login_data = login_result
            
            session_id = login_data["data"]["session_id"]
            
            # Change password
            change_result = await injector.run('auth', {
                "operation": "change_password",
                "username": "change_pwd_test",
                "password": "old_password",
                "new_password": "new_password"
            })
            
            if hasattr(change_result, 'output'):
                change_data = json.loads(change_result.output)
            else:
                change_data = change_result
            
            # No longer checking success field - function now throws exceptions on failure
            assert change_data["data"]["sessions_invalidated"] is True
            
            # Verify old session is invalid - should raise KeyError
            with pytest.raises(KeyError) as exc_info:
                await injector.run('session', {
                    "operation": "get",
                    "session_id": session_id
                })
            
            # Verify the error message contains expected text
            assert "does not exist" in str(exc_info.value)
            
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
            
            if hasattr(new_login, 'output'):
                new_data = json.loads(new_login.output)
            else:
                new_data = new_login
            
            # No longer checking success field - function now throws exceptions on failure
        
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
            
            if hasattr(reset_result, 'output'):
                reset_data = json.loads(reset_result.output)
            else:
                reset_data = reset_result
            
            # No longer checking success field - function now throws exceptions on failure
            assert reset_data["data"]["sessions_invalidated"] is True
            
            # Verify new password works
            login_result = await injector.run('auth', {
                "operation": "login",
                "username": "reset_test",
                "password": "reset_password"
            })
            
            if hasattr(login_result, 'output'):
                login_data = json.loads(login_result.output)
            else:
                login_data = login_result
            
            # No longer checking success field - function now throws exceptions on failure
        
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
            
            if hasattr(token_result, 'output'):
                token_data = json.loads(token_result.output)
            else:
                token_data = token_result
            
            # No longer checking success field - function now throws exceptions on failure
            assert "token" in token_data["data"]
            assert token_data["data"]["expires_in"] == 3600
            
            token = token_data["data"]["token"]
            
            # Verify token
            verify_result = await injector.run('auth', {
                "operation": "verify_token",
                "token": token
            })
            
            if hasattr(verify_result, 'output'):
                verify_data = json.loads(verify_result.output)
            else:
                verify_data = verify_result
            
            # No longer checking success field - function now throws exceptions on failure
            assert verify_data["data"]["valid"] is True
            assert verify_data["data"]["user_id"] == "token_test"
            assert verify_data["data"]["metadata"]["purpose"] == "api_access"
            
            # Verify invalid token
            invalid_result = await injector.run('auth', {
                "operation": "verify_token",
                "token": "invalid.token.here"
            })
            
            if hasattr(invalid_result, 'output'):
                invalid_data = json.loads(invalid_result.output)
            else:
                invalid_data = invalid_result
            
            # Should return invalid token result
            assert invalid_data["data"]["valid"] is False
        
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
            
            if hasattr(perm_result, 'output'):
                perm_data = json.loads(perm_result.output)
            else:
                perm_data = perm_result
            
            # No longer checking success field - function now throws exceptions on failure
            assert perm_data["data"]["has_permission"] is False
            
            # Assign admin role
            assign_result = await injector.run('auth', {
                "operation": "assign_role",
                "user_id": "role_test",
                "role": "admin"
            })
            
            if hasattr(assign_result, 'output'):
                assign_data = json.loads(assign_result.output)
            else:
                assign_data = assign_result
            
            # No longer checking success field - function now throws exceptions on failure
            assert "admin" in assign_data["data"]["current_roles"]
            
            # Check permission again
            admin_perm = await injector.run('auth', {
                "operation": "check_permission",
                "user_id": "role_test",
                "permission": "admin"
            })
            
            if hasattr(admin_perm, 'output'):
                admin_data = json.loads(admin_perm.output)
            else:
                admin_data = admin_perm
            
            assert admin_data["data"]["has_permission"] is True
            
            # Revoke admin role
            revoke_result = await injector.run('auth', {
                "operation": "revoke_role",
                "user_id": "role_test",
                "role": "admin"
            })
            
            if hasattr(revoke_result, 'output'):
                revoke_data = json.loads(revoke_result.output)
            else:
                revoke_data = revoke_result
            
            # No longer checking success field - function now throws exceptions on failure
            assert "admin" not in revoke_data["data"]["current_roles"]
            
            # Verify permission revoked
            final_perm = await injector.run('auth', {
                "operation": "check_permission",
                "user_id": "role_test",
                "permission": "admin"
            })
            
            if hasattr(final_perm, 'output'):
                final_data = json.loads(final_perm.output)
            else:
                final_data = final_perm
            
            assert final_data["data"]["has_permission"] is False
        
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
                    
                    if hasattr(result, 'output'):
                        data = json.loads(result.output)
                    else:
                        data = result
                    
                    assert data["data"]["has_permission"] == expected[username][permission], \
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
            
            if hasattr(get_result, 'output'):
                get_data = json.loads(get_result.output)
            else:
                get_data = get_result
            
            # No longer checking success field - function now throws exceptions on failure
            user_info = get_data["data"]
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
            
            if hasattr(update_result, 'output'):
                update_data = json.loads(update_result.output)
            else:
                update_data = update_result
            
            # No longer checking success field - function now throws exceptions on failure
            assert set(update_data["data"]["updated_fields"]) == {"age", "occupation"}
            
            # Verify update
            get_updated = await injector.run('auth', {
                "operation": "get_user",
                "user_id": "manage_test"
            })
            
            if hasattr(get_updated, 'output'):
                updated_data = json.loads(get_updated.output)
            else:
                updated_data = get_updated
            
            user_info = updated_data["data"]
            assert user_info["metadata"]["age"] == 26
            assert user_info["metadata"]["occupation"] == "Engineer"
            assert user_info["metadata"]["city"] == "New York"  # Original preserved
            
            # Delete user
            delete_result = await injector.run('auth', {
                "operation": "delete_user",
                "user_id": "manage_test"
            })
            
            if hasattr(delete_result, 'output'):
                delete_data = json.loads(delete_result.output)
            else:
                delete_data = delete_result
            
            # No longer checking success field - function now throws exceptions on failure
            assert delete_data["data"]["sessions_invalidated"] is True
            
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