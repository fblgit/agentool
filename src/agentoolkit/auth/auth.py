"""
Auth AgenTool - Provides authentication and authorization capabilities.

This toolkit provides comprehensive authentication functionality including:
- User authentication (login/logout)
- Password management
- Token generation and validation
- Role-based access control
- Multi-factor authentication support

Example Usage:
    >>> from agentoolkit.auth.auth import create_auth_agent
    >>> from agentool.core.injector import get_injector
    >>> 
    >>> # Create and register the agent
    >>> agent = create_auth_agent()
    >>> 
    >>> # Use through injector
    >>> injector = get_injector()
    >>> result = await injector.run('auth', {
    ...     "operation": "login",
    ...     "username": "user@example.com",
    ...     "password": "secure_password"
    ... })
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Literal, Set
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import RunContext

from agentool.base import BaseOperationInput
from agentool import create_agentool
from agentool.core.registry import RoutingConfig
from agentool.core.injector import get_injector


# Global user storage (in production, use database)
_users: Dict[str, Dict[str, Any]] = {}

# Global role permissions
_role_permissions: Dict[str, Set[str]] = {
    "admin": {"read", "write", "delete", "admin"},
    "user": {"read", "write"},
    "guest": {"read"}
}


class AuthInput(BaseOperationInput):
    """Input schema for authentication operations."""
    operation: Literal[
        'register', 'login', 'logout', 'verify_password', 'change_password',
        'reset_password', 'generate_token', 'verify_token', 'check_permission',
        'assign_role', 'revoke_role', 'get_user', 'update_user', 'delete_user'
    ] = Field(description="The authentication operation to perform")
    
    # User credentials
    username: Optional[str] = Field(None, description="Username or email")
    password: Optional[str] = Field(None, description="User password")
    new_password: Optional[str] = Field(None, description="New password for changes")
    
    # User data
    user_id: Optional[str] = Field(None, description="User identifier")
    email: Optional[str] = Field(None, description="User email")
    metadata: Optional[Dict[str, Any]] = Field(None, description="User metadata")
    
    # Session/Token
    session_id: Optional[str] = Field(None, description="Session identifier")
    token: Optional[str] = Field(None, description="Authentication token")
    
    # Authorization
    role: Optional[str] = Field(None, description="User role")
    permission: Optional[str] = Field(None, description="Permission to check")
    
    @field_validator('username')
    def validate_username(cls, v, info):
        """Validate username is provided for operations that require it."""
        operation = info.data.get('operation')
        if operation in ['register', 'login', 'verify_password', 'change_password', 
                         'reset_password'] and not v:
            raise ValueError(f"username is required for {operation} operation")
        return v
    
    @field_validator('password')
    def validate_password(cls, v, info):
        """Validate password is provided for operations that require it."""
        operation = info.data.get('operation')
        if operation in ['register', 'login', 'verify_password', 'change_password'] and not v:
            raise ValueError(f"password is required for {operation} operation")
        return v
    
    @field_validator('new_password')
    def validate_new_password(cls, v, info):
        """Validate new_password is provided for operations that require it."""
        operation = info.data.get('operation')
        if operation in ['change_password', 'reset_password'] and not v:
            raise ValueError(f"new_password is required for {operation} operation")
        return v
    
    @field_validator('user_id')
    def validate_user_id(cls, v, info):
        """Validate user_id is provided for operations that require it."""
        operation = info.data.get('operation')
        if operation in ['generate_token', 'check_permission', 'assign_role', 
                         'revoke_role', 'get_user', 'update_user', 'delete_user'] and not v:
            raise ValueError(f"user_id is required for {operation} operation")
        return v
    
    @field_validator('session_id')
    def validate_session_id(cls, v, info):
        """Validate session_id is provided for operations that require it."""
        operation = info.data.get('operation')
        if operation == 'logout' and not v:
            raise ValueError(f"session_id is required for {operation} operation")
        return v
    
    @field_validator('token')
    def validate_token(cls, v, info):
        """Validate token is provided for operations that require it."""
        operation = info.data.get('operation')
        if operation == 'verify_token' and not v:
            raise ValueError(f"token is required for {operation} operation")
        return v
    
    @field_validator('permission')
    def validate_permission(cls, v, info):
        """Validate permission is provided for operations that require it."""
        operation = info.data.get('operation')
        if operation == 'check_permission' and not v:
            raise ValueError(f"permission is required for {operation} operation")
        return v
    
    @field_validator('role')
    def validate_role(cls, v, info):
        """Validate role is provided for operations that require it."""
        operation = info.data.get('operation')
        if operation in ['assign_role', 'revoke_role'] and not v:
            raise ValueError(f"role is required for {operation} operation")
        return v


class AuthOutput(BaseModel):
    """Structured output for authentication operations."""
    success: bool = Field(description="Whether the operation succeeded")
    operation: str = Field(description="The operation that was performed")
    message: str = Field(description="Human-readable result message")
    data: Optional[Dict[str, Any]] = Field(None, description="Operation-specific data")


async def auth_register(ctx: RunContext[Any], username: str, password: str,
                       email: Optional[str], metadata: Optional[Dict[str, Any]]) -> AuthOutput:
    """
    Register a new user.
    
    Args:
        ctx: Runtime context
        username: Username
        password: Password
        email: Email address
        metadata: Additional user metadata
        
    Returns:
        AuthOutput with registration status
    """
    try:
        # Check if user exists
        if username in _users:
            raise ValueError(f"Username '{username}' already exists")
        
        # Hash password using crypto toolkit
        injector = get_injector()
        
        # Generate salt
        salt_result = await injector.run('crypto', {
            "operation": "generate_salt"
        })
        
        # crypto returns typed CryptoOutput
        salt = salt_result.data["salt"]
        
        # Hash password with salt
        hash_result = await injector.run('crypto', {
            "operation": "hash",
            "algorithm": "bcrypt",
            "data": password,
            "salt": salt,
            "iterations": 10000
        })
        
        # crypto returns typed CryptoOutput
        # Create user
        user_data = {
            "user_id": username,  # Using username as ID for simplicity
            "username": username,
            "email": email or f"{username}@example.com",
            "password_hash": hash_result.data["hash"],
            "salt": salt,
            "roles": ["user"],  # Default role
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_login": None,
            "metadata": metadata or {},
            "active": True,
            "mfa_enabled": False
        }
        
        # Store user
        _users[username] = user_data
        
        # Persist using storage_kv
        await injector.run('storage_kv', {
            "operation": "set",
            "key": f"user:{username}",
            "value": user_data,
            "namespace": "auth"
        })
        
        # Record business metrics
        try:
            await injector.run('metrics', {
                "operation": "increment",
                "name": "agentool.auth.registrations.success",
                "value": 1
            })
            await injector.run('metrics', {
                "operation": "increment", 
                "name": "agentool.auth.users.total",
                "value": 1
            })
        except:
            pass  # Ignore metrics errors
        
        return AuthOutput(
            success=True,
            operation="register",
            message=f"User {username} registered successfully",
            data={
                "user_id": user_data["user_id"],
                "username": username,
                "email": user_data["email"],
                "roles": user_data["roles"]
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to register user '{username}': {e}") from e


async def auth_login(ctx: RunContext[Any], username: str, password: str) -> AuthOutput:
    """
    Authenticate a user and create a session.
    
    Args:
        ctx: Runtime context
        username: Username
        password: Password
        
    Returns:
        AuthOutput with login status and session
    """
    try:
        # Get user
        if username not in _users:
            # Try loading from storage
            injector = get_injector()
            try:
                result = await injector.run('storage_kv', {
                    "operation": "get",
                    "key": f"user:{username}",
                    "namespace": "auth"
                })
                
                # storage_kv returns typed StorageKvOutput
                if result.success and result.data and result.data["exists"]:
                    _users[username] = result.data["value"]
                else:
                    raise ValueError("Invalid username or password")
            except (KeyError, TypeError):
                raise ValueError("Invalid username or password")
        
        user_data = _users[username]
        
        # Check if user is active
        if not user_data.get("active", True):
            raise PermissionError("User account is deactivated")
        
        # Verify password
        injector = get_injector()
        verify_result = await injector.run('crypto', {
            "operation": "verify_hash",
            "algorithm": "bcrypt",
            "data": password,
            "key": user_data["password_hash"],
            "salt": user_data["salt"]
        })
        
        # crypto returns typed CryptoOutput
        if not verify_result.success or not verify_result.data["valid"]:
            raise ValueError("Invalid username or password")
        
        # Create session
        session_result = await injector.run('session', {
            "operation": "create",
            "user_id": username,
            "metadata": {
                "login_time": datetime.now(timezone.utc).isoformat(),
                "ip": "127.0.0.1"  # Would get from request in production
            },
            "ttl": 86400  # 24 hours
        })
        
        # session returns typed SessionOutput
        
        # Update last login
        user_data["last_login"] = datetime.now(timezone.utc).isoformat()
        _users[username] = user_data
        
        await injector.run('storage_kv', {
            "operation": "set",
            "key": f"user:{username}",
            "value": user_data,
            "namespace": "auth"
        })
        
        # Record business metrics
        try:
            await injector.run('metrics', {
                "operation": "increment",
                "name": "agentool.auth.logins.success",
                "value": 1
            })
        except:
            pass  # Ignore metrics errors
        
        return AuthOutput(
            success=True,
            operation="login",
            message=f"User {username} logged in successfully",
            data={
                "user_id": user_data["user_id"],
                "username": username,
                "session_id": session_result.data["session_id"],
                "roles": user_data["roles"]
            }
        )
        
    except Exception as e:
        # Record login failure metric
        try:
            await injector.run('metrics', {
                "operation": "increment", 
                "name": "agentool.auth.logins.failure",
                "value": 1
            })
        except:
            pass  # Ignore metrics errors
        raise RuntimeError(f"Failed to login user '{username}': {e}") from e


async def auth_logout(ctx: RunContext[Any], session_id: str) -> AuthOutput:
    """
    Logout a user by invalidating their session.
    
    Args:
        ctx: Runtime context
        session_id: Session identifier
        
    Returns:
        AuthOutput with logout status
    """
    try:
        injector = get_injector()
        
        # Get session to find user
        try:
            session_result = await injector.run('session', {
                "operation": "get",
                "session_id": session_id
            })
            
            # session returns typed SessionOutput
            if not session_result.success:
                raise ValueError(f"Invalid or expired session: {session_id}")
            
            user_id = session_result.data["user_id"]
        except (KeyError, TypeError):
            # Session.get now throws KeyError for missing sessions
            raise ValueError(f"Invalid or expired session: {session_id}")
        
        # Delete session
        delete_result = await injector.run('session', {
            "operation": "delete",
            "session_id": session_id
        })
        
        # session returns typed SessionOutput
        
        return AuthOutput(
            success=True,
            operation="logout",
            message=f"User {user_id} logged out successfully",
            data={
                "user_id": user_id,
                "session_deleted": True
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to logout session '{session_id}': {e}") from e


async def auth_verify_password(ctx: RunContext[Any], username: str, password: str) -> AuthOutput:
    """
    Verify a user's password without creating a session.
    
    Args:
        ctx: Runtime context
        username: Username
        password: Password to verify
        
    Returns:
        AuthOutput with verification result
    """
    try:
        # Get user
        if username not in _users:
            # Return success=False for discovery operation (user not found)
            return AuthOutput(
                success=False,
                operation="verify_password",
                message="User not found",
                data={"valid": False}
            )
        
        user_data = _users[username]
        
        # Verify password
        injector = get_injector()
        verify_result = await injector.run('crypto', {
            "operation": "verify_hash",
            "algorithm": "bcrypt",
            "data": password,
            "key": user_data["password_hash"],
            "salt": user_data["salt"]
        })
        
        # crypto returns typed CryptoOutput
        return AuthOutput(
            success=True,
            operation="verify_password",
            message="Password verification completed",
            data={
                "valid": verify_result.success and verify_result.data["valid"]
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to verify password for user '{username}': {e}") from e


async def auth_change_password(ctx: RunContext[Any], username: str, 
                              password: str, new_password: str) -> AuthOutput:
    """
    Change a user's password.
    
    Args:
        ctx: Runtime context
        username: Username
        password: Current password
        new_password: New password
        
    Returns:
        AuthOutput with change status
    """
    try:
        # Verify current password
        verify_result = await auth_verify_password(ctx, username, password)
        if not verify_result.data["valid"]:
            raise ValueError("Current password is incorrect")
        
        user_data = _users[username]
        injector = get_injector()
        
        # Generate new salt
        salt_result = await injector.run('crypto', {
            "operation": "generate_salt"
        })
        
        # crypto returns typed CryptoOutput
        new_salt = salt_result.data["salt"]
        
        # Hash new password
        hash_result = await injector.run('crypto', {
            "operation": "hash",
            "algorithm": "bcrypt",
            "data": new_password,
            "salt": new_salt,
            "iterations": 10000
        })
        
        # crypto returns typed CryptoOutput
        # Update user
        user_data["password_hash"] = hash_result.data["hash"]
        user_data["salt"] = new_salt
        user_data["password_changed_at"] = datetime.now(timezone.utc).isoformat()
        
        _users[username] = user_data
        
        # Persist changes
        await injector.run('storage_kv', {
            "operation": "set",
            "key": f"user:{username}",
            "value": user_data,
            "namespace": "auth"
        })
        
        # Invalidate all user sessions
        await injector.run('session', {
            "operation": "invalidate_all",
            "user_id": username
        })
        
        return AuthOutput(
            success=True,
            operation="change_password",
            message="Password changed successfully",
            data={
                "user_id": username,
                "sessions_invalidated": True
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to change password for user '{username}': {e}") from e


async def auth_reset_password(ctx: RunContext[Any], username: str, new_password: str) -> AuthOutput:
    """
    Reset a user's password (admin operation).
    
    Args:
        ctx: Runtime context
        username: Username
        new_password: New password
        
    Returns:
        AuthOutput with reset status
    """
    try:
        # Get user
        if username not in _users:
            raise KeyError(f"User '{username}' not found")
        
        user_data = _users[username]
        injector = get_injector()
        
        # Generate new salt
        salt_result = await injector.run('crypto', {
            "operation": "generate_salt"
        })
        
        # crypto returns typed CryptoOutput
        new_salt = salt_result.data["salt"]
        
        # Hash new password
        hash_result = await injector.run('crypto', {
            "operation": "hash",
            "algorithm": "bcrypt",
            "data": new_password,
            "salt": new_salt,
            "iterations": 10000
        })
        
        # crypto returns typed CryptoOutput
        # Update user
        user_data["password_hash"] = hash_result.data["hash"]
        user_data["salt"] = new_salt
        user_data["password_reset_at"] = datetime.now(timezone.utc).isoformat()
        
        _users[username] = user_data
        
        # Persist changes
        await injector.run('storage_kv', {
            "operation": "set",
            "key": f"user:{username}",
            "value": user_data,
            "namespace": "auth"
        })
        
        # Invalidate all user sessions
        await injector.run('session', {
            "operation": "invalidate_all",
            "user_id": username
        })
        
        return AuthOutput(
            success=True,
            operation="reset_password",
            message="Password reset successfully",
            data={
                "user_id": username,
                "sessions_invalidated": True
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to reset password for user '{username}': {e}") from e


async def auth_generate_token(ctx: RunContext[Any], user_id: str, 
                             metadata: Optional[Dict[str, Any]]) -> AuthOutput:
    """
    Generate an authentication token.
    
    Args:
        ctx: Runtime context
        user_id: User identifier
        metadata: Token metadata
        
    Returns:
        AuthOutput with generated token
    """
    try:
        # Get user
        if user_id not in _users:
            raise KeyError(f"User '{user_id}' not found")
        
        user_data = _users[user_id]
        
        # Generate JWT token
        injector = get_injector()
        jwt_result = await injector.run('crypto', {
            "operation": "generate_jwt",
            "payload": {
                "user_id": user_id,
                "username": user_data["username"],
                "roles": user_data["roles"],
                "metadata": metadata or {}
            },
            "secret": "auth_secret_key",  # In production, use secure key management
            "expires_in": 3600  # 1 hour
        })
        
        # crypto returns typed CryptoOutput
        # Record token generation metric
        try:
            await injector.run('metrics', {
                "operation": "increment",
                "name": "agentool.auth.tokens.generated",
                "value": 1
            })
        except:
            pass  # Ignore metrics errors
        
        return AuthOutput(
            success=True,
            operation="generate_token",
            message="Token generated successfully",
            data={
                "token": jwt_result.data["token"],
                "expires_in": jwt_result.data["expires_in"],
                "user_id": user_id
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate token for user '{user_id}': {e}") from e


async def auth_verify_token(ctx: RunContext[Any], token: str) -> AuthOutput:
    """
    Verify an authentication token.
    
    Args:
        ctx: Runtime context
        token: Token to verify
        
    Returns:
        AuthOutput with verification result
    """
    try:
        # Verify JWT token
        injector = get_injector()
        try:
            verify_result = await injector.run('crypto', {
                "operation": "verify_jwt",
                "data": token,
                "secret": "auth_secret_key"
            })
            
            # crypto returns typed CryptoOutput
            if not verify_result.success or not verify_result.data["valid"]:
                # Record invalid token metric
                try:
                    await injector.run('metrics', {
                        "operation": "increment",
                        "name": "agentool.auth.tokens.verified.invalid",
                        "value": 1
                    })
                except:
                    pass  # Ignore metrics errors
                
                return AuthOutput(
                    success=False,
                    operation="verify_token",
                    message="Token verification failed",
                    data={
                        "valid": False,
                        "error": verify_result.data.get("error", "Invalid token")
                    }
                )
            
            payload = verify_result.data["payload"]
            
            # Record valid token metric
            try:
                await injector.run('metrics', {
                    "operation": "increment",
                    "name": "agentool.auth.tokens.verified.valid",
                    "value": 1
                })
            except:
                pass  # Ignore metrics errors
            
            return AuthOutput(
                success=True,
                operation="verify_token",
                message="Token is valid",
                data={
                    "valid": True,
                    "user_id": payload["user_id"],
                    "username": payload["username"],
                    "roles": payload["roles"],
                    "metadata": payload.get("metadata", {})
                }
            )
        except (KeyError, TypeError):
            return AuthOutput(
                success=False,
                operation="verify_token",
                message="Token verification failed",
                data={
                    "valid": False,
                    "error": "Invalid token format"
                }
            )
        
    except Exception as e:
        raise RuntimeError(f"Failed to verify token: {e}") from e


async def auth_check_permission(ctx: RunContext[Any], user_id: str, permission: str) -> AuthOutput:
    """
    Check if a user has a specific permission.
    
    Args:
        ctx: Runtime context
        user_id: User identifier
        permission: Permission to check
        
    Returns:
        AuthOutput with permission check result
    """
    try:
        # Get user
        if user_id not in _users:
            # Return success=False for discovery operation (user not found)
            return AuthOutput(
                success=False,
                operation="check_permission",
                message="User not found",
                data={"has_permission": False}
            )
        
        user_data = _users[user_id]
        user_roles = user_data.get("roles", [])
        
        # Check if any role has the permission
        has_permission = False
        for role in user_roles:
            if role in _role_permissions:
                if permission in _role_permissions[role]:
                    has_permission = True
                    break
        
        # Record permission check metrics
        try:
            await injector.run('metrics', {
                "operation": "increment",
                "name": "agentool.auth.permissions.checks",
                "value": 1
            })
            if has_permission:
                await injector.run('metrics', {
                    "operation": "increment",
                    "name": "agentool.auth.permissions.granted",
                    "value": 1
                })
            else:
                await injector.run('metrics', {
                    "operation": "increment", 
                    "name": "agentool.auth.permissions.denied",
                    "value": 1
                })
        except:
            pass  # Ignore metrics errors
        
        return AuthOutput(
            success=True,
            operation="check_permission",
            message="Permission check completed",
            data={
                "has_permission": has_permission,
                "user_id": user_id,
                "permission": permission,
                "user_roles": user_roles
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to check permission '{permission}' for user '{user_id}': {e}") from e


async def auth_assign_role(ctx: RunContext[Any], user_id: str, role: str) -> AuthOutput:
    """
    Assign a role to a user.
    
    Args:
        ctx: Runtime context
        user_id: User identifier
        role: Role to assign
        
    Returns:
        AuthOutput with assignment status
    """
    try:
        # Get user
        if user_id not in _users:
            raise KeyError(f"User '{user_id}' not found")
        
        # Check if role exists
        if role not in _role_permissions:
            raise ValueError(f"Role '{role}' does not exist")
        
        user_data = _users[user_id]
        
        # Add role if not already assigned
        if role not in user_data.get("roles", []):
            user_data.setdefault("roles", []).append(role)
            
            # Persist changes
            injector = get_injector()
            await injector.run('storage_kv', {
                "operation": "set",
                "key": f"user:{user_id}",
                "value": user_data,
                "namespace": "auth"
            })
        
        return AuthOutput(
            success=True,
            operation="assign_role",
            message=f"Role '{role}' assigned to user {user_id}",
            data={
                "user_id": user_id,
                "role": role,
                "current_roles": user_data["roles"]
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to assign role '{role}' to user '{user_id}': {e}") from e


async def auth_revoke_role(ctx: RunContext[Any], user_id: str, role: str) -> AuthOutput:
    """
    Revoke a role from a user.
    
    Args:
        ctx: Runtime context
        user_id: User identifier
        role: Role to revoke
        
    Returns:
        AuthOutput with revocation status
    """
    try:
        # Get user
        if user_id not in _users:
            raise KeyError(f"User '{user_id}' not found")
        
        user_data = _users[user_id]
        
        # Remove role if assigned
        if role in user_data.get("roles", []):
            user_data["roles"].remove(role)
            
            # Persist changes
            injector = get_injector()
            await injector.run('storage_kv', {
                "operation": "set",
                "key": f"user:{user_id}",
                "value": user_data,
                "namespace": "auth"
            })
        
        return AuthOutput(
            success=True,
            operation="revoke_role",
            message=f"Role '{role}' revoked from user {user_id}",
            data={
                "user_id": user_id,
                "role": role,
                "current_roles": user_data["roles"]
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to revoke role '{role}' from user '{user_id}': {e}") from e


async def auth_get_user(ctx: RunContext[Any], user_id: str) -> AuthOutput:
    """
    Get user information.
    
    Args:
        ctx: Runtime context
        user_id: User identifier
        
    Returns:
        AuthOutput with user data
    """
    try:
        # Get user
        if user_id not in _users:
            # Try loading from storage
            injector = get_injector()
            try:
                result = await injector.run('storage_kv', {
                    "operation": "get",
                    "key": f"user:{user_id}",
                    "namespace": "auth"
                })
                
                # storage_kv returns typed StorageKvOutput
                if result.success and result.data and result.data["exists"]:
                    _users[user_id] = result.data["value"]
                else:
                    raise KeyError(f"User '{user_id}' does not exist")
            except (KeyError, TypeError):
                raise KeyError(f"User '{user_id}' does not exist")
        
        user_data = _users[user_id].copy()
        
        # Remove sensitive data
        user_data.pop("password_hash", None)
        user_data.pop("salt", None)
        
        return AuthOutput(
            success=True,
            operation="get_user",
            message="User retrieved successfully",
            data=user_data
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve user '{user_id}': {e}") from e


async def auth_update_user(ctx: RunContext[Any], user_id: str, 
                          metadata: Optional[Dict[str, Any]]) -> AuthOutput:
    """
    Update user information.
    
    Args:
        ctx: Runtime context
        user_id: User identifier
        metadata: Updated metadata
        
    Returns:
        AuthOutput with update status
    """
    try:
        # Get user
        if user_id not in _users:
            raise KeyError(f"User '{user_id}' not found")
        
        user_data = _users[user_id]
        
        # Update metadata
        if metadata:
            user_data["metadata"].update(metadata)
        
        user_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        # Persist changes
        injector = get_injector()
        await injector.run('storage_kv', {
            "operation": "set",
            "key": f"user:{user_id}",
            "value": user_data,
            "namespace": "auth"
        })
        
        return AuthOutput(
            success=True,
            operation="update_user",
            message="User updated successfully",
            data={
                "user_id": user_id,
                "updated_fields": list(metadata.keys()) if metadata else []
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to update user '{user_id}': {e}") from e


async def auth_delete_user(ctx: RunContext[Any], user_id: str) -> AuthOutput:
    """
    Delete a user account.
    
    Args:
        ctx: Runtime context
        user_id: User identifier
        
    Returns:
        AuthOutput with deletion status
    """
    try:
        # Check if user exists
        if user_id not in _users:
            raise KeyError(f"User '{user_id}' not found")
        
        # Invalidate all sessions
        injector = get_injector()
        await injector.run('session', {
            "operation": "invalidate_all",
            "user_id": user_id
        })
        
        # Remove from memory
        del _users[user_id]
        
        # Remove from storage
        await injector.run('storage_kv', {
            "operation": "delete",
            "key": f"user:{user_id}",
            "namespace": "auth"
        })
        
        return AuthOutput(
            success=True,
            operation="delete_user",
            message="User deleted successfully",
            data={
                "user_id": user_id,
                "sessions_invalidated": True
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to delete user '{user_id}': {e}") from e


def create_auth_agent():
    """
    Create and return the auth AgenTool.
    
    Returns:
        Agent configured for authentication operations
    """
    auth_routing = RoutingConfig(
        operation_field='operation',
        operation_map={
            'register': ('auth_register', lambda x: {
                'username': x.username, 'password': x.password,
                'email': x.email, 'metadata': x.metadata
            }),
            'login': ('auth_login', lambda x: {
                'username': x.username, 'password': x.password
            }),
            'logout': ('auth_logout', lambda x: {
                'session_id': x.session_id
            }),
            'verify_password': ('auth_verify_password', lambda x: {
                'username': x.username, 'password': x.password
            }),
            'change_password': ('auth_change_password', lambda x: {
                'username': x.username, 'password': x.password,
                'new_password': x.new_password
            }),
            'reset_password': ('auth_reset_password', lambda x: {
                'username': x.username, 'new_password': x.new_password
            }),
            'generate_token': ('auth_generate_token', lambda x: {
                'user_id': x.user_id, 'metadata': x.metadata
            }),
            'verify_token': ('auth_verify_token', lambda x: {
                'token': x.token
            }),
            'check_permission': ('auth_check_permission', lambda x: {
                'user_id': x.user_id, 'permission': x.permission
            }),
            'assign_role': ('auth_assign_role', lambda x: {
                'user_id': x.user_id, 'role': x.role
            }),
            'revoke_role': ('auth_revoke_role', lambda x: {
                'user_id': x.user_id, 'role': x.role
            }),
            'get_user': ('auth_get_user', lambda x: {
                'user_id': x.user_id
            }),
            'update_user': ('auth_update_user', lambda x: {
                'user_id': x.user_id, 'metadata': x.metadata
            }),
            'delete_user': ('auth_delete_user', lambda x: {
                'user_id': x.user_id
            }),
        }
    )
    
    return create_agentool(
        name='auth',
        input_schema=AuthInput,
        routing_config=auth_routing,
        tools=[
            auth_register, auth_login, auth_logout, auth_verify_password,
            auth_change_password, auth_reset_password, auth_generate_token,
            auth_verify_token, auth_check_permission, auth_assign_role,
            auth_revoke_role, auth_get_user, auth_update_user, auth_delete_user
        ],
        output_type=AuthOutput,
        use_typed_output=True,  # Enable typed output for auth (Tier 3 - depends on crypto, storage_kv, session)
        system_prompt="Handle authentication and authorization operations securely.",
        description="Comprehensive authentication and authorization toolkit",
        version="1.0.0",
        tags=["auth", "security", "authentication", "authorization"],
        dependencies=["crypto", "storage_kv"],  # Uses crypto for password hashing, storage_kv for user storage
        examples=[
            {
                "description": "Register a new user",
                "input": {
                    "operation": "register",
                    "username": "john_doe",
                    "password": "secure_password",
                    "email": "john@example.com"
                },
                "output": {
                    "operation": "register",
                    "message": "User john_doe registered successfully"
                }
            },
            {
                "description": "Login user",
                "input": {
                    "operation": "login",
                    "username": "john_doe",
                    "password": "secure_password"
                },
                "output": {
                    "operation": "login",
                    "message": "User john_doe logged in successfully"
                }
            },
            {
                "description": "Check permission",
                "input": {
                    "operation": "check_permission",
                    "user_id": "john_doe",
                    "permission": "write"
                },
                "output": {
                    "operation": "check_permission",
                    "message": "Permission check completed"
                }
            }
        ]
    )


# Create the agent instance
agent = create_auth_agent()