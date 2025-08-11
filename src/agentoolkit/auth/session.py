"""
Session AgenTool - Provides session management capabilities for authentication and state tracking.

This toolkit provides comprehensive session management functionality including:
- Session creation and destruction
- Session validation and renewal
- User session tracking
- Session metadata management
- Activity tracking

Example Usage:
    >>> from agentoolkit.auth.session import create_session_agent
    >>> from agentool.core.injector import get_injector
    >>> 
    >>> # Create and register the agent
    >>> agent = create_session_agent()
    >>> 
    >>> # Use through injector
    >>> injector = get_injector()
    >>> result = await injector.run('session', {
    ...     "operation": "create",
    ...     "user_id": "user123",
    ...     "metadata": {"ip": "192.168.1.1", "user_agent": "Mozilla/5.0"}
    ... })
"""

import secrets
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import RunContext

from agentool.base import BaseOperationInput
from agentool import create_agentool
from agentool.core.registry import RoutingConfig
from agentool.core.injector import get_injector


# Global session storage (in production, use Redis or database)
_sessions: Dict[str, Dict[str, Any]] = {}


class SessionInput(BaseOperationInput):
    """Input schema for session operations."""
    operation: Literal[
        'create', 'get', 'update', 'delete', 'validate',
        'renew', 'list', 'invalidate_all', 'get_active'
    ] = Field(description="The session operation to perform")
    
    # Common fields
    session_id: Optional[str] = Field(None, description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    
    # Session data
    metadata: Optional[Dict[str, Any]] = Field(None, description="Session metadata")
    ttl: Optional[int] = Field(None, description="Time to live in seconds")
    
    # Update fields
    data: Optional[Dict[str, Any]] = Field(None, description="Data to store/update")
    
    @field_validator('session_id')
    def validate_session_id(cls, v, info):
        """Validate that session_id is provided for operations that require it."""
        operation = info.data.get('operation')
        if operation in ['get', 'update', 'delete', 'validate', 'renew'] and not v:
            raise ValueError(f"session_id is required for {operation} operation")
        return v
    
    @field_validator('user_id')
    def validate_user_id(cls, v, info):
        """Validate that user_id is provided for operations that require it."""
        operation = info.data.get('operation')
        if operation in ['create', 'invalidate_all'] and not v:
            raise ValueError(f"user_id is required for {operation} operation")
        return v
    
    @field_validator('ttl')
    def validate_ttl(cls, v, info):
        """Validate that ttl is positive when provided."""
        if v is not None and v <= 0:
            raise ValueError("ttl must be a positive integer when provided")
        return v


class SessionOutput(BaseModel):
    """Structured output for session operations."""
    success: bool = Field(default=True, description="Whether the operation succeeded")
    operation: str = Field(description="The operation that was performed")
    message: str = Field(description="Human-readable result message")
    data: Optional[Dict[str, Any]] = Field(None, description="Operation-specific data")


async def session_create(ctx: RunContext[Any], user_id: str, metadata: Optional[Dict[str, Any]], 
                        ttl: Optional[int]) -> SessionOutput:
    """
    Create a new session.
    
    Args:
        ctx: Runtime context
        user_id: User identifier
        metadata: Session metadata
        ttl: Time to live in seconds
        
    Returns:
        SessionOutput with created session details
        
    Raises:
        ValueError: For invalid user_id or ttl values
        RuntimeError: For storage or session creation failures
    """
    if not user_id or not isinstance(user_id, str):
        raise ValueError("user_id must be a non-empty string")
    
    if ttl is not None and (not isinstance(ttl, int) or ttl <= 0):
        raise ValueError("ttl must be a positive integer")
    
    try:
        # Generate secure session ID
        session_id = secrets.token_urlsafe(32)
        
        # Default TTL is 24 hours
        ttl = ttl or 86400
        
        # Create session data
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=ttl)).isoformat(),
            "last_activity": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
            "data": {},
            "active": True
        }
        
        # Store session
        _sessions[session_id] = session_data
        
        # Also use storage_kv for persistence
        injector = get_injector()
        await injector.run('storage_kv', {
            "operation": "set",
            "key": f"session:{session_id}",
            "value": session_data,
            "namespace": "sessions",
            "ttl": ttl
        })
        
        # Record business metrics
        try:
            await injector.run('metrics', {
                "operation": "increment",
                "name": "agentool.session.sessions.created",
                "value": 1
            })
            await injector.run('metrics', {
                "operation": "increment",
                "name": "agentool.session.sessions.active",
                "value": 1
            })
        except Exception as metrics_error:
            # Log metrics error but continue
            await injector.run('logging', {
                "operation": "log",
                "level": "ERROR",
                "message": f"Failed to record session creation metrics: {str(metrics_error)}",
                "data": {
                    "operation": "session_create",
                    "user_id": user_id,
                    "error": str(metrics_error)
                },
                "logger_name": "session"
            })
        
        return SessionOutput(
            operation="create",
            message=f"Session created successfully for user {user_id}",
            data={
                "session_id": session_id,
                "user_id": user_id,
                "expires_at": session_data["expires_at"],
                "ttl": ttl
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to create session for user {user_id}: {e}") from e


async def session_get(ctx: RunContext[Any], session_id: str) -> SessionOutput:
    """
    Get session details.
    
    Args:
        ctx: Runtime context
        session_id: Session identifier
        
    Returns:
        SessionOutput with session details or success=False if not found/expired
        
    Raises:
        ValueError: For invalid session_id
        RuntimeError: For storage errors (not for session not found/expired)
    """
    if not session_id or not isinstance(session_id, str):
        raise ValueError("session_id must be a non-empty string")
    
    try:
        # Check memory first
        if session_id in _sessions:
            session_data = _sessions[session_id]
        else:
            # Try to load from storage
            injector = get_injector()
            result = await injector.run('storage_kv', {
                "operation": "get",
                "key": f"session:{session_id}",
                "namespace": "sessions"
            })
            
            # storage_kv now returns typed output with success field
            if result.success and result.data["value"]:
                session_data = result.data["value"]
                # Cache in memory
                _sessions[session_id] = session_data
            else:
                # Discovery operation - session not found
                return SessionOutput(
                    success=False,
                    operation="get",
                    message=f"Session {session_id} does not exist",
                    data={}  # Empty data when not found
                )
        
        # Check if expired
        expires_at = datetime.fromisoformat(session_data["expires_at"])
        if datetime.now(timezone.utc) > expires_at:
            # Record expired session metric
            injector = get_injector()
            try:
                await injector.run('metrics', {
                    "operation": "increment",
                    "name": "agentool.session.sessions.expired",
                    "value": 1
                })
            except Exception as metrics_error:
                # Log metrics error but continue
                await injector.run('logging', {
                    "operation": "log",
                    "level": "ERROR",
                    "message": f"Failed to record session expiration metrics: {str(metrics_error)}",
                    "data": {
                        "operation": "session_get",
                        "session_id": session_id,
                        "error": str(metrics_error)
                    },
                    "logger_name": "session"
                })
            
            # Session expired, clean it up
            await session_delete(ctx, session_id)
            
            # Discovery operation - session expired
            return SessionOutput(
                success=False,
                operation="get",
                message=f"Session {session_id} has expired",
                data={}  # Empty data when expired
            )
        
        return SessionOutput(
            success=True,
            operation="get",
            message="Session retrieved successfully",
            data=session_data
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve session {session_id}: {e}") from e


async def session_update(ctx: RunContext[Any], session_id: str, 
                        data: Optional[Dict[str, Any]]) -> SessionOutput:
    """
    Update session data.
    
    Args:
        ctx: Runtime context
        session_id: Session identifier
        data: Data to update
        
    Returns:
        SessionOutput with update status
        
    Raises:
        ValueError: For invalid session_id
        KeyError: When session is not found
        RuntimeError: For storage errors or expired sessions
    """
    # Get existing session
    get_result = await session_get(ctx, session_id)
    
    # Check if session was found
    if not get_result.success:
        # Session not found or expired - this is NOT a discovery operation for update
        # Update requires the session to exist, so we raise an exception
        raise KeyError(f"Session {session_id} does not exist or has expired")
    
    session_data = get_result.data
    
    try:
        # Update data
        if data:
            session_data["data"].update(data)
        
        # Update last activity
        session_data["last_activity"] = datetime.now(timezone.utc).isoformat()
        
        # Store updated session
        _sessions[session_id] = session_data
        
        # Persist to storage
        injector = get_injector()
        ttl = int((datetime.fromisoformat(session_data["expires_at"]) - 
                  datetime.now(timezone.utc)).total_seconds())
        
        await injector.run('storage_kv', {
            "operation": "set",
            "key": f"session:{session_id}",
            "value": session_data,
            "namespace": "sessions",
            "ttl": max(ttl, 1)  # At least 1 second
        })
        
        return SessionOutput(
            operation="update",
            message="Session updated successfully",
            data={
                "session_id": session_id,
                "updated_fields": list(data.keys()) if data else []
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to update session {session_id}: {e}") from e


async def session_delete(ctx: RunContext[Any], session_id: str) -> SessionOutput:
    """
    Delete a session.
    
    Args:
        ctx: Runtime context
        session_id: Session identifier
        
    Returns:
        SessionOutput with deletion status
        
    Raises:
        ValueError: For invalid session_id
        RuntimeError: For storage deletion failures
    """
    if not session_id or not isinstance(session_id, str):
        raise ValueError("session_id must be a non-empty string")
    
    try:
        # Remove from memory
        existed = session_id in _sessions
        if existed:
            del _sessions[session_id]
        
        # Remove from storage
        injector = get_injector()
        await injector.run('storage_kv', {
            "operation": "delete",
            "key": f"session:{session_id}",
            "namespace": "sessions"
        })
        
        # Record business metrics
        if existed:
            try:
                await injector.run('metrics', {
                    "operation": "increment",
                    "name": "agentool.session.sessions.deleted",
                    "value": 1
                })
                await injector.run('metrics', {
                    "operation": "decrement",
                    "name": "agentool.session.sessions.active",
                    "value": 1
                })
            except Exception as metrics_error:
                # Log metrics error but continue
                await injector.run('logging', {
                    "operation": "log",
                    "level": "ERROR",
                    "message": f"Failed to record session deletion metrics: {str(metrics_error)}",
                    "data": {
                        "operation": "session_delete",
                        "session_id": session_id,
                        "error": str(metrics_error)
                    },
                    "logger_name": "session"
                })
        
        return SessionOutput(
            operation="delete",
            message="Session deleted successfully",
            data={
                "session_id": session_id,
                "existed": existed
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to delete session {session_id}: {e}") from e


async def session_validate(ctx: RunContext[Any], session_id: str) -> SessionOutput:
    """
    Validate a session.
    
    Args:
        ctx: Runtime context
        session_id: Session identifier
        
    Returns:
        SessionOutput with validation result (success=False for invalid sessions)
        
    Raises:
        ValueError: For invalid session_id
        RuntimeError: For validation errors
    """
    try:
        # Get session - this now returns success=False if not found or expired
        get_result = await session_get(ctx, session_id)
        
        # Check if session was found and not expired
        if not get_result.success:
            # Record failed validation
            try:
                injector = get_injector()
                await injector.run('metrics', {
                    "operation": "increment",
                    "name": "agentool.session.validations.failure",
                    "value": 1
                })
            except Exception as metrics_error:
                # Log metrics error but continue
                await injector.run('logging', {
                    "operation": "log",
                    "level": "ERROR",
                    "message": f"Failed to record validation failure metrics: {str(metrics_error)}",
                    "data": {
                        "operation": "session_validate",
                        "session_id": session_id,
                        "error": str(metrics_error)
                    },
                    "logger_name": "session"
                })
            
            # Session not found or expired - return validation failed result
            return SessionOutput(
                success=False,  # Discovery operation - session invalid
                operation="validate",
                message="Session validation failed",
                data={
                    "valid": False,
                    "reason": get_result.message
                }
            )
        
        session_data = get_result.data
        
        # Check if active
        if not session_data.get("active", True):
            return SessionOutput(
                success=False,  # Discovery operation - session inactive
                operation="validate",
                message="Session is inactive",
                data={
                    "valid": False,
                    "reason": "Session has been deactivated"
                }
            )
        
        # Update last activity
        await session_update(ctx, session_id, None)
        
        # Record successful validation
        try:
            injector = get_injector()
            await injector.run('metrics', {
                "operation": "increment",
                "name": "agentool.session.validations.success",
                "value": 1
            })
        except Exception as metrics_error:
            # Log metrics error but continue
            await injector.run('logging', {
                "operation": "log",
                "level": "ERROR",
                "message": f"Failed to record validation success metrics: {str(metrics_error)}",
                "data": {
                    "operation": "session_validate",
                    "session_id": session_id,
                    "error": str(metrics_error)
                },
                "logger_name": "session"
            })
        
        return SessionOutput(
            success=True,  # Session is valid
            operation="validate",
            message="Session is valid",
            data={
                "valid": True,
                "user_id": session_data["user_id"],
                "expires_at": session_data["expires_at"]
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to validate session {session_id}: {e}") from e


async def session_renew(ctx: RunContext[Any], session_id: str, ttl: Optional[int]) -> SessionOutput:
    """
    Renew a session by extending its expiration.
    
    Args:
        ctx: Runtime context
        session_id: Session identifier
        ttl: New time to live in seconds
        
    Returns:
        SessionOutput with renewal status
        
    Raises:
        ValueError: For invalid session_id or ttl values
        KeyError: When session is not found
        RuntimeError: For storage errors or expired sessions
    """
    if ttl is not None and (not isinstance(ttl, int) or ttl <= 0):
        raise ValueError("ttl must be a positive integer")
    
    # Get existing session
    get_result = await session_get(ctx, session_id)
    
    # Check if session was found
    if not get_result.success:
        # Session not found or expired - this is NOT a discovery operation for renew
        # Renew requires the session to exist, so we raise an exception
        raise KeyError(f"Session {session_id} does not exist or has expired")
    
    session_data = get_result.data
    
    try:
        # Default TTL is 24 hours
        ttl = ttl or 86400
        
        # Update expiration
        session_data["expires_at"] = (datetime.now(timezone.utc) + timedelta(seconds=ttl)).isoformat()
        session_data["last_activity"] = datetime.now(timezone.utc).isoformat()
        
        # Store updated session
        _sessions[session_id] = session_data
        
        # Persist to storage
        injector = get_injector()
        await injector.run('storage_kv', {
            "operation": "set",
            "key": f"session:{session_id}",
            "value": session_data,
            "namespace": "sessions",
            "ttl": ttl
        })
        
        return SessionOutput(
            operation="renew",
            message="Session renewed successfully",
            data={
                "session_id": session_id,
                "new_expires_at": session_data["expires_at"],
                "ttl": ttl
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to renew session {session_id}: {e}") from e


async def session_list(ctx: RunContext[Any], user_id: Optional[str]) -> SessionOutput:
    """
    List sessions, optionally filtered by user.
    
    Args:
        ctx: Runtime context
        user_id: Optional user ID filter
        
    Returns:
        SessionOutput with session list
        
    Raises:
        RuntimeError: For storage access failures
    """
    try:
        # Get all sessions from storage
        injector = get_injector()
        list_result = await injector.run('storage_kv', {
            "operation": "keys",
            "namespace": "sessions"
        })
        
        # storage_kv now returns typed output
        sessions = []
        
        # Load each session
        for key in list_result.data["keys"]:
            if key.startswith("session:"):
                session_id = key.replace("session:", "")
                try:
                    get_result = await session_get(ctx, session_id)
                    
                    # Skip if session not found or expired
                    if not get_result.success:
                        continue
                    
                    session_data = get_result.data
                    if not user_id or session_data["user_id"] == user_id:
                        sessions.append({
                            "session_id": session_data["session_id"],
                            "user_id": session_data["user_id"],
                            "created_at": session_data["created_at"],
                            "expires_at": session_data["expires_at"],
                            "last_activity": session_data["last_activity"],
                            "active": session_data.get("active", True)
                        })
                except Exception as e:
                    # Log the error but continue with other sessions
                    await injector.run('logging', {
                        "operation": "log",
                        "level": "WARN",
                        "message": f"Failed to load session {session_id} during list operation: {str(e)}",
                        "data": {
                            "operation": "session_list",
                            "session_id": session_id,
                            "error": str(e)
                        },
                        "logger_name": "session"
                    })
                    continue
        
        return SessionOutput(
            operation="list",
            message=f"Found {len(sessions)} session(s)",
            data={
                "count": len(sessions),
                "sessions": sessions
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to list sessions: {e}") from e


async def session_invalidate_all(ctx: RunContext[Any], user_id: str) -> SessionOutput:
    """
    Invalidate all sessions for a user.
    
    Args:
        ctx: Runtime context
        user_id: User identifier
        
    Returns:
        SessionOutput with invalidation status
        
    Raises:
        ValueError: For invalid user_id
        RuntimeError: For session listing or deletion failures
    """
    if not user_id or not isinstance(user_id, str):
        raise ValueError("user_id must be a non-empty string")
    
    try:
        # List user sessions
        list_result = await session_list(ctx, user_id)
        sessions = list_result.data["sessions"]
        invalidated = 0
        
        # Delete each session
        for session in sessions:
            try:
                await session_delete(ctx, session["session_id"])
                invalidated += 1
            except RuntimeError:
                # Continue with other sessions even if one fails
                continue
        
        return SessionOutput(
            operation="invalidate_all",
            message=f"Invalidated {invalidated} session(s) for user {user_id}",
            data={
                "user_id": user_id,
                "invalidated_count": invalidated
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to invalidate sessions for user {user_id}: {e}") from e


async def session_get_active(ctx: RunContext[Any]) -> SessionOutput:
    """
    Get all active sessions.
    
    Args:
        ctx: Runtime context
        
    Returns:
        SessionOutput with active sessions
        
    Raises:
        RuntimeError: For session listing failures
    """
    try:
        # List all sessions
        list_result = await session_list(ctx, None)
        all_sessions = list_result.data["sessions"]
        
        # Filter active sessions
        active_sessions = [
            session for session in all_sessions
            if session.get("active", True) and
            datetime.fromisoformat(session["expires_at"]) > datetime.now(timezone.utc)
        ]
        
        # Group by user
        users_with_sessions = {}
        for session in active_sessions:
            user_id = session["user_id"]
            if user_id not in users_with_sessions:
                users_with_sessions[user_id] = 0
            users_with_sessions[user_id] += 1
        
        return SessionOutput(
            operation="get_active",
            message=f"Found {len(active_sessions)} active session(s)",
            data={
                "total_active": len(active_sessions),
                "unique_users": len(users_with_sessions),
                "users": users_with_sessions,
                "sessions": active_sessions
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to get active sessions: {e}") from e


def create_session_agent():
    """
    Create and return the session AgenTool.
    
    Returns:
        Agent configured for session management operations
    """
    session_routing = RoutingConfig(
        operation_field='operation',
        operation_map={
            'create': ('session_create', lambda x: {
                'user_id': x.user_id, 'metadata': x.metadata, 'ttl': x.ttl
            }),
            'get': ('session_get', lambda x: {
                'session_id': x.session_id
            }),
            'update': ('session_update', lambda x: {
                'session_id': x.session_id, 'data': x.data
            }),
            'delete': ('session_delete', lambda x: {
                'session_id': x.session_id
            }),
            'validate': ('session_validate', lambda x: {
                'session_id': x.session_id
            }),
            'renew': ('session_renew', lambda x: {
                'session_id': x.session_id, 'ttl': x.ttl
            }),
            'list': ('session_list', lambda x: {
                'user_id': x.user_id
            }),
            'invalidate_all': ('session_invalidate_all', lambda x: {
                'user_id': x.user_id
            }),
            'get_active': ('session_get_active', lambda x: {}),
        }
    )
    
    return create_agentool(
        name='session',
        input_schema=SessionInput,
        routing_config=session_routing,
        tools=[
            session_create, session_get, session_update, session_delete,
            session_validate, session_renew, session_list,
            session_invalidate_all, session_get_active
        ],
        output_type=SessionOutput,
        system_prompt="Manage user sessions efficiently and securely.",
        description="Comprehensive session management for authentication and state tracking",
        version="1.0.0",
        tags=["session", "auth", "security", "state"],
        dependencies=["storage_kv"],  # Uses storage_kv for session storage
        examples=[
            {
                "description": "Create a new session",
                "input": {
                    "operation": "create",
                    "user_id": "user123",
                    "metadata": {"ip": "192.168.1.1"},
                    "ttl": 3600
                },
                "output": {
                    "operation": "create",
                    "message": "Session created successfully for user user123"
                }
            },
            {
                "description": "Validate an existing session",
                "input": {
                    "operation": "validate",
                    "session_id": "abc123..."
                },
                "output": {
                    "operation": "validate",
                    "message": "Session is valid"
                }
            }
        ]
    )


# Create the agent instance
agent = create_session_agent()