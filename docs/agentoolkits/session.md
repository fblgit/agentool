# Session AgenToolkit

> **Note**: For guidelines on creating new AgenToolkits, see [CRAFTING_AGENTOOLS.md](../CRAFTING_AGENTOOLS.md). For testing patterns and examples, refer to [tests/agentoolkit/test_session.py](../../../tests/agentoolkit/test_session.py).

## Overview

The Session AgenToolkit provides comprehensive session management capabilities for authentication and state tracking. It handles session lifecycle management, validation, renewal, and user session tracking with automatic expiration and persistent storage support.

### Key Features
- Secure session creation with cryptographic tokens
- Session validation and expiration management
- Session renewal and TTL extension
- User session tracking and management
- Activity tracking with last activity updates
- Batch session invalidation for user logout
- Persistent storage via storage_kv with TTL support
- Active session monitoring

## Creation Method

```python
from agentoolkit.auth.session import create_session_agent

# Create the agent
agent = create_session_agent()
```

The creation function returns a fully configured AgenTool with name `'session'`.

## Input Schema

### SessionInput

The input schema is a Pydantic model with the following fields:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `operation` | `Literal['create', 'get', 'update', 'delete', 'validate', 'renew', 'list', 'invalidate_all', 'get_active']` | Yes | - | The session operation to perform |
| `session_id` | `Optional[str]` | No | None | Session identifier |
| `user_id` | `Optional[str]` | No | None | User identifier |
| `metadata` | `Optional[Dict[str, Any]]` | No | None | Session metadata |
| `ttl` | `Optional[int]` | No | None | Time to live in seconds |
| `data` | `Optional[Dict[str, Any]]` | No | None | Data to store/update |

## Operations Schema

The routing configuration maps operations to tool functions:

| Operation | Tool Function | Required Parameters | Description |
|-----------|--------------|-------------------|-------------|
| `create` | `session_create` | `user_id`, `metadata`, `ttl` | Create new session |
| `get` | `session_get` | `session_id` | Get session details |
| `update` | `session_update` | `session_id`, `data` | Update session data |
| `delete` | `session_delete` | `session_id` | Delete a session |
| `validate` | `session_validate` | `session_id` | Validate session status |
| `renew` | `session_renew` | `session_id`, `ttl` | Extend session expiration |
| `list` | `session_list` | `user_id` | List sessions (optional filter) |
| `invalidate_all` | `session_invalidate_all` | `user_id` | Delete all user sessions |
| `get_active` | `session_get_active` | - | Get all active sessions |

## Output Schema

### SessionOutput

All operations return a `SessionOutput` model with:

| Field | Type | Description |
|-------|------|-------------|
| `operation` | `str` | The operation that was performed |
| `message` | `str` | Human-readable result message |
| `data` | `Optional[Dict[str, Any]]` | Operation-specific data |

### Operation-Specific Data Fields

- **create**: `session_id`, `user_id`, `expires_at`, `ttl`
- **get**: Complete session data object including:
  - `session_id`, `user_id`, `created_at`, `expires_at`
  - `last_activity`, `metadata`, `data`, `active`
- **update**: `session_id`, `updated_fields`
- **delete**: `session_id`, `existed`
- **validate**: `valid`, `user_id`, `expires_at` (or `reason` if invalid)
- **renew**: `session_id`, `new_expires_at`, `ttl`
- **list**: `count`, `sessions` (array of session summaries)
- **invalidate_all**: `user_id`, `invalidated_count`
- **get_active**: `total_active`, `unique_users`, `users` (count per user), `sessions`

### Session Data Structure

Each session contains:
```json
{
  "session_id": "secure_token",
  "user_id": "user123",
  "created_at": "ISO timestamp",
  "expires_at": "ISO timestamp",
  "last_activity": "ISO timestamp",
  "metadata": {"ip": "192.168.1.1", ...},
  "data": {"custom": "data"},
  "active": true
}
```

## Dependencies

This AgenToolkit depends on:
- **storage_kv**: For persistent session storage in the "sessions" namespace with TTL support

## Tools

### session_create
```python
async def session_create(ctx: RunContext[Any], user_id: str, metadata: Optional[Dict[str, Any]], ttl: Optional[int]) -> SessionOutput
```
Create a new session with secure token generation. Default TTL is 24 hours (86400 seconds).

**Raises:**
- `ValueError`: If user_id is invalid or ttl is not positive
- `RuntimeError`: If session creation or storage fails

### session_get
```python
async def session_get(ctx: RunContext[Any], session_id: str) -> SessionOutput
```
Get session details. Automatically checks expiration and removes expired sessions.

**Raises:**
- `ValueError`: If session_id is invalid
- `KeyError`: If session doesn't exist
- `RuntimeError`: If session is expired or retrieval fails

### session_update
```python
async def session_update(ctx: RunContext[Any], session_id: str, data: Optional[Dict[str, Any]]) -> SessionOutput
```
Update session data and refresh last activity timestamp.

**Raises:**
- `ValueError`: If session_id is invalid
- `KeyError`: If session doesn't exist
- `RuntimeError`: If session is expired or update fails

### session_delete
```python
async def session_delete(ctx: RunContext[Any], session_id: str) -> SessionOutput
```
Delete a session from both memory and persistent storage.

**Raises:**
- `ValueError`: If session_id is invalid
- `RuntimeError`: If deletion fails

### session_validate
```python
async def session_validate(ctx: RunContext[Any], session_id: str) -> SessionOutput
```
Validate session existence, expiration, and active status. Updates last activity on success.

**Raises:**
- `ValueError`: If session_id is invalid
- `RuntimeError`: If validation process fails

Note: Returns `valid: false` in data for invalid/expired sessions instead of raising exceptions.

### session_renew
```python
async def session_renew(ctx: RunContext[Any], session_id: str, ttl: Optional[int]) -> SessionOutput
```
Extend session expiration. Default TTL is 24 hours if not specified.

**Raises:**
- `ValueError`: If session_id is invalid or ttl is not positive
- `KeyError`: If session doesn't exist
- `RuntimeError`: If session is expired or renewal fails

### session_list
```python
async def session_list(ctx: RunContext[Any], user_id: Optional[str]) -> SessionOutput
```
List all sessions, optionally filtered by user_id. Skips expired sessions.

**Raises:**
- `RuntimeError`: If listing fails

### session_invalidate_all
```python
async def session_invalidate_all(ctx: RunContext[Any], user_id: str) -> SessionOutput
```
Delete all sessions for a specific user. Useful for logout-everywhere functionality.

**Raises:**
- `ValueError`: If user_id is invalid
- `RuntimeError`: If invalidation fails

### session_get_active
```python
async def session_get_active(ctx: RunContext[Any]) -> SessionOutput
```
Get all currently active (non-expired) sessions with user statistics.

**Raises:**
- `RuntimeError`: If retrieval fails

## Exceptions

| Exception Type | Scenarios |
|---------------|-----------|
| `ValueError` | - Invalid user_id or session_id<br>- Non-positive TTL value<br>- Invalid parameter types |
| `KeyError` | - Session not found<br>- Session doesn't exist in storage |
| `RuntimeError` | - Session expired<br>- Storage operation failures<br>- Session creation errors |

## Usage Examples

### Session Lifecycle Management
```python
from agentoolkit.auth.session import create_session_agent
from agentool.core.injector import get_injector

# Create and register the agent
agent = create_session_agent()
injector = get_injector()

# Create a session
result = await injector.run('session', {
    "operation": "create",
    "user_id": "user123",
    "metadata": {
        "ip": "192.168.1.1",
        "user_agent": "Mozilla/5.0",
        "device": "desktop"
    },
    "ttl": 3600  # 1 hour
})
session_id = result.data["session_id"]

# Validate the session
result = await injector.run('session', {
    "operation": "validate",
    "session_id": session_id
})

# Update session data
result = await injector.run('session', {
    "operation": "update",
    "session_id": session_id,
    "data": {"last_page": "/dashboard"}
})

# Renew session for another hour
result = await injector.run('session', {
    "operation": "renew",
    "session_id": session_id,
    "ttl": 3600
})
```

### User Session Management
```python
# List all sessions for a user
result = await injector.run('session', {
    "operation": "list",
    "user_id": "user123"
})

# Invalidate all user sessions (logout everywhere)
result = await injector.run('session', {
    "operation": "invalidate_all",
    "user_id": "user123"
})

# Get all active sessions across the system
result = await injector.run('session', {
    "operation": "get_active"
})
active_count = result.data["total_active"]
unique_users = result.data["unique_users"]
```

### Session Data Storage
```python
# Store custom data in session
result = await injector.run('session', {
    "operation": "update",
    "session_id": session_id,
    "data": {
        "cart_items": ["item1", "item2"],
        "preferences": {"theme": "dark"}
    }
})

# Retrieve session with custom data
result = await injector.run('session', {
    "operation": "get",
    "session_id": session_id
})
cart_items = result.data["data"]["cart_items"]
```

## Testing

The test suite is located at `tests/agentoolkit/test_session.py`. Tests cover:
- Session creation with various TTL values
- Session validation and expiration
- Session renewal and activity tracking
- User session listing and invalidation
- Concurrent session management
- Storage persistence and recovery
- Error conditions and edge cases

To run tests:
```bash
pytest tests/agentoolkit/test_session.py -v
```

## Notes

- Session IDs are generated using `secrets.token_urlsafe(32)` for cryptographic security
- Default TTL is 24 hours (86400 seconds) if not specified
- Sessions are stored in both memory and storage_kv for performance and persistence
- Expired sessions are automatically cleaned up on access
- Last activity is updated on validation and update operations
- The validate operation returns success/failure in data rather than raising exceptions
- Session data can store any JSON-serializable content
- TTL is enforced both in memory and in storage_kv