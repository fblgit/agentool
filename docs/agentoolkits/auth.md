# Auth AgenToolkit

> **Note**: For guidelines on creating new AgenToolkits, see [CRAFTING_AGENTOOLS.md](../CRAFTING_AGENTOOLS.md). For testing patterns and examples, refer to [tests/agentoolkit/test_auth.py](../../../tests/agentoolkit/test_auth.py).

## Overview

The Auth AgenToolkit provides comprehensive authentication and authorization capabilities. It handles user registration, login/logout, password management, token generation/verification, and role-based access control (RBAC). The toolkit integrates with crypto and storage_kv agentoolkits for secure password hashing and persistent user storage.

### Key Features
- User registration and authentication
- Secure password hashing with bcrypt (via crypto toolkit)
- Session management (via session toolkit integration)
- JWT token generation and verification
- Role-based access control (RBAC)
- Password reset and change functionality
- User profile management
- Multi-factor authentication support (placeholder)
- Persistent storage via storage_kv

## Creation Method

```python
from agentoolkit.auth.auth import create_auth_agent

# Create the agent
agent = create_auth_agent()
```

The creation function returns a fully configured AgenTool with name `'auth'`.

## Input Schema

### AuthInput

The input schema is a Pydantic model with the following fields:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `operation` | `Literal['register', 'login', 'logout', 'verify_password', 'change_password', 'reset_password', 'generate_token', 'verify_token', 'check_permission', 'assign_role', 'revoke_role', 'get_user', 'update_user', 'delete_user']` | Yes | - | The authentication operation to perform |
| `username` | `Optional[str]` | No | None | Username or email |
| `password` | `Optional[str]` | No | None | User password |
| `new_password` | `Optional[str]` | No | None | New password for changes |
| `user_id` | `Optional[str]` | No | None | User identifier |
| `email` | `Optional[str]` | No | None | User email |
| `metadata` | `Optional[Dict[str, Any]]` | No | None | User metadata |
| `session_id` | `Optional[str]` | No | None | Session identifier |
| `token` | `Optional[str]` | No | None | Authentication token |
| `role` | `Optional[str]` | No | None | User role |
| `permission` | `Optional[str]` | No | None | Permission to check |

## Operations Schema

The routing configuration maps operations to tool functions:

| Operation | Tool Function | Required Parameters | Description |
|-----------|--------------|-------------------|-------------|
| `register` | `auth_register` | `username`, `password`, `email`, `metadata` | Register a new user |
| `login` | `auth_login` | `username`, `password` | Authenticate and create session |
| `logout` | `auth_logout` | `session_id` | Invalidate user session |
| `verify_password` | `auth_verify_password` | `username`, `password` | Verify password without login |
| `change_password` | `auth_change_password` | `username`, `password`, `new_password` | Change user password |
| `reset_password` | `auth_reset_password` | `username`, `new_password` | Admin password reset |
| `generate_token` | `auth_generate_token` | `user_id`, `metadata` | Generate JWT token |
| `verify_token` | `auth_verify_token` | `token` | Verify JWT token |
| `check_permission` | `auth_check_permission` | `user_id`, `permission` | Check user permission |
| `assign_role` | `auth_assign_role` | `user_id`, `role` | Assign role to user |
| `revoke_role` | `auth_revoke_role` | `user_id`, `role` | Revoke role from user |
| `get_user` | `auth_get_user` | `user_id` | Get user information |
| `update_user` | `auth_update_user` | `user_id`, `metadata` | Update user metadata |
| `delete_user` | `auth_delete_user` | `user_id` | Delete user account |

## Output Schema

### AuthOutput

All operations return an `AuthOutput` model with:

| Field | Type | Description |
|-------|------|-------------|
| `operation` | `str` | The operation that was performed |
| `message` | `str` | Human-readable result message |
| `data` | `Optional[Dict[str, Any]]` | Operation-specific data |

### Operation-Specific Data Fields

- **register**: `user_id`, `username`, `email`, `roles`
- **login**: `user_id`, `username`, `session_id`, `roles`
- **logout**: `user_id`, `session_deleted`
- **verify_password**: `valid`
- **change_password**: `user_id`, `sessions_invalidated`
- **reset_password**: `user_id`, `sessions_invalidated`
- **generate_token**: `token`, `expires_in`, `user_id`
- **verify_token**: `valid`, `user_id`, `username`, `roles`, `metadata` (or `error` if invalid)
- **check_permission**: `has_permission`, `user_id`, `permission`, `user_roles`
- **assign_role**: `user_id`, `role`, `current_roles`
- **revoke_role**: `user_id`, `role`, `current_roles`
- **get_user**: Complete user data (without password/salt)
- **update_user**: `user_id`, `updated_fields`
- **delete_user**: `user_id`, `sessions_invalidated`

## Dependencies

This AgenToolkit depends on:
- **crypto**: For password hashing (bcrypt), JWT operations, and salt generation
- **storage_kv**: For persistent user storage in the "auth" namespace
- **session**: For session management (login/logout operations)

## Predefined Roles and Permissions

The toolkit includes default role definitions:
- **admin**: `read`, `write`, `delete`, `admin`
- **user**: `read`, `write`
- **guest**: `read`

## Tools

### auth_register
```python
async def auth_register(ctx: RunContext[Any], username: str, password: str, email: Optional[str], metadata: Optional[Dict[str, Any]]) -> AuthOutput
```
Register a new user with encrypted password storage.

**Raises:**
- `ValueError`: If username already exists
- `RuntimeError`: If registration fails

### auth_login
```python
async def auth_login(ctx: RunContext[Any], username: str, password: str) -> AuthOutput
```
Authenticate user and create a session (24-hour TTL).

**Raises:**
- `ValueError`: If credentials are invalid
- `PermissionError`: If account is deactivated
- `RuntimeError`: If login fails

### auth_logout
```python
async def auth_logout(ctx: RunContext[Any], session_id: str) -> AuthOutput
```
Invalidate user session.

**Raises:**
- `ValueError`: If session is invalid or expired
- `RuntimeError`: If logout fails

### auth_verify_password
```python
async def auth_verify_password(ctx: RunContext[Any], username: str, password: str) -> AuthOutput
```
Verify password without creating a session.

**Raises:**
- `RuntimeError`: If verification fails

### auth_change_password
```python
async def auth_change_password(ctx: RunContext[Any], username: str, password: str, new_password: str) -> AuthOutput
```
Change user password (requires current password). Invalidates all sessions.

**Raises:**
- `ValueError`: If current password is incorrect
- `RuntimeError`: If password change fails

### auth_reset_password
```python
async def auth_reset_password(ctx: RunContext[Any], username: str, new_password: str) -> AuthOutput
```
Admin operation to reset password without current password. Invalidates all sessions.

**Raises:**
- `KeyError`: If user not found
- `RuntimeError`: If reset fails

### auth_generate_token
```python
async def auth_generate_token(ctx: RunContext[Any], user_id: str, metadata: Optional[Dict[str, Any]]) -> AuthOutput
```
Generate JWT token (1-hour expiration).

**Raises:**
- `KeyError`: If user not found
- `RuntimeError`: If token generation fails

### auth_verify_token
```python
async def auth_verify_token(ctx: RunContext[Any], token: str) -> AuthOutput
```
Verify and decode JWT token.

**Raises:**
- `RuntimeError`: If verification fails

### auth_check_permission
```python
async def auth_check_permission(ctx: RunContext[Any], user_id: str, permission: str) -> AuthOutput
```
Check if user has specific permission based on roles.

**Raises:**
- `RuntimeError`: If permission check fails

### auth_assign_role
```python
async def auth_assign_role(ctx: RunContext[Any], user_id: str, role: str) -> AuthOutput
```
Assign a role to user.

**Raises:**
- `KeyError`: If user not found
- `ValueError`: If role doesn't exist
- `RuntimeError`: If assignment fails

### auth_revoke_role
```python
async def auth_revoke_role(ctx: RunContext[Any], user_id: str, role: str) -> AuthOutput
```
Revoke a role from user.

**Raises:**
- `KeyError`: If user not found
- `RuntimeError`: If revocation fails

### auth_get_user
```python
async def auth_get_user(ctx: RunContext[Any], user_id: str) -> AuthOutput
```
Get user information (excludes password/salt).

**Raises:**
- `KeyError`: If user not found
- `RuntimeError`: If retrieval fails

### auth_update_user
```python
async def auth_update_user(ctx: RunContext[Any], user_id: str, metadata: Optional[Dict[str, Any]]) -> AuthOutput
```
Update user metadata.

**Raises:**
- `KeyError`: If user not found
- `RuntimeError`: If update fails

### auth_delete_user
```python
async def auth_delete_user(ctx: RunContext[Any], user_id: str) -> AuthOutput
```
Delete user account and invalidate all sessions.

**Raises:**
- `KeyError`: If user not found
- `RuntimeError`: If deletion fails

## Exceptions

| Exception Type | Scenarios |
|---------------|-----------|
| `ValueError` | - Username already exists<br>- Invalid credentials<br>- Invalid role<br>- Incorrect password |
| `KeyError` | - User not found<br>- Session not found |
| `PermissionError` | - Account deactivated<br>- Insufficient permissions |
| `RuntimeError` | - Operation failures<br>- Dependency toolkit errors |

## Usage Examples

### User Registration and Login
```python
from agentoolkit.auth.auth import create_auth_agent
from agentool.core.injector import get_injector

# Create and register the agent
agent = create_auth_agent()
injector = get_injector()

# Register new user
result = await injector.run('auth', {
    "operation": "register",
    "username": "john_doe",
    "password": "SecurePass123!",
    "email": "john@example.com",
    "metadata": {"full_name": "John Doe"}
})

# Login user
result = await injector.run('auth', {
    "operation": "login",
    "username": "john_doe",
    "password": "SecurePass123!"
})
session_id = result.data["session_id"]

# Logout user
result = await injector.run('auth', {
    "operation": "logout",
    "session_id": session_id
})
```

### Password Management
```python
# Change password
result = await injector.run('auth', {
    "operation": "change_password",
    "username": "john_doe",
    "password": "SecurePass123!",
    "new_password": "NewSecurePass456!"
})

# Admin password reset
result = await injector.run('auth', {
    "operation": "reset_password",
    "username": "john_doe",
    "new_password": "ResetPass789!"
})
```

### Token Management
```python
# Generate token
result = await injector.run('auth', {
    "operation": "generate_token",
    "user_id": "john_doe",
    "metadata": {"scope": "api"}
})
token = result.data["token"]

# Verify token
result = await injector.run('auth', {
    "operation": "verify_token",
    "token": token
})
```

### Role-Based Access Control
```python
# Assign role
result = await injector.run('auth', {
    "operation": "assign_role",
    "user_id": "john_doe",
    "role": "admin"
})

# Check permission
result = await injector.run('auth', {
    "operation": "check_permission",
    "user_id": "john_doe",
    "permission": "delete"
})
has_permission = result.data["has_permission"]

# Revoke role
result = await injector.run('auth', {
    "operation": "revoke_role",
    "user_id": "john_doe",
    "role": "admin"
})
```

## Testing

The test suite is located at `tests/agentoolkit/test_auth.py`. Tests cover:
- User registration with duplicate checking
- Login/logout flow
- Password verification and changes
- Token generation and validation
- Role assignment and permission checking
- User data management
- Session integration
- Error conditions and edge cases

To run tests:
```bash
pytest tests/agentoolkit/test_auth.py -v
```

## Notes

- Passwords are hashed using bcrypt (via crypto toolkit) with 10,000 iterations
- User data is persisted in storage_kv under the "auth" namespace
- Sessions have a 24-hour TTL by default
- JWT tokens expire after 1 hour
- Default role "user" is assigned to new registrations
- Password changes/resets invalidate all user sessions
- The toolkit uses "auth_secret_key" for JWT signing (should be secured in production)
- User IDs use username for simplicity (can be changed to UUIDs)