# Storage KV AgenToolkit

> **Note**: For guidelines on creating new AgenToolkits, see [CRAFTING_AGENTOOLS.md](../CRAFTING_AGENTOOLS.md). For testing patterns and examples, refer to [tests/agentoolkit/test_storage_kv.py](../../../tests/agentoolkit/test_storage_kv.py).

## Overview

The Storage KV (Key-Value) AgenToolkit provides an in-memory key-value storage system with Redis-compatible interface. It offers time-to-live (TTL) support, namespace isolation, pattern matching, and atomic operations for efficient data storage and retrieval.

### Key Features
- Full CRUD operations for key-value pairs
- TTL (Time To Live) support with automatic expiration
- Namespace support for data isolation
- Pattern-based key matching with wildcard support
- Atomic operations
- Memory-efficient implementation
- Redis-compatible interface and conventions

## Creation Method

```python
from agentoolkit.storage.kv import create_storage_kv_agent

# Create the agent
agent = create_storage_kv_agent()
```

The creation function returns a fully configured AgenTool with name `'storage_kv'`.

## Input Schema

### StorageKvInput

The input schema is a Pydantic model with the following fields:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `operation` | `Literal['get', 'set', 'delete', 'exists', 'keys', 'clear', 'expire', 'ttl']` | Yes | - | The key-value operation to perform |
| `key` | `Optional[str]` | No | None | Key for the operation |
| `value` | `Optional[Any]` | No | None | Value to store (JSON serializable) |
| `ttl` | `Optional[int]` | No | None | Time to live in seconds |
| `pattern` | `Optional[str]` | No | None | Pattern for keys operation (supports * wildcard) |
| `namespace` | `str` | No | "default" | Key namespace for data isolation |

## Operations Schema

The routing configuration maps operations to tool functions:

| Operation | Tool Function | Required Parameters | Description |
|-----------|--------------|-------------------|-------------|
| `get` | `kv_get` | `key`, `namespace` | Retrieve a value by key |
| `set` | `kv_set` | `key`, `value`, `namespace`, `ttl` | Store a key-value pair with optional TTL |
| `delete` | `kv_delete` | `key`, `namespace` | Delete a key |
| `exists` | `kv_exists` | `key`, `namespace` | Check if a key exists |
| `keys` | `kv_keys` | `namespace`, `pattern` | List keys matching a pattern |
| `clear` | `kv_clear` | `namespace` | Clear all keys in a namespace |
| `expire` | `kv_expire` | `key`, `namespace`, `ttl` | Set TTL for an existing key |
| `ttl` | `kv_ttl` | `key`, `namespace` | Get remaining TTL for a key |

## Output Schema

### StorageKvOutput

All operations return a `StorageKvOutput` model with:

| Field | Type | Description |
|-------|------|-------------|
| `operation` | `str` | The operation that was performed |
| `key` | `Optional[str]` | The key that was operated on |
| `namespace` | `str` | The namespace used |
| `message` | `str` | Human-readable result message |
| `data` | `Optional[Any]` | Operation-specific data |

### Operation-Specific Data Fields

- **get**: `value`, `exists`, `ttl_remaining`, `expired_keys_cleaned`
- **set**: `stored`, `ttl`, `namespace_size`
- **delete**: `deleted`, `existed`
- **exists**: `exists`
- **keys**: `keys`, `count`, `pattern`, `expired_keys_cleaned`
- **clear**: `cleared_count`, `namespace_existed`
- **expire**: `ttl_set`, `expires_at`
- **ttl**: `ttl`, `exists`, `has_expiry`, `expires_at` (Redis conventions: -2 = key doesn't exist, -1 = no expiry)

## Dependencies

This AgenToolkit has no external dependencies on other AgenToolkits. It uses an in-memory storage backend that could be replaced with Redis or other storage systems in production.

## Tools

### kv_get
```python
async def kv_get(ctx: RunContext[Any], key: str, namespace: str) -> StorageKvOutput
```
Get a value by key. Automatically cleans up expired keys in the namespace.

**Raises:**
- `KeyError`: If the key is not found or has expired
- `RuntimeError`: For other retrieval errors

### kv_set
```python
async def kv_set(ctx: RunContext[Any], key: str, value: Any, namespace: str, ttl: Optional[int]) -> StorageKvOutput
```
Set a key-value pair with optional TTL. Creates namespace if it doesn't exist.

**Raises:**
- `RuntimeError`: If there's an error setting the key

### kv_delete
```python
async def kv_delete(ctx: RunContext[Any], key: str, namespace: str) -> StorageKvOutput
```
Delete a key. Also removes any associated TTL.

**Raises:**
- `RuntimeError`: If there's an error deleting the key

### kv_exists
```python
async def kv_exists(ctx: RunContext[Any], key: str, namespace: str) -> StorageKvOutput
```
Check if a key exists and hasn't expired.

**Raises:**
- `RuntimeError`: If there's an error checking existence

### kv_keys
```python
async def kv_keys(ctx: RunContext[Any], namespace: str, pattern: Optional[str]) -> StorageKvOutput
```
List keys matching a pattern. Supports `*` wildcard. Automatically cleans expired keys.

**Raises:**
- `RuntimeError`: If there's an error listing keys

### kv_clear
```python
async def kv_clear(ctx: RunContext[Any], namespace: str) -> StorageKvOutput
```
Clear all keys in a namespace. Also removes all TTL data.

**Raises:**
- `RuntimeError`: If there's an error clearing the namespace

### kv_expire
```python
async def kv_expire(ctx: RunContext[Any], key: str, namespace: str, ttl: int) -> StorageKvOutput
```
Set TTL for an existing key.

**Raises:**
- `KeyError`: If the key doesn't exist
- `RuntimeError`: For other TTL setting errors

### kv_ttl
```python
async def kv_ttl(ctx: RunContext[Any], key: str, namespace: str) -> StorageKvOutput
```
Get the remaining TTL for a key. Returns Redis-compatible values.

**Raises:**
- `RuntimeError`: If there's an error getting TTL

## Exceptions

| Exception Type | Scenarios |
|---------------|-----------|
| `KeyError` | - Getting a non-existent key<br>- Getting an expired key<br>- Setting TTL on non-existent key |
| `RuntimeError` | - Storage operation failures<br>- Namespace access errors<br>- Serialization issues |

## Usage Examples

### Basic Key-Value Operations
```python
from agentoolkit.storage.kv import create_storage_kv_agent
from agentool.core.injector import get_injector

# Create and register the agent
agent = create_storage_kv_agent()
injector = get_injector()

# Set a value with TTL
result = await injector.run('storage_kv', {
    "operation": "set",
    "key": "user:123",
    "value": {"name": "Alice", "email": "alice@example.com"},
    "ttl": 3600,
    "namespace": "users"
})

# Get the value
result = await injector.run('storage_kv', {
    "operation": "get",
    "key": "user:123",
    "namespace": "users"
})

# Check if key exists
result = await injector.run('storage_kv', {
    "operation": "exists",
    "key": "user:123",
    "namespace": "users"
})
```

### Pattern Matching and Namespace Management
```python
# List all keys with pattern
result = await injector.run('storage_kv', {
    "operation": "keys",
    "pattern": "user:*",
    "namespace": "users"
})

# Clear entire namespace
result = await injector.run('storage_kv', {
    "operation": "clear",
    "namespace": "temp"
})
```

### TTL Management
```python
# Set TTL on existing key
result = await injector.run('storage_kv', {
    "operation": "expire",
    "key": "session:abc",
    "ttl": 1800,
    "namespace": "sessions"
})

# Check remaining TTL
result = await injector.run('storage_kv', {
    "operation": "ttl",
    "key": "session:abc",
    "namespace": "sessions"
})
```

## Testing

The test suite is located at `tests/agentoolkit/test_storage_kv.py`. Tests cover:
- All CRUD operations
- TTL functionality and expiration
- Namespace isolation
- Pattern matching
- Error handling for missing keys
- Edge cases and boundary conditions

To run tests:
```bash
pytest tests/agentoolkit/test_storage_kv.py -v
```

## Notes

- Keys automatically expire based on TTL settings
- Expired keys are cleaned up during get and keys operations
- Namespaces provide complete isolation between different data sets
- Pattern matching uses Python's `fnmatch` module for wildcard support
- The storage backend is in-memory by default but can be replaced with persistent storage
- Follows Redis conventions for TTL return values (-2 for non-existent, -1 for no expiry)