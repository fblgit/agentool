# Configuration Management AgenToolkit

> **Note**: For guidelines on creating new AgenToolkits, see [CRAFTING_AGENTOOLS.md](../CRAFTING_AGENTOOLS.md). For testing patterns and examples, refer to [tests/agentoolkit/test_config.py](../../../tests/agentoolkit/test_config.py).

## Overview

The Configuration Management AgenToolkit provides structured configuration management with environment variable support, type validation, and hierarchical configuration patterns. It builds on top of storage_kv to provide high-level configuration management capabilities.

### Key Features
- Hierarchical configuration keys (dot notation like 'database.host.port')
- Environment variable integration with prefix support
- JSON, YAML, and environment file format support
- Type validation and automatic conversion
- Configuration reloading and validation with JSON schemas
- Default value support
- Namespace-based configuration isolation
- Nested dictionary manipulation

## Creation Method

```python
from agentoolkit.system.config import create_config_agent

# Create the agent
agent = create_config_agent()
```

The creation function returns a fully configured AgenTool with name `'config'`.

## Input Schema

### ConfigInput

The input schema is a Pydantic model with the following fields:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `operation` | `Literal['get', 'set', 'delete', 'list', 'reload', 'validate', 'load_file', 'save_file']` | Yes | - | The configuration operation to perform |
| `key` | `Optional[str]` | No | None | Configuration key (supports dot notation like 'db.host') |
| `value` | `Optional[Any]` | No | None | Configuration value (JSON serializable) |
| `namespace` | `str` | No | "app" | Configuration namespace |
| `format` | `Literal['json', 'yaml', 'env']` | No | "json" | Configuration format |
| `file_path` | `Optional[str]` | No | None | File path for load_file/save_file operations |
| `default` | `Optional[Any]` | No | None | Default value if key not found |
| `env_prefix` | `Optional[str]` | No | None | Environment variable prefix for loading |
| `validation_schema` | `Optional[Dict[str, Any]]` | No | None | JSON schema for validation |

## Operations Schema

The routing configuration maps operations to tool functions:

| Operation | Tool Function | Required Parameters | Description |
|-----------|--------------|-------------------|-------------|
| `get` | `config_get` | `key`, `namespace`, `default` | Get a configuration value with hierarchical support |
| `set` | `config_set` | `key`, `value`, `namespace` | Set a configuration value with hierarchical creation |
| `delete` | `config_delete` | `key`, `namespace` | Delete a configuration key |
| `list` | `config_list` | `namespace`, `key` | List configuration keys or get all configuration |
| `reload` | `config_reload` | `namespace`, `env_prefix` | Reload configuration from environment variables |
| `validate` | `config_validate` | `namespace`, `validation_schema` | Validate configuration against JSON schema |

## Output Schema

### ConfigOutput

All operations return a `ConfigOutput` model with:

| Field | Type | Description |
|-------|------|-------------|
| `operation` | `str` | The operation that was performed |
| `key` | `Optional[str]` | The key that was operated on |
| `namespace` | `str` | The namespace used |
| `message` | `str` | Human-readable result message |
| `data` | `Optional[Any]` | Operation-specific data |

### Operation-Specific Data Fields

- **get**: `value`, `exists`, `used_default`
- **set**: `value`, `key_parts`, `namespace_size`
- **delete**: `deleted`, `existed`
- **list**: `config`, `flattened`, `keys`, `count`
- **reload**: `env_vars_loaded`, `count`, `prefix`
- **validate**: `valid`, `schema`, `config_size`

## Dependencies

This AgenToolkit depends on:
- **storage_kv**: Used for persistent configuration storage

## Tools

### config_get
```python
async def config_get(ctx: RunContext[Any], key: str, namespace: str, default: Any) -> ConfigOutput
```
Get a configuration value using hierarchical key notation. Returns default value if key doesn't exist.

**Key Features:**
- Supports hierarchical keys (e.g., 'database.host.port')
- Returns default value if key not found
- Preserves data types from storage

### config_set
```python
async def config_set(ctx: RunContext[Any], key: str, value: Any, namespace: str) -> ConfigOutput
```
Set a configuration value using hierarchical key notation. Creates nested dictionaries as needed.

**Key Features:**
- Automatically creates nested structure for hierarchical keys
- Preserves existing configuration structure
- Supports any JSON-serializable value type

### config_delete
```python
async def config_delete(ctx: RunContext[Any], key: str, namespace: str) -> ConfigOutput
```
Delete a configuration key from the hierarchical structure.

**Key Features:**
- Removes key from nested dictionary structure
- Preserves other keys in the same branch
- Reports whether key existed before deletion

### config_list
```python
async def config_list(ctx: RunContext[Any], namespace: str, key: Optional[str]) -> ConfigOutput
```
List configuration keys or get all configuration. Optionally filter by key prefix.

**Key Features:**
- Flattens hierarchical structure for key listing
- Supports key prefix filtering
- Returns both hierarchical and flattened views

### config_reload
```python
async def config_reload(ctx: RunContext[Any], namespace: str, env_prefix: Optional[str]) -> ConfigOutput
```
Reload configuration from environment variables with automatic type conversion.

**Key Features:**
- Converts environment variable names to hierarchical keys
- Attempts JSON parsing for complex values
- Merges with existing configuration
- Configurable environment variable prefix

### config_validate
```python
async def config_validate(ctx: RunContext[Any], namespace: str, validation_schema: Dict[str, Any]) -> ConfigOutput
```
Validate configuration against a JSON schema using jsonschema library.

**Key Features:**
- Full JSON schema validation support
- Detailed error reporting on validation failure
- Requires jsonschema library to be installed

## Exceptions

| Exception Type | Scenarios |
|---------------|-----------|
| `ValueError` | - Configuration validation failures<br>- Empty namespace during validation |
| `RuntimeError` | - Storage operation failures<br>- Key manipulation errors<br>- Environment variable parsing errors |
| `ImportError` | - jsonschema library not available for validation |

## Usage Examples

### Hierarchical Configuration
```python
from agentoolkit.system.config import create_config_agent
from agentool.core.injector import get_injector

# Create and register the agent
agent = create_config_agent()
injector = get_injector()

# Set hierarchical configuration
result = await injector.run('config', {
    "operation": "set",
    "key": "database.host",
    "value": "localhost",
    "namespace": "app"
})

result = await injector.run('config', {
    "operation": "set",
    "key": "database.port",
    "value": 5432,
    "namespace": "app"
})

result = await injector.run('config', {
    "operation": "set",
    "key": "database.credentials.username",
    "value": "admin",
    "namespace": "app"
})

# Get hierarchical values
result = await injector.run('config', {
    "operation": "get",
    "key": "database.host",
    "namespace": "app"
})

# Get with default value
result = await injector.run('config', {
    "operation": "get",
    "key": "database.timeout",
    "default": 30,
    "namespace": "app"
})
```

### Environment Variable Integration
```python
# Set environment variables
# APP_DATABASE_HOST=prod.example.com
# APP_DATABASE_PORT=5432
# APP_API_KEYS='["key1", "key2", "key3"]'  # JSON array

# Reload from environment
result = await injector.run('config', {
    "operation": "reload",
    "namespace": "app",
    "env_prefix": "APP_"
})

# Check loaded values
result = await injector.run('config', {
    "operation": "get",
    "key": "database.host",
    "namespace": "app"
})
# Returns: "prod.example.com"

result = await injector.run('config', {
    "operation": "get",
    "key": "api.keys",
    "namespace": "app"
})
# Returns: ["key1", "key2", "key3"] (parsed from JSON)
```

### Configuration Management
```python
# List all configuration keys
result = await injector.run('config', {
    "operation": "list",
    "namespace": "app"
})

# List keys with prefix filter
result = await injector.run('config', {
    "operation": "list",
    "namespace": "app",
    "key": "database"
})

# Delete a configuration key
result = await injector.run('config', {
    "operation": "delete",
    "key": "database.credentials.password",
    "namespace": "app"
})
```

### Configuration Validation
```python
# Define JSON schema for validation
schema = {
    "type": "object",
    "properties": {
        "database": {
            "type": "object",
            "properties": {
                "host": {"type": "string"},
                "port": {"type": "integer", "minimum": 1, "maximum": 65535}
            },
            "required": ["host", "port"]
        },
        "api": {
            "type": "object",
            "properties": {
                "keys": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
    },
    "required": ["database"]
}

# Validate configuration
result = await injector.run('config', {
    "operation": "validate",
    "namespace": "app",
    "validation_schema": schema
})
```

### Complex Configuration Structures
```python
# Set complex nested configuration
result = await injector.run('config', {
    "operation": "set",
    "key": "services.web.instances",
    "value": [
        {"host": "web1.example.com", "port": 8080},
        {"host": "web2.example.com", "port": 8080}
    ],
    "namespace": "production"
})

# Get entire configuration for inspection
result = await injector.run('config', {
    "operation": "list",
    "namespace": "production"
})

# Access nested array element (would require multiple get operations)
result = await injector.run('config', {
    "operation": "get",
    "key": "services.web.instances",
    "namespace": "production"
})
```

### Multiple Namespaces
```python
# Development configuration
result = await injector.run('config', {
    "operation": "set",
    "key": "database.host",
    "value": "localhost",
    "namespace": "dev"
})

# Production configuration
result = await injector.run('config', {
    "operation": "set",
    "key": "database.host",
    "value": "prod-db.example.com",
    "namespace": "prod"
})

# Test configuration
result = await injector.run('config', {
    "operation": "set",
    "key": "database.host",
    "value": "test-db.example.com",
    "namespace": "test"
})
```

## Testing

The test suite is located at `tests/agentoolkit/test_config.py`. Tests cover:
- Hierarchical key manipulation (get, set, delete)
- Environment variable loading and parsing
- Default value handling
- Configuration validation with JSON schemas
- Namespace isolation
- Error handling for invalid operations
- Complex nested structure management

To run tests:
```bash
pytest tests/agentoolkit/test_config.py -v
```

## Notes

- Hierarchical keys use dot notation and automatically create nested dictionary structures
- Environment variable names are converted to lowercase with underscores replaced by dots
- JSON parsing is attempted for environment variable values, falling back to string if parsing fails
- Configuration validation requires the `jsonschema` library to be installed
- All configuration data is stored in the `storage_kv` namespace "config" with keys prefixed by "config:"
- Namespaces provide complete isolation between different configuration sets (e.g., dev, prod, test)
- The toolkit preserves data types when possible but all data must be JSON-serializable for storage
- Default values are returned immediately without storage access when keys don't exist