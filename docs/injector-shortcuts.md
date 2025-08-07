# AgenTool Injector Shortcuts Documentation

## Overview

The AgenTool injector provides convenient shortcut methods for common operations, reducing boilerplate code while maintaining full type safety and compatibility with the standard `run()` method.

## Motivation

The standard injector `run()` method requires verbose dictionary syntax for even simple operations:

```python
# Standard verbose approach
result = await injector.run('logging', {
    "operation": "log",
    "level": "INFO",
    "message": "User logged in",
    "data": {"user_id": 123},
    "logger_name": "default"
})
```

Shortcuts provide a cleaner, more intuitive API:

```python
# Shortcut approach
result = await injector.log("User logged in", "INFO", {"user_id": 123})
```

## Implementation

### Core Shortcut Methods

All shortcuts are implemented in [src/agentool/core/injector.py:821-1113](../src/agentool/core/injector.py#L821-L1113) and follow a consistent pattern:

1. **Direct wrappers** - Call `self.run()` internally
2. **Type preservation** - Return the same output as the standard method
3. **Error handling** - Gracefully handle failures with sensible defaults

### Available Shortcuts

#### Logging Operations
- `log(message, level="INFO", data=None, logger_name="default")` - [injector.py:821](../src/agentool/core/injector.py#L821)
  - Matches: `logging_log(ctx, message, level, data, logger_name)` from logging toolkit
  - Returns: `LoggingOutput` with full operation details

#### Key-Value Storage
- `kget(key, default=None, namespace="default")` - [injector.py:849](../src/agentool/core/injector.py#L849)
  - Matches: `kv_get(ctx, key, namespace)` from storage_kv toolkit
  - Returns: Stored value or default if not found (extracts from `StorageKvOutput.data['value']`)
- `kset(key, value, ttl=None, namespace="default")` - [injector.py:889](../src/agentool/core/injector.py#L889)
  - Matches: `kv_set(ctx, key, value, namespace, ttl)` from storage_kv toolkit
  - Returns: `bool` indicating success (extracts from `StorageKvOutput.success`)

#### File Storage
- `fsread(path)` - [injector.py:1006](../src/agentool/core/injector.py#L1006)
  - Matches: `fs_read(ctx, path, encoding='utf-8')` from storage_fs toolkit
  - Returns: File content as string or `None` (extracts from `StorageFsOutput.data['content']`)
- `fswrite(path, content, create_parents=True)` - [injector.py:1028](../src/agentool/core/injector.py#L1028)
  - Matches: `fs_write(ctx, path, content, mode='w', encoding='utf-8', create_parents)` from storage_fs toolkit
  - Returns: `bool` indicating success (extracts from `StorageFsOutput.success`)

#### Configuration
- `config_get(key, namespace="app", default=None)` - [injector.py:1057](../src/agentool/core/injector.py#L1057)
  - Matches: `config_get(ctx, key, namespace, default, parent)` from config toolkit
  - Returns: Configuration value or default (extracts from `ConfigOutput.data['value']`)
- `config_set(key, value, namespace="app")` - [injector.py:1086](../src/agentool/core/injector.py#L1086)
  - Matches: `config_set(ctx, key, value, namespace, ttl=None)` from config toolkit
  - Returns: `bool` indicating success (extracts from `ConfigOutput.success`)

#### Metrics
- `metric_inc(name, value=1.0, labels=None)` - [injector.py:954](../src/agentool/core/injector.py#L954)
  - Matches: `metrics_increment(ctx, name, value, labels)` from metrics toolkit
  - Returns: `MetricsOutput` with operation result
- `metric_dec(name, value=1.0, labels=None)` - [injector.py:980](../src/agentool/core/injector.py#L980)
  - Matches: `metrics_decrement(ctx, name, value, labels)` from metrics toolkit
  - Returns: `MetricsOutput` with operation result

#### References
- `ref(ref_type, key, content)` - [injector.py:1180](../src/agentool/core/injector.py#L1180)
  - Stores content in the specified storage type and returns a reference
  - Returns: Reference string like `"!ref:storage_kv:key"`
- `unref(reference)` - [injector.py:1115](../src/agentool/core/injector.py#L1115)
  - Resolves a reference string by retrieving content from the appropriate storage
  - Supports formats: `!ref:storage_kv:key`, `!ref:storage_fs:path`, `!ref:config:key`
  - Returns: Resolved content or error string

## Template System

### Modular Templates

The shortcut documentation is built from modular templates defined in [injector.py:30-92](../src/agentool/core/injector.py#L30-L92):

```python
_TEMPLATE_SHORT_DEFAULT_START  # Header
_TEMPLATE_SHORT_LOGGING        # Logging examples
_TEMPLATE_SHORT_STORAGE_KV     # KV storage examples
_TEMPLATE_SHORT_STORAGE_FS     # File storage examples
_TEMPLATE_SHORT_CONFIG         # Configuration examples
_TEMPLATE_SHORT_METRICS        # Metrics examples
_TEMPLATE_SHORT_REFERENCES     # Reference examples
_TEMPLATE_SHORT_DEFAULT_END    # Footer

_SHORTCUTS_TEMPLATES = {
    'logging': _TEMPLATE_SHORT_LOGGING,
    'storage_kv': _TEMPLATE_SHORT_STORAGE_KV,
    'storage_fs': _TEMPLATE_SHORT_STORAGE_FS,
    'config': _TEMPLATE_SHORT_CONFIG,
    'metrics': _TEMPLATE_SHORT_METRICS,
}
```

### Dynamic Documentation

The `show_shortcuts()` method ([injector.py:1226](../src/agentool/core/injector.py#L1226)) dynamically builds documentation based on dependencies:

```python
# Show all shortcuts
injector.show_shortcuts()

# Show only specific shortcuts
injector.show_shortcuts(['storage_kv', 'logging'])

# Show shortcuts for agent dependencies
config = AgenToolRegistry.get('my_agent')
injector.show_shortcuts(config.dependencies)
```

## Toolkit Signature Compatibility

### Verified Toolkit Functions

All shortcuts are thin wrappers that call the underlying toolkit functions through `self.run()`. The signatures have been verified against the actual implementations:

| Shortcut | Toolkit Function | Location |
|----------|-----------------|----------|
| `log()` | `logging_log(ctx, message, level, data, logger_name)` | agentoolkit/system/logging.py |
| `kget()` | `kv_get(ctx, key, namespace)` | agentoolkit/storage/kv.py:146 |
| `kset()` | `kv_set(ctx, key, value, namespace, ttl)` | agentoolkit/storage/kv.py:202 |
| `fsread()` | `fs_read(ctx, path, encoding)` | agentoolkit/storage/fs.py |
| `fswrite()` | `fs_write(ctx, path, content, mode, encoding, create_parents)` | agentoolkit/storage/fs.py |
| `config_get()` | `config_get(ctx, key, namespace, default, parent)` | agentoolkit/system/config.py |
| `config_set()` | `config_set(ctx, key, value, namespace, ttl)` | agentoolkit/system/config.py |
| `metric_inc()` | `metrics_increment(ctx, name, value, labels)` | agentoolkit/system/metrics.py |
| `metric_dec()` | `metrics_decrement(ctx, name, value, labels)` | agentoolkit/system/metrics.py |

## Important Considerations

### Type Safety and Output Schemas

While shortcuts simplify the API, developers still need to understand the underlying output schemas when crafting AgenTools:

1. **Output Types** - Each AgenTool has a specific output type (e.g., `StorageKvOutput`, `LoggingOutput`)
2. **Data Structure** - The `data` field contains operation-specific results
3. **Discovery Patterns** - Some operations return `success=False` for "not found" cases

Example:
```python
# The shortcut returns the full typed output
result = await injector.log("Test message")
# result is LoggingOutput with:
# - result.success: bool
# - result.operation: str
# - result.message: str
# - result.data: Dict[str, Any]

# For convenience shortcuts like kget, we extract the value
value = await injector.kget("user:123")  # Returns just the value
# Internally this does:
# result = await self.run('storage_kv', {...})
# return result.data.get("value", default) if result.success else default
```

### When to Use Shortcuts vs run()

**Use shortcuts when:**
- Performing common, simple operations
- You want cleaner, more readable code
- The convenience methods provide the return type you need

**Use run() when:**
- You need full control over all parameters
- Working with less common operations
- You need the complete output object (for convenience shortcuts)

## Extending the System

### Adding New Shortcuts

To add a new shortcut:

1. **Define the method** in `AgenToolInjector` class:
```python
async def my_shortcut(self, param1: str, param2: Any = None) -> Any:
    """
    Brief description of the shortcut.
    
    Args:
        param1: Description
        param2: Description
        
    Returns:
        Description of return value
        
    Example:
        result = await injector.my_shortcut("value")
    """
    return await self.run('agent_name', {
        "operation": "op_name",
        "field1": param1,
        "field2": param2
    })
```

2. **Add template** if it should be documented:
```python
_TEMPLATE_SHORT_MY_AGENT = """
MY AGENT
   await injector.my_shortcut("example")
"""

_SHORTCUTS_TEMPLATES['my_agent'] = _TEMPLATE_SHORT_MY_AGENT
```

3. **Consider return type**:
   - Return full result for operations where all fields are useful
   - Extract specific values for convenience methods (like `kget`)
   - Return booleans for success/failure operations

### Hidden Methods

Some methods are intentionally undocumented but still functional:

- `metric()` - Generic metric operation ([injector.py:938](../src/agentool/core/injector.py#L938))
- `chain()` - Fluent operation chaining ([injector.py:1220](../src/agentool/core/injector.py#L1220))

These are kept for internal use or future features.

## Usage Examples

### Basic Operations

```python
from agentool.core.injector import get_injector

injector = get_injector()

# Logging
await injector.log("Application started", "INFO")
await injector.log("Error occurred", "ERROR", {"code": 500, "path": "/api"})

# Key-Value Storage
await injector.kset("user:123", {"name": "Alice", "role": "admin"})
user = await injector.kget("user:123", default={})

# File Operations
config_text = await injector.fsread("/etc/app/config.json")
await injector.fswrite("/tmp/output.txt", "Results: ...")

# Configuration
db_host = await injector.config_get("database.host", default="localhost")
await injector.config_set("feature.enabled", True)

# Metrics
await injector.metric_inc("api.requests")
await injector.metric_dec("active.connections")
```

### Reference System

```python
# Store data and get a reference
ref = await injector.ref("storage_kv", "template_data", {
    "title": "Welcome",
    "body": "Hello, {{name}}!"
})
print(ref)  # "!ref:storage_kv:template_data"

# Later, resolve the reference
data = await injector.unref("!ref:storage_kv:template_data")
print(data)  # {"title": "Welcome", "body": "Hello, {{name}}!"}
```

### Working with Dependencies

```python
# In an AgenTool that uses dependencies
class MyToolInput(BaseOperationInput):
    operation: Literal['process']
    
async def my_tool_process(ctx: RunContext[Any]):
    injector = get_injector()
    
    # Use shortcuts for cleaner code
    await injector.log("Processing started")
    
    config = await injector.config_get("my_tool.setting", default="default_value")
    
    # Process...
    
    await injector.metric_inc("my_tool.processed")
    await injector.log("Processing completed")
```

## Best Practices

1. **Use shortcuts for readability** - They make code more maintainable
2. **Understand return types** - Know what each shortcut returns
3. **Handle failures gracefully** - Use default values where appropriate
4. **Log important operations** - Use the log shortcut for debugging
5. **Track metrics** - Use metric shortcuts for observability
6. **Document dependencies** - When creating AgenTools, document which shortcuts are available

## See Also

- [AGENT_RESULT.md](AGENT_RESULT.md) - Type safety documentation
- [src/agentool/core/injector.py](../src/agentool/core/injector.py) - Implementation details
- [examples/demo_agentoolkit.py](../examples/demo_agentoolkit.py) - Usage examples