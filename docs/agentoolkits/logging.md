# Logging AgenToolkit

> **Note**: For guidelines on creating new AgenToolkits, see [CRAFTING_AGENTOOLS.md](../CRAFTING_AGENTOOLS.md). For testing patterns and examples, refer to [tests/agentoolkit/test_logging.py](../../../tests/agentoolkit/test_logging.py).

## Overview

The Logging AgenToolkit provides structured logging capabilities with multiple output formats, log levels, and automatic log rotation. It supports both console and file output with configurable formatting and log level filtering.

### Key Features
- Multiple log levels (DEBUG, INFO, WARN, ERROR, CRITICAL)
- Structured logging with JSON support
- Multiple output destinations (console, file, both)
- Automatic log file rotation and cleanup
- Logger namespacing for different components
- Custom log formatting (text and JSON)
- Log level filtering and threshold management
- Integration with storage_fs for file operations

## Creation Method

```python
from agentoolkit.system.logging import create_logging_agent

# Create the agent
agent = create_logging_agent()
```

The creation function returns a fully configured AgenTool with name `'logging'`.

## Input Schema

### LoggingInput

The input schema is a Pydantic model with the following fields:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `operation` | `Literal['log', 'configure', 'get_logs', 'clear_logs', 'rotate_logs']` | Yes | - | The logging operation to perform |
| `level` | `Literal['DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']` | No | "INFO" | Log level |
| `message` | `Optional[str]` | No | None | Log message |
| `data` | `Optional[Dict[str, Any]]` | No | None | Structured log data |
| `logger_name` | `str` | No | "default" | Logger name/namespace |
| `output` | `Literal['console', 'file', 'both']` | No | "console" | Log output destination |
| `file_path` | `Optional[str]` | No | None | Log file path (for file output) |
| `format` | `Literal['text', 'json']` | No | "text" | Log format |
| `max_file_size` | `int` | No | 10485760 | Max log file size in bytes (10MB default) |
| `max_files` | `int` | No | 5 | Maximum number of rotated log files to keep |

## Operations Schema

The routing configuration maps operations to tool functions:

| Operation | Tool Function | Required Parameters | Description |
|-----------|--------------|-------------------|-------------|
| `log` | `logging_log` | `level`, `message`, `data`, `logger_name`, `output`, `file_path`, `format` | Log a message with structured data |
| `configure` | `logging_configure` | `logger_name`, `level`, `output`, `file_path`, `format`, `max_file_size`, `max_files` | Configure logger settings |
| `get_logs` | `logging_get_logs` | `logger_name`, `file_path`, `level` | Retrieve recent log entries |
| `clear_logs` | `logging_clear_logs` | `logger_name`, `file_path` | Clear log files |

## Output Schema

### LoggingOutput

All operations return a `LoggingOutput` model with:

| Field | Type | Description |
|-------|------|-------------|
| `operation` | `str` | The operation that was performed |
| `logger_name` | `str` | The logger name used |
| `message` | `str` | Human-readable result message |
| `data` | `Optional[Any]` | Operation-specific data |

### Operation-Specific Data Fields

- **log**: `level`, `outputs`, `format`, `entry_length`
- **configure**: `configuration`, `total_loggers`
- **get_logs**: `entries`, `count`, `total_lines`, `file_path`
- **clear_logs**: `cleared_files`, `count`

## Dependencies

This AgenToolkit depends on:
- **storage_fs**: Used for file-based logging operations

## Tools

### logging_log
```python
async def logging_log(ctx: RunContext[Any], level: str, message: str, data: Optional[Dict[str, Any]], 
                     logger_name: str, output: str, file_path: Optional[str], format: str) -> LoggingOutput
```
Log a message with structured data. Supports level filtering and multiple output destinations.

**Key Features:**
- Level threshold filtering (messages below minimum level are skipped)
- Automatic log file rotation when size limits are exceeded
- Structured data serialization in JSON format
- Multiple output destinations (console, file, both)

**Raises:**
- `IOError`: If error writing to log file
- `RuntimeError`: For general logging errors

### logging_configure
```python
async def logging_configure(ctx: RunContext[Any], logger_name: str, level: str, output: str, 
                          file_path: Optional[str], format: str, max_file_size: int, max_files: int) -> LoggingOutput
```
Configure logger settings including level, output destination, and rotation parameters.

**Key Features:**
- Per-logger configuration storage
- File rotation parameters
- Output destination selection
- Log format configuration

**Raises:**
- `RuntimeError`: If configuration operation fails

### logging_get_logs
```python
async def logging_get_logs(ctx: RunContext[Any], logger_name: str, file_path: Optional[str], 
                          level: Optional[str]) -> LoggingOutput
```
Retrieve recent log entries from log files with optional level filtering.

**Key Features:**
- Returns last 100 log entries
- Level-based filtering
- Parses both JSON and text format logs
- Handles missing log files gracefully

**Raises:**
- `RuntimeError`: If error retrieving logs

### logging_clear_logs
```python
async def logging_clear_logs(ctx: RunContext[Any], logger_name: str, file_path: Optional[str]) -> LoggingOutput
```
Clear log files including main log and all rotated files.

**Key Features:**
- Clears main log file and up to 10 rotated files
- Reports count of cleared files
- Creates empty log file to reset logging

**Raises:**
- `RuntimeError`: If error clearing logs

## Log Formats

### Text Format
```
[2024-01-01T12:00:00.000Z] INFO     [auth] User logged in | {"user_id":"123","ip":"192.168.1.1"}
[2024-01-01T12:00:05.123Z] ERROR    [api] Request failed | {"endpoint":"/users","error":"timeout"}
```

### JSON Format
```json
{"timestamp":"2024-01-01T12:00:00.000Z","level":"INFO","logger":"auth","message":"User logged in","data":{"user_id":"123","ip":"192.168.1.1"}}
{"timestamp":"2024-01-01T12:00:05.123Z","level":"ERROR","logger":"api","message":"Request failed","data":{"endpoint":"/users","error":"timeout"}}
```

## Log Rotation

- Automatic rotation when file size exceeds `max_file_size`
- Rotated files are named with `.1`, `.2`, etc. suffixes
- Old rotated files beyond `max_files` limit are automatically removed
- Rotation preserves log continuity by creating new empty main log file

## Exceptions

| Exception Type | Scenarios |
|---------------|-----------|
| `IOError` | - Error writing to log file<br>- File permission issues<br>- Disk space issues |
| `RuntimeError` | - General logging operation failures<br>- Configuration errors<br>- Log retrieval errors |
| `FileNotFoundError` | - Log file doesn't exist during get_logs operation |

## Usage Examples

### Basic Logging
```python
from agentoolkit.system.logging import create_logging_agent
from agentool.core.injector import get_injector

# Create and register the agent
agent = create_logging_agent()
injector = get_injector()

# Simple info log
result = await injector.run('logging', {
    "operation": "log",
    "level": "INFO",
    "message": "Application started",
    "logger_name": "app"
})

# Log with structured data
result = await injector.run('logging', {
    "operation": "log",
    "level": "INFO",
    "message": "User logged in",
    "data": {
        "user_id": "user_123",
        "ip_address": "192.168.1.100",
        "user_agent": "Mozilla/5.0"
    },
    "logger_name": "auth"
})

# Error logging
result = await injector.run('logging', {
    "operation": "log",
    "level": "ERROR",
    "message": "Database connection failed",
    "data": {
        "database": "users_db",
        "error_code": "CONNECTION_TIMEOUT",
        "retry_count": 3
    },
    "logger_name": "database"
})
```

### Logger Configuration
```python
# Configure logger for file output with JSON format
result = await injector.run('logging', {
    "operation": "configure",
    "logger_name": "api",
    "level": "WARN",
    "output": "file",
    "file_path": "/var/log/api.log",
    "format": "json",
    "max_file_size": 50 * 1024 * 1024,  # 50MB
    "max_files": 10
})

# Configure logger for both console and file output
result = await injector.run('logging', {
    "operation": "configure",
    "logger_name": "security",
    "level": "ERROR",
    "output": "both",
    "file_path": "/var/log/security.log",
    "format": "text"
})

# Log to configured logger
result = await injector.run('logging', {
    "operation": "log",
    "level": "WARN",
    "message": "Rate limit exceeded",
    "data": {
        "ip": "10.0.0.1",
        "endpoint": "/api/users",
        "rate_limit": 100,
        "current_count": 150
    },
    "logger_name": "api"
})
```

### Log Management
```python
# Get recent log entries with level filtering
result = await injector.run('logging', {
    "operation": "get_logs",
    "logger_name": "api",
    "level": "ERROR"
})

# Get all recent logs
result = await injector.run('logging', {
    "operation": "get_logs",
    "logger_name": "auth",
    "file_path": "/var/log/auth.log"
})

# Clear log files
result = await injector.run('logging', {
    "operation": "clear_logs",
    "logger_name": "api"
})
```

### Multiple Loggers and Levels
```python
# Configure different loggers for different components
components = [
    ("database", "ERROR", "/var/log/db.log"),
    ("api", "INFO", "/var/log/api.log"),
    ("auth", "WARN", "/var/log/auth.log"),
    ("metrics", "DEBUG", "/var/log/metrics.log")
]

for logger_name, level, file_path in components:
    await injector.run('logging', {
        "operation": "configure",
        "logger_name": logger_name,
        "level": level,
        "output": "both",
        "file_path": file_path,
        "format": "json"
    })

# Log to different loggers
await injector.run('logging', {
    "operation": "log",
    "level": "ERROR",
    "message": "Connection pool exhausted",
    "data": {"pool_size": 20, "active_connections": 20},
    "logger_name": "database"
})

await injector.run('logging', {
    "operation": "log",
    "level": "DEBUG",
    "message": "Metric recorded",
    "data": {"metric": "api.requests", "value": 1},
    "logger_name": "metrics"
})
```

### Advanced Usage with Rotation
```python
# Configure logger with specific rotation settings
result = await injector.run('logging', {
    "operation": "configure",
    "logger_name": "high_volume",
    "level": "INFO",
    "output": "file",
    "file_path": "/var/log/high_volume.log",
    "format": "json",
    "max_file_size": 10 * 1024 * 1024,  # 10MB files
    "max_files": 20  # Keep 20 rotated files
})

# Generate logs that will trigger rotation
for i in range(1000):
    await injector.run('logging', {
        "operation": "log",
        "level": "INFO",
        "message": f"Processing item {i}",
        "data": {
            "item_id": f"item_{i}",
            "processed_at": "2024-01-01T12:00:00Z",
            "size": i * 1024
        },
        "logger_name": "high_volume"
    })
```

## Testing

The test suite is located at `tests/agentoolkit/test_logging.py`. Tests cover:
- All log levels and filtering
- Text and JSON format output
- Console and file output destinations
- Logger configuration and management
- Log file rotation and cleanup
- Structured data logging
- Error handling for file operations
- Log retrieval and parsing

To run tests:
```bash
pytest tests/agentoolkit/test_logging.py -v
```

## Notes

- Log levels follow standard severity hierarchy: DEBUG < INFO < WARN < ERROR < CRITICAL
- Messages below the configured minimum level are silently skipped
- File rotation is automatic when size limits are exceeded during logging
- JSON format logs are single-line entries for easy parsing
- Text format includes timestamp, level, logger name, message, and optional structured data
- The toolkit integrates with storage_fs for all file operations, providing consistent error handling
- Logger configurations are stored in memory and persist for the lifetime of the application
- Clearing logs removes both main log files and all rotated versions
- Log retrieval returns up to 100 recent entries to prevent memory issues with large log files