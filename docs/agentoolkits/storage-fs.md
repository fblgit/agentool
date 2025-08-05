# Storage FS AgenToolkit

> **Note**: For guidelines on creating new AgenToolkits, see [CRAFTING_AGENTOOLS.md](../CRAFTING_AGENTOOLS.md). For testing patterns and examples, refer to [tests/agentoolkit/test_storage_fs.py](../../../tests/agentoolkit/test_storage_fs.py).

## Overview

The Storage FS (File System) AgenToolkit provides a comprehensive interface for file system operations. It offers CRUD operations for files and directories, encoding support, pattern-based filtering, recursive operations, and security-aware file handling.

### Key Features
- Full CRUD operations for files and directories
- Encoding support for text files
- Pattern-based file filtering with glob support
- Recursive operations for directories
- Automatic parent directory creation
- Structured error handling
- Security-aware file operations

## Creation Method

```python
from agentoolkit.storage.fs import create_storage_fs_agent

# Create the agent
agent = create_storage_fs_agent()
```

The creation function returns a fully configured AgenTool with name `'storage_fs'`.

## Input Schema

### StorageFsInput

The input schema is a Pydantic model with the following fields:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `operation` | `Literal['read', 'write', 'append', 'delete', 'list', 'exists', 'mkdir', 'rmdir']` | Yes | - | The file system operation to perform |
| `path` | `str` | Yes | - | File or directory path |
| `content` | `Optional[str]` | No | None | Content for write/append operations |
| `encoding` | `str` | No | "utf-8" | Text encoding for file operations |
| `recursive` | `bool` | No | False | Recursive operation for list/delete |
| `pattern` | `Optional[str]` | No | None | Pattern for file filtering in list operations |
| `create_parents` | `bool` | No | True | Create parent directories if they don't exist |

## Operations Schema

The routing configuration maps operations to tool functions:

| Operation | Tool Function | Required Parameters | Description |
|-----------|--------------|-------------------|-------------|
| `read` | `fs_read` | `path`, `encoding` | Read content from a file |
| `write` | `fs_write` | `path`, `content`, `encoding`, `create_parents` | Write content to a file (overwrite) |
| `append` | `fs_append` | `path`, `content`, `encoding`, `create_parents` | Append content to a file |
| `delete` | `fs_delete` | `path`, `recursive` | Delete a file or directory |
| `list` | `fs_list` | `path`, `recursive`, `pattern` | List directory contents |
| `exists` | `fs_exists` | `path` | Check if a file or directory exists |
| `mkdir` | `fs_mkdir` | `path`, `create_parents` | Create a directory |
| `rmdir` | `fs_rmdir` | `path`, `recursive` | Remove a directory |

## Output Schema

### StorageFsOutput

All operations return a `StorageFsOutput` model with:

| Field | Type | Description |
|-------|------|-------------|
| `operation` | `str` | The operation that was performed |
| `path` | `str` | The path that was operated on |
| `message` | `str` | Human-readable result message |
| `data` | `Optional[Any]` | Operation-specific data |

### Operation-Specific Data Fields

- **read**: `content`, `size_bytes`, `encoding`
- **write**: `content_length`, `size_bytes`, `encoding`, `created_parents`
- **append**: `content_length`, `total_size_bytes`, `encoding`
- **delete**: `type` (file/directory), `existed`, `recursive` (for directories)
- **list**: `items` (array of file/dir info), `count`, `recursive`, `pattern`
- **exists**: `exists`, `type` (file/directory), `size` (for files)
- **mkdir**: `created`, `parents_created`, `existed`
- **rmdir**: `removed`, `recursive`, `existed`

### List Operation Item Structure

Each item in the list operation contains:
- `name`: File or directory name
- `path`: Full path to the item
- `relative_path`: Relative path from listing root (recursive only)
- `type`: "file" or "directory"
- `size`: File size in bytes (None for directories)

## Dependencies

This AgenToolkit has no external dependencies on other AgenToolkits. It uses Python's built-in pathlib, os, glob, and shutil modules for file system operations.

## Tools

### fs_read
```python
async def fs_read(ctx: RunContext[Any], path: str, encoding: str) -> StorageFsOutput
```
Read content from a file with specified encoding.

**Raises:**
- `FileNotFoundError`: If the file doesn't exist
- `ValueError`: If the path is not a file
- `UnicodeDecodeError`: If encoding fails
- `PermissionError`: If read permission is denied
- `RuntimeError`: For other read errors

### fs_write
```python
async def fs_write(ctx: RunContext[Any], path: str, content: str, encoding: str, create_parents: bool) -> StorageFsOutput
```
Write content to a file, overwriting if it exists. Optionally creates parent directories.

**Raises:**
- `PermissionError`: If write permission is denied
- `FileNotFoundError`: If parent directory doesn't exist and create_parents is False
- `RuntimeError`: For other write errors

### fs_append
```python
async def fs_append(ctx: RunContext[Any], path: str, content: str, encoding: str, create_parents: bool) -> StorageFsOutput
```
Append content to a file. Creates the file if it doesn't exist.

**Raises:**
- `PermissionError`: If append permission is denied
- `RuntimeError`: For other append errors

### fs_delete
```python
async def fs_delete(ctx: RunContext[Any], path: str, recursive: bool) -> StorageFsOutput
```
Delete a file or directory. Use recursive=True for non-empty directories.

**Raises:**
- `PermissionError`: If delete permission is denied
- `OSError`: If directory is not empty and recursive is False
- `RuntimeError`: For other delete errors

### fs_list
```python
async def fs_list(ctx: RunContext[Any], path: str, recursive: bool, pattern: Optional[str]) -> StorageFsOutput
```
List directory contents with optional pattern filtering and recursive traversal.

**Raises:**
- `FileNotFoundError`: If directory doesn't exist
- `ValueError`: If path is not a directory
- `PermissionError`: If list permission is denied
- `RuntimeError`: For other list errors

### fs_exists
```python
async def fs_exists(ctx: RunContext[Any], path: str) -> StorageFsOutput
```
Check if a file or directory exists and return its type and size.

**Raises:**
- `RuntimeError`: If there's an error checking existence

### fs_mkdir
```python
async def fs_mkdir(ctx: RunContext[Any], path: str, create_parents: bool) -> StorageFsOutput
```
Create a directory. Optionally creates parent directories.

**Raises:**
- `ValueError`: If path exists but is not a directory
- `FileNotFoundError`: If parent doesn't exist and create_parents is False
- `PermissionError`: If create permission is denied
- `RuntimeError`: For other mkdir errors

### fs_rmdir
```python
async def fs_rmdir(ctx: RunContext[Any], path: str, recursive: bool) -> StorageFsOutput
```
Remove a directory. Use recursive=True for non-empty directories.

**Raises:**
- `ValueError`: If path is not a directory
- `OSError`: If directory is not empty and recursive is False
- `PermissionError`: If remove permission is denied
- `RuntimeError`: For other rmdir errors

## Exceptions

| Exception Type | Scenarios |
|---------------|-----------|
| `FileNotFoundError` | - Reading non-existent file<br>- Listing non-existent directory<br>- Parent directory missing |
| `ValueError` | - Path is wrong type (file vs directory)<br>- Invalid path format |
| `PermissionError` | - Insufficient permissions for operation<br>- Protected system files |
| `OSError` | - Directory not empty for delete/rmdir<br>- File system errors |
| `UnicodeDecodeError` | - Invalid encoding for file content |
| `RuntimeError` | - General file system operation failures |

## Usage Examples

### Basic File Operations
```python
from agentoolkit.storage.fs import create_storage_fs_agent
from agentool.core.injector import get_injector

# Create and register the agent
agent = create_storage_fs_agent()
injector = get_injector()

# Write a file
result = await injector.run('storage_fs', {
    "operation": "write",
    "path": "/tmp/test.txt",
    "content": "Hello, World!",
    "encoding": "utf-8"
})

# Read the file
result = await injector.run('storage_fs', {
    "operation": "read",
    "path": "/tmp/test.txt"
})

# Append to file
result = await injector.run('storage_fs', {
    "operation": "append",
    "path": "/tmp/test.txt",
    "content": "\nAdditional content"
})
```

### Directory Operations
```python
# Create directory with parents
result = await injector.run('storage_fs', {
    "operation": "mkdir",
    "path": "/tmp/myapp/data",
    "create_parents": True
})

# List directory contents with pattern
result = await injector.run('storage_fs', {
    "operation": "list",
    "path": "/tmp",
    "pattern": "*.txt",
    "recursive": False
})

# List recursively
result = await injector.run('storage_fs', {
    "operation": "list",
    "path": "/tmp/myapp",
    "recursive": True
})
```

### File Management
```python
# Check if file exists
result = await injector.run('storage_fs', {
    "operation": "exists",
    "path": "/tmp/test.txt"
})

# Delete a file
result = await injector.run('storage_fs', {
    "operation": "delete",
    "path": "/tmp/test.txt"
})

# Remove directory recursively
result = await injector.run('storage_fs', {
    "operation": "rmdir",
    "path": "/tmp/myapp",
    "recursive": True
})
```

## Testing

The test suite is located at `tests/agentoolkit/test_storage_fs.py`. Tests cover:
- All file operations (read, write, append, delete)
- Directory operations (create, list, remove)
- Encoding handling
- Pattern matching for listings
- Recursive operations
- Error conditions and permission handling
- Edge cases (empty files, special characters)

To run tests:
```bash
pytest tests/agentoolkit/test_storage_fs.py -v
```

## Notes

- File paths should be absolute or relative to the current working directory
- Default encoding is UTF-8, but can be changed per operation
- Pattern matching uses Python's glob syntax (*, **, ?, [])
- Recursive delete operations are irreversible - use with caution
- Parent directory creation is enabled by default for safety
- The toolkit handles both files and directories uniformly where applicable
- Items in list operations are sorted alphabetically by name