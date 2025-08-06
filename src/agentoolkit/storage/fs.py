"""
File System Storage AgenTool - Provides comprehensive file system operations.

This toolkit provides a unified interface for file system operations including
read, write, append, delete, list, and directory management operations.

Features:
- Full CRUD operations for files and directories
- Encoding support for text files
- Pattern-based file filtering
- Recursive operations
- Error handling with structured responses
- Security-aware file operations

Example Usage:
    >>> from agentoolkit.storage.fs import create_storage_fs_agent
    >>> from agentool.core.injector import get_injector
    >>> 
    >>> # Create and register the agent
    >>> agent = create_storage_fs_agent()
    >>> 
    >>> # Use through injector
    >>> injector = get_injector()
    >>> result = await injector.run('storage_fs', {
    ...     "operation": "write",
    ...     "path": "/tmp/test.txt",
    ...     "content": "Hello, World!"
    ... })
"""

import os
import glob
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from agentool.base import BaseOperationInput
from agentool import create_agentool
from agentool.core.registry import RoutingConfig


class StorageFsInput(BaseOperationInput):
    """Input schema for file system storage operations."""
    operation: Literal['read', 'write', 'append', 'delete', 'list', 'exists', 'mkdir', 'rmdir'] = Field(
        description="The file system operation to perform"
    )
    path: str = Field(description="File or directory path")
    content: Optional[str] = Field(None, description="Content for write/append operations")
    encoding: str = Field(default="utf-8", description="Text encoding for file operations")
    recursive: bool = Field(default=False, description="Recursive operation for list/delete")
    pattern: Optional[str] = Field(None, description="Pattern for file filtering in list operations")
    create_parents: bool = Field(default=True, description="Create parent directories if they don't exist")


class StorageFsOutput(BaseModel):
    """Structured output for file system operations."""
    operation: str = Field(description="The operation that was performed")
    path: str = Field(description="The path that was operated on")
    message: str = Field(description="Human-readable result message")
    data: Optional[Any] = Field(None, description="Operation-specific data")


async def fs_read(ctx: RunContext[Any], path: str, encoding: str) -> StorageFsOutput:
    """
    Read content from a file.
    
    Args:
        ctx: Runtime context
        path: Path to the file to read
        encoding: Text encoding to use
        
    Returns:
        StorageFsOutput with file content in data field
    """
    try:
        file_path = Path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        
        content = file_path.read_text(encoding=encoding)
        
        return StorageFsOutput(
            operation="read",
            path=path,
            message=f"Successfully read {len(content)} characters from {path}",
            data={
                "content": content,
                "size_bytes": file_path.stat().st_size,
                "encoding": encoding
            }
        )
        
    except UnicodeDecodeError as e:
        raise RuntimeError(f"Encoding error reading {path}: {str(e)}") from e
    except PermissionError as e:
        raise PermissionError(f"Permission denied reading {path}") from e
    except FileNotFoundError:
        raise  # Re-raise as-is
    except ValueError:
        raise  # Re-raise as-is
    except Exception as e:
        raise RuntimeError(f"Error reading {path}: {str(e)}") from e


async def fs_write(ctx: RunContext[Any], path: str, content: str, encoding: str, create_parents: bool) -> StorageFsOutput:
    """
    Write content to a file (overwrite if exists).
    
    Args:
        ctx: Runtime context
        path: Path to the file to write
        content: Content to write
        encoding: Text encoding to use
        create_parents: Whether to create parent directories
        
    Returns:
        StorageFsOutput with write operation result
    """
    try:
        file_path = Path(path)
        
        # Create parent directories if needed
        if create_parents and not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_path.write_text(content, encoding=encoding)
        
        return StorageFsOutput(
            operation="write",
            path=path,
            message=f"Successfully wrote {len(content)} characters to {path}",
            data={
                "content_length": len(content),
                "size_bytes": file_path.stat().st_size,
                "encoding": encoding,
                "created_parents": create_parents and not file_path.parent.exists()
            }
        )
        
    except PermissionError as e:
        raise PermissionError(f"Permission denied writing to {path}") from e
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Parent directory does not exist: {path}") from e
    except Exception as e:
        raise RuntimeError(f"Error writing to {path}: {str(e)}") from e


async def fs_append(ctx: RunContext[Any], path: str, content: str, encoding: str, create_parents: bool) -> StorageFsOutput:
    """
    Append content to a file.
    
    Args:
        ctx: Runtime context
        path: Path to the file to append to
        content: Content to append
        encoding: Text encoding to use
        create_parents: Whether to create parent directories
        
    Returns:
        StorageFsOutput with append operation result
    """
    try:
        file_path = Path(path)
        
        # Create parent directories if needed
        if create_parents and not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # If file doesn't exist, create it
        if not file_path.exists():
            file_path.touch()
        
        with open(file_path, 'a', encoding=encoding) as f:
            f.write(content)
        
        return StorageFsOutput(
            operation="append",
            path=path,
            message=f"Successfully appended {len(content)} characters to {path}",
            data={
                "content_length": len(content),
                "total_size_bytes": file_path.stat().st_size,
                "encoding": encoding
            }
        )
        
    except PermissionError as e:
        raise PermissionError(f"Permission denied appending to {path}") from e
    except Exception as e:
        raise RuntimeError(f"Error appending to {path}: {str(e)}") from e


async def fs_delete(ctx: RunContext[Any], path: str, recursive: bool) -> StorageFsOutput:
    """
    Delete a file or directory.
    
    Args:
        ctx: Runtime context
        path: Path to delete
        recursive: Whether to delete directories recursively
        
    Returns:
        StorageFsOutput with delete operation result
    """
    try:
        file_path = Path(path)
        
        if not file_path.exists():
            return StorageFsOutput(
                operation="delete",
                path=path,
                message=f"Path already does not exist: {path}",
                data={"existed": False}
            )
        
        if file_path.is_file():
            file_path.unlink()
            return StorageFsOutput(
                operation="delete",
                path=path,
                message=f"Successfully deleted file: {path}",
                data={"type": "file", "existed": True}
            )
        
        elif file_path.is_dir():
            if recursive:
                shutil.rmtree(file_path)
                return StorageFsOutput(
                    operation="delete",
                    path=path,
                    message=f"Successfully deleted directory recursively: {path}",
                    data={"type": "directory", "recursive": True, "existed": True}
                )
            else:
                try:
                    file_path.rmdir()
                    return StorageFsOutput(
                        operation="delete",
                        path=path,
                        message=f"Successfully deleted empty directory: {path}",
                        data={"type": "directory", "recursive": False, "existed": True}
                    )
                except OSError as e:
                    raise OSError(f"Directory not empty (use recursive=true): {path}") from e
        
    except PermissionError as e:
        raise PermissionError(f"Permission denied deleting {path}") from e
    except OSError:
        raise  # Re-raise OSError from rmdir
    except Exception as e:
        raise RuntimeError(f"Error deleting {path}: {str(e)}") from e


async def fs_list(ctx: RunContext[Any], path: str, recursive: bool, pattern: Optional[str]) -> StorageFsOutput:
    """
    List directory contents.
    
    Args:
        ctx: Runtime context
        path: Directory path to list
        recursive: Whether to list recursively
        pattern: Optional glob pattern for filtering
        
    Returns:
        StorageFsOutput with directory listing in data field
    """
    try:
        dir_path = Path(path)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")
        
        items = []
        
        if recursive:
            # Recursive listing
            if pattern:
                glob_pattern = f"**/{pattern}" if not pattern.startswith('**/') else pattern
                for item_path in dir_path.glob(glob_pattern):
                    relative_path = item_path.relative_to(dir_path)
                    items.append({
                        "name": item_path.name,
                        "path": str(item_path),
                        "relative_path": str(relative_path),
                        "type": "file" if item_path.is_file() else "directory",
                        "size": item_path.stat().st_size if item_path.is_file() else None
                    })
            else:
                for item_path in dir_path.rglob("*"):
                    relative_path = item_path.relative_to(dir_path)
                    items.append({
                        "name": item_path.name,
                        "path": str(item_path),
                        "relative_path": str(relative_path),
                        "type": "file" if item_path.is_file() else "directory",
                        "size": item_path.stat().st_size if item_path.is_file() else None
                    })
        else:
            # Non-recursive listing
            if pattern:
                for item_path in dir_path.glob(pattern):
                    items.append({
                        "name": item_path.name,
                        "path": str(item_path),
                        "type": "file" if item_path.is_file() else "directory",
                        "size": item_path.stat().st_size if item_path.is_file() else None
                    })
            else:
                for item_path in dir_path.iterdir():
                    items.append({
                        "name": item_path.name,
                        "path": str(item_path),
                        "type": "file" if item_path.is_file() else "directory",
                        "size": item_path.stat().st_size if item_path.is_file() else None
                    })
        
        # Sort items by name
        items.sort(key=lambda x: x["name"])
        
        return StorageFsOutput(
            operation="list",
            path=path,
            message=f"Successfully listed {len(items)} items in {path}",
            data={
                "items": items,
                "count": len(items),
                "recursive": recursive,
                "pattern": pattern
            }
        )
        
    except PermissionError as e:
        raise PermissionError(f"Permission denied listing {path}") from e
    except FileNotFoundError:
        raise  # Re-raise
    except ValueError:
        raise  # Re-raise
    except Exception as e:
        raise RuntimeError(f"Error listing {path}: {str(e)}") from e


async def fs_exists(ctx: RunContext[Any], path: str) -> StorageFsOutput:
    """
    Check if a file or directory exists.
    
    Args:
        ctx: Runtime context
        path: Path to check
        
    Returns:
        StorageFsOutput with existence check result
    """
    try:
        file_path = Path(path)
        exists = file_path.exists()
        
        if exists:
            file_type = "file" if file_path.is_file() else "directory"
            size = file_path.stat().st_size if file_path.is_file() else None
            
            return StorageFsOutput(
                operation="exists",
                path=path,
                message=f"Path exists: {path} ({file_type})",
                data={
                    "exists": True,
                    "type": file_type,
                    "size": size
                }
            )
        else:
            return StorageFsOutput(
                operation="exists",
                path=path,
                message=f"Path does not exist: {path}",
                data={"exists": False}
            )
            
    except Exception as e:
        raise RuntimeError(f"Error checking existence of {path}: {str(e)}") from e


async def fs_mkdir(ctx: RunContext[Any], path: str, create_parents: bool) -> StorageFsOutput:
    """
    Create a directory.
    
    Args:
        ctx: Runtime context
        path: Directory path to create
        create_parents: Whether to create parent directories
        
    Returns:
        StorageFsOutput with directory creation result
    """
    try:
        dir_path = Path(path)
        
        if dir_path.exists():
            if dir_path.is_dir():
                return StorageFsOutput(
                    operation="mkdir",
                    path=path,
                    message=f"Directory already exists: {path}",
                    data={"created": False, "existed": True}
                )
            else:
                raise ValueError(f"Path exists but is not a directory: {path}")
        
        dir_path.mkdir(parents=create_parents, exist_ok=True)
        
        return StorageFsOutput(
            operation="mkdir",
            path=path,
            message=f"Successfully created directory: {path}",
            data={
                "created": True,
                "parents_created": create_parents,
                "existed": False
            }
        )
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Parent directory does not exist (use create_parents=true): {path}") from e
    except PermissionError as e:
        raise PermissionError(f"Permission denied creating directory: {path}") from e
    except ValueError:
        raise  # Re-raise
    except Exception as e:
        raise RuntimeError(f"Error creating directory: {path}: {str(e)}") from e


async def fs_rmdir(ctx: RunContext[Any], path: str, recursive: bool) -> StorageFsOutput:
    """
    Remove a directory.
    
    Args:
        ctx: Runtime context
        path: Directory path to remove
        recursive: Whether to remove recursively
        
    Returns:
        StorageFsOutput with directory removal result
    """
    try:
        dir_path = Path(path)
        
        if not dir_path.exists():
            return StorageFsOutput(
                operation="rmdir",
                path=path,
                message=f"Directory already does not exist: {path}",
                data={"removed": False, "existed": False}
            )
        
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")
        
        if recursive:
            shutil.rmtree(dir_path)
            return StorageFsOutput(
                operation="rmdir",
                path=path,
                message=f"Successfully removed directory recursively: {path}",
                data={"removed": True, "recursive": True, "existed": True}
            )
        else:
            try:
                dir_path.rmdir()
                return StorageFsOutput(
                    operation="rmdir",
                    path=path,
                    message=f"Successfully removed empty directory: {path}",
                    data={"removed": True, "recursive": False, "existed": True}
                )
            except OSError as e:
                raise OSError(f"Directory not empty (use recursive=true): {path}") from e
        
    except PermissionError as e:
        raise PermissionError(f"Permission denied removing directory: {path}") from e
    except OSError:
        raise  # Re-raise OSError from rmdir
    except ValueError:
        raise  # Re-raise ValueError
    except Exception as e:
        raise RuntimeError(f"Error removing directory: {path}: {str(e)}") from e


def create_storage_fs_agent():
    """
    Create and return the file system storage AgenTool.
    
    Returns:
        Agent configured for file system operations
    """
    fs_routing = RoutingConfig(
        operation_field='operation',
        operation_map={
            'read': ('fs_read', lambda x: {'path': x.path, 'encoding': x.encoding}),
            'write': ('fs_write', lambda x: {
                'path': x.path, 'content': x.content, 'encoding': x.encoding,
                'create_parents': x.create_parents
            }),
            'append': ('fs_append', lambda x: {
                'path': x.path, 'content': x.content, 'encoding': x.encoding,
                'create_parents': x.create_parents
            }),
            'delete': ('fs_delete', lambda x: {'path': x.path, 'recursive': x.recursive}),
            'list': ('fs_list', lambda x: {
                'path': x.path, 'recursive': x.recursive, 'pattern': x.pattern
            }),
            'exists': ('fs_exists', lambda x: {'path': x.path}),
            'mkdir': ('fs_mkdir', lambda x: {
                'path': x.path, 'create_parents': x.create_parents
            }),
            'rmdir': ('fs_rmdir', lambda x: {'path': x.path, 'recursive': x.recursive}),
        }
    )
    
    return create_agentool(
        name='storage_fs',
        input_schema=StorageFsInput,
        routing_config=fs_routing,
        tools=[fs_read, fs_write, fs_append, fs_delete, fs_list, fs_exists, fs_mkdir, fs_rmdir],
        output_type=StorageFsOutput,
        use_typed_output=True,  # Enable typed output for storage_fs (Tier 1 - no dependencies)
        system_prompt="Handle file system operations efficiently and safely.",
        description="File system storage operations with comprehensive CRUD support",
        version="1.0.0",
        tags=["storage", "filesystem", "files", "directories", "io"],
        examples=[
            {
                "description": "Write content to a file",
                "input": {"operation": "write", "path": "/tmp/test.txt", "content": "Hello, World!"},
                "output": {
                    "success": True, 
                    "operation": "write", 
                    "path": "/tmp/test.txt",
                    "message": "Successfully wrote 13 characters to /tmp/test.txt"
                }
            },
            {
                "description": "Read file content",
                "input": {"operation": "read", "path": "/tmp/test.txt"},
                "output": {
                    "success": True,
                    "operation": "read", 
                    "path": "/tmp/test.txt",
                    "message": "Successfully read 13 characters from /tmp/test.txt",
                    "data": {"content": "Hello, World!", "size_bytes": 13, "encoding": "utf-8"}
                }
            },
            {
                "description": "List directory contents",
                "input": {"operation": "list", "path": "/tmp", "pattern": "*.txt"},
                "output": {
                    "success": True,
                    "operation": "list",
                    "path": "/tmp", 
                    "message": "Successfully listed 1 items in /tmp",
                    "data": {"items": [{"name": "test.txt", "type": "file"}], "count": 1}
                }
            }
        ]
    )


# Create the agent instance
agent = create_storage_fs_agent()