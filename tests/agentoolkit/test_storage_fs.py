"""
Tests for storage_fs toolkit.

This module tests all functionality of the file system storage toolkit
including file operations, directory management, error handling, and edge cases.
"""

import json
import os
import tempfile
import asyncio
from pathlib import Path

from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry


class TestStorageFs:
    """Test suite for storage_fs toolkit."""
    
    def setup_method(self):
        """Clear registry and injector before each test."""
        AgenToolRegistry.clear()
        get_injector().clear()
        
        # Import and create the agent
        from agentoolkit.storage.fs import create_storage_fs_agent
        agent = create_storage_fs_agent()
    
    def test_fs_write_and_read(self):
        """Test basic write and read operations."""
        
        async def run_test():
            with tempfile.TemporaryDirectory() as temp_dir:
                injector = get_injector()
                test_file = os.path.join(temp_dir, "test.txt")
                test_content = "Hello, World!"
                
                # Test write
                write_result = await injector.run('storage_fs', {
                    "operation": "write",
                    "path": test_file,
                    "content": test_content
                })
                
                # Verify write result
                if hasattr(write_result, 'output'):
                    write_data = json.loads(write_result.output)
                else:
                    write_data = write_result
                
                # No longer checking success field - function now throws exceptions on failure
                assert write_data["operation"] == "write"
                assert write_data["path"] == test_file
                assert write_data["data"]["content_length"] == len(test_content)
                
                # Test read
                read_result = await injector.run('storage_fs', {
                    "operation": "read",
                    "path": test_file
                })
                
                # Verify read result
                if hasattr(read_result, 'output'):
                    read_data = json.loads(read_result.output)
                else:
                    read_data = read_result
                
                # No longer checking success field - function now throws exceptions on failure
                assert read_data["operation"] == "read"
                assert read_data["data"]["content"] == test_content
                assert read_data["data"]["encoding"] == "utf-8"
        
        asyncio.run(run_test())
    
    def test_fs_append(self):
        """Test append operation."""
        
        async def run_test():
            with tempfile.TemporaryDirectory() as temp_dir:
                injector = get_injector()
                test_file = os.path.join(temp_dir, "append_test.txt")
                initial_content = "Line 1\n"
                additional_content = "Line 2\n"
                
                # Write initial content
                await injector.run('storage_fs', {
                    "operation": "write",
                    "path": test_file,
                    "content": initial_content
                })
                
                # Append additional content
                append_result = await injector.run('storage_fs', {
                    "operation": "append",
                    "path": test_file,
                    "content": additional_content
                })
                
                # Verify append result
                if hasattr(append_result, 'output'):
                    append_data = json.loads(append_result.output)
                else:
                    append_data = append_result
                
                # No longer checking success field - function now throws exceptions on failure
                assert append_data["operation"] == "append"
                
                # Read and verify combined content
                read_result = await injector.run('storage_fs', {
                    "operation": "read",
                    "path": test_file
                })
                
                if hasattr(read_result, 'output'):
                    read_data = json.loads(read_result.output)
                else:
                    read_data = read_result
                
                assert read_data["data"]["content"] == initial_content + additional_content
        
        asyncio.run(run_test())
    
    def test_fs_delete(self):
        """Test delete operation."""
        
        async def run_test():
            with tempfile.TemporaryDirectory() as temp_dir:
                injector = get_injector()
                test_file = os.path.join(temp_dir, "delete_test.txt")
                
                # Create a file
                await injector.run('storage_fs', {
                    "operation": "write",
                    "path": test_file,
                    "content": "To be deleted"
                })
                
                # Verify file exists
                exists_result = await injector.run('storage_fs', {
                    "operation": "exists",
                    "path": test_file
                })
                
                if hasattr(exists_result, 'output'):
                    exists_data = json.loads(exists_result.output)
                else:
                    exists_data = exists_result
                
                assert exists_data["data"]["exists"] is True
                
                # Delete the file
                delete_result = await injector.run('storage_fs', {
                    "operation": "delete",
                    "path": test_file
                })
                
                if hasattr(delete_result, 'output'):
                    delete_data = json.loads(delete_result.output)
                else:
                    delete_data = delete_result
                
                # No longer checking success field - function now throws exceptions on failure
                assert delete_data["data"]["existed"] is True
                assert delete_data["data"]["type"] == "file"
                
                # Verify file no longer exists
                exists_result = await injector.run('storage_fs', {
                    "operation": "exists",
                    "path": test_file
                })
                
                if hasattr(exists_result, 'output'):
                    exists_data = json.loads(exists_result.output)
                else:
                    exists_data = exists_result
                
                assert exists_data["data"]["exists"] is False
        
        asyncio.run(run_test())
    
    def test_fs_mkdir_and_rmdir(self):
        """Test directory creation and removal."""
        
        async def run_test():
            with tempfile.TemporaryDirectory() as temp_dir:
                injector = get_injector()
                test_dir = os.path.join(temp_dir, "test_directory")
                
                # Create directory
                mkdir_result = await injector.run('storage_fs', {
                    "operation": "mkdir",
                    "path": test_dir
                })
                
                if hasattr(mkdir_result, 'output'):
                    mkdir_data = json.loads(mkdir_result.output)
                else:
                    mkdir_data = mkdir_result
                
                # No longer checking success field - function now throws exceptions on failure
                assert mkdir_data["data"]["created"] is True
                
                # Verify directory exists
                exists_result = await injector.run('storage_fs', {
                    "operation": "exists",
                    "path": test_dir
                })
                
                if hasattr(exists_result, 'output'):
                    exists_data = json.loads(exists_result.output)
                else:
                    exists_data = exists_result
                
                assert exists_data["data"]["exists"] is True
                assert exists_data["data"]["type"] == "directory"
                
                # Remove directory
                rmdir_result = await injector.run('storage_fs', {
                    "operation": "rmdir",
                    "path": test_dir
                })
                
                if hasattr(rmdir_result, 'output'):
                    rmdir_data = json.loads(rmdir_result.output)
                else:
                    rmdir_data = rmdir_result
                
                # No longer checking success field - function now throws exceptions on failure
                assert rmdir_data["data"]["removed"] is True
        
        asyncio.run(run_test())
    
    def test_fs_list(self):
        """Test directory listing operations."""
        
        async def run_test():
            with tempfile.TemporaryDirectory() as temp_dir:
                injector = get_injector()
                
                # Create some test files
                files = ["file1.txt", "file2.py", "file3.md"]
                for filename in files:
                    await injector.run('storage_fs', {
                        "operation": "write",
                        "path": os.path.join(temp_dir, filename),
                        "content": f"Content of {filename}"
                    })
                
                # Create a subdirectory
                subdir = os.path.join(temp_dir, "subdir")
                await injector.run('storage_fs', {
                    "operation": "mkdir",
                    "path": subdir
                })
                
                # List all files
                list_result = await injector.run('storage_fs', {
                    "operation": "list",
                    "path": temp_dir
                })
                
                if hasattr(list_result, 'output'):
                    list_data = json.loads(list_result.output)
                else:
                    list_data = list_result
                
                # No longer checking success field - function now throws exceptions on failure
                assert list_data["data"]["count"] >= len(files) + 1  # files + subdir
                
                # Check that all files are listed
                item_names = [item["name"] for item in list_data["data"]["items"]]
                for filename in files:
                    assert filename in item_names
                assert "subdir" in item_names
                
                # Test pattern matching
                pattern_result = await injector.run('storage_fs', {
                    "operation": "list",
                    "path": temp_dir,
                    "pattern": "*.txt"
                })
                
                if hasattr(pattern_result, 'output'):
                    pattern_data = json.loads(pattern_result.output)
                else:
                    pattern_data = pattern_result
                
                # No longer checking success field - function now throws exceptions on failure
                txt_files = [item["name"] for item in pattern_data["data"]["items"]]
                assert "file1.txt" in txt_files
                assert "file2.py" not in txt_files
        
        asyncio.run(run_test())
    
    def test_fs_error_handling(self):
        """Test error handling for various edge cases."""
        
        async def run_test():
            injector = get_injector()
            
            # Test reading non-existent file - should raise FileNotFoundError
            try:
                read_result = await injector.run('storage_fs', {
                    "operation": "read",
                    "path": "/nonexistent/file.txt"
                })
                assert False, "Should have thrown FileNotFoundError"
            except FileNotFoundError as e:
                assert "not found" in str(e).lower()
            
            # Test listing non-existent directory - should raise FileNotFoundError
            try:
                list_result = await injector.run('storage_fs', {
                    "operation": "list",
                    "path": "/nonexistent/directory"
                })
                assert False, "Should have thrown FileNotFoundError"
            except FileNotFoundError as e:
                assert "not found" in str(e).lower()
            
            # Test deleting non-existent file (should be idempotent)
            delete_result = await injector.run('storage_fs', {
                "operation": "delete",
                "path": "/nonexistent/file.txt"
            })
            
            if hasattr(delete_result, 'output'):
                delete_data = json.loads(delete_result.output)
            else:
                delete_data = delete_result
            
            # No longer checking success field - function now throws exceptions on failure  # Idempotent
            assert delete_data["data"]["existed"] is False
        
        asyncio.run(run_test())
    
    def test_fs_recursive_operations(self):
        """Test recursive directory operations."""
        
        async def run_test():
            with tempfile.TemporaryDirectory() as temp_dir:
                injector = get_injector()
                
                # Create nested directory structure
                nested_dir = os.path.join(temp_dir, "parent", "child")
                await injector.run('storage_fs', {
                    "operation": "mkdir",
                    "path": nested_dir,
                    "create_parents": True
                })
                
                # Create files in nested structure
                await injector.run('storage_fs', {
                    "operation": "write",
                    "path": os.path.join(temp_dir, "parent", "parent_file.txt"),
                    "content": "Parent file"
                })
                
                await injector.run('storage_fs', {
                    "operation": "write",
                    "path": os.path.join(nested_dir, "child_file.txt"),
                    "content": "Child file"
                })
                
                # Test recursive listing
                list_result = await injector.run('storage_fs', {
                    "operation": "list",
                    "path": temp_dir,
                    "recursive": True
                })
                
                if hasattr(list_result, 'output'):
                    list_data = json.loads(list_result.output)
                else:
                    list_data = list_result
                
                # No longer checking success field - function now throws exceptions on failure
                
                # Check that both files are found
                all_files = [item["name"] for item in list_data["data"]["items"]]
                assert "parent_file.txt" in all_files
                assert "child_file.txt" in all_files
                
                # Test recursive delete
                delete_result = await injector.run('storage_fs', {
                    "operation": "delete",
                    "path": os.path.join(temp_dir, "parent"),
                    "recursive": True
                })
                
                if hasattr(delete_result, 'output'):
                    delete_data = json.loads(delete_result.output)
                else:
                    delete_data = delete_result
                
                # No longer checking success field - function now throws exceptions on failure
                assert delete_data["data"]["type"] == "directory"
                assert delete_data["data"]["recursive"] is True
                
                # Verify deletion
                exists_result = await injector.run('storage_fs', {
                    "operation": "exists",
                    "path": os.path.join(temp_dir, "parent")
                })
                
                if hasattr(exists_result, 'output'):
                    exists_data = json.loads(exists_result.output)
                else:
                    exists_data = exists_result
                
                assert exists_data["data"]["exists"] is False
        
        asyncio.run(run_test())
    
    def test_fs_encoding_support(self):
        """Test different text encodings."""
        
        async def run_test():
            with tempfile.TemporaryDirectory() as temp_dir:
                injector = get_injector()
                test_file = os.path.join(temp_dir, "encoding_test.txt")
                
                # Test UTF-8 with special characters
                utf8_content = "Hello 世界! 🌍 Ñoño"
                
                write_result = await injector.run('storage_fs', {
                    "operation": "write",
                    "path": test_file,
                    "content": utf8_content,
                    "encoding": "utf-8"
                })
                
                if hasattr(write_result, 'output'):
                    write_data = json.loads(write_result.output)
                else:
                    write_data = write_result
                
                # No longer checking success field - function now throws exceptions on failure
                
                # Read back with UTF-8
                read_result = await injector.run('storage_fs', {
                    "operation": "read",
                    "path": test_file,
                    "encoding": "utf-8"
                })
                
                if hasattr(read_result, 'output'):
                    read_data = json.loads(read_result.output)
                else:
                    read_data = read_result
                
                # No longer checking success field - function now throws exceptions on failure
                assert read_data["data"]["content"] == utf8_content
        
        asyncio.run(run_test())
    
    def test_fs_idempotent_operations(self):
        """Test that operations are idempotent where appropriate."""
        
        async def run_test():
            with tempfile.TemporaryDirectory() as temp_dir:
                injector = get_injector()
                test_file = os.path.join(temp_dir, "idempotent_test.txt")
                test_dir = os.path.join(temp_dir, "idempotent_dir")
                
                # Test mkdir idempotency
                mkdir_result1 = await injector.run('storage_fs', {
                    "operation": "mkdir",
                    "path": test_dir
                })
                
                mkdir_result2 = await injector.run('storage_fs', {
                    "operation": "mkdir",
                    "path": test_dir
                })
                
                if hasattr(mkdir_result1, 'output'):
                    mkdir_data1 = json.loads(mkdir_result1.output)
                else:
                    mkdir_data1 = mkdir_result1
                    
                if hasattr(mkdir_result2, 'output'):
                    mkdir_data2 = json.loads(mkdir_result2.output)
                else:
                    mkdir_data2 = mkdir_result2
                
                # No longer checking success field - function now throws exceptions on failure
                # No longer checking success field - function now throws exceptions on failure
                assert mkdir_data1["data"]["created"] is True
                assert mkdir_data2["data"]["created"] is False  # Already existed
                
                # Test delete idempotency
                delete_result1 = await injector.run('storage_fs', {
                    "operation": "delete",
                    "path": test_file
                })
                
                delete_result2 = await injector.run('storage_fs', {
                    "operation": "delete",
                    "path": test_file
                })
                
                if hasattr(delete_result1, 'output'):
                    delete_data1 = json.loads(delete_result1.output)
                else:
                    delete_data1 = delete_result1
                    
                if hasattr(delete_result2, 'output'):
                    delete_data2 = json.loads(delete_result2.output)
                else:
                    delete_data2 = delete_result2
                
                # No longer checking success field - function now throws exceptions on failure
                # No longer checking success field - function now throws exceptions on failure
                assert delete_data1["data"]["existed"] is False  # Never existed
                assert delete_data2["data"]["existed"] is False  # Still doesn't exist
        
        asyncio.run(run_test())