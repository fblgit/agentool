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
                
                # Verify write result - storage_fs returns typed StorageFsOutput
                assert write_result.operation == "write"
                assert write_result.path == test_file
                assert write_result.data["content_length"] == len(test_content)
                
                # Test read
                read_result = await injector.run('storage_fs', {
                    "operation": "read",
                    "path": test_file
                })
                
                # Verify read result - storage_fs returns typed StorageFsOutput
                assert read_result.operation == "read"
                assert read_result.data["content"] == test_content
                assert read_result.data["encoding"] == "utf-8"
        
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
                
                # Verify append result - storage_fs returns typed StorageFsOutput
                assert append_result.operation == "append"
                
                # Read and verify combined content
                read_result = await injector.run('storage_fs', {
                    "operation": "read",
                    "path": test_file
                })
                
                # storage_fs returns typed StorageFsOutput
                assert read_result.data["content"] == initial_content + additional_content
        
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
                
                # storage_fs returns typed StorageFsOutput
                assert exists_result.data["exists"] is True
                
                # Delete the file
                delete_result = await injector.run('storage_fs', {
                    "operation": "delete",
                    "path": test_file
                })
                
                # storage_fs returns typed StorageFsOutput
                assert delete_result.data["existed"] is True
                assert delete_result.data["type"] == "file"
                
                # Verify file no longer exists
                exists_result = await injector.run('storage_fs', {
                    "operation": "exists",
                    "path": test_file
                })
                
                # storage_fs returns typed StorageFsOutput
                assert exists_result.data["exists"] is False
        
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
                
                # storage_fs returns typed StorageFsOutput
                assert mkdir_result.data["created"] is True
                
                # Verify directory exists
                exists_result = await injector.run('storage_fs', {
                    "operation": "exists",
                    "path": test_dir
                })
                
                # storage_fs returns typed StorageFsOutput
                assert exists_result.data["exists"] is True
                assert exists_result.data["type"] == "directory"
                
                # Remove directory
                rmdir_result = await injector.run('storage_fs', {
                    "operation": "rmdir",
                    "path": test_dir
                })
                
                # storage_fs returns typed StorageFsOutput
                assert rmdir_result.data["removed"] is True
        
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
                
                # storage_fs returns typed StorageFsOutput
                assert list_result.data["count"] >= len(files) + 1  # files + subdir
                
                # Check that all files are listed
                item_names = [item["name"] for item in list_result.data["items"]]
                for filename in files:
                    assert filename in item_names
                assert "subdir" in item_names
                
                # Test pattern matching
                pattern_result = await injector.run('storage_fs', {
                    "operation": "list",
                    "path": temp_dir,
                    "pattern": "*.txt"
                })
                
                # storage_fs returns typed StorageFsOutput
                txt_files = [item["name"] for item in pattern_result.data["items"]]
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
            
            # storage_fs returns typed StorageFsOutput
            assert delete_result.data["existed"] is False
        
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
                
                # storage_fs returns typed StorageFsOutput
                # Check that both files are found
                all_files = [item["name"] for item in list_result.data["items"]]
                assert "parent_file.txt" in all_files
                assert "child_file.txt" in all_files
                
                # Test recursive delete
                delete_result = await injector.run('storage_fs', {
                    "operation": "delete",
                    "path": os.path.join(temp_dir, "parent"),
                    "recursive": True
                })
                
                # storage_fs returns typed StorageFsOutput
                assert delete_result.data["type"] == "directory"
                assert delete_result.data["recursive"] is True
                
                # Verify deletion
                exists_result = await injector.run('storage_fs', {
                    "operation": "exists",
                    "path": os.path.join(temp_dir, "parent")
                })
                
                # storage_fs returns typed StorageFsOutput
                assert exists_result.data["exists"] is False
        
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
                
                # storage_fs returns typed StorageFsOutput
                # No longer checking success field - function now throws exceptions on failure
                
                # Read back with UTF-8
                read_result = await injector.run('storage_fs', {
                    "operation": "read",
                    "path": test_file,
                    "encoding": "utf-8"
                })
                
                # storage_fs returns typed StorageFsOutput
                assert read_result.data["content"] == utf8_content
        
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
                
                # storage_fs returns typed StorageFsOutput
                assert mkdir_result1.data["created"] is True
                assert mkdir_result2.data["created"] is False  # Already existed
                
                # Test delete idempotency
                delete_result1 = await injector.run('storage_fs', {
                    "operation": "delete",
                    "path": test_file
                })
                
                delete_result2 = await injector.run('storage_fs', {
                    "operation": "delete",
                    "path": test_file
                })
                
                # storage_fs returns typed StorageFsOutput
                assert delete_result1.data["existed"] is False  # Never existed
                assert delete_result2.data["existed"] is False  # Still doesn't exist
        
        asyncio.run(run_test())