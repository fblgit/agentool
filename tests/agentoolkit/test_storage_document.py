"""
Tests for storage_document toolkit.

This module tests all functionality of the document storage toolkit
including document CRUD operations, chunking, metadata management, and KV storage integration.
"""

import json
import asyncio

from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry


class TestStorageDocument:
    """Test suite for storage_document toolkit."""
    
    def setup_method(self):
        """Clear registry and injector before each test."""
        AgenToolRegistry.clear()
        get_injector().clear()
        
        # Import and create agents
        from agentoolkit.storage.document import create_storage_document_agent
        from agentoolkit.storage.kv import create_storage_kv_agent, _kv_storage, _kv_expiry
        
        # Clear KV storage
        _kv_storage.clear()
        _kv_expiry.clear()
        
        # Register agents
        kv_agent = create_storage_kv_agent()
        doc_agent = create_storage_document_agent()
    
    def test_document_store_and_retrieve(self):
        """Test document storage and retrieval."""
        
        async def run_test():
            injector = get_injector()
            
            content = "This is a test document with some sample content for testing."
            metadata = {"title": "Test Document", "author": "Test Suite"}
            
            # Store document
            store_result = await injector.run('storage_document', {
                "operation": "store",
                "content": content,
                "metadata": metadata
            })
            
            if hasattr(store_result, 'output'):
                store_data = json.loads(store_result.output)
            else:
                store_data = store_result
            
            assert store_data.get("success") is True
            assert "Document stored successfully" in store_data.get("message", "")
            doc_id = store_data.get("data", {}).get("doc_id")
            assert doc_id is not None
            
            # Retrieve document
            retrieve_result = await injector.run('storage_document', {
                "operation": "retrieve",
                "doc_id": doc_id
            })
            
            if hasattr(retrieve_result, 'output'):
                retrieve_data = json.loads(retrieve_result.output)
            else:
                retrieve_data = retrieve_result
            
            assert retrieve_data.get("success") is True
            assert retrieve_data.get("data", {}).get("content") == content
            assert retrieve_data.get("data", {}).get("metadata", {}).get("title") == "Test Document"
        
        asyncio.run(run_test())
    
    def test_document_chunking(self):
        """Test document chunking functionality."""
        
        async def run_test():
            injector = get_injector()
            
            # Create a longer document
            content = " ".join(["Word" + str(i) for i in range(1000)])
            
            # Store document
            store_result = await injector.run('storage_document', {
                "operation": "store",
                "content": content,
                "doc_id": "chunk_test"
            })
            
            # Chunk the document
            chunk_result = await injector.run('storage_document', {
                "operation": "chunk",
                "doc_id": "chunk_test",
                "chunk_size": 100,
                "chunk_overlap": 20
            })
            
            if hasattr(chunk_result, 'output'):
                chunk_data = json.loads(chunk_result.output)
            else:
                chunk_data = chunk_result
            
            assert chunk_data.get("success") is True
            chunks = chunk_data.get("data", {}).get("chunks", [])
            assert len(chunks) > 0
            
            # Verify chunk structure
            first_chunk = chunks[0]
            assert "chunk_id" in first_chunk
            assert "content" in first_chunk
            assert "chunk_index" in first_chunk
            assert first_chunk["doc_id"] == "chunk_test"
            
            # Verify chunk size (approximate due to word boundaries)
            chunk_words = first_chunk["content"].split()
            assert 80 <= len(chunk_words) <= 120  # Allow some flexibility
            
            # Verify overlap exists between chunks if multiple chunks
            if len(chunks) > 1:
                chunk1_words = chunks[0]["content"].split()
                chunk2_words = chunks[1]["content"].split()
                # Check for some overlap
                overlap = set(chunk1_words[-20:]) & set(chunk2_words[:20])
                assert len(overlap) > 0
        
        asyncio.run(run_test())
    
    def test_document_list(self):
        """Test listing documents."""
        
        async def run_test():
            injector = get_injector()
            
            # Store multiple documents
            doc_ids = []
            for i in range(3):
                result = await injector.run('storage_document', {
                    "operation": "store",
                    "doc_id": f"list_test_{i}",
                    "content": f"Document {i} content",
                    "metadata": {"index": i}
                })
                
                if hasattr(result, 'output'):
                    data = json.loads(result.output)
                else:
                    data = result
                doc_ids.append(data.get("data", {}).get("doc_id"))
            
            # List all documents
            list_result = await injector.run('storage_document', {
                "operation": "list",
                "limit": 100
            })
            
            if hasattr(list_result, 'output'):
                list_data = json.loads(list_result.output)
            else:
                list_data = list_result
            
            assert list_data.get("success") is True
            documents = list_data.get("data", {}).get("documents", [])
            
            # Check our documents are in the list
            found_ids = [doc["doc_id"] for doc in documents]
            for doc_id in doc_ids:
                assert doc_id in found_ids
            
            # Test with prefix filter
            prefix_result = await injector.run('storage_document', {
                "operation": "list",
                "prefix": "list_test",
                "limit": 10
            })
            
            if hasattr(prefix_result, 'output'):
                prefix_data = json.loads(prefix_result.output)
            else:
                prefix_data = prefix_result
            
            prefix_docs = prefix_data.get("data", {}).get("documents", [])
            assert len(prefix_docs) >= 3
            for doc in prefix_docs:
                assert doc["doc_id"].startswith("list_test")
        
        asyncio.run(run_test())
    
    def test_document_delete(self):
        """Test document deletion."""
        
        async def run_test():
            injector = get_injector()
            
            # Store and chunk a document
            await injector.run('storage_document', {
                "operation": "store",
                "doc_id": "delete_test",
                "content": "Content to be deleted " * 50
            })
            
            await injector.run('storage_document', {
                "operation": "chunk",
                "doc_id": "delete_test",
                "chunk_size": 50,
                "chunk_overlap": 10
            })
            
            # Delete the document
            delete_result = await injector.run('storage_document', {
                "operation": "delete",
                "doc_id": "delete_test"
            })
            
            if hasattr(delete_result, 'output'):
                delete_data = json.loads(delete_result.output)
            else:
                delete_data = delete_result
            
            assert delete_data.get("success") is True
            assert "Document deleted" in delete_data.get("message", "")
            
            # Verify document is deleted
            try:
                retrieve_result = await injector.run('storage_document', {
                    "operation": "retrieve",
                    "doc_id": "delete_test"
                })
                
                if hasattr(retrieve_result, 'output'):
                    retrieve_data = json.loads(retrieve_result.output)
                else:
                    retrieve_data = retrieve_result
                
                assert False, "Expected KeyError for deleted document"
            except KeyError as e:
                assert "not found" in str(e)
        
        asyncio.run(run_test())
    
    def test_document_update_metadata(self):
        """Test updating document metadata."""
        
        async def run_test():
            injector = get_injector()
            
            # Store a document
            await injector.run('storage_document', {
                "operation": "store",
                "doc_id": "meta_test",
                "content": "Document with metadata",
                "metadata": {"version": 1, "status": "draft"}
            })
            
            # Update metadata
            update_result = await injector.run('storage_document', {
                "operation": "update_metadata",
                "doc_id": "meta_test",
                "metadata": {"status": "published", "reviewer": "Alice"}
            })
            
            if hasattr(update_result, 'output'):
                update_data = json.loads(update_result.output)
            else:
                update_data = update_result
            
            assert update_data.get("success") is True
            
            # Retrieve and verify metadata
            retrieve_result = await injector.run('storage_document', {
                "operation": "retrieve",
                "doc_id": "meta_test"
            })
            
            if hasattr(retrieve_result, 'output'):
                retrieve_data = json.loads(retrieve_result.output)
            else:
                retrieve_data = retrieve_result
            
            metadata = retrieve_data.get("data", {}).get("metadata", {})
            assert metadata.get("version") == 1  # Original field preserved
            assert metadata.get("status") == "published"  # Updated
            assert metadata.get("reviewer") == "Alice"  # New field added
            assert "updated_at" in metadata  # Auto-added timestamp
        
        asyncio.run(run_test())
    
    def test_document_auto_id_generation(self):
        """Test automatic document ID generation."""
        
        async def run_test():
            injector = get_injector()
            
            content = "Document without explicit ID"
            
            # Store without doc_id
            result = await injector.run('storage_document', {
                "operation": "store",
                "content": content
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert data.get("success") is True
            doc_id = data.get("data", {}).get("doc_id")
            assert doc_id is not None
            assert len(doc_id) == 16  # Hash-based ID length
            
            # Verify same content generates same ID
            result2 = await injector.run('storage_document', {
                "operation": "store",
                "content": content
            })
            
            if hasattr(result2, 'output'):
                data2 = json.loads(result2.output)
            else:
                data2 = result2
            
            doc_id2 = data2.get("data", {}).get("doc_id")
            assert doc_id == doc_id2  # Same content, same ID
        
        asyncio.run(run_test())
    
    def test_document_error_handling(self):
        """Test error handling for document operations."""
        
        async def run_test():
            injector = get_injector()
            
            # Test retrieve non-existent document
            try:
                await injector.run('storage_document', {
                    "operation": "retrieve",
                    "doc_id": "non_existent_xyz"
                })
                assert False, "Expected KeyError for non-existent document"
            except KeyError as e:
                assert "not found" in str(e)
            
            # Test chunk non-existent document
            try:
                await injector.run('storage_document', {
                    "operation": "chunk",
                    "doc_id": "non_existent_xyz",
                    "chunk_size": 100
                })
                assert False, "Expected error for chunking non-existent document"
            except (KeyError, RuntimeError) as e:
                assert "not found" in str(e) or "Failed" in str(e)
        
        asyncio.run(run_test())