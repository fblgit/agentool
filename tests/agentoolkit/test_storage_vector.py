"""
Tests for storage_vector toolkit.

This module tests all functionality of the vector storage toolkit
including collection initialization, vector operations, similarity search, and pgvector integration.
"""

import json
import asyncio
import pytest

from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry


class TestStorageVector:
    """Test suite for storage_vector toolkit."""
    
    def setup_method(self):
        """Clear registry and injector before each test."""
        AgenToolRegistry.clear()
        get_injector().clear()
        
        # Import and create the agent
        from agentoolkit.storage.vector import create_storage_vector_agent, _pool
        
        # Clear connection pool
        if _pool is not None:
            asyncio.run(_pool.close())
            from agentoolkit.storage import vector
            vector._pool = None
        
        # Register agent
        agent = create_storage_vector_agent()
    
    def teardown_method(self):
        """Clean up resources after each test."""
        # Simple cleanup - just clear references
        from agentoolkit.storage import vector
        
        # Clear pool reference - let garbage collection handle the rest
        vector._pool = None
    
    def test_vector_init_collection(self):
        """Test collection initialization."""
        
        async def run_test():
            injector = get_injector()
            collection = "test_collection"
            
            # Initialize collection
            result = await injector.run('storage_vector', {
                "operation": "init_collection",
                "collection": collection
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert data.get("success") is True
            assert f"Collection '{collection}' initialized" in data.get("message", "")
            assert data.get("data", {}).get("collection") == collection
        
        asyncio.run(run_test())
    
    def test_vector_upsert_and_search(self):
        """Test vector upsert and similarity search."""
        
        async def run_test():
            injector = get_injector()
            collection = "test_search"
            
            # Initialize collection first
            await injector.run('storage_vector', {
                "operation": "init_collection",
                "collection": collection
            })
            
            # Prepare test data
            embeddings = [
                [0.1] * 1536,  # OpenAI embedding dimension
                [0.2] * 1536,
                [0.3] * 1536
            ]
            ids = ["vec1", "vec2", "vec3"]
            metadata = [
                {"type": "doc", "category": "A"},
                {"type": "doc", "category": "B"},
                {"type": "doc", "category": "A"}
            ]
            contents = ["First document", "Second document", "Third document"]
            
            # Upsert vectors
            upsert_result = await injector.run('storage_vector', {
                "operation": "upsert",
                "collection": collection,
                "embeddings": embeddings,
                "ids": ids,
                "metadata": metadata,
                "contents": contents
            })
            
            if hasattr(upsert_result, 'output'):
                upsert_data = json.loads(upsert_result.output)
            else:
                upsert_data = upsert_result
            
            assert upsert_data.get("success") is True
            assert upsert_data.get("count") == 3
            
            # Search for similar vectors
            query_embedding = [0.15] * 1536  # Should be closest to vec1
            
            search_result = await injector.run('storage_vector', {
                "operation": "search",
                "collection": collection,
                "query_embedding": query_embedding,
                "top_k": 2
            })
            
            if hasattr(search_result, 'output'):
                search_data = json.loads(search_result.output)
            else:
                search_data = search_result
            
            assert search_data.get("success") is True
            assert search_data.get("count") == 2
            results = search_data.get("data", {}).get("results", [])
            assert len(results) == 2
            
            # Check first result is most similar
            if results:
                assert results[0]["id"] == "vec1"  # Closest to query
                assert "similarity" in results[0]
                assert results[0]["content"] == "First document"
        
        asyncio.run(run_test())
    
    def test_vector_delete(self):
        """Test vector deletion."""
        
        async def run_test():
            injector = get_injector()
            collection = "test_delete"
            
            # Initialize and add vectors
            await injector.run('storage_vector', {
                "operation": "init_collection",
                "collection": collection
            })
            
            embeddings = [[0.1] * 1536, [0.2] * 1536]
            ids = ["del1", "del2"]
            
            await injector.run('storage_vector', {
                "operation": "upsert",
                "collection": collection,
                "embeddings": embeddings,
                "ids": ids
            })
            
            # Delete one vector
            delete_result = await injector.run('storage_vector', {
                "operation": "delete",
                "collection": collection,
                "doc_ids": ["del1"]
            })
            
            if hasattr(delete_result, 'output'):
                delete_data = json.loads(delete_result.output)
            else:
                delete_data = delete_result
            
            assert delete_data.get("success") is True
            assert delete_data.get("count") >= 1
            
            # Verify deletion with search
            search_result = await injector.run('storage_vector', {
                "operation": "search",
                "collection": collection,
                "query_embedding": [0.1] * 1536,
                "top_k": 10
            })
            
            if hasattr(search_result, 'output'):
                search_data = json.loads(search_result.output)
            else:
                search_data = search_result
            
            results = search_data.get("data", {}).get("results", [])
            # Should not find del1
            for result in results:
                assert result["id"] != "del1"
        
        asyncio.run(run_test())
    
    def test_vector_list_collections(self):
        """Test listing collections."""
        
        async def run_test():
            injector = get_injector()
            
            # Create multiple collections
            collections = ["list_test1", "list_test2", "list_test3"]
            for col in collections:
                await injector.run('storage_vector', {
                    "operation": "init_collection",
                    "collection": col
                })
            
            # List collections
            list_result = await injector.run('storage_vector', {
                "operation": "list_collections"
            })
            
            if hasattr(list_result, 'output'):
                list_data = json.loads(list_result.output)
            else:
                list_data = list_result
            
            assert list_data.get("success") is True
            found_collections = list_data.get("data", {}).get("collections", [])
            
            # Check our test collections are listed
            for col in collections:
                assert col in found_collections
        
        asyncio.run(run_test())
    
    def test_vector_metadata_filter(self):
        """Test search with metadata filtering."""
        
        async def run_test():
            injector = get_injector()
            collection = "test_filter"
            
            # Initialize collection
            await injector.run('storage_vector', {
                "operation": "init_collection",
                "collection": collection
            })
            
            # Add vectors with different metadata
            embeddings = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
            ids = ["f1", "f2", "f3"]
            metadata = [
                {"category": "tech", "year": 2023},
                {"category": "science", "year": 2024},
                {"category": "tech", "year": 2024}
            ]
            
            await injector.run('storage_vector', {
                "operation": "upsert",
                "collection": collection,
                "embeddings": embeddings,
                "ids": ids,
                "metadata": metadata
            })
            
            # Search with filter (basic implementation)
            # TODO: This test may need adjustment based on filter implementation
            search_result = await injector.run('storage_vector', {
                "operation": "search",
                "collection": collection,
                "query_embedding": [0.2] * 1536,
                "top_k": 10,
                "filter": {"category": "tech"}
            })
            
            if hasattr(search_result, 'output'):
                search_data = json.loads(search_result.output)
            else:
                search_data = search_result
            
            # Basic check - filter implementation is TODO Phase 2
            assert search_data.get("success") is True
        
        asyncio.run(run_test())
    
    def test_vector_error_handling(self):
        """Test error handling for invalid operations."""
        
        async def run_test():
            injector = get_injector()
            
            # Test mismatched embeddings and IDs
            try:
                await injector.run('storage_vector', {
                    "operation": "upsert",
                    "collection": "test_error",
                    "embeddings": [[0.1] * 1536],
                    "ids": ["id1", "id2"]  # Mismatch
                })
                assert False, "Expected ValueError for mismatched embeddings and IDs"
            except ValueError as e:
                assert "must match" in str(e)
            
            # Test search on non-existent collection
            try:
                await injector.run('storage_vector', {
                    "operation": "search",
                    "collection": "non_existent_xyz",
                    "query_embedding": [0.1] * 1536,
                    "top_k": 5
                })
                # May succeed or fail depending on DB state
            except RuntimeError:
                pass  # Expected
        
        asyncio.run(run_test())


# Note: These tests require PostgreSQL with pgvector running
# Run: docker run --rm -e POSTGRES_PASSWORD=postgres -p 54320:5432 pgvector/pgvector:pg17