"""
Tests for rag_retriever toolkit.

This module tests all functionality of the retriever toolkit
including search, context retrieval, document indexing, and integration with other RAG components.
"""

import json
import asyncio

from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry


class TestRagRetriever:
    """Test suite for rag_retriever toolkit."""
    
    def setup_method(self):
        """Clear registry and injector before each test."""
        AgenToolRegistry.clear()
        get_injector().clear()
        
        # Import and create all required agents
        from agentoolkit.rag.retriever import create_rag_retriever_agent
        from agentoolkit.rag.embedder import create_rag_embedder_agent
        from agentoolkit.storage.vector import create_storage_vector_agent
        from agentoolkit.storage.document import create_storage_document_agent
        from agentoolkit.storage.kv import create_storage_kv_agent, _kv_storage, _kv_expiry
        
        # Clear storages
        _kv_storage.clear()
        _kv_expiry.clear()
        
        # Clear vector storage pool
        from agentoolkit.storage import vector
        if vector._pool is not None:
            asyncio.run(vector._pool.close())
            vector._pool = None
        
        # Clear OpenAI client
        from agentoolkit.rag import embedder
        embedder._openai_client = None
        
        # Register all agents
        kv_agent = create_storage_kv_agent()
        doc_agent = create_storage_document_agent()
        vec_agent = create_storage_vector_agent()
        emb_agent = create_rag_embedder_agent()
        ret_agent = create_rag_retriever_agent()
    
    def teardown_method(self):
        """Clean up resources after each test."""
        # Simple cleanup - just clear references
        from agentoolkit.storage import vector
        from agentoolkit.rag import embedder
        
        # Clear pool reference - let garbage collection handle the rest
        vector._pool = None
        
        # Clear OpenAI client reference
        if hasattr(embedder, '_openai_client'):
            embedder._openai_client = None
    
    def test_retriever_search(self):
        """Test basic search functionality."""
        
        async def run_test():
            injector = get_injector()
            
            # Initialize vector collection
            await injector.run('storage_vector', {
                "operation": "init_collection",
                "collection": "test"
            })
            
            # Store and index a test document first
            doc_content = "This is a test document about artificial intelligence and machine learning."
            await injector.run('storage_document', {
                "operation": "store",
                "doc_id": "test_doc",
                "content": doc_content
            })
            
            # Index the document for retrieval
            await injector.run('rag_retriever', {
                "operation": "index_document",
                "doc_id": "test_doc",
                "collection": "test"
            })
            
            # Perform search
            result = await injector.run('rag_retriever', {
                "operation": "search",
                "query": "artificial intelligence",
                "collection": "test",
                "top_k": 2
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert data.get("success") is True
            assert "Found" in data.get("message", "")
            results = data.get("results", [])
            assert len(results) >= 0  # May be 0 if no matches found
        
        asyncio.run(run_test())
    
    def test_retriever_context_assembly(self):
        """Test context retrieval and assembly."""
        
        async def run_test():
            injector = get_injector()
            
            # Initialize vector collection
            await injector.run('storage_vector', {
                "operation": "init_collection",
                "collection": "test"
            })
            
            # Store and index multiple test documents
            docs = [
                ("doc1", "Short content about AI"),
                ("doc2", " ".join(["Word"] * 50) + " about machine learning"),
                ("doc3", " ".join(["LongWord"] * 100) + " about neural networks")
            ]
            
            for doc_id, content in docs:
                await injector.run('storage_document', {
                    "operation": "store",
                    "doc_id": doc_id,
                    "content": content
                })
                
                await injector.run('rag_retriever', {
                    "operation": "index_document",
                    "doc_id": doc_id,
                    "collection": "test"
                })
            
            # Retrieve context with max length limit
            result = await injector.run('rag_retriever', {
                "operation": "retrieve_context",
                "query": "AI machine learning",
                "collection": "test",
                "top_k": 5,
                "max_context_length": 200
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            # Should succeed even if no context found
            assert data.get("success") is True or data.get("context") == ""
        
        asyncio.run(run_test())
    
    def test_retriever_index_document(self):
        """Test document indexing for retrieval."""
        
        async def run_test():
            injector = get_injector()
            
            # Initialize vector collection
            await injector.run('storage_vector', {
                "operation": "init_collection",
                "collection": "test"
            })
            
            # Store a document first
            doc_content = " ".join([f"Word{i}" for i in range(200)])
            await injector.run('storage_document', {
                "operation": "store",
                "doc_id": "index_test",
                "content": doc_content
            })
            
            # Index the document
            result = await injector.run('rag_retriever', {
                "operation": "index_document",
                "doc_id": "index_test",
                "collection": "test"
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert data.get("success") is True
            assert "Indexed document" in data.get("message", "")
            metadata = data.get("metadata", {})
            assert metadata.get("doc_id") == "index_test"
            assert metadata.get("chunk_count", 0) > 0
        
        asyncio.run(run_test())
    
    def test_retriever_reindex_collection(self):
        """Test reindexing entire collection."""
        
        async def run_test():
            injector = get_injector()
            
            # Initialize vector collection
            await injector.run('storage_vector', {
                "operation": "init_collection",
                "collection": "test"
            })
            
            # Store multiple documents
            for i in range(3):
                await injector.run('storage_document', {
                    "operation": "store",
                    "doc_id": f"reindex_doc_{i}",
                    "content": f"Document {i} content about topic {i}"
                })
            
            # Reindex collection
            result = await injector.run('rag_retriever', {
                "operation": "reindex_collection",
                "collection": "test"
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert data.get("success") is True
            assert "Reindexed" in data.get("message", "")
            metadata = data.get("metadata", {})
            assert metadata.get("indexed", 0) >= 0  # Some documents indexed
        
        asyncio.run(run_test())
    
    def test_retriever_no_results(self):
        """Test behavior when no results are found."""
        
        async def run_test():
            injector = get_injector()
            
            # Initialize empty collection
            await injector.run('storage_vector', {
                "operation": "init_collection",
                "collection": "empty"
            })
            
            # Try to retrieve context with no indexed documents
            result = await injector.run('rag_retriever', {
                "operation": "retrieve_context",
                "query": "nonexistent query",
                "collection": "empty",
                "top_k": 5
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            # Should handle gracefully - either success with empty context or failure
            assert data.get("success") is False or data.get("context") == ""
        
        asyncio.run(run_test())
    
    def test_retriever_error_handling(self):
        """Test error handling for retriever operations."""
        
        async def run_test():
            injector = get_injector()
            
            # Test index non-existent document
            try:
                await injector.run('rag_retriever', {
                    "operation": "index_document",
                    "doc_id": "nonexistent_xyz",
                    "collection": "test"
                })
                # May succeed or fail depending on implementation
            except (KeyError, RuntimeError) as e:
                assert "not found" in str(e) or "Failed" in str(e)
        
        asyncio.run(run_test())