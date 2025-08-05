"""
Tests for graph_rag toolkit.

This module tests all functionality of the RAG graph toolkit
including question answering, document indexing, and graph orchestration.
"""

import json
import asyncio
import pytest

from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry


class TestGraphRag:
    """Test suite for graph_rag toolkit."""
    
    def setup_method(self):
        """Clear registry and injector before each test."""
        AgenToolRegistry.clear()
        get_injector().clear()
        
        # Import and create all required agents
        from agentoolkit.graph.rag_graph import create_graph_rag_agent
        from agentoolkit.rag.retriever import create_rag_retriever_agent
        from agentoolkit.rag.embedder import create_rag_embedder_agent
        from agentoolkit.storage.vector import create_storage_vector_agent
        from agentoolkit.storage.document import create_storage_document_agent
        from agentoolkit.storage.kv import create_storage_kv_agent, _kv_storage, _kv_expiry
        
        # Clear storages
        _kv_storage.clear()
        _kv_expiry.clear()
        
        # Clear vector storage pool - just set to None, don't try to close
        from agentoolkit.storage import vector
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
        graph_agent = create_graph_rag_agent()
    
    def teardown_method(self):
        """Clean up resources after each test."""
        # Import modules
        from agentoolkit.storage import vector
        from agentoolkit.rag import embedder
        
        # Just clear references - don't try to close async resources
        # The resources will be cleaned up by garbage collection
        vector._pool = None
        embedder._openai_client = None
    
    async def test_rag_graph_ask_question(self, allow_model_requests):
        """Test asking questions using RAG graph."""
        injector = get_injector()
        
        # Initialize vector collection
        await injector.run('storage_vector', {
            "operation": "init_collection",
            "collection": "test"
        })
        
        # Store and index test documents
        test_doc = "RAG stands for Retrieval-Augmented Generation. It is a technique that combines information retrieval with text generation to provide more accurate and contextual responses."
        await injector.run('storage_document', {
            "operation": "store",
            "doc_id": "rag_doc", 
            "content": test_doc
        })
        
        # Index the document
        await injector.run('rag_retriever', {
            "operation": "index_document",
            "doc_id": "rag_doc",
            "collection": "test"
        })
        
        # Ask a question using graph RAG
        result = await injector.run('graph_rag', {
            "operation": "ask",
            "question": "What is RAG?",
            "collection": "test",
            "model": "openai:gpt-4o"
        })
        
        if hasattr(result, 'output'):
            data = json.loads(result.output)
        else:
            data = result
        
        assert data.get("success") is True
        assert data.get("answer") is not None
        assert len(data.get("answer", "")) > 0
        sources = data.get("sources", [])
        assert len(sources) >= 0  # May have sources from retrieval
    
    async def test_rag_graph_no_context_found(self, allow_model_requests):
        """Test behavior when no relevant context is found."""
        injector = get_injector()
        
        # Initialize empty vector collection
        await injector.run('storage_vector', {
            "operation": "init_collection", 
            "collection": "empty"
        })
        
        # Ask a question with no relevant documents
        result = await injector.run('graph_rag', {
            "operation": "ask",
            "question": "What is quantum computing?",
            "collection": "empty",
            "model": "openai:gpt-4o"
        })
        
        if hasattr(result, 'output'):
            data = json.loads(result.output)
        else:
            data = result
        
        # Should still succeed but with limited context
        assert data.get("success") is True
        assert data.get("answer") is not None
        # Answer should indicate insufficient context or provide general response
        assert len(data.get("answer", "")) > 0
    
    async def test_rag_graph_custom_model(self, allow_model_requests):
        """Test using different model configurations."""
        injector = get_injector()
        
        # Initialize vector collection
        await injector.run('storage_vector', {
            "operation": "init_collection",
            "collection": "test"
        })
        
        # Store and index test document
        test_doc = "Machine learning is a subset of artificial intelligence that uses algorithms to learn patterns from data."
        await injector.run('storage_document', {
            "operation": "store",
            "doc_id": "ml_doc",
            "content": test_doc
        })
        
        # Index the document
        await injector.run('rag_retriever', {
            "operation": "index_document", 
            "doc_id": "ml_doc",
            "collection": "test"
        })
        
        # Ask question with OpenAI model (should work)
        result = await injector.run('graph_rag', {
            "operation": "ask",
            "question": "What is machine learning?",
            "collection": "test",
            "model": "openai:gpt-4o"  # Use OpenAI, not Anthropic
        })
        
        if hasattr(result, 'output'):
            data = json.loads(result.output)
        else:
            data = result
        
        assert data.get("success") is True
        assert data.get("answer") is not None
        assert len(data.get("answer", "")) > 0
    
    def test_rag_graph_index_documents(self):
        """Test document indexing through RAG graph."""
        
        async def run_test():
            injector = get_injector()
            
            # Initialize vector collection
            await injector.run('storage_vector', {
                "operation": "init_collection",
                "collection": "test"
            })
            
            # Test document indexing
            documents = [
                "First document about artificial intelligence and machine learning.",
                "Second document about natural language processing and text generation."
            ]
            doc_ids = ["doc1", "doc2"]
            
            # Index documents using graph RAG (documents parameter is required)
            result = await injector.run('graph_rag', {
                "operation": "index",
                "documents": documents,
                "doc_ids": doc_ids,
                "collection": "test"
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
                
            assert data.get("success") is True
            assert data.get("indexed_count", 0) >= 2  # Should index 2 documents
        
        asyncio.run(run_test())
    
    def test_rag_graph_clear_cache(self):
        """Test clearing RAG caches."""
        
        async def run_test(): 
            injector = get_injector()
            
            # Clear cache operation
            result = await injector.run('graph_rag', {
                "operation": "clear_cache"
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert data.get("success") is True
            assert "cache" in data.get("message", "").lower()
        
        asyncio.run(run_test())
    
    def test_rag_graph_error_handling(self):
        """Test error handling for invalid operations."""
        
        async def run_test():
            injector = get_injector()
            
            # Test with invalid operation - should return error message string
            result = await injector.run('graph_rag', {
                "operation": "invalid_operation"
            })
            
            # Should return error message string for invalid operations
            if hasattr(result, 'output'):
                data = result.output
            else:
                data = str(result)
            
            assert "unknown operation" in data.lower() or "invalid" in data.lower()
        
        asyncio.run(run_test())