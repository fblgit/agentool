"""
Tests for rag_embedder toolkit.

This module tests all functionality of the rag_embedder toolkit
including embedding generation, caching, batch processing, and OpenAI integration.
"""

import json
import asyncio
import os
from unittest.mock import patch, MagicMock, AsyncMock

from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry


class TestRagEmbedder:
    """Test suite for rag_embedder toolkit."""
    
    def setup_method(self):
        """Clear registry and injector before each test."""
        AgenToolRegistry.clear()
        get_injector().clear()
        
        # Import and create agents
        from agentoolkit.rag.embedder import (
            create_rag_embedder_agent,
            _openai_client
        )
        from agentoolkit.storage.kv import create_storage_kv_agent, _kv_storage, _kv_expiry
        
        # Clear KV storage for caching
        _kv_storage.clear()
        _kv_expiry.clear()
        
        # Clear OpenAI client
        from agentoolkit.rag import embedder
        embedder._openai_client = None
        
        # Register agents
        kv_agent = create_storage_kv_agent()
        emb_agent = create_rag_embedder_agent()
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('openai.AsyncOpenAI')
    def test_rag_embedder_single_text(self, mock_openai_class):
        """Test embedding single text."""
        
        async def run_test():
            # Mock OpenAI client
            mock_client = AsyncMock()
            mock_openai_class.return_value = mock_client
            
            # Mock embedding response
            mock_embedding = [0.1] * 1536
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=mock_embedding)]
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            
            injector = get_injector()
            
            # Generate embedding
            result = await injector.run('rag_embedder', {
                "operation": "embed",
                "texts": ["Hello, world!"],
                "model": "text-embedding-3-small"
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert data.get("success") is True
            assert "Generated 1 embeddings" in data.get("message", "")
            embeddings = data.get("embeddings", [])
            assert len(embeddings) == 1
            assert len(embeddings[0]) == 1536
            assert data.get("api_calls") == 1
            assert data.get("cache_hits") == 0
            
            # Verify OpenAI was called
            mock_client.embeddings.create.assert_called_once()
        
        asyncio.run(run_test())
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('openai.AsyncOpenAI')
    def test_rag_embedder_caching(self, mock_openai_class):
        """Test embedding caching functionality."""
        
        async def run_test():
            # Mock OpenAI client
            mock_client = AsyncMock()
            mock_openai_class.return_value = mock_client
            
            # Mock embedding response
            mock_embedding = [0.2] * 1536
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=mock_embedding)]
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            
            injector = get_injector()
            text = "Cacheable text"
            
            # First call - should hit API
            result1 = await injector.run('rag_embedder', {
                "operation": "embed",
                "texts": [text],
                "model": "text-embedding-3-small"
            })
            
            if hasattr(result1, 'output'):
                data1 = json.loads(result1.output)
            else:
                data1 = result1
            
            assert data1.get("api_calls") == 1
            assert data1.get("cache_hits") == 0
            
            # Second call - should hit cache
            result2 = await injector.run('rag_embedder', {
                "operation": "embed",
                "texts": [text],
                "model": "text-embedding-3-small"
            })
            
            if hasattr(result2, 'output'):
                data2 = json.loads(result2.output)
            else:
                data2 = result2
            
            assert data2.get("api_calls") == 0
            assert data2.get("cache_hits") == 1
            
            # Verify OpenAI was called only once
            assert mock_client.embeddings.create.call_count == 1
            
            # Verify embeddings are the same
            assert data1.get("embeddings") == data2.get("embeddings")
        
        asyncio.run(run_test())
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('openai.AsyncOpenAI')
    def test_rag_embedder_batch(self, mock_openai_class):
        """Test batch embedding functionality."""
        
        async def run_test():
            # Mock OpenAI client
            mock_client = AsyncMock()
            mock_openai_class.return_value = mock_client
            
            # Mock batch embedding response
            mock_embeddings = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=emb) for emb in mock_embeddings
            ]
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            
            injector = get_injector()
            texts = ["Text 1", "Text 2", "Text 3"]
            
            # Batch embed
            result = await injector.run('rag_embedder', {
                "operation": "embed_batch",
                "texts": texts,
                "model": "text-embedding-3-small"
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert data.get("success") is True
            assert "Batch embedded 3 texts" in data.get("message", "")
            embeddings = data.get("embeddings", [])
            assert len(embeddings) == 3
            assert data.get("api_calls") == 3  # All new texts
            assert data.get("cache_hits") == 0
        
        asyncio.run(run_test())
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('openai.AsyncOpenAI')
    def test_rag_embedder_mixed_cache_batch(self, mock_openai_class):
        """Test batch embedding with mixed cache hits."""
        
        async def run_test():
            # Mock OpenAI client
            mock_client = AsyncMock()
            mock_openai_class.return_value = mock_client
            
            # Mock embedding responses
            mock_response1 = MagicMock()
            mock_response1.data = [MagicMock(embedding=[0.1] * 1536)]
            
            mock_response2 = MagicMock()
            mock_response2.data = [
                MagicMock(embedding=[0.2] * 1536),
                MagicMock(embedding=[0.3] * 1536)
            ]
            
            mock_client.embeddings.create = AsyncMock(
                side_effect=[mock_response1, mock_response2]
            )
            
            injector = get_injector()
            
            # First, cache one text
            await injector.run('rag_embedder', {
                "operation": "embed",
                "texts": ["Cached text"],
                "model": "text-embedding-3-small"
            })
            
            # Batch with one cached and two new
            result = await injector.run('rag_embedder', {
                "operation": "embed_batch",
                "texts": ["Cached text", "New text 1", "New text 2"],
                "model": "text-embedding-3-small"
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert data.get("success") is True
            assert data.get("cache_hits") == 1
            assert data.get("api_calls") == 2
            embeddings = data.get("embeddings", [])
            assert len(embeddings) == 3
        
        asyncio.run(run_test())
    
    def test_rag_embedder_clear_cache(self):
        """Test cache clearing functionality."""
        
        async def run_test():
            injector = get_injector()
            
            # First, add some items to cache using proper cache key format
            # Generate cache keys like the embedder does
            import hashlib
            text1 = "test text 1"
            text2 = "test text 2"
            model = "text-embedding-3-small"
            
            key1_hash = hashlib.sha256(f"{text1}:{model}".encode()).hexdigest()[:16]
            key2_hash = hashlib.sha256(f"{text2}:{model}".encode()).hexdigest()[:16]
            cache_key1 = f"embed:{model}:{key1_hash}"
            cache_key2 = f"embed:{model}:{key2_hash}"
            
            await injector.run('storage_kv', {
                "operation": "set",
                "key": cache_key1,
                "value": "[0.1, 0.2]",
                "namespace": "rag"
            })
            
            await injector.run('storage_kv', {
                "operation": "set",
                "key": cache_key2,
                "value": "[0.3, 0.4]",
                "namespace": "rag"
            })
            
            # Clear cache
            result = await injector.run('rag_embedder', {
                "operation": "clear_cache",
                "cache_key_prefix": "embed"
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert data.get("success") is True
            assert "Cleared" in data.get("message", "") or "No cached" in data.get("message", "")
            
            # Verify cache is cleared by trying to get one of the keys we set
            # The key should not exist after clearing cache (should raise KeyError)
            try:
                check_result = await injector.run('storage_kv', {
                    "operation": "get",
                    "key": cache_key1,
                    "namespace": "rag"
                })
                # If we get here, the key still exists - test should fail
                assert False, f"Expected KeyError but key still exists: {check_result}"
            except KeyError:
                # This is expected - key was successfully deleted
                pass
        
        asyncio.run(run_test())
    
    def test_rag_embedder_model_info(self):
        """Test getting model information."""
        
        async def run_test():
            injector = get_injector()
            
            # Get info for small model
            result = await injector.run('rag_embedder', {
                "operation": "get_model_info",
                "model": "text-embedding-3-small"
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert data.get("success") is True
            model_info = data.get("model_info", {})
            assert model_info.get("dimensions") == 1536
            assert model_info.get("max_tokens") == 8191
            assert "cost_per_million_tokens" in model_info
            
            # Get info for large model
            result2 = await injector.run('rag_embedder', {
                "operation": "get_model_info",
                "model": "text-embedding-3-large"
            })
            
            if hasattr(result2, 'output'):
                data2 = json.loads(result2.output)
            else:
                data2 = result2
            
            model_info2 = data2.get("model_info", {})
            assert model_info2.get("dimensions") == 3072
        
        asyncio.run(run_test())
    
    def test_rag_embedder_error_handling(self):
        """Test error handling for rag_embedder operations."""
        
        async def run_test():
            injector = get_injector()
            
            # Test with no texts
            try:
                await injector.run('rag_embedder', {
                    "operation": "embed",
                    "texts": [],
                    "model": "text-embedding-3-small"
                })
                assert False, "Expected ValueError for empty texts"
            except ValueError as e:
                assert "No texts provided" in str(e)
            
            # Test with invalid model
            try:
                await injector.run('rag_embedder', {
                    "operation": "get_model_info",
                    "model": "invalid-model"
                })
                assert False, "Expected ValueError for invalid model"
            except ValueError as e:
                assert "Unknown model" in str(e)
            
            # Test without API key
            with patch.dict(os.environ, {}, clear=True):
                try:
                    await injector.run('rag_embedder', {
                        "operation": "embed",
                        "texts": ["test"],
                        "model": "text-embedding-3-small"
                    })
                    assert False, "Expected ValueError for missing API key"
                except (ValueError, RuntimeError) as e:
                    assert "OPENAI_API_KEY" in str(e) or "Failed" in str(e)
        
        asyncio.run(run_test())
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('openai.AsyncOpenAI')
    def test_rag_embedder_large_batch(self, mock_openai_class):
        """Test handling of large batches."""
        
        async def run_test():
            # Mock OpenAI client
            mock_client = AsyncMock()
            mock_openai_class.return_value = mock_client
            
            # Create 150 texts (exceeds typical batch limit of 100)
            texts = [f"Text {i}" for i in range(150)]
            
            # Mock responses for two batches
            batch1_embeddings = [[0.1] * 1536] * 100
            batch2_embeddings = [[0.2] * 1536] * 50
            
            mock_response1 = MagicMock()
            mock_response1.data = [
                MagicMock(embedding=emb) for emb in batch1_embeddings
            ]
            
            mock_response2 = MagicMock()
            mock_response2.data = [
                MagicMock(embedding=emb) for emb in batch2_embeddings
            ]
            
            mock_client.embeddings.create = AsyncMock(
                side_effect=[mock_response1, mock_response2]
            )
            
            injector = get_injector()
            
            # Batch embed large set
            result = await injector.run('rag_embedder', {
                "operation": "embed_batch",
                "texts": texts,
                "model": "text-embedding-3-small"
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert data.get("success") is True
            embeddings = data.get("embeddings", [])
            assert len(embeddings) == 150
            
            # Verify two API calls were made (batched)
            assert mock_client.embeddings.create.call_count == 2
        
        asyncio.run(run_test())