"""
Embedder AgenTool - OpenAI embeddings with caching for RAG.

This AgenTool provides embedding generation using OpenAI's API with
KV storage caching to minimize API calls and costs.
"""

import hashlib
import json
import asyncio
from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from agentool import create_agentool
from agentool.core.registry import RoutingConfig
from agentool.core.injector import get_injector

# OpenAI client (will be initialized on first use)
_openai_client = None

class EmbedderInput(BaseModel):
    """Input schema for embedder operations."""
    operation: Literal['embed', 'embed_batch', 'clear_cache', 'get_model_info']
    # For embed/embed_batch
    texts: Optional[List[str]] = Field(None, description="Texts to embed")
    model: str = Field("text-embedding-3-small", description="OpenAI embedding model")
    # For cache operations
    cache_key_prefix: Optional[str] = Field("embed", description="Cache key prefix")

class EmbedderOutput(BaseModel):
    """Output schema for embedder operations."""
    success: bool
    message: str
    embeddings: Optional[List[List[float]]] = None
    model_info: Optional[Dict[str, Any]] = None
    cache_hits: Optional[int] = None
    api_calls: Optional[int] = None

def get_cache_key(text: str, model: str, prefix: str = "embed") -> str:
    """Generate cache key for text embedding."""
    text_hash = hashlib.sha256(f"{text}:{model}".encode()).hexdigest()[:16]
    return f"{prefix}:{model}:{text_hash}"

async def get_openai_client():
    """Get or create OpenAI client."""
    global _openai_client
    if _openai_client is None:
        import os
        from openai import AsyncOpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        _openai_client = AsyncOpenAI(api_key=api_key)
    
    return _openai_client

async def embed_single(
    ctx: RunContext[Any],
    text: str,
    model: str = "text-embedding-3-small",
    cache_key_prefix: str = "embed"
) -> List[float]:
    """Embed a single text with caching."""
    injector = get_injector()
    cache_key = get_cache_key(text, model, cache_key_prefix)
    
    # Check cache first
    try:
        cache_result = await injector.run('storage_kv', {
            "operation": "get",
            "key": cache_key,
            "namespace": "rag"
        })
        
        if hasattr(cache_result, 'output'):
            cache_data = json.loads(cache_result.output)
            if (cache_data.get('operation') == 'get' and 
                cache_data.get('data') and 
                cache_data['data'].get('exists')):
                # Cache hit - return cached embedding
                embedding = json.loads(cache_data['data']['value'])
                return embedding
    except KeyError:
        # Cache miss - key doesn't exist
        pass
    except Exception:
        # Other error - continue to generate
        pass
    
    # Generate embedding via OpenAI API
    client = await get_openai_client()
    
    try:
        response = await client.embeddings.create(
            model=model,
            input=text
        )
        
        embedding = response.data[0].embedding
        
        # Cache the embedding
        await injector.run('storage_kv', {
            "operation": "set",
            "key": cache_key,
            "value": json.dumps(embedding),
            "namespace": "rag"
        })
        
        return embedding
    except Exception as e:
        raise RuntimeError(f"Failed to generate embedding: {e}") from e

async def embed(
    ctx: RunContext[Any],
    texts: List[str],
    model: str = "text-embedding-3-small",
    cache_key_prefix: str = "embed"
) -> EmbedderOutput:
    """Embed one or more texts with caching."""
    if not texts:
        raise ValueError("No texts provided for embedding")
    
    embeddings = []
    cache_hits = 0
    api_calls = 0
    
    for text in texts:
        # Check if we have cached embedding
        injector = get_injector()
        cache_key = get_cache_key(text, model, cache_key_prefix)
        
        try:
            cache_result = await injector.run('storage_kv', {
                "operation": "get",
                "key": cache_key,
                "namespace": "rag"
            })
            
            if hasattr(cache_result, 'output'):
                cache_data = json.loads(cache_result.output)
                if (cache_data.get('operation') == 'get' and 
                    cache_data.get('data') and 
                    cache_data['data'].get('exists')):
                    # Cache hit
                    embedding = json.loads(cache_data['data']['value'])
                    embeddings.append(embedding)
                    cache_hits += 1
                    continue
        except KeyError:
            # Cache miss - key doesn't exist
            pass
        except Exception:
            # Other error - treat as cache miss
            pass
        
        # Cache miss - generate embedding
        embedding = await embed_single(ctx, text, model, cache_key_prefix)
        embeddings.append(embedding)
        api_calls += 1
    
    return EmbedderOutput(
        success=True,
        message=f"Generated {len(embeddings)} embeddings ({cache_hits} from cache, {api_calls} from API)",
        embeddings=embeddings,
        cache_hits=cache_hits,
        api_calls=api_calls
    )

async def embed_batch(
    ctx: RunContext[Any],
    texts: List[str],
    model: str = "text-embedding-3-small",
    cache_key_prefix: str = "embed"
) -> EmbedderOutput:
    """Embed texts in batch with caching and optional queue processing."""
    if not texts:
        raise ValueError("No texts provided for embedding")
    
    injector = get_injector()
    embeddings = []
    texts_to_embed = []
    text_indices = []
    cache_hits = 0
    
    # Check cache for each text
    for i, text in enumerate(texts):
        cache_key = get_cache_key(text, model, cache_key_prefix)
        
        try:
            cache_result = await injector.run('storage_kv', {
                "operation": "get",
                "key": cache_key,
                "namespace": "rag"
            })
            
            if hasattr(cache_result, 'output'):
                cache_data = json.loads(cache_result.output)
                if (cache_data.get('operation') == 'get' and 
                    cache_data.get('data') and 
                    cache_data['data'].get('exists')):
                    # Cache hit
                    embedding = json.loads(cache_data['data']['value'])
                    embeddings.append((i, embedding))
                    cache_hits += 1
                    continue
        except KeyError:
            # Cache miss - key doesn't exist
            pass
        except Exception:
            # Other error - treat as cache miss
            pass
        
        # Cache miss - add to batch
        texts_to_embed.append(text)
        text_indices.append(i)
    
    # Batch embed texts not in cache
    if texts_to_embed:
        client = await get_openai_client()
        
        # OpenAI supports batch embedding
        # TODO: Phase 2 - Use queue/scheduler for large batches
        try:
            # Split into smaller batches if needed (OpenAI has limits)
            batch_size = 100  # OpenAI's typical batch limit
            for batch_start in range(0, len(texts_to_embed), batch_size):
                batch_end = min(batch_start + batch_size, len(texts_to_embed))
                batch_texts = texts_to_embed[batch_start:batch_end]
                batch_indices = text_indices[batch_start:batch_end]
                
                response = await client.embeddings.create(
                    model=model,
                    input=batch_texts
                )
                
                # Process and cache results
                for j, embedding_data in enumerate(response.data):
                    text = batch_texts[j]
                    idx = batch_indices[j]
                    embedding = embedding_data.embedding
                    
                    # Cache the embedding
                    cache_key = get_cache_key(text, model, cache_key_prefix)
                    await injector.run('storage_kv', {
                        "operation": "set",
                        "key": cache_key,
                        "value": json.dumps(embedding),
                        "namespace": "rag"
                    })
                    
                    embeddings.append((idx, embedding))
        except Exception as e:
            raise RuntimeError(f"Failed to batch embed texts: {e}") from e
    
    # Sort embeddings by original index
    embeddings.sort(key=lambda x: x[0])
    ordered_embeddings = [emb for _, emb in embeddings]
    
    api_calls = len(texts_to_embed)
    
    return EmbedderOutput(
        success=True,
        message=f"Batch embedded {len(texts)} texts ({cache_hits} from cache, {api_calls} from API)",
        embeddings=ordered_embeddings,
        cache_hits=cache_hits,
        api_calls=api_calls
    )

async def clear_cache(ctx: RunContext[Any], cache_key_prefix: str = "embed") -> EmbedderOutput:
    """Clear embedding cache."""
    injector = get_injector()
    
    try:
        # List all keys with prefix
        list_result = await injector.run('storage_kv', {
            "operation": "keys",
            "pattern": f"{cache_key_prefix}:*",
            "namespace": "rag"
        })
        
        if hasattr(list_result, 'output'):
            list_data = json.loads(list_result.output)
            if list_data.get('operation') == 'keys' and list_data.get('data'):
                keys = list_data['data'].get('keys', [])
                
                # Delete each key
                for key in keys:
                    await injector.run('storage_kv', {
                        "operation": "delete",
                        "key": key,
                        "namespace": "rag"
                    })
                
                return EmbedderOutput(
                    success=True,
                    message=f"Cleared {len(keys)} cached embeddings",
                    cache_hits=0,
                    api_calls=0
                )
        
        return EmbedderOutput(
            success=True,
            message="No cached embeddings to clear",
            cache_hits=0,
            api_calls=0
        )
    except Exception as e:
        raise RuntimeError(f"Failed to clear cache: {e}") from e

async def get_model_info(ctx: RunContext[Any], model: str = "text-embedding-3-small") -> EmbedderOutput:
    """Get information about embedding model."""
    model_info = {
        "text-embedding-3-small": {
            "dimensions": 1536,
            "max_tokens": 8191,
            "cost_per_million_tokens": 0.02,
            "description": "Small, efficient embedding model"
        },
        "text-embedding-3-large": {
            "dimensions": 3072,
            "max_tokens": 8191,
            "cost_per_million_tokens": 0.13,
            "description": "Large, high-quality embedding model"
        },
        "text-embedding-ada-002": {
            "dimensions": 1536,
            "max_tokens": 8191,
            "cost_per_million_tokens": 0.10,
            "description": "Legacy embedding model"
        }
    }
    
    if model not in model_info:
        raise ValueError(f"Unknown model: {model}. Available models: {list(model_info.keys())}")
    
    return EmbedderOutput(
        success=True,
        message=f"Model info for {model}",
        model_info=model_info[model]
    )

# Routing configuration
routing = RoutingConfig(
    operation_field='operation',
    operation_map={
        'embed': ('embed', lambda x: {
            'texts': x.texts,
            'model': x.model,
            'cache_key_prefix': x.cache_key_prefix
        }),
        'embed_batch': ('embed_batch', lambda x: {
            'texts': x.texts,
            'model': x.model,
            'cache_key_prefix': x.cache_key_prefix
        }),
        'clear_cache': ('clear_cache', lambda x: {
            'cache_key_prefix': x.cache_key_prefix
        }),
        'get_model_info': ('get_model_info', lambda x: {
            'model': x.model
        })
    }
)

def create_rag_embedder_agent():
    """Create and return the embedder AgenTool."""
    return create_agentool(
        name='rag_embedder',
        input_schema=EmbedderInput,
        routing_config=routing,
        tools=[embed, embed_batch, clear_cache, get_model_info],
        output_type=EmbedderOutput,
        system_prompt="Generate text embeddings using OpenAI with intelligent caching.",
        description="Embedding generation with caching for RAG applications",
        version="1.0.0",
        tags=["embeddings", "openai", "cache", "rag", "vector"],
        dependencies=["storage_kv"],
        examples=[
            {
                "input": {
                    "operation": "embed",
                    "texts": ["Hello world"],
                    "model": "text-embedding-3-small"
                },
                "output": {
                    "success": True,
                    "message": "Generated 1 embeddings",
                    "cache_hits": 0,
                    "api_calls": 1
                }
            },
            {
                "input": {
                    "operation": "embed_batch",
                    "texts": ["First text", "Second text", "Third text"],
                    "model": "text-embedding-3-small"
                },
                "output": {
                    "success": True,
                    "message": "Batch embedded 3 texts",
                    "cache_hits": 1,
                    "api_calls": 2
                }
            }
        ]
    )

# Export
agent = create_rag_embedder_agent()

# TODO: Phase 2 Features
# - Add support for Cohere embeddings
# - Add support for HuggingFace embeddings
# - Implement queue/scheduler for large batch processing
# - Add embedding dimension reduction options
# - Support for multi-modal embeddings

# TODO: Phase 3 Features
# - Embedding compression techniques
# - Distributed embedding generation
# - Custom embedding models
# - Incremental embedding updates
# - Cross-lingual embedding alignment