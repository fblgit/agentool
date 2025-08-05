"""
Retriever AgenTool - Document retrieval and context assembly for RAG.

This AgenTool provides semantic search and context retrieval operations,
integrating with vector storage, document storage, and embedder agentools.
"""

import json
from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from agentool import create_agentool
from agentool.core.registry import RoutingConfig
from agentool.core.injector import get_injector

class RetrieverInput(BaseModel):
    """Input schema for retriever operations."""
    operation: Literal['search', 'retrieve_context', 'index_document', 'reindex_collection']
    # For search/retrieve_context
    query: Optional[str] = Field(None, description="Search query")
    collection: str = Field("default", description="Collection to search")
    top_k: int = Field(5, description="Number of results to return")
    include_metadata: bool = Field(True, description="Include metadata in results")
    # For index_document
    doc_id: Optional[str] = Field(None, description="Document ID to index")
    # For retrieval options
    max_context_length: Optional[int] = Field(2000, description="Maximum context length in tokens")

class RetrieverOutput(BaseModel):
    """Output schema for retriever operations."""
    success: bool
    message: str
    results: Optional[List[Dict[str, Any]]] = None
    context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    count: Optional[int] = None

class SearchResult(BaseModel):
    """Schema for search results."""
    doc_id: str
    chunk_id: Optional[str] = None
    content: str
    similarity: float
    metadata: Dict[str, Any]

async def search(
    ctx: RunContext[Any],
    query: str,
    collection: str = "default",
    top_k: int = 5,
    include_metadata: bool = True
) -> RetrieverOutput:
    """Perform semantic search for query."""
    injector = get_injector()
    
    try:
        # Step 1: Embed the query
        embed_result = await injector.run('rag_embedder', {
            "operation": "embed",
            "texts": [query],
            "model": "text-embedding-3-small"
        })
        
        if hasattr(embed_result, 'output'):
            embed_data = json.loads(embed_result.output)
            if not embed_data.get('success'):
                raise RuntimeError("Failed to embed query")
            query_embedding = embed_data['embeddings'][0]
        else:
            raise RuntimeError("Invalid embedder response")
        
        # Step 2: Search vector storage
        search_result = await injector.run('storage_vector', {
            "operation": "search",
            "collection": collection,
            "query_embedding": query_embedding,
            "top_k": top_k
        })
        
        if hasattr(search_result, 'output'):
            search_data = json.loads(search_result.output)
            if not search_data.get('success'):
                raise RuntimeError("Vector search failed")
            
            raw_results = search_data['data']['results']
        else:
            raise RuntimeError("Invalid vector search response")
        
        # Step 3: Format results
        results = []
        for result in raw_results:
            search_result = SearchResult(
                doc_id=result.get('metadata', {}).get('doc_id', result['id']),
                chunk_id=result.get('id'),
                content=result.get('content', ''),
                similarity=result.get('similarity', 0.0),
                metadata=result.get('metadata', {}) if include_metadata else {}
            )
            results.append(search_result.model_dump())
        
        return RetrieverOutput(
            success=True,
            message=f"Found {len(results)} relevant results",
            results=results,
            count=len(results),
            metadata={"query": query, "collection": collection}
        )
    except Exception as e:
        raise RuntimeError(f"Search failed: {e}") from e

async def retrieve_context(
    ctx: RunContext[Any],
    query: str,
    collection: str = "default",
    top_k: int = 5,
    max_context_length: int = 2000
) -> RetrieverOutput:
    """Retrieve and assemble context for query."""
    injector = get_injector()
    
    try:
        # Step 1: Perform search
        search_output = await search(ctx, query, collection, top_k, include_metadata=True)
        
        if not search_output.success or not search_output.results:
            return RetrieverOutput(
                success=False,
                message="No relevant context found",
                context="",
                metadata={"query": query}
            )
        
        # Step 2: Assemble context from search results
        context_parts = []
        total_length = 0
        sources = []
        
        for result in search_output.results:
            content = result['content']
            doc_id = result['doc_id']
            chunk_id = result.get('chunk_id')
            
            # Simple token count (approximate)
            content_length = len(content.split())
            
            # Check if adding this would exceed max length
            if total_length + content_length > max_context_length:
                if total_length == 0:
                    # If first chunk is too long, truncate it
                    words = content.split()[:max_context_length]
                    content = " ".join(words)
                    context_parts.append(content)
                    sources.append({"doc_id": doc_id, "chunk_id": chunk_id})
                break
            
            context_parts.append(content)
            sources.append({"doc_id": doc_id, "chunk_id": chunk_id})
            total_length += content_length
        
        # Step 3: Format context
        context = "\n\n---\n\n".join(context_parts)
        
        return RetrieverOutput(
            success=True,
            message=f"Retrieved context from {len(sources)} sources",
            context=context,
            metadata={
                "query": query,
                "sources": sources,
                "total_chunks": len(sources),
                "approx_tokens": total_length
            },
            count=len(sources)
        )
    except Exception as e:
        raise RuntimeError(f"Context retrieval failed: {e}") from e

async def index_document(
    ctx: RunContext[Any],
    doc_id: str,
    collection: str = "default"
) -> RetrieverOutput:
    """Index a document for retrieval."""
    injector = get_injector()
    
    try:
        # Step 1: Retrieve document
        doc_result = await injector.run('storage_document', {
            "operation": "retrieve",
            "doc_id": doc_id
        })
        
        if hasattr(doc_result, 'output'):
            doc_data = json.loads(doc_result.output)
            if not doc_data.get('success'):
                raise RuntimeError(f"Document not found: {doc_id}")
            document = doc_data['data']
        else:
            raise RuntimeError("Invalid document storage response")
        
        # Step 2: Chunk document if not already chunked
        if not document.get('metadata', {}).get('chunked'):
            chunk_result = await injector.run('storage_document', {
                "operation": "chunk",
                "doc_id": doc_id,
                "chunk_size": 500,
                "chunk_overlap": 50
            })
            
            if hasattr(chunk_result, 'output'):
                chunk_data = json.loads(chunk_result.output)
                if not chunk_data.get('success'):
                    raise RuntimeError("Failed to chunk document")
                chunks = chunk_data['data']['chunks']
            else:
                raise RuntimeError("Invalid chunking response")
        else:
            # Retrieve existing chunks
            chunks = []
            chunk_count = document['metadata'].get('chunk_count', 0)
            for i in range(chunk_count):
                chunk_key = f"doc:chunk:{doc_id}:{i}"
                chunk_result = await injector.run('storage_kv', {
                    "operation": "get",
                    "key": chunk_key
                })
                
                if hasattr(chunk_result, 'output'):
                    chunk_data = json.loads(chunk_result.output)
                    if chunk_data.get('success') and chunk_data.get('data'):
                        chunk = json.loads(chunk_data['data']['value'])
                        chunks.append(chunk)
        
        # Step 3: Generate embeddings for chunks
        chunk_texts = [chunk['content'] for chunk in chunks]
        
        embed_result = await injector.run('rag_embedder', {
            "operation": "embed_batch",
            "texts": chunk_texts,
            "model": "text-embedding-3-small"
        })
        
        if hasattr(embed_result, 'output'):
            embed_data = json.loads(embed_result.output)
            if not embed_data.get('success'):
                raise RuntimeError("Failed to generate embeddings")
            embeddings = embed_data['embeddings']
        else:
            raise RuntimeError("Invalid embedder response")
        
        # Step 4: Store embeddings in vector storage
        ids = [chunk['chunk_id'] for chunk in chunks]
        metadata_list = []
        for chunk in chunks:
            meta = chunk['metadata'].copy()
            meta['doc_id'] = doc_id
            meta['chunk_id'] = chunk['chunk_id']
            metadata_list.append(meta)
        
        # Initialize collection if needed
        await injector.run('storage_vector', {
            "operation": "init_collection",
            "collection": collection
        })
        
        # Upsert vectors
        vector_result = await injector.run('storage_vector', {
            "operation": "upsert",
            "collection": collection,
            "embeddings": embeddings,
            "ids": ids,
            "metadata": metadata_list,
            "contents": chunk_texts
        })
        
        if hasattr(vector_result, 'output'):
            vector_data = json.loads(vector_result.output)
            if not vector_data.get('success'):
                raise RuntimeError("Failed to store vectors")
        
        return RetrieverOutput(
            success=True,
            message=f"Indexed document {doc_id} with {len(chunks)} chunks",
            count=len(chunks),
            metadata={
                "doc_id": doc_id,
                "collection": collection,
                "chunk_count": len(chunks)
            }
        )
    except Exception as e:
        raise RuntimeError(f"Document indexing failed: {e}") from e

async def reindex_collection(
    ctx: RunContext[Any],
    collection: str = "default"
) -> RetrieverOutput:
    """Reindex all documents in a collection."""
    injector = get_injector()
    
    try:
        # Step 1: List all documents
        list_result = await injector.run('storage_document', {
            "operation": "list",
            "limit": 1000  # TODO: Phase 2 - Handle pagination
        })
        
        if hasattr(list_result, 'output'):
            list_data = json.loads(list_result.output)
            if not list_data.get('success'):
                raise RuntimeError("Failed to list documents")
            documents = list_data['data']['documents']
        else:
            raise RuntimeError("Invalid document listing response")
        
        # Step 2: Clear existing collection
        await injector.run('storage_vector', {
            "operation": "init_collection",
            "collection": collection
        })
        
        # Step 3: Index each document
        indexed_count = 0
        failed_docs = []
        
        for doc_info in documents:
            doc_id = doc_info['doc_id']
            try:
                await index_document(ctx, doc_id, collection)
                indexed_count += 1
            except Exception as e:
                failed_docs.append({"doc_id": doc_id, "error": str(e)})
        
        return RetrieverOutput(
            success=True,
            message=f"Reindexed {indexed_count} documents ({len(failed_docs)} failed)",
            count=indexed_count,
            metadata={
                "collection": collection,
                "total_docs": len(documents),
                "indexed": indexed_count,
                "failed": failed_docs
            }
        )
    except Exception as e:
        raise RuntimeError(f"Collection reindexing failed: {e}") from e

# Routing configuration
routing = RoutingConfig(
    operation_field='operation',
    operation_map={
        'search': ('search', lambda x: {
            'query': x.query,
            'collection': x.collection,
            'top_k': x.top_k,
            'include_metadata': x.include_metadata
        }),
        'retrieve_context': ('retrieve_context', lambda x: {
            'query': x.query,
            'collection': x.collection,
            'top_k': x.top_k,
            'max_context_length': x.max_context_length
        }),
        'index_document': ('index_document', lambda x: {
            'doc_id': x.doc_id,
            'collection': x.collection
        }),
        'reindex_collection': ('reindex_collection', lambda x: {
            'collection': x.collection
        })
    }
)

def create_rag_retriever_agent():
    """Create and return the retriever AgenTool."""
    return create_agentool(
        name='rag_retriever',
        input_schema=RetrieverInput,
        routing_config=routing,
        tools=[search, retrieve_context, index_document, reindex_collection],
        output_type=RetrieverOutput,
        system_prompt="Retrieve relevant context for RAG applications using semantic search.",
        description="Document retrieval and context assembly for RAG pipelines",
        version="1.0.0",
        tags=["retrieval", "search", "context", "rag", "semantic"],
        dependencies=["rag_embedder", "storage_vector", "storage_document", "storage_kv"],
        examples=[
            {
                "input": {
                    "operation": "search",
                    "query": "What is RAG?",
                    "collection": "documents",
                    "top_k": 3
                },
                "output": {
                    "success": True,
                    "message": "Found 3 relevant results",
                    "count": 3
                }
            },
            {
                "input": {
                    "operation": "retrieve_context",
                    "query": "How does vector search work?",
                    "collection": "documents",
                    "top_k": 5,
                    "max_context_length": 1000
                },
                "output": {
                    "success": True,
                    "message": "Retrieved context from 5 sources",
                    "context": "Vector search uses...",
                    "count": 5
                }
            }
        ]
    )

# Export
agent = create_rag_retriever_agent()

# TODO: Phase 2 Features
# - Query expansion and refinement
# - Hybrid search (keyword + semantic)
# - Reranking with cross-encoder models
# - Multi-collection search
# - Faceted search with metadata filters

# TODO: Phase 3 Features
# - Multi-hop reasoning
# - Query decomposition
# - Conversational context tracking
# - Incremental indexing
# - Distributed retrieval