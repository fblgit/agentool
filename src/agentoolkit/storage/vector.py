"""
Vector Storage AgenTool - PGVector operations for RAG.

This AgenTool provides vector storage operations using PostgreSQL with pgvector extension.
Supports embedding storage, similarity search, and metadata management.
"""

import json
import asyncio
import asyncpg
from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import RunContext

from agentool.base import BaseOperationInput
from agentool import create_agentool
from agentool.core.registry import RoutingConfig

# Storage backend - connection pool
_pool: Optional[asyncpg.Pool] = None

# Configuration
DB_CONFIG = {
    "host": "localhost",
    "port": 54320,
    "database": "postgres",
    "user": "postgres",
    "password": "postgres"
}

class StorageVectorInput(BaseOperationInput):
    """Input schema for vector storage operations."""
    operation: Literal['init_collection', 'upsert', 'search', 'delete', 'list_collections'] = Field(
        description="The vector storage operation to perform"
    )
    collection: str = Field("default", description="Collection/table name")
    # For upsert
    embeddings: Optional[List[List[float]]] = Field(None, description="Vector embeddings")
    ids: Optional[List[str]] = Field(None, description="Document IDs")
    metadata: Optional[List[Dict[str, Any]]] = Field(None, description="Document metadata")
    contents: Optional[List[str]] = Field(None, description="Original text content")
    # For search
    query_embedding: Optional[List[float]] = Field(None, description="Query vector")
    top_k: Optional[int] = Field(5, description="Number of results to return")
    filter: Optional[Dict[str, Any]] = Field(None, description="Metadata filter")
    # For delete
    doc_ids: Optional[List[str]] = Field(None, description="IDs to delete")

class StorageVectorOutput(BaseModel):
    """Structured output for vector storage operations."""
    success: bool = Field(description="Whether the operation succeeded")
    operation: str = Field(description="The operation that was performed")
    message: str = Field(description="Human-readable result message")
    data: Optional[Dict[str, Any]] = Field(None, description="Operation-specific data")

async def get_connection() -> asyncpg.Pool:
    """Get or create database connection pool."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(**DB_CONFIG)
    return _pool

async def init_collection(ctx: RunContext[Any], collection: str) -> StorageVectorOutput:
    """Initialize a vector collection (table) with pgvector."""
    pool = await get_connection()
    
    try:
        async with pool.acquire() as conn:
            # Create vector extension if not exists
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create table for collection
            table_name = f"vectors_{collection}"
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    embedding vector(1536),  -- OpenAI embedding dimension
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Create index for similarity search
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx 
                ON {table_name} 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
            
            return StorageVectorOutput(
                success=True,
                operation="init_collection",
                message=f"Collection '{collection}' initialized successfully",
                data={"collection": collection, "table": table_name}
            )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize collection: {e}") from e

async def upsert(
    ctx: RunContext[Any], 
    collection: str, 
    embeddings: List[List[float]], 
    ids: List[str],
    metadata: Optional[List[Dict[str, Any]]] = None,
    contents: Optional[List[str]] = None
) -> StorageVectorOutput:
    """Upsert vectors with metadata into collection."""
    if len(embeddings) != len(ids):
        raise ValueError("Number of embeddings must match number of IDs")
    
    pool = await get_connection()
    table_name = f"vectors_{collection}"
    
    # Prepare metadata and contents
    if metadata is None:
        metadata = [{} for _ in ids]
    if contents is None:
        contents = ["" for _ in ids]
    
    try:
        async with pool.acquire() as conn:
            # Upsert each vector
            for i, (vec_id, embedding, meta, content) in enumerate(
                zip(ids, embeddings, metadata, contents)
            ):
                # Convert embedding to pgvector format
                embedding_str = f"[{','.join(map(str, embedding))}]"
                meta_json = json.dumps(meta)
                
                await conn.execute(f"""
                    INSERT INTO {table_name} (id, content, embedding, metadata, updated_at)
                    VALUES ($1, $2, $3::vector, $4::jsonb, NOW())
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                """, vec_id, content, embedding_str, meta_json)
            
            return StorageVectorOutput(
                success=True,
                operation="upsert",
                message=f"Upserted {len(ids)} vectors to collection '{collection}'",
                data={"collection": collection, "ids": ids, "count": len(ids)}
            )
    except Exception as e:
        raise RuntimeError(f"Failed to upsert vectors: {e}") from e

async def search(
    ctx: RunContext[Any],
    collection: str,
    query_embedding: List[float],
    top_k: int = 5,
    filter: Optional[Dict[str, Any]] = None
) -> StorageVectorOutput:
    """Search for similar vectors using cosine similarity."""
    pool = await get_connection()
    table_name = f"vectors_{collection}"
    
    # Convert query embedding to pgvector format
    query_str = f"[{','.join(map(str, query_embedding))}]"
    
    try:
        async with pool.acquire() as conn:
            # Build query with optional metadata filter
            base_query = f"""
                SELECT id, content, metadata,
                       1 - (embedding <=> $1::vector) as similarity
                FROM {table_name}
            """
            
            params = [query_str]
            where_clause = ""
            
            if filter:
                # Add JSONB filter conditions
                # TODO: Phase 2 - More sophisticated filtering
                filter_conditions = []
                for key, value in filter.items():
                    filter_conditions.append(f"metadata->>{key!r} = {value!r}")
                where_clause = " WHERE " + " AND ".join(filter_conditions)
            
            query = f"""
                {base_query}
                {where_clause}
                ORDER BY embedding <=> $1::vector
                LIMIT $2
            """
            params.append(top_k)
            
            rows = await conn.fetch(query, *params)
            
            results = []
            for row in rows:
                results.append({
                    "id": row["id"],
                    "content": row["content"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "similarity": float(row["similarity"])
                })
            
            return StorageVectorOutput(
                success=True,
                operation="search",
                message=f"Found {len(results)} similar vectors",
                data={"results": results, "collection": collection, "count": len(results)}
            )
    except Exception as e:
        raise RuntimeError(f"Failed to search vectors: {e}") from e

async def delete(ctx: RunContext[Any], collection: str, doc_ids: List[str]) -> StorageVectorOutput:
    """Delete vectors by IDs."""
    pool = await get_connection()
    table_name = f"vectors_{collection}"
    
    try:
        async with pool.acquire() as conn:
            # Delete vectors
            result = await conn.execute(f"""
                DELETE FROM {table_name}
                WHERE id = ANY($1::text[])
            """, doc_ids)
            
            # Extract number of deleted rows
            deleted_count = int(result.split()[-1])
            
            return StorageVectorOutput(
                success=True,
                operation="delete",
                message=f"Deleted {deleted_count} vectors from collection '{collection}'",
                data={"collection": collection, "deleted_ids": doc_ids[:deleted_count], "count": deleted_count}
            )
    except Exception as e:
        raise RuntimeError(f"Failed to delete vectors: {e}") from e

async def list_collections(ctx: RunContext[Any]) -> StorageVectorOutput:
    """List all vector collections."""
    pool = await get_connection()
    
    try:
        async with pool.acquire() as conn:
            # Find all tables with vectors_ prefix
            rows = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name LIKE 'vectors_%'
            """)
            
            collections = [
                row["table_name"].replace("vectors_", "") 
                for row in rows
            ]
            
            return StorageVectorOutput(
                success=True,
                operation="list_collections",
                message=f"Found {len(collections)} collections",
                data={"collections": collections, "count": len(collections)}
            )
    except Exception as e:
        raise RuntimeError(f"Failed to list collections: {e}") from e

# Routing configuration
routing = RoutingConfig(
    operation_field='operation',
    operation_map={
        'init_collection': ('init_collection', lambda x: {'collection': x.collection}),
        'upsert': ('upsert', lambda x: {
            'collection': x.collection,
            'embeddings': x.embeddings,
            'ids': x.ids,
            'metadata': x.metadata,
            'contents': x.contents
        }),
        'search': ('search', lambda x: {
            'collection': x.collection,
            'query_embedding': x.query_embedding,
            'top_k': x.top_k,
            'filter': x.filter
        }),
        'delete': ('delete', lambda x: {
            'collection': x.collection,
            'doc_ids': x.doc_ids
        }),
        'list_collections': ('list_collections', lambda x: {})
    }
)

def create_vector_agent():
    """Create and return the vector storage AgenTool."""
    return create_agentool(
        name='storage_vector',
        input_schema=StorageVectorInput,
        output_type=StorageVectorOutput,
        use_typed_output=True,
        routing_config=routing,
        tools=[init_collection, upsert, search, delete, list_collections],
        system_prompt="Manage vector embeddings with PGVector for RAG applications.",
        description="Vector storage operations for similarity search and retrieval",
        version="1.0.0",
        tags=["storage", "vector", "pgvector", "rag", "embeddings"],
        dependencies=[],
        examples=[
            {
                "input": {
                    "operation": "init_collection",
                    "collection": "documents"
                },
                "output": {
                    "success": True,
                    "message": "Collection 'documents' initialized successfully",
                    "data": {"collection": "documents", "table": "vectors_documents"}
                }
            },
            {
                "input": {
                    "operation": "search",
                    "collection": "documents",
                    "query_embedding": [0.1, 0.2, 0.3],
                    "top_k": 3
                },
                "output": {
                    "success": True,
                    "message": "Found 3 similar vectors",
                    "count": 3,
                    "data": {"results": []}
                }
            }
        ]
    )


# TODO: Phase 2 Features
# - Add support for Chromadb backend
# - Add support for Pinecone backend
# - Implement hybrid search (keyword + semantic)
# - Add more sophisticated metadata filtering
# - Support for multiple embedding dimensions

# TODO: Phase 3 Features
# - Automatic index optimization
# - Distributed vector storage
# - Cross-collection search
# - Vector compression techniques
# - Incremental indexing