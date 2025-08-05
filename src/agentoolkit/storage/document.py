"""
Document Storage AgenTool - Document management and chunking for RAG.

This AgenTool provides document storage, retrieval, and chunking operations.
Uses KV storage for document content and metadata management.
"""

import hashlib
import json
from typing import Dict, Any, Optional, List, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from agentool import create_agentool
from agentool.core.registry import RoutingConfig
from agentool.core.injector import get_injector

class DocumentStorageInput(BaseModel):
    """Input schema for document storage operations."""
    operation: Literal['store', 'retrieve', 'chunk', 'list', 'delete', 'update_metadata']
    # For store/retrieve/delete
    doc_id: Optional[str] = Field(None, description="Document identifier")
    # For store
    content: Optional[str] = Field(None, description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    # For chunk
    chunk_size: Optional[int] = Field(500, description="Target chunk size in tokens")
    chunk_overlap: Optional[int] = Field(50, description="Overlap between chunks")
    # For list
    prefix: Optional[str] = Field(None, description="Filter documents by ID prefix")
    limit: Optional[int] = Field(100, description="Maximum documents to return")

class DocumentStorageOutput(BaseModel):
    """Output schema for document storage operations."""
    success: bool
    message: str
    data: Optional[Any] = None
    count: Optional[int] = None

class DocumentChunk(BaseModel):
    """Schema for document chunks."""
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    metadata: Dict[str, Any]

def generate_doc_id(content: str) -> str:
    """Generate a unique document ID from content hash."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]

def simple_tokenize(text: str) -> List[str]:
    """Simple tokenization by splitting on whitespace."""
    # TODO: Phase 2 - Use proper tokenizer (tiktoken)
    return text.split()

def chunk_text(
    text: str, 
    chunk_size: int = 500, 
    chunk_overlap: int = 50
) -> List[str]:
    """Chunk text into overlapping segments."""
    tokens = simple_tokenize(text)
    chunks = []
    
    for i in range(0, len(tokens), chunk_size - chunk_overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        if chunk_tokens:
            chunks.append(" ".join(chunk_tokens))
    
    return chunks

async def store_document(
    ctx: RunContext[Any], 
    doc_id: Optional[str],
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> DocumentStorageOutput:
    """Store a document with metadata."""
    injector = get_injector()
    
    # Generate ID if not provided
    if not doc_id:
        doc_id = generate_doc_id(content)
    
    # Prepare metadata
    if metadata is None:
        metadata = {}
    
    metadata.update({
        "created_at": datetime.now().isoformat(),
        "content_length": len(content),
        "token_count": len(simple_tokenize(content))
    })
    
    # Store document content in KV storage
    try:
        # Store content
        await injector.run('storage_kv', {
            "operation": "set",
            "key": f"doc:content:{doc_id}",
            "value": content
        })
        
        # Store metadata
        await injector.run('storage_kv', {
            "operation": "set",
            "key": f"doc:meta:{doc_id}",
            "value": json.dumps(metadata)
        })
        
        # Add to document index
        doc_index = []
        try:
            doc_index_result = await injector.run('storage_kv', {
                "operation": "get",
                "key": "doc:index"
            })
            
            if hasattr(doc_index_result, 'output'):
                index_data = json.loads(doc_index_result.output)
                if (index_data.get('operation') == 'get' and 
                    index_data.get('data') and 
                    index_data['data'].get('exists')):
                    doc_index = json.loads(index_data['data'].get('value', '[]'))
        except KeyError:
            # Index doesn't exist yet - start with empty list
            doc_index = []
        
        if doc_id not in doc_index:
            doc_index.append(doc_id)
            await injector.run('storage_kv', {
                "operation": "set",
                "key": "doc:index",
                "value": json.dumps(doc_index)
            })
        
        return DocumentStorageOutput(
            success=True,
            message=f"Document stored successfully with ID: {doc_id}",
            data={"doc_id": doc_id, "metadata": metadata}
        )
    except Exception as e:
        raise RuntimeError(f"Failed to store document: {e}") from e

async def retrieve_document(ctx: RunContext[Any], doc_id: str) -> DocumentStorageOutput:
    """Retrieve a document by ID."""
    injector = get_injector()
    
    try:
        # Get content
        content_result = await injector.run('storage_kv', {
            "operation": "get",
            "key": f"doc:content:{doc_id}"
        })
        
        if hasattr(content_result, 'output'):
            content_data = json.loads(content_result.output)
            if (not content_data.get('operation') == 'get' or 
                not content_data.get('data') or 
                not content_data['data'].get('exists')):
                raise KeyError(f"Document not found: {doc_id}")
            content = content_data['data']['value']
        else:
            raise KeyError(f"Document not found: {doc_id}")
        
        # Get metadata
        meta_result = await injector.run('storage_kv', {
            "operation": "get",
            "key": f"doc:meta:{doc_id}"
        })
        
        metadata = {}
        if hasattr(meta_result, 'output'):
            meta_data = json.loads(meta_result.output)
            if (meta_data.get('operation') == 'get' and 
                meta_data.get('data') and 
                meta_data['data'].get('exists')):
                metadata = json.loads(meta_data['data']['value'])
        
        return DocumentStorageOutput(
            success=True,
            message=f"Document retrieved: {doc_id}",
            data={
                "doc_id": doc_id,
                "content": content,
                "metadata": metadata
            }
        )
    except KeyError as e:
        raise e
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve document: {e}") from e

async def chunk_document(
    ctx: RunContext[Any],
    doc_id: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> DocumentStorageOutput:
    """Chunk a document into smaller segments."""
    injector = get_injector()
    
    try:
        # Retrieve document
        doc_result = await retrieve_document(ctx, doc_id)
        doc_data = doc_result.data
        content = doc_data['content']
        metadata = doc_data['metadata']
        
        # Chunk the content
        chunks = chunk_text(content, chunk_size, chunk_overlap)
        
        # Create chunk objects
        document_chunks = []
        for i, chunk_content in enumerate(chunks):
            chunk = DocumentChunk(
                chunk_id=f"{doc_id}:chunk:{i}",
                doc_id=doc_id,
                content=chunk_content,
                chunk_index=i,
                metadata={
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap
                }
            )
            document_chunks.append(chunk.model_dump())
            
            # Store chunk in KV storage
            await injector.run('storage_kv', {
                "operation": "set",
                "key": f"doc:chunk:{doc_id}:{i}",
                "value": json.dumps(chunk.model_dump())
            })
        
        # Update document metadata with chunk info
        metadata['chunked'] = True
        metadata['chunk_count'] = len(chunks)
        metadata['chunk_size'] = chunk_size
        metadata['chunk_overlap'] = chunk_overlap
        
        await injector.run('storage_kv', {
            "operation": "set",
            "key": f"doc:meta:{doc_id}",
            "value": json.dumps(metadata)
        })
        
        return DocumentStorageOutput(
            success=True,
            message=f"Document chunked into {len(chunks)} segments",
            count=len(chunks),
            data={"doc_id": doc_id, "chunks": document_chunks}
        )
    except Exception as e:
        raise RuntimeError(f"Failed to chunk document: {e}") from e

async def list_documents(
    ctx: RunContext[Any],
    prefix: Optional[str] = None,
    limit: int = 100
) -> DocumentStorageOutput:
    """List all documents with optional prefix filter."""
    injector = get_injector()
    
    try:
        # Get document index
        index_result = await injector.run('storage_kv', {
            "operation": "get",
            "key": "doc:index"
        })
        
        doc_ids = []
        if hasattr(index_result, 'output'):
            index_data = json.loads(index_result.output)
            if (index_data.get('operation') == 'get' and 
                index_data.get('data') and 
                index_data['data'].get('exists')):
                doc_ids = json.loads(index_data['data'].get('value', '[]'))
        
        # Filter by prefix if provided
        if prefix:
            doc_ids = [doc_id for doc_id in doc_ids if doc_id.startswith(prefix)]
        
        # Apply limit
        doc_ids = doc_ids[:limit]
        
        # Get metadata for each document
        documents = []
        for doc_id in doc_ids:
            meta_result = await injector.run('storage_kv', {
                "operation": "get",
                "key": f"doc:meta:{doc_id}"
            })
            
            metadata = {}
            if hasattr(meta_result, 'output'):
                meta_data = json.loads(meta_result.output)
                if (meta_data.get('operation') == 'get' and 
                    meta_data.get('data') and 
                    meta_data['data'].get('exists')):
                    metadata = json.loads(meta_data['data']['value'])
            
            documents.append({
                "doc_id": doc_id,
                "metadata": metadata
            })
        
        return DocumentStorageOutput(
            success=True,
            message=f"Found {len(documents)} documents",
            count=len(documents),
            data={"documents": documents}
        )
    except Exception as e:
        raise RuntimeError(f"Failed to list documents: {e}") from e

async def delete_document(ctx: RunContext[Any], doc_id: str) -> DocumentStorageOutput:
    """Delete a document and its chunks."""
    injector = get_injector()
    
    try:
        # Check if document exists
        doc_result = await retrieve_document(ctx, doc_id)
        metadata = doc_result.data['metadata']
        
        # Delete content
        await injector.run('storage_kv', {
            "operation": "delete",
            "key": f"doc:content:{doc_id}"
        })
        
        # Delete metadata
        await injector.run('storage_kv', {
            "operation": "delete",
            "key": f"doc:meta:{doc_id}"
        })
        
        # Delete chunks if they exist
        if metadata.get('chunked'):
            chunk_count = metadata.get('chunk_count', 0)
            for i in range(chunk_count):
                await injector.run('storage_kv', {
                    "operation": "delete",
                    "key": f"doc:chunk:{doc_id}:{i}"
                })
        
        # Remove from index
        index_result = await injector.run('storage_kv', {
            "operation": "get",
            "key": "doc:index"
        })
        
        if hasattr(index_result, 'output'):
            index_data = json.loads(index_result.output)
            if (index_data.get('operation') == 'get' and 
                index_data.get('data') and 
                index_data['data'].get('exists')):
                doc_index = json.loads(index_data['data'].get('value', '[]'))
                if doc_id in doc_index:
                    doc_index.remove(doc_id)
                    await injector.run('storage_kv', {
                        "operation": "set",
                        "key": "doc:index",
                        "value": json.dumps(doc_index)
                    })
        
        return DocumentStorageOutput(
            success=True,
            message=f"Document deleted: {doc_id}",
            data={"doc_id": doc_id}
        )
    except Exception as e:
        raise RuntimeError(f"Failed to delete document: {e}") from e

async def update_metadata(
    ctx: RunContext[Any],
    doc_id: str,
    metadata: Dict[str, Any]
) -> DocumentStorageOutput:
    """Update document metadata."""
    injector = get_injector()
    
    try:
        # Get existing metadata
        meta_result = await injector.run('storage_kv', {
            "operation": "get",
            "key": f"doc:meta:{doc_id}"
        })
        
        existing_metadata = {}
        if hasattr(meta_result, 'output'):
            meta_data = json.loads(meta_result.output)
            if (meta_data.get('operation') == 'get' and 
                meta_data.get('data') and 
                meta_data['data'].get('exists')):
                existing_metadata = json.loads(meta_data['data']['value'])
        
        # Merge metadata
        existing_metadata.update(metadata)
        existing_metadata['updated_at'] = datetime.now().isoformat()
        
        # Store updated metadata
        await injector.run('storage_kv', {
            "operation": "set",
            "key": f"doc:meta:{doc_id}",
            "value": json.dumps(existing_metadata)
        })
        
        return DocumentStorageOutput(
            success=True,
            message=f"Metadata updated for document: {doc_id}",
            data={"doc_id": doc_id, "metadata": existing_metadata}
        )
    except Exception as e:
        raise RuntimeError(f"Failed to update metadata: {e}") from e

# Routing configuration
routing = RoutingConfig(
    operation_field='operation',
    operation_map={
        'store': ('store_document', lambda x: {
            'doc_id': x.doc_id,
            'content': x.content,
            'metadata': x.metadata
        }),
        'retrieve': ('retrieve_document', lambda x: {'doc_id': x.doc_id}),
        'chunk': ('chunk_document', lambda x: {
            'doc_id': x.doc_id,
            'chunk_size': x.chunk_size,
            'chunk_overlap': x.chunk_overlap
        }),
        'list': ('list_documents', lambda x: {
            'prefix': x.prefix,
            'limit': x.limit
        }),
        'delete': ('delete_document', lambda x: {'doc_id': x.doc_id}),
        'update_metadata': ('update_metadata', lambda x: {
            'doc_id': x.doc_id,
            'metadata': x.metadata
        })
    }
)

def create_storage_document_agent():
    """Create and return the document storage AgenTool."""
    return create_agentool(
        name='storage_document',
        input_schema=DocumentStorageInput,
        routing_config=routing,
        tools=[
            store_document, 
            retrieve_document, 
            chunk_document, 
            list_documents,
            delete_document,
            update_metadata
        ],
        output_type=DocumentStorageOutput,
        system_prompt="Manage document storage, retrieval, and chunking for RAG applications.",
        description="Document management with chunking support for RAG pipelines",
        version="1.0.0",
        tags=["storage", "document", "chunking", "rag", "text"],
        dependencies=["storage_kv"],
        examples=[
            {
                "input": {
                    "operation": "store",
                    "content": "This is a sample document.",
                    "metadata": {"title": "Sample"}
                },
                "output": {
                    "success": True,
                    "message": "Document stored successfully",
                    "data": {"doc_id": "abc123"}
                }
            },
            {
                "input": {
                    "operation": "chunk",
                    "doc_id": "abc123",
                    "chunk_size": 100,
                    "chunk_overlap": 20
                },
                "output": {
                    "success": True,
                    "message": "Document chunked into 5 segments",
                    "count": 5
                }
            }
        ]
    )

# Export
agent = create_storage_document_agent()

# TODO: Phase 2 Features
# - Add PDF parsing support
# - Add HTML parsing support
# - Use proper tokenizer (tiktoken) for accurate chunking
# - Implement semantic chunking based on content structure
# - Add document versioning

# TODO: Phase 3 Features
# - Incremental document updates
# - Document deduplication
# - Language detection and multi-lingual support
# - OCR support for scanned documents
# - Automatic metadata extraction