"""
RAG Graph AgenTool - Orchestration of RAG pipelines using pydantic_graph.

This AgenTool provides graph-based orchestration for RAG workflows,
coordinating document indexing and question answering flows.
"""

import json
from typing import Dict, Any, Optional, List, Literal
from dataclasses import dataclass, field
from pydantic import BaseModel, Field as PydanticField
from pydantic_ai import RunContext, Agent
from pydantic_ai.messages import ModelMessage
from pydantic_graph import BaseNode, End, Graph, GraphRunContext
from agentool import create_agentool
from agentool.core.registry import RoutingConfig
from agentool.core.injector import get_injector

# Input/Output schemas for AgenTool
class RAGGraphInput(BaseModel):
    """Input schema for RAG graph operations."""
    operation: Literal['ask', 'index', 'index_batch', 'clear_cache']
    # For ask
    question: Optional[str] = PydanticField(None, description="Question to answer")
    # For index/index_batch
    documents: Optional[List[str]] = PydanticField(None, description="Documents to index")
    doc_ids: Optional[List[str]] = PydanticField(None, description="Document IDs")
    # Common options
    collection: str = PydanticField("default", description="Collection name")
    model: str = PydanticField("openai:gpt-4o", description="LLM model to use")

class RAGGraphOutput(BaseModel):
    """Output schema for RAG graph operations."""
    success: bool
    message: str
    answer: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    indexed_count: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

# Graph state management
@dataclass
class RAGState:
    """State for RAG graph execution."""
    question: Optional[str] = None
    context: Optional[str] = None
    sources: List[Dict[str, Any]] = field(default_factory=list)
    answer: Optional[str] = None
    agent_messages: List[ModelMessage] = field(default_factory=list)
    collection: str = "default"
    model: str = "openai:gpt-4o"

# Graph nodes
@dataclass
class RetrieveContext(BaseNode[RAGState]):
    """Retrieve context for the question."""
    
    async def run(self, ctx: GraphRunContext[RAGState]) -> "GenerateAnswer":
        injector = get_injector()
        
        # Retrieve context using retriever agentool
        result = await injector.run('rag_retriever', {
            "operation": "retrieve_context",
            "query": ctx.state.question,
            "collection": ctx.state.collection,
            "top_k": 5,
            "max_context_length": 2000
        })
        
        if hasattr(result, 'output'):
            result_data = json.loads(result.output)
            if result_data.get('success'):
                ctx.state.context = result_data.get('context', '')
                ctx.state.sources = result_data.get('metadata', {}).get('sources', [])
        
        return GenerateAnswer()

@dataclass
class GenerateAnswer(BaseNode[RAGState, None, Dict[str, Any]]):
    """Generate answer using LLM with retrieved context."""
    
    async def run(self, ctx: GraphRunContext[RAGState]) -> End[Dict[str, Any]]:
        # Create agent for answer generation
        answer_agent = Agent(
            ctx.state.model,
            system_prompt="""You are a helpful assistant that answers questions based on the provided context.

Instructions:
- Answer based only on the provided context
- If the context doesn't contain enough information, say so
- Be concise and accurate
- Cite sources when possible""",
            output_type=str
        )
        
        # Generate prompt with context
        prompt = f"""Context:
{ctx.state.context}

Question: {ctx.state.question}

Please provide a clear and accurate answer based on the context above."""
        
        # Generate answer
        result = await answer_agent.run(
            prompt,
            message_history=ctx.state.agent_messages
        )
        
        ctx.state.answer = result.output
        ctx.state.agent_messages += result.new_messages()
        
        # Return result
        return End({
            "answer": ctx.state.answer,
            "sources": ctx.state.sources,
            "context_used": ctx.state.context
        })

# Create the RAG graph
rag_answer_graph = Graph(
    nodes=[RetrieveContext, GenerateAnswer],
    state_type=RAGState
)

# AgenTool functions
async def ask(
    ctx: RunContext[Any],
    question: str,
    collection: str = "default",
    model: str = "openai:gpt-4o"
) -> RAGGraphOutput:
    """Answer a question using RAG."""
    try:
        # Initialize state
        state = RAGState(
            question=question,
            collection=collection,
            model=model
        )
        
        # Run the graph
        result = await rag_answer_graph.run(
            RetrieveContext(),
            state=state
        )
        
        # Extract output
        output_data = result.output
        
        return RAGGraphOutput(
            success=True,
            message="Question answered successfully",
            answer=output_data.get("answer"),
            sources=output_data.get("sources"),
            metadata={
                "question": question,
                "collection": collection,
                "model": model,
                "context_length": len(output_data.get("context_used") or "")
            }
        )
    except Exception as e:
        raise RuntimeError(f"Failed to answer question: {e}") from e

async def index(
    ctx: RunContext[Any],
    documents: List[str],
    doc_ids: Optional[List[str]],
    collection: str = "default"
) -> RAGGraphOutput:
    """Index documents for RAG."""
    injector = get_injector()
    
    if not documents:
        raise ValueError("No documents provided for indexing")
    
    # Generate IDs if not provided
    if not doc_ids:
        import hashlib
        doc_ids = [
            hashlib.sha256(doc.encode()).hexdigest()[:16]
            for doc in documents
        ]
    
    if len(documents) != len(doc_ids):
        raise ValueError("Number of documents must match number of IDs")
    
    try:
        indexed_count = 0
        failed_docs = []
        
        for doc_content, doc_id in zip(documents, doc_ids):
            try:
                # Store document
                await injector.run('storage_document', {
                    "operation": "store",
                    "doc_id": doc_id,
                    "content": doc_content,
                    "metadata": {"collection": collection}
                })
                
                # Index document for retrieval
                await injector.run('rag_retriever', {
                    "operation": "index_document",
                    "doc_id": doc_id,
                    "collection": collection
                })
                
                indexed_count += 1
            except Exception as e:
                failed_docs.append({"doc_id": doc_id, "error": str(e)})
        
        return RAGGraphOutput(
            success=True,
            message=f"Indexed {indexed_count} documents ({len(failed_docs)} failed)",
            indexed_count=indexed_count,
            metadata={
                "collection": collection,
                "total_docs": len(documents),
                "failed_docs": failed_docs
            }
        )
    except Exception as e:
        raise RuntimeError(f"Failed to index documents: {e}") from e

async def index_batch(
    ctx: RunContext[Any],
    documents: List[str],
    doc_ids: Optional[List[str]],
    collection: str = "default"
) -> RAGGraphOutput:
    """Index documents in batch (same as index for now)."""
    # TODO: Phase 2 - Optimize batch indexing with parallel processing
    return await index(ctx, documents, doc_ids, collection)

async def clear_cache(ctx: RunContext[Any], collection: str = "default") -> RAGGraphOutput:
    """Clear embedding cache and optionally vector storage."""
    injector = get_injector()
    
    try:
        # Clear embedding cache
        embed_result = await injector.run('rag_embedder', {
            "operation": "clear_cache",
            "cache_key_prefix": "embed"
        })
        
        cache_cleared = 0
        if hasattr(embed_result, 'output'):
            embed_data = json.loads(embed_result.output)
            if embed_data.get('success'):
                cache_cleared = embed_data.get('cache_hits', 0)
        
        # TODO: Phase 2 - Add option to clear vector storage collection
        
        return RAGGraphOutput(
            success=True,
            message=f"Cleared {cache_cleared} cached embeddings",
            metadata={"collection": collection, "cache_cleared": cache_cleared}
        )
    except Exception as e:
        raise RuntimeError(f"Failed to clear cache: {e}") from e

# Routing configuration
routing = RoutingConfig(
    operation_field='operation',
    operation_map={
        'ask': ('ask', lambda x: {
            'question': x.question,
            'collection': x.collection,
            'model': x.model
        }),
        'index': ('index', lambda x: {
            'documents': x.documents,
            'doc_ids': x.doc_ids,
            'collection': x.collection
        }),
        'index_batch': ('index_batch', lambda x: {
            'documents': x.documents,
            'doc_ids': x.doc_ids,
            'collection': x.collection
        }),
        'clear_cache': ('clear_cache', lambda x: {
            'collection': x.collection
        })
    }
)

def create_graph_rag_agent():
    """Create and return the RAG graph AgenTool."""
    return create_agentool(
        name='graph_rag',
        input_schema=RAGGraphInput,
        routing_config=routing,
        tools=[ask, index, index_batch, clear_cache],
        output_type=RAGGraphOutput,
        system_prompt="Orchestrate RAG pipelines for document indexing and question answering.",
        description="Graph-based RAG orchestration using pydantic_graph",
        version="1.0.0",
        tags=["rag", "graph", "orchestration", "pipeline", "qa"],
        dependencies=["rag_retriever", "storage_document", "rag_embedder"],
        examples=[
            {
                "input": {
                    "operation": "ask",
                    "question": "What is RAG?",
                    "collection": "documents"
                },
                "output": {
                    "success": True,
                    "message": "Question answered successfully",
                    "answer": "RAG stands for Retrieval-Augmented Generation..."
                }
            },
            {
                "input": {
                    "operation": "index",
                    "documents": ["RAG is a technique..."],
                    "doc_ids": ["doc1"],
                    "collection": "documents"
                },
                "output": {
                    "success": True,
                    "message": "Indexed 1 documents",
                    "indexed_count": 1
                }
            }
        ]
    )

# Export
agent = create_graph_rag_agent()

# TODO: Phase 2 Features
# - Add query refinement node
# - Implement answer validation node
# - Add feedback loop for improving answers
# - Support for conversation history
# - Parallel document indexing

# TODO: Phase 3 Features
# - Multi-step reasoning graphs
# - Dynamic graph construction based on query type
# - Graph state persistence and recovery
# - Distributed graph execution
# - Advanced answer synthesis with citations