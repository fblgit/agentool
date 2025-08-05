"""
RAG Agent - Main LLM agent for RAG workflows.

This agent provides high-level RAG capabilities with customizable system prompts
loaded from Jinja2 templates.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from agentool.core.injector import get_injector

@dataclass
class RAGAgentDeps:
    """Dependencies for RAG agent."""
    collection: str = "default"
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "openai:gpt-4o"
    max_context_length: int = 2000
    top_k: int = 5

class RAGQuestion(BaseModel):
    """Input schema for RAG questions."""
    question: str
    include_sources: bool = True

class RAGAnswer(BaseModel):
    """Output schema for RAG answers."""
    answer: str
    confidence: float
    sources: Optional[list] = None

def load_template(template_name: str, **kwargs) -> str:
    """Load and render a Jinja2 template."""
    templates_dir = Path(__file__).parent.parent / "templates"
    
    # Create templates directory if it doesn't exist
    templates_dir.mkdir(exist_ok=True)
    
    # Check if template exists, if not create a default one
    template_path = templates_dir / template_name
    if not template_path.exists():
        # Create default RAG template
        if template_name == "rag_system.j2":
            default_template = """You are a helpful assistant that answers questions based on retrieved context.

{% if instructions %}
Additional Instructions:
{{ instructions }}
{% endif %}

Guidelines:
- Answer based only on the provided context
- If the context doesn't contain enough information, say so clearly
- Be concise and accurate
- Cite sources when possible
- Maintain a {{ tone | default('professional') }} tone

{% if domain %}
Domain Expertise: {{ domain }}
{% endif %}"""
            template_path.write_text(default_template)
    
    # Load and render template
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template(template_name)
    return template.render(**kwargs)

# Create the main RAG agent
rag_agent = Agent(
    'openai:gpt-4o',
    deps_type=RAGAgentDeps,
    output_type=RAGAnswer,
    system_prompt=load_template(
        'rag_system.j2',
        tone='helpful and professional',
        domain='general knowledge'
    )
)

@rag_agent.system_prompt
async def add_collection_info(ctx: RunContext[RAGAgentDeps]) -> str:
    """Add collection information to system prompt."""
    return f"You are searching in the '{ctx.deps.collection}' collection."

@rag_agent.tool
async def retrieve_context(
    ctx: RunContext[RAGAgentDeps],
    query: str
) -> Dict[str, Any]:
    """Retrieve relevant context for a query."""
    injector = get_injector()
    
    # Use retriever agentool
    result = await injector.run('retriever', {
        "operation": "retrieve_context",
        "query": query,
        "collection": ctx.deps.collection,
        "top_k": ctx.deps.top_k,
        "max_context_length": ctx.deps.max_context_length
    })
    
    if hasattr(result, 'output'):
        import json
        result_data = json.loads(result.output)
        if result_data.get('success'):
            return {
                "context": result_data.get('context', ''),
                "sources": result_data.get('metadata', {}).get('sources', [])
            }
    
    return {"context": "", "sources": []}

@rag_agent.tool
async def search_similar(
    ctx: RunContext[RAGAgentDeps],
    query: str,
    top_k: Optional[int] = None
) -> list:
    """Search for similar documents."""
    injector = get_injector()
    
    # Use retriever for search
    result = await injector.run('retriever', {
        "operation": "search",
        "query": query,
        "collection": ctx.deps.collection,
        "top_k": top_k or ctx.deps.top_k,
        "include_metadata": True
    })
    
    if hasattr(result, 'output'):
        import json
        result_data = json.loads(result.output)
        if result_data.get('success'):
            return result_data.get('results', [])
    
    return []

# Helper functions for direct usage
async def ask_rag(
    question: str,
    collection: str = "default",
    model: str = "openai:gpt-4o"
) -> RAGAnswer:
    """Ask a question using RAG."""
    deps = RAGAgentDeps(
        collection=collection,
        llm_model=model
    )
    
    # First retrieve context
    injector = get_injector()
    context_result = await injector.run('retriever', {
        "operation": "retrieve_context",
        "query": question,
        "collection": collection,
        "top_k": deps.top_k,
        "max_context_length": deps.max_context_length
    })
    
    context = ""
    sources = []
    if hasattr(context_result, 'output'):
        import json
        context_data = json.loads(context_result.output)
        if context_data.get('success'):
            context = context_data.get('context', '')
            sources = context_data.get('metadata', {}).get('sources', [])
    
    # Generate answer with context
    prompt = f"""Context:
{context}

Question: {question}

Please provide a clear, accurate answer based on the context above. Rate your confidence (0-1) based on how well the context supports your answer."""
    
    result = await rag_agent.run(prompt, deps=deps)
    
    # Extract confidence from answer if not provided
    if not hasattr(result.output, 'confidence'):
        result.output.confidence = 0.8  # Default confidence
    
    if sources:
        result.output.sources = sources
    
    return result.output

async def index_documents(
    documents: list,
    doc_ids: Optional[list] = None,
    collection: str = "default"
) -> Dict[str, Any]:
    """Index documents for RAG."""
    injector = get_injector()
    
    # Use rag_graph for indexing
    result = await injector.run('rag_graph', {
        "operation": "index",
        "documents": documents,
        "doc_ids": doc_ids,
        "collection": collection
    })
    
    if hasattr(result, 'output'):
        import json
        return json.loads(result.output)
    
    return {"success": False, "message": "Indexing failed"}

# Example usage functions
async def create_custom_rag_agent(
    system_template: str = "rag_system.j2",
    **template_kwargs
) -> Agent:
    """Create a custom RAG agent with specific configuration."""
    return Agent(
        'openai:gpt-4o',
        deps_type=RAGAgentDeps,
        output_type=RAGAnswer,
        system_prompt=load_template(system_template, **template_kwargs)
    )

# Export main agent
__all__ = [
    'rag_agent',
    'RAGAgentDeps',
    'RAGQuestion',
    'RAGAnswer',
    'ask_rag',
    'index_documents',
    'create_custom_rag_agent',
    'load_template'
]