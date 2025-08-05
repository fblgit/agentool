"""
AgentToolkit - Foundational toolkit ecosystem for AgenTools.

This package provides a comprehensive set of foundational toolkits that can be
composed together to build sophisticated multi-agent systems. Each toolkit
follows the established patterns from the AgenTool framework.

Available Toolkits:
- storage.fs: File system operations
- storage.kv: Key-value storage with TTL support
- storage.vector: Vector storage with PGVector for RAG
- storage.document: Document storage and chunking for RAG
- system.config: Configuration management
- system.logging: Structured logging
- system.scheduler: Job scheduling and execution engine
- system.queue: Message queuing and data bus
- system.templates: Jinja2 template rendering with storage integration
- auth.auth: Authentication and authorization
- auth.session: Session management
- security.crypto: Cryptographic operations
- network.http: HTTP client operations
- observability.metrics: Metrics collection and tracking
- management.agentool: AgenTool registry management and introspection
- rag.embedder: OpenAI embeddings with caching
- rag.retriever: Document retrieval and context assembly
- graph.rag_graph: RAG pipeline orchestration
"""

__version__ = "1.0.0"
__author__ = "AgenTool Framework"

# Storage agents
from .storage.fs import create_storage_fs_agent
from .storage.kv import create_storage_kv_agent
from .storage.vector import agent as vector_storage_agent
from .storage.document import agent as document_storage_agent

# System agents
from .system.config import create_config_agent
from .system.logging import create_logging_agent
from .system.scheduler import create_scheduler_agent
from .system.queue import create_queue_agent
from .system.templates import create_templates_agent

# Auth agents
from .auth.auth import create_auth_agent
from .auth.session import create_session_agent

# Security agents
from .security.crypto import create_crypto_agent

# Network agents
from .network.http import create_http_agent

# Observability agents
from .observability.metrics import create_metrics_agent

# Management agents
from .management.agentool import create_agentool_management_agent

# RAG agents
from .rag.embedder import agent as embedder_agent
from .rag.retriever import agent as retriever_agent

# Graph agents
from .graph.rag_graph import agent as rag_graph_agent

__all__ = [
    # Storage
    'create_storage_fs_agent',
    'create_storage_kv_agent',
    'vector_storage_agent',
    'document_storage_agent',
    # System
    'create_config_agent',
    'create_logging_agent',
    'create_scheduler_agent',
    'create_queue_agent',
    'create_templates_agent',
    # Auth
    'create_auth_agent',
    'create_session_agent',
    # Security
    'create_crypto_agent',
    # Network
    'create_http_agent',
    # Observability
    'create_metrics_agent',
    # Management
    'create_agentool_management_agent',
    # RAG
    'embedder_agent',
    'retriever_agent',
    # Graph
    'rag_graph_agent'
]