"""
Test helpers for GraphToolkit tests.

Provides proper initialization of all agentoolkit components for testing.
"""

from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry


def setup_agentoolkit_components():
    """
    Initialize all agentoolkit components needed for GraphToolkit tests.
    
    This ensures storage_kv, storage_fs, metrics, logging, and templates
    are properly initialized and available.
    """
    # Clear registry and injector to start fresh
    AgenToolRegistry.clear()
    get_injector().clear()
    
    # Initialize storage components
    from agentoolkit.storage.kv import create_storage_kv_agent, _kv_storage, _kv_expiry
    from agentoolkit.storage.fs import create_storage_fs_agent
    
    # Clear global storage state
    _kv_storage.clear()
    _kv_expiry.clear()
    
    # Create and register storage agents
    kv_agent = create_storage_kv_agent()
    fs_agent = create_storage_fs_agent()
    
    # Initialize metrics
    try:
        from agentoolkit.metrics import create_metrics_agent
        metrics_agent = create_metrics_agent()
    except ImportError:
        # Metrics might not be fully implemented
        pass
    
    # Initialize logging
    try:
        from agentoolkit.logging import create_logging_agent
        logging_agent = create_logging_agent()
    except ImportError:
        # Logging might not be fully implemented
        pass
    
    # Initialize template engine
    try:
        from agentoolkit.templates import create_template_agent
        template_agent = create_template_agent()
    except ImportError:
        # Templates might not be fully implemented
        pass
    
    # Initialize observability
    try:
        from agentoolkit.observability import create_observability_agent
        observability_agent = create_observability_agent()
    except ImportError:
        # Observability might not be fully implemented
        pass
    
    return get_injector()


def teardown_agentoolkit_components():
    """Clean up agentoolkit components after tests."""
    # Clear storage
    try:
        from agentoolkit.storage.kv import _kv_storage, _kv_expiry
        _kv_storage.clear()
        _kv_expiry.clear()
    except ImportError:
        pass
    
    # Clear registry and injector
    AgenToolRegistry.clear()
    get_injector().clear()