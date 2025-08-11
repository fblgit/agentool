"""
Test helpers for GraphToolkit tests.

Provides proper initialization of all agentoolkit components for testing.
Uses the centralized GraphToolkit initialization system.
"""

from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry


def setup_agentoolkit_components():
    """
    Initialize all agentoolkit components needed for GraphToolkit tests.
    
    Uses the centralized GraphToolkit initialization system with test configuration.
    """
    from graphtoolkit.core.initialization import (
        initialize_graphtoolkit,
        test_config,
        reset_graphtoolkit
    )
    
    # Reset completely to start fresh
    reset_graphtoolkit()
    
    # Initialize with test configuration
    initialize_graphtoolkit(test_config())
    
    return get_injector()


def teardown_agentoolkit_components():
    """Clean up agentoolkit components after tests."""
    from graphtoolkit.core.initialization import cleanup_graphtoolkit, reset_graphtoolkit
    
    # Use centralized cleanup
    cleanup_graphtoolkit()
    
    # Complete reset for next test
    reset_graphtoolkit()