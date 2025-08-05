"""
AgenTool Workflow Components for AI Code Generation.

This module provides a complete workflow for generating new AgenTools
by analyzing requirements, creating specifications, crafting implementations,
evaluating the results, and generating comprehensive test suites.
"""

from .workflow_analyzer import create_workflow_analyzer_agent
from .workflow_specifier import create_workflow_specifier_agent
from .workflow_crafter import create_workflow_crafter_agent
from .workflow_evaluator import create_workflow_evaluator_agent
from .workflow_test_analyzer import create_workflow_test_analyzer_agent
from .workflow_test_stubber import create_workflow_test_stubber_agent
from .workflow_test_crafter import create_workflow_test_crafter_agent

__all__ = [
    'create_workflow_analyzer_agent',
    'create_workflow_specifier_agent',
    'create_workflow_crafter_agent',
    'create_workflow_evaluator_agent',
    'create_workflow_test_analyzer_agent',
    'create_workflow_test_stubber_agent',
    'create_workflow_test_crafter_agent'
]

# Auto-create all agents when module is imported
def initialize_workflow_agents():
    """Initialize all workflow AgenTools and register them."""
    agents = [
        create_workflow_analyzer_agent(),
        create_workflow_specifier_agent(),
        create_workflow_crafter_agent(),
        create_workflow_evaluator_agent(),
        create_workflow_test_analyzer_agent(),
        create_workflow_test_stubber_agent(),
        create_workflow_test_crafter_agent()
    ]
    return agents