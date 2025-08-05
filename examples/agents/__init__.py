"""
AgenTool example agents.

This module contains example AgenTool agents that demonstrate
how to combine schemas, tools, and routing configurations into
complete working agents.
"""

from .storage import create_storage_agent

__all__ = ['create_storage_agent']