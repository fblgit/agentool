"""LLM Agents for various workflows.

This package provides agents for:
- RAG (Retrieval-Augmented Generation) workflows
- AI Code Generation workflows for AgenTools
"""

from .rag_agent import rag_agent

# AI Code Generation Workflow imports
from .models import (
    AnalyzerOutput,
    SpecificationOutput,
    CodeOutput,
    ValidationOutput,
    WorkflowMetadata
)

# from .base import BaseAgenToolAgent

# Old agent implementations - replaced by workflow AgenTools
# from .analyzer_agent import AnalyzerAgent, create_analyzer_agent
# from .specification_agent import SpecificationAgent, create_specification_agent
# from .crafter_agent import CrafterAgent, create_crafter_agent
# from .evaluator_agent import EvaluatorAgent, create_evaluator_agent

from .workflow import (
    WorkflowState,
    AnalyzerNode,
    SpecificationNode,
    CrafterNode,
    EvaluatorNode,
    agentool_generation_graph,
    run_agentool_generation_workflow
)

__all__ = [
    # RAG
    "rag_agent",
    
    # AI Code Generation Models
    'AnalyzerOutput',
    'SpecificationOutput',
    'CodeOutput',
    'ValidationOutput',
    'WorkflowMetadata',
    
    # Base
    # 'BaseAgenToolAgent',
    
    # Agents - replaced by workflow AgenTools
    # 'AnalyzerAgent',
    # 'create_analyzer_agent',
    # 'SpecificationAgent',
    # 'create_specification_agent',
    # 'CrafterAgent',
    # 'create_crafter_agent',
    # 'EvaluatorAgent',
    # 'create_evaluator_agent',
    
    # Workflow
    'WorkflowState',
    'AnalyzerNode',
    'SpecificationNode',
    'CrafterNode',
    'EvaluatorNode',
    'agentool_generation_graph',
    'run_agentool_generation_workflow'
]