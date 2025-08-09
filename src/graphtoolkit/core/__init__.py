"""GraphToolkit Core Module.

Core types, registry, and factory for the meta-framework.
"""

from .factory import (
    build_domain_workflow,
    create_domain_workflow,
    create_node_instance,
    create_workflow_graph,
    create_workflow_state,
    register_node_class,
    update_state_with_result,
    validate_workflow_definition,
)
from .registry import PHASE_REGISTRY, PhaseRegistry, get_registry
from .types import (
    ConditionConfig,
    ConditionOperator,
    ModelParameters,
    NodeConfig,
    PhaseDefinition,
    ProcessingState,
    RefinementRecord,
    # Enums
    RetryBackoff,
    # Core types
    StorageRef,
    StorageType,
    TemplateConfig,
    TokenUsage,
    ValidationResult,
    WorkflowDefinition,
    WorkflowState,
)

__all__ = [
    # Types
    'RetryBackoff',
    'ConditionOperator', 
    'StorageType',
    'StorageRef',
    'TemplateConfig',
    'ModelParameters',
    'NodeConfig',
    'ConditionConfig',
    'PhaseDefinition',
    'WorkflowDefinition',
    'ValidationResult',
    'RefinementRecord',
    'WorkflowState',
    'TokenUsage',
    'ProcessingState',
    
    # Registry
    'PhaseRegistry',
    'PHASE_REGISTRY',
    'get_registry',
    
    # Factory
    'register_node_class',
    'create_node_instance',
    'create_workflow_state',
    'build_domain_workflow',
    'create_workflow_graph',
    'create_domain_workflow',
    'update_state_with_result',
    'validate_workflow_definition'
]