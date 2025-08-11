"""GraphToolkit Domain Definitions.

This module contains domain-specific workflow definitions for various use cases.
Each domain follows the same meta-framework pattern with 4-phase workflows.
"""

# Import domain modules to register phases
from . import agentool, api, blockchain, documentation, workflow, smoke

# Export domain names for easy reference
AVAILABLE_DOMAINS = [
    'agentool',      # AgenTool creation workflow
    'api',           # API design workflow
    'workflow',      # Workflow orchestration
    'documentation', # Documentation generation
    'blockchain',    # Smart contract development
    'smoke'          # Lightweight E2E testing domain
]

# Export phase names by domain
DOMAIN_PHASES = {
    'agentool': ['analyzer', 'specifier', 'crafter', 'evaluator'],
    'api': ['analyzer', 'designer', 'generator', 'validator'],
    'workflow': ['analyzer', 'step_designer', 'orchestrator', 'tester'],
    'documentation': ['content_analyzer', 'structure_designer', 'writer', 'reviewer'],
    'blockchain': ['contract_analyzer', 'smart_contract_designer', 'auditor', 'optimizer'],
    'smoke': ['ingredient_analyzer', 'recipe_designer', 'recipe_crafter', 'recipe_evaluator']
}

# Helper function to get all phases for a domain
def get_domain_phases(domain: str):
    """Get all phase names for a domain.
    
    Args:
        domain: Domain name
        
    Returns:
        List of phase names for the domain
    """
    if domain not in DOMAIN_PHASES:
        raise ValueError(f'Unknown domain: {domain}. Available: {AVAILABLE_DOMAINS}')
    return DOMAIN_PHASES[domain]


# Helper function to build workflow definition
def build_workflow_definition(domain: str):
    """Build a complete workflow definition for a domain.
    
    Args:
        domain: Domain name
        
    Returns:
        WorkflowDefinition for the domain
    """
    from datetime import datetime

    from ..core.registry import get_phase
    from ..core.types import NodeConfig, RetryBackoff, WorkflowDefinition
    
    if domain not in DOMAIN_PHASES:
        raise ValueError(f'Unknown domain: {domain}. Available: {AVAILABLE_DOMAINS}')
    
    phases = {}
    phase_sequence = []
    
    for phase_name in DOMAIN_PHASES[domain]:
        phase_key = f'{domain}.{phase_name}'
        phase_def = get_phase(phase_key)
        if phase_def:
            phases[phase_name] = phase_def
            phase_sequence.append(phase_name)
    
    # Default node configurations
    node_configs = {
        # Storage operations - usually not retryable (local)
        'dependency_check': NodeConfig(
            node_type='dependency_check',
            retryable=False,
            max_retries=0
        ),
        'load_dependencies': NodeConfig(
            node_type='storage_load',
            retryable=True,  # Can retry on transient storage issues
            max_retries=2,
            retry_backoff=RetryBackoff.LINEAR
        ),
        'save_phase_output': NodeConfig(
            node_type='storage_save',
            retryable=True,
            max_retries=2,
            retry_backoff=RetryBackoff.LINEAR
        ),
        
        # LLM operations - retryable (API can have transient failures)
        'llm_call': NodeConfig(
            node_type='llm_inference',
            retryable=True,
            max_retries=3,
            retry_backoff=RetryBackoff.EXPONENTIAL,
            retry_delay=2.0
        ),
        
        # Deterministic operations - no retry needed
        'template_render': NodeConfig(
            node_type='template',
            retryable=False,
            max_retries=0
        ),
        'schema_validation': NodeConfig(
            node_type='validation',
            retryable=False,
            max_retries=0
        ),
        'state_update': NodeConfig(
            node_type='state_update',
            retryable=False,
            max_retries=0
        ),
        
        # Control flow - no retry
        'quality_gate': NodeConfig(
            node_type='quality_gate',
            retryable=False,
            max_retries=0
        ),
        
        # Specifier-specific nodes
        'prepare_specifier_iteration': NodeConfig(
            node_type='prepare_specifier_iteration',
            retryable=True,  # Can retry on storage/agentool_mgmt failures
            max_retries=2,
            retry_backoff=RetryBackoff.LINEAR
        ),
        'specifier_tool_iterator': NodeConfig(
            node_type='specifier_tool_iterator',
            retryable=True,  # Can retry individual tools, but handles failures gracefully
            max_retries=1,   # Limited retries since it handles failures per-tool
            retry_backoff=RetryBackoff.LINEAR,
            iter_enabled=True  # This is an iteration node
        ),
        
        # Generic aliases
        'save_output': NodeConfig(
            node_type='storage_save',
            retryable=True,
            max_retries=2,
            retry_backoff=RetryBackoff.LINEAR
        ),
        'process_tools': NodeConfig(  # Legacy compatibility
            node_type='process_tools',
            retryable=True,
            max_retries=2,
            iter_enabled=True
        )
    }
    
    return WorkflowDefinition(
        domain=domain,
        phases=phases,
        phase_sequence=phase_sequence,
        node_configs=node_configs,
        enable_refinement=True,
        enable_parallel=False,  # Will be enabled in Phase 6
        max_execution_time=3600,
        version='1.0.0',
        created_at=datetime.now()
    )


__all__ = [
    'AVAILABLE_DOMAINS',
    'DOMAIN_PHASES',
    'get_domain_phases',
    'build_workflow_definition'
]