"""GraphToolkit - State-Driven Meta-Framework for AI Workflows.

A powerful framework for building domain-agnostic, AI-powered workflows
using atomic node decomposition and state-driven execution.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

# Import nodes to trigger registration
from . import nodes
from .core.deps import ModelConfig, StorageConfig, WorkflowDeps
from .core.registry import get_registry
from .core.types import (
    NodeConfig,
    PhaseDefinition,
    StorageRef,
    TokenUsage,
    ValidationResult,
    WorkflowDefinition,
    WorkflowState,
)
from .domains import AVAILABLE_DOMAINS, DOMAIN_PHASES, build_workflow_definition

logger = logging.getLogger(__name__)

__version__ = '1.0.0'


# Public API Types
class WorkflowResult:
    """Result from workflow execution."""
    def __init__(
        self,
        state: WorkflowState,
        outputs: Dict[str, Any],
        success: bool = True,
        error: Optional[str] = None
    ):
        self.state = state
        self.outputs = outputs
        self.success = success
        self.error = error
        self.workflow_id = state.workflow_id
        self.domain = state.domain
        
    @property
    def generated_code(self) -> Optional[str]:
        """Get generated code for AgenTool domain."""
        if self.domain == 'agentool':
            return self.outputs.get('crafter', {}).get('code')
        return None
    
    @property
    def final_output(self) -> Any:
        """Get the final phase output."""
        if self.state.completed_phases:
            last_phase = list(self.state.completed_phases)[-1]
            return self.outputs.get(last_phase)
        return None


class Workflow:
    """Workflow instance ready for execution."""
    def __init__(
        self,
        definition: WorkflowDefinition,
        state: WorkflowState,
        deps: Optional[WorkflowDeps] = None
    ):
        self.definition = definition
        self.state = state
        self.deps = deps or WorkflowDeps.create_default()
        
    async def run(self) -> WorkflowResult:
        """Execute the workflow."""
        from .core.executor import WorkflowExecutor
        executor = WorkflowExecutor(self.deps)
        return await executor.run(self.state)


# Main Public API Functions

def create_workflow(
    domain: str,
    input_data: Dict[str, Any],
    workflow_id: Optional[str] = None,
    phases: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Workflow:
    """Create a workflow instance for execution.
    
    Args:
        domain: Domain name (agentool, api, workflow, documentation, blockchain)
        input_data: Initial input data for the workflow
        workflow_id: Optional workflow ID (auto-generated if not provided)
        phases: Optional list of phases to run (default: all domain phases)
        config: Optional configuration for dependencies
        
    Returns:
        Workflow instance ready for execution
        
    Example:
        >>> workflow = create_workflow(
        ...     domain="agentool",
        ...     input_data={"catalog": {...}, "missing_tools": [...]}
        ... )
        >>> result = await workflow.run()
    """
    if domain not in AVAILABLE_DOMAINS:
        raise ValueError(f'Unknown domain: {domain}. Available: {AVAILABLE_DOMAINS}')
    
    # Build workflow definition
    workflow_def = build_workflow_definition(domain)
    
    # Filter phases if specified
    if phases:
        workflow_def = _filter_phases(workflow_def, phases)
    
    # Create workflow state
    workflow_id = workflow_id or f"{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    state = WorkflowState(
        workflow_id=workflow_id,
        domain=domain,
        workflow_def=workflow_def,
        current_phase=workflow_def.phase_sequence[0] if workflow_def.phase_sequence else '',
        domain_data={'input': input_data}
    )
    
    # Create dependencies
    deps = WorkflowDeps.from_config(config) if config else WorkflowDeps.create_default()
    
    # Populate phase registry in deps
    deps = _update_deps_registry(deps, get_registry())
    
    return Workflow(workflow_def, state, deps)


async def execute_workflow(
    workflow: Workflow,
    timeout: Optional[int] = None
) -> WorkflowResult:
    """Execute a workflow instance.
    
    Args:
        workflow: Workflow instance to execute
        timeout: Optional timeout in seconds
        
    Returns:
        WorkflowResult with outputs from all phases
        
    Example:
        >>> workflow = create_workflow("api", {"requirements": "REST API"})
        >>> result = await execute_workflow(workflow)
        >>> print(result.outputs)
    """
    if timeout:
        return await asyncio.wait_for(workflow.run(), timeout=timeout)
    else:
        return await workflow.run()


async def run_domain_workflow(
    domain: str,
    input_data: Dict[str, Any],
    phases: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
    timeout: Optional[int] = None
) -> WorkflowResult:
    """High-level API to create and run a domain workflow in one call.
    
    Args:
        domain: Domain name (agentool, api, workflow, documentation, blockchain)
        input_data: Input data for the workflow
        phases: Optional list of phases to run (default: all)
        config: Optional configuration dict
        timeout: Optional timeout in seconds
        
    Returns:
        WorkflowResult with outputs from all phases
        
    Example:
        >>> result = await run_domain_workflow(
        ...     "blockchain",
        ...     {"requirements": "ERC20 token with vesting"},
        ...     phases=["contract_analyzer", "smart_contract_designer"]
        ... )
        >>> print(result.outputs["smart_contract_designer"]["contract_code"])
    """
    workflow = create_workflow(domain, input_data, phases=phases, config=config)
    return await execute_workflow(workflow, timeout=timeout)


# Helper Functions

def _filter_phases(
    workflow_def: WorkflowDefinition,
    phases: List[str]
) -> WorkflowDefinition:
    """Filter workflow definition to only include specified phases."""
    from dataclasses import replace
    
    filtered_phases = {
        name: phase_def
        for name, phase_def in workflow_def.phases.items()
        if name in phases
    }
    
    filtered_sequence = [
        phase for phase in workflow_def.phase_sequence
        if phase in phases
    ]
    
    return replace(
        workflow_def,
        phases=filtered_phases,
        phase_sequence=filtered_sequence
    )


def _update_deps_registry(
    deps: WorkflowDeps,
    registry: Dict[str, PhaseDefinition]
) -> WorkflowDeps:
    """Update dependencies with phase registry."""
    from dataclasses import replace
    return replace(deps, phase_registry=registry)


def list_domains() -> List[str]:
    """List all available domains.
    
    Returns:
        List of domain names
        
    Example:
        >>> domains = list_domains()
        >>> print(domains)
        ['agentool', 'api', 'workflow', 'documentation', 'blockchain']
    """
    return AVAILABLE_DOMAINS


def list_domain_phases(domain: str) -> List[str]:
    """List all phases for a domain.
    
    Args:
        domain: Domain name
        
    Returns:
        List of phase names
        
    Example:
        >>> phases = list_domain_phases("agentool")
        >>> print(phases)
        ['analyzer', 'specifier', 'crafter', 'evaluator']
    """
    if domain not in DOMAIN_PHASES:
        raise ValueError(f'Unknown domain: {domain}')
    return DOMAIN_PHASES[domain]


def get_workflow_status(workflow: Workflow) -> Dict[str, Any]:
    """Get current status of a workflow.
    
    Args:
        workflow: Workflow instance
        
    Returns:
        Status dictionary
        
    Example:
        >>> status = get_workflow_status(workflow)
        >>> print(status["completed_phases"])
        {'analyzer', 'specifier'}
    """
    return {
        'workflow_id': workflow.state.workflow_id,
        'domain': workflow.state.domain,
        'current_phase': workflow.state.current_phase,
        'completed_phases': list(workflow.state.completed_phases),
        'quality_scores': workflow.state.quality_scores,
        'refinement_counts': workflow.state.refinement_count,
        'has_errors': 'error' in workflow.state.domain_data
    }


# Export main components
__all__ = [
    # Main API
    'create_workflow',
    'execute_workflow',
    'run_domain_workflow',
    
    # Types
    'Workflow',
    'WorkflowResult',
    'WorkflowState',
    'WorkflowDefinition',
    'WorkflowDeps',
    
    # Utilities
    'list_domains',
    'list_domain_phases',
    'get_workflow_status',
    
    # Constants
    'AVAILABLE_DOMAINS',
    '__version__'
]