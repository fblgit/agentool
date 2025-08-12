"""GraphToolkit Factory System.

Factory functions for creating graphs, nodes, and workflows from definitions.
"""

import logging
from dataclasses import replace
from typing import Any, Dict, List, Optional, Type

try:
    from pydantic_graph import BaseNode, End, Graph, GraphRunContext
    HAS_PYDANTIC_GRAPH = True
except ImportError:
    # Define stubs for development without pydantic_graph
    HAS_PYDANTIC_GRAPH = False
    
    class Graph:
        def __init__(self, nodes: List[Any]):
            self.nodes = nodes
    
    class BaseNode:
        pass
    
    class End:
        def __init__(self, result: Any):
            self.result = result
    
    class GraphRunContext:
        def __init__(self, state: Any, deps: Any):
            self.state = state
            self.deps = deps

from .registry import PHASE_REGISTRY
from .types import StorageRef, WorkflowDefinition, WorkflowState

logger = logging.getLogger(__name__)


# Node class registry - will be populated by node modules
NODE_CLASSES: Dict[str, Type[BaseNode]] = {}


def get_node_registry() -> Dict[str, Type[BaseNode]]:
    """Get the global node class registry."""
    return NODE_CLASSES


def register_node_class(node_id: str, node_class: Type[BaseNode]) -> None:
    """Register a node class for instantiation.
    
    Args:
        node_id: Node identifier
        node_class: Node class to register
    """
    NODE_CLASSES[node_id] = node_class
    logger.debug(f'Registered node class: {node_id} -> {node_class.__name__}')


def create_node_instance(node_id: str, **kwargs) -> BaseNode:
    """Create a node instance from ID.
    
    Args:
        node_id: Node identifier
        **kwargs: Additional arguments for node constructor
        
    Returns:
        Node instance
        
    Raises:
        ValueError: If node ID is not registered
    """
    if node_id not in NODE_CLASSES:
        # Try to import the node module to trigger registration
        _try_import_node(node_id)
    
    if node_id not in NODE_CLASSES:
        raise ValueError(f'Unknown node ID: {node_id}. Available: {list(NODE_CLASSES.keys())}')
    
    node_class = NODE_CLASSES[node_id]
    return node_class(**kwargs)


def _try_import_node(node_id: str) -> None:
    """Try to import a node module to trigger registration."""
    # Map node IDs to module paths
    module_map = {
        # Storage nodes
        'dependency_check': 'graphtoolkit.nodes.atomic.storage',
        'load_dependencies': 'graphtoolkit.nodes.atomic.storage',
        'save_output': 'graphtoolkit.nodes.atomic.storage',
        'save_phase_output': 'graphtoolkit.nodes.atomic.storage',
        'load_storage': 'graphtoolkit.nodes.atomic.storage',
        'save_storage': 'graphtoolkit.nodes.atomic.storage',
        'batch_load': 'graphtoolkit.nodes.atomic.storage',
        'batch_save': 'graphtoolkit.nodes.atomic.storage',
        
        # Template nodes
        'template_render': 'graphtoolkit.nodes.atomic.templates',
        'template_validate': 'graphtoolkit.nodes.atomic.templates',
        'template_save': 'graphtoolkit.nodes.atomic.templates',
        'template_exec': 'graphtoolkit.nodes.atomic.templates',
        
        # LLM nodes
        'llm_call': 'graphtoolkit.nodes.atomic.llm',
        'prompt_builder': 'graphtoolkit.nodes.atomic.llm',
        'response_parser': 'graphtoolkit.nodes.atomic.llm',
        'batch_llm': 'graphtoolkit.nodes.atomic.llm',
        
        # Validation nodes
        'schema_validation': 'graphtoolkit.nodes.atomic.validation',
        'quality_gate': 'graphtoolkit.nodes.atomic.validation',
        'dependency_validation': 'graphtoolkit.nodes.atomic.validation',
        'data_validation': 'graphtoolkit.nodes.atomic.validation',
        'syntax_validation': 'graphtoolkit.nodes.atomic.validation',
        'import_validation': 'graphtoolkit.nodes.atomic.validation',
        
        # Control nodes
        'state_update': 'graphtoolkit.nodes.atomic.control',
        'next_phase': 'graphtoolkit.nodes.atomic.control',
        'refinement': 'graphtoolkit.nodes.atomic.control',
        'conditional': 'graphtoolkit.nodes.atomic.control',
        'loop': 'graphtoolkit.nodes.atomic.control',
        'branch': 'graphtoolkit.nodes.atomic.control',
        'parallel': 'graphtoolkit.nodes.atomic.control',
        'state_based_conditional': 'graphtoolkit.nodes.atomic.control',
        'sequential_map': 'graphtoolkit.nodes.atomic.control',
        
        # Transform nodes
        'json_parse': 'graphtoolkit.nodes.atomic.transform',
        'json_serialize': 'graphtoolkit.nodes.atomic.transform',
        'code_format': 'graphtoolkit.nodes.atomic.transform',
        'data_merge': 'graphtoolkit.nodes.atomic.transform',
        'data_filter': 'graphtoolkit.nodes.atomic.transform',
        
        # Execution nodes
        'test_execution': 'graphtoolkit.nodes.atomic.execution',
        'coverage_analysis': 'graphtoolkit.nodes.atomic.execution',
        'code_execution': 'graphtoolkit.nodes.atomic.execution',
        
        # Generator nodes
        'simple_generator': 'graphtoolkit.nodes.atomic.generators',
        'advanced_generator': 'graphtoolkit.nodes.atomic.generators',
        'generator_routing': 'graphtoolkit.nodes.atomic.generators',
        
        # Approval nodes
        'approval': 'graphtoolkit.nodes.atomic.approval',
        'refinement_loop': 'graphtoolkit.nodes.atomic.approval',
        'quality_check': 'graphtoolkit.nodes.atomic.approval',
        
        # Iteration operation nodes
        'process_tools': 'graphtoolkit.nodes.atomic.iteration_ops',
        'process_endpoints': 'graphtoolkit.nodes.atomic.iteration_ops',
        'process_steps': 'graphtoolkit.nodes.atomic.iteration_ops',
        'process_contracts': 'graphtoolkit.nodes.atomic.iteration_ops',
        'batch_validate': 'graphtoolkit.nodes.atomic.iteration_ops',
        'specifier_tool_iterator': 'graphtoolkit.nodes.atomic.iteration_ops',  # V1-compatible specifier
        'prepare_specifier_iteration': 'graphtoolkit.nodes.atomic.storage',  # V1-compatible prep
        
        # Iteration nodes
        'iterate': 'graphtoolkit.nodes.iteration',
        'batch_process': 'graphtoolkit.nodes.iteration',
        'map': 'graphtoolkit.nodes.iteration',
        'filter': 'graphtoolkit.nodes.iteration',
        'aggregate': 'graphtoolkit.nodes.iteration',
        'parallel_map': 'graphtoolkit.nodes.iteration',
        
        # Generic nodes
        'generic_phase': 'graphtoolkit.nodes.generic',
        
        # Base nodes
        'error': 'graphtoolkit.nodes.base'
    }
    
    module_path = module_map.get(node_id)
    if module_path:
        try:
            import importlib
            importlib.import_module(module_path)
            logger.debug(f'Imported module for node: {node_id}')
        except ImportError as e:
            logger.error(f'Could not import module for node {node_id}: {e}')
            from ..exceptions import NodeExecutionError
            raise NodeExecutionError(f'Failed to import required module for node {node_id}: {e}') from e


def create_workflow_state(
    workflow_def: WorkflowDefinition,
    workflow_id: str,
    initial_data: Optional[Dict[str, Any]] = None
) -> WorkflowState:
    """Create initial workflow state.
    
    Args:
        workflow_def: Workflow definition
        workflow_id: Unique workflow identifier
        initial_data: Optional initial domain data
        
    Returns:
        Initial workflow state
    """
    # Get first phase
    if not workflow_def.phase_sequence:
        raise ValueError('Workflow definition has no phases')
    
    first_phase = workflow_def.phase_sequence[0]
    phase_def = workflow_def.phases.get(first_phase)
    
    if not phase_def:
        raise ValueError(f'Phase {first_phase} not found in workflow definition')
    
    # Get first node in first phase
    first_node = phase_def.atomic_nodes[0] if phase_def.atomic_nodes else ''
    
    return WorkflowState(
        workflow_id=workflow_id,
        domain=workflow_def.domain,
        workflow_def=workflow_def,
        current_phase=first_phase,
        current_node=first_node,
        domain_data=initial_data or {}
    )


def build_domain_workflow(
    domain: str,
    phases: List[str],
    enable_refinement: bool = True,
    enable_parallel: bool = False
) -> WorkflowDefinition:
    """Build a workflow definition for a domain.
    
    Args:
        domain: Domain name
        phases: List of phase names
        enable_refinement: Enable refinement loops
        enable_parallel: Enable parallel execution
        
    Returns:
        Complete workflow definition
    """
    return PHASE_REGISTRY.create_workflow_definition(
        domain=domain,
        phases=phases,
        enable_refinement=enable_refinement,
        enable_parallel=enable_parallel
    )


def create_workflow_graph(
    workflow_def: WorkflowDefinition,
    start_node_class: Optional[Type[BaseNode]] = None
) -> Graph:
    """Create a pydantic_graph Graph from workflow definition.
    
    Args:
        workflow_def: Workflow definition
        start_node_class: Optional custom start node class
        
    Returns:
        Configured graph ready for execution
    """
    if not HAS_PYDANTIC_GRAPH:
        logger.error('pydantic_graph not available - this is a required dependency')
        from ..exceptions import DependencyError
        raise DependencyError('pydantic_graph is required but not available. Install with: pip install pydantic-graph')
    
    # Import GenericPhaseNode if not provided
    if start_node_class is None:
        try:
            from ..nodes.generic import GenericPhaseNode
            start_node_class = GenericPhaseNode
        except ImportError:
            raise ImportError('GenericPhaseNode not available, provide start_node_class')
    
    # Collect all node classes used in workflow
    node_classes = set()
    node_classes.add(start_node_class)
    
    for phase_def in workflow_def.phases.values():
        for node_id in phase_def.atomic_nodes:
            try:
                # This will trigger import and registration
                create_node_instance(node_id)
                if node_id in NODE_CLASSES:
                    node_classes.add(NODE_CLASSES[node_id])
            except ValueError as e:
                logger.error(f'Could not load node class for: {node_id}')
                from ..exceptions import NodeExecutionError
                raise NodeExecutionError(f'Failed to load required node class for: {node_id}') from e
    
    # Create graph with all node classes
    return Graph(nodes=list(node_classes))


def create_domain_workflow(
    domain: str,
    phases: List[str],
    workflow_id: str,
    initial_data: Optional[Dict[str, Any]] = None
) -> tuple[WorkflowDefinition, WorkflowState, Graph]:
    """High-level function to create a complete workflow.
    
    Args:
        domain: Domain name
        phases: List of phase names
        workflow_id: Unique workflow identifier
        initial_data: Optional initial domain data
        
    Returns:
        Tuple of (workflow_def, initial_state, graph)
    """
    # Build workflow definition
    workflow_def = build_domain_workflow(domain, phases)
    
    # Create initial state
    initial_state = create_workflow_state(workflow_def, workflow_id, initial_data)
    
    # Create graph
    graph = create_workflow_graph(workflow_def)
    
    return workflow_def, initial_state, graph


def update_state_with_result(
    state: WorkflowState,
    phase_name: str,
    result: Any,
    storage_ref: Optional[StorageRef] = None
) -> WorkflowState:
    """Helper to update state after phase completion.
    
    Args:
        state: Current state
        phase_name: Completed phase name
        result: Phase result
        storage_ref: Optional storage reference
        
    Returns:
        Updated state
    """
    new_domain_data = {
        **state.domain_data,
        f'{phase_name}_output': result
    }
    
    new_state = replace(
        state,
        completed_phases=state.completed_phases | {phase_name},
        domain_data=new_domain_data
    )
    
    if storage_ref:
        new_state = new_state.with_storage_ref(phase_name, storage_ref)
    
    # Move to next phase
    next_phase = state.workflow_def.get_next_phase(phase_name)
    if next_phase:
        phase_def = state.workflow_def.phases.get(next_phase)
        if phase_def:
            new_state = replace(
                new_state,
                current_phase=next_phase,
                current_node=phase_def.atomic_nodes[0] if phase_def.atomic_nodes else ''
            )
    
    return new_state


def validate_workflow_definition(workflow_def: WorkflowDefinition) -> List[str]:
    """Validate a workflow definition for issues.
    
    Args:
        workflow_def: Workflow definition to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check phases exist
    if not workflow_def.phases:
        errors.append('Workflow has no phases defined')
    
    if not workflow_def.phase_sequence:
        errors.append('Workflow has no phase sequence defined')
    
    # Check phase sequence matches phases
    for phase_name in workflow_def.phase_sequence:
        if phase_name not in workflow_def.phases:
            errors.append(f'Phase {phase_name} in sequence but not defined')
    
    # Check each phase
    for phase_name, phase_def in workflow_def.phases.items():
        if not phase_def.atomic_nodes:
            errors.append(f'Phase {phase_name} has no atomic nodes')
        
        # Check dependencies are in workflow
        for dep in phase_def.dependencies:
            if dep not in workflow_def.phases:
                errors.append(f'Phase {phase_name} depends on undefined phase {dep}')
            
            # Check dependency comes before in sequence
            if dep in workflow_def.phase_sequence and phase_name in workflow_def.phase_sequence:
                dep_idx = workflow_def.phase_sequence.index(dep)
                phase_idx = workflow_def.phase_sequence.index(phase_name)
                if dep_idx >= phase_idx:
                    errors.append(f'Phase {phase_name} depends on {dep} which comes after it')
    
    # Check node configs
    for phase_def in workflow_def.phases.values():
        for node_id in phase_def.atomic_nodes:
            if node_id not in workflow_def.node_configs:
                errors.append(f'No configuration for node {node_id}')
    
    return errors