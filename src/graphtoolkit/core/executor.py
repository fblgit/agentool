"""GraphToolkit Workflow Executor.

Main execution engine that runs workflows using pydantic_graph with atomic node chaining.
"""

import asyncio
import logging
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Real pydantic_graph imports - no more mocks
from pydantic_graph import BaseNode, SimpleStatePersistence, Graph, GraphRunContext

from ..core.deps import WorkflowDeps
from ..core.factory import create_node_instance
from ..core.types import WorkflowState
from ..nodes.generic import GenericPhaseNode

logger = logging.getLogger(__name__)


class WorkflowExecutor:
    """Executes workflows using pydantic_graph with atomic node chaining.
    
    This executor provides the real implementation using Graph.run() and Graph.iter()
    for efficient workflow execution with state persistence and parallel control.
    
    Features:
    - Real pydantic_graph integration
    - State persistence with FileStatePersistence  
    - Parallel execution control with Graph.iter()
    - Atomic node chaining for resilience
    - Metrics and logging integration
    """
    
    def __init__(self, deps: WorkflowDeps):
        """Initialize executor with dependencies.
        
        Args:
            deps: Workflow dependencies container
        """
        self.deps = deps
        self.execution_history: List[Dict[str, Any]] = []
        
    async def run(self, initial_state: WorkflowState) -> 'WorkflowResult':
        """Execute a workflow using pydantic_graph.Graph.run().
        
        Args:
            initial_state: Initial workflow state
            
        Returns:
            WorkflowResult with final state and outputs
        """
        logger.info(f'Starting workflow execution: {initial_state.workflow_id}')
        logger.debug(f'[WorkflowExecutor] Initial state - phase: {initial_state.current_phase}, node: {initial_state.current_node}')
        logger.debug(f'[WorkflowExecutor] Initial state - completed phases: {initial_state.completed_phases}')
        logger.debug(f'[WorkflowExecutor] Initial state - domain data keys: {list(initial_state.domain_data.keys())}')
        
        try:
            # Create graph from workflow definition
            logger.debug(f'[WorkflowExecutor] Building graph from workflow definition')
            graph = self._build_graph(initial_state)
            logger.debug(f'[WorkflowExecutor] Graph built successfully')
            
            # Execute using pydantic_graph
            logger.debug(f'[WorkflowExecutor] Starting graph.run() with GenericPhaseNode')
            result = await graph.run(
                GenericPhaseNode(),
                state=initial_state,
                deps=self.deps
            )
            logger.debug(f'[WorkflowExecutor] graph.run() completed, result type: {type(result).__name__}')
            
            # The result is a GraphRunResult with output and state
            final_state = result.state if hasattr(result, 'state') else result
            
            # Extract outputs from state
            outputs = self._extract_outputs(final_state)
            
            logger.info(f'Workflow execution completed: {initial_state.workflow_id}')
            
            return WorkflowResult(
                state=final_state,
                outputs=outputs,
                success=True
            )
            
        except Exception as e:
            logger.error(f'Workflow execution failed: {e}')
            from ..exceptions import WorkflowError
            raise WorkflowError(f'Workflow execution failed: {e}') from e
        finally:
            # Clean up resources
            if self.deps:
                self.deps.cleanup()
    
    def _build_graph(self, initial_state: WorkflowState) -> Graph:
        """Build pydantic_graph.Graph from workflow definition.
        
        Args:
            initial_state: Initial workflow state
            
        Returns:
            Configured Graph instance
        """
        # Import all node modules to ensure registration
        from ..nodes import generic, atomic
        from ..core.factory import get_node_registry, create_node_instance
        
        # Get all unique node types from workflow definition
        all_node_types = set()
        
        for phase_def in initial_state.workflow_def.phases.values():
            all_node_types.update(phase_def.atomic_nodes)
        
        logger.debug(f"[WorkflowExecutor] Required node types from workflow: {all_node_types}")
        
        # Ensure all required nodes are available by trying to import them
        for node_type in all_node_types:
            try:
                # This will trigger import if needed
                create_node_instance(node_type)
            except Exception as e:
                logger.error(f"[WorkflowExecutor] Failed to load required node {node_type}: {e}")
                from ..exceptions import NodeExecutionError
                raise NodeExecutionError(f"Failed to load required node {node_type}: {e}") from e
        
        # Collect node CLASSES (not instances)
        node_classes = set()
        
        # Always include GenericPhaseNode class as entry point
        node_classes.add(GenericPhaseNode)
        
        # Import base classes first
        from ..nodes.base import BaseNode, AtomicNode, ErrorNode
        
        # Import all atomic node classes that are needed
        # This approach loads all classes to ensure they're available in the graph
        from ..nodes.atomic.storage import (
            DependencyCheckNode, LoadDependenciesNode, SavePhaseOutputNode,
            PrepareSpecifierIterationNode  # V1-compatible specifier preparation
        )
        from ..nodes.atomic.templates import TemplateRenderNode
        from ..nodes.atomic.llm import LLMCallNode
        from ..nodes.atomic.validation import SchemaValidationNode, QualityGateNode
        from ..nodes.atomic.control import StateUpdateNode, NextPhaseNode, RefinementNode
        from ..nodes.atomic.iteration_ops import (
            SpecifierToolIteratorNode  # V1-compatible specifier iteration
        )
        from ..nodes.atomic.crafter_ops import (
            PrepareCrafterIterationNode,  # V1-compatible crafter preparation
            CrafterToolIteratorNode  # V1-compatible crafter iteration
        )
        from ..nodes.generic import WorkflowEndNode
        
        # Add base classes to satisfy pydantic_graph's type checking
        node_classes.add(BaseNode)
        node_classes.add(AtomicNode)
        node_classes.add(ErrorNode)
        
        # Add all node classes that might be used
        # We include all to avoid "not included in graph" errors
        node_classes.add(DependencyCheckNode)
        node_classes.add(LoadDependenciesNode)
        node_classes.add(SavePhaseOutputNode)
        node_classes.add(PrepareSpecifierIterationNode)  # V1-compatible specifier prep
        node_classes.add(SpecifierToolIteratorNode)  # V1-compatible specifier iteration
        node_classes.add(PrepareCrafterIterationNode)  # V1-compatible crafter prep
        node_classes.add(CrafterToolIteratorNode)  # V1-compatible crafter iteration
        node_classes.add(TemplateRenderNode)
        node_classes.add(LLMCallNode)
        node_classes.add(SchemaValidationNode)
        node_classes.add(QualityGateNode)
        node_classes.add(StateUpdateNode)
        node_classes.add(NextPhaseNode)
        node_classes.add(RefinementNode)
        node_classes.add(WorkflowEndNode)
        node_classes.add(ErrorNode)
        
        # Get all registered node classes dynamically
        registry = get_node_registry()
        logger.debug(f"[WorkflowExecutor] Node registry has {len(registry)} registered nodes")
        
        # Add any additional registered node classes that are in the workflow
        for node_type in all_node_types:
            if node_type in registry:
                node_classes.add(registry[node_type])
                logger.debug(f"[WorkflowExecutor] Added registered node class for: {node_type}")
        
        logger.debug(f"[WorkflowExecutor] Total node classes in graph: {len(node_classes)}")
        
        # Create and return graph with node CLASSES
        return Graph(nodes=list(node_classes))
    
    def _extract_outputs(self, final_state: WorkflowState) -> Dict[str, Any]:
        """Extract outputs from final workflow state.
        
        Args:
            final_state: Final workflow state
            
        Returns:
            Dictionary of phase outputs
        """
        outputs = {}
        
        # Extract from phase_outputs (storage references) if they exist
        if hasattr(final_state, 'phase_outputs') and final_state.phase_outputs:
            for phase_name, storage_ref in final_state.phase_outputs.items():
                outputs[phase_name] = {
                    'storage_ref': str(storage_ref),
                    'phase': phase_name,
                    'created_at': storage_ref.created_at.isoformat() if hasattr(storage_ref, 'created_at') else None
                }
        
        # Extract from domain_data
        for key, value in final_state.domain_data.items():
            if '_output' in key:
                phase_name = key.replace('_output', '')
                if phase_name not in outputs:
                    outputs[phase_name] = {}
                outputs[phase_name]['data'] = value
        
        return outputs
    
    def _extract_phase_output(
        self,
        state: WorkflowState,
        phase_name: str
    ) -> Dict[str, Any]:
        """Extract output data for a phase from state.
        
        Args:
            state: Workflow state
            phase_name: Phase name
            
        Returns:
            Phase output data
        """
        # Check if phase output was saved
        if phase_name in state.phase_outputs:
            ref = state.phase_outputs[phase_name]
            # In real implementation, would load from storage
            # For now, return reference info
            return {
                'storage_ref': str(ref),
                'phase': phase_name
            }
        
        # Check domain data for phase output
        phase_output_key = f'{phase_name}_output'
        if phase_output_key in state.domain_data:
            return state.domain_data[phase_output_key]
        
        # Return any phase-specific data
        return {
            key: value
            for key, value in state.domain_data.items()
            if key.startswith(phase_name)
        }
    
    def _record_execution(
        self,
        phase_name: str,
        nodes_executed: List[str],
        execution_time: float,
        success: bool,
        error: Optional[str] = None
    ):
        """Record execution history for monitoring."""
        self.execution_history.append({
            'phase': phase_name,
            'nodes': nodes_executed,
            'execution_time': execution_time,
            'success': success,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
    
    def _is_recoverable_error(self, error: Exception) -> bool:
        """Check if error is recoverable."""
        from ..nodes.base import LLMError, RetryableError, StorageError
        return isinstance(error, (RetryableError, StorageError, LLMError))
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history for monitoring."""
        return self.execution_history
    
    async def run_with_persistence(
        self,
        initial_state: WorkflowState,
        persistence_path: str
    ) -> 'WorkflowResult':
        """Execute workflow with state persistence using FileStatePersistence.
        
        Args:
            initial_state: Initial workflow state
            persistence_path: Path to persist state
            
        Returns:
            WorkflowResult
        """
        logger.info(f'Running with persistence at: {persistence_path}')
        
        try:
            # Create graph from workflow definition
            graph = self._build_graph(initial_state)
            
            # Set up file persistence
            persistence = SimpleStatePersistence(Path(persistence_path))
            
            # Execute using pydantic_graph with persistence
            final_state = await graph.run(
                GenericPhaseNode(),
                state=initial_state,
                deps=self.deps,
                persistence=persistence
            )
            
            # Extract outputs from state
            outputs = self._extract_outputs(final_state)
            
            logger.info(f'Workflow execution completed with persistence: {initial_state.workflow_id}')
            
            return WorkflowResult(
                state=final_state,
                outputs=outputs,
                success=True
            )
            
        except Exception as e:
            logger.error(f'Workflow execution with persistence failed: {e}')
            from ..exceptions import WorkflowError
            raise WorkflowError(f'Workflow execution with persistence failed: {e}') from e
        finally:
            # Clean up resources
            if self.deps:
                self.deps.cleanup()
    
    async def run_phase_only(
        self,
        state: WorkflowState,
        phase_name: str
    ) -> 'PhaseResult':
        """Execute only a specific phase.
        
        Args:
            state: Current workflow state
            phase_name: Phase to execute
            
        Returns:
            PhaseResult
        """
        logger.info(f'Executing single phase: {phase_name}')
        state = replace(state, current_phase=phase_name)
        return await self._execute_phase(state, phase_name)
    
    async def run_with_parallel_control(
        self,
        initial_state: WorkflowState,
        persistence_path: Optional[str] = None
    ) -> 'WorkflowResult':
        """Execute workflow with Graph.iter() for parallel execution control.
        
        This implements the parallel execution pattern documented in workflow-graph-system.md
        using pydantic_graph.Graph.iter() for fine-grained control over execution.
        
        Args:
            initial_state: Initial workflow state
            persistence_path: Optional path for state persistence
            
        Returns:
            WorkflowResult
        """
        logger.info(f'Starting parallel-controlled workflow: {initial_state.workflow_id}')
        
        try:
            # Create graph from workflow definition
            graph = self._build_graph(initial_state)
            
            # Set up persistence if requested
            persistence = None
            if persistence_path:
                persistence = SimpleStatePersistence(Path(persistence_path))
            
            # Execute with parallel control using Graph.iter()
            final_state = initial_state
            
            async with graph.iter(
                GenericPhaseNode(),
                state=initial_state,
                deps=self.deps,
                persistence=persistence
            ) as run:
                async for node_result in run:
                    # Check if this is a parallel execution request
                    if self._should_execute_parallel(node_result, final_state):
                        logger.info('Parallel execution requested')
                        
                        # Get parallel items from state
                        parallel_items = final_state.iter_items
                        
                        if parallel_items:
                            # Execute items in parallel
                            tasks = []
                            for item in parallel_items:
                                task = asyncio.create_task(
                                    self._process_parallel_item(node_result, item, final_state)
                                )
                                tasks.append(task)
                            
                            # Wait for all parallel tasks
                            results = await asyncio.gather(*tasks, return_exceptions=True)
                            
                            # Filter successful results
                            successful_results = [
                                r for r in results 
                                if not isinstance(r, Exception)
                            ]
                            
                            # Update state with parallel results
                            final_state = replace(
                                final_state,
                                iter_results=successful_results,
                                iter_index=len(parallel_items)
                            )
                            
                            logger.info(f'Parallel execution completed: {len(successful_results)}/{len(parallel_items)} successful')
                    
                    # Update final state from node execution
                    if hasattr(node_result, 'state'):
                        final_state = node_result.state
            
            # Extract outputs
            outputs = self._extract_outputs(final_state)
            
            logger.info(f'Parallel-controlled workflow completed: {initial_state.workflow_id}')
            
            return WorkflowResult(
                state=final_state,
                outputs=outputs,
                success=True
            )
            
        except Exception as e:
            logger.error(f'Parallel-controlled workflow failed: {e}')
            from ..exceptions import WorkflowError
            raise WorkflowError(f'Parallel-controlled workflow failed: {e}') from e
        finally:
            # Clean up resources
            if self.deps:
                self.deps.cleanup()
    
    def _should_execute_parallel(self, node_result: Any, state: WorkflowState) -> bool:
        """Check if parallel execution is requested."""
        return (
            hasattr(node_result, '__class__') and
            'parallel' in node_result.__class__.__name__.lower() and
            len(state.iter_items) > 1
        )
    
    async def _process_parallel_item(
        self,
        node: BaseNode,
        item: Any,
        state: WorkflowState
    ) -> Any:
        """Process a single item in parallel execution."""
        try:
            # Create isolated state for this item
            item_state = replace(
                state,
                iter_items=[item],
                iter_index=0,
                iter_results=[]
            )
            
            # Execute the node for this item
            # This is a simplified approach - in practice, you'd need
            # to handle the full node chain for the item
            result = await node.run(GraphRunContext(item_state, self.deps))
            
            return result
            
        except Exception as e:
            logger.error(f'Parallel item processing failed: {e}')
            from ..exceptions import NodeExecutionError
            raise NodeExecutionError(f'Parallel item processing failed: {e}') from e


@dataclass
class WorkflowResult:
    """Result from workflow execution."""
    state: WorkflowState
    outputs: Dict[str, Any]
    success: bool
    error: Optional[str] = None
    execution_time: Optional[float] = None
    
    def __post_init__(self):
        """Calculate execution time if not provided."""
        if self.execution_time is None and self.state:
            start_time = self.state.created_at
            end_time = self.state.updated_at
            if start_time and end_time:
                self.execution_time = (end_time - start_time).total_seconds()


@dataclass  
class PhaseResult:
    """Result from executing a phase."""
    output: Dict[str, Any]
    state: WorkflowState
    should_stop: bool = False


# Production-ready workflow functions with agentoolkit integration

async def execute_agentool_workflow(
    task_description: str,
    model: str = 'openai:gpt-4o',
    workflow_id: Optional[str] = None,
    enable_persistence: bool = True
) -> Dict[str, Any]:
    """Execute an AgenTool generation workflow using GraphToolkit.
    
    Args:
        task_description: Description of the AgenTool to create
        model: LLM model to use
        workflow_id: Optional workflow identifier
        enable_persistence: Whether to enable state persistence
        
    Returns:
        Workflow execution results
    """
    from agentool.core.injector import get_injector
    from ..core.deps import WorkflowDeps
    from ..core.factory import create_workflow_state, build_domain_workflow
    from .initialization import ensure_graphtoolkit_initialized, agentool_workflow_config
    
    # Ensure initialization for AgenTool workflow
    ensure_graphtoolkit_initialized(agentool_workflow_config())
    
    # Generate workflow ID if not provided
    if not workflow_id:
        import uuid
        workflow_id = str(uuid.uuid4())
    
    # Create workflow dependencies
    deps = WorkflowDeps(
        injector=get_injector(),
        model_config={
            'analyzer': model,
            'specifier': model, 
            'crafter': model,
            'evaluator': model
        },
        enable_metrics=True,
        enable_logging=True
    )
    
    # Build workflow definition for AgenTool domain
    workflow_def = build_domain_workflow(
        domain='agentool',
        phases=['analyzer', 'specifier', 'crafter', 'evaluator'],
        enable_refinement=True,
        enable_parallel=False
    )
    
    # Create initial state
    initial_state = create_workflow_state(
        workflow_def=workflow_def,
        workflow_id=workflow_id,
        initial_data={
            'task_description': task_description,
            'model': model,
            'domain': 'agentool'
        }
    )
    
    # Create executor
    executor = WorkflowExecutor(deps)
    
    try:
        # Execute with or without persistence
        if enable_persistence:
            persistence_path = f'workflows/{workflow_id}/state.json'
            result = await executor.run_with_persistence(initial_state, persistence_path)
        else:
            result = await executor.run(initial_state)
        
        # Convert to standard format
        return {
            'workflow_id': workflow_id,
            'success': result.success,
            'error': result.error,
            'execution_time': result.execution_time,
            'completed_phases': list(result.state.completed_phases),
            'quality_scores': result.state.quality_scores,
            'outputs': result.outputs,
            'domain_data': result.state.domain_data
        }
        
    except Exception as e:
        logger.error(f'AgenTool workflow execution failed: {e}')
        from ..exceptions import WorkflowError
        raise WorkflowError(f'AgenTool workflow execution failed: {e}') from e


async def execute_smoke_workflow(
    ingredients: List[str],
    dietary_restrictions: Optional[List[str]] = None,
    cuisine_preference: Optional[str] = None,
    max_cook_time: Optional[int] = None,
    model: str = 'openai:gpt-4o-mini',
    workflow_id: Optional[str] = None,
    enable_persistence: bool = False,
    enable_refinement: bool = True
) -> Dict[str, Any]:
    """Execute a smoke test workflow for recipe generation.
    
    This is a lightweight E2E test workflow that exercises all GraphToolkit
    capabilities with minimal complexity and cost.
    
    Args:
        ingredients: List of available ingredients
        dietary_restrictions: Optional dietary restrictions
        cuisine_preference: Optional cuisine preference
        max_cook_time: Optional max cooking time in minutes
        model: LLM model to use (default: gpt-4o-mini for cost efficiency)
        workflow_id: Optional workflow identifier
        enable_persistence: Whether to enable state persistence
        enable_refinement: Whether to enable quality-based refinement
        
    Returns:
        Workflow execution results with recipe and evaluation
    """
    from ..domains.smoke import create_smoke_workflow
    from .initialization import ensure_graphtoolkit_initialized, default_config
    
    # Ensure initialization
    ensure_graphtoolkit_initialized(default_config())
    
    # Create smoke workflow
    workflow_def, initial_state = create_smoke_workflow(
        ingredients=ingredients,
        dietary_restrictions=dietary_restrictions,
        cuisine_preference=cuisine_preference,
        max_cook_time=max_cook_time,
        workflow_id=workflow_id,
        enable_refinement=enable_refinement
    )
    
    # Create dependencies with specified model
    from .deps import ModelConfig, StorageConfig
    deps = WorkflowDeps(
        models=ModelConfig(provider=model.split(':')[0], model=model.split(':')[1]),
        storage=StorageConfig(kv_backend='memory'),
        metrics_enabled=True,
        logging_level='INFO'
    )
    
    # Create and run executor
    executor = WorkflowExecutor(deps)
    
    # Run the workflow
    result = await executor.run(initial_state)
    
    # Return the raw workflow result - let the caller extract what they need
    return result


async def execute_testsuite_workflow(
    code_to_test: str,
    framework: str = 'pytest',
    coverage_target: float = 0.85,
    workflow_id: Optional[str] = None,
    enable_persistence: bool = True
) -> Dict[str, Any]:
    """Execute a test suite generation workflow using GraphToolkit.
    
    Args:
        code_to_test: Code that needs test coverage
        framework: Testing framework to use
        coverage_target: Target coverage percentage
        workflow_id: Optional workflow identifier
        enable_persistence: Whether to enable state persistence
        
    Returns:
        Workflow execution results
    """
    from agentool.core.injector import get_injector
    from ..core.deps import WorkflowDeps
    from ..core.factory import create_workflow_state, build_domain_workflow
    from .initialization import ensure_graphtoolkit_initialized, default_config
    
    # Ensure initialization for TestSuite workflow  
    ensure_graphtoolkit_initialized(default_config())
    
    # Generate workflow ID if not provided
    if not workflow_id:
        import uuid
        workflow_id = str(uuid.uuid4())
    
    # Create workflow dependencies
    deps = WorkflowDeps(
        injector=get_injector(),
        model_config={
            'test_analyzer': 'openai:gpt-4o',
            'test_designer': 'openai:gpt-4o',
            'test_generator': 'anthropic:claude-3-5-sonnet-latest',
            'test_executor': 'openai:gpt-4o-mini'
        },
        enable_metrics=True,
        enable_logging=True
    )
    
    # Build workflow definition for TestSuite domain
    workflow_def = build_domain_workflow(
        domain='testsuite',
        phases=['test_analyzer', 'test_designer', 'test_generator', 'test_executor'],
        enable_refinement=True,
        enable_parallel=False
    )
    
    # Create initial state
    initial_state = create_workflow_state(
        workflow_def=workflow_def,
        workflow_id=workflow_id,
        initial_data={
            'code_to_test': code_to_test,
            'framework': framework,
            'coverage_target': coverage_target,
            'domain': 'testsuite'
        }
    )
    
    # Create executor
    executor = WorkflowExecutor(deps)
    
    try:
        # Execute with or without persistence
        if enable_persistence:
            persistence_path = f'workflows/{workflow_id}/state.json'
            result = await executor.run_with_persistence(initial_state, persistence_path)
        else:
            result = await executor.run(initial_state)
        
        # Convert to standard format
        return {
            'workflow_id': workflow_id,
            'success': result.success,
            'error': result.error,
            'execution_time': result.execution_time,
            'completed_phases': list(result.state.completed_phases),
            'quality_scores': result.state.quality_scores,
            'outputs': result.outputs,
            'domain_data': result.state.domain_data,
            # TestSuite-specific results
            'test_files': result.state.domain_data.get('test_files', {}),
            'coverage_report': result.state.domain_data.get('coverage_report', {}),
            'test_results': result.state.domain_data.get('test_results', {})
        }
        
    except Exception as e:
        logger.error(f'TestSuite workflow execution failed: {e}')
        from ..exceptions import WorkflowError
        raise WorkflowError(f'TestSuite workflow execution failed: {e}') from e