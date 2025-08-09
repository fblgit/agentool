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
from pydantic_graph import BaseNode, FileStatePersistence, Graph, GraphRunContext

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
        
        try:
            # Create graph from workflow definition
            graph = self._build_graph(initial_state)
            
            # Execute using pydantic_graph
            final_state = await graph.run(
                GenericPhaseNode(),
                state=initial_state,
                deps=self.deps
            )
            
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
            return WorkflowResult(
                state=initial_state,
                outputs={},
                success=False,
                error=str(e)
            )
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
        # Get all unique node types from workflow definition
        all_node_types = set()
        
        for phase_def in initial_state.workflow_def.phases.values():
            all_node_types.update(phase_def.atomic_nodes)
        
        # Create node instances
        nodes = []
        
        # Always include GenericPhaseNode as entry point
        nodes.append(GenericPhaseNode())
        
        # Add all atomic nodes that can be created
        for node_type in all_node_types:
            try:
                node = create_node_instance(node_type)
                if node:
                    nodes.append(node)
            except Exception as e:
                logger.warning(f'Could not create node {node_type}: {e}')
        
        # Create and return graph
        return Graph(nodes=nodes)
    
    def _extract_outputs(self, final_state: WorkflowState) -> Dict[str, Any]:
        """Extract outputs from final workflow state.
        
        Args:
            final_state: Final workflow state
            
        Returns:
            Dictionary of phase outputs
        """
        outputs = {}
        
        # Extract from phase_outputs (storage references)
        for phase_name, storage_ref in final_state.phase_outputs.items():
            outputs[phase_name] = {
                'storage_ref': str(storage_ref),
                'phase': phase_name,
                'created_at': storage_ref.created_at.isoformat()
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
            persistence = FileStatePersistence(Path(persistence_path))
            
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
            return WorkflowResult(
                state=initial_state,
                outputs={},
                success=False,
                error=str(e)
            )
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
                persistence = FileStatePersistence(Path(persistence_path))
            
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
            return WorkflowResult(
                state=initial_state,
                outputs={},
                success=False,
                error=str(e)
            )
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
            return e


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