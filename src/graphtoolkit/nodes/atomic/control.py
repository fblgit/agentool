"""GraphToolkit Control Flow Atomic Nodes.

Control flow nodes for state updates, phase transitions, and conditional logic.
"""

import logging
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Optional, Union

from pydantic_graph import GraphRunContext

from ...core.factory import create_node_instance, register_node_class
from ...core.types import RefinementRecord, WorkflowState
from ..base import AtomicNode, BaseNode, End, NonRetryableError

logger = logging.getLogger(__name__)


@dataclass
class StateUpdateNode(AtomicNode[WorkflowState, Any, WorkflowState]):
    """Update workflow state after phase operations.
    Marks phase complete and updates metadata.
    """
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> WorkflowState:
        """Update state to mark phase complete."""
        phase_name = ctx.state.current_phase
        
        logger.debug(f"[StateUpdateNode] Marking phase {phase_name} as complete")
        logger.debug(f"[StateUpdateNode] Current completed phases: {ctx.state.completed_phases}")
        
        # Mark phase as complete
        new_state = replace(
            ctx.state,
            completed_phases=ctx.state.completed_phases | {phase_name},
            updated_at=datetime.now()
        )
        
        logger.info(f'[StateUpdateNode] Phase {phase_name} marked as complete')
        logger.debug(f"[StateUpdateNode] New completed phases: {new_state.completed_phases}")
        return new_state
    
    async def update_state_in_place(self, state: WorkflowState, result: WorkflowState) -> None:
        """Don't store WorkflowState in domain_data to avoid recursion."""
        # We already updated the state in perform_operation
        # Don't store the entire WorkflowState in domain_data
        pass


@dataclass
class NextPhaseNode(BaseNode[WorkflowState, Any, WorkflowState]):
    """Determine and transition to the next phase.
    """
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> Union[BaseNode, End[WorkflowState]]:
        """Find next phase and transition to it."""
        logger.debug(f"[NextPhaseNode] === ENTRY === Current phase: {ctx.state.current_phase}")
        logger.debug(f"[NextPhaseNode] Current node: {ctx.state.current_node}")
        logger.debug(f"[NextPhaseNode] Completed phases: {ctx.state.completed_phases}")
        logger.debug(f"[NextPhaseNode] Phase sequence: {ctx.state.workflow_def.phase_sequence}")
        logger.debug(f"[NextPhaseNode] Domain data keys: {list(ctx.state.domain_data.keys())}")
        
        # Get next phase from workflow definition
        next_phase = ctx.state.workflow_def.get_next_phase(ctx.state.current_phase)
        logger.debug(f"[NextPhaseNode] Next phase from workflow_def: {next_phase}")
        
        if next_phase:
            logger.info(f'[NextPhaseNode] Transitioning from {ctx.state.current_phase} to {next_phase}')
            
            # Update state to new phase
            phase_def = ctx.state.workflow_def.phases.get(next_phase)
            if not phase_def:
                raise NonRetryableError(f'Phase {next_phase} not found in workflow')
            
            # Set current node to first atomic node of new phase
            first_node = phase_def.atomic_nodes[0] if phase_def.atomic_nodes else None
            
            # Update state and directly run the first node of the new phase
            new_state = replace(
                ctx.state,
                current_phase=next_phase,
                current_node=first_node
            )
            
            # UPDATE STATE IN PLACE - pydantic_graph state is mutable during execution
            ctx.state.current_phase = next_phase
            ctx.state.current_node = first_node
            
            logger.debug(f"[NextPhaseNode] State updated in place to phase: {next_phase}")
            logger.debug(f"[NextPhaseNode] Returning GenericPhaseNode")
            
            # Return GenericPhaseNode which will now see the updated state
            from ..generic import GenericPhaseNode
            return GenericPhaseNode()
        
        # No more phases - workflow complete
        logger.info(f'[NextPhaseNode] Workflow complete for {ctx.state.workflow_id}')
        logger.debug(f"[NextPhaseNode] Final completed phases: {ctx.state.completed_phases}")
        logger.debug(f"[NextPhaseNode] === EXIT === Returning End")
        return End(ctx.state)


@dataclass
class RefinementNode(BaseNode[WorkflowState, Any, WorkflowState]):
    """Trigger refinement of the current phase.
    """
    feedback: str
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Set up refinement and restart phase."""
        phase_name = ctx.state.current_phase
        
        logger.debug(f"[RefinementNode] === ENTRY === Phase: {phase_name}, Feedback: {self.feedback[:50]}...")
        logger.debug(f"[RefinementNode] Current refinement count: {ctx.state.refinement_count}")
        
        # Increment refinement count
        refinement_count = ctx.state.refinement_count.get(phase_name, 0) + 1
        logger.info(f"[RefinementNode] Starting refinement {refinement_count} for {phase_name}")
        
        # Get current quality score
        quality_score = ctx.state.quality_scores.get(phase_name, 0.0)
        
        # Create refinement record
        record = RefinementRecord(
            iteration=refinement_count,
            timestamp=datetime.now(),
            previous_score=quality_score,
            new_score=0.0,  # Will be updated after refinement
            feedback=self.feedback,
            changes_made=[],
            code_before_ref=ctx.state.phase_outputs.get(phase_name),
            code_after_ref=None
        )
        
        # Update refinement history
        phase_history = ctx.state.refinement_history.get(phase_name, [])
        
        # Update state for refinement
        new_state = replace(
            ctx.state,
            refinement_count={
                **ctx.state.refinement_count,
                phase_name: refinement_count
            },
            refinement_history={
                **ctx.state.refinement_history,
                phase_name: phase_history + [record]
            },
            domain_data={
                **ctx.state.domain_data,
                f'{phase_name}_refinement_feedback': self.feedback,
                f'{phase_name}_previous_score': quality_score
            }
        )
        
        logger.info(f'Triggering refinement {refinement_count} for {phase_name}')
        
        # Restart phase from template rendering
        # (dependencies are already loaded)
        new_state = replace(new_state, current_node='template_render')
        return create_node_instance('template_render')


@dataclass
class ConditionalNode(BaseNode[WorkflowState, Any, WorkflowState]):
    """Conditional branching based on state.
    """
    condition_name: str
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Evaluate condition and branch."""
        # Get condition from workflow definition
        condition = ctx.state.workflow_def.conditions.get(self.condition_name)
        
        if not condition:
            raise NonRetryableError(f'Condition {self.condition_name} not found')
        
        # Evaluate condition
        result = condition.evaluate(ctx.state)
        
        logger.info(f'Condition {self.condition_name} evaluated to {result}')
        
        # Update state with condition result
        new_state = replace(
            ctx.state,
            domain_data={
                **ctx.state.domain_data,
                f'condition_{self.condition_name}': result
            }
        )
        
        # Determine next node based on result
        if result:
            # Continue to next node in sequence
            next_node_id = self.get_next_node(GraphRunContext(state=new_state, deps=ctx.deps))
            if next_node_id:
                new_state = replace(new_state, current_node=next_node_id)
                return create_node_instance(next_node_id)
        else:
            # Skip to a different node or end phase
            # This could be configured in the condition
            pass
        
        # Default: continue to next phase
        return NextPhaseNode()


@dataclass
class LoopNode(BaseNode[WorkflowState, Any, WorkflowState]):
    """Loop control for iterative operations.
    """
    max_iterations: int = 10
    loop_condition: Optional[str] = None
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Check loop condition and continue or exit."""
        # Get current iteration count
        loop_key = f'{ctx.state.current_phase}_loop_count'
        loop_count = ctx.state.domain_data.get(loop_key, 0)
        
        # Check max iterations
        if loop_count >= self.max_iterations:
            logger.info(f'Max iterations {self.max_iterations} reached')
            # Exit loop, continue to next node
            next_node_id = self.get_next_node(ctx)
            if next_node_id:
                new_state = replace(ctx.state, current_node=next_node_id)
                return create_node_instance(next_node_id)
            return End(ctx.state)
        
        # Check loop condition if specified
        if self.loop_condition:
            condition = ctx.state.workflow_def.conditions.get(self.loop_condition)
            if condition and not condition.evaluate(ctx.state):
                logger.info(f'Loop condition {self.loop_condition} not met, exiting loop')
                # Exit loop
                next_node_id = self.get_next_node(ctx)
                if next_node_id:
                    new_state = replace(ctx.state, current_node=next_node_id)
                    return create_node_instance(next_node_id)
                return End(ctx.state)
        
        # Continue loop
        new_state = replace(
            ctx.state,
            domain_data={
                **ctx.state.domain_data,
                loop_key: loop_count + 1
            }
        )
        
        logger.info(f'Continuing loop iteration {loop_count + 1}')
        
        # Return to start of loop (would be configured)
        # For now, return to template render
        new_state = replace(new_state, current_node='template_render')
        return create_node_instance('template_render')


@dataclass
class BranchNode(BaseNode[WorkflowState, Any, WorkflowState]):
    """Branch to different nodes based on state.
    """
    branches: dict[str, str]  # condition_name -> node_id mapping
    default_node: Optional[str] = None
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Evaluate branches and route to appropriate node."""
        for condition_name, node_id in self.branches.items():
            condition = ctx.state.workflow_def.conditions.get(condition_name)
            if condition and condition.evaluate(ctx.state):
                logger.info(f'Branch condition {condition_name} met, routing to {node_id}')
                new_state = replace(ctx.state, current_node=node_id)
                return create_node_instance(node_id)
        
        # No conditions met, use default
        if self.default_node:
            logger.info(f'No branch conditions met, using default node {self.default_node}')
            new_state = replace(ctx.state, current_node=self.default_node)
            return create_node_instance(self.default_node)
        
        # No default, continue to next node
        next_node_id = self.get_next_node(ctx)
        if next_node_id:
            new_state = replace(ctx.state, current_node=next_node_id)
            return create_node_instance(next_node_id)
        
        return End(ctx.state)


@dataclass
class ParallelNode(BaseNode[WorkflowState, Any, WorkflowState]):
    """Execute multiple nodes in parallel using pydantic_graph's Graph.iter().
    
    This node marks items for parallel execution. The actual parallelism
    is controlled by the Graph execution engine using Graph.iter() pattern.
    
    Per workflow-graph-system.md documentation:
    ```python
    async with graph.iter(StartNode(), state=state) as run:
        async for node in run:
            if isinstance(node, ParallelNode):
                # Graph.iter() handles parallel execution
                for item in state.iter_items:
                    task = asyncio.create_task(process_item(graph, node, item))
                results = await asyncio.gather(*tasks)
    ```
    """
    parallel_nodes: list[str]
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Mark nodes for parallel execution.
        
        In production with pydantic_graph:
        - This node signals to Graph.iter() that parallel execution should begin
        - Graph.iter() creates asyncio tasks for each parallel node
        - Results are gathered and aggregated back into state
        
        Current implementation (without pydantic_graph):
        - Store parallel nodes in state for sequential processing
        - Real Graph.iter() integration would handle actual parallelism
        """
        logger.info(f'Marking nodes for parallel execution: {self.parallel_nodes}')
        
        # Store parallel execution request in state
        # Graph.iter() would detect this and spawn parallel tasks
        new_state = replace(
            ctx.state,
            domain_data={
                **ctx.state.domain_data,
                'parallel_nodes': self.parallel_nodes,
                'parallel_execution_requested': True,
                'parallel_index': 0
            }
        )
        
        # In real Graph.iter() integration:
        # - Graph engine would detect parallel_execution_requested flag
        # - Spawn asyncio tasks for each node in parallel_nodes
        # - Use asyncio.gather() to collect results
        # - Continue with aggregation node
        
        # Current fallback: sequential execution
        if self.parallel_nodes:
            first_node = self.parallel_nodes[0]
            new_state = replace(new_state, current_node=first_node)
            logger.warning('Graph.iter() not available - falling back to sequential execution')
            return create_node_instance(first_node)
        
        return End(new_state)


@dataclass
class StateBasedConditionalNode(BaseNode[WorkflowState, Any, WorkflowState]):
    """State-based conditional branching as documented in workflow-graph-system.md.
    Evaluates conditions from WorkflowDefinition to determine next node.
    """
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Evaluate state-based conditions and branch."""
        # Get current phase definition
        phase_def = ctx.state.workflow_def.phases.get(ctx.state.current_phase)
        if not phase_def:
            raise NonRetryableError(f'Phase {ctx.state.current_phase} not found')
        
        # Check quality gate condition
        if 'quality_check' in ctx.state.workflow_def.conditions:
            condition = ctx.state.workflow_def.conditions['quality_check']
            quality_score = ctx.state.quality_scores.get(ctx.state.current_phase, 0.0)
            
            if quality_score >= getattr(condition, 'threshold', 0.8):
                logger.info(f'Quality check passed: {quality_score}')
                return NextPhaseNode()
            else:
                # Check refinement limit
                if 'refinement_limit' in ctx.state.workflow_def.conditions:
                    limit_condition = ctx.state.workflow_def.conditions['refinement_limit']
                    refinement_count = ctx.state.refinement_count.get(ctx.state.current_phase, 0)
                    
                    if refinement_count >= getattr(limit_condition, 'expected_value', 3):
                        logger.info(f'Refinement limit reached: {refinement_count}')
                        return NextPhaseNode()
                    else:
                        logger.info('Quality check failed, triggering refinement')
                        return RefinementNode(feedback='Quality score below threshold')
        
        # Check complexity routing
        if 'complexity_routing' in ctx.state.workflow_def.conditions:
            condition = ctx.state.workflow_def.conditions['complexity_routing']
            complexity = ctx.state.domain_data.get('complexity', 'normal')
            
            if complexity == 'high':
                logger.info('High complexity detected, using advanced node')
                return create_node_instance('advanced_generator')
            else:
                logger.info('Normal complexity, using simple node')
                return create_node_instance('simple_generator')
        
        # Default: continue to next node
        next_node_id = self.get_next_node(ctx)
        if next_node_id:
            new_state = replace(ctx.state, current_node=next_node_id)
            return create_node_instance(next_node_id)
        
        return End(ctx.state)


@dataclass
class SequentialMapNode(BaseNode[WorkflowState, Any, WorkflowState]):
    """Map operation over collection sequentially.
    Each item is processed one at a time through the node chain.
    """
    operation_node: str  # Node to apply to each item
    input_field: str = 'iter_items'  # Field in domain_data containing items
    output_field: str = 'iter_results'  # Field to store results
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Process items sequentially through operation node."""
        # Get items to process
        items = ctx.state.domain_data.get(self.input_field, [])
        current_index = ctx.state.iter_index
        
        if current_index >= len(items):
            # All items processed
            logger.info(f'Sequential map complete: processed {len(items)} items')
            
            # Store results and continue
            new_state = replace(
                ctx.state,
                domain_data={
                    **ctx.state.domain_data,
                    self.output_field: ctx.state.iter_results,
                    f'{self.output_field}_count': len(ctx.state.iter_results)
                },
                iter_index=0  # Reset for potential next iteration
            )
            
            # Continue to next node
            next_node_id = self.get_next_node(GraphRunContext(state=new_state, deps=ctx.deps))
            if next_node_id:
                new_state = replace(new_state, current_node=next_node_id)
                return create_node_instance(next_node_id)
            return End(new_state)
        
        # Process current item
        current_item = items[current_index]
        logger.debug(f'Processing item {current_index + 1}/{len(items)}')
        
        # Store current item in state for operation node
        new_state = replace(
            ctx.state,
            domain_data={
                **ctx.state.domain_data,
                'current_item': current_item,
                'current_item_index': current_index
            }
        )
        
        # Execute operation node for this item
        # After it completes, it will return here with incremented index
        new_state = replace(new_state, current_node=self.operation_node)
        
        # Create a wrapper that will return to this node after operation
        return SequentialMapReturnNode(
            map_node_id='sequential_map',
            operation_complete=False
        )


@dataclass  
class SequentialMapReturnNode(BaseNode[WorkflowState, Any, WorkflowState]):
    """Helper node to return to SequentialMapNode after operation.
    """
    map_node_id: str
    operation_complete: bool
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Return to map node or continue with operation."""
        if not self.operation_complete:
            # Execute the operation
            operation_node = ctx.state.domain_data.get('sequential_map_operation')
            if operation_node:
                return create_node_instance(operation_node)
            # Fallback: treat current item as result
            result = ctx.state.domain_data.get('current_item')
        else:
            # Operation complete, get result
            result = ctx.state.domain_data.get('operation_result', 
                                              ctx.state.domain_data.get('current_item'))
        
        # Add result to iter_results
        new_results = ctx.state.iter_results + [result]
        new_state = replace(
            ctx.state,
            iter_results=new_results,
            iter_index=ctx.state.iter_index + 1
        )
        
        # Return to SequentialMapNode for next item
        return create_node_instance(self.map_node_id)


# Register control nodes
register_node_class('state_update', StateUpdateNode)
register_node_class('next_phase', NextPhaseNode)
register_node_class('refinement', RefinementNode)
register_node_class('conditional', ConditionalNode)
register_node_class('loop', LoopNode)
register_node_class('branch', BranchNode)
register_node_class('parallel', ParallelNode)
register_node_class('state_based_conditional', StateBasedConditionalNode)
register_node_class('sequential_map', SequentialMapNode)
register_node_class('sequential_map_return', SequentialMapReturnNode)