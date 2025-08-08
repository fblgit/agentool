"""
GraphToolkit Control Flow Atomic Nodes.

Control flow nodes for state updates, phase transitions, and conditional logic.
"""

from typing import Any, Optional, Union
from dataclasses import dataclass, replace
from datetime import datetime
import logging

from ..base import (
    BaseNode,
    AtomicNode,
    NonRetryableError,
    GraphRunContext,
    End
)
from ...core.types import (
    WorkflowState,
    RefinementRecord
)
from ...core.factory import register_node_class, create_node_instance


logger = logging.getLogger(__name__)


@dataclass
class StateUpdateNode(AtomicNode[WorkflowState, Any, WorkflowState]):
    """
    Update workflow state after phase operations.
    Marks phase complete and updates metadata.
    """
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> WorkflowState:
        """Update state to mark phase complete."""
        phase_name = ctx.state.current_phase
        
        # Mark phase as complete
        new_state = replace(
            ctx.state,
            completed_phases=ctx.state.completed_phases | {phase_name},
            updated_at=datetime.now()
        )
        
        logger.info(f"Phase {phase_name} marked as complete")
        return new_state
    
    async def update_state(self, state: WorkflowState, result: WorkflowState) -> WorkflowState:
        """State is already updated in perform_operation."""
        return result


@dataclass
class NextPhaseNode(BaseNode[WorkflowState, Any, WorkflowState]):
    """
    Determine and transition to the next phase.
    """
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> Union[BaseNode, End[WorkflowState]]:
        """Find next phase and transition to it."""
        # Get next phase from workflow definition
        next_phase = ctx.state.workflow_def.get_next_phase(ctx.state.current_phase)
        
        if next_phase:
            logger.info(f"Transitioning from {ctx.state.current_phase} to {next_phase}")
            
            # Update state to new phase
            phase_def = ctx.state.workflow_def.phases.get(next_phase)
            if not phase_def:
                raise NonRetryableError(f"Phase {next_phase} not found in workflow")
            
            # Set current node to first atomic node of new phase
            first_node = phase_def.atomic_nodes[0] if phase_def.atomic_nodes else None
            
            new_state = replace(
                ctx.state,
                current_phase=next_phase,
                current_node=first_node
            )
            
            # Return GenericPhaseNode to start the new phase
            from ..generic import GenericPhaseNode
            return GenericPhaseNode()
        
        # No more phases - workflow complete
        logger.info(f"Workflow complete for {ctx.state.workflow_id}")
        return End(ctx.state)


@dataclass
class RefinementNode(BaseNode[WorkflowState, Any, WorkflowState]):
    """
    Trigger refinement of the current phase.
    """
    feedback: str
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Set up refinement and restart phase."""
        phase_name = ctx.state.current_phase
        
        # Increment refinement count
        refinement_count = ctx.state.refinement_count.get(phase_name, 0) + 1
        
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
        
        logger.info(f"Triggering refinement {refinement_count} for {phase_name}")
        
        # Restart phase from template rendering
        # (dependencies are already loaded)
        new_state = replace(new_state, current_node='template_render')
        return create_node_instance('template_render')


@dataclass
class ConditionalNode(BaseNode[WorkflowState, Any, WorkflowState]):
    """
    Conditional branching based on state.
    """
    condition_name: str
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Evaluate condition and branch."""
        # Get condition from workflow definition
        condition = ctx.state.workflow_def.conditions.get(self.condition_name)
        
        if not condition:
            raise NonRetryableError(f"Condition {self.condition_name} not found")
        
        # Evaluate condition
        result = condition.evaluate(ctx.state)
        
        logger.info(f"Condition {self.condition_name} evaluated to {result}")
        
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
            next_node_id = self.get_next_node(new_state)
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
    """
    Loop control for iterative operations.
    """
    max_iterations: int = 10
    loop_condition: Optional[str] = None
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Check loop condition and continue or exit."""
        # Get current iteration count
        loop_key = f"{ctx.state.current_phase}_loop_count"
        loop_count = ctx.state.domain_data.get(loop_key, 0)
        
        # Check max iterations
        if loop_count >= self.max_iterations:
            logger.info(f"Max iterations {self.max_iterations} reached")
            # Exit loop, continue to next node
            next_node_id = self.get_next_node(ctx.state)
            if next_node_id:
                new_state = replace(ctx.state, current_node=next_node_id)
                return create_node_instance(next_node_id)
            return End(ctx.state)
        
        # Check loop condition if specified
        if self.loop_condition:
            condition = ctx.state.workflow_def.conditions.get(self.loop_condition)
            if condition and not condition.evaluate(ctx.state):
                logger.info(f"Loop condition {self.loop_condition} not met, exiting loop")
                # Exit loop
                next_node_id = self.get_next_node(ctx.state)
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
        
        logger.info(f"Continuing loop iteration {loop_count + 1}")
        
        # Return to start of loop (would be configured)
        # For now, return to template render
        new_state = replace(new_state, current_node='template_render')
        return create_node_instance('template_render')


@dataclass
class BranchNode(BaseNode[WorkflowState, Any, WorkflowState]):
    """
    Branch to different nodes based on state.
    """
    branches: dict[str, str]  # condition_name -> node_id mapping
    default_node: Optional[str] = None
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Evaluate branches and route to appropriate node."""
        for condition_name, node_id in self.branches.items():
            condition = ctx.state.workflow_def.conditions.get(condition_name)
            if condition and condition.evaluate(ctx.state):
                logger.info(f"Branch condition {condition_name} met, routing to {node_id}")
                new_state = replace(ctx.state, current_node=node_id)
                return create_node_instance(node_id)
        
        # No conditions met, use default
        if self.default_node:
            logger.info(f"No branch conditions met, using default node {self.default_node}")
            new_state = replace(ctx.state, current_node=self.default_node)
            return create_node_instance(self.default_node)
        
        # No default, continue to next node
        next_node_id = self.get_next_node(ctx.state)
        if next_node_id:
            new_state = replace(ctx.state, current_node=next_node_id)
            return create_node_instance(next_node_id)
        
        return End(ctx.state)


@dataclass
class ParallelNode(BaseNode[WorkflowState, Any, WorkflowState]):
    """
    Execute multiple nodes in parallel.
    Note: In pydantic_graph, this would use Graph.iter() for control.
    """
    parallel_nodes: list[str]
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """
        Set up parallel execution.
        In practice, this would use Graph.iter() for parallel control.
        """
        # For now, we'll execute sequentially
        # Real implementation would use Graph.iter() pattern
        
        logger.info(f"Executing nodes in parallel: {self.parallel_nodes}")
        
        # Store parallel nodes in state for processing
        new_state = replace(
            ctx.state,
            domain_data={
                **ctx.state.domain_data,
                'parallel_nodes': self.parallel_nodes,
                'parallel_index': 0
            }
        )
        
        # Start with first parallel node
        if self.parallel_nodes:
            first_node = self.parallel_nodes[0]
            new_state = replace(new_state, current_node=first_node)
            return create_node_instance(first_node)
        
        return End(new_state)


# Register control nodes
register_node_class('state_update', StateUpdateNode)
register_node_class('next_phase', NextPhaseNode)
register_node_class('refinement', RefinementNode)
register_node_class('conditional', ConditionalNode)
register_node_class('loop', LoopNode)
register_node_class('branch', BranchNode)
register_node_class('parallel', ParallelNode)