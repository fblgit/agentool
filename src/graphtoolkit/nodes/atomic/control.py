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
class StateUpdateNode(AtomicNode[WorkflowState, Any, None]):
    """Update workflow state after phase operations.
    Marks phase complete and updates metadata.
    """
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> None:
        """Update state to mark phase complete."""
        phase_name = ctx.state.current_phase
        
        logger.debug(f"[StateUpdateNode] Marking phase {phase_name} as complete")
        logger.debug(f"[StateUpdateNode] Current completed phases: {ctx.state.completed_phases}")
        
        # Mark phase as complete - directly modify the state
        ctx.state.completed_phases.add(phase_name)
        ctx.state.updated_at = datetime.now()
        
        logger.info(f'[StateUpdateNode] Phase {phase_name} marked as complete')
        logger.debug(f"[StateUpdateNode] New completed phases: {ctx.state.completed_phases}")
        return None
    
    async def update_state_in_place(self, state: WorkflowState, result: Any) -> None:
        """Nothing to do - state was already updated in perform_operation."""
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


# Register control nodes
register_node_class('state_update', StateUpdateNode)
register_node_class('next_phase', NextPhaseNode)
register_node_class('refinement', RefinementNode)