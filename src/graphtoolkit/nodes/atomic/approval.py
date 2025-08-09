"""GraphToolkit Approval and Refinement Loop Nodes.

Nodes for quality approval and refinement loop management.
Per workflow-graph-system.md lines 632-634, 328-329.
"""

import logging
from dataclasses import dataclass, replace
from typing import Any, Optional

from ...core.factory import register_node_class
from ...core.types import WorkflowState
from ..base import BaseNode, GraphRunContext

logger = logging.getLogger(__name__)


@dataclass
class ApprovalNode(BaseNode[WorkflowState, Any, WorkflowState]):
    """Approval node for quality-passed phases.
    
    Per workflow-graph-system.md line 632:
    - Returned when quality check passes
    - Marks phase as approved
    - Continues to next phase
    """
    approval_message: Optional[str] = None
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Mark phase as approved and continue."""
        phase_name = ctx.state.current_phase
        
        # Log approval
        message = self.approval_message or f'Phase {phase_name} approved with quality score'
        quality_score = ctx.state.quality_scores.get(phase_name, 1.0)
        logger.info(f'{message}: {quality_score:.2f}')
        
        # Update state to mark approval
        new_state = replace(
            ctx.state,
            domain_data={
                **ctx.state.domain_data,
                f'{phase_name}_approved': True,
                f'{phase_name}_approval_message': message,
                'last_approval': phase_name
            }
        )
        
        # Continue to next phase
        from .control import NextPhaseNode
        return NextPhaseNode()


@dataclass
class RefinementLoopNode(BaseNode[WorkflowState, Any, WorkflowState]):
    """Manages refinement loops for quality improvement.
    
    Per workflow-graph-system.md lines 328-329:
    - Added when enable_refinement is true
    - Manages refinement iteration logic
    - Prevents infinite loops
    """
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Manage refinement loop logic."""
        phase_name = ctx.state.current_phase
        
        # Check if refinement is enabled
        if not ctx.state.workflow_def.enable_refinement:
            logger.info('Refinement disabled, continuing to next phase')
            from .control import NextPhaseNode
            return NextPhaseNode()
        
        # Get phase definition
        phase_def = ctx.state.workflow_def.phases.get(phase_name)
        if not phase_def or not phase_def.allow_refinement:
            logger.info(f'Refinement not allowed for phase {phase_name}')
            from .control import NextPhaseNode
            return NextPhaseNode()
        
        # Check current refinement count
        refinement_count = ctx.state.refinement_count.get(phase_name, 0)
        max_refinements = phase_def.max_refinements
        
        # Check quality score
        quality_score = ctx.state.quality_scores.get(phase_name, 0.0)
        quality_threshold = phase_def.quality_threshold
        
        # Determine if we should refine
        should_refine = (
            quality_score < quality_threshold and
            refinement_count < max_refinements
        )
        
        if should_refine:
            logger.info(
                f'Triggering refinement for {phase_name}: '
                f'score {quality_score:.2f} < threshold {quality_threshold:.2f}, '
                f'iteration {refinement_count + 1}/{max_refinements}'
            )
            
            # Generate refinement feedback
            feedback = self._generate_refinement_feedback(
                phase_name, quality_score, quality_threshold, ctx.state
            )
            
            # Trigger refinement
            from .control import RefinementNode
            return RefinementNode(feedback=feedback)
        
        # Quality met or max refinements reached
        if quality_score >= quality_threshold:
            logger.info(
                f'Quality threshold met for {phase_name}: '
                f'{quality_score:.2f} >= {quality_threshold:.2f}'
            )
            return ApprovalNode(
                approval_message=f'Phase {phase_name} approved after {refinement_count} refinements'
            )
        else:
            logger.warning(
                f'Max refinements ({max_refinements}) reached for {phase_name}, '
                f'accepting quality {quality_score:.2f}'
            )
            # Mark that we accepted below threshold
            new_state = replace(
                ctx.state,
                domain_data={
                    **ctx.state.domain_data,
                    f'{phase_name}_below_threshold': True,
                    f'{phase_name}_final_score': quality_score
                }
            )
            
            # Continue despite low quality
            from .control import NextPhaseNode
            return NextPhaseNode()
    
    def _generate_refinement_feedback(
        self,
        phase_name: str,
        current_score: float,
        threshold: float,
        state: WorkflowState
    ) -> str:
        """Generate feedback for refinement."""
        feedback_parts = [
            f'Phase {phase_name} requires refinement.',
            f'Current quality: {current_score:.2f}',
            f'Required quality: {threshold:.2f}',
            f'Gap: {threshold - current_score:.2f}'
        ]
        
        # Add specific feedback based on validation results
        validation_result = state.validation_results.get(phase_name)
        if validation_result:
            if validation_result.errors:
                feedback_parts.append(f"Errors to fix: {', '.join(validation_result.errors[:3])}")
            if validation_result.warnings:
                feedback_parts.append(f"Warnings: {', '.join(validation_result.warnings[:3])}")
        
        # Add domain-specific feedback
        if state.domain == 'agentool':
            feedback_parts.append('Ensure all tool specifications are complete')
            feedback_parts.append('Verify import statements and type hints')
        elif state.domain == 'api':
            feedback_parts.append('Ensure all endpoints have proper validation')
            feedback_parts.append('Check response schemas match specifications')
        
        return '\n'.join(feedback_parts)


@dataclass
class QualityCheckNode(BaseNode[WorkflowState, Any, WorkflowState]):
    """Quality check node that branches based on quality.
    
    Per workflow-graph-system.md lines 629-634:
    - Evaluates quality score against threshold
    - Returns ApprovalNode if passed
    - Returns RefinementNode if failed
    """
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Check quality and branch accordingly."""
        phase_name = ctx.state.current_phase
        phase_def = ctx.state.workflow_def.phases.get(phase_name)
        
        if not phase_def:
            logger.warning(f'No phase definition for {phase_name}')
            from .control import NextPhaseNode
            return NextPhaseNode()
        
        # Get quality score
        quality_score = ctx.state.quality_scores.get(phase_name, 0.0)
        threshold = phase_def.quality_threshold
        
        # Check if quality meets threshold
        if quality_score >= threshold:
            logger.info(f'Quality check passed: {quality_score:.2f} >= {threshold:.2f}')
            return ApprovalNode()
        else:
            logger.info(f'Quality check failed: {quality_score:.2f} < {threshold:.2f}')
            # Use RefinementLoopNode to manage refinement logic
            return RefinementLoopNode()


# Register approval and refinement nodes
register_node_class('approval', ApprovalNode)
register_node_class('refinement_loop', RefinementLoopNode)
register_node_class('quality_check', QualityCheckNode)