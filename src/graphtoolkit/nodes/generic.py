"""GraphToolkit GenericPhaseNode.

The meta-framework orchestrator that starts phases by returning first atomic node.
"""

import logging
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Union

try:
    from pydantic_graph import BaseNode as PydanticBaseNode, End, GraphRunContext
    HAS_PYDANTIC_GRAPH = True
except ImportError:
    # Development stubs
    HAS_PYDANTIC_GRAPH = False
    
    class PydanticBaseNode:
        pass
    
    class GraphRunContext:
        def __init__(self, state, deps):
            self.state = state
            self.deps = deps
    
    class End:
        def __init__(self, result):
            self.result = result

from ..core.factory import create_node_instance, register_node_class
from ..core.types import WorkflowState
from .base import NonRetryableError

logger = logging.getLogger(__name__)


@dataclass
class GenericPhaseNode(PydanticBaseNode):
    """Starts a phase by returning its first atomic node.
    
    This is the key innovation of the meta-framework:
    - Does NOT execute the phase work itself
    - Returns first atomic node which chains to rest
    - Reads all configuration from state
    - No constructor parameters needed
    """
    
    async def run(self, ctx: GraphRunContext[WorkflowState, Any]) -> Union[PydanticBaseNode, End[WorkflowState]]:
        """Start phase execution by returning first atomic node.
        
        The atomic nodes will chain together:
        DependencyCheck → LoadDependencies → TemplateRender → 
        LLMCall → SchemaValidation → SaveOutput → 
        StateUpdate → QualityGate
        """
        logger.debug(f"[GenericPhaseNode] === ENTRY === Workflow: {ctx.state.workflow_id}")
        logger.debug(f"[GenericPhaseNode] Current phase: {ctx.state.current_phase}")
        logger.debug(f"[GenericPhaseNode] Current node: {ctx.state.current_node}")
        logger.debug(f"[GenericPhaseNode] Completed phases: {ctx.state.completed_phases}")
        logger.debug(f"[GenericPhaseNode] Domain data keys: {list(ctx.state.domain_data.keys())}")
        logger.debug(f"[GenericPhaseNode] Retry counts: {ctx.state.retry_counts}")
        logger.debug(f"[GenericPhaseNode] Phase outputs: {list(ctx.state.phase_outputs.keys())}")
        
        # Get current phase from state
        if not ctx.state.current_phase:
            logger.debug(f"[GenericPhaseNode] No current phase set, checking phase sequence")
            # If no current phase, start with first phase
            if ctx.state.workflow_def.phase_sequence:
                first_phase = ctx.state.workflow_def.phase_sequence[0]
                logger.info(f"[GenericPhaseNode] Setting first phase: {first_phase}")
                new_state = replace(ctx.state, current_phase=first_phase)
                ctx = GraphRunContext(state=new_state, deps=ctx.deps)
            else:
                logger.error(f"[GenericPhaseNode] FATAL: No phases defined in workflow")
                raise NonRetryableError('No phases defined in workflow')
        
        phase_name = ctx.state.current_phase
        logger.debug(f"[GenericPhaseNode] Looking up phase definition for: {phase_name}")
        phase_def = ctx.state.workflow_def.phases.get(phase_name)
        
        if not phase_def:
            logger.error(f"[GenericPhaseNode] FATAL: Phase {phase_name} not found")
            logger.error(f"[GenericPhaseNode] Available phases: {list(ctx.state.workflow_def.phases.keys())}")
            raise NonRetryableError(f'Phase {phase_name} not found in workflow definition')
        
        logger.info(f'[GenericPhaseNode] Starting phase {phase_name} with {len(phase_def.atomic_nodes)} atomic nodes')
        logger.debug(f"[GenericPhaseNode] Atomic nodes: {phase_def.atomic_nodes}")
        
        # Check if phase already completed
        if phase_name in ctx.state.completed_phases:
            logger.info(f'[GenericPhaseNode] Phase {phase_name} already completed, moving to next')
            # Move to next phase
            from .atomic.control import NextPhaseNode
            return NextPhaseNode()
        
        # Get first atomic node in the phase
        if not phase_def.atomic_nodes:
            logger.warning(f'[GenericPhaseNode] Phase {phase_name} has no atomic nodes defined')
            # Mark phase as complete and move on
            new_state = replace(
                ctx.state,
                completed_phases=ctx.state.completed_phases | {phase_name}
            )
            from .atomic.control import NextPhaseNode
            return NextPhaseNode()
        
        first_node_id = phase_def.atomic_nodes[0]
        
        # Update state to track current node
        new_state = replace(
            ctx.state,
            current_node=first_node_id
        )
        
        logger.info(f'[GenericPhaseNode] Phase {phase_name} starting with node {first_node_id}')
        logger.debug(f"[GenericPhaseNode] === EXIT === Returning {first_node_id} node")
        
        # Return first atomic node - it will chain to the rest
        # Each atomic node knows how to find the next node in sequence
        return create_node_instance(first_node_id)


@dataclass
class WorkflowStartNode(PydanticBaseNode):
    """Entry point for workflow execution.
    Sets up initial state and returns GenericPhaseNode.
    """
    
    async def run(self, ctx: GraphRunContext[WorkflowState, Any]) -> Union[PydanticBaseNode, End[WorkflowState]]:
        """Initialize workflow and start first phase."""
        logger.info(f'Starting workflow {ctx.state.workflow_id} for domain {ctx.state.domain}')
        
        # Validate workflow definition
        if not ctx.state.workflow_def:
            raise NonRetryableError('No workflow definition in state')
        
        if not ctx.state.workflow_def.phase_sequence:
            raise NonRetryableError('No phase sequence defined')
        
        # Set current phase if not set
        if not ctx.state.current_phase:
            first_phase = ctx.state.workflow_def.phase_sequence[0]
            new_state = replace(ctx.state, current_phase=first_phase)
            ctx = GraphRunContext(state=new_state, deps=ctx.deps)
        
        # Log workflow configuration
        logger.info(f'Workflow has {len(ctx.state.workflow_def.phases)} phases: {ctx.state.workflow_def.phase_sequence}')
        
        # Return GenericPhaseNode to start first phase
        return GenericPhaseNode()


@dataclass
class WorkflowEndNode(PydanticBaseNode):
    """Terminal node for successful workflow completion.
    """
    
    async def run(self, ctx: GraphRunContext[WorkflowState, Any]) -> End[WorkflowState]:
        """Mark workflow as complete and return final state."""
        logger.info(f'Workflow {ctx.state.workflow_id} completed successfully')
        
        # Log summary
        logger.info(f'Completed phases: {ctx.state.completed_phases}')
        logger.info(f'Quality scores: {ctx.state.quality_scores}')
        
        # Calculate total token usage
        total_tokens = 0
        for phase, usage in ctx.state.total_token_usage.items():
            total_tokens += usage.total_tokens
        logger.info(f'Total tokens used: {total_tokens}')
        
        # Update final state
        final_state = replace(
            ctx.state,
            domain_data={
                **ctx.state.domain_data,
                'workflow_complete': True,
                'completion_time': datetime.now().isoformat()
            }
        )
        
        return End(final_state)


# Register the generic nodes
register_node_class('generic_phase', GenericPhaseNode)
register_node_class('workflow_start', WorkflowStartNode)
register_node_class('workflow_end', WorkflowEndNode)