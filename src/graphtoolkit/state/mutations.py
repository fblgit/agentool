"""
GraphToolkit State Mutation Patterns.

Immutable state mutation helpers and patterns.
"""

from typing import Any, Dict, Optional, Set, List
from dataclasses import replace
from datetime import datetime
import logging

from ..core.types import (
    WorkflowState,
    StorageRef,
    RefinementRecord,
    ValidationResult,
    TokenUsage
)


logger = logging.getLogger(__name__)


def update_phase_complete(
    state: WorkflowState,
    phase_name: str,
    output_ref: Optional[StorageRef] = None
) -> WorkflowState:
    """
    Mark a phase as complete and optionally store its output reference.
    
    Args:
        state: Current workflow state
        phase_name: Name of the completed phase
        output_ref: Optional storage reference for phase output
        
    Returns:
        New state with phase marked complete
    """
    updates = {
        'completed_phases': state.completed_phases | {phase_name},
        'updated_at': datetime.now()
    }
    
    if output_ref:
        updates['phase_outputs'] = {
            **state.phase_outputs,
            phase_name: output_ref
        }
    
    # Determine next phase
    next_phase = state.workflow_def.get_next_phase(phase_name)
    if next_phase:
        updates['current_phase'] = next_phase
        # Set current node to first node of next phase
        phase_def = state.workflow_def.phases.get(next_phase)
        if phase_def and phase_def.atomic_nodes:
            updates['current_node'] = phase_def.atomic_nodes[0]
    
    return replace(state, **updates)


def update_for_refinement(
    state: WorkflowState,
    phase_name: str,
    feedback: str,
    quality_score: float
) -> WorkflowState:
    """
    Update state for phase refinement.
    
    Args:
        state: Current workflow state
        phase_name: Phase to refine
        feedback: Refinement feedback
        quality_score: Current quality score
        
    Returns:
        New state prepared for refinement
    """
    # Increment refinement count
    new_refinement_count = {
        **state.refinement_count,
        phase_name: state.refinement_count.get(phase_name, 0) + 1
    }
    
    # Add refinement feedback to domain data
    new_domain_data = {
        **state.domain_data,
        f'{phase_name}_feedback': feedback,
        f'{phase_name}_previous_score': quality_score
    }
    
    # Create refinement record
    record = RefinementRecord(
        iteration=new_refinement_count[phase_name],
        timestamp=datetime.now(),
        previous_score=state.quality_scores.get(phase_name, 0.0),
        new_score=quality_score,
        feedback=feedback,
        changes_made=[],
        code_before_ref=state.phase_outputs.get(phase_name),
        code_after_ref=None
    )
    
    # Update refinement history
    phase_history = state.refinement_history.get(phase_name, [])
    new_refinement_history = {
        **state.refinement_history,
        phase_name: phase_history + [record]
    }
    
    return replace(
        state,
        current_phase=phase_name,  # Stay in same phase
        current_node='template_render',  # Restart from template rendering
        refinement_count=new_refinement_count,
        domain_data=new_domain_data,
        refinement_history=new_refinement_history,
        updated_at=datetime.now()
    )


def update_quality_score(
    state: WorkflowState,
    phase_name: str,
    score: float
) -> WorkflowState:
    """
    Update quality score for a phase.
    
    Args:
        state: Current workflow state
        phase_name: Phase name
        score: Quality score (0.0 to 1.0)
        
    Returns:
        New state with updated quality score
    """
    return replace(
        state,
        quality_scores={
            **state.quality_scores,
            phase_name: score
        },
        updated_at=datetime.now()
    )


def update_validation_result(
    state: WorkflowState,
    phase_name: str,
    result: ValidationResult
) -> WorkflowState:
    """
    Update validation result for a phase.
    
    Args:
        state: Current workflow state
        phase_name: Phase name
        result: Validation result
        
    Returns:
        New state with validation result
    """
    return replace(
        state,
        validation_results={
            **state.validation_results,
            phase_name: result
        },
        updated_at=datetime.now()
    )


def update_token_usage(
    state: WorkflowState,
    phase_name: str,
    usage: TokenUsage
) -> WorkflowState:
    """
    Update or accumulate token usage for a phase.
    
    Args:
        state: Current workflow state
        phase_name: Phase name
        usage: Token usage to add
        
    Returns:
        New state with updated token usage
    """
    existing_usage = state.total_token_usage.get(phase_name)
    
    if existing_usage:
        # Accumulate usage
        new_usage = existing_usage + usage
    else:
        new_usage = usage
    
    return replace(
        state,
        total_token_usage={
            **state.total_token_usage,
            phase_name: new_usage
        },
        updated_at=datetime.now()
    )


def add_domain_data(
    state: WorkflowState,
    key: str,
    value: Any
) -> WorkflowState:
    """
    Add or update domain-specific data.
    
    Args:
        state: Current workflow state
        key: Data key
        value: Data value
        
    Returns:
        New state with updated domain data
    """
    return replace(
        state,
        domain_data={
            **state.domain_data,
            key: value
        },
        updated_at=datetime.now()
    )


def merge_domain_data(
    state: WorkflowState,
    data: Dict[str, Any]
) -> WorkflowState:
    """
    Merge multiple domain data entries.
    
    Args:
        state: Current workflow state
        data: Dictionary of data to merge
        
    Returns:
        New state with merged domain data
    """
    return replace(
        state,
        domain_data={
            **state.domain_data,
            **data
        },
        updated_at=datetime.now()
    )


def increment_retry_count(
    state: WorkflowState,
    retry_key: str
) -> WorkflowState:
    """
    Increment retry count for a specific operation.
    
    Args:
        state: Current workflow state
        retry_key: Unique key for the retry operation
        
    Returns:
        New state with incremented retry count
    """
    current_count = state.retry_counts.get(retry_key, 0)
    
    return replace(
        state,
        retry_counts={
            **state.retry_counts,
            retry_key: current_count + 1
        },
        updated_at=datetime.now()
    )


def reset_retry_count(
    state: WorkflowState,
    retry_key: str
) -> WorkflowState:
    """
    Reset retry count for a specific operation.
    
    Args:
        state: Current workflow state
        retry_key: Unique key for the retry operation
        
    Returns:
        New state with reset retry count
    """
    new_retry_counts = dict(state.retry_counts)
    new_retry_counts.pop(retry_key, None)
    
    return replace(
        state,
        retry_counts=new_retry_counts,
        updated_at=datetime.now()
    )


def update_iteration_state(
    state: WorkflowState,
    items: Optional[List[Any]] = None,
    results: Optional[List[Any]] = None,
    index: Optional[int] = None
) -> WorkflowState:
    """
    Update iteration state for processing items.
    
    Args:
        state: Current workflow state
        items: Items to iterate over (if starting)
        results: Results to append (if processing)
        index: Current iteration index
        
    Returns:
        New state with updated iteration state
    """
    updates = {'updated_at': datetime.now()}
    
    if items is not None:
        updates['iter_items'] = items
        updates['iter_index'] = 0
        updates['iter_results'] = []
    
    if results is not None:
        updates['iter_results'] = state.iter_results + results
    
    if index is not None:
        updates['iter_index'] = index
    
    return replace(state, **updates)


def validate_state_transition(
    old_state: WorkflowState,
    new_state: WorkflowState
) -> bool:
    """
    Validate that a state transition is valid.
    
    Args:
        old_state: Previous state
        new_state: New state
        
    Returns:
        True if transition is valid
        
    Raises:
        ValueError: If transition is invalid
    """
    # Workflow ID must not change
    if old_state.workflow_id != new_state.workflow_id:
        raise ValueError("Workflow ID cannot be changed")
    
    # Domain must not change
    if old_state.domain != new_state.domain:
        raise ValueError("Domain cannot be changed")
    
    # Workflow definition must not change
    if old_state.workflow_def != new_state.workflow_def:
        raise ValueError("Workflow definition cannot be changed during execution")
    
    # Completed phases should only grow
    if not old_state.completed_phases.issubset(new_state.completed_phases):
        raise ValueError("Completed phases cannot be removed")
    
    # Phase outputs should only grow
    if not set(old_state.phase_outputs.keys()).issubset(set(new_state.phase_outputs.keys())):
        raise ValueError("Phase outputs cannot be removed")
    
    return True


def create_recovery_state(
    state: WorkflowState,
    error: str,
    node_id: Optional[str] = None
) -> WorkflowState:
    """
    Create a recovery state after an error.
    
    Args:
        state: Current workflow state
        error: Error message
        node_id: Node that caused the error
        
    Returns:
        New state with error information
    """
    return replace(
        state,
        domain_data={
            **state.domain_data,
            'error': error,
            'error_node': node_id,
            'error_time': datetime.now().isoformat(),
            'recovery_attempted': True
        },
        updated_at=datetime.now()
    )