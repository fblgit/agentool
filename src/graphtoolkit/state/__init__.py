"""
GraphToolkit State Management.

State mutation and persistence utilities.
"""

from .mutations import (
    update_phase_complete,
    update_for_refinement,
    update_quality_score,
    update_validation_result,
    update_token_usage,
    add_domain_data,
    merge_domain_data,
    increment_retry_count,
    reset_retry_count,
    update_iteration_state,
    validate_state_transition,
    create_recovery_state
)

from .persistence import (
    StateSerializer,
    StatePersistence,
    FileStatePersistence,
    KVStatePersistence,
    create_checkpoint,
    restore_checkpoint
)

__all__ = [
    # Mutations
    'update_phase_complete',
    'update_for_refinement',
    'update_quality_score',
    'update_validation_result',
    'update_token_usage',
    'add_domain_data',
    'merge_domain_data',
    'increment_retry_count',
    'reset_retry_count',
    'update_iteration_state',
    'validate_state_transition',
    'create_recovery_state',
    
    # Persistence
    'StateSerializer',
    'StatePersistence',
    'FileStatePersistence',
    'KVStatePersistence',
    'create_checkpoint',
    'restore_checkpoint'
]