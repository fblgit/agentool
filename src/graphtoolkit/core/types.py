"""GraphToolkit Core Type Definitions.

This module contains the canonical type definitions for the state-driven meta-framework.
These types define the uniform plane where all workflow configuration lives.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Type

from pydantic import BaseModel


class RetryBackoff(str, Enum):
    """Retry backoff strategies."""
    NONE = 'none'
    LINEAR = 'linear'
    EXPONENTIAL = 'exponential'


class ConditionOperator(str, Enum):
    """Condition comparison operators."""
    EQUALS = '=='
    NOT_EQUALS = '!='
    GREATER_THAN = '>'
    GREATER_EQUAL = '>='
    LESS_THAN = '<'
    LESS_EQUAL = '<='
    IN = 'in'
    NOT_IN = 'not_in'


class StorageType(str, Enum):
    """Storage backend types."""
    KV = 'kv'
    FS = 'fs'


@dataclass(frozen=True)
class StorageRef:
    """Reference to stored data."""
    storage_type: StorageType
    key: str
    created_at: datetime
    version: Optional[int] = None
    size_bytes: Optional[int] = None
    
    def __str__(self) -> str:
        return f'{self.storage_type.value}://{self.key}'


@dataclass(frozen=True)
class TemplateConfig:
    """Template configuration for a phase."""
    system_template: str  # Path to system prompt template
    user_template: str  # Path to user prompt template
    variables: Optional[Dict[str, str]] = None  # Default variables
    fragments: Optional[List[str]] = None  # Reusable template fragments


@dataclass(frozen=True)
class ModelParameters:
    """LLM model parameters."""
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None


@dataclass(frozen=True)
class NodeConfig:
    """Configuration for atomic node behavior."""
    node_type: str
    retryable: bool = False
    max_retries: int = 0
    retry_backoff: RetryBackoff = RetryBackoff.EXPONENTIAL
    retry_delay: float = 1.0  # Base delay in seconds
    
    # Iteration support
    iter_enabled: bool = False
    iter_in_type: Optional[Type] = None
    iter_out_type: Optional[Type] = None
    
    # Caching
    cacheable: bool = False
    cache_ttl: int = 3600  # Seconds
    
    # Timeouts
    timeout: Optional[float] = None  # Seconds


@dataclass(frozen=True)
class ConditionConfig:
    """Configuration for state-driven conditions."""
    condition_type: Literal['quality_gate', 'state_path', 'threshold', 'custom']
    
    # For quality_gate
    quality_field: Optional[str] = None
    threshold: Optional[float] = None
    
    # For state_path conditions
    state_path: Optional[str] = None
    operator: Optional[ConditionOperator] = None
    expected_value: Optional[Any] = None
    
    # For custom conditions
    custom_evaluator: Optional[str] = None  # Function name in registry
    
    def evaluate(self, state: 'WorkflowState') -> bool:
        """Evaluate condition against state."""
        if self.condition_type == 'quality_gate':
            score = state.quality_scores.get(self.quality_field, 0.0)
            return score >= self.threshold
            
        elif self.condition_type == 'state_path':
            # Navigate state path (e.g., "domain_data.complexity")
            value = state
            for part in self.state_path.split('.'):
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    value = getattr(value, part, None)
                if value is None:
                    return False
            
            # Compare with expected value
            if self.operator == ConditionOperator.EQUALS:
                return value == self.expected_value
            elif self.operator == ConditionOperator.NOT_EQUALS:
                return value != self.expected_value
            elif self.operator == ConditionOperator.GREATER_THAN:
                return value > self.expected_value
            elif self.operator == ConditionOperator.GREATER_EQUAL:
                return value >= self.expected_value
            elif self.operator == ConditionOperator.LESS_THAN:
                return value < self.expected_value
            elif self.operator == ConditionOperator.LESS_EQUAL:
                return value <= self.expected_value
            elif self.operator == ConditionOperator.IN:
                return value in self.expected_value
            elif self.operator == ConditionOperator.NOT_IN:
                return value not in self.expected_value
                
        elif self.condition_type == 'threshold':
            # Similar to state_path but specifically for numeric thresholds
            value = state
            for part in self.state_path.split('.'):
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    value = getattr(value, part, None)
                if value is None:
                    return False
            return self.operator == ConditionOperator.GREATER_EQUAL and value >= self.expected_value
            
        return False


@dataclass(frozen=True)
class PhaseDefinition:
    """Definition of a workflow phase."""
    phase_name: str
    atomic_nodes: List[str]  # Ordered list of atomic node IDs
    input_schema: Type[BaseModel]
    output_schema: Type[BaseModel]
    dependencies: List[str] = field(default_factory=list)  # Previous phases
    
    # Templates
    templates: Optional[TemplateConfig] = None
    
    # Storage
    storage_pattern: str = 'workflow/{workflow_id}/{phase}'
    storage_type: StorageType = StorageType.KV
    additional_storage_patterns: Optional[Dict[str, str]] = None  # Additional storage patterns
    
    # Quality control
    quality_threshold: float = 0.8
    allow_refinement: bool = True
    max_refinements: int = 3
    
    # Model configuration
    model_config: Optional[ModelParameters] = None
    
    # Domain hint
    domain: Optional[str] = None
    
    # Iteration configuration
    iteration_config: Optional[Dict[str, Any]] = None  # Iteration settings for phases that process multiple items


@dataclass(frozen=True)
class WorkflowDefinition:
    """Complete workflow definition - the uniform plane."""
    domain: str
    phases: Dict[str, PhaseDefinition]
    phase_sequence: List[str]
    node_configs: Dict[str, NodeConfig]
    
    # Conditional logic
    conditions: Dict[str, ConditionConfig] = field(default_factory=dict)
    
    # Global settings
    enable_refinement: bool = True
    enable_parallel: bool = False
    max_execution_time: int = 3600  # Seconds
    
    # Metadata
    version: str = '1.0.0'
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_phase(self, phase_name: str) -> Optional[PhaseDefinition]:
        """Get phase definition by name."""
        return self.phases.get(phase_name)
    
    def get_next_phase(self, current_phase: str) -> Optional[str]:
        """Get next phase in sequence."""
        try:
            idx = self.phase_sequence.index(current_phase)
            if idx + 1 < len(self.phase_sequence):
                return self.phase_sequence[idx + 1]
        except ValueError:
            pass
        return None


@dataclass(frozen=True)
class ValidationResult:
    """Result of validation operations."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RefinementRecord:
    """Record of a refinement iteration."""
    iteration: int
    timestamp: datetime
    previous_score: float
    new_score: float
    feedback: str
    changes_made: List[str]
    code_before_ref: Optional[StorageRef] = None
    code_after_ref: Optional[StorageRef] = None


@dataclass
class WorkflowState:
    """State-driven workflow execution.
    
    Per workflow-graph-system.md design:
    - WorkflowDefinition is part of state (the uniform plane)
    - Enables state-driven conditional logic and dynamic workflows
    - Configuration access via ctx.state.workflow_def
    """
    # Workflow definition (the uniform plane)  
    workflow_def: 'WorkflowDefinition'
    
    # Core identity
    workflow_id: str
    domain: str
    
    # Execution position
    current_phase: str
    current_node: str = ''
    completed_phases: Set[str] = field(default_factory=set)
    
    # Storage references (not data itself)
    phase_outputs: Dict[str, StorageRef] = field(default_factory=dict)
    
    # Domain-flexible data
    domain_data: Dict[str, Any] = field(default_factory=dict)
    
    # Quality & refinement tracking
    quality_scores: Dict[str, float] = field(default_factory=dict)
    refinement_count: Dict[str, int] = field(default_factory=dict)
    refinement_history: Dict[str, List[RefinementRecord]] = field(default_factory=dict)
    validation_results: Dict[str, ValidationResult] = field(default_factory=dict)
    
    # Iteration state
    iter_items: List[Any] = field(default_factory=list)
    iter_results: List[Any] = field(default_factory=list) 
    iter_index: int = 0
    
    # Retry tracking (per node instance)
    retry_counts: Dict[str, int] = field(default_factory=dict)
    
    # Token usage tracking
    total_token_usage: Dict[str, 'TokenUsage'] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def get_current_node_config(self) -> Optional[NodeConfig]:
        """Get configuration for current node."""
        return self.workflow_def.node_configs.get(self.current_node)
    
    def get_next_atomic_node(self) -> Optional[str]:
        """Get next node in current phase."""
        phase_def = self.workflow_def.phases.get(self.current_phase)
        if not phase_def or not self.current_node:
            return None
        
        try:
            current_idx = phase_def.atomic_nodes.index(self.current_node)
            if current_idx + 1 < len(phase_def.atomic_nodes):
                return phase_def.atomic_nodes[current_idx + 1]
        except ValueError:
            pass
        return None
    
    def get_current_phase_def(self) -> Optional['PhaseDefinition']:
        """Get current phase definition."""
        return self.workflow_def.phases.get(self.current_phase)
    
    def with_storage_ref(self, phase: str, ref: StorageRef) -> 'WorkflowState':
        """Helper to update storage reference."""
        from dataclasses import replace
        return replace(
            self,
            phase_outputs={**self.phase_outputs, phase: ref},
            updated_at=datetime.now()
        )


@dataclass(frozen=True)
class TokenUsage:
    """Token usage tracking."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    
    def __add__(self, other: 'TokenUsage') -> 'TokenUsage':
        """Add token usage."""
        if self.model != other.model:
            raise ValueError(f'Cannot add usage for different models: {self.model} vs {other.model}')
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            model=self.model
        )


@dataclass(frozen=True)
class ProcessingState:
    """Transient processing state (not persisted)."""
    last_result: Optional[Any] = None
    last_error: Optional[str] = None
    last_retryable_error: Optional[str] = None
    retry_after: Optional[float] = None
    
    # For parallel processing
    parallel_results: List[Any] = field(default_factory=list)
    parallel_errors: List[str] = field(default_factory=list)
    
    # For conditional branching
    last_condition: Optional[bool] = None
    branch_taken: Optional[str] = None
    branch_history: List[str] = field(default_factory=list)
    
    # Error tracking
    tools_failed: List[str] = field(default_factory=list)
    processing_errors: Dict[str, str] = field(default_factory=dict)
    error_count: int = 0