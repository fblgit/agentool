"""
AgenToolkit Domain for GraphToolkit
====================================

This domain implements the AgenTool creation workflow using the GraphToolkit
meta-framework. It provides phases for analyzing, specifying, crafting, and
evaluating AgenTools based on task descriptions and the existing tool catalog.

State Management Pattern: MUTABLE STATE
---------------------------------------
This domain uses direct state mutation for efficiency and simplicity.

Rules:
1. Modify state collections directly: ctx.state.domain_data['key'] = value
2. Update mutable fields in place: ctx.state.completed_phases.add(phase)
3. NO use of replace() or immutable patterns
4. All nodes receive the same state instance (mutations are visible)

This approach prioritizes:
- Performance (no copying overhead)
- Simplicity (direct updates)
- Consistency with GraphToolkit patterns

Storage Pattern: HIERARCHICAL KV
---------------------------------
workflow/{workflow_id}/input/*   - Input data (catalog, prompt)
workflow/{workflow_id}/render/*  - Rendered templates per phase
workflow/{workflow_id}/output/*  - LLM outputs per phase

Examples:
- workflow/abc123/input/catalog      # The AgenTool catalog
- workflow/abc123/input/prompt       # User's task description
- workflow/abc123/render/analyzer    # Rendered analyzer template
- workflow/abc123/output/analyzer    # Analyzer LLM output
- workflow/abc123/render/specifier   # Rendered specifier template
- workflow/abc123/output/specifier   # Specifier LLM output

This enables:
- Clean separation of concerns
- Easy debugging of each phase
- Consistent retrieval patterns
- Scalable addition of new phases
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from ..core.types import PhaseDefinition, StorageType, TemplateConfig, ModelParameters
from ..core.registry import register_phase

# Import the AnalyzerOutput, CodeOutput, and EvaluationOutput from agents.models
from agents.models import AnalyzerOutput, ToolSpecification, CodeOutput, EvaluationOutput


# Input schemas for each phase

class AgenToolkitAnalyzerInput(BaseModel):
    """Input for the analyzer phase of AgenToolkit workflow."""
    task_description: str = Field(
        min_length=10,
        max_length=2000,
        description="Description of the AgenTool capability to create"
    )
    constraints: Optional[List[str]] = Field(
        default=None,
        description="Optional constraints or requirements for the tool"
    )
    examples: Optional[List[str]] = Field(
        default=None,
        description="Optional examples of desired behavior"
    )


class AgenToolkitSpecifierInput(BaseModel):
    """Input for the specifier phase - uses analyzer output."""
    tool_name: str = Field(
        description="Name of the tool to specify (from analyzer's missing_tools)"
    )
    analyzer_output: Dict[str, Any] = Field(
        description="Complete output from analyzer phase"
    )


class AgenToolkitCrafterInput(BaseModel):
    """Input for the crafter phase - uses specification."""
    specification: Dict[str, Any] = Field(
        description="Tool specification from specifier phase"
    )
    existing_tools: List[str] = Field(
        description="List of existing tools to integrate with"
    )


class AgenToolkitEvaluatorInput(BaseModel):
    """Input for the evaluator phase - uses crafted implementation."""
    implementation: str = Field(
        description="The crafted Python implementation"
    )
    specification: Dict[str, Any] = Field(
        description="Original specification to evaluate against"
    )


class AgenToolkitRefinerInput(BaseModel):
    """Input for the refiner phase - uses previous implementation and evaluation feedback."""
    previous_code: str = Field(
        description="The previous implementation from crafter phase"
    )
    specification: Dict[str, Any] = Field(
        description="Original specification to implement"
    )
    evaluation_feedback: Dict[str, Any] = Field(
        description="Evaluation results with issues and improvements to address"
    )


# Output schemas (reusing from agents.models where possible)
# Using CodeOutput from agents.models for crafter phase
# Using EvaluationOutput from agents.models for evaluator phase
# Using CodeOutput from agents.models for refiner phase (same as crafter)


# Register phases for agentoolkit domain

# Phase 1: Analyzer - Analyze task and identify required tools
analyzer_phase = PhaseDefinition(
    phase_name='analyzer',
    domain='agentoolkit',
    atomic_nodes=[
        'dependency_check',      # No dependencies for first phase
        'load_dependencies',     # Loads catalog from agentool_mgmt
        'template_render',       # Renders analyzer template
        'llm_call',             # Calls LLM with rendered template
        'schema_validation',     # Validates against AnalyzerOutput
        'save_phase_output',     # Saves to output/analyzer
        'state_update',         # Marks phase complete
        'quality_gate'          # Checks quality threshold
    ],
    input_schema=AgenToolkitAnalyzerInput,
    output_schema=AnalyzerOutput,
    dependencies=[],  # First phase has no dependencies
    templates=TemplateConfig(
        system_template='agentool/system/analyzer.jinja',
        user_template='agentool/prompts/analyze_catalog.jinja',
        variables={'schema_json': AnalyzerOutput.model_json_schema()}
    ),
    storage_pattern='workflow/{workflow_id}/output/analyzer',
    storage_type=StorageType.KV,
    additional_storage_patterns={
        'catalog': 'workflow/{workflow_id}/input/catalog',
        'prompt': 'workflow/{workflow_id}/input/prompt',
        'rendered': 'workflow/{workflow_id}/render/analyzer'
    },
    quality_threshold=0.8,
    allow_refinement=True,
    max_refinements=2,
    model_config=ModelParameters(
        temperature=0.7,
        max_tokens=2000
    )
)

# Phase 2: Specifier - Create detailed specification for each missing tool
# Note: We need to import ToolSpecificationLLM for the output schema
from agents.models import ToolSpecificationLLM

specifier_phase = PhaseDefinition(
    phase_name='specifier',
    domain='agentoolkit',
    atomic_nodes=[
        'dependency_check',
        'load_dependencies',     # Loads analyzer output
        'iteration_control',     # NEW: Start iteration over missing_tools
        'template_render',       # Renders specifier template for current tool
        'llm_call',             # Calls LLM for specification
        'schema_validation',     # Validates against ToolSpecificationLLM
        'save_iteration_output', # NEW: Saves individual specification (returns to iteration_control)
        'aggregation',          # NEW: Aggregate all specifications
        'save_phase_output',    # Saves aggregated output
        'state_update',
        'quality_gate'
    ],
    input_schema=AgenToolkitSpecifierInput,
    output_schema=ToolSpecificationLLM,  # Individual spec uses LLM version
    dependencies=['analyzer'],
    templates=TemplateConfig(
        system_template='agentool/system/specifier.jinja',  # Fixed path
        user_template='agentool/prompts/create_specification.jinja',  # Fixed path
        variables={'schema_json': ToolSpecificationLLM.model_json_schema()}
    ),
    storage_pattern='workflow/{workflow_id}/output/specifier',
    storage_type=StorageType.KV,
    additional_storage_patterns={
        'rendered': 'workflow/{workflow_id}/render/specifier'
    },
    iteration_config={
        'enabled': True,
        'items_source': 'analyzer_output.missing_tools',
        'item_storage_pattern': 'workflow/{workflow_id}/specification/{item_name}'
    },
    quality_threshold=0.85,
    allow_refinement=False,  # No refinement during iteration
    max_refinements=0,
    model_config=ModelParameters(
        temperature=0.5,  # Lower temperature for more consistent specs
        max_tokens=3000
    )
)

# Phase 3: Crafter - Implement the AgenTool based on specification
crafter_phase = PhaseDefinition(
    phase_name='crafter',
    domain='agentoolkit',
    atomic_nodes=[
        'dependency_check',
        'load_dependencies',     # Loads specifications from specifier
        'iteration_control',     # Start iteration over missing_tools
        'template_render',       # Renders crafter template for current tool
        'llm_call',             # Calls LLM for implementation
        'schema_validation',     # Validates against ImplementationOutput
        'save_iteration_output', # Saves individual implementation
        'aggregation',          # Aggregate all implementations
        'save_phase_output',    # Saves aggregated output
        'state_update',
        'quality_gate'
    ],
    input_schema=AgenToolkitCrafterInput,
    output_schema=CodeOutput,  # Using CodeOutput from agents.models
    dependencies=['analyzer', 'specifier'],  # Needs both for context
    templates=TemplateConfig(
        system_template='agentool/system/crafter.jinja',  # Fixed path
        user_template='agentool/prompts/craft_implementation.jinja',  # Fixed path
        variables={}  # No schema needed, LLM outputs raw code
    ),
    storage_pattern='workflow/{workflow_id}/output/crafter',
    storage_type=StorageType.KV,
    additional_storage_patterns={
        'rendered': 'workflow/{workflow_id}/render/crafter'
    },
    iteration_config={
        'enabled': True,
        'items_source': 'analyzer_output.missing_tools',  # Same as specifier
        'item_storage_pattern': 'workflow/{workflow_id}/crafter/{item_name}'
    },
    quality_threshold=0.85,
    allow_refinement=False,  # No refinement during iteration
    max_refinements=0,
    model_config=ModelParameters(
        temperature=0.3,  # Low temperature for code generation
        max_tokens=4000
    )
)

# Phase 4: Evaluator - Evaluate each implementation (with iteration)
evaluator_phase = PhaseDefinition(
    phase_name='evaluator',
    domain='agentoolkit',
    atomic_nodes=[
        'dependency_check',
        'load_dependencies',     # Loads crafter and specifier outputs
        'iteration_control',     # Start iteration over missing_tools
        'template_render',       # Renders evaluator template for current tool
        'llm_call',             # Calls LLM for evaluation
        'schema_validation',     # Validates against EvaluationOutput
        'save_iteration_output', # Saves individual evaluation
        'aggregation',          # Aggregate all evaluations
        'save_phase_output',    # Saves aggregated output
        'state_update',
        'quality_gate'
    ],
    input_schema=AgenToolkitEvaluatorInput,
    output_schema=EvaluationOutput,  # From agents.models
    dependencies=['analyzer', 'specifier', 'crafter'],  # Needs analysis, specs, and implementations
    templates=TemplateConfig(
        system_template='agentool/system/evaluator.jinja',  # Fixed path
        user_template='agentool/prompts/evaluate_code.jinja',  # Fixed path
        variables={'schema_json': EvaluationOutput.model_json_schema()}
    ),
    storage_pattern='workflow/{workflow_id}/output/evaluator',
    storage_type=StorageType.KV,
    additional_storage_patterns={
        'rendered': 'workflow/{workflow_id}/render/evaluator'
    },
    iteration_config={
        'enabled': True,
        'items_source': 'analyzer_output.missing_tools',  # Same as specifier/crafter
        'item_storage_pattern': 'workflow/{workflow_id}/evaluation/{item_name}'
    },
    quality_threshold=0.8,  # Lower threshold - evaluation is assessment, not generation
    allow_refinement=False,  # No refinement for evaluation
    max_refinements=0,
    model_config=ModelParameters(
        temperature=0.3,  # Low temperature for consistent evaluation
        max_tokens=3000
    )
)

# Phase 5: Refiner - Refine implementations that failed evaluation
refiner_phase = PhaseDefinition(
    phase_name='refiner',
    domain='agentoolkit',
    atomic_nodes=[
        'dependency_check',
        'load_dependencies',     # Loads evaluator, crafter, specifier outputs
        'iteration_control',     # Start iteration over tools_to_refine
        'template_render',       # Renders refiner template for current tool
        'llm_call',             # Calls LLM for refined implementation
        'schema_validation',     # Validates against CodeOutput (same as crafter)
        'save_iteration_output', # Saves individual refined implementation
        'aggregation',          # Aggregate all refinements
        'save_phase_output',    # Saves aggregated output
        'state_update',
        'quality_gate'
    ],
    input_schema=AgenToolkitRefinerInput,
    output_schema=CodeOutput,  # Same as crafter - outputs refined code
    dependencies=['analyzer', 'specifier', 'crafter', 'evaluator'],  # Needs all previous phases
    templates=TemplateConfig(
        system_template='agentool/system/refiner.jinja',  # System prompt for refinement
        user_template='agentool/prompts/refine_implementation.jinja',  # User prompt with feedback
        variables={}  # No schema needed, outputs raw code like crafter
    ),
    storage_pattern='workflow/{workflow_id}/output/refiner',
    storage_type=StorageType.KV,
    additional_storage_patterns={
        'rendered': 'workflow/{workflow_id}/render/refiner'
    },
    iteration_config={
        'enabled': True,
        'items_source': 'tools_to_refine',  # Filtered list of tools needing refinement
        'item_storage_pattern': 'workflow/{workflow_id}/refine/{item_name}'
    },
    quality_threshold=0.9,  # Higher threshold for refined code
    allow_refinement=False,  # No further refinement during iteration
    max_refinements=0,
    model_config=ModelParameters(
        temperature=0.2,  # Even lower temperature for precise fixes
        max_tokens=4000
    )
)

# Register all phases
register_phase('agentoolkit.analyzer', analyzer_phase)
register_phase('agentoolkit.specifier', specifier_phase)
register_phase('agentoolkit.crafter', crafter_phase)
register_phase('agentoolkit.evaluator', evaluator_phase)
register_phase('agentoolkit.refiner', refiner_phase)

# Workflow configuration for convenience
AGENTOOLKIT_WORKFLOW_PHASES = [
    'analyzer',
    'specifier',
    'crafter',
    'evaluator',
    'refiner'  # Optional - only run if tools need refinement
]

# Export key components
__all__ = [
    'AgenToolkitAnalyzerInput',
    'AgenToolkitSpecifierInput',
    'AgenToolkitCrafterInput',
    'AgenToolkitEvaluatorInput',
    'AgenToolkitRefinerInput',
    'AnalyzerOutput',
    'ToolSpecification',
    'CodeOutput',
    'EvaluationOutput',
    'analyzer_phase',
    'specifier_phase',
    'crafter_phase',
    'evaluator_phase',
    'refiner_phase',
    'AGENTOOLKIT_WORKFLOW_PHASES'
]