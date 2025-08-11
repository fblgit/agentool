"""GraphToolkit AgenTool Domain.

Complete phase definitions for the AgenTool workflow domain.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from pydantic import BaseModel, Field

from ..core.registry import get_registry
from ..core.types import ModelParameters, PhaseDefinition, StorageType, TemplateConfig, WorkflowState

# Import V1 models for compatibility
try:
    from agents.models import (
        AnalyzerOutput as V1AnalyzerOutput,
        MissingToolSpec,
        ToolSpecification,
        ToolSpecificationLLM,
        ExistingToolInfo
    )
except ImportError:
    # Fallback if V1 models not available
    V1AnalyzerOutput = None
    MissingToolSpec = None
    ToolSpecification = None
    ToolSpecificationLLM = None
    ExistingToolInfo = None

# Input/Output Schemas for AgenTool phases

class AnalyzerInput(BaseModel):
    """Input schema for analyzer phase."""
    task_description: str = Field(description='Description of the AgenTool to create')
    domain: str = Field(default='agentool', description='Workflow domain')
    catalog: Optional[Dict[str, Any]] = Field(default=None, description='Existing tool catalog')
    guidelines: Optional[List[str]] = Field(default=None, description='Development guidelines')


# Use V1 AnalyzerOutput if available, otherwise use simplified version
if V1AnalyzerOutput:
    AnalyzerOutput = V1AnalyzerOutput
else:
    class AnalyzerOutput(BaseModel):
        """Output schema for analyzer phase."""
        name: str = Field(description='Descriptive name for the solution')
        description: str = Field(description='2-3 sentence explanation of the system')
        system_design: str = Field(description='Detailed architecture explanation')
        guidelines: List[str] = Field(description='Patterns and practices to follow')
        existing_tools: List[str] = Field(description='Exact names of existing tools to reuse')
        missing_tools: List[Dict[str, Any]] = Field(description='Specifications for new tools')
        success: bool = Field(default=True, description='Whether analysis succeeded')


class SpecifierInput(BaseModel):
    """Input schema for specifier phase."""
    missing_tools: List[str] = Field(description='Tools to specify')
    system_design: Dict[str, Any] = Field(description='System design from analyzer')
    guidelines: Optional[List[str]] = Field(default=None, description='Specification guidelines')


class SpecifierOutput(BaseModel):
    """Output schema for specifier phase."""
    specifications: List[Dict[str, Any]] = Field(description='Tool specifications')
    tool_count: int = Field(description='Number of tools specified')
    success: bool = Field(default=True, description='Whether specification succeeded')

# If V1 models available, we'll use ToolSpecification for actual specification


# Import V1 models for exact crafter compatibility  
try:
    from agents.models import CodeOutput, SpecificationOutput, ValidationOutput
    from agentoolkit.workflows.workflow_crafter import WorkflowCrafterInput, WorkflowCrafterOutput
    from agentoolkit.workflows.workflow_evaluator import WorkflowEvaluatorOutput
except ImportError:
    # Fallback definitions if V1 models not available
    class CodeOutput(BaseModel):
        code: str = Field(description='Generated implementation code')
        file_path: str = Field(description='File path for the implementation')
        
    class WorkflowCrafterInput(BaseModel):
        workflow_id: str = Field(description='Workflow identifier')
        model: str = Field(default='openai:gpt-4o', description='LLM model to use')
        
    class WorkflowCrafterOutput(BaseModel):
        success: bool = Field(description='Whether code generation succeeded')
        message: str = Field(description='Status message')
        data: Dict[str, Any] = Field(description='Generated code and metadata')
        state_ref: str = Field(description='Reference to stored state')
    
    class ValidationOutput(BaseModel):
        syntax_valid: bool = Field(description='Whether Python syntax is valid')
        imports_valid: bool = Field(description='Whether all imports are available')
        tests_passed: bool = Field(description='Whether tests passed')
        issues: List[str] = Field(description='Issues found during validation')
        fixes_applied: List[str] = Field(description='Fixes applied')
        improvements: List[str] = Field(description='Improvements made')
        final_code: str = Field(description='Final validated code')
        ready_for_deployment: bool = Field(description='Ready for deployment')
        
    class WorkflowEvaluatorOutput(BaseModel):
        success: bool = Field(description='Whether evaluation succeeded')
        operation: str = Field(description='Operation performed')
        message: str = Field(description='Status message')
        data: Dict[str, Any] = Field(description='Validation results and final code')
        state_ref: str = Field(description='Reference to stored state')


class CrafterInput(BaseModel):
    """Input schema for crafter phase - V1 compatible."""
    workflow_id: str = Field(description='Workflow identifier')
    model: str = Field(default='openai:gpt-4o', description='LLM model to use')


class CrafterOutput(BaseModel):
    """Output schema for crafter phase - V1 compatible."""
    success: bool = Field(description='Whether code generation succeeded')
    message: str = Field(description='Status message')
    data: Dict[str, Any] = Field(description='Generated code and metadata')
    total_tools: int = Field(description='Number of tools implemented')
    implementations: List[Dict[str, Any]] = Field(description='List of implementations with metadata')


# Use V1 evaluator models for exact compatibility
class EvaluatorInput(BaseModel):
    """Input schema for evaluator phase - V1 compatible."""
    workflow_id: str = Field(description='Workflow identifier to retrieve code from')
    model: str = Field(default='openai:gpt-4o', description='LLM model to use for evaluation')
    auto_fix: bool = Field(default=True, description='Whether to automatically fix issues found')


# Use V1 WorkflowEvaluatorOutput if available, otherwise use fallback
if 'WorkflowEvaluatorOutput' in globals():
    EvaluatorOutput = WorkflowEvaluatorOutput
else:
    class EvaluatorOutput(BaseModel):
        """Output schema for evaluator phase - V1 compatible fallback."""
        success: bool = Field(description='Whether evaluation succeeded')
        operation: str = Field(description='Operation performed')
        message: str = Field(description='Status message')
        data: Dict[str, Any] = Field(description='Validation results and final code')
        state_ref: str = Field(description='Reference to stored state')


# Phase Definitions

ANALYZER_PHASE = PhaseDefinition(
    phase_name='analyzer',
    atomic_nodes=[
        'dependency_check',
        'load_dependencies',  # Loads catalog from agentool_mgmt
        'save_catalog',       # V1: Store catalog at workflow/{workflow_id}/catalog  
        'template_render',
        'llm_call',
        'schema_validation',
        'save_analysis',      # V1: Store analysis at workflow/{workflow_id}/analysis
        'save_missing_tools', # V1: Store each missing tool individually
        'state_update',
        'quality_gate'
    ],
    input_schema=AnalyzerInput,
    output_schema=AnalyzerOutput,
    dependencies=[],  # No dependencies for first phase
    templates=TemplateConfig(
        system_template='agentool/system/analyzer.jinja',
        user_template='agentool/prompts/analyze_catalog.jinja'
    ),
    storage_pattern='workflow/{workflow_id}/analysis',  # V1 main storage pattern
    storage_type=StorageType.KV,
    quality_threshold=0.8,
    allow_refinement=True,
    max_refinements=3,
    model_config=ModelParameters(
        temperature=0.7,  # V1 uses 0.7
        max_tokens=2000
    ),
    domain='agentool'
)

# Use V1 models for exact compatibility if available
if V1AnalyzerOutput:
    SpecifierInputSchema = V1AnalyzerOutput  # Specifier uses analyzer output as input
    try:
        from agents.models import SpecificationOutput
        SpecifierOutputSchema = SpecificationOutput
    except ImportError:
        SpecifierOutputSchema = SpecifierOutput
else:
    SpecifierInputSchema = SpecifierInput
    SpecifierOutputSchema = SpecifierOutput

SPECIFIER_PHASE = PhaseDefinition(
    phase_name='specifier',
    atomic_nodes=[
        'dependency_check',           # Check analyzer dependency is satisfied
        'load_dependencies',          # Load analyzer output from storage 
        'prepare_specifier_iteration', # V1-compatible: Load missing tools, collect existing tools
        'specifier_tool_iterator',    # V1-compatible: Iterate over missing tools with LLM
        'save_output',               # Store final SpecificationOutput
        'state_update',              # Update workflow state
        'quality_gate'               # Check specification quality
    ],
    input_schema=SpecifierInputSchema,  # V1: Uses analyzer output directly
    output_schema=SpecifierOutputSchema,  # V1: SpecificationOutput with tool list
    dependencies=['analyzer'],  # Requires analyzer phase to complete first
    templates=TemplateConfig(
        system_template='system/specification.jinja',  # V1 EXACT path - no agentool prefix
        user_template='prompts/create_specification.jinja'  # V1 EXACT path - no agentool prefix
    ),
    storage_pattern='workflow/{workflow_id}/specs',  # V1 storage pattern
    storage_type=StorageType.KV,
    quality_threshold=0.85,
    allow_refinement=True,
    max_refinements=3,
    model_config=ModelParameters(
        temperature=0.5,  # V1 uses 0.5 for consistent output
        max_tokens=3000   # V1 needs more tokens for detailed specifications  
    ),
    domain='agentool'
)

CRAFTER_PHASE = PhaseDefinition(
    phase_name='crafter',
    atomic_nodes=[
        'dependency_check',           # Check specifier dependency
        'load_dependencies',          # Load analysis and specifications from storage
        'save_catalog',               # Store catalog if needed for templates
        'prepare_crafter_iteration',  # V1-compatible: Set up iteration over specs
        'crafter_tool_iterator',      # V1-compatible: Generate code for each tool (CORE ITERATION)
        'save_implementation_summary', # V1: Store summary with all implementations
        'state_update',              # Update workflow state
        'quality_gate'               # Check implementation quality
    ],
    input_schema=CrafterInput,       # V1 compatible with workflow_id and model
    output_schema=CrafterOutput,     # V1 compatible output structure
    dependencies=['specifier'],     # Requires specifications from specifier phase
    templates=TemplateConfig(
        system_template='system/crafter.jinja',  # V1 EXACT path
        user_template='prompts/craft_implementation.jinja'  # V1 EXACT path
    ),
    storage_pattern='workflow/{workflow_id}/implementations',  # V1 storage pattern
    storage_type=StorageType.KV,
    quality_threshold=0.9,
    allow_refinement=True,
    max_refinements=5,
    model_config=ModelParameters(
        temperature=0.3,  # V1 uses 0.3 for code generation
        max_tokens=4000  # V1 uses 4000 tokens for code
    ),
    domain='agentool',
    # V1 Additional Storage Patterns
    additional_storage_patterns={
        'implementations_summary': 'workflow/{workflow_id}/implementations_summary',
        'individual_implementation': 'workflow/{workflow_id}/implementations/{tool_name}', 
        'file_system': 'generated/{workflow_id}/src/{file_path}',
        'skeleton': 'workflow/{workflow_id}/skeleton/{tool_name}',
        'current_missing_tool': 'workflow/{workflow_id}/current_missing_tool_for_craft/{tool_name}'
    }
)

EVALUATOR_PHASE = PhaseDefinition(
    phase_name='evaluator',
    atomic_nodes=[
        'dependency_check',                # Check crafter dependency
        'load_dependencies',              # Load specifications from crafter phase
        'prepare_evaluator_iteration',    # V1-compatible: Load specifications and prepare iteration
        'evaluator_tool_iterator',        # V1-compatible: Iterate over each implementation (CORE ITERATION)
        'save_validation_summary',        # V1: Store validation summary with all tools
        'save_summary_markdown',          # V1: Create SUMMARY.md file
        'state_update',                  # Update workflow state
        'quality_gate'                   # Check overall evaluation quality
    ],
    input_schema=EvaluatorInput,         # V1 compatible with workflow_id, model, auto_fix
    output_schema=EvaluatorOutput,       # V1 compatible WorkflowEvaluatorOutput
    dependencies=['crafter'],            # Requires implementations from crafter phase
    templates=TemplateConfig(
        system_template='system/evaluator.jinja',  # V1 EXACT path - no agentool prefix
        user_template='prompts/evaluate_code.jinja'  # V1 EXACT path - no agentool prefix
    ),
    storage_pattern='workflow/{workflow_id}/validations_summary',  # V1 main storage pattern
    storage_type=StorageType.KV,
    quality_threshold=0.9,
    allow_refinement=True,
    max_refinements=3,
    model_config=ModelParameters(
        temperature=0.2,  # V1 uses 0.2 for evaluation
        max_tokens=4000   # V1 uses 4000 tokens for validation
    ),
    domain='agentool',
    # V1 Additional Storage Patterns for evaluator
    additional_storage_patterns={
        'individual_validation': 'workflow/{workflow_id}/validations/{tool_name}',
        'final_code_fs': 'generated/{workflow_id}/final/{file_name}', 
        'summary_markdown': 'generated/{workflow_id}/SUMMARY.md',
        'current_implementation_code': 'workflow/{workflow_id}/current_implementation_code',
        'reference_implementation': 'workflow/{workflow_id}/reference_implementation'
    }
)


def register_agentool_domain():
    """Register all AgenTool domain phases with the registry."""
    registry = get_registry()
    
    # Register each phase
    registry.register_phase('agentool.analyzer', ANALYZER_PHASE)
    registry.register_phase('agentool.specifier', SPECIFIER_PHASE)
    registry.register_phase('agentool.crafter', CRAFTER_PHASE)
    registry.register_phase('agentool.evaluator', EVALUATOR_PHASE)
    
    # Also register test phases if needed
    # registry.register_phase("agentool.test_analyzer", TEST_ANALYZER_PHASE)
    # registry.register_phase("agentool.test_stubber", TEST_STUBBER_PHASE)
    # registry.register_phase("agentool.test_crafter", TEST_CRAFTER_PHASE)


# Auto-register on import
register_agentool_domain()


async def create_agentool_workflow(
    task_description: str,
    workflow_id: Optional[str] = None,
    enable_refinement: bool = True,
    guidelines: Optional[List[str]] = None
):
    """Create an AgenTool workflow with proper initialization.
    
    Args:
        task_description: Description of the AgenTool to create
        workflow_id: Optional workflow identifier
        enable_refinement: Whether to enable quality-based refinement
        guidelines: Optional development guidelines
        
    Returns:
        Tuple of (WorkflowDefinition, WorkflowState)
    """
    from ..core.types import WorkflowDefinition, WorkflowState
    from ..domains import build_workflow_definition
    import uuid
    
    # Generate workflow ID if not provided
    if not workflow_id:
        workflow_id = f"agentool-{uuid.uuid4().hex[:8]}"
    
    # Build workflow definition
    workflow_def = build_workflow_definition('agentool')
    from dataclasses import replace
    workflow_def = replace(workflow_def, enable_refinement=enable_refinement)
    
    # Fetch catalog from agentool_mgmt if available
    catalog = {}
    try:
        from agentool.core.injector import get_injector
        injector = get_injector()
        
        # Try to get catalog
        import asyncio
        catalog_result = await injector.run('agentool_mgmt', {
            'operation': 'export_catalog',
            'format': 'json'
        })
        if catalog_result.success:
            catalog = catalog_result.data.get('catalog', {})
        else:
            from ..exceptions import CatalogError
            raise CatalogError(f"Failed to export catalog: {catalog_result.message}")
    except ImportError:
        # agentool not available, continue
        pass
    
    # Create initial state with V1-compatible data
    initial_state = WorkflowState(
        workflow_def=workflow_def,
        workflow_id=workflow_id,
        domain='agentool',
        current_phase='analyzer',
        current_node='dependency_check',
        completed_phases=set(),
        phase_outputs={},
        domain_data={
            'task_description': task_description,
            'catalog': catalog,
            'guidelines': guidelines or [],
            'input': AnalyzerInput(
                task_description=task_description,
                domain='agentool',
                catalog=catalog,
                guidelines=guidelines
            ).model_dump()
        },
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    return workflow_def, initial_state