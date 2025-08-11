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


class CrafterInput(BaseModel):
    """Input schema for crafter phase."""
    specifications: List[Dict[str, Any]] = Field(description='Tool specifications to implement')
    existing_code: Optional[Dict[str, str]] = Field(default=None, description='Existing code base')


class CrafterOutput(BaseModel):
    """Output schema for crafter phase."""
    implementations: Dict[str, str] = Field(description='Tool implementations (name -> code)')
    imports: List[str] = Field(description='Required imports')
    success: bool = Field(default=True, description='Whether crafting succeeded')


class EvaluatorInput(BaseModel):
    """Input schema for evaluator phase."""
    implementations: Dict[str, str] = Field(description='Tool implementations to evaluate')
    specifications: List[Dict[str, Any]] = Field(description='Original specifications')


class EvaluatorOutput(BaseModel):
    """Output schema for evaluator phase."""
    validation_results: Dict[str, Dict[str, Any]] = Field(description='Validation results per tool')
    quality_scores: Dict[str, float] = Field(description='Quality scores per tool')
    final_code: Dict[str, str] = Field(description='Final validated code')
    success: bool = Field(default=True, description='Whether evaluation succeeded')


# Phase Definitions

ANALYZER_PHASE = PhaseDefinition(
    phase_name='analyzer',
    atomic_nodes=[
        'dependency_check',
        'load_dependencies',
        'template_render',
        'llm_call',
        'schema_validation',
        'save_output',
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
    storage_pattern='workflow/{workflow_id}/analyzer',
    storage_type=StorageType.KV,
    quality_threshold=0.8,
    allow_refinement=True,
    max_refinements=3,
    model_config=ModelParameters(
        temperature=0.7,
        max_tokens=2000
    ),
    domain='agentool'
)

SPECIFIER_PHASE = PhaseDefinition(
    phase_name='specifier',
    atomic_nodes=[
        'dependency_check',
        'load_dependencies',
        'prepare_specifier_iteration',  # Prepare missing tools for iteration
        'process_tools',  # Iterate over missing tools
        'save_output',
        'state_update',
        'quality_gate'
    ],
    input_schema=SpecifierInput,
    output_schema=SpecifierOutput,
    dependencies=['analyzer'],
    templates=TemplateConfig(
        system_template='agentool/system/specifier.jinja',
        user_template='agentool/prompts/create_specification.jinja'
    ),
    storage_pattern='workflow/{workflow_id}/specifier',
    storage_type=StorageType.KV,
    quality_threshold=0.85,
    allow_refinement=True,
    max_refinements=3,
    model_config=ModelParameters(
        temperature=0.5,
        max_tokens=3000
    ),
    domain='agentool'
)

CRAFTER_PHASE = PhaseDefinition(
    phase_name='crafter',
    atomic_nodes=[
        'dependency_check',
        'load_dependencies',
        'template_render',
        'llm_call',
        'schema_validation',
        'save_output',
        'state_update',
        'quality_gate'
    ],
    input_schema=CrafterInput,
    output_schema=CrafterOutput,
    dependencies=['specifier'],
    templates=TemplateConfig(
        system_template='agentool/system/crafter.jinja',
        user_template='agentool/prompts/craft_implementation.jinja'
    ),
    storage_pattern='workflow/{workflow_id}/crafter',
    storage_type=StorageType.KV,
    quality_threshold=0.9,
    allow_refinement=True,
    max_refinements=5,
    model_config=ModelParameters(
        temperature=0.3,
        max_tokens=4000
    ),
    domain='agentool'
)

EVALUATOR_PHASE = PhaseDefinition(
    phase_name='evaluator',
    atomic_nodes=[
        'dependency_check',
        'load_dependencies',
        'template_render',
        'llm_call',
        'schema_validation',
        'save_output',
        'state_update',
        'quality_gate'
    ],
    input_schema=EvaluatorInput,
    output_schema=EvaluatorOutput,
    dependencies=['crafter'],
    templates=TemplateConfig(
        system_template='agentool/system/evaluator.jinja',
        user_template='agentool/prompts/evaluate_code.jinja'
    ),
    storage_pattern='workflow/{workflow_id}/evaluator',
    storage_type=StorageType.KV,
    quality_threshold=0.95,
    allow_refinement=True,
    max_refinements=3,
    model_config=ModelParameters(
        temperature=0.2,
        max_tokens=2000
    ),
    domain='agentool'
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
        try:
            catalog_result = await injector.run('agentool_mgmt', {
                'operation': 'export_catalog',
                'format': 'json'
            })
            if catalog_result.success:
                catalog = catalog_result.data.get('catalog', {})
        except:
            # If agentool_mgmt not available, continue without catalog
            pass
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