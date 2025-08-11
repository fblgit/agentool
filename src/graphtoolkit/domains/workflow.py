"""GraphToolkit Workflow Domain Definition.

Domain for designing and orchestrating workflow systems.
Phases: Process Analyzer → Step Designer → Orchestrator → Tester
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..core.registry import register_phase
from ..core.types import ModelParameters, PhaseDefinition, StorageType, TemplateConfig

# Input/Output Schemas for Workflow Domain

class WorkflowAnalyzerInput(BaseModel):
    """Input for workflow analysis phase."""
    process_description: str = Field(description='Description of the process to analyze')
    domain_context: str = Field(description='Business domain context')
    requirements: List[str] = Field(default_factory=list, description='Process requirements')
    constraints: Dict[str, Any] = Field(default_factory=dict, description='Process constraints')
    existing_workflows: Optional[List[Dict[str, Any]]] = Field(None, description='Existing workflow definitions')


class WorkflowAnalyzerOutput(BaseModel):
    """Output from workflow analysis phase."""
    process_steps: List[Dict[str, Any]] = Field(description='Identified process steps')
    dependencies: Dict[str, List[str]] = Field(description='Step dependencies')
    data_flow: Dict[str, Any] = Field(description='Data flow between steps')
    decision_points: List[Dict[str, Any]] = Field(description='Decision points in workflow')
    parallel_opportunities: List[str] = Field(description='Steps that can run in parallel')
    recommendations: List[str] = Field(description='Process improvement recommendations')


class StepDesignerInput(BaseModel):
    """Input for step design phase."""
    process_steps: List[Dict[str, Any]] = Field(description='Process steps to design')
    dependencies: Dict[str, List[str]] = Field(description='Step dependencies')
    data_flow: Dict[str, Any] = Field(description='Data flow requirements')
    design_patterns: Optional[List[str]] = Field(None, description='Preferred design patterns')


class StepDesignerOutput(BaseModel):
    """Output from step design phase."""
    step_definitions: List[Dict[str, Any]] = Field(description='Detailed step definitions')
    step_interfaces: Dict[str, Dict[str, Any]] = Field(description='Input/output interfaces for each step')
    error_handling: Dict[str, List[str]] = Field(description='Error handling strategies')
    retry_policies: Dict[str, Dict[str, Any]] = Field(description='Retry policies for each step')
    monitoring_points: List[Dict[str, Any]] = Field(description='Monitoring and logging points')


class OrchestratorInput(BaseModel):
    """Input for orchestration phase."""
    step_definitions: List[Dict[str, Any]] = Field(description='Step definitions to orchestrate')
    dependencies: Dict[str, List[str]] = Field(description='Step dependencies')
    step_interfaces: Dict[str, Dict[str, Any]] = Field(description='Step interfaces')
    execution_strategy: str = Field(default='sequential', description='Execution strategy')


class OrchestratorOutput(BaseModel):
    """Output from orchestration phase."""
    workflow_definition: Dict[str, Any] = Field(description='Complete workflow definition')
    execution_graph: Dict[str, Any] = Field(description='DAG of execution')
    state_machine: Dict[str, Any] = Field(description='State machine definition')
    compensation_logic: Dict[str, Any] = Field(description='Compensation/rollback logic')
    configuration: Dict[str, Any] = Field(description='Workflow configuration')


class TesterInput(BaseModel):
    """Input for testing phase."""
    workflow_definition: Dict[str, Any] = Field(description='Workflow to test')
    test_scenarios: List[Dict[str, Any]] = Field(description='Test scenarios')
    test_data: Dict[str, Any] = Field(description='Test data sets')
    validation_criteria: List[str] = Field(description='Validation criteria')


class TesterOutput(BaseModel):
    """Output from testing phase."""
    test_results: List[Dict[str, Any]] = Field(description='Test execution results')
    coverage_report: Dict[str, float] = Field(description='Test coverage metrics')
    performance_metrics: Dict[str, Any] = Field(description='Performance test results')
    edge_cases: List[Dict[str, Any]] = Field(description='Edge cases identified')
    recommendations: List[str] = Field(description='Testing recommendations')
    quality_score: float = Field(description='Overall quality score')


# Phase Definitions for Workflow Domain

analyzer_phase = PhaseDefinition(
    phase_name='analyzer',
    atomic_nodes=[
        'dependency_check',
        'load_dependencies', 
        'template_render',
        'llm_call',
        'schema_validation',
        'save_phase_output',
        'state_update',
        'quality_gate'
    ],
    input_schema=WorkflowAnalyzerInput,
    output_schema=WorkflowAnalyzerOutput,
    dependencies=[],
    templates=TemplateConfig(
        system_template='templates/system/workflow_analyzer.jinja',
        user_template='templates/prompts/analyze_process.jinja',
        variables={
            'analysis_depth': 'comprehensive',
            'focus_areas': 'steps,dependencies,parallelism'
        }
    ),
    storage_pattern='workflow/{workflow_id}/analysis',
    storage_type=StorageType.KV,
    quality_threshold=0.8,
    allow_refinement=True,
    max_refinements=2,
    model_config=ModelParameters(
        temperature=0.7,
        max_tokens=2000
    ),
    domain='workflow'
)

step_designer_phase = PhaseDefinition(
    phase_name='step_designer',
    atomic_nodes=[
        'dependency_check',
        'load_dependencies',
        'template_render',
        'llm_call',
        'schema_validation',
        'save_phase_output',
        'state_update',
        'quality_gate'
    ],
    input_schema=StepDesignerInput,
    output_schema=StepDesignerOutput,
    dependencies=['analyzer'],
    templates=TemplateConfig(
        system_template='templates/system/step_designer.jinja',
        user_template='templates/prompts/design_steps.jinja',
        variables={
            'design_style': 'detailed',
            'include_error_handling': 'true'
        }
    ),
    storage_pattern='workflow/{workflow_id}/step_design',
    storage_type=StorageType.KV,
    quality_threshold=0.85,
    allow_refinement=True,
    max_refinements=3,
    model_config=ModelParameters(
        temperature=0.6,
        max_tokens=3000
    ),
    domain='workflow'
)

orchestrator_phase = PhaseDefinition(
    phase_name='orchestrator',
    atomic_nodes=[
        'dependency_check',
        'load_dependencies',
        'template_render',
        'llm_call',
        'schema_validation',
        'save_phase_output',
        'state_update',
        'quality_gate'
    ],
    input_schema=OrchestratorInput,
    output_schema=OrchestratorOutput,
    dependencies=['analyzer', 'step_designer'],
    templates=TemplateConfig(
        system_template='templates/system/workflow_orchestrator.jinja',
        user_template='templates/prompts/orchestrate_workflow.jinja',
        variables={
            'orchestration_style': 'dag',
            'include_compensation': 'true'
        }
    ),
    storage_pattern='workflow/{workflow_id}/orchestration',
    storage_type=StorageType.KV,
    quality_threshold=0.9,
    allow_refinement=True,
    max_refinements=2,
    model_config=ModelParameters(
        temperature=0.5,
        max_tokens=4000
    ),
    domain='workflow'
)

tester_phase = PhaseDefinition(
    phase_name='tester',
    atomic_nodes=[
        'dependency_check',
        'load_dependencies',
        'template_render',
        'llm_call',
        'schema_validation',
        'save_phase_output',
        'state_update',
        'quality_gate'
    ],
    input_schema=TesterInput,
    output_schema=TesterOutput,
    dependencies=['analyzer', 'step_designer', 'orchestrator'],
    templates=TemplateConfig(
        system_template='templates/system/workflow_tester.jinja',
        user_template='templates/prompts/test_workflow.jinja',
        variables={
            'test_depth': 'thorough',
            'include_edge_cases': 'true'
        }
    ),
    storage_pattern='workflow/{workflow_id}/testing',
    storage_type=StorageType.KV,
    quality_threshold=0.85,
    allow_refinement=True,
    max_refinements=2,
    model_config=ModelParameters(
        temperature=0.6,
        max_tokens=2500
    ),
    domain='workflow'
)


def register_workflow_phases():
    """Register all workflow domain phases."""
    register_phase('workflow.analyzer', analyzer_phase)
    register_phase('workflow.step_designer', step_designer_phase)
    register_phase('workflow.orchestrator', orchestrator_phase)
    register_phase('workflow.tester', tester_phase)


# Auto-register on import
register_workflow_phases()