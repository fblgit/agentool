"""GraphToolkit TestSuite Domain.

Complete phase definitions for the TestSuite workflow domain.
This domain handles generation and execution of comprehensive test suites
for any codebase or generated code.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..core.registry import get_registry
from ..core.types import ModelParameters, PhaseDefinition, StorageType, TemplateConfig

# Input/Output Schemas for TestSuite phases

class TestAnalyzerInput(BaseModel):
    """Input schema for test analyzer phase."""
    code_to_test: str = Field(description='Code that needs test coverage')
    file_path: Optional[str] = Field(default=None, description='Path to code file')
    existing_tests: Optional[List[str]] = Field(default=None, description='Existing test files')
    coverage_target: float = Field(default=0.85, description='Target code coverage percentage')
    test_types: List[str] = Field(default=['unit', 'integration'], description='Types of tests to generate')
    framework: str = Field(default='pytest', description='Testing framework to use')

class TestAnalyzerOutput(BaseModel):
    """Output schema for test analyzer phase."""
    functions_to_test: List[Dict[str, Any]] = Field(description='Functions/methods requiring tests')
    test_categories: Dict[str, List[str]] = Field(description='Test categories and their functions')
    complexity_analysis: Dict[str, Any] = Field(description='Code complexity metrics')
    coverage_gaps: List[str] = Field(description='Areas lacking test coverage')
    recommended_fixtures: List[str] = Field(description='Test fixtures to create')
    success: bool = Field(default=True, description='Whether analysis succeeded')

class TestDesignerInput(BaseModel):
    """Input schema for test designer phase."""
    functions_to_test: List[Dict[str, Any]] = Field(description='Functions requiring tests')
    test_categories: Dict[str, List[str]] = Field(description='Categorized test requirements')
    framework: str = Field(default='pytest', description='Testing framework')
    coverage_target: float = Field(default=0.85, description='Target coverage')

class TestDesignerOutput(BaseModel):
    """Output schema for test designer phase."""
    test_designs: List[Dict[str, Any]] = Field(description='Detailed test case designs')
    fixture_designs: List[Dict[str, Any]] = Field(description='Test fixture designs')
    mock_requirements: List[str] = Field(description='External dependencies to mock')
    test_data_requirements: Dict[str, Any] = Field(description='Test data needed')
    execution_order: List[str] = Field(description='Test execution sequence')
    success: bool = Field(default=True, description='Whether design succeeded')

class TestGeneratorInput(BaseModel):
    """Input schema for test generator phase."""
    test_designs: List[Dict[str, Any]] = Field(description='Test case designs to implement')
    fixture_designs: List[Dict[str, Any]] = Field(description='Fixture designs')
    framework: str = Field(default='pytest', description='Testing framework')
    code_context: Optional[str] = Field(default=None, description='Original code for context')

class TestGeneratorOutput(BaseModel):
    """Output schema for test generator phase."""
    test_files: Dict[str, str] = Field(description='Generated test files (filename -> content)')
    fixture_code: Optional[str] = Field(default=None, description='Generated fixture code')
    requirements: List[str] = Field(description='Test dependencies/requirements')
    execution_commands: List[str] = Field(description='Commands to run tests')
    estimated_coverage: float = Field(description='Estimated coverage percentage')
    success: bool = Field(default=True, description='Whether generation succeeded')

class TestExecutorInput(BaseModel):
    """Input schema for test executor phase."""
    test_files: Dict[str, str] = Field(description='Test files to execute')
    code_under_test: Optional[str] = Field(default=None, description='Code being tested')
    framework: str = Field(default='pytest', description='Testing framework')
    run_integration: bool = Field(default=True, description='Whether to run integration tests')
    collect_coverage: bool = Field(default=True, description='Whether to collect coverage metrics')

class TestExecutorOutput(BaseModel):
    """Output schema for test executor phase."""
    test_results: Dict[str, Any] = Field(description='Test execution results')
    coverage_report: Optional[Dict[str, Any]] = Field(default=None, description='Code coverage report')
    passed_tests: int = Field(description='Number of passed tests')
    failed_tests: int = Field(description='Number of failed tests')
    execution_time: float = Field(description='Total execution time in seconds')
    issues_found: List[str] = Field(description='Issues discovered during testing')
    success: bool = Field(default=True, description='Whether execution succeeded')

# Phase Definitions

TEST_ANALYZER_PHASE = PhaseDefinition(
    phase_name='test_analyzer',
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
    input_schema=TestAnalyzerInput,
    output_schema=TestAnalyzerOutput,
    dependencies=[],  # No dependencies for first phase
    templates=TemplateConfig(
        system_template='system/test_analyzer',
        user_template='prompts/testsuite/analyze_code',
        variables={
            'framework': 'pytest',
            'purpose': 'Analyze code for comprehensive test generation'
        }
    ),
    storage_pattern='workflow/{workflow_id}/test_analyzer',
    storage_type=StorageType.KV,
    quality_threshold=0.8,
    allow_refinement=True,
    max_refinements=3,
    model_config=ModelParameters(
        temperature=0.7,
        max_tokens=2500
    ),
    domain='testsuite'
)

TEST_DESIGNER_PHASE = PhaseDefinition(
    phase_name='test_designer',
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
    input_schema=TestDesignerInput,
    output_schema=TestDesignerOutput,
    dependencies=['test_analyzer'],
    templates=TemplateConfig(
        system_template='system/test_designer',
        user_template='prompts/testsuite/design_tests',
        variables={
            'framework': 'pytest',
            'purpose': 'Design comprehensive test cases and fixtures'
        }
    ),
    storage_pattern='workflow/{workflow_id}/test_designer',
    storage_type=StorageType.KV,
    quality_threshold=0.85,
    allow_refinement=True,
    max_refinements=3,
    model_config=ModelParameters(
        temperature=0.6,
        max_tokens=3500
    ),
    domain='testsuite'
)

TEST_GENERATOR_PHASE = PhaseDefinition(
    phase_name='test_generator',
    atomic_nodes=[
        'dependency_check',
        'load_dependencies',
        'template_render',
        'llm_call',
        'schema_validation',
        'syntax_validation',  # Validate generated test code
        'save_output',
        'state_update',
        'quality_gate'
    ],
    input_schema=TestGeneratorInput,
    output_schema=TestGeneratorOutput,
    dependencies=['test_designer'],
    templates=TemplateConfig(
        system_template='system/test_generator',
        user_template='prompts/testsuite/generate_tests',
        variables={
            'framework': 'pytest',
            'purpose': 'Generate complete test implementations'
        }
    ),
    storage_pattern='workflow/{workflow_id}/test_generator',
    storage_type=StorageType.KV,
    quality_threshold=0.9,
    allow_refinement=True,
    max_refinements=5,
    model_config=ModelParameters(
        temperature=0.4,
        max_tokens=4500
    ),
    domain='testsuite'
)

TEST_EXECUTOR_PHASE = PhaseDefinition(
    phase_name='test_executor',
    atomic_nodes=[
        'dependency_check',
        'load_dependencies',
        'test_execution',  # Custom node for running tests
        'coverage_analysis',  # Custom node for coverage analysis
        'schema_validation',
        'save_output',
        'state_update',
        'quality_gate'
    ],
    input_schema=TestExecutorInput,
    output_schema=TestExecutorOutput,
    dependencies=['test_generator'],
    templates=TemplateConfig(
        system_template='system/test_executor',
        user_template='prompts/testsuite/execute_tests',
        variables={
            'framework': 'pytest',
            'purpose': 'Execute tests and analyze results'
        }
    ),
    storage_pattern='workflow/{workflow_id}/test_executor',
    storage_type=StorageType.KV,
    quality_threshold=0.95,
    allow_refinement=True,
    max_refinements=2,
    model_config=ModelParameters(
        temperature=0.3,
        max_tokens=2000
    ),
    domain='testsuite'
)

def register_testsuite_domain():
    """Register all TestSuite domain phases with the registry."""
    registry = get_registry()
    
    # Register each phase
    registry.register_phase('testsuite.test_analyzer', TEST_ANALYZER_PHASE)
    registry.register_phase('testsuite.test_designer', TEST_DESIGNER_PHASE)
    registry.register_phase('testsuite.test_generator', TEST_GENERATOR_PHASE)
    registry.register_phase('testsuite.test_executor', TEST_EXECUTOR_PHASE)

# Auto-register on import
register_testsuite_domain()