"""Data models for the AI Code Generation Workflow.

These models represent incremental data points collected at each phase,
avoiding duplication and focusing on new information only.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ExistingToolInfo(BaseModel):
    """Information about an existing AgenTool that can be reused."""
    name: str = Field(description="AgenTool name in registry")
    description: str = Field(description="What this tool does")
    operations_schema: Dict[str, Any] = Field(description="Available operations and their schemas")
    input_schema: Dict[str, Any] = Field(description="Pydantic input schema definition")
    output_schema: Dict[str, Any] = Field(description="Pydantic output schema definition")


class MissingToolSpec(BaseModel):
    """Specification for an AgenTool that needs to be created."""
    name: str = Field(description="Name following naming conventions (lowercase_with_underscores)")
    description: str = Field(description="Clear, single-sentence description of what this tool does")
    required_functionality: str = Field(description="Detailed description of all operations and capabilities this tool must provide")
    required_tools: List[str] = Field(default_factory=list, description="Names of existing tools from catalog that this tool will use as dependencies")
    dependencies: List[str] = Field(default_factory=list, description="External Python packages or libraries needed (e.g., 'redis', 'numpy')")


class AnalyzerOutput(BaseModel):
    """Complete analysis output from catalog examination.
    
    This is the primary output from the analyzer phase, containing
    all the insights needed for subsequent phases.
    """
    name: str = Field(description="Descriptive name for the overall solution/capability being built")
    description: str = Field(description="Clear explanation of what this system will accomplish")
    system_design: str = Field(description="High-level architecture and integration approach explaining how tools work together")
    guidelines: List[str] = Field(description="Patterns and best practices extracted from existing tools to follow")
    existing_tools: List[str] = Field(description="Names of existing tools from the catalog to use as dependencies")
    missing_tools: List[MissingToolSpec] = Field(description="Specifications for new tools that need to be created")


class ToolSpecification(BaseModel):
    """Detailed specification for a single AgenTool to be implemented."""
    name: str = Field(description="AgenTool name following naming conventions")
    description: str = Field(description="Clear description of tool purpose")
    input_schema: Dict[str, Any] = Field(description="Pydantic input model as dict")
    output_schema: Dict[str, Any] = Field(description="Pydantic output model as dict")
    examples: List[Dict[str, Any]] = Field(description="Usage examples with input/output")
    errors: List[str] = Field(description="Possible error conditions to handle")
    extended_intents: List[str] = Field(description="Additional use cases covered")
    required_tools: List[str] = Field(description="AgenTool dependencies by name")
    dependencies: List[str] = Field(description="Python package dependencies")
    implementation_guidelines: List[str] = Field(description="Specific coding requirements")


class SpecificationOutput(BaseModel):
    """Collection of all specifications for missing tools.
    
    Output from the specification phase containing detailed
    specs for each tool that needs to be created.
    """
    specifications: List[ToolSpecification] = Field(
        description="Complete specifications for each missing tool"
    )


class CodeOutput(BaseModel):
    """Generated implementation code from the crafter phase."""
    code: str = Field(description="Complete Python implementation following AgenTool patterns")
    file_path: str = Field(description="Suggested file path for saving the implementation")


class ValidationOutput(BaseModel):
    """Validation and test results from the evaluator phase."""
    syntax_valid: bool = Field(description="Whether Python syntax is correct")
    imports_valid: bool = Field(description="Whether all imports are available")
    tests_passed: bool = Field(description="Whether all tests pass")
    issues: List[str] = Field(description="Issues found during validation")
    fixes_applied: List[str] = Field(description="Corrections made to the code")
    improvements: List[str] = Field(description="Enhancements applied")
    final_code: str = Field(description="Final validated and improved implementation")
    ready_for_deployment: bool = Field(description="Whether code is production-ready")


class WorkflowMetadata(BaseModel):
    """Metadata tracking for the entire workflow execution."""
    
    workflow_id: str = Field(description="Unique identifier for this workflow run")
    started_at: str = Field(description="ISO timestamp when workflow started")
    completed_at: Optional[str] = Field(None, description="ISO timestamp when workflow completed")
    total_duration_seconds: Optional[float] = Field(None, description="Total execution time")
    
    # Phase tracking
    phase_durations: Dict[str, float] = Field(
        default_factory=dict,
        description="Duration of each phase in seconds"
    )
    current_phase: str = Field(description="Current phase being executed")
    status: str = Field(description="Overall workflow status (running, completed, failed)")
    
    # Model usage
    models_used: Dict[str, str] = Field(
        default_factory=dict,
        description="LLM models used in each phase"
    )
    
    # Errors
    errors: List[str] = Field(
        default_factory=list,
        description="Any errors encountered during execution"
    )


class TestCaseSpec(BaseModel):
    """Specification for an individual test case."""
    name: str = Field(description="Test method name following test_* convention")
    description: str = Field(description="What this test validates")
    test_type: str = Field(description="Type of test: unit, integration, edge_case, error_handling")
    operation: Optional[str] = Field(None, description="Operation being tested if applicable")
    inputs: Dict[str, Any] = Field(description="Test input data")
    expected_output: Dict[str, Any] = Field(description="Expected output or behavior")
    dependencies_required: List[str] = Field(default_factory=list, description="Real dependencies to create")
    assertions: List[str] = Field(description="Key assertions to validate")


class TestAnalysisOutput(BaseModel):
    """Complete test analysis output from test analyzer phase."""
    tool_name: str = Field(description="Name of the tool being tested")
    test_cases: List[TestCaseSpec] = Field(description="All test cases to implement")
    coverage_requirements: Dict[str, Any] = Field(description="Coverage targets by category")
    dependency_setup: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Real dependencies to create and their initialization order"
    )
    fixtures_needed: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Test fixtures and setup requirements"
    )
    test_data_samples: Dict[str, Any] = Field(
        default_factory=dict,
        description="Sample data for different test scenarios"
    )
    integration_points: List[str] = Field(
        default_factory=list,
        description="External tools/services to test integration with"
    )
    global_state_to_clear: List[str] = Field(
        default_factory=list,
        description="Global state stores that need clearing in setup"
    )


class TestStubOutput(BaseModel):
    """Test stub output from test stubber phase."""
    code: str = Field(description="Complete test file skeleton with placeholders")
    file_path: str = Field(description="Path where test file will be saved")
    placeholders_count: int = Field(description="Number of test placeholders to fill")
    structure_elements: Dict[str, bool] = Field(
        description="Checklist of structural elements included",
        default_factory=lambda: {
            "imports": False,
            "agent_creation": False,
            "setup_teardown": False,
            "test_placeholders": False,
            "real_dependencies": False
        }
    )


class TestImplementationOutput(BaseModel):
    """Final test implementation output from test crafter phase."""
    code: str = Field(description="Complete test implementation")
    file_path: str = Field(description="Path where test file is saved")
    test_count: int = Field(description="Total number of tests implemented")
    coverage_achieved: Dict[str, float] = Field(
        default_factory=dict,
        description="Coverage percentages by category"
    )
    runnable: bool = Field(description="Whether tests are ready to execute")
    dependencies_created: bool = Field(description="Whether all real dependencies are properly created")