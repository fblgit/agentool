"""Data models for the AI Code Generation Workflow.

These models represent incremental data points collected at each phase,
avoiding duplication and focusing on new information only.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, ValidationError


class ExistingToolInfo(BaseModel):
    """Information about an existing AgenTool that can be reused.
    
    Represents a complete registry record for an existing tool that can be
    used as a dependency or pattern reference.
    """
    name: str = Field(
        description="Exact registered name of the AgenTool (e.g., 'storage_kv', 'auth', 'metrics'). Must match registry key."
    )
    description: str = Field(
        description="One-sentence description of the tool's purpose and capabilities (e.g., 'Provides secure key-value storage with TTL support')"
    )
    operations_schema: Dict[str, Any] = Field(
        description="""Map of operation names to their parameter schemas. Example:
        {
            "get": {"params": ["key", "default"], "returns": "Any"},
            "set": {"params": ["key", "value", "ttl"], "returns": "bool"},
            "delete": {"params": ["key"], "returns": "bool"}
        }"""
    )
    input_schema: Dict[str, Any] = Field(
        description="""Complete JSON Schema for the tool's input model, including operation enum and all parameters.
        Must follow the pattern used by the tool for proper integration."""
    )
    output_schema: Dict[str, Any] = Field(
        description="""Complete JSON Schema for the tool's output model, typically with success, message, and data fields.
        Used to understand the tool's response format."""
    )


class MissingToolSpec(BaseModel):
    """Specification for an AgenTool that needs to be created.
    
    High-level requirements for a new tool identified during analysis phase.
    """
    name: str = Field(
        description="""Proposed name following lowercase_underscore convention (e.g., 'cache_manager', 'rate_limiter').
        Should be descriptive and consistent with existing naming patterns."""
    )
    description: str = Field(
        description="""Clear one-sentence summary of the tool's purpose (e.g., 'Manages distributed caching with invalidation support').
        Focus on the primary capability without implementation details."""
    )
    required_functionality: str = Field(
        description="""Comprehensive description of all required operations and capabilities. Include:
        - Core operations needed (e.g., 'Must support get, set, delete with TTL')
        - Data handling requirements (e.g., 'Handle JSON-serializable objects up to 1MB')
        - Performance requirements (e.g., 'Sub-millisecond response for cache hits')
        - Integration needs (e.g., 'Must work with existing auth system')
        Example: 'Provide rate limiting with configurable windows (fixed, sliding), multiple strategies (token bucket, leaky bucket),
        per-user and per-endpoint limits, redis backend support, and metric tracking.'"""
    )
    required_tools: List[str] = Field(
        default_factory=list,
        description="""Exact names of existing AgenTools this will depend on (e.g., ['storage_kv', 'metrics', 'logging']).
        These must be tools identified in the catalog analysis phase."""
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="""External Python packages needed with optional version constraints (e.g., ['redis>=4.0.0', 'python-dateutil', 'pyjwt==2.8.0']).
        Only include if specific external libraries are required beyond standard library."""
    )


class AnalyzerOutput(BaseModel):
    """Complete analysis output from catalog examination.
    
    This is the primary output from the analyzer phase, containing
    all the insights needed for subsequent phases.
    """
    name: str = Field(
        description="""Descriptive name for the solution being built (e.g., 'Distributed Task Processing System', 'Multi-tenant API Gateway').
        Should clearly indicate the primary capability being delivered."""
    )
    description: str = Field(
        description="""2-3 sentence explanation of what this system accomplishes and its key benefits.
        Example: 'Provides reliable distributed task processing with automatic retries and dead letter queues.
        Integrates with existing monitoring and ensures exactly-once processing guarantees.'"""
    )
    system_design: str = Field(
        description="""Detailed architecture explanation including:
        - Component interactions and data flow
        - Integration points between tools
        - Error handling and resilience patterns
        - Performance considerations
        Example: 'The system uses a producer-consumer pattern with storage_kv for task queuing, auth for access control,
        metrics for monitoring throughput, and a new task_processor tool for execution. Tasks flow from API endpoints
        through auth validation, get queued in storage_kv with TTL, and are processed by workers that report metrics.'"""
    )
    guidelines: List[str] = Field(
        description="""Specific patterns and practices to follow based on existing tools. Examples:
        - 'Use async/await consistently as seen in storage_kv and http tools'
        - 'Follow the structured output pattern with success/message/data fields'
        - 'Implement operation-based routing with Literal types'
        - 'Include comprehensive error handling with specific exception types'
        - 'Add metrics tracking for all operations'"""
    )
    existing_tools: List[str] = Field(
        description="""Exact names of existing tools to reuse (e.g., ['storage_kv', 'auth', 'metrics', 'logging']).
        Must be actual registered tool names from the catalog."""
    )
    missing_tools: List[MissingToolSpec] = Field(
        description="""Complete specifications for each new tool that needs creation.
        Order by dependency (tools with no dependencies first)."""
    )


class ToolSpecification(BaseModel):
    """Detailed specification for a single AgenTool to be implemented.
    
    This specification must provide complete information for implementing
    a fully functional AgenTool including schemas, examples, and dependencies.
    """
    name: str = Field(
        description="AgenTool name following lowercase_underscore convention (e.g., 'storage_kv', 'auth_manager')"
    )
    description: str = Field(
        description="Clear one-sentence description of what this tool does (e.g., 'Manages key-value storage with TTL support')"
    )
    input_schema: Dict[str, Any] = Field(
        description="""JSON Schema for the Pydantic input model. Must include:
        - type: 'object'
        - properties: Dict with 'operation' field (enum/Literal) plus other fields
        - required: List of required field names
        - Field descriptions for each property
        Example: {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["get", "set", "delete"], "description": "Operation to perform"},
                "key": {"type": "string", "description": "Storage key"},
                "value": {"type": ["string", "object", "null"], "description": "Value to store (for set operation)"}
            },
            "required": ["operation", "key"]
        }"""
    )
    output_schema: Dict[str, Any] = Field(
        description="""JSON Schema for the Pydantic output model. Must include:
        - type: 'object'
        - properties: Dict with at least 'success', 'message', and 'data' fields
        - required: List of required field names (usually ['success', 'message'])
        Example: {
            "type": "object",
            "properties": {
                "success": {"type": "boolean", "description": "Whether operation succeeded"},
                "message": {"type": "string", "description": "Human-readable result message"},
                "data": {"type": ["object", "null"], "description": "Operation-specific return data"}
            },
            "required": ["success", "message"]
        }"""
    )
    examples: List[Dict[str, Any]] = Field(
        description="""List of input/output examples covering all operations. Each example must have:
        - description: What this example demonstrates
        - input: Complete input JSON matching input_schema
        - output: Expected output JSON matching output_schema
        Example: [
            {
                "description": "Get a value from storage",
                "input": {"operation": "get", "key": "user:123"},
                "output": {"success": true, "message": "Retrieved value", "data": {"name": "Alice"}}
            },
            {
                "description": "Handle missing key",
                "input": {"operation": "get", "key": "nonexistent"},
                "output": {"success": false, "message": "Key not found", "data": null}
            }
        ]"""
    )
    errors: List[str] = Field(
        description="List of specific error conditions to handle (e.g., 'KeyError when key not found', 'ValueError for invalid TTL', 'ConnectionError for storage backend')"
    )
    extended_intents: List[str] = Field(
        description="Additional capabilities beyond basic operations (e.g., 'Support batch operations', 'Handle TTL expiration', 'Provide namespace isolation')"
    )
    required_tools: List[str] = Field(
        description="Names of existing AgenTools this tool depends on (e.g., ['storage_kv', 'logging', 'metrics']). Must be exact registered names.",
        default_factory=list
    )
    dependencies: List[str] = Field(
        description="External Python packages needed (e.g., ['redis==4.5.0', 'numpy>=1.20.0']). Use exact versions or ranges.",
        default_factory=list
    )
    implementation_guidelines: List[str] = Field(
        description="Specific implementation requirements (e.g., 'Use async/await for all I/O operations', 'Follow existing error handling patterns', 'Include proper logging at INFO level')"
    )


class ToolSpecificationLLM(BaseModel):
    """LLM-compatible version of ToolSpecification with renamed schema fields.
    
    This version is used when interacting with OpenAI's structured output feature
    which reserves 'input_schema' and 'output_schema' field names for internal use.
    The fields are renamed to 'tool_input_schema' and 'tool_output_schema' to avoid conflicts.
    """
    name: str = Field(
        description="AgenTool name following lowercase_underscore convention (e.g., 'storage_kv', 'auth_manager')"
    )
    description: str = Field(
        description="Clear one-sentence description of what this tool does (e.g., 'Manages key-value storage with TTL support')"
    )
    tool_input_schema: Dict[str, Any] = Field(
        description="""JSON Schema for the Pydantic input model. Must include:
        - type: 'object'
        - properties: Dict with 'operation' field (enum/Literal) plus other fields
        - required: List of required field names
        - Field descriptions for each property
        Example: {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["get", "set", "delete"], "description": "Operation to perform"},
                "key": {"type": "string", "description": "Storage key"},
                "value": {"type": ["string", "object", "null"], "description": "Value to store (for set operation)"}
            },
            "required": ["operation", "key"]
        }"""
    )
    tool_output_schema: Dict[str, Any] = Field(
        description="""JSON Schema for the Pydantic output model. Must include:
        - type: 'object'
        - properties: Dict with at least 'success', 'message', and 'data' fields
        - required: List of required field names (usually ['success', 'message'])
        Example: {
            "type": "object",
            "properties": {
                "success": {"type": "boolean", "description": "Whether operation succeeded"},
                "message": {"type": "string", "description": "Human-readable result message"},
                "data": {"type": ["object", "null"], "description": "Operation-specific return data"}
            },
            "required": ["success", "message"]
        }"""
    )
    examples: List[Dict[str, Any]] = Field(
        description="""List of input/output examples covering all operations. Each example must have:
        - description: What this example demonstrates
        - input: Complete input JSON matching tool_input_schema
        - output: Expected output JSON matching tool_output_schema
        Example: [
            {
                "description": "Get a value from storage",
                "input": {"operation": "get", "key": "user:123"},
                "output": {"success": true, "message": "Retrieved value", "data": {"name": "Alice"}}
            },
            {
                "description": "Handle missing key",
                "input": {"operation": "get", "key": "nonexistent"},
                "output": {"success": false, "message": "Key not found", "data": null}
            }
        ]"""
    )
    errors: List[str] = Field(
        description="List of specific error conditions to handle (e.g., 'KeyError when key not found', 'ValueError for invalid TTL', 'ConnectionError for storage backend')"
    )
    extended_intents: List[str] = Field(
        description="Additional capabilities beyond basic operations (e.g., 'Support batch operations', 'Handle TTL expiration', 'Provide namespace isolation')"
    )
    required_tools: List[str] = Field(
        description="Names of existing AgenTools this tool depends on (e.g., ['storage_kv', 'logging', 'metrics']). Must be exact registered names.",
        default_factory=list
    )
    dependencies: List[str] = Field(
        description="External Python packages needed (e.g., ['redis==4.5.0', 'numpy>=1.20.0']). Use exact versions or ranges.",
        default_factory=list
    )
    implementation_guidelines: List[str] = Field(
        description="Specific implementation requirements (e.g., 'Use async/await for all I/O operations', 'Follow existing error handling patterns', 'Include proper logging at INFO level')"
    )
    
    def to_tool_specification(self) -> ToolSpecification:
        """Convert to internal ToolSpecification format with original field names."""
        data = self.model_dump()
        # Rename fields back to original names
        data['input_schema'] = data.pop('tool_input_schema')
        data['output_schema'] = data.pop('tool_output_schema')
        return ToolSpecification(**data)


class SpecificationOutput(BaseModel):
    """Collection of all specifications for missing tools.
    
    Output from the specification phase containing detailed
    specs for each tool that needs to be created.
    """
    specifications: List[ToolSpecification] = Field(
        description="Complete specifications for each missing tool"
    )


class CodeOutput(BaseModel):
    """Generated implementation code from the crafter phase.
    
    Contains production-ready code that follows all AgenTool patterns and conventions.
    """
    code: str = Field(
        description="""Complete, production-ready Python implementation including:
        - All imports (from agentool, dependencies, standard library)
        - Pydantic models for input/output schemas
        - Tool functions with proper async/await and error handling
        - Routing configuration mapping operations to functions
        - Agent creation with create_agentool()
        - Module-level agent instance
        Must be syntactically correct and follow established patterns."""
    )
    file_path: str = Field(
        description="""Suggested file path following project structure (e.g., 'src/agentoolkit/network/rate_limiter.py').
        Should match the appropriate category folder and use snake_case filename."""
    )
    
    @field_validator('code')
    @classmethod
    def validate_code(cls, v: str) -> str:
        """Validate that generated code is complete and follows AgenTool patterns."""
        if not v or not v.strip():
            raise ValueError("code cannot be empty. The crafter must generate a complete AgenTool implementation.")
        
        # Check minimum length for a basic AgenTool
        if len(v.strip()) < 500:
            raise ValueError(f"code is too short ({len(v)} chars). Expected a complete AgenTool with imports, schemas, functions, and routing. Ensure full implementation is generated.")
        
        # Check for required AgenTool components
        if 'from agentool import' not in v and 'import agentool' not in v:
            raise ValueError("code missing agentool imports. Must include 'from agentool import create_agentool, BaseOperationInput' or similar.")
        
        if 'BaseOperationInput' not in v:
            raise ValueError("code missing BaseOperationInput schema. All AgenTools must have an input schema inheriting from BaseOperationInput.")
        
        if 'create_agentool(' not in v:
            raise ValueError("code missing create_agentool() call. Must create the agent using create_agentool() function.")
        
        if 'RoutingConfig' not in v and 'routing_config=' not in v:
            raise ValueError("code missing routing configuration. Must define RoutingConfig for operation mapping.")
        
        return v
    


class ValidationOutput(BaseModel):
    """Validation and test results from the evaluator phase.
    
    Comprehensive validation results with all fixes and improvements applied.
    """
    syntax_valid: bool = Field(
        description="Whether the Python code compiles without syntax errors. Must be true for deployment."
    )
    imports_valid: bool = Field(
        description="Whether all imports resolve correctly (agentool modules, dependencies, standard library). Must be true for deployment."
    )
    tests_passed: bool = Field(
        description="Whether all unit and integration tests pass successfully. Should be true for production readiness."
    )
    issues: List[str] = Field(
        description="""Specific issues found during validation. Examples:
        - 'Missing error handling for NetworkError in fetch operation'
        - 'Incorrect type annotation for TTL parameter (should be Optional[int])'
        - 'Operation "delete" not implemented despite being in schema'
        - 'Circular dependency detected with metrics tool'"""
    )
    fixes_applied: List[str] = Field(
        description="""Corrections made to resolve issues. Examples:
        - 'Added try/except block for NetworkError with proper error propagation'
        - 'Fixed type annotation to Optional[int] with None default'
        - 'Implemented missing delete operation with proper cleanup'
        - 'Refactored to use lazy loading for metrics dependency'"""
    )
    improvements: List[str] = Field(
        description="""Enhancements beyond bug fixes. Examples:
        - 'Added input validation for email format'
        - 'Implemented connection pooling for better performance'
        - 'Added retry logic with exponential backoff'
        - 'Enhanced logging with structured context'"""
    )
    final_code: str = Field(
        description="""Complete, validated, and improved implementation ready for deployment.
        Includes all fixes and enhancements applied during validation."""
    )
    ready_for_deployment: bool = Field(
        description="""Whether code meets all production requirements:
        syntax_valid=True AND imports_valid=True AND critical tests pass AND no blocking issues remain."""
    )
    
    @field_validator('final_code')
    @classmethod
    def validate_final_code(cls, v: str) -> str:
        """Validate that final_code is substantial and not empty."""
        if not v or not v.strip():
            raise ValueError("final_code cannot be empty. The evaluator must provide the complete, validated implementation.")
        
        # Check minimum length (at least a basic class/function structure)
        if len(v.strip()) < 200:
            raise ValueError(f"final_code is too short ({len(v)} chars). Expected a complete implementation with imports, class/function definitions, and logic. Ensure the full validated code is provided.")
        
        # Check for basic Python structure
        code_lower = v.lower()
        if 'import' not in code_lower and 'from' not in code_lower:
            raise ValueError("final_code appears incomplete - missing import statements. Provide the complete implementation including all imports.")
        
        if 'def ' not in v and 'class ' not in v:
            raise ValueError("final_code appears incomplete - missing function or class definitions. Provide the complete implementation.")
        
        return v
    
    @field_validator('issues', 'fixes_applied', 'improvements')
    @classmethod
    def validate_lists_not_none(cls, v: List[str]) -> List[str]:
        """Ensure lists are not None and convert empty to empty list."""
        return v if v is not None else []
    
    @field_validator('ready_for_deployment')
    @classmethod
    def validate_deployment_readiness(cls, v: bool, info) -> bool:
        """Validate deployment readiness is consistent with other fields."""
        data = info.data
        syntax_valid = data.get('syntax_valid', False)
        imports_valid = data.get('imports_valid', False)
        
        # If marked ready for deployment, syntax and imports must be valid
        if v and (not syntax_valid or not imports_valid):
            raise ValueError("ready_for_deployment cannot be True when syntax_valid or imports_valid is False")
        
        return v


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
    """Specification for an individual test case.
    
    Defines a complete test case with inputs, expected outputs, and validation logic.
    """
    name: str = Field(
        description="""Test method name following test_* convention (e.g., 'test_get_existing_key', 'test_set_with_ttl').
        Should be descriptive and indicate what is being tested."""
    )
    description: str = Field(
        description="""Clear explanation of what this test validates (e.g., 'Verifies that getting an existing key returns the correct value',
        'Ensures TTL expiration removes keys after timeout')."""
    )
    test_type: str = Field(
        description="""Category of test. Must be one of:
        - 'unit': Tests single operation in isolation
        - 'integration': Tests interaction with dependencies
        - 'edge_case': Tests boundary conditions (empty input, max size, etc.)
        - 'error_handling': Tests error conditions and exception handling"""
    )
    operation: Optional[str] = Field(
        None,
        description="""Specific operation being tested if applicable (e.g., 'get', 'set', 'delete').
        Should match operation names from the tool's schema."""
    )
    inputs: Dict[str, Any] = Field(
        description="""Complete input data for the test. Example:
        {"operation": "set", "key": "test_key", "value": {"data": "test"}, "ttl": 60}
        Must match the tool's input schema."""
    )
    expected_output: Dict[str, Any] = Field(
        description="""Expected response or behavior. Example:
        {"success": true, "message": "Value stored successfully", "data": {"key": "test_key"}}
        For error cases: {"success": false, "message": "Key not found", "data": null}"""
    )
    dependencies_required: List[str] = Field(
        default_factory=list,
        description="""Names of real AgenTool dependencies needed for this test (e.g., ['storage_kv', 'metrics']).
        These will be created/mocked in test setup."""
    )
    assertions: List[str] = Field(
        description="""Specific assertions to validate. Examples:
        - 'assert result.success is True'
        - 'assert result.data["key"] == "test_key"'
        - 'assert "not found" in result.message.lower()'
        - 'assert mock_storage.get.called_once_with("test_key")'"""
    )


class TestAnalysisOutput(BaseModel):
    """Complete test analysis output from test analyzer phase.
    
    Comprehensive test plan with all test cases and setup requirements.
    """
    tool_name: str = Field(
        description="""Exact name of the AgenTool being tested (e.g., 'rate_limiter', 'cache_manager').
        Must match the tool's registered name."""
    )
    test_cases: List[TestCaseSpec] = Field(
        description="""Complete list of test cases covering all operations and edge cases.
        Should include at least one test per operation plus error handling tests."""
    )
    coverage_requirements: Dict[str, Any] = Field(
        description="""Coverage targets by category. Example:
        {
            "operations": ["get", "set", "delete", "clear"],
            "error_cases": ["key_not_found", "invalid_ttl", "storage_error"],
            "edge_cases": ["empty_key", "large_value", "zero_ttl"],
            "min_coverage_percent": 80
        }"""
    )
    dependency_setup: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="""Dependencies and their initialization order. Example:
        {
            "tier_1": ["storage_kv", "storage_fs"],
            "tier_2": ["metrics", "logging"],
            "tier_3": ["auth"]
        }
        Order matters for avoiding circular dependencies."""
    )
    fixtures_needed: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="""Test fixtures required. Example:
        [
            {"name": "sample_user", "type": "dict", "value": {"id": 123, "name": "Test User"}},
            {"name": "mock_redis", "type": "mock", "target": "redis.Redis"},
            {"name": "temp_dir", "type": "fixture", "cleanup": true}
        ]"""
    )
    test_data_samples: Dict[str, Any] = Field(
        default_factory=dict,
        description="""Sample data for different scenarios. Example:
        {
            "valid_inputs": [{"key": "test1", "value": "data1"}],
            "invalid_inputs": [{"key": "", "value": null}],
            "edge_cases": [{"key": "x" * 1000, "value": {"nested": "data"}}],
            "performance_data": [{"count": 1000, "size": "1kb"}]
        }"""
    )
    integration_points: List[str] = Field(
        default_factory=list,
        description="""External tools to test integration with (e.g., ['storage_kv', 'metrics', 'auth']).
        These should be tested with real instances, not mocks."""
    )
    global_state_to_clear: List[str] = Field(
        default_factory=list,
        description="""Global state that must be cleared between tests. Examples:
        - 'storage_kv:test_*' (all test keys in storage)
        - 'metrics:test.*' (all test metrics)
        - 'cache:*' (entire cache)
        - 'registry:test_tools' (test tool registrations)"""
    )


class TestStubOutput(BaseModel):
    """Test stub output from test stubber phase.
    
    Test file skeleton with structure and placeholders for implementation.
    """
    code: str = Field(
        description="""Complete test file skeleton including:
        - All necessary imports (pytest, asyncio, agentool modules)
        - Test class definition with proper inheritance
        - setUp/tearDown methods for test isolation
        - Placeholder test methods with # TODO comments
        - Basic structure for dependency creation
        Example structure with clear TODO markers for implementation."""
    )
    file_path: str = Field(
        description="""Path for the test file (e.g., 'tests/agentoolkit/test_rate_limiter.py').
        Should follow project test structure and naming conventions."""
    )
    placeholders_count: int = Field(
        description="""Number of test method placeholders that need implementation.
        Each placeholder represents a test case from the analysis phase."""
    )
    structure_elements: Dict[str, bool] = Field(
        description="""Checklist confirming structural elements are included:
        - 'imports': All necessary imports added
        - 'agent_creation': Agent instance creation in setUp
        - 'setup_teardown': Proper test isolation methods
        - 'test_placeholders': All test method stubs created
        - 'real_dependencies': Dependency creation logic included""",
        default_factory=lambda: {
            "imports": False,
            "agent_creation": False,
            "setup_teardown": False,
            "test_placeholders": False,
            "real_dependencies": False
        }
    )


class TestImplementationOutput(BaseModel):
    """Final test implementation output from test crafter phase.
    
    Complete, runnable test suite with all test cases implemented.
    """
    code: str = Field(
        description="""Complete, executable test implementation including:
        - All test methods with proper assertions
        - Real dependency creation and injection
        - Proper async/await for async operations
        - Comprehensive error case testing
        - Performance benchmarks if applicable
        Must be immediately runnable with pytest."""
    )
    file_path: str = Field(
        description="""Final path where test file is saved (e.g., 'tests/agentoolkit/test_rate_limiter.py').
        File should be created and ready to run."""
    )
    test_count: int = Field(
        description="""Total number of test methods implemented.
        Should match or exceed the count from analysis phase."""
    )
    coverage_achieved: Dict[str, float] = Field(
        default_factory=dict,
        description="""Coverage percentages achieved by category. Example:
        {
            "operations": 100.0,  # All operations have tests
            "error_cases": 90.0,  # Most error cases covered
            "edge_cases": 85.0,   # Good edge case coverage
            "overall": 92.5       # Overall test coverage
        }"""
    )
    runnable: bool = Field(
        description="""Whether tests can be executed immediately with 'pytest <file_path>'.
        True means all imports work, dependencies are created, and syntax is valid."""
    )
    dependencies_created: bool = Field(
        description="""Confirms all required AgenTool dependencies are properly created and registered.
        Tests should not fail due to missing dependencies."""
    )