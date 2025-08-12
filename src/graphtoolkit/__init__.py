"""GraphToolkit - State-driven meta-framework for AI-powered workflows.

GraphToolkit provides a uniform plane where workflow structure, behavior, and types 
are defined in state configuration, with atomic nodes acting as simple executors 
that chain together based on state.

Universal Pattern: InputSchema → Variables → TemplateRender → LLM Call → OutputSchema → Storage

Key Features:
- Domain-agnostic workflow engine
- Atomic node chaining for resilience  
- State-driven configuration and execution
- Template-based prompt management
- Quality gates and refinement loops
- Integration with existing agentoolkits
- Parallel execution with pydantic_graph
"""

from typing import Optional

# Core types and interfaces
from .core.types import (
    WorkflowDefinition,
    WorkflowState,
    PhaseDefinition,
    NodeConfig,
    TemplateConfig,
    ModelParameters,
    StorageRef,
    ValidationResult,
    ConditionConfig,
    TokenUsage
)

# Factory functions for creating workflows
from .core.factory import (
    build_domain_workflow,
    create_workflow_state,
    create_workflow_graph,
    create_domain_workflow,
    validate_workflow_definition
)

# Main executor for running workflows
from .core.executor import (
    WorkflowExecutor,
    WorkflowResult
)

# Registry for managing phases and domains
from .core.registry import (
    get_registry,
    register_phase
)

# Import initialization system
from .core.initialization import (
    initialize_graphtoolkit,
    ensure_graphtoolkit_initialized,
    InitializationConfig,
    default_config,
    test_config,
    graphtoolkit_context
)

# Import all domains to trigger registration
from .domains import smoke

# Import all atomic nodes to trigger registration
from .nodes.atomic import (
    storage, templates, llm, validation, control, execution
)

__version__ = '0.1.0'

__all__ = [
    # Core types
    'WorkflowDefinition',
    'WorkflowState', 
    'PhaseDefinition',
    'NodeConfig',
    'TemplateConfig',
    'ModelParameters',
    'StorageRef',
    'ValidationResult',
    'ConditionConfig',
    'TokenUsage',
    
    # Factory functions
    'build_domain_workflow',
    'create_workflow_state',
    'create_workflow_graph', 
    'create_domain_workflow',
    'validate_workflow_definition',
    
    # Execution
    'WorkflowExecutor',
    'WorkflowResult',
    
    # Registry
    'get_registry',
    'register_phase',
    
    # Initialization
    'initialize_graphtoolkit',
    'ensure_graphtoolkit_initialized',
    'InitializationConfig',
    'default_config',
    'test_config',
    'graphtoolkit_context',
    
    # High-level API
    'GraphToolkit',
    'list_available_domains',
    'get_domain_phases'
]


class GraphToolkit:
    """High-level GraphToolkit API for easy workflow management."""
    
    def __init__(self, config: Optional[InitializationConfig] = None):
        """Initialize GraphToolkit with registry access.
        
        Args:
            config: Optional initialization configuration.
        """
        # Ensure initialization before accessing registry
        ensure_graphtoolkit_initialized(config)
        self.registry = get_registry()
    
    def list_domains(self) -> list[str]:
        """List all registered domains."""
        return self.registry.list_domains()
    
    def get_domain_phases(self, domain: str) -> dict[str, PhaseDefinition]:
        """Get all phases for a specific domain."""
        return self.registry.get_domain_phases(domain)
    
    def create_workflow(
        self,
        domain: str,
        phases: list[str],
        workflow_id: str = None,
        initial_data: dict = None,
        enable_refinement: bool = True,
        enable_parallel: bool = False
    ) -> tuple[WorkflowDefinition, WorkflowState]:
        """Create a complete workflow definition and initial state.
        
        Args:
            domain: Domain name (smoke, etc.)
            phases: List of phase names to include
            workflow_id: Optional workflow identifier
            initial_data: Optional initial domain data
            enable_refinement: Enable quality gates and refinement loops
            enable_parallel: Enable parallel execution where applicable
            
        Returns:
            Tuple of (workflow_definition, initial_state)
        """
        if not workflow_id:
            import uuid
            workflow_id = str(uuid.uuid4())
        
        # Build workflow definition
        workflow_def = build_domain_workflow(
            domain=domain,
            phases=phases,
            enable_refinement=enable_refinement,
            enable_parallel=enable_parallel
        )
        
        # Create initial state
        initial_state = create_workflow_state(
            workflow_def=workflow_def,
            workflow_id=workflow_id,
            initial_data=initial_data or {}
        )
        
        return workflow_def, initial_state
    
    async def execute_workflow(
        self,
        domain: str,
        phases: list[str],
        initial_data: dict,
        workflow_id: str = None,
        model_config: dict[str, str] = None,
        enable_persistence: bool = True
    ) -> dict:
        """Execute a workflow with the specified configuration.
        
        Args:
            domain: Domain name
            phases: List of phase names
            initial_data: Initial data for workflow
            workflow_id: Optional workflow identifier
            model_config: Optional model configuration per phase
            enable_persistence: Whether to enable state persistence
            
        Returns:
            Workflow execution results
        """
        if domain == 'agentool':
            return await execute_agentool_workflow(
                task_description=initial_data.get('task_description', ''),
                model=model_config.get('default', 'openai:gpt-4o') if model_config else 'openai:gpt-4o',
                workflow_id=workflow_id,
                enable_persistence=enable_persistence
            )
        elif domain == 'testsuite':
            return await execute_testsuite_workflow(
                code_to_test=initial_data.get('code_to_test', ''),
                framework=initial_data.get('framework', 'pytest'),
                coverage_target=initial_data.get('coverage_target', 0.85),
                workflow_id=workflow_id,
                enable_persistence=enable_persistence
            )
        else:
            raise ValueError(f'Domain {domain} not supported yet. Available: agentool, testsuite')
    
    def validate_workflow(self, workflow_def: WorkflowDefinition) -> list[str]:
        """Validate a workflow definition for issues.
        
        Args:
            workflow_def: Workflow definition to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        return validate_workflow_definition(workflow_def)


# Global GraphToolkit instance for convenience
_graphtoolkit = GraphToolkit()


def create_agentool_workflow(
    task_description: str,
    workflow_id: str = None,
    model: str = 'openai:gpt-4o'
) -> tuple[WorkflowDefinition, WorkflowState]:
    """Create an AgenTool generation workflow.
    
    Args:
        task_description: Description of the AgenTool to create
        workflow_id: Optional workflow identifier
        model: LLM model to use
        
    Returns:
        Tuple of (workflow_definition, initial_state)
    """
    return _graphtoolkit.create_workflow(
        domain='agentool',
        phases=['analyzer', 'specifier', 'crafter', 'evaluator'],
        workflow_id=workflow_id,
        initial_data={
            'task_description': task_description,
            'model': model,
            'domain': 'agentool'
        }
    )


def create_testsuite_workflow(
    code_to_test: str,
    framework: str = 'pytest',
    coverage_target: float = 0.85,
    workflow_id: str = None
) -> tuple[WorkflowDefinition, WorkflowState]:
    """Create a test suite generation workflow.
    
    Args:
        code_to_test: Code that needs test coverage
        framework: Testing framework to use
        coverage_target: Target coverage percentage
        workflow_id: Optional workflow identifier
        
    Returns:
        Tuple of (workflow_definition, initial_state)
    """
    return _graphtoolkit.create_workflow(
        domain='testsuite',
        phases=['test_analyzer', 'test_designer', 'test_generator', 'test_executor'],
        workflow_id=workflow_id,
        initial_data={
            'code_to_test': code_to_test,
            'framework': framework,
            'coverage_target': coverage_target,
            'domain': 'testsuite'
        }
    )


def list_available_domains() -> list[str]:
    """List all available workflow domains."""
    return _graphtoolkit.list_domains()


def get_domain_phases(domain: str) -> dict[str, PhaseDefinition]:
    """Get all phases for a specific domain."""
    return _graphtoolkit.get_domain_phases(domain)


# Example usage patterns for documentation
"""
Example Usage:

# Simple AgenTool generation
from graphtoolkit import execute_agentool_workflow

result = await execute_agentool_workflow(
    task_description="Create a session management AgenTool",
    model="openai:gpt-4o"
)

# Test suite generation
from graphtoolkit import execute_testsuite_workflow

result = await execute_testsuite_workflow(
    code_to_test=my_python_code,
    framework="pytest",
    coverage_target=0.90
)

# Advanced workflow creation and execution
from graphtoolkit import GraphToolkit

toolkit = GraphToolkit()
workflow_def, initial_state = toolkit.create_workflow(
    domain='agentool',
    phases=['analyzer', 'specifier', 'crafter', 'evaluator'],
    initial_data={'task_description': 'Create TODO manager'},
    enable_refinement=True
)

result = await toolkit.execute_workflow(
    domain='agentool',
    phases=['analyzer', 'specifier', 'crafter', 'evaluator'],
    initial_data={'task_description': 'Create TODO manager'},
    model_config={
        'analyzer': 'openai:gpt-4o-mini',
        'specifier': 'openai:gpt-4o',
        'crafter': 'anthropic:claude-3-5-sonnet-latest',
        'evaluator': 'openai:gpt-4o'
    }
)

# Domain discovery
from graphtoolkit import list_available_domains, get_domain_phases

domains = list_available_domains()  # ['agentool', 'testsuite', 'api', ...]
phases = get_domain_phases('agentool')  # {'analyzer': PhaseDefinition(...), ...}
"""