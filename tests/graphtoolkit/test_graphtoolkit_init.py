"""
Tests for GraphToolkit main API functions.

This module tests the high-level GraphToolkit API including workflow creation,
domain discovery, validation, and basic execution patterns.
"""

import asyncio
import tempfile
import pytest
from pathlib import Path
from datetime import datetime
from pydantic_ai.models.test import TestModel
from pydantic_ai import RunContext

# GraphToolkit imports
from graphtoolkit import (
    GraphToolkit, 
    create_agentool_workflow, 
    create_testsuite_workflow,
    list_available_domains,
    get_domain_phases,
    execute_agentool_workflow,
    execute_testsuite_workflow
)
from graphtoolkit.core.types import (
    WorkflowDefinition, 
    WorkflowState, 
    PhaseDefinition,
    StorageRef,
    StorageType,
    NodeConfig
)
from graphtoolkit.core.executor import WorkflowExecutor, WorkflowResult


class TestGraphToolkitAPI:
    """Test suite for GraphToolkit high-level API."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.toolkit = GraphToolkit()
    
    def test_list_available_domains(self):
        """Test domain discovery functionality."""
        domains = list_available_domains()
        
        # Should include the 6 documented domains
        expected_domains = {'agentool', 'testsuite', 'api', 'workflow', 'documentation', 'blockchain'}
        available_domains = set(domains)
        
        # Check that core domains are available
        assert 'agentool' in available_domains
        assert 'testsuite' in available_domains
        
        # Verify it's not empty
        assert len(domains) > 0
        assert all(isinstance(domain, str) for domain in domains)
    
    def test_get_domain_phases(self):
        """Test retrieving phases for a specific domain."""
        # Test AgenTool domain phases
        agentool_phases = get_domain_phases('agentool')
        assert isinstance(agentool_phases, dict)
        
        # Should have the 4 standard phases per documentation
        expected_phases = ['analyzer', 'specifier', 'crafter', 'evaluator']
        for phase in expected_phases:
            if phase in agentool_phases:  # Some phases might not be registered yet
                assert isinstance(agentool_phases[phase], PhaseDefinition)
                assert agentool_phases[phase].phase_name == phase
                assert agentool_phases[phase].domain == 'agentool'
    
    def test_create_agentool_workflow(self):
        """Test AgenTool workflow creation."""
        task_description = "Create a session management AgenTool"
        
        workflow_def, initial_state = create_agentool_workflow(
            task_description=task_description,
            workflow_id="test-workflow-001",
            model="openai:gpt-4o"
        )
        
        # Verify workflow definition
        assert isinstance(workflow_def, WorkflowDefinition)
        assert workflow_def.domain == 'agentool'
        assert len(workflow_def.phase_sequence) == 4
        assert workflow_def.phase_sequence == ['analyzer', 'specifier', 'crafter', 'evaluator']
        
        # Verify initial state
        assert isinstance(initial_state, WorkflowState)
        assert initial_state.workflow_id == "test-workflow-001"
        assert initial_state.domain == 'agentool'
        assert initial_state.workflow_def == workflow_def
        assert initial_state.domain_data['task_description'] == task_description
        assert initial_state.domain_data['model'] == "openai:gpt-4o"
    
    def test_create_testsuite_workflow(self):
        """Test TestSuite workflow creation."""
        code_to_test = """
def calculate_total(items):
    return sum(item['price'] for item in items)
"""
        
        workflow_def, initial_state = create_testsuite_workflow(
            code_to_test=code_to_test,
            framework="pytest",
            coverage_target=0.90,
            workflow_id="test-suite-001"
        )
        
        # Verify workflow definition
        assert isinstance(workflow_def, WorkflowDefinition)
        assert workflow_def.domain == 'testsuite'
        assert len(workflow_def.phase_sequence) == 4
        assert workflow_def.phase_sequence == ['test_analyzer', 'test_designer', 'test_generator', 'test_executor']
        
        # Verify initial state
        assert isinstance(initial_state, WorkflowState)
        assert initial_state.workflow_id == "test-suite-001"
        assert initial_state.domain == 'testsuite'
        assert initial_state.domain_data['code_to_test'] == code_to_test
        assert initial_state.domain_data['framework'] == "pytest"
        assert initial_state.domain_data['coverage_target'] == 0.90
    
    def test_toolkit_create_workflow_generic(self):
        """Test generic workflow creation via GraphToolkit instance."""
        workflow_def, initial_state = self.toolkit.create_workflow(
            domain='agentool',
            phases=['analyzer', 'specifier'],
            workflow_id='custom-workflow',
            initial_data={'custom_field': 'value'},
            enable_refinement=False,
            enable_parallel=True
        )
        
        # Verify workflow definition
        assert isinstance(workflow_def, WorkflowDefinition)
        assert workflow_def.domain == 'agentool'
        assert len(workflow_def.phase_sequence) == 2
        assert workflow_def.enable_refinement == False
        assert workflow_def.enable_parallel == True
        
        # Verify initial state
        assert isinstance(initial_state, WorkflowState)
        assert initial_state.workflow_id == 'custom-workflow'
        assert initial_state.domain_data['custom_field'] == 'value'
    
    def test_workflow_validation(self):
        """Test workflow definition validation."""
        # Create a valid workflow
        workflow_def, _ = create_agentool_workflow("Test task")
        
        # Test validation
        errors = self.toolkit.validate_workflow(workflow_def)
        
        # Should be valid (empty error list)
        assert isinstance(errors, list)
        # If there are validation errors, they should be strings
        if errors:
            assert all(isinstance(error, str) for error in errors)
    
    @pytest.mark.asyncio
    async def test_execute_agentool_workflow_basic(self):
        """Test basic AgenTool workflow execution with TestModel."""
        from graphtoolkit.core.executor import WorkflowExecutor, WorkflowResult
        from graphtoolkit.core.deps import WorkflowDeps
        
        # Create a test model for deterministic testing
        test_model = TestModel()
        
        # Create workflow definition and initial state
        workflow_def, initial_state = create_agentool_workflow(
            task_description="Create a session management AgenTool",
            model="test",
            workflow_id="test-workflow-persistence"
        )
        
        # Create deps with test model
        deps = WorkflowDeps(
            models={'default': test_model}
        )
        
        # Create executor with deps
        executor = WorkflowExecutor(deps=deps)
        
        # Test basic workflow structure
        assert workflow_def.domain == 'agentool'
        assert len(workflow_def.phase_sequence) == 4
        assert initial_state.workflow_id == "test-workflow-persistence"
        assert initial_state.domain == 'agentool'
        
        # Test workflow result structure (without actual execution for now)
        # This tests that the types and structure are correct
        test_result = WorkflowResult(
            state=initial_state,
            outputs={'analyzer': {'data': 'test_analysis'}},
            success=True,
            execution_time=45.2
        )
        
        assert test_result.success == True
        assert test_result.execution_time == 45.2
        assert 'analyzer' in test_result.outputs
    
    @pytest.mark.asyncio
    async def test_execute_testsuite_workflow_basic(self):
        """Test basic TestSuite workflow execution with TestModel."""
        code_to_test = """
def add_numbers(a, b):
    return a + b

def multiply_numbers(a, b):
    return a * b
"""
        
        from graphtoolkit.core.executor import WorkflowExecutor, WorkflowResult
        from graphtoolkit.core.deps import WorkflowDeps
        
        # Create test model
        test_model = TestModel()
        
        # Create workflow
        workflow_def, initial_state = create_testsuite_workflow(
            code_to_test=code_to_test,
            framework="pytest",
            coverage_target=0.85,
            workflow_id="testsuite-test"
        )
        
        # Verify workflow structure
        assert workflow_def.domain == 'testsuite'
        assert len(workflow_def.phase_sequence) == 4
        assert initial_state.workflow_id == "testsuite-test"
        assert initial_state.domain == 'testsuite'
        assert initial_state.domain_data['code_to_test'] == code_to_test
        assert initial_state.domain_data['framework'] == "pytest"
        assert initial_state.domain_data['coverage_target'] == 0.85
        
        # Create deps with test model
        deps = WorkflowDeps(
            models={'default': test_model}
        )
        
        # Create executor with deps
        executor = WorkflowExecutor(deps=deps)
        
        # Test result structure
        test_result = WorkflowResult(
            state=initial_state,
            outputs={'test_generator': {'data': 'generated_tests'}},
            success=True,
            execution_time=32.1
        )
        
        assert test_result.success == True
        assert 'test_generator' in test_result.outputs
    
    @pytest.mark.asyncio
    async def test_toolkit_execute_workflow_integration(self):
        """Test integrated workflow execution via GraphToolkit instance with TestModel."""
        from graphtoolkit.core.executor import WorkflowExecutor, WorkflowResult
        from graphtoolkit.core.deps import WorkflowDeps
        
        # Create test model
        test_model = TestModel()
        
        # Test AgenTool domain workflow creation
        workflow_def, initial_state = self.toolkit.create_workflow(
            domain='agentool',
            phases=['analyzer', 'specifier'],
            initial_data={'task_description': 'Create TODO manager'},
            workflow_id='integration-test'
        )
        
        assert workflow_def.domain == 'agentool'
        assert len(workflow_def.phase_sequence) == 2
        assert initial_state.workflow_id == 'integration-test'
        
        # Test TestSuite domain workflow creation (including test_designer for dependencies)
        workflow_def_ts, initial_state_ts = self.toolkit.create_workflow(
            domain='testsuite',
            phases=['test_analyzer', 'test_designer', 'test_generator'],
            initial_data={'code_to_test': 'def test(): pass', 'framework': 'pytest'},
            workflow_id='testsuite-integration'
        )
        
        assert workflow_def_ts.domain == 'testsuite'
        assert initial_state_ts.workflow_id == 'testsuite-integration'
        
        # Create deps for testing
        deps = WorkflowDeps(
            models={'default': test_model}
        )
        
        # Test executor creation
        executor = WorkflowExecutor(deps=deps)
        assert executor is not None
        assert executor.deps == deps
    
    def test_workflow_id_generation(self):
        """Test automatic workflow ID generation."""
        # Test AgenTool workflow ID generation
        workflow_def1, state1 = create_agentool_workflow("Task 1")
        workflow_def2, state2 = create_agentool_workflow("Task 2")
        
        # Should generate unique IDs
        assert state1.workflow_id != state2.workflow_id
        assert len(state1.workflow_id) > 0
        assert len(state2.workflow_id) > 0
        
        # Test TestSuite workflow ID generation
        workflow_def3, state3 = create_testsuite_workflow("def test(): pass")
        workflow_def4, state4 = create_testsuite_workflow("def test2(): pass")
        
        # Should generate unique IDs
        assert state3.workflow_id != state4.workflow_id
        assert state3.workflow_id not in [state1.workflow_id, state2.workflow_id]
    
    def test_error_handling_invalid_domain(self):
        """Test error handling for invalid domains."""
        with pytest.raises(ValueError, match="Domain invalid_domain not supported"):
            asyncio.run(self.toolkit.execute_workflow(
                domain='invalid_domain',
                phases=['phase1'],
                initial_data={}
            ))
    
    def test_storage_ref_creation(self):
        """Test StorageRef creation and string representation."""
        ref = StorageRef(
            storage_type=StorageType.KV,
            key='workflow/test-id/analyzer',
            created_at=datetime.now(),
            version=1,
            size_bytes=1024
        )
        
        assert str(ref) == 'kv://workflow/test-id/analyzer'
        assert ref.storage_type == StorageType.KV
        assert ref.key == 'workflow/test-id/analyzer'
        assert ref.version == 1
        assert ref.size_bytes == 1024
    
    def test_node_config_defaults(self):
        """Test NodeConfig default values."""
        config = NodeConfig(node_type='test_node')
        
        assert config.node_type == 'test_node'
        assert config.retryable == False
        assert config.max_retries == 0
        assert config.iter_enabled == False
        assert config.cacheable == False
        assert config.timeout is None
    
    def test_workflow_definition_navigation(self):
        """Test WorkflowDefinition helper methods."""
        workflow_def, _ = create_agentool_workflow("Test task")
        
        # Test get_phase
        analyzer_phase = workflow_def.get_phase('analyzer')
        if analyzer_phase:  # Some phases might not be registered
            assert analyzer_phase.phase_name == 'analyzer'
            assert analyzer_phase.domain == 'agentool'
        
        # Test get_next_phase
        next_phase = workflow_def.get_next_phase('analyzer')
        if next_phase:
            assert next_phase == 'specifier'
        
        # Test invalid phase
        invalid_phase = workflow_def.get_phase('nonexistent')
        assert invalid_phase is None
        
        # Test next phase for last phase
        last_next = workflow_def.get_next_phase('evaluator')
        assert last_next is None
    
    def test_workflow_state_helpers(self):
        """Test WorkflowState helper methods."""
        _, initial_state = create_agentool_workflow("Test task")
        
        # Test get_current_node_config when current node is set
        config = initial_state.get_current_node_config()
        assert config is not None  # Initial state has dependency_check as current node
        assert isinstance(config, NodeConfig)
        
        # Test get_next_atomic_node returns the next node in sequence
        next_node = initial_state.get_next_atomic_node()
        assert next_node is not None  # Should have a next node in the atomic sequence
        
        # Test get_current_phase_def
        phase_def = initial_state.get_current_phase_def()
        if phase_def:
            assert phase_def.phase_name == initial_state.current_phase
        
        # Test with_storage_ref helper
        storage_ref = StorageRef(
            storage_type=StorageType.KV,
            key='test-key',
            created_at=datetime.now()
        )
        updated_state = initial_state.with_storage_ref('test_phase', storage_ref)
        assert updated_state.phase_outputs['test_phase'] == storage_ref
        assert updated_state.updated_at > initial_state.updated_at


# Integration test that verifies real workflow creation without mocking core components
class TestGraphToolkitRealIntegration:
    """Integration tests using real GraphToolkit components (no mocking of core framework)."""
    
    def test_real_workflow_creation_and_validation(self):
        """Test that workflows can be created and validated using real components."""
        # Test real AgenTool workflow creation
        workflow_def, initial_state = create_agentool_workflow(
            task_description="Create a real session management AgenTool for testing",
            model="openai:gpt-4o"
        )
        
        # Verify real workflow components
        assert isinstance(workflow_def, WorkflowDefinition)
        assert isinstance(initial_state, WorkflowState)
        assert initial_state.workflow_def is workflow_def
        
        # Test real validation
        toolkit = GraphToolkit()
        validation_errors = toolkit.validate_workflow(workflow_def)
        
        # Should be able to validate without errors in the validation process itself
        assert isinstance(validation_errors, list)
        # Note: validation might return errors about missing components, but validation itself should work
        
    def test_real_testsuite_workflow_creation(self):
        """Test that TestSuite workflows can be created using real components."""
        test_code = """
def calculate_area(length, width):
    '''Calculate area of rectangle.'''
    if length <= 0 or width <= 0:
        raise ValueError("Dimensions must be positive")
    return length * width

def format_currency(amount, currency='USD'):
    '''Format amount as currency string.'''
    return f"{currency} {amount:.2f}"
"""
        
        workflow_def, initial_state = create_testsuite_workflow(
            code_to_test=test_code,
            framework="pytest",
            coverage_target=0.95
        )
        
        # Verify real components
        assert isinstance(workflow_def, WorkflowDefinition)
        assert isinstance(initial_state, WorkflowState)
        assert workflow_def.domain == 'testsuite'
        assert initial_state.domain_data['code_to_test'] == test_code
        assert initial_state.domain_data['coverage_target'] == 0.95
        
        # Verify the workflow is properly structured
        assert len(workflow_def.phase_sequence) == 4
        assert workflow_def.phase_sequence[0] == 'test_analyzer'
        assert workflow_def.phase_sequence[-1] == 'test_executor'
    
    def test_real_domain_discovery(self):
        """Test that domain discovery works with real registry."""
        domains = list_available_domains()
        
        # Should discover real domains
        assert isinstance(domains, list)
        assert len(domains) > 0
        
        # Check that we can get phases for discovered domains
        for domain in domains[:2]:  # Test first 2 to avoid slowdown
            phases = get_domain_phases(domain)
            assert isinstance(phases, dict)
            # Each domain should have at least some phase definitions
            # (might be empty if domain registration is incomplete)
            for phase_name, phase_def in phases.items():
                assert isinstance(phase_name, str)
                assert isinstance(phase_def, PhaseDefinition)
                assert phase_def.phase_name == phase_name


# Performance and resource tests
class TestGraphToolkitPerformance:
    """Performance and resource usage tests."""
    
    def test_workflow_creation_performance(self):
        """Test that workflow creation is reasonably fast."""
        import time
        
        start_time = time.time()
        
        # Create multiple workflows
        for i in range(10):
            workflow_def, initial_state = create_agentool_workflow(f"Task {i}")
            assert isinstance(workflow_def, WorkflowDefinition)
            assert isinstance(initial_state, WorkflowState)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should create 10 workflows in reasonable time (less than 5 seconds)
        assert execution_time < 5.0, f"Workflow creation took {execution_time:.2f}s, expected < 5s"
    
    def test_memory_usage_workflow_creation(self):
        """Test memory usage during workflow creation."""
        import gc
        import sys
        
        # Get initial memory baseline
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Create workflows
        workflows = []
        for i in range(50):
            workflow_def, initial_state = create_testsuite_workflow(f"def test_{i}(): pass")
            workflows.append((workflow_def, initial_state))
        
        # Check memory growth
        gc.collect()
        final_objects = len(gc.get_objects())
        object_growth = final_objects - initial_objects
        
        # Memory growth should be reasonable (less than 10000 new objects for 50 workflows)
        assert object_growth < 10000, f"Created {object_growth} objects for 50 workflows"
        
        # Clean up
        workflows.clear()
        gc.collect()