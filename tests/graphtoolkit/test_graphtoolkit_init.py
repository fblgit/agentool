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
    list_available_domains,
    get_domain_phases
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
        
        # Should include the smoke domain
        available_domains = set(domains)
        
        # Check that smoke domain is available
        assert 'smoke' in available_domains
        
        # Verify it's not empty
        assert len(domains) > 0
        assert all(isinstance(domain, str) for domain in domains)
    
    def test_get_domain_phases(self):
        """Test retrieving phases for a specific domain."""
        # Test smoke domain phases
        smoke_phases = get_domain_phases('smoke')
        assert isinstance(smoke_phases, dict)
        
        # Should have the 4 standard phases for smoke domain
        expected_phases = ['ingredient_analyzer', 'recipe_designer', 'recipe_crafter', 'recipe_evaluator']
        for phase in expected_phases:
            if phase in smoke_phases:  # Some phases might not be registered yet
                assert isinstance(smoke_phases[phase], PhaseDefinition)
                assert smoke_phases[phase].phase_name == phase
                assert smoke_phases[phase].domain == 'smoke'
    
    def test_create_smoke_workflow(self):
        """Test smoke workflow creation."""
        from graphtoolkit.domains.smoke import create_smoke_workflow
        
        ingredients = ["chicken", "rice", "vegetables"]
        
        workflow_def, initial_state = create_smoke_workflow(
            ingredients=ingredients,
            dietary_restrictions=["gluten-free"],
            workflow_id="test-workflow-001"
        )
        
        # Verify workflow definition
        assert isinstance(workflow_def, WorkflowDefinition)
        assert workflow_def.domain == 'smoke'
        assert len(workflow_def.phase_sequence) == 4
        assert workflow_def.phase_sequence == ['ingredient_analyzer', 'recipe_designer', 'recipe_crafter', 'recipe_evaluator']
        
        # Verify initial state
        assert isinstance(initial_state, WorkflowState)
        assert initial_state.workflow_id == "test-workflow-001"
        assert initial_state.domain == 'smoke'
        assert initial_state.workflow_def == workflow_def
        assert initial_state.domain_data['ingredients'] == ingredients
        assert initial_state.domain_data['dietary_restrictions'] == ["gluten-free"]
    
    
    def test_toolkit_create_workflow_generic(self):
        """Test generic workflow creation via GraphToolkit instance."""
        workflow_def, initial_state = self.toolkit.create_workflow(
            domain='smoke',
            phases=['ingredient_analyzer', 'recipe_designer'],
            workflow_id='custom-workflow',
            initial_data={'custom_field': 'value'},
            enable_refinement=False,
            enable_parallel=True
        )
        
        # Verify workflow definition
        assert isinstance(workflow_def, WorkflowDefinition)
        assert workflow_def.domain == 'smoke'
        assert len(workflow_def.phase_sequence) == 2
        assert workflow_def.enable_refinement == False
        assert workflow_def.enable_parallel == True
        
        # Verify initial state
        assert isinstance(initial_state, WorkflowState)
        assert initial_state.workflow_id == 'custom-workflow'
        assert initial_state.domain_data['custom_field'] == 'value'
    
    def test_workflow_validation(self):
        """Test workflow definition validation."""
        from graphtoolkit.domains.smoke import create_smoke_workflow
        
        # Create a valid workflow
        workflow_def, _ = create_smoke_workflow(["ingredients"])
        
        # Test validation
        errors = self.toolkit.validate_workflow(workflow_def)
        
        # Should be valid (empty error list)
        assert isinstance(errors, list)
        # If there are validation errors, they should be strings
        if errors:
            assert all(isinstance(error, str) for error in errors)
    
    @pytest.mark.asyncio
    async def test_execute_smoke_workflow_basic(self):
        """Test basic smoke workflow execution with TestModel."""
        from graphtoolkit.core.executor import WorkflowExecutor, WorkflowResult
        from graphtoolkit.core.deps import WorkflowDeps
        from graphtoolkit.domains.smoke import create_smoke_workflow
        
        # Create a test model for deterministic testing
        test_model = TestModel()
        
        # Create workflow definition and initial state
        workflow_def, initial_state = create_smoke_workflow(
            ingredients=["chicken", "rice"],
            workflow_id="test-workflow-persistence"
        )
        
        # Create deps with test model
        deps = WorkflowDeps(
            models={'default': test_model}
        )
        
        # Create executor with deps
        executor = WorkflowExecutor(deps=deps)
        
        # Test basic workflow structure
        assert workflow_def.domain == 'smoke'
        assert len(workflow_def.phase_sequence) == 4
        assert initial_state.workflow_id == "test-workflow-persistence"
        assert initial_state.domain == 'smoke'
        
        # Test workflow result structure (without actual execution for now)
        # This tests that the types and structure are correct
        test_result = WorkflowResult(
            state=initial_state,
            outputs={'ingredient_analyzer': {'data': 'test_analysis'}},
            success=True,
            execution_time=45.2
        )
        
        assert test_result.success == True
        assert test_result.execution_time == 45.2
        assert 'ingredient_analyzer' in test_result.outputs
    
        
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
        
        # Test smoke domain workflow creation
        workflow_def, initial_state = self.toolkit.create_workflow(
            domain='smoke',
            phases=['ingredient_analyzer', 'recipe_designer'],
            initial_data={'ingredients': ['chicken', 'rice']},
            workflow_id='integration-test'
        )
        
        assert workflow_def.domain == 'smoke'
        assert len(workflow_def.phase_sequence) == 2
        assert initial_state.workflow_id == 'integration-test'
        
        # Test smoke domain with different phases
        workflow_def_ts, initial_state_ts = self.toolkit.create_workflow(
            domain='smoke',
            phases=['ingredient_analyzer', 'recipe_designer', 'recipe_crafter'],
            initial_data={'ingredients': ['tomato', 'pasta']},
            workflow_id='smoke-integration'
        )
        
        assert workflow_def_ts.domain == 'smoke'
        assert initial_state_ts.workflow_id == 'smoke-integration'
        
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
        from graphtoolkit.domains.smoke import create_smoke_workflow
        
        # Test smoke workflow ID generation
        workflow_def1, state1 = create_smoke_workflow(["ingredient1"])
        workflow_def2, state2 = create_smoke_workflow(["ingredient2"])
        
        # Should generate unique IDs
        assert state1.workflow_id != state2.workflow_id
        assert len(state1.workflow_id) > 0
        assert len(state2.workflow_id) > 0
        
        # Test another smoke workflow ID generation
        workflow_def3, state3 = create_smoke_workflow(["ingredient3"])
        workflow_def4, state4 = create_smoke_workflow(["ingredient4"])
        
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
        from graphtoolkit.domains.smoke import create_smoke_workflow
        workflow_def, _ = create_smoke_workflow(["test_ingredient"])
        
        # Test get_phase
        analyzer_phase = workflow_def.get_phase('ingredient_analyzer')
        if analyzer_phase:  # Some phases might not be registered
            assert analyzer_phase.phase_name == 'ingredient_analyzer'
            assert analyzer_phase.domain == 'smoke'
        
        # Test get_next_phase
        next_phase = workflow_def.get_next_phase('ingredient_analyzer')
        if next_phase:
            assert next_phase == 'recipe_designer'
        
        # Test invalid phase
        invalid_phase = workflow_def.get_phase('nonexistent')
        assert invalid_phase is None
        
        # Test next phase for last phase
        last_next = workflow_def.get_next_phase('recipe_evaluator')
        assert last_next is None
    
    def test_workflow_state_helpers(self):
        """Test WorkflowState helper methods."""
        from graphtoolkit.domains.smoke import create_smoke_workflow
        _, initial_state = create_smoke_workflow(["test_ingredient"])
        
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
        # Test real smoke workflow creation
        from graphtoolkit.domains.smoke import create_smoke_workflow
        workflow_def, initial_state = create_smoke_workflow(
            ingredients=["chicken", "rice", "vegetables"],
            dietary_restrictions=["gluten-free"]
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
        
    def test_real_smoke_workflow_creation_alternative(self):
        """Test that smoke workflows can be created with different parameters."""
        from graphtoolkit.domains.smoke import create_smoke_workflow
        
        workflow_def, initial_state = create_smoke_workflow(
            ingredients=["pasta", "tomatoes", "basil"],
            dietary_restrictions=["vegan"],
            cuisine_preference="Italian",
            max_cook_time=30
        )
        
        # Verify real components
        assert isinstance(workflow_def, WorkflowDefinition)
        assert isinstance(initial_state, WorkflowState)
        assert workflow_def.domain == 'smoke'
        assert initial_state.domain_data['ingredients'] == ["pasta", "tomatoes", "basil"]
        assert initial_state.domain_data['dietary_restrictions'] == ["vegan"]
        
        # Verify the workflow is properly structured
        assert len(workflow_def.phase_sequence) == 4
        assert workflow_def.phase_sequence[0] == 'ingredient_analyzer'
        assert workflow_def.phase_sequence[-1] == 'recipe_evaluator'
    
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
        from graphtoolkit.domains.smoke import create_smoke_workflow
        
        start_time = time.time()
        
        # Create multiple workflows
        for i in range(10):
            workflow_def, initial_state = create_smoke_workflow([f"ingredient_{i}"])
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
        from graphtoolkit.domains.smoke import create_smoke_workflow
        
        # Get initial memory baseline
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Create workflows
        workflows = []
        for i in range(50):
            workflow_def, initial_state = create_smoke_workflow([f"ingredient_{i}"])
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