"""
Tests for WorkflowState mutations and persistence.

This module tests the state management aspects of GraphToolkit including:
- WorkflowState immutability and update patterns
- State persistence with pydantic_graph
- State transitions and phase tracking  
- Domain data management and storage references
- Token usage and metrics tracking
- Refinement history and quality scores
"""

import tempfile
import pytest
import json
import pickle
from dataclasses import replace
from pathlib import Path
from datetime import datetime, timedelta
from pydantic_ai.models.test import TestModel
from pydantic import BaseModel

# GraphToolkit imports
from graphtoolkit.core.types import (
    WorkflowState,
    WorkflowDefinition,
    PhaseDefinition,
    NodeConfig,
    StorageRef,
    StorageType,
    ValidationResult,
    RefinementRecord,
    TokenUsage,
    TemplateConfig,
    ModelParameters
)

# pydantic_graph imports for persistence testing
# Note: SimpleStatePersistence is not a real class, we'll use a simple file-based approach
# from pydantic_graph import SimpleStatePersistence


class TestWorkflowStateImmutability:
    """Test WorkflowState immutability patterns."""
    
    def setup_method(self):
        """Set up test state."""
        # Create proper input/output schemas
        class TestInput(BaseModel):
            data: str
        
        class TestOutput(BaseModel):
            result: str
        
        # Create a complete workflow definition
        phase_def = PhaseDefinition(
            phase_name='test_phase',
            atomic_nodes=['node1', 'node2'],
            input_schema=TestInput,
            output_schema=TestOutput,
            templates=TemplateConfig(
                system_template='templates/test_system.jinja',
                user_template='templates/test_user.jinja'
            ),
            model_config=ModelParameters(temperature=0.8)
        )
        
        self.workflow_def = WorkflowDefinition(
            domain='test',
            phases={'test_phase': phase_def},
            phase_sequence=['test_phase'],
            node_configs={
                'node1': NodeConfig(node_type='test', retryable=True)
            }
        )
        
        self.initial_state = WorkflowState(
            workflow_def=self.workflow_def,
            workflow_id='test-workflow',
            domain='test',
            current_phase='test_phase',
            current_node='node1',
            domain_data={'initial': 'data'}
        )
    
    def test_state_immutability(self):
        """Test that WorkflowState updates don't mutate original state."""
        original_domain_data = self.initial_state.domain_data.copy()
        original_completed_phases = self.initial_state.completed_phases.copy()
        
        # Update state
        updated_state = replace(
            self.initial_state,
            domain_data={'updated': 'data'},
            completed_phases={'test_phase'}
        )
        
        # Original state should be unchanged
        assert self.initial_state.domain_data == original_domain_data
        assert self.initial_state.completed_phases == original_completed_phases
        
        # Updated state should have new values
        assert updated_state.domain_data == {'updated': 'data'}
        assert updated_state.completed_phases == {'test_phase'}
        
        # States should be different objects
        assert self.initial_state is not updated_state
    
    def test_with_storage_ref_helper(self):
        """Test with_storage_ref helper method."""
        storage_ref = StorageRef(
            storage_type=StorageType.KV,
            key='test/key',
            created_at=datetime.now()
        )
        
        original_updated_at = self.initial_state.updated_at
        
        # Add storage reference
        updated_state = self.initial_state.with_storage_ref('test_phase', storage_ref)
        
        # Original state unchanged
        assert 'test_phase' not in self.initial_state.phase_outputs
        
        # Updated state has new reference
        assert 'test_phase' in updated_state.phase_outputs
        assert updated_state.phase_outputs['test_phase'] == storage_ref
        
        # Updated timestamp should be newer
        assert updated_state.updated_at > original_updated_at
    
    def test_state_navigation_helpers(self):
        """Test state helper methods."""
        # Test get_current_node_config
        config = self.initial_state.get_current_node_config()
        assert config is not None
        assert config.node_type == 'test'
        assert config.retryable == True
        
        # Test get_next_atomic_node
        next_node = self.initial_state.get_next_atomic_node()
        assert next_node == 'node2'
        
        # Test when at last node
        last_node_state = replace(self.initial_state, current_node='node2')
        next_from_last = last_node_state.get_next_atomic_node()
        assert next_from_last is None
        
        # Test get_current_phase_def
        phase_def = self.initial_state.get_current_phase_def()
        assert phase_def is not None
        assert phase_def.phase_name == 'test_phase'
    
    def test_state_helper_edge_cases(self):
        """Test state helpers with missing data."""
        # State with invalid current node
        invalid_state = replace(self.initial_state, current_node='nonexistent')
        
        config = invalid_state.get_current_node_config()
        assert config is None
        
        next_node = invalid_state.get_next_atomic_node()
        assert next_node is None
        
        # State with invalid current phase
        invalid_phase_state = replace(self.initial_state, current_phase='nonexistent')
        
        phase_def = invalid_phase_state.get_current_phase_def()
        assert phase_def is None


class TestStateTransitions:
    """Test state transitions during workflow execution."""
    
    def setup_method(self):
        """Set up multi-phase workflow."""
        # Create proper input/output schemas
        class AnalyzerInput(BaseModel):
            task_description: str
        
        class AnalyzerOutput(BaseModel):
            analysis: str
            missing_tools: list
        
        class CrafterInput(BaseModel):
            analysis: str
        
        class CrafterOutput(BaseModel):
            generated_code: str
        
        # Create multi-phase workflow
        analyzer_phase = PhaseDefinition(
            phase_name='analyzer',
            atomic_nodes=['dependency_check', 'analyze'],
            input_schema=AnalyzerInput,
            output_schema=AnalyzerOutput
        )
        
        crafter_phase = PhaseDefinition(
            phase_name='crafter',
            atomic_nodes=['load_analysis', 'generate_code'],
            input_schema=CrafterInput,
            output_schema=CrafterOutput,
            dependencies=['analyzer']
        )
        
        self.workflow_def = WorkflowDefinition(
            domain='agentool',
            phases={'analyzer': analyzer_phase, 'crafter': crafter_phase},
            phase_sequence=['analyzer', 'crafter'],
            node_configs={
                'dependency_check': NodeConfig(node_type='storage'),
                'analyze': NodeConfig(node_type='llm'),
                'load_analysis': NodeConfig(node_type='storage'),
                'generate_code': NodeConfig(node_type='llm')
            }
        )
        
        self.initial_state = WorkflowState(
            workflow_def=self.workflow_def,
            workflow_id='transition-test',
            domain='agentool',
            current_phase='analyzer',
            current_node='dependency_check'
        )
    
    def test_node_transitions(self):
        """Test transitions between nodes within a phase."""
        # Start with first node
        assert self.initial_state.current_node == 'dependency_check'
        assert self.initial_state.get_next_atomic_node() == 'analyze'
        
        # Move to next node
        next_state = replace(self.initial_state, current_node='analyze')
        assert next_state.current_node == 'analyze'
        assert next_state.get_next_atomic_node() is None  # Last node in phase
    
    def test_phase_completion(self):
        """Test phase completion tracking."""
        # Complete analyzer phase
        completed_state = replace(
            self.initial_state,
            completed_phases={'analyzer'},
            current_phase='crafter',
            current_node='load_analysis'
        )
        
        assert 'analyzer' in completed_state.completed_phases
        assert completed_state.current_phase == 'crafter'
        assert completed_state.current_node == 'load_analysis'
        
        # Test next phase lookup
        next_phase = self.workflow_def.get_next_phase('analyzer')
        assert next_phase == 'crafter'
        
        # No next phase after crafter
        next_after_crafter = self.workflow_def.get_next_phase('crafter')
        assert next_after_crafter is None
    
    def test_dependency_tracking(self):
        """Test dependency satisfaction checking."""
        # Crafter phase depends on analyzer
        crafter_def = self.workflow_def.phases['crafter']
        assert 'analyzer' in crafter_def.dependencies
        
        # State with analyzer completed
        state_with_analyzer = replace(
            self.initial_state,
            completed_phases={'analyzer'},
            current_phase='crafter'
        )
        
        # Check if dependencies are satisfied
        for dep in crafter_def.dependencies:
            assert dep in state_with_analyzer.completed_phases
    
    def test_workflow_progression(self):
        """Test complete workflow progression."""
        states = []
        current_state = self.initial_state
        states.append(current_state)
        
        # Simulate node transitions
        progression = [
            ('analyzer', 'dependency_check'),
            ('analyzer', 'analyze'),
            ('crafter', 'load_analysis'),  # Phase transition
            ('crafter', 'generate_code')
        ]
        
        for phase, node in progression[1:]:
            # Mark previous phase complete if changing phases
            completed_phases = current_state.completed_phases
            if phase != current_state.current_phase:
                completed_phases = completed_phases | {current_state.current_phase}
            
            current_state = replace(
                current_state,
                current_phase=phase,
                current_node=node,
                completed_phases=completed_phases
            )
            states.append(current_state)
        
        # Final state should have both phases completed
        final_state = replace(
            current_state,
            completed_phases=current_state.completed_phases | {current_state.current_phase}
        )
        
        assert len(final_state.completed_phases) == 2
        assert 'analyzer' in final_state.completed_phases
        assert 'crafter' in final_state.completed_phases


class TestDomainDataManagement:
    """Test domain data management and evolution."""
    
    def setup_method(self):
        """Set up state with domain data."""
        self.workflow_def = WorkflowDefinition(
            domain='agentool',
            phases={},
            phase_sequence=[],
            node_configs={}
        )
        
        self.state = WorkflowState(
            workflow_def=self.workflow_def,
            workflow_id='domain-test',
            domain='agentool',
            current_phase='analyzer',
            domain_data={
                'task_description': 'Create a session manager',
                'model': 'openai:gpt-4o'
            }
        )
    
    def test_domain_data_accumulation(self):
        """Test accumulating domain data across phases."""
        # Add analyzer results
        analyzer_state = replace(
            self.state,
            domain_data={
                **self.state.domain_data,
                'analyzer_output': {
                    'missing_tools': ['session_create', 'session_read'],
                    'complexity': 'medium'
                }
            }
        )
        
        # Add crafter results
        crafter_state = replace(
            analyzer_state,
            domain_data={
                **analyzer_state.domain_data,
                'crafter_output': {
                    'generated_code': 'class SessionManager: ...',
                    'test_code': 'def test_session_create(): ...'
                }
            }
        )
        
        # All data should be preserved
        final_data = crafter_state.domain_data
        assert 'task_description' in final_data
        assert 'analyzer_output' in final_data
        assert 'crafter_output' in final_data
        
        # Individual phase data should be intact
        assert len(final_data['analyzer_output']['missing_tools']) == 2
        assert 'SessionManager' in final_data['crafter_output']['generated_code']
    
    def test_domain_specific_patterns(self):
        """Test domain-specific data patterns."""
        # AgenTool domain pattern
        agentool_data = {
            'task_description': 'Create user auth system',
            'analyzer_output': {'tools': ['auth_login', 'auth_logout']},
            'specifier_output': {'specifications': [...]},
            'crafter_output': {'generated_code': '...'},
            'evaluator_output': {'quality_score': 0.89}
        }
        
        agentool_state = replace(self.state, domain_data=agentool_data)
        
        # TestSuite domain pattern
        testsuite_data = {
            'code_to_test': 'def calculate(x): return x * 2',
            'framework': 'pytest',
            'coverage_target': 0.85,
            'test_analyzer_output': {'complexity': 3},
            'test_generator_output': {'test_files': {'test_calc.py': '...'}},
            'test_executor_output': {'coverage': 0.92}
        }
        
        testsuite_state = replace(
            self.state,
            domain='testsuite',
            domain_data=testsuite_data
        )
        
        # Verify domain-specific data
        assert agentool_state.domain_data['task_description'] == 'Create user auth system'
        assert testsuite_state.domain_data['code_to_test'] == 'def calculate(x): return x * 2'
        assert testsuite_state.domain_data['framework'] == 'pytest'
    
    def test_data_size_tracking(self):
        """Test handling of large domain data."""
        import sys
        
        # Create large data structure
        large_data = {
            'large_list': list(range(10000)),
            'large_dict': {f'key_{i}': f'value_{i}' for i in range(1000)},
            'large_string': 'x' * 100000
        }
        
        large_state = replace(self.state, domain_data=large_data)
        
        # State should handle large data
        assert len(large_state.domain_data['large_list']) == 10000
        assert len(large_state.domain_data['large_dict']) == 1000
        assert len(large_state.domain_data['large_string']) == 100000
        
        # Memory usage should be reasonable (rough check)
        state_size = sys.getsizeof(large_state)
        assert state_size < 10 * 1024 * 1024  # Less than 10MB


class TestStorageReferences:
    """Test storage reference management."""
    
    def setup_method(self):
        """Set up state with storage references."""
        self.workflow_def = WorkflowDefinition(
            domain='test',
            phases={},
            phase_sequence=[],
            node_configs={}
        )
        
        self.state = WorkflowState(
            workflow_def=self.workflow_def,
            workflow_id='storage-test',
            domain='test',
            current_phase='test_phase'
        )
    
    def test_storage_ref_creation_and_access(self):
        """Test creating and accessing storage references."""
        # Create different types of storage references
        kv_ref = StorageRef(
            storage_type=StorageType.KV,
            key='workflow/test/analysis',
            created_at=datetime.now(),
            version=1,
            size_bytes=1024
        )
        
        fs_ref = StorageRef(
            storage_type=StorageType.FS,
            key='workflow/test/generated_code.py',
            created_at=datetime.now(),
            size_bytes=4096
        )
        
        # Add to state
        updated_state = self.state.with_storage_ref('analysis', kv_ref)
        final_state = updated_state.with_storage_ref('code_generation', fs_ref)
        
        # Verify references
        assert 'analysis' in final_state.phase_outputs
        assert 'code_generation' in final_state.phase_outputs
        
        assert final_state.phase_outputs['analysis'] == kv_ref
        assert final_state.phase_outputs['code_generation'] == fs_ref
        
        # Test string representations
        assert str(kv_ref) == 'kv://workflow/test/analysis'
        assert str(fs_ref) == 'fs://workflow/test/generated_code.py'
    
    def test_storage_ref_versioning(self):
        """Test storage reference versioning for refinement."""
        base_time = datetime.now()
        
        # Original version
        v1_ref = StorageRef(
            storage_type=StorageType.KV,
            key='workflow/test/phase_output',
            created_at=base_time,
            version=1
        )
        
        # Refined version
        v2_ref = StorageRef(
            storage_type=StorageType.KV,
            key='workflow/test/phase_output/v2',
            created_at=base_time + timedelta(minutes=5),
            version=2
        )
        
        # Add both versions
        state_v1 = self.state.with_storage_ref('phase_v1', v1_ref)
        state_v2 = state_v1.with_storage_ref('phase_v2', v2_ref)
        
        # Should maintain version tracking
        assert state_v2.phase_outputs['phase_v1'].version == 1
        assert state_v2.phase_outputs['phase_v2'].version == 2
        assert state_v2.phase_outputs['phase_v2'].created_at > state_v2.phase_outputs['phase_v1'].created_at
    
    def test_storage_ref_batch_operations(self):
        """Test managing multiple storage references."""
        refs = {}
        
        # Create batch of references
        for i in range(5):
            ref = StorageRef(
                storage_type=StorageType.KV,
                key=f'batch/item_{i}',
                created_at=datetime.now()
            )
            refs[f'item_{i}'] = ref
        
        # Add all references
        current_state = self.state
        for name, ref in refs.items():
            current_state = current_state.with_storage_ref(name, ref)
        
        # Verify all references
        assert len(current_state.phase_outputs) == 5
        for name, ref in refs.items():
            assert name in current_state.phase_outputs
            assert current_state.phase_outputs[name] == ref


class TestQualityAndRefinement:
    """Test quality scores and refinement history."""
    
    def setup_method(self):
        """Set up state for quality tracking."""
        self.workflow_def = WorkflowDefinition(
            domain='agentool',
            phases={},
            phase_sequence=[],
            node_configs={}
        )
        
        self.state = WorkflowState(
            workflow_def=self.workflow_def,
            workflow_id='quality-test',
            domain='agentool',
            current_phase='crafter'
        )
    
    def test_quality_score_tracking(self):
        """Test quality score management."""
        # Add quality scores for different phases
        quality_state = replace(
            self.state,
            quality_scores={
                'analyzer': 0.87,
                'specifier': 0.91,
                'crafter': 0.73,  # Below threshold
                'evaluator': 0.89
            }
        )
        
        # Verify scores
        assert quality_state.quality_scores['analyzer'] == 0.87
        assert quality_state.quality_scores['crafter'] == 0.73
        
        # Check if any scores are below threshold (0.8)
        below_threshold = [
            phase for phase, score in quality_state.quality_scores.items()
            if score < 0.8
        ]
        assert 'crafter' in below_threshold
    
    def test_refinement_history_tracking(self):
        """Test refinement history management."""
        initial_time = datetime.now()
        
        # Create refinement record
        refinement = RefinementRecord(
            iteration=1,
            timestamp=initial_time,
            previous_score=0.65,
            new_score=0.87,
            feedback="Improved error handling and added documentation",
            changes_made=[
                "Added try-catch blocks",
                "Added comprehensive docstrings", 
                "Improved variable naming"
            ]
        )
        
        # Add to state
        refinement_state = replace(
            self.state,
            refinement_count={'crafter': 1},
            refinement_history={'crafter': [refinement]}
        )
        
        # Verify refinement tracking
        assert refinement_state.refinement_count['crafter'] == 1
        assert len(refinement_state.refinement_history['crafter']) == 1
        
        history_entry = refinement_state.refinement_history['crafter'][0]
        assert history_entry.iteration == 1
        assert history_entry.new_score > history_entry.previous_score
        assert len(history_entry.changes_made) == 3
    
    def test_multiple_refinement_cycles(self):
        """Test multiple refinement cycles for same phase."""
        refinements = []
        base_time = datetime.now()
        
        # Create multiple refinement records
        scores = [(0.65, 0.75), (0.75, 0.82), (0.82, 0.89)]
        for i, (prev, new) in enumerate(scores):
            refinement = RefinementRecord(
                iteration=i + 1,
                timestamp=base_time + timedelta(minutes=i * 10),
                previous_score=prev,
                new_score=new,
                feedback=f"Refinement {i + 1} feedback",
                changes_made=[f"Change {i + 1}"]
            )
            refinements.append(refinement)
        
        # Add all refinements
        multi_refinement_state = replace(
            self.state,
            refinement_count={'crafter': 3},
            refinement_history={'crafter': refinements}
        )
        
        # Verify progression
        assert multi_refinement_state.refinement_count['crafter'] == 3
        assert len(multi_refinement_state.refinement_history['crafter']) == 3
        
        # Verify score improvement over iterations
        history = multi_refinement_state.refinement_history['crafter']
        assert history[0].new_score == 0.75
        assert history[1].new_score == 0.82
        assert history[2].new_score == 0.89
        assert history[-1].new_score > history[0].previous_score
    
    def test_validation_results_tracking(self):
        """Test validation result management."""
        # Create validation results
        syntax_validation = ValidationResult(
            valid=True,
            errors=[],
            warnings=['Unused import on line 5'],
            metadata={'lines_checked': 150}
        )
        
        import_validation = ValidationResult(
            valid=False,
            errors=['Module "nonexistent" not found'],
            warnings=[],
            metadata={'imports_checked': 8}
        )
        
        # Add to state
        validation_state = replace(
            self.state,
            validation_results={
                'syntax': syntax_validation,
                'imports': import_validation
            }
        )
        
        # Verify validation tracking
        assert validation_state.validation_results['syntax'].valid == True
        assert validation_state.validation_results['imports'].valid == False
        assert len(validation_state.validation_results['syntax'].warnings) == 1
        assert len(validation_state.validation_results['imports'].errors) == 1


class TestTokenUsageTracking:
    """Test token usage and metrics tracking."""
    
    def setup_method(self):
        """Set up state for token tracking."""
        self.workflow_def = WorkflowDefinition(
            domain='agentool',
            phases={},
            phase_sequence=[],
            node_configs={}
        )
        
        self.state = WorkflowState(
            workflow_def=self.workflow_def,
            workflow_id='token-test',
            domain='agentool',
            current_phase='analyzer'
        )
    
    def test_token_usage_tracking(self):
        """Test token usage accumulation."""
        # Create token usage records
        analyzer_tokens = TokenUsage(
            prompt_tokens=1200,
            completion_tokens=450,
            total_tokens=1650,
            model='openai:gpt-4o'
        )
        
        crafter_tokens = TokenUsage(
            prompt_tokens=2100,
            completion_tokens=800,
            total_tokens=2900,
            model='openai:gpt-4o'
        )
        
        # Add to state
        token_state = replace(
            self.state,
            total_token_usage={
                'analyzer': analyzer_tokens,
                'crafter': crafter_tokens
            }
        )
        
        # Verify tracking
        assert token_state.total_token_usage['analyzer'].total_tokens == 1650
        assert token_state.total_token_usage['crafter'].total_tokens == 2900
        
        # Calculate total usage across phases
        total_prompt = sum(usage.prompt_tokens for usage in token_state.total_token_usage.values())
        total_completion = sum(usage.completion_tokens for usage in token_state.total_token_usage.values())
        total_all = sum(usage.total_tokens for usage in token_state.total_token_usage.values())
        
        assert total_prompt == 3300
        assert total_completion == 1250
        assert total_all == 4550
    
    def test_token_usage_addition(self):
        """Test TokenUsage addition operation."""
        usage1 = TokenUsage(
            prompt_tokens=500,
            completion_tokens=200,
            total_tokens=700,
            model='openai:gpt-4o'
        )
        
        usage2 = TokenUsage(
            prompt_tokens=300,
            completion_tokens=150,
            total_tokens=450,
            model='openai:gpt-4o'
        )
        
        # Add token usages
        combined = usage1 + usage2
        
        assert combined.prompt_tokens == 800
        assert combined.completion_tokens == 350
        assert combined.total_tokens == 1150
        assert combined.model == 'openai:gpt-4o'
    
    def test_token_usage_different_models_error(self):
        """Test error when adding different model token usage."""
        usage1 = TokenUsage(
            prompt_tokens=500,
            completion_tokens=200,
            total_tokens=700,
            model='openai:gpt-4o'
        )
        
        usage2 = TokenUsage(
            prompt_tokens=300,
            completion_tokens=150,
            total_tokens=450,
            model='anthropic:claude-3-5-sonnet-latest'
        )
        
        # Should raise error for different models
        with pytest.raises(ValueError, match="Cannot add usage for different models"):
            combined = usage1 + usage2


class TestStatePersistence:
    """Test state persistence with pydantic_graph integration."""
    
    def setup_method(self):
        """Set up persistence test environment."""
        self.workflow_def = WorkflowDefinition(
            domain='persistence_test',
            phases={},
            phase_sequence=[],
            node_configs={}
        )
        
        self.state = WorkflowState(
            workflow_def=self.workflow_def,
            workflow_id='persistence-test',
            domain='persistence_test',
            current_phase='test_phase',
            domain_data={'test': 'data'},
            quality_scores={'test_phase': 0.85}
        )
    
    def test_state_serialization(self):
        """Test that WorkflowState can be serialized."""
        # Test JSON serialization (for simple data)
        simple_state = WorkflowState(
            workflow_def=self.workflow_def,
            workflow_id='json-test',
            domain='test',
            current_phase='phase1'
        )
        
        # Convert to dict for JSON serialization
        state_dict = {
            'workflow_id': simple_state.workflow_id,
            'domain': simple_state.domain,
            'current_phase': simple_state.current_phase,
            'completed_phases': list(simple_state.completed_phases),
            'domain_data': simple_state.domain_data,
            'quality_scores': simple_state.quality_scores
        }
        
        # Should be JSON serializable
        json_str = json.dumps(state_dict)
        restored_dict = json.loads(json_str)
        
        assert restored_dict['workflow_id'] == simple_state.workflow_id
        assert restored_dict['domain'] == simple_state.domain
        assert restored_dict['current_phase'] == simple_state.current_phase
    
    def test_state_pickle_serialization(self):
        """Test state serialization with pickle."""
        # Full state with complex objects
        storage_ref = StorageRef(
            storage_type=StorageType.KV,
            key='test/key',
            created_at=datetime.now()
        )
        
        complex_state = replace(
            self.state,
            phase_outputs={'test_phase': storage_ref},
            domain_data={'complex': {'nested': {'data': [1, 2, 3]}}}
        )
        
        # Serialize with pickle
        pickled = pickle.dumps(complex_state)
        restored = pickle.loads(pickled)
        
        # Verify restoration
        assert restored.workflow_id == complex_state.workflow_id
        assert restored.domain == complex_state.domain
        assert 'test_phase' in restored.phase_outputs
        assert restored.phase_outputs['test_phase'].key == storage_ref.key
        assert restored.domain_data['complex']['nested']['data'] == [1, 2, 3]
    
    def test_pydantic_graph_persistence_pattern(self):
        """Test state persistence pattern for pydantic_graph integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            persistence_file = Path(temp_dir) / "state.json"
            
            # Create state snapshots
            state_snapshots = [
                self.state,
                replace(self.state, current_node='node2'),
                replace(self.state, completed_phases={'test_phase'})
            ]
            
            # Simulate state persistence during graph execution
            # Convert states to serializable format
            serializable_snapshots = []
            for i, state in enumerate(state_snapshots):
                # Extract serializable parts of state
                snapshot = {
                    'workflow_id': state.workflow_id,
                    'domain': state.domain,
                    'current_phase': state.current_phase,
                    'current_node': state.current_node,
                    'completed_phases': list(state.completed_phases),
                    'domain_data': state.domain_data,
                    'quality_scores': state.quality_scores,
                    'snapshot_id': f'snapshot_{i}'
                }
                serializable_snapshots.append(snapshot)
            
            # Save to file
            with open(persistence_file, 'w') as f:
                json.dump(serializable_snapshots, f)
            
            # Verify persistence file exists
            assert persistence_file.exists()
            
            # Verify we can restore
            with open(persistence_file, 'r') as f:
                restored = json.load(f)
            
            assert len(restored) == 3
            assert restored[0]['workflow_id'] == self.state.workflow_id
    
    def test_state_memory_efficiency(self):
        """Test memory efficiency of state objects."""
        import sys
        
        # Create multiple state versions
        states = []
        base_state = self.state
        
        for i in range(100):
            new_state = replace(
                base_state,
                domain_data={**base_state.domain_data, f'iteration_{i}': f'data_{i}'}
            )
            states.append(new_state)
        
        # Memory usage should scale reasonably
        total_size = sum(sys.getsizeof(state) for state in states)
        avg_size = total_size / len(states)
        
        # Each state should be relatively small (less than 10KB)
        assert avg_size < 10 * 1024, f"Average state size {avg_size} bytes too large"
    
    def test_large_state_handling(self):
        """Test handling of states with large data."""
        # Create state with large domain data
        large_data = {
            'generated_code': 'x' * 50000,  # 50KB of code
            'analysis_results': list(range(5000)),  # 5000 items
            'dependencies': {f'dep_{i}': f'version_{i}' for i in range(1000)}  # 1000 deps
        }
        
        large_state = replace(self.state, domain_data=large_data)
        
        # State should handle large data without issues
        assert len(large_state.domain_data['generated_code']) == 50000
        assert len(large_state.domain_data['analysis_results']) == 5000
        assert len(large_state.domain_data['dependencies']) == 1000
        
        # Operations should still work
        updated_large_state = replace(
            large_state,
            quality_scores={'test_phase': 0.92}
        )
        
        assert updated_large_state.quality_scores['test_phase'] == 0.92
        # Original large data should be preserved
        assert len(updated_large_state.domain_data['generated_code']) == 50000


class TestStateEvolutionPatterns:
    """Test common state evolution patterns during workflow execution."""
    
    def setup_method(self):
        """Set up workflow for state evolution testing."""
        self.workflow_def = WorkflowDefinition(
            domain='evolution_test',
            phases={},
            phase_sequence=['phase1', 'phase2', 'phase3'],
            node_configs={}
        )
        
        self.initial_state = WorkflowState(
            workflow_def=self.workflow_def,
            workflow_id='evolution-test',
            domain='evolution_test',
            current_phase='phase1',
            current_node='node1'
        )
    
    def test_typical_workflow_evolution(self):
        """Test typical state evolution through workflow."""
        # Phase 1: Initial processing
        phase1_state = replace(
            self.initial_state,
            domain_data={'input': 'user_request', 'phase1_result': 'processed'},
            quality_scores={'phase1': 0.85}
        )
        
        # Phase 1 completion
        phase1_complete = replace(
            phase1_state,
            completed_phases={'phase1'},
            current_phase='phase2',
            current_node='node2_1',
            phase_outputs={
                'phase1': StorageRef(
                    storage_type=StorageType.KV,
                    key='workflow/evolution-test/phase1',
                    created_at=datetime.now()
                )
            }
        )
        
        # Phase 2: Build on phase 1
        phase2_state = replace(
            phase1_complete,
            domain_data={
                **phase1_complete.domain_data,
                'phase2_result': 'enhanced_processing'
            },
            quality_scores={
                **phase1_complete.quality_scores,
                'phase2': 0.78  # Below threshold
            }
        )
        
        # Phase 2 refinement
        phase2_refined = replace(
            phase2_state,
            refinement_count={'phase2': 1},
            refinement_history={
                'phase2': [
                    RefinementRecord(
                        iteration=1,
                        timestamp=datetime.now(),
                        previous_score=0.78,
                        new_score=0.87,
                        feedback="Improved processing logic",
                        changes_made=["Enhanced algorithm", "Added validation"]
                    )
                ]
            },
            quality_scores={
                **phase2_state.quality_scores,
                'phase2': 0.87  # Improved
            }
        )
        
        # Verify evolution
        assert len(phase2_refined.completed_phases) == 1
        assert phase2_refined.quality_scores['phase2'] == 0.87
        assert phase2_refined.refinement_count['phase2'] == 1
        assert len(phase2_refined.domain_data) == 3  # input, phase1_result, phase2_result
    
    def test_error_recovery_pattern(self):
        """Test state evolution during error recovery."""
        # Initial failure state
        error_state = replace(
            self.initial_state,
            domain_data={
                'error': 'Network timeout',
                'error_node': 'api_call',
                'error_count': 1
            }
        )
        
        # Retry attempt state
        retry_state = replace(
            error_state,
            retry_counts={'phase1_api_call_evolution-test': 1},
            domain_data={
                **error_state.domain_data,
                'retry_attempt': 1,
                'last_error': 'Network timeout'
            }
        )
        
        # Recovery success state
        recovery_state = replace(
            retry_state,
            domain_data={
                'success': 'Recovered after retry',
                'retry_successful': True,
                'total_retries': 1,
                'api_result': 'success_data'
            }
        )
        
        # Verify recovery tracking
        assert recovery_state.retry_counts['phase1_api_call_evolution-test'] == 1
        assert recovery_state.domain_data['retry_successful'] == True
        assert 'error' not in recovery_state.domain_data  # Error cleared on success
    
    def test_parallel_processing_state_evolution(self):
        """Test state evolution during parallel processing."""
        # Setup for parallel processing
        parallel_setup = replace(
            self.initial_state,
            iter_items=['item1', 'item2', 'item3', 'item4'],
            iter_index=0
        )
        
        # Parallel processing in progress
        parallel_progress = replace(
            parallel_setup,
            iter_results=['result1', 'result2'],
            iter_index=2
        )
        
        # Parallel processing complete
        parallel_complete = replace(
            parallel_progress,
            iter_results=['result1', 'result2', 'result3', 'result4'],
            iter_index=4,
            domain_data={
                'parallel_results': ['result1', 'result2', 'result3', 'result4'],
                'parallel_count': 4,
                'parallel_complete': True
            }
        )
        
        # Verify parallel evolution
        assert len(parallel_complete.iter_results) == 4
        assert parallel_complete.iter_index == 4
        assert parallel_complete.domain_data['parallel_count'] == 4
    
    def test_complex_nested_evolution(self):
        """Test complex state evolution with multiple concerns."""
        # Build complex state step by step
        current_state = self.initial_state
        
        # Add domain data
        current_state = replace(
            current_state,
            domain_data={'step': 1, 'complexity': 'high'}
        )
        
        # Add storage references
        storage_ref = StorageRef(
            storage_type=StorageType.KV,
            key='complex/data',
            created_at=datetime.now()
        )
        current_state = current_state.with_storage_ref('complex_data', storage_ref)
        
        # Add quality tracking
        current_state = replace(
            current_state,
            quality_scores={'phase1': 0.75}  # Below threshold
        )
        
        # Add refinement
        refinement = RefinementRecord(
            iteration=1,
            timestamp=datetime.now(),
            previous_score=0.75,
            new_score=0.89,
            feedback="Complex refinement",
            changes_made=["Multiple improvements"]
        )
        current_state = replace(
            current_state,
            refinement_count={'phase1': 1},
            refinement_history={'phase1': [refinement]},
            quality_scores={'phase1': 0.89}
        )
        
        # Add token usage
        tokens = TokenUsage(
            prompt_tokens=2500,
            completion_tokens=800,
            total_tokens=3300,
            model='openai:gpt-4o'
        )
        current_state = replace(
            current_state,
            total_token_usage={'phase1': tokens}
        )
        
        # Verify complex state
        assert current_state.domain_data['complexity'] == 'high'
        assert 'complex_data' in current_state.phase_outputs
        assert current_state.quality_scores['phase1'] == 0.89
        assert current_state.refinement_count['phase1'] == 1
        assert current_state.total_token_usage['phase1'].total_tokens == 3300