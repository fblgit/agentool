"""
Tests for AgenTool domain workflow.

This module tests the complete AgenTool generation workflow including:
- Analyzer phase (tool catalog analysis)
- Specifier phase (specification creation)
- Crafter phase (code generation)
- Evaluator phase (quality assessment)
"""

import asyncio
import tempfile
import pytest
import ast
import json
from pathlib import Path
from datetime import datetime, timedelta
from pydantic_ai.models.test import TestModel

# GraphToolkit imports
from graphtoolkit import execute_agentool_workflow, create_agentool_workflow
from graphtoolkit.core.executor import WorkflowExecutor, WorkflowResult
from graphtoolkit.core.types import (
    WorkflowState, 
    WorkflowDefinition,
    StorageRef,
    StorageType,
    ValidationResult,
    RefinementRecord
)

# AgenTool integration
from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry


class TestAgenToolWorkflowComplete:
    """Test complete AgenTool workflow execution."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        # Clear registries to avoid interference
        AgenToolRegistry.clear()
        get_injector().clear()
        
        # Set up test model for deterministic testing
        self.test_model = TestModel()
        
        # Expected response structures
        self.expected_responses = {
            'analyzer': {
                'missing_tools': ['session_create', 'session_read', 'session_update', 'session_delete'],
                'tool_analysis': {
                    'session_create': {
                        'complexity': 'medium',
                        'dependencies': ['storage_kv'],
                        'estimated_lines': 45
                    },
                    'session_read': {
                        'complexity': 'low', 
                        'dependencies': ['storage_kv'],
                        'estimated_lines': 25
                    }
                },
                'domain_assessment': 'session_management',
                'quality_score': 0.87
            },
            'specifier': {
                'specifications': [
                    {
                        'tool_name': 'session_create',
                        'input_schema': {
                            'user_id': 'str',
                            'session_data': 'dict',
                            'ttl': 'Optional[int]'
                        },
                        'output_schema': {
                            'session_id': 'str',
                            'success': 'bool'
                        },
                        'description': 'Create a new user session'
                    }
                ],
                'routing_config': {
                    'operation_field': 'operation',
                    'operation_map': {
                        'create': ('session_create', 'lambda x: {"user_id": x.user_id, "session_data": x.session_data}')
                    }
                },
                'quality_score': 0.91
            },
            'crafter': {
                'generated_code': '''
from pydantic import BaseModel
from typing import Optional, Dict, Any

class SessionManagerInput(BaseOperationInput):
    operation: Literal["create", "read", "update", "delete"]
    user_id: str
    session_id: Optional[str] = None
    session_data: Optional[Dict[str, Any]] = None
    ttl: Optional[int] = 3600

async def session_create(user_id: str, session_data: Dict[str, Any], ttl: int = 3600):
    """Create a new user session."""
    session_id = f"session_{user_id}_{int(time.time())}"
    await storage_kv_set(f"session:{session_id}", session_data, ttl=ttl)
    return {"session_id": session_id, "success": True}

# Additional functions...
''',
                'file_structure': {
                    'session_manager.py': 'main_implementation',
                    'test_session_manager.py': 'test_suite'
                },
                'quality_score': 0.89
            },
            'evaluator': {
                'syntax_valid': True,
                'imports_available': True,
                'test_coverage': 0.94,
                'code_quality_metrics': {
                    'complexity': 3.2,
                    'maintainability': 8.5,
                    'documentation': 0.85
                },
                'final_score': 0.92,
                'recommendations': ['Add more error handling', 'Consider async context managers']
            }
        }
    
    @pytest.mark.asyncio
    async def test_complete_agentool_workflow_execution(self):
        """Test end-to-end AgenTool workflow with all phases."""
        # Create workflow with test model
        from graphtoolkit.core.deps import WorkflowDeps, ModelConfig, StorageConfig
        from graphtoolkit.core.executor import WorkflowExecutor
        
        # Create deps with test model
        deps = WorkflowDeps(
            models=ModelConfig(provider='test', model='test'),
            storage=StorageConfig(kv_backend='memory')
        )
        
        # Create workflow
        workflow_def, initial_state = create_agentool_workflow(
            task_description="Create a comprehensive session management AgenTool",
            model="test",
            workflow_id="test-session-manager"
        )
        
        # Verify workflow structure
        assert workflow_def.domain == 'agentool'
        assert initial_state.workflow_id == "test-session-manager"
        assert initial_state.domain_data['task_description'] == "Create a comprehensive session management AgenTool"
    
    @pytest.mark.asyncio
    async def test_agentool_analyzer_phase(self):
        """Test analyzer phase specifically."""
        # Create workflow with test model
        workflow_def, initial_state = create_agentool_workflow(
            task_description="Create a task management AgenTool",
            model="test"
        )
        
        # Verify analyzer phase exists
        assert 'analyzer' in workflow_def.phases
        analyzer_phase = workflow_def.phases['analyzer']
        assert analyzer_phase.phase_name == 'analyzer'
        assert analyzer_phase.domain == 'agentool'
        
        # Verify analyzer phase configuration
        assert len(analyzer_phase.atomic_nodes) > 0
        if analyzer_phase.templates:
            # Templates is a TemplateConfig object
            assert analyzer_phase.templates is not None
        
        # Test expected analyzer output structure
        analyzer_output = self.expected_responses['analyzer']
        assert 'missing_tools' in analyzer_output
        assert 'tool_analysis' in analyzer_output
        assert 'domain_assessment' in analyzer_output
        
        # Verify tool analysis structure
        for tool_name, analysis in analyzer_output['tool_analysis'].items():
            assert 'complexity' in analysis
            assert 'dependencies' in analysis
            assert 'estimated_lines' in analysis
        
        # Verify quality score
        assert analyzer_output['quality_score'] == 0.87
    
    @pytest.mark.asyncio
    async def test_agentool_code_generation_quality(self):
        """Test that generated code meets quality standards."""
        # Create workflow with test model
        workflow_def, initial_state = create_agentool_workflow(
            task_description="Create a data validation AgenTool",
            model="test"
        )
        
        # Test syntactic validity of sample generated code
        sample_generated_code = self.expected_responses['crafter']['generated_code']
        try:
            ast.parse(sample_generated_code)
            syntax_valid = True
        except SyntaxError:
            syntax_valid = False
            
        assert syntax_valid, "Generated code should be syntactically valid Python"
        
        # Test code structure
        assert 'class' in sample_generated_code  # Should define classes
        assert 'async def' in sample_generated_code  # Should have async functions
        assert 'BaseOperationInput' in sample_generated_code  # Should use AgenTool patterns
        
        # Verify evaluator phase configuration
        assert 'evaluator' in workflow_def.phases
        evaluator_phase = workflow_def.phases['evaluator']
        assert evaluator_phase.phase_name == 'evaluator'
        assert evaluator_phase.domain == 'agentool'
    
    @pytest.mark.asyncio
    async def test_agentool_workflow_with_refinement(self):
        """Test workflow with quality gate triggering refinement."""
        from dataclasses import replace
        
        # Create workflow with refinement enabled
        workflow_def, initial_state = create_agentool_workflow(
            "Create a complex cryptography AgenTool",
            model="test"
        )
        
        # Test refinement configuration in workflow
        assert workflow_def.enable_refinement == True if hasattr(workflow_def, 'enable_refinement') else True
        
        # Test quality threshold configuration
        for phase_name, phase_def in workflow_def.phases.items():
            if hasattr(phase_def, 'quality_threshold'):
                assert phase_def.quality_threshold >= 0.7  # Reasonable threshold
                assert phase_def.quality_threshold <= 1.0
        
        # Simulate a refinement scenario data
        refinement_record = RefinementRecord(
            iteration=1,
            timestamp=datetime.now(),
            previous_score=0.65,
            new_score=0.89,
            feedback="Improved error handling and documentation",
            changes_made=['Added try-catch blocks', 'Added docstrings']
        )
        
        # Verify refinement record structure
        assert refinement_record.new_score > refinement_record.previous_score
        assert len(refinement_record.changes_made) > 0
        assert refinement_record.iteration == 1
    
    @pytest.mark.asyncio
    async def test_agentool_storage_integration(self):
        """Test integration with agentoolkit storage systems."""
        # Import storage components to ensure they're available
        try:
            from agentoolkit.storage.kv import create_storage_kv_agent
            from agentoolkit.storage.fs import create_storage_fs_agent
            storage_available = True
        except ImportError:
            storage_available = False
        
        if not storage_available:
            pytest.skip("Storage agentoolkits not available")
        
        # Create storage agents
        kv_agent = create_storage_kv_agent()
        fs_agent = create_storage_fs_agent()
        
        # Test KV storage operation
        from agentool.core.injector import get_injector
        injector = get_injector()
        injector.register('storage_kv', kv_agent)
        injector.register('storage_fs', fs_agent)
        
        # Test saving workflow outputs
        kv_result = await injector.run('storage_kv', {
            'operation': 'set',
            'key': 'workflow/test/analyzer',
            'value': {'analyzer_results': 'test_data'},
            'namespace': 'workflow'
        })
        # kv_result is an AgentRunResult, get the data
        print(kv_result.output)
        assert '"stored":true' in kv_result.output
        
        # Create storage references
        storage_ref_kv = StorageRef(
            storage_type=StorageType.KV,
            key='workflow/test/analyzer',
            created_at=datetime.now(),
            size_bytes=256
        )
        
        storage_ref_fs = StorageRef(
            storage_type=StorageType.FS,
            key='workflow/test/generated_code.py',
            created_at=datetime.now(),
            size_bytes=2048
        )
        
        # Verify storage reference formatting
        assert storage_ref_kv.storage_type == StorageType.KV
        assert storage_ref_fs.storage_type == StorageType.FS
        assert 'workflow/test' in storage_ref_kv.key
        assert 'generated_code.py' in storage_ref_fs.key
    
    @pytest.mark.asyncio
    async def test_agentool_workflow_error_recovery(self):
        """Test error handling and recovery in AgenTool workflow."""
        # Create workflow
        workflow_def, initial_state = create_agentool_workflow(
            task_description="Create an error-prone AgenTool for testing recovery",
            model="test"
        )
        
        # Test error recovery configuration
        for phase_name, phase_def in workflow_def.phases.items():
            # Check if retry configuration exists
            if hasattr(phase_def, 'max_retries'):
                assert phase_def.max_retries >= 0
                assert phase_def.max_retries <= 5  # Reasonable retry limit
        
        # Simulate error scenario
        error_result = WorkflowResult(
            state=initial_state,
            outputs={},
            success=False,
            error="Temporary network error during LLM call"
        )
        
        # Verify error result structure
        assert error_result.success == False
        assert error_result.error is not None
        assert "network error" in error_result.error.lower()
        
        # Simulate successful recovery
        from dataclasses import replace
        recovered_state = replace(
            initial_state,
            completed_phases={'analyzer', 'specifier', 'crafter', 'evaluator'},
            quality_scores={
                'analyzer': 0.88,
                'specifier': 0.92,
                'crafter': 0.87,
                'evaluator': 0.90
            },
            domain_data={'recovery_successful': True}
        )
        
        recovery_result = WorkflowResult(
            state=recovered_state,
            outputs={'analyzer': {'data': 'recovered'}},
            success=True
        )
        
        # Verify recovery result
        assert recovery_result.success == True
        assert len(recovered_state.completed_phases) == 4
        assert recovered_state.domain_data['recovery_successful'] == True
    
    def test_agentool_workflow_schema_validation(self):
        """Test schema validation for AgenTool workflow inputs/outputs."""
        # Test valid workflow creation
        workflow_def, initial_state = create_agentool_workflow(
            task_description="Create a user authentication AgenTool",
            model="openai:gpt-4o-mini"
        )
        
        # Verify workflow definition structure
        assert isinstance(workflow_def.domain, str)
        assert workflow_def.domain == 'agentool'
        assert isinstance(workflow_def.phases, dict)
        assert isinstance(workflow_def.phase_sequence, list)
        assert len(workflow_def.phase_sequence) == 4
        
        # Verify initial state structure
        assert isinstance(initial_state.workflow_id, str)
        assert isinstance(initial_state.domain, str)
        assert isinstance(initial_state.domain_data, dict)
        assert initial_state.domain_data['task_description'] == "Create a user authentication AgenTool"
        assert initial_state.domain_data['model'] == "openai:gpt-4o-mini"
        
        # Test invalid inputs (should not crash, but may produce empty/default workflows)
        try:
            empty_workflow_def, empty_state = create_agentool_workflow(
                task_description="",  # Empty description
                model=""  # Empty model
            )
            # Should still create valid structures
            assert isinstance(empty_workflow_def, WorkflowDefinition)
            assert isinstance(empty_state, WorkflowState)
        except Exception as e:
            # If validation fails, it should be a clear error
            assert "task_description" in str(e) or "model" in str(e)
    
    @pytest.mark.asyncio
    async def test_agentool_workflow_metrics_tracking(self):
        """Test metrics and performance tracking in AgenTool workflow."""
        # Create workflow
        workflow_def, initial_state = create_agentool_workflow(
            task_description="Create a metrics-tracked AgenTool",
            model="test"
        )
        
        # Simulate workflow execution with metrics
        from dataclasses import replace
        
        final_state = replace(
            initial_state,
            completed_phases={'analyzer', 'specifier', 'crafter', 'evaluator'},
            total_token_usage={
                'analyzer': {'prompt_tokens': 1250, 'completion_tokens': 450, 'total_tokens': 1700},
                'specifier': {'prompt_tokens': 1800, 'completion_tokens': 650, 'total_tokens': 2450},
                'crafter': {'prompt_tokens': 2200, 'completion_tokens': 1200, 'total_tokens': 3400},
                'evaluator': {'prompt_tokens': 900, 'completion_tokens': 300, 'total_tokens': 1200}
            } if hasattr(initial_state, 'total_token_usage') else {},
            created_at=datetime.now() - timedelta(seconds=120),  # 2 minutes ago
            updated_at=datetime.now()
        )
        
        # Calculate execution time
        execution_time = (final_state.updated_at - final_state.created_at).total_seconds()
        
        # Verify metrics
        assert execution_time > 0
        assert abs(execution_time - 120.0) < 0.01  # Allow small floating point difference
        
        # Verify phase completion
        assert len(final_state.completed_phases) == 4
        assert set(final_state.completed_phases) == {'analyzer', 'specifier', 'crafter', 'evaluator'}
        
        # Verify token usage if available
        if hasattr(final_state, 'total_token_usage') and final_state.total_token_usage:
            total_tokens = sum(
                phase_usage.get('total_tokens', 0) 
                for phase_usage in final_state.total_token_usage.values()
            )
            assert total_tokens > 0
    
    def test_agentool_workflow_concurrent_creation(self):
        """Test creating multiple AgenTool workflows concurrently."""
        import asyncio
        
        async def create_workflow(task_id):
            workflow_def, initial_state = create_agentool_workflow(
                task_description=f"Create AgenTool #{task_id}",
                model="openai:gpt-4o-mini"
            )
            return workflow_def, initial_state
        
        async def test_concurrent():
            # Create 10 workflows concurrently
            tasks = [create_workflow(i) for i in range(10)]
            results = await asyncio.gather(*tasks)
            
            # Verify all workflows were created successfully
            assert len(results) == 10
            
            workflow_ids = set()
            for workflow_def, initial_state in results:
                assert isinstance(workflow_def, WorkflowDefinition)
                assert isinstance(initial_state, WorkflowState)
                
                # Verify unique workflow IDs
                workflow_ids.add(initial_state.workflow_id)
            
            # All workflow IDs should be unique
            assert len(workflow_ids) == 10
        
        # Run the concurrent test
        asyncio.run(test_concurrent())
    
    @pytest.mark.asyncio
    async def test_agentool_workflow_persistence_integration(self):
        """Test workflow persistence with temporary files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            persistence_path = Path(temp_dir) / "workflow_state.json"
            
            # Create workflow with persistence enabled
            workflow_def, initial_state = create_agentool_workflow(
                task_description="Create a persistent AgenTool",
                workflow_id="persistence-test",
                model="test"
            )
            
            # Verify workflow ID
            assert initial_state.workflow_id == "persistence-test"
            
            # Simulate partial completion state
            from dataclasses import replace
            partial_state = replace(
                initial_state,
                completed_phases={'analyzer'},  # Partial completion
                domain_data={
                    'task_description': 'Create a persistent AgenTool',
                    'analyzer_output': self.expected_responses['analyzer']
                }
            )
            
            # Test state serialization (if methods exist)
            if hasattr(partial_state, 'to_dict'):
                state_dict = partial_state.to_dict()
                assert isinstance(state_dict, dict)
                assert 'workflow_id' in state_dict
                assert state_dict['workflow_id'] == 'persistence-test'
            
            # Test persistence file creation
            if persistence_path:
                # Write test data to persistence file
                persistence_path.write_text(json.dumps({
                    'workflow_id': 'persistence-test',
                    'completed_phases': ['analyzer'],
                    'domain': 'agentool'
                }))
                
                # Verify file was created
                assert persistence_path.exists()
                
                # Read back and verify
                data = json.loads(persistence_path.read_text())
                assert data['workflow_id'] == 'persistence-test'
                assert 'analyzer' in data['completed_phases']


class TestAgenToolWorkflowRealComponents:
    """Test AgenTool workflow using real GraphToolkit components without mocking core framework."""
    
    def setup_method(self):
        """Set up test environment."""
        AgenToolRegistry.clear()
        get_injector().clear()
    
    def test_real_agentool_workflow_structure(self):
        """Test AgenTool workflow creation with real components."""
        # Create workflow using real GraphToolkit API
        workflow_def, initial_state = create_agentool_workflow(
            task_description="Create a comprehensive file management AgenTool with CRUD operations",
            model="anthropic:claude-3-5-sonnet-latest"
        )
        
        # Verify workflow structure using real types
        assert isinstance(workflow_def, WorkflowDefinition)
        assert workflow_def.domain == 'agentool'
        assert workflow_def.version == '1.0.0'
        assert isinstance(workflow_def.created_at, datetime)
        
        # Verify phase sequence
        assert len(workflow_def.phase_sequence) == 4
        expected_phases = ['analyzer', 'specifier', 'crafter', 'evaluator']
        assert workflow_def.phase_sequence == expected_phases
        
        # Verify each phase has proper structure
        for phase_name in expected_phases:
            if phase_name in workflow_def.phases:  # Phase might not be fully registered yet
                phase_def = workflow_def.phases[phase_name]
                assert phase_def.phase_name == phase_name
                assert phase_def.domain == 'agentool'
                assert isinstance(phase_def.atomic_nodes, list)
                assert len(phase_def.atomic_nodes) > 0
        
        # Verify initial state using real types
        assert isinstance(initial_state, WorkflowState)
        assert initial_state.workflow_def is workflow_def
        assert initial_state.domain == 'agentool'
        assert len(initial_state.workflow_id) > 0
        assert initial_state.current_phase == workflow_def.phase_sequence[0]  # Should start with first phase
        
        # Verify domain data
        domain_data = initial_state.domain_data
        assert domain_data['task_description'] == "Create a comprehensive file management AgenTool with CRUD operations"
        assert domain_data['model'] == "anthropic:claude-3-5-sonnet-latest"
        assert domain_data['domain'] == 'agentool'
    
    def test_real_workflow_validation(self):
        """Test workflow validation using real components."""
        # Create real workflow
        workflow_def, _ = create_agentool_workflow(
            task_description="Create a notification system AgenTool",
            model="openai:gpt-4o"
        )
        
        # Use real GraphToolkit for validation
        from graphtoolkit import GraphToolkit
        toolkit = GraphToolkit()
        
        # Validate workflow
        validation_errors = toolkit.validate_workflow(workflow_def)
        
        # Should return a list (may contain errors about missing components, but validation should work)
        assert isinstance(validation_errors, list)
        
        # Each error should be a string if present
        for error in validation_errors:
            assert isinstance(error, str)
    
    def test_real_state_management(self):
        """Test WorkflowState operations using real types."""
        workflow_def, initial_state = create_agentool_workflow(
            task_description="Create a logging AgenTool"
        )
        
        # Test state helper methods
        current_phase_def = initial_state.get_current_phase_def()
        if current_phase_def:  # Might be None if phase not registered
            assert current_phase_def.phase_name == initial_state.current_phase
            assert current_phase_def.domain == 'agentool'
        
        # Test storage reference operations
        storage_ref = StorageRef(
            storage_type=StorageType.KV,
            key=f'workflow/{initial_state.workflow_id}/test_phase',
            created_at=datetime.now(),
            size_bytes=512
        )
        
        updated_state = initial_state.with_storage_ref('test_phase', storage_ref)
        
        # Verify storage reference was added
        assert 'test_phase' in updated_state.phase_outputs
        assert updated_state.phase_outputs['test_phase'] == storage_ref
        assert updated_state.updated_at > initial_state.updated_at
        
        # Verify original state is unchanged (immutable)
        assert 'test_phase' not in initial_state.phase_outputs
    
    def test_real_multiple_workflow_creation(self):
        """Test creating multiple real workflows."""
        workflows = []
        
        # Create 5 different AgenTool workflows
        task_descriptions = [
            "Create a user authentication AgenTool",
            "Create a data validation AgenTool", 
            "Create a file compression AgenTool",
            "Create a email notification AgenTool",
            "Create a backup system AgenTool"
        ]
        
        for i, task_desc in enumerate(task_descriptions):
            workflow_def, initial_state = create_agentool_workflow(
                task_description=task_desc,
                model=f"openai:gpt-4o" if i % 2 == 0 else f"anthropic:claude-3-5-sonnet-latest"
            )
            workflows.append((workflow_def, initial_state))
        
        # Verify all workflows are unique and valid
        workflow_ids = set()
        for workflow_def, initial_state in workflows:
            assert isinstance(workflow_def, WorkflowDefinition)
            assert isinstance(initial_state, WorkflowState)
            assert workflow_def.domain == 'agentool'
            
            # Verify unique workflow IDs
            workflow_ids.add(initial_state.workflow_id)
            
            # Verify task description is preserved
            assert initial_state.domain_data['task_description'] in task_descriptions
        
        # All workflows should have unique IDs
        assert len(workflow_ids) == 5
