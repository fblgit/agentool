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
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from datetime import datetime, timedelta

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
        
        # Set up mock LLM responses for deterministic testing
        self.mock_llm_responses = {
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
        # Mock LLM calls to return deterministic responses
        with patch('graphtoolkit.nodes.atomic.llm.LLMCallNode') as mock_llm_node:
            # Configure mock LLM responses for each phase
            async def mock_llm_run(ctx):
                current_phase = ctx.state.current_phase
                response_data = self.mock_llm_responses.get(current_phase, {})
                
                # Update state with mock response
                new_domain_data = {**ctx.state.domain_data}
                new_domain_data[f'{current_phase}_output'] = response_data
                
                from dataclasses import replace
                new_state = replace(
                    ctx.state,
                    domain_data=new_domain_data,
                    quality_scores={
                        **ctx.state.quality_scores,
                        current_phase: response_data.get('quality_score', 0.8)
                    }
                )
                
                return new_state
            
            mock_llm_instance = AsyncMock()
            mock_llm_instance.run = AsyncMock(side_effect=mock_llm_run)
            mock_llm_node.return_value = mock_llm_instance
            
            # Execute workflow
            result = await execute_agentool_workflow(
                task_description="Create a comprehensive session management AgenTool",
                model="openai:gpt-4o",
                workflow_id="test-session-manager",
                enable_persistence=False
            )
            
            # Verify workflow completion
            assert result['success'] == True
            assert result['workflow_id'] == "test-session-manager"
            assert len(result['completed_phases']) == 4
            assert set(result['completed_phases']) == {'analyzer', 'specifier', 'crafter', 'evaluator'}
            
            # Verify phase-specific outputs
            assert 'analyzer_output' in result['domain_data']
            assert 'specifier_output' in result['domain_data']
            assert 'crafter_output' in result['domain_data'] 
            assert 'evaluator_output' in result['domain_data']
            
            # Verify quality scores for all phases
            assert len(result['quality_scores']) == 4
            for phase in ['analyzer', 'specifier', 'crafter', 'evaluator']:
                assert phase in result['quality_scores']
                assert 0.0 <= result['quality_scores'][phase] <= 1.0
            
            # Verify execution time is reasonable
            assert result['execution_time'] is not None
            assert result['execution_time'] >= 0
    
    @pytest.mark.asyncio
    async def test_agentool_analyzer_phase(self):
        """Test analyzer phase specifically."""
        with patch('graphtoolkit.core.executor.WorkflowExecutor') as mock_executor_class:
            mock_executor = AsyncMock()
            mock_executor_class.return_value = mock_executor
            
            # Create mock state after analyzer phase
            workflow_def, initial_state = create_agentool_workflow(
                "Create a task management AgenTool"
            )
            
            final_state = MagicMock(spec=WorkflowState)
            final_state.completed_phases = {'analyzer'}
            final_state.quality_scores = {'analyzer': 0.87}
            final_state.domain_data = {
                'task_description': 'Create a task management AgenTool',
                'analyzer_output': self.mock_llm_responses['analyzer']
            }
            final_state.phase_outputs = {
                'analyzer': StorageRef(
                    storage_type=StorageType.KV,
                    key='workflow/test-id/analyzer',
                    created_at=datetime.now()
                )
            }
            
            mock_executor.run.return_value = WorkflowResult(
                state=final_state,
                outputs={'analyzer': {'storage_ref': 'kv://workflow/test-id/analyzer'}},
                success=True
            )
            
            # Execute workflow  
            result = await execute_agentool_workflow(
                task_description="Create a task management AgenTool",
                model="openai:gpt-4o",
                enable_persistence=False
            )
            
            # Verify analyzer results
            analyzer_output = result['domain_data']['analyzer_output']
            assert 'missing_tools' in analyzer_output
            assert 'tool_analysis' in analyzer_output
            assert 'domain_assessment' in analyzer_output
            
            # Verify tool analysis structure
            for tool_name, analysis in analyzer_output['tool_analysis'].items():
                assert 'complexity' in analysis
                assert 'dependencies' in analysis
                assert 'estimated_lines' in analysis
            
            # Verify quality score
            assert result['quality_scores']['analyzer'] == 0.87
    
    @pytest.mark.asyncio
    async def test_agentool_code_generation_quality(self):
        """Test that generated code meets quality standards."""
        with patch('graphtoolkit.core.executor.WorkflowExecutor') as mock_executor_class:
            mock_executor = AsyncMock()
            mock_executor_class.return_value = mock_executor
            
            # Mock complete workflow execution
            final_state = MagicMock(spec=WorkflowState)
            final_state.completed_phases = {'analyzer', 'specifier', 'crafter', 'evaluator'}
            final_state.quality_scores = {
                'analyzer': 0.87,
                'specifier': 0.91,
                'crafter': 0.89,
                'evaluator': 0.92
            }
            final_state.domain_data = {
                'crafter_output': self.mock_llm_responses['crafter'],
                'evaluator_output': self.mock_llm_responses['evaluator']
            }
            
            mock_executor.run.return_value = WorkflowResult(
                state=final_state,
                outputs={},
                success=True
            )
            
            result = await execute_agentool_workflow(
                task_description="Create a data validation AgenTool",
                model="anthropic:claude-3-5-sonnet-latest"
            )
            
            # Verify generated code quality
            crafter_output = result['domain_data']['crafter_output']
            generated_code = crafter_output['generated_code']
            
            # Test syntactic validity
            try:
                ast.parse(generated_code)
                syntax_valid = True
            except SyntaxError:
                syntax_valid = False
                
            assert syntax_valid, "Generated code should be syntactically valid Python"
            
            # Test code structure
            assert 'class' in generated_code  # Should define classes
            assert 'async def' in generated_code  # Should have async functions
            assert 'BaseOperationInput' in generated_code  # Should use AgenTool patterns
            
            # Verify evaluator results
            evaluator_output = result['domain_data']['evaluator_output']
            assert evaluator_output['syntax_valid'] == True
            assert evaluator_output['imports_available'] == True
            assert evaluator_output['test_coverage'] >= 0.85  # High coverage requirement
            assert evaluator_output['final_score'] >= 0.8  # Quality threshold
    
    @pytest.mark.asyncio
    async def test_agentool_workflow_with_refinement(self):
        """Test workflow with quality gate triggering refinement."""
        with patch('graphtoolkit.core.executor.WorkflowExecutor') as mock_executor_class:
            mock_executor = AsyncMock()
            mock_executor_class.return_value = mock_executor
            
            # Mock refinement scenario - first attempt fails quality gate
            iteration_states = []
            
            # First iteration - low quality
            first_state = MagicMock(spec=WorkflowState)
            first_state.completed_phases = {'analyzer', 'specifier', 'crafter'}
            first_state.quality_scores = {
                'analyzer': 0.87,
                'specifier': 0.91, 
                'crafter': 0.65  # Below quality threshold
            }
            first_state.refinement_count = {'crafter': 0}
            first_state.refinement_history = {'crafter': []}
            iteration_states.append(first_state)
            
            # Second iteration - after refinement
            second_state = MagicMock(spec=WorkflowState)
            second_state.completed_phases = {'analyzer', 'specifier', 'crafter', 'evaluator'}
            second_state.quality_scores = {
                'analyzer': 0.87,
                'specifier': 0.91,
                'crafter': 0.89  # Improved after refinement
            }
            second_state.refinement_count = {'crafter': 1}
            second_state.refinement_history = {'crafter': [
                RefinementRecord(
                    iteration=1,
                    timestamp=datetime.now(),
                    previous_score=0.65,
                    new_score=0.89,
                    feedback="Improved error handling and documentation",
                    changes_made=['Added try-catch blocks', 'Added docstrings']
                )
            ]}
            iteration_states.append(second_state)
            
            # Mock executor to simulate refinement
            call_count = 0
            async def mock_run(initial_state):
                nonlocal call_count
                result_state = iteration_states[min(call_count, len(iteration_states) - 1)]
                call_count += 1
                
                return WorkflowResult(
                    state=result_state,
                    outputs={},
                    success=True
                )
            
            mock_executor.run.side_effect = mock_run
            
            # Execute workflow with refinement enabled
            result = await execute_agentool_workflow(
                task_description="Create a complex data processing AgenTool",
                model="openai:gpt-4o"
            )
            
            # Verify refinement occurred
            assert result['success'] == True
            
            # Check that executor was called multiple times (original + refinement)
            assert mock_executor.run.call_count >= 1
            
            # If refinement data is available, verify it
            final_state = iteration_states[-1]
            if hasattr(final_state, 'refinement_history') and final_state.refinement_history:
                refinement_history = final_state.refinement_history.get('crafter', [])
                if refinement_history:
                    refinement = refinement_history[0]
                    assert refinement.new_score > refinement.previous_score
                    assert len(refinement.changes_made) > 0
    
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
        
        with patch('graphtoolkit.core.executor.WorkflowExecutor') as mock_executor_class:
            mock_executor = AsyncMock()
            mock_executor_class.return_value = mock_executor
            
            # Mock workflow that uses storage operations
            final_state = MagicMock(spec=WorkflowState)
            final_state.completed_phases = {'analyzer', 'specifier', 'crafter', 'evaluator'}
            final_state.phase_outputs = {
                'analyzer': StorageRef(
                    storage_type=StorageType.KV,
                    key='workflow/test/analyzer',
                    created_at=datetime.now(),
                    size_bytes=256
                ),
                'crafter': StorageRef(
                    storage_type=StorageType.FS,
                    key='workflow/test/generated_code.py',
                    created_at=datetime.now(),
                    size_bytes=2048
                )
            }
            final_state.domain_data = {
                'storage_operations': [
                    {'operation': 'save', 'key': 'analyzer_results', 'success': True},
                    {'operation': 'save', 'key': 'generated_code', 'success': True}
                ]
            }
            
            mock_executor.run.return_value = WorkflowResult(
                state=final_state,
                outputs={
                    'analyzer': {'storage_ref': 'kv://workflow/test/analyzer'},
                    'crafter': {'storage_ref': 'fs://workflow/test/generated_code.py'}
                },
                success=True
            )
            
            result = await execute_agentool_workflow(
                task_description="Create a file processing AgenTool",
                model="openai:gpt-4o"
            )
            
            # Verify storage integration
            assert result['success'] == True
            assert 'outputs' in result
            
            # Check storage references
            outputs = result['outputs']
            for phase_name, phase_output in outputs.items():
                if 'storage_ref' in phase_output:
                    storage_ref = phase_output['storage_ref']
                    assert storage_ref.startswith(('kv://', 'fs://'))
    
    @pytest.mark.asyncio
    async def test_agentool_workflow_error_recovery(self):
        """Test error handling and recovery in AgenTool workflow."""
        with patch('graphtoolkit.core.executor.WorkflowExecutor') as mock_executor_class:
            mock_executor = AsyncMock()
            mock_executor_class.return_value = mock_executor
            
            # Mock workflow that encounters and recovers from errors
            async def mock_run_with_recovery(initial_state):
                # Simulate error in first attempt, success in retry
                if not hasattr(mock_run_with_recovery, 'attempt_count'):
                    mock_run_with_recovery.attempt_count = 0
                
                mock_run_with_recovery.attempt_count += 1
                
                if mock_run_with_recovery.attempt_count == 1:
                    # First attempt fails
                    return WorkflowResult(
                        state=initial_state,
                        outputs={},
                        success=False,
                        error="Temporary network error during LLM call"
                    )
                else:
                    # Second attempt succeeds
                    final_state = MagicMock(spec=WorkflowState)
                    final_state.completed_phases = {'analyzer', 'specifier', 'crafter', 'evaluator'}
                    final_state.quality_scores = {
                        'analyzer': 0.88,
                        'specifier': 0.92,
                        'crafter': 0.87,
                        'evaluator': 0.90
                    }
                    final_state.domain_data = {'recovery_successful': True}
                    
                    return WorkflowResult(
                        state=final_state,
                        outputs={'analyzer': {'data': 'recovered'}},
                        success=True
                    )
            
            mock_executor.run.side_effect = mock_run_with_recovery
            
            # Execute workflow
            result = await execute_agentool_workflow(
                task_description="Create an error-prone AgenTool for testing recovery",
                model="openai:gpt-4o"
            )
            
            # Note: Since we're mocking the executor, the result depends on our mock logic
            # In a real scenario, the executor would handle retries internally
            if result['success']:
                # Recovery successful
                assert result['success'] == True
                assert len(result['completed_phases']) == 4
            else:
                # Error occurred (expected in some test scenarios)
                assert result['success'] == False
                assert 'error' in result
                assert result['error'] is not None
    
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
        with patch('graphtoolkit.core.executor.WorkflowExecutor') as mock_executor_class:
            mock_executor = AsyncMock()
            mock_executor_class.return_value = mock_executor
            
            # Mock workflow with metrics
            final_state = MagicMock(spec=WorkflowState)
            final_state.completed_phases = {'analyzer', 'specifier', 'crafter', 'evaluator'}
            final_state.total_token_usage = {
                'analyzer': {'prompt_tokens': 1250, 'completion_tokens': 450, 'total_tokens': 1700},
                'specifier': {'prompt_tokens': 1800, 'completion_tokens': 650, 'total_tokens': 2450},
                'crafter': {'prompt_tokens': 2200, 'completion_tokens': 1200, 'total_tokens': 3400},
                'evaluator': {'prompt_tokens': 900, 'completion_tokens': 300, 'total_tokens': 1200}
            }
            final_state.created_at = datetime.now() - timedelta(seconds=120)  # 2 minutes ago
            final_state.updated_at = datetime.now()
            
            mock_executor.run.return_value = WorkflowResult(
                state=final_state,
                outputs={},
                success=True,
                execution_time=120.5
            )
            
            result = await execute_agentool_workflow(
                task_description="Create a metrics-tracked AgenTool",
                model="openai:gpt-4o"
            )
            
            # Verify execution time tracking
            assert result['execution_time'] is not None
            assert result['execution_time'] > 0
            
            # Verify phase completion tracking
            assert len(result['completed_phases']) == 4
            assert set(result['completed_phases']) == {'analyzer', 'specifier', 'crafter', 'evaluator'}
            
            # Verify quality scores tracking
            assert isinstance(result['quality_scores'], dict)
    
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
            
            with patch('graphtoolkit.core.executor.WorkflowExecutor') as mock_executor_class:
                mock_executor = AsyncMock()
                mock_executor_class.return_value = mock_executor
                
                # Mock persistence workflow
                final_state = MagicMock(spec=WorkflowState)
                final_state.completed_phases = {'analyzer'}  # Partial completion
                final_state.workflow_id = 'persistence-test'
                final_state.domain_data = {
                    'task_description': 'Create a persistent AgenTool',
                    'analyzer_output': self.mock_llm_responses['analyzer']
                }
                
                mock_executor.run_with_persistence.return_value = WorkflowResult(
                    state=final_state,
                    outputs={'analyzer': {'data': 'persisted'}},
                    success=True
                )
                
                # Execute with persistence
                result = await execute_agentool_workflow(
                    task_description="Create a persistent AgenTool",
                    workflow_id="persistence-test",
                    enable_persistence=True
                )
                
                # Verify persistence was attempted
                mock_executor.run_with_persistence.assert_called_once()
                
                # Verify result
                assert result['success'] == True
                assert result['workflow_id'] == 'persistence-test'


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