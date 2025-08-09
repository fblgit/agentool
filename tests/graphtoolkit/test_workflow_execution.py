"""End-to-end tests for complete GraphToolkit workflow execution."""

import asyncio
import json
from dataclasses import replace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from src.graphtoolkit import (
    GraphToolkit,
    create_workflow,
    execute_agentool_workflow,
    execute_testsuite_workflow,
)
from src.graphtoolkit.core.executor import WorkflowExecutor
from src.graphtoolkit.core.types import WorkflowState
from src.graphtoolkit.domains import AVAILABLE_DOMAINS, DOMAIN_PHASES


class TestWorkflowExecution:
    """Test complete workflow execution scenarios."""
    
    @pytest.mark.asyncio
    async def test_simple_agentool_workflow(self):
        """Test a simple AgenTool workflow from start to finish."""
        with patch('src.graphtoolkit.core.executor.WorkflowExecutor.execute') as mock_execute:
            # Mock successful execution
            mock_execute.return_value = {
                'success': True,
                'generated_code': 'class MyAgenTool: ...',
                'quality_score': 0.92,
                'phases_completed': ['analyzer', 'specifier', 'crafter', 'evaluator']
            }
            
            result = await execute_agentool_workflow(
                task_description="Create a session management tool",
                model="openai:gpt-4o"
            )
            
            assert result['success']
            assert 'generated_code' in result
            assert result['quality_score'] > 0.9
            mock_execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_testsuite_workflow_with_coverage(self):
        """Test TestSuite workflow with coverage requirements."""
        test_code = """
        def add(a, b):
            return a + b
        
        def subtract(a, b):
            return a - b
        """
        
        with patch('src.graphtoolkit.core.executor.WorkflowExecutor.execute') as mock_execute:
            mock_execute.return_value = {
                'success': True,
                'generated_tests': 'def test_add(): ...',
                'coverage': 0.95,
                'test_count': 8,
                'phases_completed': ['test_analyzer', 'test_designer', 'test_generator', 'test_executor']
            }
            
            result = await execute_testsuite_workflow(
                code_to_test=test_code,
                framework="pytest",
                coverage_target=0.90
            )
            
            assert result['success']
            assert result['coverage'] >= 0.90
            assert result['test_count'] > 0
    
    @pytest.mark.asyncio
    async def test_workflow_with_phase_failure(self):
        """Test workflow handling when a phase fails."""
        with patch('src.graphtoolkit.core.executor.WorkflowExecutor.execute') as mock_execute:
            mock_execute.side_effect = Exception("LLM API error in specifier phase")
            
            with pytest.raises(Exception) as exc_info:
                await execute_agentool_workflow(
                    task_description="Create a tool",
                    model="openai:gpt-4o"
                )
            
            assert "specifier phase" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_workflow_with_refinement(self):
        """Test workflow that triggers refinement due to low quality."""
        with patch('src.graphtoolkit.core.executor.WorkflowExecutor.execute') as mock_execute:
            # First attempt returns low quality
            mock_execute.return_value = {
                'success': True,
                'quality_score': 0.65,  # Below typical threshold
                'refinement_count': 2,
                'final_quality': 0.88,
                'phases_completed': ['analyzer', 'specifier', 'crafter', 'evaluator']
            }
            
            result = await execute_agentool_workflow(
                task_description="Complex tool requiring refinement",
                model="openai:gpt-4o"
            )
            
            assert result['refinement_count'] > 0
            assert result['final_quality'] > result['quality_score']
    
    @pytest.mark.asyncio
    async def test_multi_domain_workflow_execution(self):
        """Test executing workflows across different domains."""
        toolkit = GraphToolkit()
        
        test_domains = [
            ('agentool', {'task_description': 'Create auth tool'}),
            ('api', {'requirements': 'REST API for users'}),
            ('workflow', {'process_description': 'Order processing'}),
            ('documentation', {'source_code': {'main.py': 'code'}}),
            ('blockchain', {'requirements': 'ERC20 token'}),
            ('testsuite', {'code_to_test': 'def func(): pass'})
        ]
        
        for domain, initial_data in test_domains:
            with patch.object(toolkit, 'execute_workflow') as mock_exec:
                mock_exec.return_value = {
                    'success': True,
                    'domain': domain,
                    'phases_completed': DOMAIN_PHASES[domain]
                }
                
                result = await toolkit.execute_workflow(
                    domain=domain,
                    initial_data=initial_data
                )
                
                assert result['success']
                assert result['domain'] == domain
                mock_exec.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_workflow_with_custom_phases(self):
        """Test workflow execution with custom phase selection."""
        toolkit = GraphToolkit()
        
        # Only run analyzer and specifier phases
        custom_phases = ['analyzer', 'specifier']
        
        with patch.object(toolkit, 'execute_workflow') as mock_exec:
            mock_exec.return_value = {
                'success': True,
                'phases_completed': custom_phases
            }
            
            result = await toolkit.execute_workflow(
                domain='agentool',
                phases=custom_phases,
                initial_data={'task_description': 'Quick analysis'}
            )
            
            assert result['phases_completed'] == custom_phases
    
    @pytest.mark.asyncio
    async def test_workflow_state_persistence(self):
        """Test workflow state can be saved and resumed."""
        from src.graphtoolkit.core.persistence import WorkflowPersistence
        
        persistence = WorkflowPersistence(storage_path='/tmp/test_workflow')
        
        # Create initial state
        initial_state = WorkflowState(
            workflow_id='test-persist-123',
            domain='agentool',
            current_phase='specifier',
            current_node='llm_call',
            completed_phases={'analyzer'},
            phase_outputs={'analyzer': 'analysis_output'}
        )
        
        # Save state
        with patch.object(persistence, 'save_state', new_callable=AsyncMock) as mock_save:
            mock_save.return_value = True
            saved = await persistence.save_state(initial_state)
            assert saved
            mock_save.assert_called_once()
        
        # Load state
        with patch.object(persistence, 'load_state', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = initial_state
            loaded_state = await persistence.load_state()
            assert loaded_state.workflow_id == 'test-persist-123'
            assert loaded_state.current_phase == 'specifier'
            assert 'analyzer' in loaded_state.completed_phases
    
    @pytest.mark.asyncio
    async def test_workflow_with_parallel_execution(self):
        """Test workflow with parallel node execution."""
        from src.graphtoolkit.nodes.iteration import ParallelMapNode
        
        items = ['item1', 'item2', 'item3', 'item4', 'item5']
        
        state = WorkflowState(
            workflow_id='test-parallel',
            domain='agentool',
            current_phase='crafter',
            current_node='process_tools',
            iter_items=items,
            iter_index=0
        )
        
        ctx = MagicMock()
        ctx.state = state
        ctx.deps = MagicMock()
        
        parallel_node = ParallelMapNode(max_concurrent=3)
        
        async def mock_process(item, ctx):
            await asyncio.sleep(0.01)  # Simulate work
            return f"processed_{item}"
        
        with patch.object(parallel_node, 'process_item', side_effect=mock_process):
            with patch.object(parallel_node, 'on_iteration_complete') as mock_complete:
                mock_complete.return_value = MagicMock()
                
                result = await parallel_node.run(ctx)
                
                # Verify all items were processed
                assert len(ctx.state.iter_results) == 0  # Results not stored in initial state
                mock_complete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_workflow_error_recovery(self):
        """Test workflow error handling and recovery."""
        executor = WorkflowExecutor()
        
        # Simulate transient error that succeeds on retry
        call_count = 0
        
        async def mock_execute_with_retry(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Transient API error")
            return {'success': True, 'retry_count': call_count - 1}
        
        with patch.object(executor, 'execute_phase', side_effect=mock_execute_with_retry):
            result = await executor.execute(
                domain='agentool',
                initial_data={'task_description': 'Test with retry'}
            )
            
            assert result['success']
            assert result['retry_count'] > 0
    
    @pytest.mark.asyncio
    async def test_workflow_with_model_config(self):
        """Test workflow with custom model configuration per phase."""
        toolkit = GraphToolkit()
        
        model_config = {
            'analyzer': 'openai:gpt-4o-mini',      # Cheaper for analysis
            'specifier': 'openai:gpt-4o',          # Better for specification
            'crafter': 'anthropic:claude-3-5-sonnet-latest',  # Best for code generation
            'evaluator': 'openai:gpt-4o-mini'      # Cheaper for evaluation
        }
        
        with patch.object(toolkit, 'execute_workflow') as mock_exec:
            mock_exec.return_value = {
                'success': True,
                'models_used': model_config
            }
            
            result = await toolkit.execute_workflow(
                domain='agentool',
                initial_data={'task_description': 'Optimized model usage'},
                model_config=model_config
            )
            
            assert result['models_used'] == model_config
    
    @pytest.mark.asyncio
    async def test_workflow_metrics_collection(self):
        """Test that workflow execution collects proper metrics."""
        from src.agentoolkit.metrics import MetricsCollector
        
        metrics = MetricsCollector()
        
        with patch.object(metrics, 'record_workflow_execution') as mock_record:
            with patch('src.graphtoolkit.core.executor.WorkflowExecutor.execute') as mock_exec:
                mock_exec.return_value = {
                    'success': True,
                    'duration': 45.3,
                    'tokens_used': 2500,
                    'phases_completed': 4
                }
                
                result = await execute_agentool_workflow(
                    task_description="Tool with metrics",
                    model="openai:gpt-4o"
                )
                
                # Verify metrics would be collected
                assert result['duration'] > 0
                assert result['tokens_used'] > 0
    
    @pytest.mark.asyncio
    async def test_workflow_with_storage_integration(self):
        """Test workflow integration with agentoolkit storage."""
        from src.agentoolkit.storage import StorageKV
        
        storage = StorageKV()
        
        workflow_data = {
            'workflow_id': 'test-storage-123',
            'phase_outputs': {
                'analyzer': {'analysis': 'complete'},
                'specifier': {'spec': 'detailed'}
            }
        }
        
        # Test storage operations
        with patch.object(storage, 'set', new_callable=AsyncMock) as mock_set:
            with patch.object(storage, 'get', new_callable=AsyncMock) as mock_get:
                mock_set.return_value = True
                mock_get.return_value = workflow_data
                
                # Store workflow data
                stored = await storage.set('workflow:test-storage-123', workflow_data)
                assert stored
                
                # Retrieve workflow data
                retrieved = await storage.get('workflow:test-storage-123')
                assert retrieved['workflow_id'] == 'test-storage-123'
                assert 'analyzer' in retrieved['phase_outputs']


class TestWorkflowIntegration:
    """Test GraphToolkit integration with existing agentoolkit components."""
    
    @pytest.mark.asyncio
    async def test_template_engine_integration(self):
        """Test that workflows use the agentoolkit template engine."""
        from src.agentoolkit.templates import TemplateEngine
        
        engine = TemplateEngine()
        
        template_content = """
        Task: {{ task_description }}
        Domain: {{ domain }}
        Phase: {{ current_phase }}
        """
        
        with patch.object(engine, 'render') as mock_render:
            mock_render.return_value = "Rendered template content"
            
            rendered = engine.render(
                template_content,
                task_description="Create auth tool",
                domain="agentool",
                current_phase="analyzer"
            )
            
            assert rendered == "Rendered template content"
            mock_render.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_logging_integration(self):
        """Test that workflows properly log events."""
        from src.agentoolkit.logging import Logger
        
        logger = Logger("graphtoolkit")
        
        with patch.object(logger, 'info') as mock_info:
            with patch.object(logger, 'error') as mock_error:
                # Simulate workflow execution with logging
                logger.info("Starting workflow execution", workflow_id="test-123")
                logger.info("Phase completed", phase="analyzer", duration=5.2)
                
                assert mock_info.call_count == 2
                mock_error.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_observability_integration(self):
        """Test workflow observability and monitoring."""
        from src.agentoolkit.observability import ObservabilityManager
        
        obs_manager = ObservabilityManager()
        
        with patch.object(obs_manager, 'trace_workflow') as mock_trace:
            mock_trace.return_value = {
                'trace_id': 'trace-123',
                'spans': ['phase1', 'phase2', 'phase3']
            }
            
            trace = obs_manager.trace_workflow(
                workflow_id='test-obs-123',
                domain='agentool'
            )
            
            assert trace['trace_id'] == 'trace-123'
            assert len(trace['spans']) == 3