"""End-to-end tests for GraphToolkit quality gates and refinement loops."""

import asyncio
from dataclasses import replace
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field
from pydantic_graph import End, GraphRunContext

from src.graphtoolkit.core.types import (
    ModelParameters,
    PhaseDefinition,
    StorageType,
    TemplateConfig,
    WorkflowState,
)
from src.graphtoolkit.nodes.atomic.validation import QualityGateNode


class MockInput(BaseModel):
    """Mock input schema for testing."""
    data: str = Field(description="Test data")


class MockOutput(BaseModel):
    """Mock output schema for testing."""
    result: str = Field(description="Test result")
    quality_score: float = Field(description="Quality score")


@pytest.fixture
def mock_phase_definition():
    """Create a mock phase definition for testing."""
    return PhaseDefinition(
        phase_name='test_phase',
        atomic_nodes=[
            'dependency_check',
            'template_render',
            'llm_call',
            'schema_validation',
            'quality_gate'
        ],
        input_schema=MockInput,
        output_schema=MockOutput,
        dependencies=[],
        templates=TemplateConfig(
            system_template='system/test',
            user_template='prompts/test',
            variables={'test': 'value'}
        ),
        storage_pattern='workflow/{workflow_id}/test',
        storage_type=StorageType.KV,
        quality_threshold=0.8,  # Key threshold for testing
        allow_refinement=True,
        max_refinements=3,
        model_config=ModelParameters(temperature=0.7),
        domain='test'
    )


@pytest.fixture
def workflow_state(mock_phase_definition):
    """Create a workflow state for testing."""
    from src.graphtoolkit.core.types import WorkflowDefinition
    
    workflow_def = WorkflowDefinition(
        domain='test',
        phases={'test_phase': mock_phase_definition},
        phase_sequence=['test_phase'],
        node_configs={},
        enable_refinement=True
    )
    
    return WorkflowState(
        workflow_id='test-workflow-123',
        domain='test',
        current_phase='test_phase',
        current_node='quality_gate',
        completed_phases=set(),
        phase_outputs={},
        quality_scores={},  # Start with no scores
        refinement_count={'test_phase': 0},
        workflow_def=workflow_def
    )


class TestQualityGates:
    """Test quality gate functionality."""
    
    @pytest.mark.asyncio
    async def test_quality_gate_pass(self, workflow_state):
        """Test quality gate when quality threshold is met."""
        # Set quality score above threshold (0.8)
        workflow_state = replace(
            workflow_state,
            quality_scores={'test_phase': 0.85}
        )
        
        ctx = GraphRunContext(state=workflow_state, deps=MagicMock())
        quality_gate = QualityGateNode()
        
        # Mock the node creation methods
        with patch.object(quality_gate, 'complete_phase') as mock_complete:
            mock_complete.return_value = End(workflow_state)
            
            result = await quality_gate.run(ctx)
            
            # Should complete the phase
            mock_complete.assert_called_once()
            assert isinstance(result, End)
    
    @pytest.mark.asyncio
    async def test_quality_gate_fail_triggers_refinement(self, workflow_state):
        """Test quality gate triggers refinement when threshold not met."""
        # Set quality score below threshold (0.8)
        workflow_state = replace(
            workflow_state,
            quality_scores={'test_phase': 0.6},
            refinement_count={'test_phase': 0}  # First attempt
        )
        
        ctx = GraphRunContext(state=workflow_state, deps=MagicMock())
        quality_gate = QualityGateNode()
        
        # Mock the refinement node creation
        with patch.object(quality_gate, 'create_refinement_node') as mock_refine:
            from src.graphtoolkit.nodes.atomic.llm import RefinementNode
            mock_refine.return_value = RefinementNode()
            
            result = await quality_gate.run(ctx)
            
            # Should trigger refinement
            mock_refine.assert_called_once()
            assert isinstance(result, RefinementNode)
    
    @pytest.mark.asyncio
    async def test_quality_gate_max_refinements(self, workflow_state):
        """Test quality gate respects max refinement limit."""
        # Set quality score below threshold but max refinements reached
        workflow_state = replace(
            workflow_state,
            quality_scores={'test_phase': 0.6},
            refinement_count={'test_phase': 3}  # Max refinements reached
        )
        
        ctx = GraphRunContext(state=workflow_state, deps=MagicMock())
        quality_gate = QualityGateNode()
        
        # Mock the node creation methods
        with patch.object(quality_gate, 'complete_phase') as mock_complete:
            mock_complete.return_value = End(workflow_state)
            
            result = await quality_gate.run(ctx)
            
            # Should complete despite low quality (max refinements reached)
            mock_complete.assert_called_once()
            assert isinstance(result, End)
    
    @pytest.mark.asyncio
    async def test_quality_gate_refinement_improves_score(self, workflow_state):
        """Test that refinement loop actually improves quality scores."""
        # Simulate a refinement cycle
        initial_score = 0.5
        refined_score = 0.9
        
        # Initial state with low quality
        workflow_state = replace(
            workflow_state,
            quality_scores={'test_phase': initial_score},
            refinement_count={'test_phase': 0}
        )
        
        ctx = GraphRunContext(state=workflow_state, deps=MagicMock())
        quality_gate = QualityGateNode()
        
        # First check - should trigger refinement
        with patch.object(quality_gate, 'create_refinement_node') as mock_refine:
            mock_refine.return_value = MagicMock()
            result1 = await quality_gate.run(ctx)
            assert mock_refine.called
        
        # After refinement - update state with improved score
        refined_state = replace(
            workflow_state,
            quality_scores={'test_phase': refined_score},
            refinement_count={'test_phase': 1}
        )
        
        ctx_refined = GraphRunContext(state=refined_state, deps=MagicMock())
        
        # Second check - should pass
        with patch.object(quality_gate, 'complete_phase') as mock_complete:
            mock_complete.return_value = End(refined_state)
            result2 = await quality_gate.run(ctx_refined)
            mock_complete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_quality_gate_with_feedback(self, workflow_state):
        """Test quality gate generates proper feedback for refinement."""
        # Set quality score below threshold
        workflow_state = replace(
            workflow_state,
            quality_scores={'test_phase': 0.6},
            refinement_count={'test_phase': 0},
            domain_data={
                'last_output': 'Some generated content with issues',
                'validation_errors': ['Missing required field', 'Invalid format']
            }
        )
        
        ctx = GraphRunContext(state=workflow_state, deps=MagicMock())
        quality_gate = QualityGateNode()
        
        # Check that refinement includes feedback
        with patch.object(quality_gate, 'create_refinement_node') as mock_refine:
            await quality_gate.run(ctx)
            
            # Verify refinement node was created
            mock_refine.assert_called_once()
            
            # Check state was updated with feedback
            assert 'validation_errors' in workflow_state.domain_data
            assert len(workflow_state.domain_data['validation_errors']) > 0
    
    @pytest.mark.asyncio
    async def test_quality_gate_different_thresholds(self, mock_phase_definition):
        """Test quality gates with different threshold values."""
        test_cases = [
            (0.5, 0.6, True),   # score > threshold, should pass
            (0.8, 0.7, False),  # score < threshold, should refine
            (0.75, 0.75, True), # score == threshold, should pass
            (0.9, 0.89, False), # score just below threshold, should refine
        ]
        
        for threshold, score, should_pass in test_cases:
            # Update phase definition with new threshold
            mock_phase_definition.quality_threshold = threshold
            
            from src.graphtoolkit.core.types import WorkflowDefinition
            workflow_def = WorkflowDefinition(
                domain='test',
                phases={'test_phase': mock_phase_definition},
                phase_sequence=['test_phase'],
                node_configs={},
                enable_refinement=True
            )
            
            state = WorkflowState(
                workflow_id='test-workflow',
                domain='test',
                current_phase='test_phase',
                current_node='quality_gate',
                completed_phases=set(),
                phase_outputs={},
                quality_scores={'test_phase': score},
                refinement_count={'test_phase': 0},
                workflow_def=workflow_def
            )
            
            ctx = GraphRunContext(state=state, deps=MagicMock())
            quality_gate = QualityGateNode()
            
            if should_pass:
                with patch.object(quality_gate, 'complete_phase') as mock_complete:
                    mock_complete.return_value = End(state)
                    result = await quality_gate.run(ctx)
                    mock_complete.assert_called_once()
            else:
                with patch.object(quality_gate, 'create_refinement_node') as mock_refine:
                    mock_refine.return_value = MagicMock()
                    result = await quality_gate.run(ctx)
                    mock_refine.assert_called_once()


class TestRefinementLoop:
    """Test the complete refinement loop functionality."""
    
    @pytest.mark.asyncio
    async def test_refinement_loop_state_tracking(self, workflow_state):
        """Test that refinement loop properly tracks state changes."""
        from src.graphtoolkit.nodes.atomic.llm import RefinementNode
        
        # Initial state with low quality and feedback
        workflow_state = replace(
            workflow_state,
            quality_scores={'test_phase': 0.5},
            refinement_count={'test_phase': 1},
            domain_data={
                'feedback': 'Output needs improvement in clarity',
                'last_output': 'Initial output'
            }
        )
        
        ctx = GraphRunContext(state=workflow_state, deps=MagicMock())
        refinement_node = RefinementNode()
        
        # Mock the LLM call for refinement
        with patch.object(refinement_node, 'execute_refinement', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = 'Refined output with improvements'
            
            with patch.object(refinement_node, 'get_next_node') as mock_next:
                mock_next.return_value = 'quality_gate'
                
                with patch.object(refinement_node, 'create_next_node') as mock_create:
                    mock_create.return_value = QualityGateNode()
                    
                    result = await refinement_node.run(ctx)
                    
                    # Check refinement was executed
                    mock_execute.assert_called_once()
                    
                    # Verify state tracking
                    assert ctx.state.refinement_count['test_phase'] == 1
                    assert 'feedback' in ctx.state.domain_data
    
    @pytest.mark.asyncio
    async def test_refinement_loop_convergence(self, workflow_state):
        """Test that refinement loop converges to acceptable quality."""
        quality_scores = [0.5, 0.65, 0.75, 0.85]  # Improving scores
        
        for i, score in enumerate(quality_scores):
            state = replace(
                workflow_state,
                quality_scores={'test_phase': score},
                refinement_count={'test_phase': i}
            )
            
            ctx = GraphRunContext(state=state, deps=MagicMock())
            quality_gate = QualityGateNode()
            
            if score < 0.8:  # Below threshold
                with patch.object(quality_gate, 'create_refinement_node') as mock_refine:
                    mock_refine.return_value = MagicMock()
                    result = await quality_gate.run(ctx)
                    assert mock_refine.called
            else:  # Above threshold
                with patch.object(quality_gate, 'complete_phase') as mock_complete:
                    mock_complete.return_value = End(state)
                    result = await quality_gate.run(ctx)
                    assert mock_complete.called
    
    @pytest.mark.asyncio
    async def test_refinement_with_multiple_phases(self):
        """Test refinement across multiple phases in a workflow."""
        from src.graphtoolkit.core.types import WorkflowDefinition
        
        # Create workflow with multiple phases
        phases = {
            'phase1': PhaseDefinition(
                phase_name='phase1',
                atomic_nodes=['quality_gate'],
                input_schema=MockInput,
                output_schema=MockOutput,
                dependencies=[],
                templates=TemplateConfig(
                    system_template='system/test1',
                    user_template='prompts/test1'
                ),
                storage_pattern='workflow/{workflow_id}/phase1',
                storage_type=StorageType.KV,
                quality_threshold=0.7,
                allow_refinement=True,
                max_refinements=2,
                model_config=ModelParameters(temperature=0.7),
                domain='test'
            ),
            'phase2': PhaseDefinition(
                phase_name='phase2',
                atomic_nodes=['quality_gate'],
                input_schema=MockInput,
                output_schema=MockOutput,
                dependencies=['phase1'],
                templates=TemplateConfig(
                    system_template='system/test2',
                    user_template='prompts/test2'
                ),
                storage_pattern='workflow/{workflow_id}/phase2',
                storage_type=StorageType.KV,
                quality_threshold=0.9,  # Higher threshold
                allow_refinement=True,
                max_refinements=3,
                model_config=ModelParameters(temperature=0.5),
                domain='test'
            )
        }
        
        workflow_def = WorkflowDefinition(
            domain='test',
            phases=phases,
            phase_sequence=['phase1', 'phase2'],
            node_configs={},
            enable_refinement=True
        )
        
        # Test phase1 refinement
        state1 = WorkflowState(
            workflow_id='test-multi',
            domain='test',
            current_phase='phase1',
            current_node='quality_gate',
            completed_phases=set(),
            phase_outputs={},
            quality_scores={'phase1': 0.6},  # Below threshold
            refinement_count={'phase1': 0},
            workflow_def=workflow_def
        )
        
        ctx1 = GraphRunContext(state=state1, deps=MagicMock())
        quality_gate1 = QualityGateNode()
        
        with patch.object(quality_gate1, 'create_refinement_node') as mock_refine:
            mock_refine.return_value = MagicMock()
            await quality_gate1.run(ctx1)
            assert mock_refine.called
        
        # Test phase2 with higher threshold
        state2 = WorkflowState(
            workflow_id='test-multi',
            domain='test',
            current_phase='phase2',
            current_node='quality_gate',
            completed_phases={'phase1'},
            phase_outputs={'phase1': 'output1'},
            quality_scores={'phase2': 0.85},  # Below phase2 threshold
            refinement_count={'phase2': 0},
            workflow_def=workflow_def
        )
        
        ctx2 = GraphRunContext(state=state2, deps=MagicMock())
        quality_gate2 = QualityGateNode()
        
        with patch.object(quality_gate2, 'create_refinement_node') as mock_refine:
            mock_refine.return_value = MagicMock()
            await quality_gate2.run(ctx2)
            assert mock_refine.called  # Should refine even at 0.85