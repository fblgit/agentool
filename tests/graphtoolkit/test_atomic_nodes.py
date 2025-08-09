"""
Tests for atomic node chaining and retry logic.

This module tests the atomic node patterns including:
- BaseNode retry mechanisms and error handling
- AtomicNode chaining patterns
- Storage nodes (DependencyCheck, LoadDependencies, SavePhaseOutput)
- LLM nodes with error recovery  
- Validation nodes with quality gates
- Transform nodes with data processing
"""

import asyncio
import tempfile
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import replace
from datetime import datetime, timedelta

# GraphToolkit imports
from graphtoolkit.nodes.base import (
    BaseNode, 
    AtomicNode, 
    ErrorNode,
    RetryableError,
    NonRetryableError, 
    StorageError,
    LLMError,
    ValidationError
)
from graphtoolkit.nodes.atomic.storage import (
    DependencyCheckNode,
    LoadDependenciesNode, 
    SavePhaseOutputNode,
    LoadStorageNode,
    SaveStorageNode,
    BatchLoadNode,
    BatchSaveNode
)
from graphtoolkit.core.types import (
    WorkflowState,
    WorkflowDefinition,
    PhaseDefinition,
    NodeConfig,
    StorageRef,
    StorageType,
    RetryBackoff
)

# pydantic_graph imports  
from pydantic_graph import GraphRunContext, End


class TestBaseNodeRetryLogic:
    """Test BaseNode retry mechanisms and error handling."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create a mock state with retry configuration
        self.workflow_def = MagicMock()
        self.workflow_def.node_configs = {
            'retryable_node': NodeConfig(
                node_type='test',
                retryable=True,
                max_retries=3,
                retry_backoff=RetryBackoff.EXPONENTIAL,
                retry_delay=0.1  # Fast retries for tests
            ),
            'non_retryable_node': NodeConfig(
                node_type='test',
                retryable=False,
                max_retries=0
            )
        }
        
        self.mock_state = MagicMock(spec=WorkflowState)
        self.mock_state.workflow_def = self.workflow_def
        self.mock_state.current_node = 'retryable_node'
        self.mock_state.retry_counts = {}
        self.mock_state.current_phase = 'test_phase'
        self.mock_state.workflow_id = 'test-workflow'
        
        self.mock_deps = MagicMock()
        
        # Mock get_current_node_config method
        def get_node_config():
            return self.workflow_def.node_configs.get(self.mock_state.current_node)
        
        self.mock_state.get_current_node_config = get_node_config
    
    @pytest.mark.asyncio
    async def test_retryable_error_with_retry_config(self):
        """Test that retryable errors trigger retry when configured."""
        
        class TestRetryableNode(BaseNode):
            def __init__(self):
                super().__init__()
                self.attempt_count = 0
            
            async def execute(self, ctx):
                self.attempt_count += 1
                if self.attempt_count <= 2:  # Fail first 2 attempts
                    raise RetryableError("Temporary failure")
                return End("success")
        
        node = TestRetryableNode()
        ctx = GraphRunContext(self.mock_state, self.mock_deps)
        
        # Mock the retry count update mechanism
        with patch.object(node, '_handle_retryable_error') as mock_handle:
            # First call should trigger retry handling
            await node.run(ctx)
            mock_handle.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_non_retryable_error_handling(self):
        """Test that non-retryable errors go directly to error handling."""
        
        class TestNonRetryableNode(BaseNode):
            async def execute(self, ctx):
                raise NonRetryableError("Permanent failure")
        
        node = TestNonRetryableNode()
        ctx = GraphRunContext(self.mock_state, self.mock_deps)
        
        # Mock the error handling
        with patch.object(node, '_handle_non_retryable_error') as mock_handle:
            mock_handle.return_value = ErrorNode(error="Permanent failure")
            
            result = await node.run(ctx)
            mock_handle.assert_called_once()
            assert isinstance(result, ErrorNode)
    
    @pytest.mark.asyncio 
    async def test_max_retries_exceeded(self):
        """Test behavior when max retries are exceeded."""
        
        class AlwaysFailingNode(BaseNode):
            async def execute(self, ctx):
                raise RetryableError("Always fails")
        
        node = AlwaysFailingNode()
        
        # Set state to have already reached max retries
        self.mock_state.retry_counts = {'test_phase_retryable_node_test-workflow': 3}
        ctx = GraphRunContext(self.mock_state, self.mock_deps)
        
        # Mock the error handling to return ErrorNode
        with patch.object(node, '_handle_retryable_error') as mock_handle:
            mock_handle.return_value = ErrorNode(error="Max retries exceeded")
            
            result = await node.run(ctx)
            assert isinstance(result, ErrorNode)
    
    @pytest.mark.asyncio
    async def test_backoff_strategies(self):
        """Test different retry backoff strategies."""
        
        class BackoffTestNode(BaseNode):
            def __init__(self):
                super().__init__()
                self.backoff_times = []
            
            async def _apply_backoff(self, retry_count, config):
                start_time = time.time()
                await super()._apply_backoff(retry_count, config)
                self.backoff_times.append(time.time() - start_time)
            
            async def execute(self, ctx):
                return End("success")
        
        node = BackoffTestNode()
        
        # Test exponential backoff
        config = NodeConfig(
            node_type='test',
            retry_backoff=RetryBackoff.EXPONENTIAL,
            retry_delay=0.01  # Very fast for testing
        )
        
        await node._apply_backoff(1, config)
        await node._apply_backoff(2, config)
        await node._apply_backoff(3, config)
        
        # Exponential backoff should increase: 0.01, 0.02, 0.04
        assert node.backoff_times[1] > node.backoff_times[0]
        assert node.backoff_times[2] > node.backoff_times[1]
        
        # Test linear backoff
        node = BackoffTestNode()
        config = NodeConfig(
            node_type='test',
            retry_backoff=RetryBackoff.LINEAR,
            retry_delay=0.01
        )
        
        await node._apply_backoff(1, config)
        await node._apply_backoff(2, config)
        
        # Linear backoff should be: 0.01, 0.02
        assert abs(node.backoff_times[1] - 0.02) < 0.005  # Allow small variance
    
    @pytest.mark.asyncio
    async def test_error_node_execution(self):
        """Test ErrorNode behavior."""
        error_node = ErrorNode(error="Test error", node_id="test_node")
        ctx = GraphRunContext(self.mock_state, self.mock_deps)
        
        result = await error_node.execute(ctx)
        
        # ErrorNode should return End
        assert isinstance(result, End)
        
        # Should update state with error information
        if hasattr(result.value, 'domain_data'):
            assert 'error' in result.value.domain_data
            assert result.value.domain_data['error'] == "Test error"


class TestAtomicNodeChaining:
    """Test atomic node chaining patterns."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create mock workflow definition with atomic nodes
        phase_def = MagicMock(spec=PhaseDefinition)
        phase_def.atomic_nodes = ['node_1', 'node_2', 'node_3']
        
        self.workflow_def = MagicMock()
        self.workflow_def.phases = {'test_phase': phase_def}
        
        self.mock_state = MagicMock(spec=WorkflowState)
        self.mock_state.workflow_def = self.workflow_def
        self.mock_state.current_phase = 'test_phase'
        self.mock_state.current_node = 'node_1'
        self.mock_state.domain_data = {}
        
        # Mock get_next_atomic_node method
        def get_next_node():
            nodes = phase_def.atomic_nodes
            try:
                current_idx = nodes.index(self.mock_state.current_node)
                if current_idx + 1 < len(nodes):
                    return nodes[current_idx + 1]
            except ValueError:
                pass
            return None
        
        self.mock_state.get_next_atomic_node = get_next_node
        
        self.mock_deps = MagicMock()
    
    @pytest.mark.asyncio
    async def test_atomic_node_chaining(self):
        """Test that atomic nodes chain to the next node properly."""
        
        class TestAtomicNode(AtomicNode):
            def __init__(self, node_id):
                super().__init__()
                self.node_id = node_id
            
            async def perform_operation(self, ctx):
                return f"result_{self.node_id}"
            
            async def update_state(self, state, result):
                new_data = {**state.domain_data, f'{self.node_id}_result': result}
                return replace(state, domain_data=new_data)
            
            def create_next_node(self, node_id):
                # Return a mock next node for testing
                return TestAtomicNode(node_id)
        
        node1 = TestAtomicNode('node_1')
        ctx = GraphRunContext(self.mock_state, self.mock_deps)
        
        with patch('graphtoolkit.nodes.base.GraphRunContext') as mock_ctx_class:
            # Mock GraphRunContext creation for chaining
            mock_ctx_class.return_value = ctx
            
            result = await node1.execute(ctx)
            
            # Should return next node in chain
            assert isinstance(result, TestAtomicNode)
            assert result.node_id == 'node_2'
    
    @pytest.mark.asyncio
    async def test_atomic_node_phase_completion(self):
        """Test phase completion when no more nodes in chain."""
        
        class TestAtomicNode(AtomicNode):
            async def perform_operation(self, ctx):
                return "final_result"
            
            async def update_state(self, state, result):
                return replace(state, domain_data={**state.domain_data, 'final': result})
        
        # Set to last node in phase
        self.mock_state.current_node = 'node_3'
        
        node = TestAtomicNode()
        ctx = GraphRunContext(self.mock_state, self.mock_deps)
        
        # Mock complete_phase to return End
        with patch.object(node, 'complete_phase') as mock_complete:
            mock_complete.return_value = End(self.mock_state)
            
            result = await node.execute(ctx)
            
            # Should complete phase when no more nodes
            mock_complete.assert_called_once()
            assert isinstance(result, End)


class TestStorageNodes:
    """Test storage atomic nodes."""
    
    def setup_method(self):
        """Set up test environment with storage mocks."""
        # Create mock phase definition
        phase_def = MagicMock(spec=PhaseDefinition)
        phase_def.dependencies = ['previous_phase']
        phase_def.storage_pattern = 'workflow/{workflow_id}/{phase}'
        phase_def.storage_type = StorageType.KV
        phase_def.allow_refinement = True
        
        self.workflow_def = MagicMock()
        self.workflow_def.phases = {'current_phase': phase_def}
        
        self.mock_state = MagicMock(spec=WorkflowState)
        self.mock_state.workflow_def = self.workflow_def
        self.mock_state.current_phase = 'current_phase'
        self.mock_state.current_node = 'load_dependencies'
        self.mock_state.completed_phases = {'previous_phase'}
        self.mock_state.phase_outputs = {
            'previous_phase': StorageRef(
                storage_type=StorageType.KV,
                key='workflow/test-id/previous_phase',
                created_at=datetime.now()
            )
        }
        self.mock_state.domain_data = {
            'current_phase_output': {'result': 'test_data'}
        }
        self.mock_state.workflow_id = 'test-workflow'
        self.mock_state.refinement_count = {}
        
        self.mock_state.get_current_phase_def = lambda: phase_def
        
        # Mock storage client
        self.mock_storage_client = AsyncMock()
        self.mock_deps = MagicMock()
        self.mock_deps.get_storage_client.return_value = self.mock_storage_client
    
    @pytest.mark.asyncio
    async def test_dependency_check_node_success(self):
        """Test DependencyCheckNode when all dependencies are satisfied."""
        node = DependencyCheckNode()
        ctx = GraphRunContext(self.mock_state, self.mock_deps)
        
        with patch('graphtoolkit.nodes.atomic.storage.create_node_instance') as mock_create:
            mock_next_node = MagicMock()
            mock_create.return_value = mock_next_node
            
            result = await node.execute(ctx)
            
            # Should create and return next node
            mock_create.assert_called_once()
            assert result == mock_next_node
    
    @pytest.mark.asyncio
    async def test_dependency_check_node_missing_dependency(self):
        """Test DependencyCheckNode with missing dependency."""
        # Remove completed dependency
        self.mock_state.completed_phases = set()
        
        node = DependencyCheckNode()
        ctx = GraphRunContext(self.mock_state, self.mock_deps)
        
        with pytest.raises(NonRetryableError, match="Missing dependency"):
            await node.execute(ctx)
    
    @pytest.mark.asyncio
    async def test_load_dependencies_node_success(self):
        """Test LoadDependenciesNode successful loading."""
        node = LoadDependenciesNode()
        ctx = GraphRunContext(self.mock_state, self.mock_deps)
        
        # Mock successful storage load
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = {'loaded': 'dependency_data'}
        self.mock_storage_client.run.return_value = mock_result
        
        # Mock metrics tracking
        with patch.object(node, '_track_storage_operation') as mock_track:
            result = await node.perform_operation(ctx)
            
            # Should load dependency data
            assert 'previous_phase' in result
            assert result['previous_phase'] == {'loaded': 'dependency_data'}
            
            # Should track storage operation
            mock_track.assert_called_once()
            args = mock_track.call_args[0]
            assert args[0] == 'load'  # operation
            assert args[1] == 'kv'    # storage type
    
    @pytest.mark.asyncio
    async def test_load_dependencies_node_storage_error(self):
        """Test LoadDependenciesNode with storage failure."""
        node = LoadDependenciesNode()
        ctx = GraphRunContext(self.mock_state, self.mock_deps)
        
        # Mock storage failure
        self.mock_storage_client.run.side_effect = Exception("Storage connection failed")
        
        with pytest.raises(StorageError, match="Failed to load"):
            await node.perform_operation(ctx)
    
    @pytest.mark.asyncio
    async def test_save_phase_output_node_success(self):
        """Test SavePhaseOutputNode successful save."""
        node = SavePhaseOutputNode()
        ctx = GraphRunContext(self.mock_state, self.mock_deps)
        
        # Mock successful storage save
        mock_result = MagicMock()
        mock_result.success = True
        self.mock_storage_client.run.return_value = mock_result
        
        with patch.object(node, '_track_storage_operation') as mock_track:
            result = await node.perform_operation(ctx)
            
            # Should return StorageRef
            assert isinstance(result, StorageRef)
            assert result.storage_type == StorageType.KV
            assert result.key == 'workflow/test-workflow/current_phase'
            
            # Should track storage operation
            mock_track.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_phase_output_node_no_output_data(self):
        """Test SavePhaseOutputNode with missing output data."""
        # Remove output data
        self.mock_state.domain_data = {}
        
        node = SavePhaseOutputNode()
        ctx = GraphRunContext(self.mock_state, self.mock_deps)
        
        with pytest.raises(NonRetryableError, match="No output data"):
            await node.perform_operation(ctx)
    
    @pytest.mark.asyncio
    async def test_batch_load_node(self):
        """Test BatchLoadNode parallel loading."""
        keys = ['key1', 'key2', 'key3']
        node = BatchLoadNode(storage_keys=keys)
        ctx = GraphRunContext(self.mock_state, self.mock_deps)
        
        # Mock successful loads for all keys
        mock_results = [
            MagicMock(success=True, data=f'data_{i}') 
            for i in range(len(keys))
        ]
        self.mock_storage_client.run.side_effect = mock_results
        
        with patch('graphtoolkit.nodes.atomic.storage.create_node_instance') as mock_create:
            mock_next_node = MagicMock()
            mock_create.return_value = mock_next_node
            
            result = await node.execute(ctx)
            
            # Should call storage for each key
            assert self.mock_storage_client.run.call_count == len(keys)
            
            # Should return next node
            assert result == mock_next_node
    
    @pytest.mark.asyncio
    async def test_batch_save_node(self):
        """Test BatchSaveNode parallel saving."""
        node = BatchSaveNode(storage_prefix='batch_test')
        
        # Set up iteration results to save
        self.mock_state.iter_results = ['item1', 'item2', 'item3']
        
        ctx = GraphRunContext(self.mock_state, self.mock_deps)
        
        # Mock successful saves
        mock_result = MagicMock(success=True)
        self.mock_storage_client.run.return_value = mock_result
        
        with patch.object(node, '_continue_chain') as mock_continue:
            mock_continue.return_value = MagicMock()
            
            await node.execute(ctx)
            
            # Should save all items
            assert self.mock_storage_client.run.call_count == 3
            
            # Should continue chain with storage references
            mock_continue.assert_called_once()
            refs = mock_continue.call_args[0][1]  # Second argument is refs
            assert len(refs) == 3
            assert all(isinstance(ref, StorageRef) for ref in refs)


class TestStorageNodeIntegration:
    """Test storage nodes with real agentoolkit integration."""
    
    def setup_method(self):
        """Set up with agentoolkit mocks."""
        # Try to import agentoolkit components
        try:
            from agentoolkit.storage.kv import create_storage_kv_agent
            self.has_agentoolkit = True
        except ImportError:
            self.has_agentoolkit = False
    
    @pytest.mark.skipif(not hasattr(pytest, 'importorskip'), reason="Storage integration optional")
    @pytest.mark.asyncio
    async def test_storage_integration_kv(self):
        """Test integration with agentoolkit KV storage."""
        if not self.has_agentoolkit:
            pytest.skip("Agentoolkit not available")
        
        from agentool.core.injector import get_injector
        
        # Set up real agentoolkit storage
        get_injector().clear()
        from agentoolkit.storage.kv import create_storage_kv_agent
        create_storage_kv_agent()
        
        # Create storage node with real dependencies  
        phase_def = MagicMock(spec=PhaseDefinition)
        phase_def.dependencies = []
        phase_def.storage_type = StorageType.KV
        phase_def.storage_pattern = 'test/integration/{phase}'
        
        mock_state = MagicMock(spec=WorkflowState)
        mock_state.current_phase = 'test_phase'
        mock_state.workflow_id = 'integration-test'
        mock_state.domain_data = {'test_phase_output': {'test': 'data'}}
        mock_state.refinement_count = {}
        mock_state.get_current_phase_def = lambda: phase_def
        
        # Create real deps with injector
        mock_deps = MagicMock()
        mock_deps.get_storage_client = lambda: get_injector()
        
        node = SavePhaseOutputNode()
        ctx = GraphRunContext(mock_state, mock_deps)
        
        # Should be able to save to real storage
        with patch.object(node, '_track_storage_operation'):
            result = await node.perform_operation(ctx)
            
            assert isinstance(result, StorageRef)
            assert result.storage_type == StorageType.KV
            assert 'test/integration/test_phase' in result.key


class TestLLMNodeErrorRecovery:
    """Test LLM nodes with error recovery patterns."""
    
    def setup_method(self):
        """Set up LLM test environment."""
        # Create node config with retry enabled for LLM calls
        self.node_config = NodeConfig(
            node_type='llm_call',
            retryable=True,
            max_retries=2,
            retry_backoff=RetryBackoff.LINEAR,
            retry_delay=0.05  # Fast for testing
        )
        
        # Mock state
        self.mock_state = MagicMock(spec=WorkflowState)
        self.mock_state.current_node = 'llm_call'
        self.mock_state.retry_counts = {}
        self.mock_state.get_current_node_config = lambda: self.node_config
        
        self.mock_deps = MagicMock()
    
    @pytest.mark.asyncio
    async def test_llm_retry_on_api_error(self):
        """Test LLM node retries on API failures."""
        
        class MockLLMNode(BaseNode):
            def __init__(self):
                super().__init__()
                self.call_count = 0
            
            async def execute(self, ctx):
                self.call_count += 1
                if self.call_count <= 1:  # Fail first call
                    raise LLMError("API rate limit exceeded")
                return End(f"Success on attempt {self.call_count}")
        
        node = MockLLMNode()
        ctx = GraphRunContext(self.mock_state, self.mock_deps)
        
        # Mock retry handling to actually retry
        original_handle = node._handle_retryable_error
        
        async def mock_retry_handle(ctx, error, config):
            if node.call_count < 2:  # Allow retry
                self.mock_state.retry_counts['llm_call'] = node.call_count
                return node  # Return self for retry
            return await original_handle(ctx, error, config)
        
        with patch.object(node, '_handle_retryable_error', side_effect=mock_retry_handle):
            result = await node.run(ctx)
            
            # Should eventually succeed after retry
            assert node.call_count >= 2
    
    @pytest.mark.asyncio 
    async def test_llm_permanent_failure(self):
        """Test LLM node handling of permanent failures."""
        
        class FailingLLMNode(BaseNode):
            async def execute(self, ctx):
                raise LLMError("Invalid API key")
        
        # Configure as non-retryable for this test
        self.node_config.retryable = False
        
        node = FailingLLMNode()
        ctx = GraphRunContext(self.mock_state, self.mock_deps)
        
        with patch.object(node, '_handle_non_retryable_error') as mock_handle:
            mock_handle.return_value = ErrorNode(error="Invalid API key")
            
            result = await node.run(ctx)
            
            assert isinstance(result, ErrorNode)
            mock_handle.assert_called_once()


class TestValidationNodes:
    """Test validation nodes and quality gates."""
    
    @pytest.mark.asyncio
    async def test_validation_error_triggers_refinement(self):
        """Test that ValidationError triggers refinement rather than retry."""
        
        class ValidationTestNode(BaseNode):
            async def execute(self, ctx):
                # Simulate quality check failure
                raise ValidationError("Quality score 0.65 below threshold 0.8")
        
        mock_state = MagicMock(spec=WorkflowState)
        mock_state.current_node = 'quality_gate'
        mock_state.retry_counts = {}
        mock_state.get_current_node_config = lambda: NodeConfig(node_type='validation')
        
        node = ValidationTestNode()
        ctx = GraphRunContext(mock_state, MagicMock())
        
        # ValidationError should be treated as non-retryable
        with patch.object(node, '_handle_non_retryable_error') as mock_handle:
            mock_handle.return_value = ErrorNode(error="Quality check failed")
            
            result = await node.run(ctx)
            
            mock_handle.assert_called_once()
            error_arg = mock_handle.call_args[0][1]
            assert isinstance(error_arg, ValidationError)


class TestNodeChainPerformance:
    """Test performance characteristics of node chains."""
    
    @pytest.mark.asyncio
    async def test_long_chain_performance(self):
        """Test performance of long atomic node chains."""
        import time
        
        class FastAtomicNode(AtomicNode):
            def __init__(self, node_id):
                super().__init__()
                self.node_id = node_id
            
            async def perform_operation(self, ctx):
                # Minimal work
                return f"result_{self.node_id}"
            
            async def update_state(self, state, result):
                return state  # No state changes for performance test
        
        # Simulate 20 nodes in chain
        start_time = time.time()
        
        node_chain = []
        for i in range(20):
            node = FastAtomicNode(f'node_{i}')
            node_chain.append(node)
        
        creation_time = time.time() - start_time
        
        # Node creation should be fast
        assert creation_time < 1.0, f"Node creation took {creation_time:.3f}s, expected < 1s"
        
        # Test individual node execution performance  
        mock_state = MagicMock()
        mock_state.domain_data = {}
        mock_deps = MagicMock()
        
        start_time = time.time()
        
        for node in node_chain[:5]:  # Test first 5 nodes
            ctx = GraphRunContext(mock_state, mock_deps)
            try:
                result = await node.perform_operation(ctx)
                assert result is not None
            except NotImplementedError:
                pass  # Expected for some abstract methods
        
        execution_time = time.time() - start_time
        
        # Individual operations should be very fast
        assert execution_time < 0.5, f"Node execution took {execution_time:.3f}s, expected < 0.5s"
    
    @pytest.mark.asyncio
    async def test_parallel_storage_operations(self):
        """Test parallel storage operations performance."""
        import time
        
        # Mock storage operations with controlled delay
        async def mock_storage_op(*args, **kwargs):
            await asyncio.sleep(0.01)  # 10ms delay
            return MagicMock(success=True, data='test_data')
        
        mock_storage_client = AsyncMock()
        mock_storage_client.run.side_effect = mock_storage_op
        
        mock_deps = MagicMock()
        mock_deps.get_storage_client.return_value = mock_storage_client
        
        mock_state = MagicMock(spec=WorkflowState)
        mock_state.domain_data = {}
        
        # Test BatchLoadNode with 10 keys
        keys = [f'key_{i}' for i in range(10)]
        node = BatchLoadNode(storage_keys=keys)
        
        ctx = GraphRunContext(mock_state, mock_deps)
        
        start_time = time.time()
        
        with patch('graphtoolkit.nodes.atomic.storage.create_node_instance'):
            await node.execute(ctx)
        
        execution_time = time.time() - start_time
        
        # Parallel execution should be much faster than sequential
        # 10 operations × 10ms = 100ms if sequential
        # Should complete in ~15-30ms if parallel
        assert execution_time < 0.05, f"Parallel load took {execution_time:.3f}s, expected < 0.05s"
        
        # Should have made all storage calls
        assert mock_storage_client.run.call_count == 10


class TestNodeConfigurationReading:
    """Test node configuration reading from state."""
    
    def setup_method(self):
        """Set up configuration test environment."""
        self.node_configs = {
            'configurable_node': NodeConfig(
                node_type='test',
                retryable=True,
                max_retries=5,
                retry_backoff=RetryBackoff.EXPONENTIAL,
                retry_delay=0.2,
                iter_enabled=True,
                cacheable=True,
                timeout=30.0
            )
        }
        
        self.workflow_def = MagicMock()
        self.workflow_def.node_configs = self.node_configs
        
        self.mock_state = MagicMock(spec=WorkflowState)
        self.mock_state.workflow_def = self.workflow_def
        self.mock_state.current_node = 'configurable_node'
        
        def get_node_config():
            return self.node_configs.get(self.mock_state.current_node)
        
        self.mock_state.get_current_node_config = get_node_config
    
    def test_node_config_reading(self):
        """Test that nodes can read their configuration from state."""
        
        class ConfigurableTestNode(BaseNode):
            def get_config(self, ctx):
                return self._get_node_config(ctx)
        
        node = ConfigurableTestNode()
        ctx = GraphRunContext(self.mock_state, MagicMock())
        
        config = node.get_config(ctx)
        
        # Should read config from state
        assert config is not None
        assert config.node_type == 'test'
        assert config.retryable == True
        assert config.max_retries == 5
        assert config.retry_backoff == RetryBackoff.EXPONENTIAL
        assert config.retry_delay == 0.2
        assert config.iter_enabled == True
        assert config.cacheable == True
        assert config.timeout == 30.0
    
    def test_missing_node_config(self):
        """Test behavior when node config is missing."""
        self.mock_state.current_node = 'missing_node'
        
        class TestNode(BaseNode):
            def get_config(self, ctx):
                return self._get_node_config(ctx)
        
        node = TestNode()
        ctx = GraphRunContext(self.mock_state, MagicMock())
        
        config = node.get_config(ctx)
        
        # Should return None for missing config
        assert config is None


class TestRealNodeChainExecution:
    """Test node chain execution with real GraphToolkit components."""
    
    @pytest.mark.asyncio
    async def test_real_dependency_check_execution(self):
        """Test DependencyCheckNode with real types."""
        # Create real phase definition
        from graphtoolkit.core.types import PhaseDefinition, WorkflowDefinition, WorkflowState
        
        phase_def = PhaseDefinition(
            phase_name='test_phase',
            atomic_nodes=['dependency_check', 'load_dependencies'],
            input_schema=MagicMock,  # Simplified for test
            output_schema=MagicMock,
            dependencies=['previous_phase']
        )
        
        workflow_def = WorkflowDefinition(
            domain='test',
            phases={'test_phase': phase_def},
            phase_sequence=['test_phase'],
            node_configs={}
        )
        
        state = WorkflowState(
            workflow_def=workflow_def,
            workflow_id='real-test',
            domain='test',
            current_phase='test_phase',
            current_node='dependency_check',
            completed_phases={'previous_phase'}  # Dependency satisfied
        )
        
        node = DependencyCheckNode()
        ctx = GraphRunContext(state, MagicMock())
        
        # Should succeed with satisfied dependency
        with patch('graphtoolkit.nodes.atomic.storage.create_node_instance') as mock_create:
            mock_next_node = MagicMock()
            mock_create.return_value = mock_next_node
            
            result = await node.execute(ctx)
            
            assert result == mock_next_node
            mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_real_error_node_execution(self):
        """Test ErrorNode with real WorkflowState."""
        from graphtoolkit.core.types import WorkflowState, WorkflowDefinition
        
        workflow_def = WorkflowDefinition(
            domain='test',
            phases={},
            phase_sequence=[],
            node_configs={}
        )
        
        state = WorkflowState(
            workflow_def=workflow_def,
            workflow_id='error-test',
            domain='test',
            current_phase='error_phase',
            domain_data={'existing': 'data'}
        )
        
        error_node = ErrorNode(error="Test execution error", node_id="failing_node")
        ctx = GraphRunContext(state, MagicMock())
        
        result = await error_node.execute(ctx)
        
        # Should return End with error information
        assert isinstance(result, End)
        final_state = result.value
        
        # Error information should be added to domain_data
        assert 'error' in final_state.domain_data
        assert final_state.domain_data['error'] == "Test execution error"
        assert final_state.domain_data['error_node'] == "failing_node"
        assert 'error_time' in final_state.domain_data
        
        # Original data should be preserved
        assert final_state.domain_data['existing'] == 'data'