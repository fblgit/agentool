"""GraphToolkit Base Node Patterns.

Base classes and patterns for all nodes in the meta-framework.
"""

import asyncio
import logging
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Optional, TypeVar, Union

# Real pydantic_graph imports - no more mocks
from pydantic_graph import BaseNode as PydanticBaseNode, End, GraphRunContext

from ..core.metrics import MetricsMixin
from ..core.types import NodeConfig, RetryBackoff

logger = logging.getLogger(__name__)


# Type variables following pydantic_graph patterns
StateT = TypeVar('StateT')  # State type for graph
DepsT = TypeVar('DepsT')    # Dependencies type for graph  
OutputT = TypeVar('OutputT')  # Final graph return type


# Error hierarchy for state-driven retry
class WorkflowError(Exception):
    """Base class for all workflow errors."""
    pass


class RetryableError(WorkflowError):
    """Error that can be retried (transient failures)."""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after  # Seconds to wait before retry


class NonRetryableError(WorkflowError):
    """Error that should not be retried (permanent failures)."""
    pass


class StorageError(RetryableError):
    """Storage operation failed (network, permissions, etc)."""
    pass


class LLMError(RetryableError):
    """LLM call failed (now retryable via config)."""
    pass


class ValidationError(NonRetryableError):
    """Validation failed (triggers refinement, not retry)."""
    pass


@dataclass
class BaseNode(PydanticBaseNode[StateT, DepsT, OutputT], MetricsMixin):
    """Base node with state-driven retry and chaining support.
    Properly inherits from pydantic_graph.BaseNode.
    
    All nodes:
    1. Read configuration from state (via deps)
    2. Perform their single operation  
    3. Update state if needed
    4. Return the next node in chain
    """
    
    async def run(self, ctx: GraphRunContext[StateT, DepsT]) -> Union['BaseNode', End[OutputT]]:
        """Execute node with state-driven configuration and metrics tracking."""
        # Get node info for metrics
        node_name = self._get_node_name()
        phase = getattr(ctx.state, 'current_phase', 'unknown')
        
        logger.debug(f"[{node_name}] === ENTRY === Phase: {phase}, Workflow: {getattr(ctx.state, 'workflow_id', 'unknown')}")
        logger.debug(f"[{node_name}] Current node in state: {getattr(ctx.state, 'current_node', 'unknown')}")
        logger.debug(f"[{node_name}] State keys: {list(ctx.state.__dict__.keys()) if hasattr(ctx.state, '__dict__') else 'N/A'}")
        logger.debug(f"[{node_name}] Domain data keys: {list(ctx.state.domain_data.keys()) if hasattr(ctx.state, 'domain_data') else 'N/A'}")
        logger.debug(f"[{node_name}] Retry counts: {ctx.state.retry_counts if hasattr(ctx.state, 'retry_counts') else 'N/A'}")
        
        # Get our configuration from deps
        node_config = self._get_node_config(ctx)
        logger.debug(f"[{node_name}] Retrieved node config: {node_config}")
        logger.debug(f"[{node_name}] Config: retryable={getattr(node_config, 'retryable', False)}, max_retries={getattr(node_config, 'max_retries', 0)}")
        
        # Track execution start
        start_time = await self._track_execution_start(node_name, phase)
        
        # Check if we should retry
        if node_config and node_config.retryable:
            retry_key = self._get_retry_key(ctx.state)
            logger.debug(f"[{node_name}] Retry key: {retry_key}")
            retry_count = ctx.state.retry_counts.get(retry_key, 0)
            logger.debug(f"[{node_name}] Current retry count: {retry_count}")
            
            if retry_count > 0:
                logger.info(f"[{node_name}] This is retry attempt {retry_count} for {retry_key}")
                # Track retry attempt
                if hasattr(self, '_last_error'):
                    logger.debug(f"[{node_name}] Last error: {self._last_error}")
                    await self._track_retry_attempt(node_name, phase, retry_count, self._last_error)
                
                # Apply backoff before retry
                logger.debug(f"[{node_name}] Applying backoff strategy: {node_config.retry_backoff}")
                await self._apply_backoff(retry_count, node_config)
        
        try:
            # Execute our specific operation
            logger.debug(f"[{node_name}] Calling execute method")
            result = await self.execute(ctx)
            logger.debug(f"[{node_name}] Execute returned: {type(result).__name__}")
            
            # Track success
            await self._track_execution_success(node_name, phase, start_time, result)
            
            logger.debug(f"[{node_name}] === EXIT === Success, returning {type(result).__name__}")
            return result
            
        except RetryableError as e:
            logger.warning(f"[{node_name}] Retryable error caught: {e}")
            # Store error for retry tracking
            self._last_error = e
            await self._track_execution_failure(node_name, phase, start_time, e)
            logger.debug(f"[{node_name}] Handling retryable error with config: {node_config}")
            return await self._handle_retryable_error(ctx, e, node_config)
            
        except NonRetryableError as e:
            logger.error(f"[{node_name}] Non-retryable error caught: {e}")
            await self._track_execution_failure(node_name, phase, start_time, e)
            return await self._handle_non_retryable_error(ctx, e)
            
        except Exception as e:
            logger.error(f"[{node_name}] Unexpected error caught: {e}", exc_info=True)
            # Treat unknown errors as non-retryable
            non_retryable_error = NonRetryableError(str(e))
            await self._track_execution_failure(node_name, phase, start_time, non_retryable_error)
            return await self._handle_non_retryable_error(ctx, non_retryable_error)
    
    async def execute(self, ctx: GraphRunContext[StateT, DepsT]) -> Union['BaseNode', End[OutputT]]:
        """Execute the node's specific operation.
        Must be implemented by subclasses.
        """
        raise NotImplementedError
    
    def _get_node_config(self, ctx: GraphRunContext[StateT, DepsT]) -> Optional[NodeConfig]:
        """Get configuration for this node from state."""
        if hasattr(ctx.state, 'get_current_node_config'):
            return ctx.state.get_current_node_config()
        return None
    
    def _get_retry_key(self, state: StateT) -> str:
        """Generate unique retry key for this node instance."""
        phase = getattr(state, 'current_phase', 'unknown')
        node = getattr(state, 'current_node', 'unknown')
        workflow_id = getattr(state, 'workflow_id', 'unknown')
        return f'{phase}_{node}_{workflow_id}'
    
    async def _apply_backoff(self, retry_count: int, config: NodeConfig) -> None:
        """Apply backoff strategy before retry."""
        if config.retry_backoff == RetryBackoff.NONE:
            return
        elif config.retry_backoff == RetryBackoff.LINEAR:
            delay = config.retry_delay * retry_count
        elif config.retry_backoff == RetryBackoff.EXPONENTIAL:
            delay = config.retry_delay * (2 ** (retry_count - 1))
        else:
            delay = config.retry_delay
        
        logger.debug(f'Applying {config.retry_backoff} backoff: {delay}s')
        await asyncio.sleep(delay)
    
    async def _handle_retryable_error(
        self,
        ctx: GraphRunContext[StateT, DepsT],
        error: RetryableError,
        config: Optional[NodeConfig]
    ) -> Union['BaseNode', End[OutputT]]:
        """Handle retryable error with self-return pattern."""
        logger.debug(f"[_handle_retryable_error] Config: {config}, retryable: {config.retryable if config else None}")
        
        if not config or not config.retryable:
            # Not configured for retry, convert to non-retryable
            logger.info(f"[_handle_retryable_error] Node not configured for retry, treating as non-retryable")
            return await self._handle_non_retryable_error(ctx, NonRetryableError(str(error)))
        
        retry_key = self._get_retry_key(ctx.state)
        retry_count = ctx.state.retry_counts.get(retry_key, 0)
        logger.debug(f"[_handle_retryable_error] Retry key: {retry_key}, count: {retry_count}, max: {config.max_retries}")
        
        if retry_count < config.max_retries:
            # Retry by returning ourselves with updated retry count
            logger.info(f'[_handle_retryable_error] Retrying {retry_key}: attempt {retry_count + 1}/{config.max_retries}')
            
            # Update retry count in state
            # Note: In actual implementation, state update happens through context
            # Increment retry count - modify in place
            ctx.state.retry_counts[retry_key] = retry_count + 1
            logger.debug(f"[_handle_retryable_error] Updated retry count to {retry_count + 1}")
            
            # Return self to retry
            logger.debug(f"[_handle_retryable_error] Returning new instance of {self.__class__.__name__} for retry")
            return self.__class__()
        else:
            # Max retries exceeded
            logger.error(f'[_handle_retryable_error] Max retries exceeded for {retry_key} ({retry_count}/{config.max_retries})')
            return ErrorNode(error=str(error), node_id=ctx.state.current_node)
    
    async def _handle_non_retryable_error(
        self,
        ctx: GraphRunContext[StateT, DepsT],
        error: NonRetryableError
    ) -> Union['BaseNode', End[OutputT]]:
        """Handle non-retryable error."""
        logger.error(f'Non-retryable error in {ctx.state.current_node}: {error}')
        return ErrorNode(error=str(error), node_id=ctx.state.current_node)
    
    def get_next_node(self, state: StateT) -> Optional[str]:
        """Get the next node in the current phase's sequence."""
        if hasattr(state, 'get_next_atomic_node'):
            return state.get_next_atomic_node()
        return None
    
    def create_next_node(self, node_id: str) -> 'BaseNode':
        """Create the next node instance."""
        from ..core.factory import create_node_instance
        return create_node_instance(node_id)


@dataclass
class ErrorNode(BaseNode[StateT, DepsT, StateT]):
    """Terminal error node for handling failures.
    """
    error: str
    node_id: Optional[str] = None
    
    async def execute(self, ctx: GraphRunContext[StateT, DepsT]) -> End[StateT]:
        """Record error and end execution."""
        # Update state with error information - modify in place
        if hasattr(ctx.state, 'domain_data'):
            ctx.state.domain_data['error'] = self.error
            ctx.state.domain_data['error_node'] = self.node_id
            ctx.state.domain_data['error_time'] = datetime.now().isoformat()
        
        return End(ctx.state)


@dataclass  
class AtomicNode(BaseNode[StateT, DepsT, OutputT]):
    """Base class for atomic nodes that chain together.
    
    Pattern:
    1. Execute single operation
    2. Update state
    3. Return next node in chain
    """
    
    async def execute(self, ctx: GraphRunContext[StateT, DepsT]) -> Union['BaseNode', End[OutputT]]:
        """Execute atomic operation and chain to next."""
        node_name = self.__class__.__name__
        logger.debug(f"[{node_name}] AtomicNode.execute starting")
        logger.debug(f"[{node_name}] Current state.current_node: {ctx.state.current_node if hasattr(ctx.state, 'current_node') else 'N/A'}")
        logger.debug(f"[{node_name}] Current state.current_phase: {ctx.state.current_phase if hasattr(ctx.state, 'current_phase') else 'N/A'}")
        
        # Perform our atomic operation
        logger.debug(f"[{node_name}] Calling perform_operation")
        try:
            result = await self.perform_operation(ctx)
            logger.debug(f"[{node_name}] perform_operation returned: {type(result).__name__ if result else 'None'}")
        except Exception as e:
            logger.error(f"[{node_name}] perform_operation raised exception: {e}", exc_info=True)
            raise
        
        # Update state with result - directly modify the mutable state
        logger.debug(f"[{node_name}] Updating state in place")
        await self.update_state_in_place(ctx.state, result)
        logger.debug(f"[{node_name}] State updated, domain_data keys: {list(ctx.state.domain_data.keys()) if hasattr(ctx.state, 'domain_data') else 'N/A'}")
        
        # Get next node in chain based on current position
        next_node_id = self.get_next_node(ctx.state)
        logger.debug(f"[{node_name}] Next node ID: {next_node_id}")
        
        if next_node_id:
            # Update current_node in state for next execution
            old_node = ctx.state.current_node if hasattr(ctx.state, 'current_node') else 'unknown'
            ctx.state.current_node = next_node_id
            logger.info(f"[{node_name}] Chaining from {old_node} to {next_node_id}")
            
            # Chain to next node - it will receive the same context with updated state
            next_node = self.create_next_node(next_node_id)
            logger.debug(f"[{node_name}] Created next node: {type(next_node).__name__}")
            return next_node
        else:
            # No more nodes in this phase
            logger.info(f"[{node_name}] No more nodes in phase {ctx.state.current_phase if hasattr(ctx.state, 'current_phase') else 'unknown'}, completing phase")
            return await self.complete_phase(ctx, ctx.state)
    
    async def perform_operation(self, ctx: GraphRunContext[StateT, DepsT]) -> Any:
        """Perform the atomic operation.
        Must be implemented by subclasses.
        """
        raise NotImplementedError
    
    async def update_state_in_place(self, state: StateT, result: Any) -> None:
        """Update state with operation result - modifies state in place.
        Default implementation stores in domain_data.
        """
        if hasattr(state, 'domain_data'):
            key = f'{state.current_node}_result'
            state.domain_data[key] = result
    
    async def update_state(self, state: StateT, result: Any) -> StateT:
        """Legacy method for compatibility - now just modifies in place."""
        await self.update_state_in_place(state, result)
        return state
    
    async def complete_phase(self, ctx: GraphRunContext[StateT, DepsT], state: StateT) -> Union['BaseNode', End[OutputT]]:
        """Handle phase completion.
        Default implementation moves to next phase.
        """
        # Mark phase as complete - modify in place
        if hasattr(state, 'completed_phases') and hasattr(state, 'current_phase'):
            state.completed_phases.add(state.current_phase)
            
            # Get next phase
            if hasattr(ctx.state, 'workflow_def'):
                next_phase = ctx.state.workflow_def.get_next_phase(state.current_phase)
                if next_phase:
                    # Move to next phase
                    from .atomic.control import NextPhaseNode
                    return NextPhaseNode()
        
        # No more phases - workflow complete
        return End(state)


# Register error node
from ..core.factory import register_node_class

register_node_class('error', ErrorNode)