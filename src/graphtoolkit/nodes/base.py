"""
GraphToolkit Base Node Patterns.

Base classes and patterns for all nodes in the meta-framework.
"""

from typing import TypeVar, Generic, Optional, Dict, Any, Union
from dataclasses import dataclass, field, replace
from datetime import datetime
import asyncio
import logging

try:
    from pydantic_graph import BaseNode as PydanticBaseNode, GraphRunContext, End
    HAS_PYDANTIC_GRAPH = True
except ImportError:
    # Development stubs
    HAS_PYDANTIC_GRAPH = False
    
    class PydanticBaseNode:
        pass
    
    class GraphRunContext:
        def __init__(self, state: Any, deps: Any):
            self.state = state
            self.deps = deps
    
    class End:
        def __init__(self, result: Any):
            self.result = result

from ..core.types import (
    WorkflowState,
    NodeConfig,
    RetryBackoff
)


logger = logging.getLogger(__name__)


# Type variables
StateT = TypeVar('StateT', bound='WorkflowState')
DepsT = TypeVar('DepsT')
OutputT = TypeVar('OutputT')


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
class BaseNode(PydanticBaseNode, Generic[StateT, DepsT, OutputT]):
    """
    Base node with state-driven retry and chaining support.
    
    All nodes:
    1. Read configuration from state
    2. Perform their single operation
    3. Update state if needed
    4. Return the next node in chain
    """
    
    async def run(self, ctx: GraphRunContext[StateT, DepsT]) -> Union[PydanticBaseNode, End[OutputT]]:
        """Execute node with state-driven configuration."""
        # Get our configuration from state
        node_config = self._get_node_config(ctx.state)
        
        # Check if we should retry
        if node_config and node_config.retryable:
            retry_key = self._get_retry_key(ctx.state)
            retry_count = ctx.state.retry_counts.get(retry_key, 0)
            
            if retry_count > 0:
                # Apply backoff before retry
                await self._apply_backoff(retry_count, node_config)
        
        try:
            # Execute our specific operation
            return await self.execute(ctx)
            
        except RetryableError as e:
            return await self._handle_retryable_error(ctx, e, node_config)
            
        except NonRetryableError as e:
            return await self._handle_non_retryable_error(ctx, e)
            
        except Exception as e:
            # Treat unknown errors as non-retryable
            return await self._handle_non_retryable_error(ctx, NonRetryableError(str(e)))
    
    async def execute(self, ctx: GraphRunContext[StateT, DepsT]) -> Union[PydanticBaseNode, End[OutputT]]:
        """
        Execute the node's specific operation.
        Must be implemented by subclasses.
        """
        raise NotImplementedError
    
    def _get_node_config(self, state: StateT) -> Optional[NodeConfig]:
        """Get configuration for this node from state."""
        if hasattr(state, 'workflow_def') and hasattr(state, 'current_node'):
            return state.workflow_def.node_configs.get(state.current_node)
        return None
    
    def _get_retry_key(self, state: StateT) -> str:
        """Generate unique retry key for this node instance."""
        phase = getattr(state, 'current_phase', 'unknown')
        node = getattr(state, 'current_node', 'unknown')
        workflow_id = getattr(state, 'workflow_id', 'unknown')
        return f"{phase}_{node}_{workflow_id}"
    
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
        
        logger.debug(f"Applying {config.retry_backoff} backoff: {delay}s")
        await asyncio.sleep(delay)
    
    async def _handle_retryable_error(
        self,
        ctx: GraphRunContext[StateT, DepsT],
        error: RetryableError,
        config: Optional[NodeConfig]
    ) -> Union[PydanticBaseNode, End[OutputT]]:
        """Handle retryable error with self-return pattern."""
        if not config or not config.retryable:
            # Not configured for retry, convert to non-retryable
            return await self._handle_non_retryable_error(ctx, NonRetryableError(str(error)))
        
        retry_key = self._get_retry_key(ctx.state)
        retry_count = ctx.state.retry_counts.get(retry_key, 0)
        
        if retry_count < config.max_retries:
            # Retry by returning ourselves with updated retry count
            logger.info(f"Retrying {retry_key}: attempt {retry_count + 1}/{config.max_retries}")
            
            # Update retry count in state
            # Note: In actual implementation, state update happens through context
            # This is a simplified representation
            new_state = replace(
                ctx.state,
                retry_counts={**ctx.state.retry_counts, retry_key: retry_count + 1}
            )
            
            # Return self to retry
            return self.__class__()
        else:
            # Max retries exceeded
            logger.error(f"Max retries exceeded for {retry_key}")
            return ErrorNode(error=str(error), node_id=ctx.state.current_node)
    
    async def _handle_non_retryable_error(
        self,
        ctx: GraphRunContext[StateT, DepsT],
        error: NonRetryableError
    ) -> Union[PydanticBaseNode, End[OutputT]]:
        """Handle non-retryable error."""
        logger.error(f"Non-retryable error in {ctx.state.current_node}: {error}")
        return ErrorNode(error=str(error), node_id=ctx.state.current_node)
    
    def get_next_node(self, state: StateT) -> Optional[str]:
        """Get the next node in the current phase's sequence."""
        if hasattr(state, 'workflow_def') and hasattr(state, 'current_phase'):
            phase_def = state.workflow_def.phases.get(state.current_phase)
            if phase_def and hasattr(state, 'current_node'):
                try:
                    current_idx = phase_def.atomic_nodes.index(state.current_node)
                    if current_idx + 1 < len(phase_def.atomic_nodes):
                        return phase_def.atomic_nodes[current_idx + 1]
                except (ValueError, IndexError):
                    pass
        return None
    
    def create_next_node(self, node_id: str) -> PydanticBaseNode:
        """Create the next node instance."""
        from ..core.factory import create_node_instance
        return create_node_instance(node_id)


@dataclass
class ErrorNode(BaseNode[StateT, DepsT, StateT]):
    """
    Terminal error node for handling failures.
    """
    error: str
    node_id: Optional[str] = None
    
    async def execute(self, ctx: GraphRunContext[StateT, DepsT]) -> End[StateT]:
        """Record error and end execution."""
        # Update state with error information
        if hasattr(ctx.state, 'domain_data'):
            new_state = replace(
                ctx.state,
                domain_data={
                    **ctx.state.domain_data,
                    'error': self.error,
                    'error_node': self.node_id,
                    'error_time': datetime.now().isoformat()
                }
            )
        else:
            new_state = ctx.state
        
        return End(new_state)


@dataclass  
class AtomicNode(BaseNode[StateT, DepsT, OutputT]):
    """
    Base class for atomic nodes that chain together.
    
    Pattern:
    1. Execute single operation
    2. Update state
    3. Return next node in chain
    """
    
    async def execute(self, ctx: GraphRunContext[StateT, DepsT]) -> Union[PydanticBaseNode, End[OutputT]]:
        """Execute atomic operation and chain to next."""
        # Perform our atomic operation
        result = await self.perform_operation(ctx)
        
        # Update state with result
        new_state = await self.update_state(ctx.state, result)
        
        # Get next node in chain
        next_node_id = self.get_next_node(new_state)
        
        if next_node_id:
            # Update current_node in state for next execution
            new_state = replace(new_state, current_node=next_node_id)
            
            # Chain to next node
            return self.create_next_node(next_node_id)
        else:
            # No more nodes in this phase
            return await self.complete_phase(ctx, new_state)
    
    async def perform_operation(self, ctx: GraphRunContext[StateT, DepsT]) -> Any:
        """
        Perform the atomic operation.
        Must be implemented by subclasses.
        """
        raise NotImplementedError
    
    async def update_state(self, state: StateT, result: Any) -> StateT:
        """
        Update state with operation result.
        Default implementation stores in domain_data.
        """
        if hasattr(state, 'domain_data'):
            key = f"{state.current_node}_result"
            return replace(
                state,
                domain_data={**state.domain_data, key: result}
            )
        return state
    
    async def complete_phase(self, ctx: GraphRunContext[StateT, DepsT], state: StateT) -> Union[PydanticBaseNode, End[OutputT]]:
        """
        Handle phase completion.
        Default implementation moves to next phase.
        """
        # Mark phase as complete
        if hasattr(state, 'completed_phases') and hasattr(state, 'current_phase'):
            new_state = replace(
                state,
                completed_phases=state.completed_phases | {state.current_phase}
            )
            
            # Get next phase
            if hasattr(state, 'workflow_def'):
                next_phase = state.workflow_def.get_next_phase(state.current_phase)
                if next_phase:
                    # Move to next phase
                    from .control import NextPhaseNode
                    return NextPhaseNode()
        
        # No more phases - workflow complete
        return End(state)


# Register error node
from ..core.factory import register_node_class
register_node_class('error', ErrorNode)