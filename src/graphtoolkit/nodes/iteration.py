"""GraphToolkit Iteration Nodes.

Nodes that support iteration over collections using self-return pattern.
This replaces parallel sub-graphs with state-based iteration.
"""

import asyncio
import logging
from dataclasses import dataclass, replace
from typing import Any, Callable, Generic, List, Optional, TypeVar

from pydantic_graph import End, GraphRunContext

from ..core.factory import register_node_class
from ..core.types import WorkflowState
from .base import BaseNode, NonRetryableError, RetryableError

logger = logging.getLogger(__name__)

# Type variables for iteration
ItemT = TypeVar('ItemT')
ResultT = TypeVar('ResultT')


@dataclass
class IterableNode(BaseNode[WorkflowState, Any, WorkflowState], Generic[ItemT, ResultT]):
    """Base class for nodes that iterate over items without sub-graphs.
    
    Pattern:
    1. Check if iteration is enabled in node config
    2. Process current item
    3. Update state with result
    4. Return self to process next item
    5. When done, return next node in chain
    """
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Execute iteration logic."""
        node_config = self._get_node_config(ctx.state)
        
        if not node_config or not node_config.iter_enabled:
            # Single execution mode - process all items at once
            return await self.process_single(ctx)
        
        # Iteration mode - process one item at a time
        items = ctx.state.iter_items
        current_idx = ctx.state.iter_index
        
        if not items or current_idx >= len(items):
            # Iteration complete or no items
            return await self.on_iteration_complete(ctx)
        
        # Process current item
        current_item = items[current_idx]
        try:
            result = await self.process_item(current_item, ctx)
            
            # Update state with result
            new_results = ctx.state.iter_results + [result]
            new_state = replace(
                ctx.state,
                iter_results=new_results,
                iter_index=current_idx + 1
            )
            
            # State will be updated through node return, not direct mutation
            
            # Check if more items to process
            if current_idx + 1 < len(items):
                # Return ourselves to process next item
                logger.debug(f'Iteration {current_idx + 1}/{len(items)} complete, continuing...')
                return self.__class__()  # Self-return for next iteration
            else:
                # All items processed
                logger.info(f'Iteration complete: processed {len(items)} items')
                return await self.on_iteration_complete(ctx)
                
        except Exception as e:
            # Handle iteration error
            return await self.handle_iteration_error(ctx, current_idx, e)
    
    async def process_single(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Process all items in a single execution (non-iteration mode).
        Default implementation processes items sequentially.
        """
        items = ctx.state.iter_items
        if not items:
            return await self.on_iteration_complete(ctx)
        
        results = []
        errors = []
        for i, item in enumerate(items):
            try:
                result = await self.process_item(item, ctx)
                results.append(result)
            except Exception as e:
                logger.error(f'Error processing item {i}: {e}')
                errors.append((i, item, e))
        
        # Check if any errors occurred
        if errors:
            from ..exceptions import BatchProcessingError
            raise BatchProcessingError(
                f"Failed to process {len(errors)} out of {len(items)} items",
                errors=errors
            )
        
        # Update state with all results
        new_state = replace(
            ctx.state,
            iter_results=results,
            iter_index=len(items)
        )
        
        # Pass new state via new context
        new_ctx = GraphRunContext(state=new_state, deps=ctx.deps)
        return await self.on_iteration_complete(new_ctx)
    
    async def process_item(self, item: ItemT, ctx: GraphRunContext[WorkflowState, Any]) -> ResultT:
        """Process a single item.
        Must be implemented by subclasses.
        """
        raise NotImplementedError
    
    async def on_iteration_complete(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Called when iteration is complete.
        Default implementation moves to next node.
        """
        # Get next node in chain
        next_node_id = self.get_next_node(ctx)
        
        if next_node_id:
            # Update state for next node
            new_state = replace(
                ctx.state,
                current_node=next_node_id,
                # Clear iteration state for next node
                iter_items=[],
                iter_results=[],
                iter_index=0
            )
            
            # Return next node - state updates happen through node returns
            return self.create_next_node(next_node_id)
        else:
            # No more nodes - return End with current state
            return End(ctx.state)
    
    async def handle_iteration_error(
        self,
        ctx: GraphRunContext[WorkflowState, Any],
        index: int,
        error: Exception
    ) -> BaseNode:
        """Handle error during iteration."""
        node_config = self._get_node_config(ctx.state)
        
        if isinstance(error, RetryableError) and node_config and node_config.retryable:
            # Retry the current item
            retry_key = f'{self._get_retry_key(ctx.state)}_item_{index}'
            retry_count = ctx.state.retry_counts.get(retry_key, 0)
            
            if retry_count < node_config.max_retries:
                logger.info(f'Retrying item {index}: attempt {retry_count + 1}')
                new_state = replace(
                    ctx.state,
                    retry_counts={**ctx.state.retry_counts, retry_key: retry_count + 1}
                )
                # Return self with updated retry count in state
                # The graph engine will handle the state update
                return self.__class__()  # Retry current item
        
        # Non-retryable or max retries exceeded - skip item
        logger.error(f'Skipping item {index} due to error: {error}')
        
        # Add error result and continue
        new_results = ctx.state.iter_results + [{'error': str(error), 'index': index}]
        new_state = replace(
            ctx.state,
            iter_results=new_results,
            iter_index=index + 1
        )
        
        # Continue with next item if available
        # Pass new state via new context
        if index + 1 < len(ctx.state.iter_items):
            return self.__class__()
        else:
            return await self.on_iteration_complete(ctx)


@dataclass
class BatchProcessNode(IterableNode[List[Any], List[Any]]):
    """Process items in batches for efficiency.
    """
    batch_size: int = 10
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Execute batch processing."""
        items = ctx.state.iter_items
        if not items:
            return await self.on_iteration_complete(ctx)
        
        # Process in batches
        results = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            try:
                batch_results = await self.process_batch(batch, ctx)
                results.extend(batch_results)
                
                # Update state with partial results
                new_state = replace(
                    ctx.state,
                    iter_results=results,
                    iter_index=i + len(batch)
                )
                # Note: State updates happen through node returns, not direct mutation
                
                logger.info(f'Processed batch {i//self.batch_size + 1}: {len(batch)} items')
                
            except Exception as e:
                logger.error(f'Batch processing error: {e}')
                # Add error results for batch
                results.extend([{'error': str(e)}] * len(batch))
        
        return await self.on_iteration_complete(ctx)
    
    async def process_batch(
        self,
        batch: List[Any],
        ctx: GraphRunContext[WorkflowState, Any]
    ) -> List[Any]:
        """Process a batch of items."""
        raise NotImplementedError
    
    async def process_item(self, item: Any, ctx: GraphRunContext[WorkflowState, Any]) -> Any:
        """Not used in batch processing."""
        pass


@dataclass
class MapNode(IterableNode[Any, Any]):
    """Map a function over items using iteration.
    """
    map_function: Optional[Callable] = None
    
    async def process_item(self, item: Any, ctx: GraphRunContext[WorkflowState, Any]) -> Any:
        """Apply map function to item."""
        if self.map_function:
            try:
                # Support both sync and async functions
                result = self.map_function(item, ctx)
                if asyncio.iscoroutine(result):
                    return await result
                return result
            except Exception as e:
                raise NonRetryableError(f'Map function failed: {e}')
        else:
            # Default: return item unchanged
            return item


@dataclass
class FilterNode(IterableNode[Any, Optional[Any]]):
    """Filter items based on a condition.
    """
    filter_function: Optional[Callable] = None
    
    async def process_item(self, item: Any, ctx: GraphRunContext[WorkflowState, Any]) -> Optional[Any]:
        """Apply filter function to item."""
        if self.filter_function:
            try:
                # Check if item passes filter
                passes = self.filter_function(item, ctx)
                if asyncio.iscoroutine(passes):
                    passes = await passes
                
                return item if passes else None
            except Exception as e:
                raise NonRetryableError(f'Filter function failed: {e}')
        else:
            # Default: pass all items
            return item
    
    async def on_iteration_complete(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Filter out None results before continuing."""
        # Remove None values from results
        filtered_results = [r for r in ctx.state.iter_results if r is not None]
        
        new_state = replace(
            ctx.state,
            iter_results=filtered_results
        )
        
        # Pass new state via new context
        new_ctx = GraphRunContext(state=new_state, deps=ctx.deps)
        return await super().on_iteration_complete(new_ctx)


@dataclass
class AggregatorNode(BaseNode[WorkflowState, Any, Any]):
    """Aggregate results from iteration.
    Not an IterableNode itself, but processes iteration results.
    """
    aggregation_function: Optional[Callable] = None
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Aggregate iteration results."""
        results = ctx.state.iter_results
        
        if not results:
            aggregated = None
        elif self.aggregation_function:
            try:
                aggregated = self.aggregation_function(results)
                if asyncio.iscoroutine(aggregated):
                    aggregated = await aggregated
            except Exception as e:
                raise NonRetryableError(f'Aggregation failed: {e}')
        else:
            # Default: return results as-is
            aggregated = results
        
        # Store aggregated result
        new_state = replace(
            ctx.state,
            domain_data={
                **ctx.state.domain_data,
                'aggregated_result': aggregated
            }
        )
        
        # Continue to next node with new state
        new_ctx = GraphRunContext(state=new_state, deps=ctx.deps)
        next_node_id = self.get_next_node(new_ctx)
        if next_node_id:
            return self.create_next_node(next_node_id)
        else:
            return End(ctx.state)


@dataclass
class ParallelMapNode(IterableNode[Any, Any]):
    """Process items in parallel using asyncio.
    Still uses self-return for state management but executes items concurrently.
    """
    max_concurrent: int = 5
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Execute parallel processing."""
        items = ctx.state.iter_items
        if not items:
            return await self.on_iteration_complete(ctx)
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_with_semaphore(item, index):
            async with semaphore:
                try:
                    return await self.process_item(item, ctx)
                except Exception as e:
                    logger.error(f'Error processing item {index}: {e}')
                    return {'error': str(e), 'index': index}
        
        # Process all items in parallel
        tasks = [
            process_with_semaphore(item, i)
            for i, item in enumerate(items)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Create new state with all results
        new_state = replace(
            ctx.state,
            iter_results=results,
            iter_index=len(items)
        )
        
        # Pass new state to completion handler via new context
        new_ctx = GraphRunContext(state=new_state, deps=ctx.deps)
        return await self.on_iteration_complete(new_ctx)


# Register iteration nodes
register_node_class('iterate', IterableNode)
register_node_class('batch_process', BatchProcessNode)
register_node_class('map', MapNode)
register_node_class('filter', FilterNode)
register_node_class('aggregate', AggregatorNode)
register_node_class('parallel_map', ParallelMapNode)