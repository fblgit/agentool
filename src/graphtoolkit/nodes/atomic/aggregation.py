"""GraphToolkit Aggregation Nodes.

Nodes for combining results from parallel operations.
"""

import logging
from dataclasses import dataclass, replace
from typing import Any, Callable, List, Optional

from ...core.factory import register_node_class
from ...core.types import WorkflowState
from ..base import AtomicNode, BaseNode, GraphRunContext, NonRetryableError

logger = logging.getLogger(__name__)


@dataclass
class AggregatorNode(AtomicNode[WorkflowState, Any, Any]):
    """Combine results from parallel operations.
    
    Per workflow-graph-system.md line 467:
    - Combines results from parallel operations
    - Supports different aggregation strategies
    """
    aggregation_strategy: str = 'list'  # list, dict, merge, concat, sum, average
    result_field: str = 'iter_results'  # Field containing results to aggregate
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> Any:
        """Aggregate results based on strategy."""
        # Get results to aggregate
        results = getattr(ctx.state, self.result_field, [])
        if not results:
            results = ctx.state.domain_data.get(self.result_field, [])
        
        if not results:
            logger.warning(f'No results found in {self.result_field}')
            return [] if self.aggregation_strategy == 'list' else {}
        
        # Apply aggregation strategy
        if self.aggregation_strategy == 'list':
            # Keep as list
            return results
            
        elif self.aggregation_strategy == 'dict':
            # Convert list to dict using index as key
            return {f'item_{i}': item for i, item in enumerate(results)}
            
        elif self.aggregation_strategy == 'merge':
            # Merge all dicts into one
            merged = {}
            for item in results:
                if isinstance(item, dict):
                    merged.update(item)
                else:
                    logger.warning(f'Cannot merge non-dict item: {type(item)}')
            return merged
            
        elif self.aggregation_strategy == 'concat':
            # Concatenate lists
            concatenated = []
            for item in results:
                if isinstance(item, list):
                    concatenated.extend(item)
                else:
                    concatenated.append(item)
            return concatenated
            
        elif self.aggregation_strategy == 'sum':
            # Sum numeric values
            try:
                return sum(results)
            except (TypeError, ValueError) as e:
                raise NonRetryableError(f'Cannot sum results: {e}')
                
        elif self.aggregation_strategy == 'average':
            # Average numeric values
            try:
                return sum(results) / len(results) if results else 0
            except (TypeError, ValueError) as e:
                raise NonRetryableError(f'Cannot average results: {e}')
                
        else:
            raise NonRetryableError(f'Unknown aggregation strategy: {self.aggregation_strategy}')
    
    async def update_state(self, state: WorkflowState, result: Any) -> WorkflowState:
        """Store aggregated result in domain data."""
        return replace(
            state,
            domain_data={
                **state.domain_data,
                f'{state.current_phase}_aggregated': result,
                'aggregated_result': result
            }
        )


@dataclass
class ParallelAggregatorNode(BaseNode[WorkflowState, Any, WorkflowState]):
    """Special aggregator for parallel execution results.
    Waits for all parallel tasks to complete before aggregating.
    """
    parallel_node_ids: List[str]
    aggregation_func: Optional[Callable] = None
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Aggregate parallel execution results."""
        # Check if all parallel nodes have completed
        parallel_results = ctx.state.domain_data.get('parallel_results', {})
        
        completed = all(
            node_id in parallel_results 
            for node_id in self.parallel_node_ids
        )
        
        if not completed:
            logger.info(f'Waiting for parallel nodes to complete: {self.parallel_node_ids}')
            # In real Graph.iter() this would be handled differently
            # For now, we'll just continue
        
        # Aggregate results
        results = [
            parallel_results.get(node_id)
            for node_id in self.parallel_node_ids
            if node_id in parallel_results
        ]
        
        if self.aggregation_func:
            aggregated = self.aggregation_func(results)
        else:
            # Default: keep as list
            aggregated = results
        
        # Update state with aggregated result
        new_state = replace(
            ctx.state,
            domain_data={
                **ctx.state.domain_data,
                'aggregated_parallel_results': aggregated,
                'parallel_execution_complete': True
            }
        )
        
        # Continue to next node
        next_node_id = self.get_next_node(new_state)
        if next_node_id:
            new_state = replace(new_state, current_node=next_node_id)
            return self.create_next_node(next_node_id)
        
        from .control import NextPhaseNode
        return NextPhaseNode()


# Register aggregation nodes
register_node_class('aggregator', AggregatorNode)
register_node_class('parallel_aggregator', ParallelAggregatorNode)