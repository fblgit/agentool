"""GraphToolkit Iteration Atomic Nodes.

Atomic nodes for handling iteration over collections in workflows.
Enables phases to process multiple items (e.g., multiple missing tools).
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ...core.factory import register_node_class
from ...core.types import WorkflowState
from ..base import AtomicNode, BaseNode, GraphRunContext, NonRetryableError

logger = logging.getLogger(__name__)


@dataclass
class IterationControlNode(BaseNode[WorkflowState, Any, WorkflowState]):
    """Controls iteration over a list of items.
    
    This node manages iteration state and determines whether to:
    1. Process the next item (return to template_render)
    2. Complete iteration (move to aggregation)
    
    The iteration state is tracked in domain_data:
    - {phase}_iteration_items: List of items to process
    - {phase}_iteration_index: Current index
    - {phase}_iteration_current: Current item being processed
    """
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Manage iteration control flow."""
        phase_name = ctx.state.current_phase
        logger.info(f"[IterationControlNode] Phase: {phase_name}")
        
        # Get phase definition to check iteration config
        phase_def = ctx.state.get_current_phase_def()
        if not phase_def:
            raise NonRetryableError(f'Phase {phase_name} not found')
        
        # Check if this phase has iteration config
        if not hasattr(phase_def, 'iteration_config') or not phase_def.iteration_config:
            logger.info(f"[IterationControlNode] No iteration config for {phase_name}, skipping")
            # Skip to aggregation
            from ...core.factory import create_node_instance
            return create_node_instance('aggregation')
        
        iter_config = phase_def.iteration_config
        if not iter_config.get('enabled', False):
            logger.info(f"[IterationControlNode] Iteration disabled for {phase_name}")
            from ...core.factory import create_node_instance
            return create_node_instance('aggregation')
        
        # Get items to iterate over
        iter_key = f"{phase_name}_iteration"
        
        # Initialize iteration if not started
        if f"{iter_key}_items" not in ctx.state.domain_data:
            items = self._get_iteration_items(ctx, iter_config)
            if not items:
                logger.info(f"[IterationControlNode] No items to iterate for {phase_name}")
                from ...core.factory import create_node_instance
                return create_node_instance('aggregation')
            
            logger.info(f"[IterationControlNode] Initializing iteration with {len(items)} items")
            ctx.state.domain_data[f"{iter_key}_items"] = items
            ctx.state.domain_data[f"{iter_key}_index"] = 0
            ctx.state.domain_data[f"{iter_key}_results"] = []
        
        items = ctx.state.domain_data[f"{iter_key}_items"]
        current_index = ctx.state.domain_data[f"{iter_key}_index"]
        
        logger.info(f"[IterationControlNode] Processing item {current_index + 1}/{len(items)}")
        
        if current_index < len(items):
            # Set current item for processing
            current_item = items[current_index]
            ctx.state.domain_data[f"{iter_key}_current"] = current_item
            logger.info(f"[IterationControlNode] Set current item: {current_item.get('name', current_item) if isinstance(current_item, dict) else current_item}")
            
            # Return to template_render to process this item
            from ...core.factory import create_node_instance
            return create_node_instance('template_render')
        else:
            # Iteration complete
            logger.info(f"[IterationControlNode] Iteration complete for {phase_name}")
            from ...core.factory import create_node_instance
            return create_node_instance('aggregation')
    
    def _get_iteration_items(self, ctx: GraphRunContext[WorkflowState, Any], iter_config: Dict[str, Any]) -> List[Any]:
        """Extract items to iterate over from domain_data."""
        items_source = iter_config.get('items_source', '')
        logger.debug(f"[IterationControlNode] Getting items from source: {items_source}")
        
        # Navigate through dot notation (e.g., 'analyzer_output.missing_tools')
        parts = items_source.split('.')
        data = ctx.state.domain_data
        
        for part in parts:
            if isinstance(data, dict):
                data = data.get(part)
            elif hasattr(data, part):
                data = getattr(data, part)
            else:
                logger.warning(f"[IterationControlNode] Could not access {part} in {type(data).__name__}")
                return []
            
            if data is None:
                logger.warning(f"[IterationControlNode] {part} is None")
                return []
        
        # Ensure we have a list
        if not isinstance(data, list):
            logger.warning(f"[IterationControlNode] Expected list, got {type(data).__name__}")
            return []
        
        logger.info(f"[IterationControlNode] Found {len(data)} items to iterate")
        return data


@dataclass
class SaveIterationOutputNode(BaseNode[WorkflowState, Any, WorkflowState]):
    """Saves individual iteration output.
    
    Stores both the rendered template and the LLM output for the current
    iteration item, then increments the iteration index.
    
    Returns IterationControlNode to continue or complete iteration.
    """
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Save the current iteration's output."""
        phase_name = ctx.state.current_phase
        phase_def = ctx.state.get_current_phase_def()
        
        if not phase_def:
            raise NonRetryableError(f'Phase {phase_name} not found')
        
        iter_config = phase_def.iteration_config if hasattr(phase_def, 'iteration_config') else {}
        iter_key = f"{phase_name}_iteration"
        
        # Get current item and output
        current_item = ctx.state.domain_data.get(f"{iter_key}_current")
        if not current_item:
            logger.warning(f"[SaveIterationOutputNode] No current item for {phase_name}")
            return "skipped"
        
        # Get the LLM response (should have been validated already)
        llm_output = ctx.state.domain_data.get(f"{phase_name}_llm_response")
        if not llm_output:
            logger.warning(f"[SaveIterationOutputNode] No LLM output for {phase_name}")
            return "skipped"
        
        # Determine item name for storage
        if isinstance(current_item, dict):
            item_name = current_item.get('name', f'item_{ctx.state.domain_data[f"{iter_key}_index"]}')
        else:
            item_name = f'item_{ctx.state.domain_data[f"{iter_key}_index"]}'
        
        logger.info(f"[SaveIterationOutputNode] Saving output for item: {item_name}")
        
        # Generate storage keys
        item_storage_pattern = iter_config.get('item_storage_pattern', f'workflow/{{workflow_id}}/iteration/{phase_name}/{{item_name}}')
        spec_key = item_storage_pattern.format(
            workflow_id=ctx.state.workflow_id,
            item_name=item_name
        )
        render_key = f"workflow/{ctx.state.workflow_id}/render/{phase_name}/{item_name}"
        
        # Get storage client
        storage_client = ctx.deps.get_storage_client()
        
        # Save specification
        spec_data = llm_output.model_dump() if hasattr(llm_output, 'model_dump') else llm_output
        spec_result = await storage_client.run('storage_kv', {
            'operation': 'set',
            'key': spec_key,
            'value': spec_data,
            'namespace': 'workflow'
        })
        
        if spec_result.success:
            logger.info(f"[SaveIterationOutputNode] Saved specification to {spec_key}")
        else:
            logger.error(f"[SaveIterationOutputNode] Failed to save specification: {spec_result.message}")
        
        # Save rendered template if available
        rendered = ctx.state.domain_data.get('rendered_prompts')
        if rendered:
            render_result = await storage_client.run('storage_kv', {
                'operation': 'set',
                'key': render_key,
                'value': rendered,
                'namespace': 'workflow'
            })
            
            if render_result.success:
                logger.info(f"[SaveIterationOutputNode] Saved render to {render_key}")
            else:
                logger.error(f"[SaveIterationOutputNode] Failed to save render: {render_result.message}")
        
        # Add to results list
        results = ctx.state.domain_data.get(f"{iter_key}_results", [])
        results.append({
            'item': current_item,
            'output': spec_data,
            'storage_key': spec_key
        })
        ctx.state.domain_data[f"{iter_key}_results"] = results
        
        # Increment iteration index
        current_index = ctx.state.domain_data.get(f"{iter_key}_index", 0)
        ctx.state.domain_data[f"{iter_key}_index"] = current_index + 1
        logger.info(f"[SaveIterationOutputNode] Incremented index to {current_index + 1}")
        
        # Return to IterationControlNode to check for more items
        logger.info(f"[SaveIterationOutputNode] Returning to IterationControlNode")
        return IterationControlNode()


@dataclass
class AggregationNode(AtomicNode[WorkflowState, Any, Dict[str, Any]]):
    """Aggregates iteration results.
    
    Collects all individual outputs from iteration and creates
    an aggregated result that represents the complete phase output.
    """
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> Dict[str, Any]:
        """Aggregate all iteration results."""
        phase_name = ctx.state.current_phase
        phase_def = ctx.state.get_current_phase_def()
        
        if not phase_def:
            raise NonRetryableError(f'Phase {phase_name} not found')
        
        iter_key = f"{phase_name}_iteration"
        logger.info(f"[AggregationNode] Aggregating results for {phase_name}")
        
        # Check if we have iteration results
        if f"{iter_key}_results" not in ctx.state.domain_data:
            logger.info(f"[AggregationNode] No iteration results, checking for direct output")
            # No iteration was performed, might be a non-iterating phase
            return {}
        
        results = ctx.state.domain_data[f"{iter_key}_results"]
        logger.info(f"[AggregationNode] Found {len(results)} iteration results")
        
        # For specifier phase, aggregate into SpecificationOutput format
        if phase_name == 'specifier':
            specifications = []
            for result in results:
                output = result.get('output', {})
                # Convert from ToolSpecificationLLM to ToolSpecification if needed
                if 'tool_input_schema' in output and 'tool_output_schema' in output:
                    # It's a ToolSpecificationLLM, convert it
                    spec = {
                        **output,
                        'input_schema': output.pop('tool_input_schema'),
                        'output_schema': output.pop('tool_output_schema')
                    }
                    specifications.append(spec)
                else:
                    specifications.append(output)
            
            aggregated = {
                'specifications': specifications
            }
        else:
            # Generic aggregation
            aggregated = {
                'items': [r['output'] for r in results],
                'count': len(results)
            }
        
        # Store aggregated result
        storage_client = ctx.deps.get_storage_client()
        aggregated_key = f"workflow/{ctx.state.workflow_id}/specifications"
        
        agg_result = await storage_client.run('storage_kv', {
            'operation': 'set',
            'key': aggregated_key,
            'value': aggregated,
            'namespace': 'workflow'
        })
        
        if agg_result.success:
            logger.info(f"[AggregationNode] Saved aggregated result to {aggregated_key}")
        else:
            logger.error(f"[AggregationNode] Failed to save aggregated result: {agg_result.message}")
        
        return aggregated
    
    async def update_state_in_place(self, state: WorkflowState, result: Dict[str, Any]) -> None:
        """Update state with aggregated output."""
        phase_name = state.current_phase
        # Store as phase output
        state.domain_data[f"{phase_name}_output"] = result
        logger.info(f"[AggregationNode] Stored aggregated output as {phase_name}_output")


# Register iteration nodes
register_node_class('iteration_control', IterationControlNode)
register_node_class('save_iteration_output', SaveIterationOutputNode)
register_node_class('aggregation', AggregationNode)