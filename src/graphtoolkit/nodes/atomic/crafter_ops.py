"""GraphToolkit Crafter Operation Nodes.

V1-compatible crafter iteration nodes that exactly replicate workflow_crafter.py behavior.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from ...core.factory import register_node_class
from ...core.types import WorkflowState
from ..base import BaseNode, GraphRunContext, NonRetryableError
from ..iteration import IterableNode

logger = logging.getLogger(__name__)


@dataclass
class PrepareCrafterIterationNode(BaseNode[WorkflowState, Any, None]):
    """V1-Compatible crafter preparation node.
    
    Mimics V1 workflow_crafter.py lines 95-136:
    - Loads specifications from storage
    - Prepares iteration state
    - Sets up for per-tool code generation
    """
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Prepare crafter iteration by loading specifications."""
        logger.info(f"Preparing crafter iteration for workflow {ctx.state.workflow_id}")
        
        from ...core.initialization import ensure_graphtoolkit_initialized
        from agentool.core.injector import get_injector
        
        ensure_graphtoolkit_initialized()
        injector = get_injector()
        
        try:
            # V1 lines 95-108: Load specifications from storage_kv
            specs_key = f'workflow/{ctx.state.workflow_id}/specs'
            logger.info(f"Loading specifications from {specs_key}")
            
            specs_result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': specs_key
            })
            
            if not specs_result.success or not specs_result.data.get('exists'):
                raise NonRetryableError(f'No specifications found at {specs_key}')
            
            # V1 lines 110-115: Parse SpecificationOutput
            try:
                from agents.models import SpecificationOutput
                spec_output = SpecificationOutput(**json.loads(specs_result.data['value']))
                specifications = [spec.model_dump() for spec in spec_output.specifications]
            except ImportError:
                # Fallback without V1 models
                spec_data = json.loads(specs_result.data['value'])
                specifications = spec_data.get('specifications', [])
            
            # V1 lines 117-122: Early exit if no specifications
            if not specifications:
                logger.info("No specifications to craft")
                ctx.state.domain_data.update({
                    'implementations': {},
                    'no_specifications': True
                })
                # Skip to save_output
                from ...core.factory import create_node_instance
                return create_node_instance('save_output')
            
            # V1 lines 124-136: Log and prepare iteration
            logger.info(f"Loaded {len(specifications)} specifications for crafting")
            
            # V1: Store individual specifications for template reference
            for spec in specifications:
                tool_name = spec.get('name', 'unknown')
                spec_key = f'workflow/{ctx.state.workflow_id}/specifications/{tool_name}'
                await injector.run('storage_kv', {
                    'operation': 'set',
                    'key': spec_key,
                    'value': json.dumps(spec)
                })
            
            # Setup iteration state for CrafterToolIteratorNode
            ctx.state.iter_items = specifications
            ctx.state.iter_index = 0
            ctx.state.iter_results = []
            
            # Store specifications in domain_data for template access
            ctx.state.domain_data['specifications'] = specifications
            
            # Chain to iterator
            from ...core.factory import create_node_instance
            return create_node_instance('crafter_tool_iterator')
            
        except Exception as e:
            logger.error(f'Failed to prepare crafter iteration: {e}')
            raise NonRetryableError(f'Crafter preparation failed: {e}')


@dataclass
class CrafterToolIteratorNode(IterableNode[Dict[str, Any], Dict[str, Any]]):
    """V1-Compatible crafter tool iterator.
    
    Mimics exact V1 behavior from workflow_crafter.py craft_single_tool():
    - Processes each specification individually with LLM
    - Uses skeleton templates for code generation
    - Stores implementations with V1 storage patterns
    - Tracks metrics per tool
    """
    
    async def prepare_iteration(self, ctx: GraphRunContext[WorkflowState, Any]) -> None:
        """Prepare for iteration - specifications already loaded by PrepareCrafterIterationNode."""
        # Specifications are already in ctx.state.iter_items
        logger.info(f'Starting crafter iteration for {len(ctx.state.iter_items)} tools')
    
    async def process_item(
        self,
        specification: Dict[str, Any],
        ctx: GraphRunContext[WorkflowState, Any]
    ) -> Dict[str, Any]:
        """Process a single specification to generate implementation (V1 craft_single_tool)."""
        tool_name = specification.get('name', 'unknown')
        logger.info(f'Crafting implementation for tool: {tool_name}')
        
        from ...core.initialization import ensure_graphtoolkit_initialized
        from agentool.core.injector import get_injector
        import json
        
        ensure_graphtoolkit_initialized()
        injector = get_injector()
        
        try:
            # V1 lines 181-187: Store missing tool data for template reference
            missing_tool_key = f'workflow/{ctx.state.workflow_id}/current_missing_tool_for_craft/{tool_name}'
            
            # Load missing tool from analysis (if available)
            missing_tool_data = {'name': tool_name}
            try:
                analysis_key = f'workflow/{ctx.state.workflow_id}/analysis'
                analysis_result = await injector.run('storage_kv', {
                    'operation': 'get',
                    'key': analysis_key
                })
                
                if analysis_result.success and analysis_result.data.get('exists'):
                    from agents.models import AnalyzerOutput
                    analysis = AnalyzerOutput(**json.loads(analysis_result.data['value']))
                    
                    # Find matching missing tool
                    for mt in analysis.missing_tools:
                        if mt.name == tool_name:
                            missing_tool_data = mt.model_dump()
                            break
            except (ImportError, Exception):
                # Use basic missing tool data if no V1 models
                pass
            
            await injector.run('storage_kv', {
                'operation': 'set',
                'key': missing_tool_key,
                'value': json.dumps(missing_tool_data)
            })
            
            # V1 lines 150-162: Load and render skeleton template 
            skeleton_result = await injector.run('templates', {
                'operation': 'render',
                'template_name': 'skeletons/agentool_comprehensive',  # V1 EXACT path
                'variables': {
                    'tool_name': tool_name
                }
            })
            
            if not skeleton_result.success:
                raise NonRetryableError(f'Skeleton template render failed: {skeleton_result.message}')
            
            skeleton_code = skeleton_result.data.get('rendered', '')
            
            # V1 lines 189-195: Store skeleton for template reference
            skeleton_key = f'workflow/{ctx.state.workflow_id}/skeleton/{tool_name}'
            await injector.run('storage_kv', {
                'operation': 'set',
                'key': skeleton_key,
                'value': skeleton_code
            })
            
            # V1 lines 164-178: Load system template (V1 uses no variables)
            system_result = await injector.run('templates', {
                'operation': 'render',
                'template_name': 'system/crafter',  # V1 EXACT path
                'variables': {}  # V1 system template uses no variables
            })
            
            if not system_result.success:
                raise NonRetryableError(f'System template render failed: {system_result.message}')
            
            system_prompt = system_result.data.get('rendered', '')
            
            # V1 lines 197-209: Render user template with V1 EXACT references
            user_result = await injector.run('templates', {
                'operation': 'render',
                'template_name': 'prompts/craft_implementation',  # V1 EXACT path
                'variables': {
                    'agentool_to_implement': f'!ref:storage_kv:{missing_tool_key}',
                    'existing_tools_schemas': f'!ref:storage_kv:workflow/{ctx.state.workflow_id}/existing_tools',
                    'spec_output': f'!ref:storage_kv:workflow/{ctx.state.workflow_id}/specifications/{tool_name}',
                    'all_specifications': f'!ref:storage_kv:workflow/{ctx.state.workflow_id}/specs',  # V1 passes ALL specs
                    'analysis_output': f'!ref:storage_kv:workflow/{ctx.state.workflow_id}/analysis',
                    'skeleton': f'!ref:storage_kv:{skeleton_key}'
                }
            })
            
            if not user_result.success:
                raise NonRetryableError(f'User template render failed: {user_result.message}')
            
            user_prompt = user_result.data.get('rendered', '')
            
            # V1 lines 305-314: Create LLM agent with V1 configuration
            try:
                from pydantic_ai import Agent
                
                # Get model from phase config or use V1 default
                model_name = 'openai:gpt-4o'  # V1 default model
                
                # V1 uses string output type, not structured output
                agent = Agent(
                    model_name,
                    output_type=str,  # V1 uses raw string output for code
                    system_prompt=system_prompt
                )
                
                # V1 lines 316-317: Generate implementation
                result = await agent.run(user_prompt)
                raw_output = result.output
                
                # V1 lines 318-325: Extract code from markdown code block
                import re
                code_match = re.search(r'```python\n(.*?)```', raw_output, re.DOTALL)
                if code_match:
                    generated_code = code_match.group(1).strip()
                else:
                    # Fallback if no code block found
                    generated_code = raw_output.strip()
                
                # V1 lines 326-329: Create CodeOutput object
                try:
                    from agents.models import CodeOutput
                    implementation = CodeOutput(
                        code=generated_code,
                        file_path=f"{tool_name}.py"
                    )
                except ImportError:
                    # Fallback without V1 models
                    implementation = {
                        'code': generated_code,
                        'file_path': f"{tool_name}.py"
                    }
                
                # V1 lines 319-350: Track token metrics
                usage = result.usage()
                await self._track_token_metrics(ctx, tool_name, usage, model_name)
                
                # V1 lines 352-356: Store individual implementation
                impl_key = f'workflow/{ctx.state.workflow_id}/implementations/{tool_name}'
                impl_data = implementation.model_dump() if hasattr(implementation, 'model_dump') else implementation
                await injector.run('storage_kv', {
                    'operation': 'set',
                    'key': impl_key,
                    'value': json.dumps(impl_data)
                })
                
                # V1 lines 358-371: Store code file - V1 EXACT path pattern
                code_path = f'generated/{ctx.state.workflow_id}/src/{tool_name}.py'  # V1 uses src/ subdirectory
                code_content = implementation.code if hasattr(implementation, 'code') else implementation.get('code', '')
                await injector.run('storage_fs', {
                    'operation': 'write',
                    'path': code_path,
                    'content': code_content,
                    'create_parents': True
                })
                
                logger.info(f'Generated implementation for {tool_name}: {len(code_content)} chars')
                
                # V1 return format matching craft_single_tool return
                return {
                    'tool_name': tool_name,
                    'file_path': code_path,  # V1 uses 'file_path' not 'path' 
                    'state_key': impl_key,   # V1 tracks storage key
                    'code_output': implementation  # V1 includes full CodeOutput object
                }
                
            except ImportError:
                # Fallback without V1 models
                logger.warning('V1 models not available, using simplified implementation')
                return {
                    'name': tool_name,
                    'code': f'# Implementation for {tool_name}\n# Generated without V1 models',
                    'imports': [],
                    'dependencies': [],
                    'error': 'V1 models not available'
                }
                
        except Exception as e:
            logger.error(f'Failed to craft tool {tool_name}: {e}')
            # V1 continues processing other tools on failure
            return {
                'name': tool_name,
                'error': str(e),
                'failed': True
            }
    
    async def _track_token_metrics(self, ctx: GraphRunContext[WorkflowState, Any], tool_name: str, usage, model: str) -> None:
        """Track token metrics per tool (V1 lines 319-350)."""
        from agentool.core.injector import get_injector
        injector = get_injector()
        
        try:
            # V1 EXACT metrics pattern
            labels = {
                'workflow_id': ctx.state.workflow_id,
                'agent': 'workflow_crafter',  # V1 exact agent name
                'tool': tool_name,
                'model': model
            }
            
            await injector.run('metrics', {
                'operation': 'increment',
                'name': 'agentool.workflow.tokens.request',
                'value': usage.request_tokens,
                'labels': labels
            })
            
            await injector.run('metrics', {
                'operation': 'increment',
                'name': 'agentool.workflow.tokens.response',
                'value': usage.response_tokens,
                'labels': labels
            })
            
            await injector.run('metrics', {
                'operation': 'increment',
                'name': 'agentool.workflow.tokens.total',
                'value': usage.total_tokens,
                'labels': labels
            })
            
        except Exception as e:
            logger.warning(f'Failed to track metrics for {tool_name}: {e}')
    
    async def on_iteration_complete(self, ctx: GraphRunContext[WorkflowState, Any]) -> 'BaseNode':
        """Complete iteration and prepare final output (V1 lines 391-423)."""
        from agentool.core.injector import get_injector
        import json
        
        injector = get_injector()
        
        try:
            # Filter out failed implementations
            successful_impls = [impl for impl in ctx.state.iter_results if not impl.get('failed', False)]
            
            # V1 lines 391-408: Create summary data (V1 format)
            summary_data = {
                'implementations': [{
                    'tool_name': impl['tool_name'],
                    'file_path': impl['file_path'],
                    'state_key': impl['state_key'],
                    'lines': impl['code_output'].code.count('\n') if hasattr(impl['code_output'], 'code') else 0
                } for impl in successful_impls],
                'total_tools': len(successful_impls),
                'total_lines': sum(
                    impl['code_output'].code.count('\n') if hasattr(impl['code_output'], 'code') else 0 
                    for impl in successful_impls
                ),
                'files': [impl['file_path'] for impl in successful_impls]
            }
            
            # V1 lines 410-415: Store implementations summary (V1 EXACT pattern)
            summary_key = f'workflow/{ctx.state.workflow_id}/implementations_summary'
            await injector.run('storage_kv', {
                'operation': 'set',
                'key': summary_key,
                'value': json.dumps(summary_data)
            })
            
            logger.info(f'Stored {len(successful_impls)} implementations at {summary_key}')
            
            # Update domain data for next phase (V1 format)
            ctx.state.domain_data.update({
                'implementations_summary': summary_data,
                'implementation_count': len(successful_impls),
                'failed_tools': len([impl for impl in ctx.state.iter_results if impl.get('failed', False)]),
                'total_tools': summary_data['total_tools'],
                'total_lines': summary_data['total_lines']
            })
            
            return await super().on_iteration_complete(ctx)
            
        except Exception as e:
            logger.error(f'Failed to complete crafter iteration: {e}')
            raise NonRetryableError(f'Crafter completion failed: {e}')


@dataclass
class SaveImplementationSummaryNode(BaseNode[WorkflowState, Any, None]):
    """V1-Compatible implementation summary saver.
    
    Saves final implementation summary to match V1 workflow_crafter.py behavior.
    """
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Save implementation summary data."""
        logger.info(f"Saving implementation summary for workflow {ctx.state.workflow_id}")
        
        from ...core.initialization import ensure_graphtoolkit_initialized
        from agentool.core.injector import get_injector
        
        ensure_graphtoolkit_initialized()
        injector = get_injector()
        
        try:
            # Get summary from domain_data (set by crafter_tool_iterator)
            summary_data = ctx.state.domain_data.get('implementations_summary')
            
            if not summary_data:
                logger.warning("No implementation summary data found")
                summary_data = {
                    'implementations': [],
                    'total_tools': 0,
                    'total_lines': 0,
                    'files': []
                }
            
            # V1 EXACT storage pattern for implementations_summary  
            summary_key = f'workflow/{ctx.state.workflow_id}/implementations_summary'
            await injector.run('storage_kv', {
                'operation': 'set',
                'key': summary_key,
                'value': json.dumps(summary_data)
            })
            
            logger.info(f'Stored implementation summary with {summary_data.get("total_tools", 0)} tools')
            
            # Chain to next node
            from ...core.factory import create_node_instance
            return create_node_instance('state_update')
            
        except Exception as e:
            logger.error(f'Failed to save implementation summary: {e}')
            raise NonRetryableError(f'Implementation summary save failed: {e}')


# Register crafter nodes
register_node_class('prepare_crafter_iteration', PrepareCrafterIterationNode)
register_node_class('crafter_tool_iterator', CrafterToolIteratorNode)
register_node_class('save_implementation_summary', SaveImplementationSummaryNode)