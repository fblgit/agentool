"""GraphToolkit Template Atomic Nodes.

Template rendering nodes that integrate with the existing template system.
"""

import logging
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, List

from ...core.factory import register_node_class
from ...core.types import WorkflowState
from ..base import AtomicNode, GraphRunContext, NonRetryableError

logger = logging.getLogger(__name__)


@dataclass
class TemplateRenderNode(AtomicNode[WorkflowState, Any, Dict[str, str]]):
    """Render templates for the current phase.
    Deterministic operation - no retry needed.
    """
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> Dict[str, str]:
        """Render system and user templates using agentoolkit."""
        # Get phase definition from state (state-driven pattern)
        phase_def = ctx.state.get_current_phase_def()
        if not phase_def:
            raise NonRetryableError(f'Phase {ctx.state.current_phase} not found')
        
        if not phase_def.templates:
            # No templates for this phase, return empty
            logger.info(f'No templates configured for {ctx.state.current_phase}')
            return {}
        
        # Prepare template variables
        variables = await self._prepare_variables(ctx)
        
        # For crafter phase, render the skeleton if needed
        if ctx.state.current_phase == 'crafter' and '_tool_name_for_skeleton' in variables:
            tool_name = variables.pop('_tool_name_for_skeleton')
            try:
                from agentool.core.injector import get_injector
                injector = get_injector()
                skeleton_result = await injector.run('templates', {
                    'operation': 'render',
                    'template_name': 'agentool/skeletons/agentool_comprehensive',
                    'variables': {'tool_name': tool_name}
                })
                
                if hasattr(skeleton_result, 'output'):
                    # It's an AgentRunResult - parse the output
                    import json
                    try:
                        output_data = json.loads(skeleton_result.output)
                        if output_data.get('success'):
                            variables['skeleton'] = output_data.get('data', {}).get('rendered', '')
                            logger.debug(f"[TemplateRenderNode] Rendered skeleton for {tool_name}")
                        else:
                            logger.warning(f"[TemplateRenderNode] Failed to render skeleton: {output_data.get('message', 'unknown error')}")
                            variables['skeleton'] = ''
                    except json.JSONDecodeError as e:
                        logger.warning(f"[TemplateRenderNode] Failed to parse skeleton result: {e}")
                        variables['skeleton'] = ''
                elif hasattr(skeleton_result, 'success') and skeleton_result.success:
                    if skeleton_result.data and isinstance(skeleton_result.data, dict):
                        variables['skeleton'] = skeleton_result.data.get('rendered', '')
                        logger.debug(f"[TemplateRenderNode] Rendered skeleton for {tool_name}")
                    else:
                        logger.warning(f"[TemplateRenderNode] No skeleton content in result")
                        variables['skeleton'] = ''
                else:
                    error_msg = skeleton_result.message if hasattr(skeleton_result, 'message') else 'unknown error'
                    logger.error(f"[TemplateRenderNode] Failed to render skeleton: {error_msg}")
                    raise NonRetryableError(f"Failed to render skeleton: {error_msg}")
            except Exception as e:
                logger.error(f"[TemplateRenderNode] Could not render skeleton: {e}")
                raise NonRetryableError(f"Could not render skeleton: {e}")
        
        # Use the existing template system through injector
        from ...core.initialization import ensure_graphtoolkit_initialized
        from agentool.core.injector import get_injector
        
        # Ensure initialization before using injector
        ensure_graphtoolkit_initialized()
        injector = get_injector()
        
        rendered = {}
        
        # Render system template
        if phase_def.templates.system_template:
            try:
                # Template name should be the path without extension (e.g., "smoke/system/analyzer")
                template_name = phase_def.templates.system_template.replace('.jinja', '').replace('.j2', '')
                result = await injector.run('templates', {
                    'operation': 'render',
                    'template_name': template_name,
                    'variables': variables
                })
                
                # The injector might return AgentRunResult or TemplatesOutput
                logger.debug(f"[TemplateRenderNode] System template result type: {type(result).__name__}")
                
                # Handle both AgentRunResult and TemplatesOutput
                if hasattr(result, 'output'):
                    # It's an AgentRunResult - parse the output
                    import json
                    try:
                        output_data = json.loads(result.output)
                        if output_data.get('success'):
                            rendered['system_prompt'] = output_data.get('data', {}).get('rendered', '')
                            logger.info(f'Rendered system template for {ctx.state.current_phase}')
                        else:
                            logger.error(f'Template render failed: {output_data.get("message", "unknown error")}')
                            from ...exceptions import TemplateError
                            raise TemplateError(f'System template render failed: {output_data.get("message", "unknown error")}')
                    except json.JSONDecodeError as e:
                        logger.error(f'Failed to parse template result output: {result.output}')
                        from ...exceptions import TemplateError
                        raise TemplateError(f'Failed to parse system template result: {e}') from e
                elif hasattr(result, 'success') and result.success:
                    # TemplatesOutput has 'data' field with rendered content
                    if result.data and isinstance(result.data, dict):
                        rendered['system_prompt'] = result.data.get('rendered', '')
                        logger.info(f'Rendered system template for {ctx.state.current_phase}')
                        logger.debug(f"[TemplateRenderNode] System prompt content (first 100 chars): {rendered['system_prompt'][:100] if rendered['system_prompt'] else 'EMPTY'}")
                    else:
                        logger.error(f'No rendered content in system template result')
                        from ...exceptions import TemplateError
                        raise TemplateError('System template render produced no content')
                else:
                    logger.error(f'Failed to render system template: {result.message if hasattr(result, "message") else "unknown error"}')
                    from ...exceptions import TemplateError
                    raise TemplateError(f'System template render failed: {result.message if hasattr(result, "message") else "unknown error"}')
            except Exception as e:
                logger.error(f'Error rendering system template: {e}')
                from ...exceptions import TemplateError
                raise TemplateError(f'Error rendering system template: {e}') from e
        
        # Render user template
        if phase_def.templates.user_template:
            try:
                # Template name should be the path without extension (e.g., "smoke/prompts/analyze_ingredients")
                template_name = phase_def.templates.user_template.replace('.jinja', '').replace('.j2', '')
                result = await injector.run('templates', {
                    'operation': 'render',
                    'template_name': template_name,
                    'variables': variables
                })
                
                # The injector might return AgentRunResult or TemplatesOutput
                logger.debug(f"[TemplateRenderNode] User template result type: {type(result).__name__}")
                
                # Handle both AgentRunResult and TemplatesOutput
                if hasattr(result, 'output'):
                    # It's an AgentRunResult - parse the output
                    import json
                    try:
                        output_data = json.loads(result.output)
                        if output_data.get('success'):
                            rendered['user_prompt'] = output_data.get('data', {}).get('rendered', '')
                            logger.info(f'Rendered user template for {ctx.state.current_phase}')
                        else:
                            logger.error(f'User template render failed: {output_data.get("message", "unknown error")}')
                            from ...exceptions import TemplateError
                            raise TemplateError(f'User template render failed: {output_data.get("message", "unknown error")}')
                    except json.JSONDecodeError as e:
                        logger.error(f'Failed to parse user template result output: {result.output}')
                        from ...exceptions import TemplateError
                        raise TemplateError(f'Failed to parse user template result: {e}') from e
                elif hasattr(result, 'success') and result.success:
                    # TemplatesOutput has 'data' field with rendered content
                    if result.data and isinstance(result.data, dict):
                        rendered['user_prompt'] = result.data.get('rendered', '')
                        logger.info(f'Rendered user template for {ctx.state.current_phase}')
                        logger.debug(f"[TemplateRenderNode] User prompt content (first 100 chars): {rendered['user_prompt'][:100] if rendered['user_prompt'] else 'EMPTY'}")
                    else:
                        logger.error(f'No rendered content in user template result')
                        from ...exceptions import TemplateError
                        raise TemplateError('User template render produced no content')
                else:
                    logger.error(f'Failed to render user template: {result.message if hasattr(result, "message") else "unknown error"}')
                    from ...exceptions import TemplateError
                    raise TemplateError(f'User template render failed: {result.message if hasattr(result, "message") else "unknown error"}')
            except Exception as e:
                logger.error(f'Error rendering user template: {e}')
                from ...exceptions import TemplateError
                raise TemplateError(f'Error rendering user template: {e}') from e
        
        if not rendered:
            raise NonRetryableError(f'No templates rendered for {ctx.state.current_phase}')
        
        return rendered
    
    async def _prepare_variables(self, ctx: GraphRunContext[WorkflowState, Any]) -> Dict[str, Any]:
        """Prepare variables for template rendering, ensuring all are JSON-serializable."""
        import json
        
        logger.debug(f"[TemplateRenderNode] _prepare_variables ENTRY - Phase: {ctx.state.current_phase}")
        logger.debug(f"[TemplateRenderNode] State domain_data keys: {list(ctx.state.domain_data.keys())}")
        
        def make_serializable(obj, depth=0):
            """Convert any object to a JSON-serializable form - simplified for Pydantic."""
            # Prevent infinite recursion
            if depth > 5:
                logger.warning(f"[TemplateRenderNode] Max depth reached for {type(obj).__name__}")
                return f"<max_depth: {type(obj).__name__}>"
            
            # Handle basic types first
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            
            # Handle Pydantic models - this should be the main path
            if hasattr(obj, 'model_dump'):
                try:
                    logger.debug(f"[TemplateRenderNode] Serializing Pydantic model: {type(obj).__name__}")
                    return obj.model_dump()
                except Exception as e:
                    logger.error(f"[TemplateRenderNode] Failed to dump Pydantic model {type(obj).__name__}: {e}")
                    return str(obj)
            
            # Handle collections
            if isinstance(obj, (list, tuple)):
                return [make_serializable(item, depth + 1) for item in obj]
            elif isinstance(obj, set):
                return [make_serializable(item, depth + 1) for item in obj]  
            elif isinstance(obj, dict):
                return {str(k): make_serializable(v, depth + 1) for k, v in obj.items()}
            
            # Handle dataclasses
            elif hasattr(obj, '__dataclass_fields__'):
                try:
                    from dataclasses import asdict
                    logger.debug(f"[TemplateRenderNode] Serializing dataclass: {type(obj).__name__}")
                    return asdict(obj)
                except Exception as e:
                    logger.error(f"[TemplateRenderNode] Failed to serialize dataclass {type(obj).__name__}: {e}")
                    return str(obj)
            
            # Fallback to string
            else:
                logger.debug(f"[TemplateRenderNode] Converting to string: {type(obj).__name__}")
                return str(obj)
        
        state = ctx.state
        variables = {
            'workflow_id': state.workflow_id,
            'domain': state.domain,
            'phase': state.current_phase,
        }
        
        # Dependencies are already stored as {phase_name}_output in domain_data
        # So they'll be picked up in the loop below
        
        # Add any domain-specific data with detailed logging
        logger.debug(f"[TemplateRenderNode] Processing domain_data items:")
        for key, value in state.domain_data.items():
            # Skip internal keys but keep input
            if key not in ['rendered_prompts', 'error', 'error_node', 'error_time']:
                logger.debug(f"[TemplateRenderNode] Processing domain_data[{key}]: {type(value).__name__}")
                try:
                    serialized = make_serializable(value)
                    variables[key] = serialized
                    logger.debug(f"[TemplateRenderNode] Successfully serialized domain_data[{key}]")
                    # Extra debug for key outputs
                    if key in ['recipe_designer_output', 'recipe_crafter_output']:
                        logger.info(f"[TemplateRenderNode] {key} type after serialization: {type(serialized).__name__}")
                        if isinstance(serialized, dict):
                            logger.info(f"[TemplateRenderNode] {key} keys: {list(serialized.keys())[:5]}")
                except Exception as e:
                    logger.error(f"[TemplateRenderNode] Failed to serialize domain_data[{key}]: {e}")
                    variables[key] = str(value)
        
        # Add validation results if present (for refinement)
        if state.validation_results:
            logger.debug(f"[TemplateRenderNode] Processing validation results: {list(state.validation_results.keys())}")
            for phase_name, validation_result in state.validation_results.items():
                logger.debug(f"[TemplateRenderNode] Serializing validation_result for {phase_name}: {type(validation_result).__name__}")
                try:
                    variables[f'validation_{phase_name}'] = make_serializable(validation_result)
                except Exception as e:
                    logger.error(f"[TemplateRenderNode] Failed to serialize validation for {phase_name}: {e}")
                    variables[f'validation_{phase_name}'] = str(validation_result)
        
        # Add iteration-specific variables if in iteration
        phase_name = ctx.state.current_phase
        iter_key = f"{phase_name}_iteration"
        if f"{iter_key}_current" in ctx.state.domain_data:
            logger.debug(f"[TemplateRenderNode] Found iteration context for {phase_name}")
            current_item = ctx.state.domain_data[f"{iter_key}_current"]
            
            # For specifier phase, add specific variables
            if phase_name == 'specifier':
                variables['agentool_to_implement'] = make_serializable(current_item)
                variables['analysis_output'] = make_serializable(ctx.state.domain_data.get('analyzer_output', {}))
                
                # Load existing tool schemas for required tools
                if isinstance(current_item, dict) and 'required_tools' in current_item:
                    required_tools = current_item.get('required_tools', [])
                    existing_schemas = self._load_tool_schemas(ctx, required_tools)
                    variables['existing_tools_schemas'] = make_serializable(existing_schemas)
                else:
                    variables['existing_tools_schemas'] = []
                
                logger.debug(f"[TemplateRenderNode] Added specifier iteration variables for tool: {current_item.get('name', 'unknown')}")
            
            # For crafter phase, add specific variables
            elif phase_name == 'crafter':
                variables['agentool_to_implement'] = make_serializable(current_item)
                variables['analysis_output'] = make_serializable(ctx.state.domain_data.get('analyzer_output', {}))
                
                # Get the specification for this tool
                tool_name = current_item.name if hasattr(current_item, 'name') else current_item.get('name', 'unknown')
                spec_key = f"workflow/{ctx.state.workflow_id}/specification/{tool_name}"
                
                # Load the specification from storage
                storage_client = ctx.deps.get_storage_client()
                spec_result = await storage_client.run('storage_kv', {
                    'operation': 'get',
                    'key': spec_key,
                    'namespace': 'workflow'
                })
                
                if spec_result.success:
                    spec_output = spec_result.data.get('value', {})
                    variables['spec_output'] = make_serializable(spec_output)
                    logger.debug(f"[TemplateRenderNode] Loaded specification for {tool_name}")
                else:
                    logger.warning(f"[TemplateRenderNode] Could not load specification for {tool_name}")
                    variables['spec_output'] = {}
                
                # Get all specifications for context
                specs_key = f"workflow/{ctx.state.workflow_id}/specifications"
                specs_result = await storage_client.run('storage_kv', {
                    'operation': 'get',
                    'key': specs_key,
                    'namespace': 'workflow'
                })
                
                if specs_result.success:
                    all_specs = specs_result.data.get('value', {})
                    variables['all_specifications'] = make_serializable(all_specs)
                else:
                    variables['all_specifications'] = {}
                
                # Load existing tool schemas
                if isinstance(current_item, dict) and 'required_tools' in current_item:
                    required_tools = current_item.get('required_tools', [])
                elif hasattr(current_item, 'required_tools'):
                    required_tools = current_item.required_tools
                else:
                    required_tools = []
                    
                existing_schemas = self._load_tool_schemas(ctx, required_tools)
                variables['existing_tools_schemas'] = make_serializable(existing_schemas)
                
                # Store tool name to render skeleton later
                variables['_tool_name_for_skeleton'] = tool_name
                # Placeholder for skeleton - will be rendered in perform_operation
                variables['skeleton'] = ''
                
                logger.debug(f"[TemplateRenderNode] Added crafter iteration variables for tool: {tool_name}")
            
            # For evaluator phase, add specific variables
            elif phase_name == 'evaluator':
                variables['agentool_to_implement'] = make_serializable(current_item)
                variables['analysis_output'] = make_serializable(ctx.state.domain_data.get('analyzer_output', {}))
                
                # Get the tool name
                tool_name = current_item.name if hasattr(current_item, 'name') else current_item.get('name', 'unknown')
                
                # Get the specification for this tool
                spec_key = f"workflow/{ctx.state.workflow_id}/specification/{tool_name}"
                storage_client = ctx.deps.get_storage_client()
                spec_result = await storage_client.run('storage_kv', {
                    'operation': 'get',
                    'key': spec_key,
                    'namespace': 'workflow'
                })
                
                if spec_result.success:
                    spec_output = spec_result.data.get('value', {})
                    variables['spec_output'] = make_serializable(spec_output)
                    logger.debug(f"[TemplateRenderNode] Loaded specification for {tool_name}")
                else:
                    logger.warning(f"[TemplateRenderNode] Could not load specification for {tool_name}")
                    variables['spec_output'] = {}
                
                # Get the implementation code for this tool
                impl_key = f"workflow/{ctx.state.workflow_id}/crafter/{tool_name}"
                impl_result = await storage_client.run('storage_kv', {
                    'operation': 'get',
                    'key': impl_key,
                    'namespace': 'workflow'
                })
                
                if impl_result.success:
                    impl_output = impl_result.data.get('value', {})
                    # Extract the code from CodeOutput if it's stored as such
                    if isinstance(impl_output, dict) and 'code' in impl_output:
                        implementation_code = impl_output['code']
                    elif isinstance(impl_output, str):
                        implementation_code = impl_output
                    else:
                        implementation_code = str(impl_output)
                    
                    variables['implementation_code'] = implementation_code
                    logger.debug(f"[TemplateRenderNode] Loaded implementation code for {tool_name} ({len(implementation_code)} chars)")
                else:
                    logger.warning(f"[TemplateRenderNode] Could not load implementation for {tool_name}")
                    variables['implementation_code'] = ""
                
                # Get the skeleton template (render it for this tool)
                try:
                    from agentool.core.injector import get_injector
                    injector = get_injector()
                    skeleton_result = await injector.run('templates', {
                        'operation': 'render',
                        'template_name': 'agentool/skeletons/agentool_comprehensive',
                        'variables': {'tool_name': tool_name}
                    })
                    
                    if hasattr(skeleton_result, 'output'):
                        # AgentRunResult - parse the output
                        import json
                        try:
                            output_data = json.loads(skeleton_result.output)
                            if output_data.get('success'):
                                variables['skeleton'] = output_data.get('data', {}).get('rendered', '')
                            else:
                                variables['skeleton'] = ''
                        except json.JSONDecodeError:
                            variables['skeleton'] = ''
                    elif hasattr(skeleton_result, 'success') and skeleton_result.success:
                        if skeleton_result.data and isinstance(skeleton_result.data, dict):
                            variables['skeleton'] = skeleton_result.data.get('rendered', '')
                        else:
                            variables['skeleton'] = ''
                    else:
                        variables['skeleton'] = ''
                        
                    logger.debug(f"[TemplateRenderNode] Rendered skeleton for {tool_name}")
                except Exception as e:
                    logger.warning(f"[TemplateRenderNode] Could not render skeleton for {tool_name}: {e}")
                    variables['skeleton'] = ''
                
                logger.debug(f"[TemplateRenderNode] Added evaluator iteration variables for tool: {tool_name}")
            
            else:
                # Generic iteration variables
                variables['current_item'] = make_serializable(current_item)
                variables['iteration_index'] = ctx.state.domain_data.get(f"{iter_key}_index", 0)
        
        # Add phase-specific variables from state
        phase_def = ctx.state.get_current_phase_def()
        if phase_def and phase_def.templates and phase_def.templates.variables:
            logger.debug(f"[TemplateRenderNode] Processing phase-specific template variables")
            for k, v in phase_def.templates.variables.items():
                logger.debug(f"[TemplateRenderNode] Serializing template variable {k}: {type(v).__name__}")
                variables[k] = make_serializable(v)
        
        # Add output schema for phases that need it (like analyzer)
        if phase_def and phase_def.output_schema:
            try:
                schema_json = json.dumps(phase_def.output_schema.model_json_schema(), indent=2)
                variables['schema_json'] = schema_json
                logger.debug(f"[TemplateRenderNode] Added schema_json for {phase_def.output_schema.__name__}")
            except Exception as e:
                logger.error(f"[TemplateRenderNode] Failed to generate schema_json: {e}")
        
        # Verify all variables are JSON-serializable with detailed debugging
        logger.debug(f"[TemplateRenderNode] Final variable count: {len(variables)}")
        logger.debug(f"[TemplateRenderNode] Final variable keys: {list(variables.keys())}")
        
        try:
            json.dumps(variables)
            logger.debug(f"[TemplateRenderNode] All variables successfully JSON serializable")
        except TypeError as e:
            logger.error(f"[TemplateRenderNode] JSON serialization failed: {e}")
            # Try to identify the problematic variable
            for k, v in variables.items():
                try:
                    json.dumps({k: v})
                    logger.debug(f"[TemplateRenderNode] Variable '{k}' is serializable")
                except Exception as var_error:
                    logger.error(f"[TemplateRenderNode] Variable '{k}' is not serializable: {var_error}")
                    logger.debug(f"[TemplateRenderNode] Converting '{k}' to string: {type(v).__name__}")
                    variables[k] = str(v)
        
        logger.debug(f"[TemplateRenderNode] _prepare_variables EXIT - returning {len(variables)} variables")
        return variables
    
    def _load_tool_schemas(self, ctx: GraphRunContext[WorkflowState, Any], tool_names: List[str]) -> List[Dict[str, Any]]:
        """Load schemas for existing tools from the catalog."""
        schemas = []
        
        # Get catalog from domain_data (should have been loaded in analyzer phase)
        catalog = ctx.state.domain_data.get('catalog', {})
        if not catalog:
            catalog = ctx.state.domain_data.get('catalog_output', {})
        
        if isinstance(catalog, dict) and 'agentools' in catalog:
            agentools = catalog['agentools']
            for tool_name in tool_names:
                # Find the tool in the catalog
                for tool in agentools:
                    if tool.get('name') == tool_name:
                        schemas.append(tool)
                        logger.debug(f"[TemplateRenderNode] Found schema for tool: {tool_name}")
                        break
                else:
                    logger.warning(f"[TemplateRenderNode] Tool {tool_name} not found in catalog")
        else:
            logger.warning(f"[TemplateRenderNode] No catalog available for loading tool schemas")
        
        return schemas
    
    async def update_state_in_place(self, state: WorkflowState, result: Dict[str, str]) -> None:
        """Update state with rendered templates - modifies in place."""
        logger.info(f"Storing rendered prompts for {state.current_phase}: {list(result.keys())}")
        state.domain_data['rendered_prompts'] = result
        logger.info(f"State now has domain_data keys: {list(state.domain_data.keys())}")
        
        # Also store in KV using the render/* pattern if configured
        phase_def = state.get_current_phase_def()
        if phase_def and phase_def.additional_storage_patterns and 'rendered' in phase_def.additional_storage_patterns:
            render_key = phase_def.additional_storage_patterns['rendered'].format(
                workflow_id=state.workflow_id
            )
            logger.debug(f"[TemplateRenderNode] Storing rendered template at {render_key}")
            
            # Combine system and user prompts for storage
            combined_render = {
                'system_prompt': result.get('system_prompt', ''),
                'user_prompt': result.get('user_prompt', ''),
                'phase': state.current_phase,
                'timestamp': str(state.updated_at)
            }
            
            try:
                from ...core.initialization import ensure_graphtoolkit_initialized
                from agentool.core.injector import get_injector
                ensure_graphtoolkit_initialized()
                injector = get_injector()
                
                storage_result = await injector.run('storage_kv', {
                    'operation': 'set',
                    'key': render_key,
                    'value': combined_render,
                    'namespace': 'workflow'
                })
                
                if storage_result.success:
                    logger.info(f"[TemplateRenderNode] Stored rendered template at {render_key}")
                else:
                    logger.warning(f"[TemplateRenderNode] Failed to store rendered template: {storage_result.message}")
            except Exception as e:
                logger.warning(f"[TemplateRenderNode] Could not store rendered template: {e}")


# Register template nodes
register_node_class('template_render', TemplateRenderNode)
