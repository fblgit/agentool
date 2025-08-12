"""GraphToolkit Template Atomic Nodes.

Template rendering nodes that integrate with the existing template system.
"""

import logging
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

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
        variables = self._prepare_variables(ctx)
        
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
    
    def _prepare_variables(self, ctx: GraphRunContext[WorkflowState, Any]) -> Dict[str, Any]:
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
    
    async def update_state_in_place(self, state: WorkflowState, result: Dict[str, str]) -> None:
        """Update state with rendered templates - modifies in place."""
        logger.info(f"Storing rendered prompts for {state.current_phase}: {list(result.keys())}")
        state.domain_data['rendered_prompts'] = result
        logger.info(f"State now has domain_data keys: {list(state.domain_data.keys())}")


# Register template nodes
register_node_class('template_render', TemplateRenderNode)