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
        from agentool.core.injector import get_injector
        injector = get_injector()
        
        rendered = {}
        
        # Render system template
        if phase_def.templates.system_template:
            try:
                result = await injector.run('templates', {
                    'operation': 'render',
                    'template_name': phase_def.templates.system_template.replace('templates/', '').replace('.jinja', ''),
                    'variables': variables
                })
                
                if result.success and result.data:
                    rendered['system_prompt'] = result.data.get('rendered', '')
                    logger.info(f'Rendered system template for {ctx.state.current_phase}')
                else:
                    logger.warning(f'Failed to render system template: {result.message}')
                    
            except Exception as e:
                logger.error(f'Error rendering system template: {e}')
        
        # Render user template
        if phase_def.templates.user_template:
            try:
                result = await injector.run('templates', {
                    'operation': 'render',
                    'template_name': phase_def.templates.user_template.replace('templates/', '').replace('.jinja', ''),
                    'variables': variables
                })
                
                if result.success and result.data:
                    rendered['user_prompt'] = result.data.get('rendered', '')
                    logger.info(f'Rendered user template for {ctx.state.current_phase}')
                else:
                    logger.warning(f'Failed to render user template: {result.message}')
                    
            except Exception as e:
                logger.error(f'Error rendering user template: {e}')
        
        if not rendered:
            raise NonRetryableError(f'No templates rendered for {ctx.state.current_phase}')
        
        return rendered
    
    def _prepare_variables(self, ctx: GraphRunContext[WorkflowState, Any]) -> Dict[str, Any]:
        """Prepare variables for template rendering."""
        state = ctx.state
        variables = {
            'workflow_id': state.workflow_id,
            'domain': state.domain,
            'phase': state.current_phase,
        }
        
        # Add loaded dependencies
        if 'loaded_dependencies' in state.domain_data:
            dependencies = state.domain_data['loaded_dependencies']
            for dep_name, dep_data in dependencies.items():
                variables[f'dep_{dep_name}'] = dep_data
        
        # Add any domain-specific data
        for key, value in state.domain_data.items():
            if key not in ['loaded_dependencies', 'rendered_prompts']:
                variables[key] = value
        
        # Add phase-specific variables from state
        phase_def = ctx.state.get_current_phase_def()
        if phase_def and phase_def.templates and phase_def.templates.variables:
            variables.update(phase_def.templates.variables)
        
        return variables
    
    async def update_state(self, state: WorkflowState, result: Dict[str, str]) -> WorkflowState:
        """Update state with rendered templates."""
        return replace(
            state,
            domain_data={
                **state.domain_data,
                'rendered_prompts': result
            }
        )


@dataclass
class TemplateValidateNode(AtomicNode[WorkflowState, Any, bool]):
    """Validate template syntax before rendering.
    """
    template_content: str
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> bool:
        """Validate template syntax."""
        from agentool.core.injector import get_injector
        injector = get_injector()
        
        try:
            result = await injector.run('templates', {
                'operation': 'validate',
                'template_content': self.template_content
            })
            
            if result.success and result.data:
                return result.data.get('valid', False)
            
            return False
            
        except Exception as e:
            logger.error(f'Template validation error: {e}')
            return False


@dataclass
class TemplateSaveNode(AtomicNode[WorkflowState, Any, bool]):
    """Save a template to the template directory.
    """
    template_name: str
    template_content: str
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> bool:
        """Save template to storage."""
        from agentool.core.injector import get_injector
        injector = get_injector()
        
        try:
            result = await injector.run('templates', {
                'operation': 'save',
                'template_name': self.template_name,
                'template_content': self.template_content
            })
            
            return result.success
            
        except Exception as e:
            logger.error(f'Template save error: {e}')
            return False


@dataclass
class TemplateExecNode(AtomicNode[WorkflowState, Any, str]):
    """Execute ad-hoc template rendering.
    """
    template_content: str
    variables: Optional[Dict[str, Any]] = None
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> str:
        """Execute template rendering."""
        from agentool.core.injector import get_injector
        injector = get_injector()
        
        # Merge provided variables with state variables
        all_variables = self._prepare_variables(ctx.state)
        if self.variables:
            all_variables.update(self.variables)
        
        try:
            result = await injector.run('templates', {
                'operation': 'exec',
                'template_content': self.template_content,
                'variables': all_variables,
                'strict': False
            })
            
            if result.success and result.data:
                return result.data.get('rendered', '')
            
            raise NonRetryableError(f'Template execution failed: {result.message}')
            
        except Exception as e:
            raise NonRetryableError(f'Template execution error: {e}')
    
    def _prepare_variables(self, state: WorkflowState) -> Dict[str, Any]:
        """Prepare variables from state."""
        return {
            'workflow_id': state.workflow_id,
            'domain': state.domain,
            'phase': state.current_phase,
            **state.domain_data
        }


# Register template nodes
register_node_class('template_render', TemplateRenderNode)
register_node_class('template_validate', TemplateValidateNode)
register_node_class('template_save', TemplateSaveNode)
register_node_class('template_exec', TemplateExecNode)