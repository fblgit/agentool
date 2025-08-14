"""GraphToolkit LLM Atomic Nodes.

LLM interaction nodes with state-driven retry configuration.
"""

import json
import logging
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional

from ...core.factory import register_node_class
from ...core.types import ModelParameters, TokenUsage, WorkflowState
from ..base import AtomicNode, GraphRunContext, LLMError, NonRetryableError

logger = logging.getLogger(__name__)


@dataclass
class LLMCallNode(AtomicNode[WorkflowState, Any, Any]):
    """Execute LLM call with rendered prompts.
    Expensive operation - retryable via configuration.
    """
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> Any:
        """Execute LLM call with prompts from state."""
        logger.debug(f"[LLMCallNode] === ENTRY === Phase: {ctx.state.current_phase}")
        logger.debug(f"[LLMCallNode] Workflow: {ctx.state.workflow_id}")
        
        phase_def = ctx.state.workflow_def.phases.get(ctx.state.current_phase)
        if not phase_def:
            logger.error(f"[LLMCallNode] Phase {ctx.state.current_phase} not found")
            raise NonRetryableError(f'Phase {ctx.state.current_phase} not found')
        
        # Get rendered prompts from state
        logger.debug(f"[LLMCallNode] Looking for rendered_prompts in domain_data. Keys: {list(ctx.state.domain_data.keys())}")
        prompts = ctx.state.domain_data.get('rendered_prompts', {})
        if not prompts:
            logger.error(f"[LLMCallNode] No rendered_prompts found. domain_data keys: {list(ctx.state.domain_data.keys())}")
            raise NonRetryableError('No rendered prompts available')
        
        system_prompt = prompts.get('system_prompt', '')
        user_prompt = prompts.get('user_prompt', '')
        
        if not user_prompt:
            raise NonRetryableError('User prompt is required')
        
        # Get model configuration
        model_params = phase_def.model_config or ModelParameters()
        
        # Determine model to use
        model = self._get_model_for_phase(ctx)
        logger.debug(f"[LLMCallNode] Using model: {model}")
        logger.debug(f"[LLMCallNode] Output schema: {phase_def.output_schema.__name__ if phase_def.output_schema else 'None'}")
        
        try:
            # Make LLM call
            logger.debug(f"[LLMCallNode] Making LLM call with prompts")
            logger.debug(f"[LLMCallNode] System prompt: {system_prompt[:100]}..." if system_prompt else "[LLMCallNode] No system prompt")
            logger.debug(f"[LLMCallNode] User prompt: {user_prompt[:100]}..." if user_prompt else "[LLMCallNode] No user prompt")
            
            response = await self._call_llm(
                ctx,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                params=model_params,
                output_schema=phase_def.output_schema
            )
            
            logger.info(f'[LLMCallNode] LLM call successful for {ctx.state.current_phase}')
            logger.debug(f"[LLMCallNode] Response type: {type(response).__name__}")
            logger.debug(f"[LLMCallNode] === EXIT === Success")
            return response
            
        except Exception as e:
            # LLM errors are now retryable via configuration
            logger.error(f"[LLMCallNode] LLM call failed: {e}")
            logger.debug(f"[LLMCallNode] === EXIT === Raising LLMError")
            raise LLMError(f'LLM call failed: {e}')
    
    def _get_model_for_phase(self, ctx: GraphRunContext[WorkflowState, Any]) -> str:
        """Determine which model to use for this phase."""
        # Check if deps has model configuration
        if hasattr(ctx.deps, 'models') and ctx.deps.models:
            # Use the configured model
            if ctx.deps.models.provider and ctx.deps.models.model:
                return f"{ctx.deps.models.provider}:{ctx.deps.models.model}"
            
        # Default model
        return 'openai:gpt-4o'
    
    async def _call_llm(
        self,
        ctx: GraphRunContext[WorkflowState, Any],
        model: str,
        system_prompt: str,
        user_prompt: str,
        params: ModelParameters,
        output_schema: Optional[type] = None
    ) -> Any:
        """Make the actual LLM call."""
        # In production, this would use the actual LLM client
        # For now, we'll use a mock or the injector pattern
        
        if hasattr(ctx.deps, 'llm_client'):
            # Use LLM client from deps
            response = await ctx.deps.llm_client.complete(
                model=model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                temperature=params.temperature,
                max_tokens=params.max_tokens,
                top_p=params.top_p,
                frequency_penalty=params.frequency_penalty,
                presence_penalty=params.presence_penalty,
                stop=params.stop_sequences,
                response_format={'type': 'json'} if output_schema else None
            )
            
            # Parse response if schema provided
            if output_schema:
                try:
                    data = json.loads(response.content)
                    return output_schema(**data)
                except (json.JSONDecodeError, ValueError) as e:
                    raise NonRetryableError(f'Failed to parse LLM response: {e}')
            
            return response.content
        
        # No LLM client available, try to use pydantic-ai
        try:
            from pydantic_ai import Agent
            from pydantic_ai.models import ALLOW_MODEL_REQUESTS
            
            logger.info(f"Creating pydantic-ai agent with model: {model}")
            logger.info(f"Output schema: {output_schema}")
            
            # Handle test model specially
            if model == 'test:test':
                from pydantic_ai.models.test import TestModel
                model_instance = TestModel()
            else:
                # For real models, we need to allow model requests
                # Set the flag to allow real API calls
                import pydantic_ai.models
                pydantic_ai.models.ALLOW_MODEL_REQUESTS = True
                model_instance = model
            
            # Special handling for crafter phase - always use str output
            if ctx.state.current_phase == 'crafter':
                logger.info("[LLMCallNode] Crafter phase detected - using raw string output")
                agent = Agent(
                    model_instance,
                    system_prompt=system_prompt,
                    output_type=str  # Always use str for crafter
                )
            else:
                # Create agent with appropriate model
                agent = Agent(
                    model_instance,
                    system_prompt=system_prompt,
                    output_type=output_schema if output_schema else str
                )
            
            logger.info(f"Running agent with prompt: {user_prompt[:100]}...")
            # Run the agent
            result = await agent.run(user_prompt)
            
            logger.info(f"Agent result type: {type(result)}, has output: {hasattr(result, 'output')}")
            
            # Get the output
            raw_output = result.output if hasattr(result, 'output') else result
            
            # For crafter phase, extract code from markdown blocks and create CodeOutput
            if ctx.state.current_phase == 'crafter':
                # Extract code from markdown code block
                import re
                code_match = re.search(r'```python\n(.*?)```', str(raw_output), re.DOTALL)
                if code_match:
                    generated_code = code_match.group(1).strip()
                    logger.info(f"[LLMCallNode] Extracted {len(generated_code)} chars of code from markdown block")
                else:
                    # Fallback if no code block found
                    generated_code = str(raw_output).strip()
                    logger.warning("[LLMCallNode] No markdown code block found, using raw output")
                
                # Get tool name for file path
                iter_key = f"{ctx.state.current_phase}_iteration"
                current_item = ctx.state.domain_data.get(f"{iter_key}_current")
                if current_item:
                    if hasattr(current_item, 'name'):
                        tool_name = current_item.name
                    elif isinstance(current_item, dict) and 'name' in current_item:
                        tool_name = current_item['name']
                    else:
                        tool_name = 'generated_tool'
                else:
                    tool_name = 'generated_tool'
                
                # Import CodeOutput if not already imported
                from agents.models import CodeOutput
                
                # Create CodeOutput object
                code_output = CodeOutput(
                    code=generated_code,
                    file_path=f"{tool_name}.py"
                )
                
                logger.info(f"[LLMCallNode] Created CodeOutput for {tool_name} with {len(generated_code)} chars")
                return code_output
            
            # Return the output for other phases
            return raw_output
            
        except ImportError as e:
            logger.error(f"Import error: {e}")
            raise NonRetryableError('No LLM client available. Install pydantic-ai or configure LLM client in deps.')
        except Exception as e:
            logger.error(f"LLM call error: {e}", exc_info=True)
            raise LLMError(f'LLM call failed: {e}')
    
    async def update_state_in_place(self, state: WorkflowState, result: Any) -> None:
        """Update state with LLM response - modifies in place."""
        phase_name = state.current_phase
        
        # Store raw response
        state.domain_data[f'{phase_name}_llm_response'] = result
        state.domain_data[f'{phase_name}_output'] = result  # Also store as output for SavePhaseOutputNode
        
        # Update token usage if available
        # In production, this would come from the LLM response
        # For now, we'll create a mock token usage
        token_usage = TokenUsage(
            prompt_tokens=len(state.domain_data.get('rendered_prompts', {}).get('user_prompt', '')) // 4,
            completion_tokens=len(str(result)) // 4,
            total_tokens=0,
            model='test'  # Simple model string for testing
        )
        token_usage = replace(
            token_usage,
            total_tokens=token_usage.prompt_tokens + token_usage.completion_tokens
        )
        
        state.total_token_usage[phase_name] = token_usage


# Register LLM nodes
register_node_class('llm_call', LLMCallNode)