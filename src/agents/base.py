"""Base agent class for the AI Code Generation Workflow.

This module provides the base class for all LLM agents in the workflow,
handling template loading and agent initialization.
"""

import json
from typing import Optional, Type, TypeVar, Generic, Any
from pydantic import BaseModel
from pydantic_ai import Agent
from agentool.core.injector import get_injector

T = TypeVar('T', bound=BaseModel)


class BaseAgenToolAgent(Generic[T]):
    """Base class for all AgenTool workflow agents.
    
    This class handles:
    - Loading system prompts from templates
    - Initializing pydantic-ai agents
    - Rendering user prompts with variables
    - Managing agent lifecycle
    """
    
    def __init__(
        self,
        model: str,
        system_template: str,
        output_type: Type[T],
        name: str
    ):
        """Initialize the base agent.
        
        Args:
            model: LLM model identifier (e.g., "openai:gpt-4o")
            system_template: Name of the system prompt template (without path)
            output_type: Pydantic model class for output validation
            name: Name of this agent for logging
        """
        self.model = model
        self.system_template = system_template
        self.output_type = output_type
        self.name = name
        self.agent: Optional[Agent[None, T]] = None
        self.system_prompt: Optional[str] = None
        
    async def initialize(self, **system_variables) -> None:
        """Initialize the agent by loading the system prompt template.
        
        Args:
            **system_variables: Variables to render in the system prompt template
        """
        injector = get_injector()
        
        # Load and render system prompt from template
        try:
            result = await injector.run('templates', {
                'operation': 'render',
                'template_name': f'system/{self.system_template}',
                'variables': system_variables
            })
            
            if hasattr(result, 'output'):
                result_data = json.loads(result.output)
            else:
                result_data = result.data if hasattr(result, 'data') else result
            
            self.system_prompt = result_data['rendered']
            
        except Exception as e:
            # If template doesn't exist or fails, use a default
            print(f"Warning: Failed to load template {self.system_template}: {e}")
            self.system_prompt = f"You are an expert {self.name} agent."
        
        # Create the pydantic-ai agent
        self.agent = Agent(
            self.model,
            output_type=self.output_type,
            system_prompt=self.system_prompt
        )
        
    async def generate(
        self,
        user_prompt_template: str,
        variables: dict,
        message_history: Optional[list] = None
    ) -> T:
        """Generate output using the agent with a rendered prompt.
        
        Args:
            user_prompt_template: Name of the user prompt template
            variables: Variables to render in the user prompt
            message_history: Optional message history for context
            
        Returns:
            Generated output matching the output_type schema
            
        Raises:
            RuntimeError: If agent is not initialized
        """
        if not self.agent:
            raise RuntimeError(f"Agent {self.name} not initialized. Call initialize() first.")
        
        injector = get_injector()
        
        # Render the user prompt from template
        try:
            result = await injector.run('templates', {
                'operation': 'render',
                'template_name': f'prompts/{user_prompt_template}',
                'variables': variables
            })
            
            if hasattr(result, 'output'):
                result_data = json.loads(result.output)
            else:
                result_data = result.data if hasattr(result, 'data') else result
            
            rendered_prompt = result_data['rendered']
            
        except Exception as e:
            # Fallback to direct string if template fails
            print(f"Warning: Failed to render template {user_prompt_template}: {e}")
            rendered_prompt = str(variables.get('task_description', 'Process this task'))
        
        # Generate using pydantic-ai agent
        result = await self.agent.generate(
            rendered_prompt,
            message_history=message_history or []
        )
        
        return result.output
        
    async def generate_with_raw_prompt(
        self,
        prompt: str,
        message_history: Optional[list] = None
    ) -> T:
        """Generate output using a raw prompt string.
        
        Args:
            prompt: Raw prompt string
            message_history: Optional message history
            
        Returns:
            Generated output matching the output_type schema
            
        Raises:
            RuntimeError: If agent is not initialized
        """
        if not self.agent:
            raise RuntimeError(f"Agent {self.name} not initialized. Call initialize() first.")
        
        result = await self.agent.generate(
            prompt,
            message_history=message_history or []
        )
        
        return result.output
        
    async def save_to_state(self, output: T, phase: str, workflow_id: str) -> str:
        """Save agent output to state storage and return reference.
        
        Args:
            output: The output to save
            phase: Phase name (analyzer, specification, crafter, evaluator)
            workflow_id: Unique workflow identifier
            
        Returns:
            Reference key for the saved state
        """
        injector = get_injector()
        
        # Create state key
        state_key = f"workflow/{workflow_id}/{phase}"
        
        # Save to storage_kv
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': state_key,
            'value': output.model_dump_json()
        })
        
        # Log the save
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'workflow',
            'message': f"Saved {phase} output to state",
            'data': {
                'workflow_id': workflow_id,
                'phase': phase,
                'state_key': state_key
            }
        })
        
        return state_key
        
    async def load_from_state(self, state_ref: str) -> dict:
        """Load data from state storage using reference.
        
        Args:
            state_ref: Reference key to the saved state
            
        Returns:
            Loaded data as dictionary
            
        Raises:
            RuntimeError: If state cannot be loaded
        """
        injector = get_injector()
        
        try:
            result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': state_ref
            })
            
            if hasattr(result, 'output'):
                result_data = json.loads(result.output)
            else:
                result_data = result.data if hasattr(result, 'data') else result
            
            if result_data.get('exists'):
                # Parse the stored JSON
                return json.loads(result_data['value'])
            else:
                raise RuntimeError(f"State not found: {state_ref}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load state from {state_ref}: {e}") from e