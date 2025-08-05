"""
AgenTool Model Provider for pydantic-ai.

This module provides the AgenToolModel class, which is a synthetic LLM provider
that enables programmatic agent behavior by routing JSON input to tools.

The AgenToolModel acts as a bridge between pydantic-ai's agent interface and
deterministic tool execution, enabling:

- Schema-driven input validation
- Deterministic routing based on operation fields
- Structured JSON input/output
- Integration with pydantic-ai's model registry
- Cost-effective alternative to LLM calls for structured operations

Example:
    >>> from pydantic_ai import Agent
    >>> from agentool import AgenToolModel
    >>> 
    >>> model = AgenToolModel('storage')
    >>> agent = Agent(model=model)
    >>> result = await agent.run('{"operation": "read", "key": "test"}')
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from pydantic import BaseModel
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
    ToolReturnPart,
    RetryPromptPart,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import Usage


@dataclass(init=False)
class AgenToolModel(Model):
    """
    A synthetic model provider that converts JSON input to tool calls.
    
    This model enables agents to behave programmatically by:
    1. Parsing JSON input from user messages
    2. Generating a tool call to the special '__agentool_manager__' tool
    3. Returning the tool execution result as the final response
    
    The actual routing logic is handled by the AgenToolManager,
    keeping this model simple and focused.
    """
    
    name: str
    _model_name: str = field(init=False)
    _system: str = field(default='agentool', init=False)
    
    def __init__(self, name: str, **kwargs):
        """Initialize the AgenToolModel.
        
        Args:
            name: The name of the AgenTool (e.g., 'storage', 'compute')
        """
        self.name = name
        self._model_name = f'agentool:{name}'
        super().__init__(**kwargs)
    
    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Process a request and generate appropriate tool calls.
        
        This implements a two-phase conversation flow:
        
        Phase 1 (Initial request):
        - Extracts JSON input from the user message
        - Generates a tool call to '__agentool_manager__' with the parsed data
        
        Phase 2 (Tool response):
        - Receives the tool execution result from the manager
        - Converts non-string results to JSON format
        - Returns the result as a text response
        
        Args:
            messages: The conversation history containing user and model messages
            model_settings: Optional model-specific settings (ignored)
            model_request_parameters: Request parameters including tool definitions
        
        Returns:
            ModelResponse containing either a tool call or the final text result
        
        Note:
            The actual validation and routing logic is handled by the
            AgenToolManager, keeping this model focused on message handling.
        """
        timestamp = datetime.now(tz=timezone.utc)
        
        # Check if configuration exists for this AgenTool
        from .registry import AgenToolRegistry
        config = AgenToolRegistry.get(self.name)
        if config is None:
            return ModelResponse(
                parts=[TextPart(f"No configuration found for AgenTool '{self.name}'")],
                model_name=self._model_name,
                usage=Usage(request_tokens=10, response_tokens=20, total_tokens=30, requests=1),
                timestamp=timestamp
            )
        
        # Check if we've already made a tool call
        has_model_response = any(isinstance(m, ModelResponse) for m in messages)
        
        if has_model_response:
            # Look for tool returns in the last message
            tool_returns = []
            if messages:
                last_message = messages[-1]
                if isinstance(last_message, ModelRequest):
                    for part in last_message.parts:
                        if isinstance(part, ToolReturnPart):
                            tool_returns.append(part)
            
            # Return tool results as final response
            if tool_returns:
                if len(tool_returns) == 1:
                    content = tool_returns[0].content
                    
                    # Check if we have an output_type configured
                    config = AgenToolRegistry.get(self.name)
                    if config and config.output_type is not None:
                        if isinstance(content, BaseModel):
                            # We have a typed output - serialize it to JSON
                            content = content.model_dump_json()
                        elif isinstance(content, str):
                            # Already a string, keep as is
                            pass
                        else:
                            # Other types, serialize to JSON
                            content = json.dumps(content)
                    elif not isinstance(content, str):
                        # Ensure content is a string
                        content = json.dumps(content)
                else:
                    content = json.dumps({tr.tool_name: tr.content for tr in tool_returns})
                
                return ModelResponse(
                    parts=[TextPart(content)],
                    model_name=self._model_name,
                    usage=Usage(request_tokens=10, response_tokens=20, total_tokens=30, requests=1),
                    timestamp=timestamp
                )
        
        # Extract user input
        user_input = self._extract_user_input(messages)
        if not user_input:
            return ModelResponse(
                parts=[TextPart("No input provided")],
                model_name=self._model_name,
                usage=Usage(request_tokens=10, response_tokens=10, total_tokens=20, requests=1),
                timestamp=timestamp
            )
        
        # Parse JSON input
        try:
            input_data = json.loads(user_input)
        except json.JSONDecodeError as e:
            return ModelResponse(
                parts=[TextPart(f"Invalid JSON input: {e}")],
                model_name=self._model_name,
                usage=Usage(request_tokens=10, response_tokens=20, total_tokens=30, requests=1),
                timestamp=timestamp
            )
        
        # Generate tool call to the manager
        parts = [ToolCallPart(
            tool_name='__agentool_manager__',
            args=input_data,
            tool_call_id=f"agentool_{int(timestamp.timestamp() * 1000)}"
        )]
        
        return ModelResponse(
            parts=parts,
            model_name=self._model_name,
            usage=Usage(request_tokens=20, response_tokens=10, total_tokens=30, requests=1),
            timestamp=timestamp
        )
    
    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        """Streaming is not supported for AgenTools."""
        raise NotImplementedError("AgenTools do not support streaming")
        yield  # This line will never be reached but satisfies the async generator protocol
    
    @property
    def model_name(self) -> str:
        """The model name."""
        return self._model_name
    
    @property
    def system(self) -> str:
        """The system / model provider."""
        return self._system
    
    def _extract_user_input(self, messages: list[ModelMessage]) -> str | None:
        """Extract the last user message content."""
        for message in reversed(messages):
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, UserPromptPart):
                        if isinstance(part.content, str):
                            return part.content
                        elif isinstance(part.content, list):
                            # Look for text content in the list
                            for item in part.content:
                                if isinstance(item, str):
                                    return item
        return None