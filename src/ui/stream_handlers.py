"""
Stream handlers for converting pydantic-ai streams to Streamlit updates.

This module provides utilities for handling different types of streaming
events from pydantic-ai and converting them to appropriate UI updates.
"""

from typing import AsyncIterator, Optional, Callable, Dict, Any
import asyncio
import streamlit as st
from datetime import datetime
import json

from pydantic_ai.messages import (
    ModelResponse,
    ModelRequest,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    SystemPromptPart,
)
from pydantic_ai.result import StreamedRunResult


class StreamHandler:
    """
    Handle streaming updates from pydantic-ai agents.
    
    This class provides methods to process different types of streaming
    events and update the Streamlit UI accordingly.
    """
    
    def __init__(self, container: st.container = None):
        """
        Initialize the stream handler.
        
        Args:
            container: Streamlit container for updates (default: main container)
        """
        self.container = container or st.container()
        self.current_text = ""
        self.tool_calls = []
        self.metrics = {
            'tokens': 0,
            'tool_calls': 0,
            'start_time': None,
            'end_time': None
        }
    
    async def handle_text_stream(
        self,
        stream: AsyncIterator[str],
        placeholder: st.empty,
        delta_mode: bool = True
    ):
        """
        Handle text streaming from an agent.
        
        Args:
            stream: Async iterator of text updates
            placeholder: Streamlit placeholder for updates
            delta_mode: If True, stream contains deltas; if False, full text
        """
        self.metrics['start_time'] = datetime.now()
        
        if delta_mode:
            async for delta in stream:
                self.current_text += delta
                placeholder.markdown(self.current_text)
                self.metrics['tokens'] += len(delta.split())
        else:
            async for full_text in stream:
                self.current_text = full_text
                placeholder.markdown(full_text)
                self.metrics['tokens'] = len(full_text.split())
        
        self.metrics['end_time'] = datetime.now()
    
    async def handle_structured_stream(
        self,
        result: StreamedRunResult,
        update_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Handle structured output streaming.
        
        Args:
            result: StreamedRunResult from pydantic-ai
            update_callback: Optional callback for each update
        """
        try:
            async for message, is_last in result.stream_structured(debounce_by=0.1):
                # Validate the structured output
                try:
                    validated = await result.validate_structured_output(
                        message,
                        allow_partial=not is_last
                    )
                    
                    if update_callback:
                        update_callback(validated)
                    
                    # Update UI with structured data
                    with self.container:
                        st.json(validated)
                        
                except Exception as e:
                    if is_last:
                        st.error(f"Validation error: {e}")
        
        except Exception as e:
            st.error(f"Stream error: {e}")
    
    async def handle_event_stream(
        self,
        stream: AsyncIterator[Any],
        event_container: st.container = None
    ):
        """
        Handle event streaming with different event types.
        
        Args:
            stream: Async iterator of events
            event_container: Container for event display
        """
        container = event_container or self.container
        
        async for event in stream:
            with container:
                if isinstance(event, TextPart):
                    st.markdown(f"💬 **Text**: {event.content}")
                
                elif isinstance(event, ToolCallPart):
                    self.tool_calls.append(event)
                    self.metrics['tool_calls'] += 1
                    
                    st.info(f"🔧 **Tool Call**: `{event.tool_name}`")
                    if event.args:
                        with st.expander("Arguments"):
                            st.json(event.args)
                
                elif isinstance(event, ToolReturnPart):
                    st.success(f"✅ **Tool Result**")
                    if hasattr(event, 'content') and event.content:
                        with st.expander("Result"):
                            st.text(str(event.content))
                
                elif isinstance(event, UserPromptPart):
                    st.markdown(f"👤 **User**: {event.content}")
                
                elif isinstance(event, SystemPromptPart):
                    with st.expander("System Prompt"):
                        st.text(event.content)
                
                else:
                    # Generic event display
                    st.write(f"📋 Event: {type(event).__name__}")
    
    def create_status_updater(self, status_container: st.status) -> Callable[[str], None]:
        """
        Create a callback function for updating a status container.
        
        Args:
            status_container: Streamlit status container
            
        Returns:
            Callback function that updates the status
        """
        def update_status(message: str):
            status_container.update(label=message, state="running")
        
        return update_status
    
    def display_metrics(self):
        """Display collected metrics."""
        if self.metrics['start_time'] and self.metrics['end_time']:
            duration = (self.metrics['end_time'] - self.metrics['start_time']).total_seconds()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Tokens", self.metrics['tokens'])
            
            with col2:
                st.metric("Tool Calls", self.metrics['tool_calls'])
            
            with col3:
                st.metric("Duration", f"{duration:.2f}s")


class MultiPhaseStreamHandler:
    """
    Handle streaming for multi-phase workflows.
    
    This class manages streaming updates across multiple workflow phases,
    maintaining separate contexts for each phase.
    """
    
    def __init__(self):
        self.phase_handlers: Dict[str, StreamHandler] = {}
        self.active_phase: Optional[str] = None
    
    def create_phase_handler(self, phase_name: str, container: st.container) -> StreamHandler:
        """
        Create a handler for a specific phase.
        
        Args:
            phase_name: Name of the phase
            container: Container for this phase's updates
            
        Returns:
            StreamHandler for the phase
        """
        handler = StreamHandler(container)
        self.phase_handlers[phase_name] = handler
        return handler
    
    def set_active_phase(self, phase_name: str):
        """Set the currently active phase."""
        self.active_phase = phase_name
    
    def get_active_handler(self) -> Optional[StreamHandler]:
        """Get the handler for the active phase."""
        if self.active_phase:
            return self.phase_handlers.get(self.active_phase)
        return None
    
    async def handle_workflow_stream(
        self,
        stream: AsyncIterator[Any],
        phase_callback: Callable[[str], None]
    ):
        """
        Handle streaming for an entire workflow.
        
        Args:
            stream: Stream of workflow events
            phase_callback: Callback when phase changes
        """
        async for event in stream:
            # Detect phase changes
            if hasattr(event, 'phase'):
                self.set_active_phase(event.phase)
                phase_callback(event.phase)
            
            # Route to appropriate handler
            handler = self.get_active_handler()
            if handler:
                # Process event with active handler
                await handler.handle_event_stream(
                    self._single_event_stream(event)
                )
    
    async def _single_event_stream(self, event: Any) -> AsyncIterator[Any]:
        """Convert a single event to an async iterator."""
        yield event
    
    def display_all_metrics(self):
        """Display metrics for all phases."""
        for phase_name, handler in self.phase_handlers.items():
            with st.expander(f"{phase_name} Metrics"):
                handler.display_metrics()