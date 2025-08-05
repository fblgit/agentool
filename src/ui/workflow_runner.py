"""
Async workflow runner for Streamlit UI.

This module provides the async wrapper around the workflow execution,
handling streaming updates and progress tracking.
"""

import asyncio
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
import json
import sys
import os
import uuid

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agentool.core.injector import get_injector, AgenToolInjector
from agents.workflow import run_agentool_generation_workflow
from pydantic_ai.messages import (
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)


@dataclass
class WorkflowState:
    """Track the state of a workflow execution."""
    is_running: bool = False
    completed: bool = False
    error: Optional[str] = None
    current_phase: Optional[str] = None
    phases_completed: List[str] = field(default_factory=list)
    workflow_id: Optional[str] = None
    
    def start(self):
        """Start the workflow."""
        self.is_running = True
        self.completed = False
        self.error = None
        self.phases_completed = []
    
    def stop(self):
        """Stop the workflow."""
        self.is_running = False
    
    def complete(self):
        """Mark workflow as completed."""
        self.is_running = False
        self.completed = True
    
    def set_phase(self, phase: str):
        """Set the current phase."""
        self.current_phase = phase
    
    def complete_phase(self, phase: str):
        """Mark a phase as completed."""
        if phase not in self.phases_completed:
            self.phases_completed.append(phase)
    
    def get_progress(self) -> float:
        """Get overall progress as a fraction."""
        total_phases = 4  # Analyzer, Specifier, Crafter, Evaluator
        return len(self.phases_completed) / total_phases


class WorkflowRunner:
    """
    Async runner for the AgenTool workflow with streaming support.
    
    This class wraps the workflow execution and provides callbacks
    for UI updates, streaming responses, and artifact tracking.
    """
    
    def __init__(
        self,
        task: str,
        model: str = "openai:gpt-4o",
        debug: bool = False,
        stream_updates: bool = True
    ):
        self.task = task
        self.model = model
        self.debug = debug
        self.stream_updates = stream_updates
        self.state = WorkflowState()
        self._stop_requested = False
        
        # Callbacks
        self.on_phase_start: Optional[Callable[[str], None]] = None
        self.on_phase_complete: Optional[Callable[[str, float], None]] = None
        self.on_phase_error: Optional[Callable[[str, str], None]] = None
        self.on_artifact_created: Optional[Callable[[str, Any], None]] = None
        self.on_stream_update: Optional[Callable[[str], None]] = None
        self.on_metrics_update: Optional[Callable[[Dict[str, Any]], None]] = None
        
        # Injector for accessing services
        self.injector = get_injector()
    
    def stop(self):
        """Request workflow stop."""
        self._stop_requested = True
        self.state.stop()
    
    async def run(self) -> Optional[str]:
        """
        Run the workflow asynchronously.
        
        Returns:
            Workflow ID if successful, None if failed
        """
        try:
            self.state.start()
            
            # Hook into workflow events
            self._setup_event_handlers()
            
            # Run the main workflow
            workflow_id = await self._run_workflow_phases()
            
            self.state.workflow_id = workflow_id
            self.state.complete()
            
            return workflow_id
            
        except Exception as e:
            self.state.error = str(e)
            if self.on_phase_error:
                self.on_phase_error(self.state.current_phase or "unknown", str(e))
            return None
    
    def _setup_event_handlers(self):
        """Set up event handlers for workflow monitoring."""
        # We'll intercept injector calls to monitor progress
        original_run = self.injector.run
        
        async def monitored_run(agent_name: str, input_data: Dict[str, Any]):
            # Track specific workflow agents
            workflow_agents = {
                'workflow_analyzer': 'analyzer',
                'workflow_specifier': 'specifier',
                'workflow_crafter': 'crafter',
                'workflow_evaluator': 'evaluator'
            }
            
            if agent_name in workflow_agents:
                phase = workflow_agents[agent_name]
                self.state.set_phase(phase)
                
                if self.on_phase_start:
                    self.on_phase_start(phase)
                
                start_time = datetime.now()
            
            # Call original
            result = await original_run(agent_name, input_data)
            
            # Track completion
            if agent_name in workflow_agents:
                phase = workflow_agents[agent_name]
                duration = (datetime.now() - start_time).total_seconds()
                
                self.state.complete_phase(phase)
                
                if self.on_phase_complete:
                    self.on_phase_complete(phase, duration)
            
            # Track artifacts
            if agent_name == 'storage_kv' and input_data.get('operation') == 'set':
                key = input_data.get('key', '')
                if 'workflow/' in key and self.on_artifact_created:
                    # Extract artifact name
                    parts = key.split('/')
                    if len(parts) >= 3:
                        artifact_name = parts[2]
                        try:
                            value = json.loads(input_data.get('value', '{}'))
                            self.on_artifact_created(artifact_name, value)
                        except:
                            pass
            
            return result
        
        # Monkey patch for monitoring
        self.injector.run = monitored_run
    
    async def _run_workflow_phases(self) -> str:
        """Run the main workflow phases."""
        # Import workflow components
        from agents.workflow import run_agentool_generation_workflow
        
        # Initialize all AgenToolkits
        await self._initialize_agentoolkits()
        
        # Run the workflow directly
        result = await run_agentool_generation_workflow(self.task, model=self.model)
        
        # Extract workflow_id from result
        # The function returns result.output which is a dict
        if isinstance(result, dict):
            workflow_id = result.get('workflow_id', self.state.workflow_id)
        else:
            # Fallback - generate a new ID
            workflow_id = str(uuid.uuid4())
        
        return workflow_id
    
    async def _initialize_agentoolkits(self):
        """Initialize all required AgenToolkits."""
        import agentoolkit
        
        # Call all create_* functions to register the agents
        for name in dir(agentoolkit):
            if name.startswith('create_'):
                create_func = getattr(agentoolkit, name)
                try:
                    # Special case for templates agent - provide absolute path
                    if name == 'create_templates_agent':
                        import os
                        templates_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../templates'))
                        agent = create_func(templates_dir=templates_path)
                    else:
                        agent = create_func()
                except Exception as e:
                    if self.debug:
                        print(f"Warning: Could not initialize {name}: {e}")
        
        # Initialize workflow agents
        try:
            from agentoolkit.workflows import (
                create_workflow_analyzer_agent,
                create_workflow_specifier_agent,
                create_workflow_crafter_agent,
                create_workflow_evaluator_agent
            )
            
            create_workflow_analyzer_agent()
            create_workflow_specifier_agent()
            create_workflow_crafter_agent()
            create_workflow_evaluator_agent()
        except ImportError as e:
            if self.debug:
                print(f"Warning: Could not import workflow agents: {e}")
    
    async def _handle_stream_update(self, update: Any):
        """Handle streaming updates from the workflow."""
        if self._stop_requested:
            raise asyncio.CancelledError("Workflow stopped by user")
        
        if self.on_stream_update:
            # Convert update to string representation
            if isinstance(update, str):
                self.on_stream_update(update)
            elif isinstance(update, TextPart):
                self.on_stream_update(update.content)
            elif isinstance(update, ToolCallPart):
                self.on_stream_update(f"🔧 Calling tool: {update.tool_name}")
            elif isinstance(update, ToolReturnPart):
                self.on_stream_update(f"✅ Tool result received")
            else:
                self.on_stream_update(str(update))
    
    async def get_artifacts(self) -> Dict[str, Any]:
        """
        Retrieve all artifacts from the workflow.
        
        Returns:
            Dictionary of artifact name to data
        """
        artifacts = {}
        
        if not self.state.workflow_id:
            return artifacts
        
        # Get all workflow keys
        try:
            result = await self.injector.run('storage_kv', {
                'operation': 'keys',
                'pattern': f'workflow/{self.state.workflow_id}/*'
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result.data if hasattr(result, 'data') else result
            
            keys = data.get('data', {}).get('keys', [])
            
            # Retrieve each artifact
            for key in keys:
                try:
                    get_result = await self.injector.run('storage_kv', {
                        'operation': 'get',
                        'key': key
                    })
                    
                    if hasattr(get_result, 'output'):
                        value_data = json.loads(get_result.output)
                    else:
                        value_data = get_result.data if hasattr(get_result, 'data') else get_result
                    
                    if value_data.get('data', {}).get('exists'):
                        artifact_name = key.split('/')[-1]
                        artifacts[artifact_name] = json.loads(value_data['data']['value'])
                except:
                    pass
        
        except Exception as e:
            if self.debug:
                print(f"Error retrieving artifacts: {e}")
        
        return artifacts