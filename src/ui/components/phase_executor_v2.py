# -*- coding: utf-8 -*-
"""
Phase Executor V2 - Fragment-based workflow phase execution.

This component handles individual workflow phase execution with real-time
updates using Streamlit fragments, enabling non-blocking UI updates.
"""

import streamlit as st
import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from agentool.core.injector import AgenToolInjector
from agents.workflow import WorkflowState
from .container_manager import get_container_manager
from .theme_manager import get_theme_manager
from ..utils.fragments import create_auto_fragment


class PhaseStatus(Enum):
    """Phase execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PhaseConfig:
    """Configuration for a workflow phase."""
    agent_name: str
    operation: str
    description: str
    icon: str
    input_builder: Callable[[WorkflowState], Dict[str, Any]]
    result_extractor: Callable[[Any], Dict[str, Any]]
    artifact_patterns: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    estimated_duration: float = 30.0


@dataclass
class PhaseResult:
    """Result from executing a workflow phase."""
    phase_name: str
    status: PhaseStatus
    success: bool
    duration: float
    started_at: datetime
    completed_at: Optional[datetime] = None
    data: Optional[Dict[str, Any]] = None
    artifacts: List[str] = field(default_factory=list)
    summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: float = 0.0
    logs: List[str] = field(default_factory=list)


class PhaseExecutorV2:
    """
    Fragment-based phase executor for non-blocking workflow execution.
    
    This executor runs phases in fragments, allowing real-time UI updates
    without interrupting the workflow execution.
    """
    
    def __init__(self, injector: AgenToolInjector, debug_mode: bool = False):
        """
        Initialize the phase executor.
        
        Args:
            injector: AgenTool injector instance
            debug_mode: Whether to show debug information
        """
        self.injector = injector
        self.debug_mode = debug_mode
        self.container_manager = get_container_manager()
        self.theme_manager = get_theme_manager()
        self.phase_configs = self._create_phase_configs()
        
        # Session state is already initialized in app_v2.py
    
    def _create_phase_configs(self) -> Dict[str, PhaseConfig]:
        """Create configuration for each workflow phase."""
        return {
            "analyzer": PhaseConfig(
                agent_name="workflow_analyzer",
                operation="analyze",
                description="Analyzing task requirements and identifying tools",
                icon="🔍",
                input_builder=lambda state: {
                    "operation": "analyze",
                    "task_description": state.task_description,
                    "workflow_id": state.workflow_id,
                    "model": state.model
                },
                result_extractor=self._extract_analyzer_result,
                artifact_patterns=[
                    "workflow/{workflow_id}/catalog",
                    "workflow/{workflow_id}/analysis",
                    "workflow/{workflow_id}/missing_tools"
                ],
                prerequisites=[],
                estimated_duration=15.0
            ),
            
            "specification": PhaseConfig(
                agent_name="workflow_specifier",
                operation="specify",
                description="Creating detailed specifications for each tool",
                icon="📋",
                input_builder=lambda state: {
                    "operation": "specify",
                    "workflow_id": state.workflow_id,
                    "model": state.model
                },
                result_extractor=self._extract_specification_result,
                artifact_patterns=[
                    "workflow/{workflow_id}/specs",
                    "workflow/{workflow_id}/specifications/{tool_name}"
                ],
                prerequisites=["analyzer"],
                estimated_duration=20.0
            ),
            
            "crafter": PhaseConfig(
                agent_name="workflow_crafter",
                operation="craft",
                description="Generating implementation code",
                icon="💻",
                input_builder=lambda state: {
                    "operation": "craft",
                    "workflow_id": state.workflow_id,
                    "model": state.model
                },
                result_extractor=self._extract_crafter_result,
                artifact_patterns=[
                    "workflow/{workflow_id}/implementations/{tool_name}",
                    "generated/{workflow_id}/{file_path}"
                ],
                prerequisites=["specification"],
                estimated_duration=30.0
            ),
            
            "evaluator": PhaseConfig(
                agent_name="workflow_evaluator",
                operation="evaluate",
                description="Validating and improving generated code",
                icon="✅",
                input_builder=lambda state: {
                    "operation": "evaluate",
                    "workflow_id": state.workflow_id,
                    "model": state.model,
                    "auto_fix": True
                },
                result_extractor=self._extract_evaluator_result,
                artifact_patterns=[
                    "workflow/{workflow_id}/validations/{tool_name}",
                    "generated/{workflow_id}/final/",
                    "generated/{workflow_id}/SUMMARY.md"
                ],
                prerequisites=["crafter"],
                estimated_duration=25.0
            ),
            
            "test_analyzer": PhaseConfig(
                agent_name="workflow_test_analyzer",
                operation="analyze",
                description="Analyzing test requirements",
                icon="🧪",
                input_builder=lambda state: {
                    "operation": "analyze",
                    "workflow_id": state.workflow_id,
                    "model": state.model
                },
                result_extractor=self._extract_test_analyzer_result,
                artifact_patterns=[
                    "workflow/{workflow_id}/test_analysis/{tool_name}"
                ],
                prerequisites=["specification"],
                estimated_duration=15.0
            ),
            
            "test_stubber": PhaseConfig(
                agent_name="workflow_test_stubber",
                operation="stub",
                description="Creating test structure and placeholders",
                icon="🏗️",
                input_builder=lambda state: {
                    "operation": "stub",
                    "workflow_id": state.workflow_id,
                    "model": state.model
                },
                result_extractor=self._extract_test_stubber_result,
                artifact_patterns=[
                    "workflow/{workflow_id}/test_stub/{tool_name}",
                    "generated/{workflow_id}/test_stubs/test_{tool_name}.py"
                ],
                prerequisites=["test_analyzer"],
                estimated_duration=20.0
            ),
            
            "test_crafter": PhaseConfig(
                agent_name="workflow_test_crafter",
                operation="craft",
                description="Implementing comprehensive tests",
                icon="🔨",
                input_builder=lambda state: {
                    "operation": "craft",
                    "workflow_id": state.workflow_id,
                    "model": state.model
                },
                result_extractor=self._extract_test_crafter_result,
                artifact_patterns=[
                    "workflow/{workflow_id}/test_implementation/{tool_name}",
                    "generated/{workflow_id}/tests/test_{tool_name}.py",
                    "generated/{workflow_id}/tests/TEST_SUMMARY_{tool_name}.md"
                ],
                prerequisites=["test_stubber"],
                estimated_duration=25.0
            )
        }
    
    def render_phase_card(self, phase_name: str, workflow_state: WorkflowState, 
                          container_key: str) -> PhaseResult:
        """
        Render a phase card with fragment-based execution.
        
        Args:
            phase_name: Name of the phase
            workflow_state: Current workflow state
            container_key: Container key for the phase card
            
        Returns:
            PhaseResult object
        """
        if phase_name not in self.phase_configs:
            raise ValueError(f"Unknown phase: {phase_name}")
        
        config = self.phase_configs[phase_name]
        
        # Create or get phase container
        container = self.container_manager.get_or_create_container(
            f"{container_key}_{phase_name}",
            height=None,
            border=False
        )
        
        # Create fragment for this phase
        phase_fragment = self._create_phase_fragment(phase_name, config, workflow_state)
        
        # Render the phase card
        with container:
            phase_fragment()
        
        # Return result if available
        return st.session_state.phase_executor_v2['phase_results'].get(
            phase_name,
            PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.PENDING,
                success=False,
                duration=0,
                started_at=datetime.now()
            )
        )
    
    def _create_phase_fragment(self, phase_name: str, config: PhaseConfig, 
                               workflow_state: WorkflowState):
        """Create a function for phase execution and display."""
        
        def phase_fragment():
            # Get or create phase result
            if phase_name not in st.session_state.phase_executor_v2['phase_results']:
                st.session_state.phase_executor_v2['phase_results'][phase_name] = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.PENDING,
                    success=False,
                    duration=0,
                    started_at=datetime.now()
                )
            
            result = st.session_state.phase_executor_v2['phase_results'][phase_name]
            
            # Apply theme styling
            status_class = f"phase-{result.status.value}"
            
            # Phase card container
            with st.container():
                
                # Header row
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"### {config.icon} {phase_name.replace('_', ' ').title()}")
                    
                    if result.status == PhaseStatus.RUNNING:
                        st.caption(config.description)
                
                with col2:
                    # Status badge
                    self._render_status_badge(result.status)
                
                with col3:
                    # Duration or estimate
                    if result.status == PhaseStatus.COMPLETED:
                        st.metric("Duration", f"{result.duration:.1f}s")
                    elif result.status == PhaseStatus.RUNNING:
                        elapsed = (datetime.now() - result.started_at).total_seconds()
                        st.metric("Elapsed", f"{elapsed:.1f}s")
                    else:
                        st.caption(f"Est: {config.estimated_duration:.0f}s")
                
                # Progress bar for running phases
                if result.status == PhaseStatus.RUNNING:
                    progress = result.progress
                    st.progress(progress, text=f"Progress: {progress:.0%}")
                    
                    # Live logs
                    if result.logs:
                        with st.expander("Live logs", expanded=True):
                            for log in result.logs[-5:]:  # Show last 5 logs
                                st.text(log)
                
                # Results section for completed phases
                elif result.status == PhaseStatus.COMPLETED:
                    with st.expander("Results", expanded=False):
                        # Metrics
                        if result.summary:
                            cols = st.columns(min(len(result.summary), 4))
                            for i, (key, value) in enumerate(list(result.summary.items())[:4]):
                                with cols[i]:
                                    st.metric(
                                        key.replace('_', ' ').title(),
                                        value if not isinstance(value, (list, dict)) else len(value)
                                    )
                        
                        # Artifacts
                        if result.artifacts:
                            st.markdown("**Artifacts Created:**")
                            self._render_artifact_pills(result.artifacts[:5])
                            if len(result.artifacts) > 5:
                                st.caption(f"... and {len(result.artifacts) - 5} more")
                
                # Error display for failed phases
                elif result.status == PhaseStatus.FAILED:
                    with st.expander("Error details", expanded=True):
                        st.error(result.error or "Unknown error occurred")
                        if self.debug_mode and result.logs:
                            st.code('\n'.join(result.logs))
                
                # Action buttons
                if result.status == PhaseStatus.PENDING and self._can_execute_phase(phase_name, workflow_state):
                    if st.button(f"▶️ Start {phase_name}", key=f"start_{phase_name}", use_container_width=True):
                        self._start_phase_execution(phase_name, config, workflow_state)
                
                elif result.status == PhaseStatus.FAILED:
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"🔄 Retry", key=f"retry_{phase_name}", use_container_width=True):
                            self._start_phase_execution(phase_name, config, workflow_state)
                    with col2:
                        if st.button(f"⏭️ Skip", key=f"skip_{phase_name}", use_container_width=True):
                            self._skip_phase(phase_name)
        
        return phase_fragment
    
    def _start_phase_execution(self, phase_name: str, config: PhaseConfig, 
                               workflow_state: WorkflowState):
        """Start executing a phase."""
        # Log to live feed
        from .live_feed import log_phase_start
        log_phase_start(phase_name, config.description)
        
        # Update phase status
        st.session_state.phase_executor_v2['phase_results'][phase_name] = PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.RUNNING,
            success=False,
            duration=0,
            started_at=datetime.now(),
            progress=0.0,
            logs=["Starting phase execution..."]
        )
        
        st.session_state.phase_executor_v2['current_phase'] = phase_name
        
        # Execute phase asynchronously
        self._execute_phase_async(phase_name, config, workflow_state)
    
    def _execute_phase_async(self, phase_name: str, config: PhaseConfig, 
                             workflow_state: WorkflowState):
        """Execute a phase asynchronously with progress updates."""
        result = st.session_state.phase_executor_v2['phase_results'][phase_name]
        
        try:
            # Build input
            result.logs.append("Building phase input...")
            result.progress = 0.1
            phase_input = config.input_builder(workflow_state)
            
            # Execute phase
            result.logs.append(f"Executing {config.agent_name}...")
            result.progress = 0.3
            
            if phase_name in ["test_analyzer", "test_stubber", "test_crafter"]:
                # Multi-tool phases
                exec_result = self._execute_multi_tool_phase(
                    phase_name, config, phase_input, workflow_state, result
                )
            else:
                # Single execution phases
                exec_result = asyncio.run(self.injector.run(config.agent_name, phase_input))
            
            # Process results
            result.logs.append("Processing results...")
            result.progress = 0.8
            
            data = self._extract_result_data(exec_result)
            extracted_data = config.result_extractor(data)
            
            # Capture artifacts
            artifacts = self._capture_artifacts(config, workflow_state.workflow_id, extracted_data)
            
            # Update result
            result.status = PhaseStatus.COMPLETED
            result.success = True
            result.completed_at = datetime.now()
            result.duration = (result.completed_at - result.started_at).total_seconds()
            result.data = extracted_data
            result.artifacts = artifacts
            
            # Log completion to live feed
            from .live_feed import log_phase_complete
            log_phase_complete(phase_name, result.duration, len(artifacts))
            result.summary = self._create_phase_summary(phase_name, extracted_data)
            result.progress = 1.0
            result.logs.append(f"Phase completed successfully in {result.duration:.1f}s")
            
            # Clear current phase so next phase can start
            st.session_state.phase_executor_v2['current_phase'] = None
            
            # Force UI update
            st.rerun()
            
            # Update metrics
            st.session_state.phase_executor_v2['metrics']['phases_completed'] += 1
            st.session_state.phase_executor_v2['metrics']['total_duration'] += result.duration
            st.session_state.phase_executor_v2['metrics']['artifacts_created'] += len(artifacts)
            
            # Store artifacts by phase
            st.session_state.phase_executor_v2['artifacts_by_phase'][phase_name] = artifacts
            
            # Update workflow state
            setattr(workflow_state, f"{phase_name}_completed", True)
            
            if workflow_state.metadata:
                workflow_state.metadata.phase_durations[phase_name] = result.duration
                workflow_state.metadata.models_used[phase_name] = workflow_state.model
            
        except Exception as e:
            # Handle failure
            result.status = PhaseStatus.FAILED
            result.success = False
            result.completed_at = datetime.now()
            result.duration = (result.completed_at - result.started_at).total_seconds()
            result.error = str(e)
            result.logs.append(f"Error: {e}")
            
            # Log error to live feed
            from .live_feed import log_phase_error
            log_phase_error(phase_name, str(e))
            
            # Update metrics
            st.session_state.phase_executor_v2['metrics']['phases_failed'] += 1
            
            if self.debug_mode:
                import traceback
                result.logs.extend(traceback.format_exc().split('\n'))
        
        finally:
            # Clear current phase
            if st.session_state.phase_executor_v2['current_phase'] == phase_name:
                st.session_state.phase_executor_v2['current_phase'] = None
    
    def _execute_multi_tool_phase(self, phase_name: str, config: PhaseConfig,
                                  base_input: Dict[str, Any], workflow_state: WorkflowState,
                                  result: PhaseResult) -> Dict[str, Any]:
        """Execute phases that iterate over multiple tools."""
        # Get list of tools from specifications
        specs_key = f'workflow/{workflow_state.workflow_id}/specs'
        specs_result = asyncio.run(self.injector.run('storage_kv', {
            'operation': 'get',
            'key': specs_key
        }))
        
        if not specs_result.success or not specs_result.data.get('exists', False):
            raise RuntimeError(f"No specifications found for {phase_name}")
        
        spec_output = json.loads(specs_result.data['value'])
        tools = spec_output.get('specifications', [])
        tools_processed = 0
        results = []
        
        total_tools = len(tools)
        
        # Process each tool
        for i, spec in enumerate(tools):
            tool_name = spec.get('name')
            if not tool_name:
                continue
            
            # Update progress
            result.progress = 0.3 + (0.5 * (i / total_tools))
            result.logs.append(f"Processing tool {i+1}/{total_tools}: {tool_name}")
            
            # Check prerequisites
            if not self._check_tool_prerequisites(phase_name, tool_name, workflow_state):
                result.logs.append(f"Skipping {tool_name} - prerequisites not met")
                continue
            
            # Execute for this tool
            tool_input = base_input.copy()
            tool_input['tool_name'] = tool_name
            
            tool_result = asyncio.run(self.injector.run(config.agent_name, tool_input))
            result_data = self._extract_result_data(tool_result)
            
            if result_data.get('success', False):
                tools_processed += 1
                results.append(result_data)
                result.logs.append(f"✅ Completed {tool_name}")
            else:
                result.logs.append(f"⚠️ Failed {tool_name}")
        
        result.logs.append(f"Processed {tools_processed}/{total_tools} tools")
        
        return {
            'success': tools_processed > 0,
            'message': f'Processed {tools_processed} tools',
            'data': {
                'tools_processed': tools_processed,
                'results': results,
                'tools': [spec.get('name') for spec in tools if spec.get('name')]
            }
        }
    
    def _check_tool_prerequisites(self, phase_name: str, tool_name: str, 
                                  workflow_state: WorkflowState) -> bool:
        """Check if prerequisites are met for a tool in a phase."""
        if phase_name == "test_stubber":
            check_key = f'workflow/{workflow_state.workflow_id}/test_analysis/{tool_name}'
        elif phase_name == "test_crafter":
            check_key = f'workflow/{workflow_state.workflow_id}/test_stub/{tool_name}'
        else:
            return True
        
        check_result = asyncio.run(self.injector.run('storage_kv', {
            'operation': 'exists',
            'key': check_key
        }))
        
        return check_result.success and check_result.data.get('exists', False)
    
    def _can_execute_phase(self, phase_name: str, workflow_state: WorkflowState) -> bool:
        """Check if a phase can be executed based on prerequisites."""
        config = self.phase_configs.get(phase_name)
        if not config:
            return False
        
        # Check prerequisites - look at phase results instead of workflow_state
        for prereq in config.prerequisites:
            prereq_result = st.session_state.phase_executor_v2['phase_results'].get(prereq)
            if not prereq_result or prereq_result.status != PhaseStatus.COMPLETED:
                # Prerequisite phase not completed
                return False
        
        # Check if not already running or completed
        result = st.session_state.phase_executor_v2['phase_results'].get(phase_name)
        if result and result.status in [PhaseStatus.RUNNING, PhaseStatus.COMPLETED]:
            return False
        
        # Check if another phase is running
        if st.session_state.phase_executor_v2['current_phase']:
            return False
        
        return True
    
    def _is_phase_running(self, phase_name: str) -> bool:
        """Check if a phase is currently running."""
        result = st.session_state.phase_executor_v2['phase_results'].get(phase_name)
        return result and result.status == PhaseStatus.RUNNING
    
    def _skip_phase(self, phase_name: str):
        """Skip a phase."""
        result = st.session_state.phase_executor_v2['phase_results'].get(phase_name)
        if result:
            result.status = PhaseStatus.SKIPPED
            result.completed_at = datetime.now()
            result.duration = 0
    
    def _render_status_badge(self, status: PhaseStatus):
        """Render a status badge."""
        status_config = {
            PhaseStatus.PENDING: ("⏳", "Pending", ""),
            PhaseStatus.RUNNING: ("⚡", "Running", "status-running"),
            PhaseStatus.COMPLETED: ("✅", "Complete", "status-complete"),
            PhaseStatus.FAILED: ("❌", "Failed", "status-error"),
            PhaseStatus.SKIPPED: ("⏭️", "Skipped", "")
        }
        
        icon, text, css_class = status_config.get(status, ("", "Unknown", ""))
        
        st.markdown(
            f'<span class="status-badge {css_class}">{icon} {text}</span>',
            unsafe_allow_html=True
        )
    
    def _render_artifact_pills(self, artifacts: List[str]):
        """Render artifacts as pills."""
        for artifact in artifacts:
            # Determine type from artifact path
            if "storage_kv:" in artifact:
                badge_class = "artifact-kv"
                icon = "🔑"
            elif "storage_fs:" in artifact:
                badge_class = "artifact-fs"
                icon = "📁"
            else:
                badge_class = ""
                icon = "📄"
            
            name = artifact.split('/')[-1] if '/' in artifact else artifact
            
            st.markdown(
                f'<span class="artifact-badge {badge_class}">{icon} {name}</span>',
                unsafe_allow_html=True
            )
    
    def _extract_result_data(self, result: Any) -> Dict[str, Any]:
        """Extract data from result."""
        if hasattr(result, 'success') and hasattr(result, 'data'):
            return {'success': result.success, 'data': result.data} if result.success else {'success': False}
        return result if isinstance(result, dict) else {}
    
    def _capture_artifacts(self, config: PhaseConfig, workflow_id: str,
                           extracted_data: Optional[Dict[str, Any]] = None) -> List[str]:
        """Capture artifacts created during phase execution."""
        artifacts = []
        
        # Process based on extracted data
        if extracted_data:
            # Handle specifications
            if 'specifications' in extracted_data:
                for spec in extracted_data.get('specifications', []):
                    if isinstance(spec, dict) and 'name' in spec:
                        artifacts.append(f"storage_kv:workflow/{workflow_id}/specifications/{spec['name']}")
            
            # Handle implementations
            if 'implementations' in extracted_data:
                for impl in extracted_data.get('implementations', []):
                    if isinstance(impl, dict):
                        tool_name = impl.get('tool_name', '')
                        if tool_name:
                            artifacts.append(f"storage_kv:workflow/{workflow_id}/implementations/{tool_name}")
            
            # Handle validations
            if 'validations' in extracted_data:
                for validation in extracted_data.get('validations', []):
                    if isinstance(validation, dict):
                        tool_name = validation.get('tool_name', '')
                        if tool_name:
                            artifacts.append(f"storage_kv:workflow/{workflow_id}/validations/{tool_name}")
        
        # Add standard patterns
        for pattern in config.artifact_patterns:
            # Skip patterns with {tool_name} if we don't have tool names
            if '{tool_name}' in pattern:
                # Try to get tool names from extracted_data
                tool_names = []
                if extracted_data:
                    # Look for tool names in various places
                    if 'tool_name' in extracted_data:
                        tool_names.append(extracted_data['tool_name'])
                    elif 'specifications' in extracted_data:
                        for spec in extracted_data.get('specifications', []):
                            if isinstance(spec, dict) and 'name' in spec:
                                tool_names.append(spec['name'])
                    elif 'implementations' in extracted_data:
                        for impl in extracted_data.get('implementations', []):
                            if isinstance(impl, dict) and 'tool_name' in impl:
                                tool_names.append(impl['tool_name'])
                
                # Create artifacts for each tool name
                for tool_name in tool_names:
                    expanded = pattern.replace("{workflow_id}", workflow_id)
                    expanded = expanded.replace("{tool_name}", tool_name)
                    
                    if expanded.startswith("workflow/"):
                        artifacts.append(f"storage_kv:{expanded}")
                    elif expanded.startswith("generated/"):
                        artifacts.append(f"storage_fs:{expanded}")
            else:
                # Pattern without {tool_name}
                pattern = pattern.replace("{workflow_id}", workflow_id)
                
                if pattern.startswith("workflow/"):
                    artifacts.append(f"storage_kv:{pattern}")
                elif pattern.startswith("generated/"):
                    artifacts.append(f"storage_fs:{pattern}")
        
        return list(set(artifacts))  # Remove duplicates
    
    def _create_phase_summary(self, phase_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of phase results."""
        if phase_name == "analyzer":
            return {
                "existing_tools": len(data.get("existing_tools", [])),
                "missing_tools": len(data.get("missing_tools", [])),
                "solution_name": data.get("name", "Unknown")
            }
        
        elif phase_name == "specification":
            return {
                "specifications_created": len(data.get("specifications", [])),
                "tools": [spec.get("name") for spec in data.get("specifications", [])][:5]
            }
        
        elif phase_name == "crafter":
            if 'implementations' in data:
                return {
                    "lines_of_code": data.get('total_lines', 0),
                    "tools_generated": len(data.get('implementations', [])),
                    "files": len(data.get('files', []))
                }
            else:
                code = data.get("code", "")
                return {
                    "lines_of_code": code.count("\n") if code else 0,
                    "file_path": data.get("file_path", "")
                }
        
        elif phase_name == "evaluator":
            return {
                "tools_validated": data.get('tools_validated', 0),
                "tools_ready": data.get('tools_ready', 0),
                "total_issues": data.get('total_issues', 0),
                "total_fixes": data.get('total_fixes', 0)
            }
        
        elif phase_name in ["test_analyzer", "test_stubber", "test_crafter"]:
            tools_count = len(data.get('tools', []))
            return {
                f"{phase_name.replace('test_', '')}s_created": tools_count,
                "tools_processed": data.get('tools_processed', 0)
            }
        
        return {}
    
    # Result extractors
    def _extract_analyzer_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract analyzer phase results."""
        return data.get("data", {})
    
    def _extract_specification_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract specification phase results."""
        return data.get("data", {})
    
    def _extract_crafter_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract crafter phase results."""
        return data.get("data", {})
    
    def _extract_evaluator_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract evaluator phase results."""
        validation_data = data.get("data", {})
        validation_data["issues_count"] = len(validation_data.get("issues", []))
        validation_data["fixes_count"] = len(validation_data.get("fixes_applied", []))
        return validation_data
    
    def _extract_test_analyzer_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract test analyzer phase results."""
        return data.get("data", {})
    
    def _extract_test_stubber_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract test stubber phase results."""
        return data.get("data", {})
    
    def _extract_test_crafter_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract test crafter phase results."""
        return data.get("data", {})