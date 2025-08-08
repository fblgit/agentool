# -*- coding: utf-8 -*-
"""
Workflow UI V2 - Modernized UI with fragment architecture for non-blocking execution.

This is the main orchestrator that combines all the modernized components
to provide a seamless, non-blocking workflow execution experience.
"""

import streamlit as st
import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import time

from agentool.core.injector import get_injector
from agents.workflow import WorkflowState, WorkflowMetadata

# Import modernized components
from .components.container_manager import get_container_manager, UIContainerManager
from .components.theme_manager import get_theme_manager, ThemeManager, ThemeMode
from .components.phase_executor_v2 import PhaseExecutorV2, PhaseStatus
from .components.artifact_viewer_v2 import ArtifactViewerV2
from .components.live_feed import LiveFeed, log_phase_start, log_phase_complete, log_phase_error, log_info
from .components.workflow_viewer_v2 import WorkflowViewerV2
from .utils.fragments import create_auto_fragment, FragmentManager
from .utils.styling import apply_custom_css, apply_responsive_layout, create_animated_header


def get_workflow_phases(generate_tests: bool = True) -> List[str]:
    """Get the list of workflow phases based on configuration.
    
    Args:
        generate_tests: Whether to include test generation phases
        
    Returns:
        List of phase names in execution order
    """
    main_phases = ['analyzer', 'specification', 'crafter', 'evaluator']
    test_phases = ['test_analyzer', 'test_stubber', 'test_crafter'] if generate_tests else []
    return main_phases + test_phases


class WorkflowUIV2:
    """
    Main workflow UI with fragment-based architecture for non-blocking execution.
    
    This UI orchestrates all the modernized components to provide real-time
    updates without page reruns interrupting workflow execution.
    """
    
    def __init__(self, debug_mode: bool = False):
        """
        Initialize the workflow UI.
        
        Args:
            debug_mode: Whether to show debug information
        """
        self.debug_mode = debug_mode
        self.injector = get_injector()
        
        # Initialize managers
        self.container_manager = get_container_manager()
        self.theme_manager = get_theme_manager()
        self.fragment_manager = FragmentManager()
        
        # Initialize components
        self.phase_executor = PhaseExecutorV2(self.injector, debug_mode)
        self.artifact_viewer = ArtifactViewerV2(debug_mode)
        self.live_feed = LiveFeed(max_items=200, auto_scroll=True)
        self.workflow_viewer = WorkflowViewerV2()
        
        # Session state is already initialized in app_v2.py
    
    
    def render(self):
        """
        Render the main workflow UI.
        
        This is the entry point for the UI that sets up the layout
        and coordinates all components.
        
        Note: Page config must be set in app_v2.py before this is called.
        """
        # Continue with rendering
        self.render_content()
    
    def render_content(self):
        """
        Render the UI content after page config has been set.
        
        This method handles all the UI rendering without setting page config.
        """
        # Apply custom styling
        apply_custom_css()
        apply_responsive_layout()
        self.theme_manager.apply_theme()
        
        # Header
        self._render_header()
        
        # Main layout
        if st.session_state.workflow_ui_v2['show_sidebar']:
            self._render_with_sidebar()
        else:
            self._render_full_width()
        
        # Footer with metrics
        self._render_footer()
    
    def _render_header(self):
        """Render the UI header with title and controls."""
        # Don't use managed containers for the header to avoid widget ID changes
        with st.container():
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                create_animated_header(
                    "🚀 AgenTool Workflow Engine v2",
                    "Fragment-based, non-blocking workflow execution"
                )
            
            with col2:
                # Tab selection with selectbox to avoid disconnection
                tabs = {
                    'execution': '⚡ Execution',
                    'artifacts': '📦 Artifacts',
                    'visualization': '📊 Visualization',
                    'logs': '📝 Logs'
                }
                
                # Get current tab
                current_tab = st.session_state.workflow_ui_v2.get('current_tab', 'execution')
                if current_tab not in tabs:
                    current_tab = 'execution'
                
                # Use selectbox which handles state better than radio
                selected = st.selectbox(
                    "View",
                    options=list(tabs.keys()),
                    index=list(tabs.keys()).index(current_tab),
                    format_func=lambda x: tabs[x],
                    key="main_tab_selector",
                    label_visibility="collapsed",
                    on_change=lambda: st.session_state.workflow_ui_v2.update({'current_tab': st.session_state.main_tab_selector})
                )
            
            with col3:
                # Layout controls
                col3_1, col3_2 = st.columns(2)
                
                with col3_1:
                    # Use checkbox instead of button to avoid rerun
                    st.session_state.workflow_ui_v2['show_sidebar'] = st.checkbox(
                        "📐",
                        value=st.session_state.workflow_ui_v2['show_sidebar'],
                        key="toggle_sidebar",
                        help="Toggle sidebar",
                        label_visibility="collapsed"
                    )
                
                with col3_2:
                    # Theme toggle using selectbox to avoid rerun
                    current_mode = self.theme_manager.mode
                    theme_options = {'dark': '🌙 Dark', 'light': '☀️ Light'}
                    
                    def update_theme():
                        theme_value = st.session_state.theme_selector
                        new_mode = ThemeMode.DARK if theme_value == 'dark' else ThemeMode.LIGHT
                        self.theme_manager.mode = new_mode
                        st.session_state.theme_manager['mode'] = new_mode.value
                    
                    selected_theme = st.selectbox(
                        "Theme",
                        list(theme_options.keys()),
                        index=0 if current_mode == ThemeMode.DARK else 1,
                        format_func=lambda x: theme_options[x],
                        key="theme_selector",
                        label_visibility="collapsed",
                        on_change=update_theme
                    )
    
    def _render_with_sidebar(self):
        """Render UI with sidebar layout."""
        # Sidebar
        with st.sidebar:
            self._render_sidebar()
        
        # Main content area
        self._render_main_content()
    
    def _render_full_width(self):
        """Render UI in full width mode."""
        # Create columns for pseudo-sidebar
        col1, col2 = st.columns([1, 4])
        
        with col1:
            with st.container():
                self._render_sidebar()
        
        with col2:
            self._render_main_content()
    
    def _render_sidebar(self):
        """Render the sidebar with workflow configuration."""
        st.markdown("### ⚙️ Workflow Configuration")
        
        # Workflow input form
        with st.form("workflow_config"):
            # Task description
            task_description = st.text_area(
                "Task Description",
                placeholder="Describe what you want to build...",
                height=200,
                key="task_input"
            )
            
            # Model selection
            model_options = ["openai:gpt-5-nano", "openai:gpt-5-mini", "openai:gpt-5-chat-latest", "openai:gpt-5", "openai:gpt-4o", "openai:gpt-4o-mini", "anthropic:claude-3-5-sonnet-latest"]
            model = st.selectbox(
                "Main Model (Default for all phases)",
                model_options,
                key="model_input"
            )
            
            # Test generation first (so we know whether to show test phases)
            generate_tests = st.checkbox(
                "Generate Tests",
                value=True,
                key="tests_input"
            )
            
            st.divider()
            
            # Per-phase model configuration - simple flat layout
            st.markdown("**Phase-Specific Models**")
            st.caption("Customize model for each phase (defaults to main model)")
            
            phase_models = {}
            
            # Define phases with friendly names
            phases = [
                ('analyzer', 'Analyzer'),
                ('specification', 'Specification'),
                ('crafter', 'Crafter'),
                ('evaluator', 'Evaluator')
            ]
            
            # Add test phases if tests are enabled
            if generate_tests:
                phases.extend([
                    ('test_analyzer', 'Test Analyzer'),
                    ('test_stubber', 'Test Stubber'),
                    ('test_crafter', 'Test Crafter')
                ])
            
            # Get the default model index - use the current value in session state if available
            current_main_model = st.session_state.get('model_input', model_options[0])
            default_model_index = model_options.index(current_main_model) if current_main_model in model_options else 0
            
            for phase_key, phase_name in phases:
                # Simple model selector for each phase, defaulting to main model
                selected_model = st.selectbox(
                    phase_name,
                    model_options,
                    index=default_model_index,
                    key=f"model_{phase_key}"
                )
                # Always add to phase_models - we'll filter later when form is submitted
                phase_models[phase_key] = selected_model
            
            st.divider()
            
            # Execution mode
            execution_mode = st.radio(
                "Execution Mode",
                ["automatic", "manual", "step"],
                format_func=lambda x: {
                    "automatic": "🚀 Automatic",
                    "manual": "🎮 Manual",
                    "step": "👣 Step-by-Step"
                }[x],
                key="execution_mode_input"
            )
            
            # Submit button
            submitted = st.form_submit_button(
                "▶️ Start Workflow",
                use_container_width=True,
                disabled=st.session_state.workflow_ui_v2['workflow_active']
            )
            
            if submitted and task_description:
                # Collect actual phase models from session state (form values)
                actual_phase_models = {}
                
                # Log all phase model values for debugging
                log_info(f"Form submitted - Main model: {model}")
                
                for phase_key, phase_name in phases:
                    phase_model_key = f"model_{phase_key}"
                    if phase_model_key in st.session_state:
                        phase_model = st.session_state[phase_model_key]
                        log_info(f"Phase {phase_key}: {phase_model} (session state)")
                        # Only include if different from main model
                        if phase_model != model:
                            actual_phase_models[phase_key] = phase_model
                            log_info(f"  -> Added {phase_key} to overrides with model {phase_model}")
                
                log_info(f"Final phase_models dict: {actual_phase_models}")
                
                self._start_workflow(task_description, model, generate_tests, execution_mode, actual_phase_models)
        
        # Workflow controls
        if st.session_state.workflow_ui_v2['workflow_active']:
            st.divider()
            st.markdown("### 🎮 Workflow Controls")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("⏸️ Pause", key="pause_workflow", use_container_width=True):
                    st.session_state.phase_executor_v2['execution_paused'] = True
                    log_info("Workflow paused")
            
            with col2:
                if st.button("🛑 Stop", key="stop_workflow", use_container_width=True):
                    self._stop_workflow()
        
        # View controls
        st.divider()
        st.markdown("### 👁️ View Options")
        
        st.session_state.workflow_ui_v2['show_live_feed'] = st.checkbox(
            "Show Live Feed",
            value=st.session_state.workflow_ui_v2['show_live_feed'],
            key="toggle_live_feed"
        )
        
        st.session_state.workflow_ui_v2['show_visualizations'] = st.checkbox(
            "Show Visualizations",
            value=st.session_state.workflow_ui_v2['show_visualizations'],
            key="toggle_visualizations"
        )
        
        # Debug mode
        if st.checkbox("Debug Mode", value=self.debug_mode, key="debug_mode"):
            self.debug_mode = True
            self.phase_executor.debug_mode = True
    
    def _render_main_content(self):
        """Render the main content area based on selected tab."""
        current_tab = st.session_state.workflow_ui_v2.get('current_tab', 'execution')
        
        # Debug info
        if self.debug_mode:
            with st.expander("Debug Info", expanded=False):
                st.write(f"Current tab: {current_tab}")
                st.write(f"Session state keys: {list(st.session_state.keys())}")
                st.write(f"workflow_ui_v2 state: {st.session_state.workflow_ui_v2}")
        
        try:
            if current_tab == 'execution':
                self._render_execution_tab()
            elif current_tab == 'artifacts':
                self._render_artifacts_tab()
            elif current_tab == 'visualization':
                self._render_visualization_tab()
            elif current_tab == 'logs':
                self._render_logs_tab()
            else:
                st.error(f"Unknown tab: {current_tab}")
        except Exception as e:
            st.error(f"Error rendering {current_tab} tab: {e}")
            if self.debug_mode:
                import traceback
                st.code(traceback.format_exc())
    
    def _render_execution_tab(self):
        """Render the execution tab with phase cards and live feed."""
        if st.session_state.workflow_ui_v2['show_live_feed']:
            # Split view with live feed
            col1, col2 = st.columns([3, 2])
            
            with col1:
                self._render_phase_execution()
            
            with col2:
                self.live_feed.render("execution_feed", height=600)
        else:
            # Full width phase execution
            self._render_phase_execution()
    
    def _render_phase_execution(self):
        """Render phase execution cards."""
        workflow_state = st.session_state.workflow_ui_v2.get('workflow_state')
        
        if not workflow_state:
            # No workflow active
            st.info("👋 Configure and start a workflow from the sidebar to begin.")
            
            # Show sample tasks
            st.markdown("#### 💡 Sample Tasks")
            samples = [
                "Create a data validation toolkit with schema checking and error reporting",
                "Build a REST API client with authentication and rate limiting",
                "Implement a caching system with TTL and eviction policies",
                "Create a logging framework with multiple handlers and formatters"
            ]
            
            for sample in samples:
                if st.button(f"📝 {sample}", key=f"sample_{sample[:20]}", use_container_width=True):
                    st.session_state.task_input = sample
                    # Don't rerun - the form will pick up the value
        else:
            # Show workflow info
            with st.expander("📋 Workflow Details", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Workflow ID", workflow_state.workflow_id[:8])
                
                with col2:
                    st.metric("Model", workflow_state.model.split(':')[-1])
                
                with col3:
                    if st.session_state.workflow_ui_v2['start_time']:
                        elapsed = (datetime.now() - st.session_state.workflow_ui_v2['start_time']).total_seconds()
                        st.metric("Elapsed", f"{elapsed:.0f}s")
                
                st.markdown("**Task:**")
                st.text(workflow_state.task_description)
            
            # Render phase cards
            phases = get_workflow_phases(workflow_state.generate_tests)
            
            # Group phases
            main_phases = ['analyzer', 'specification', 'crafter', 'evaluator']
            test_phases = ['test_analyzer', 'test_stubber', 'test_crafter'] if workflow_state.generate_tests else []
            
            # Main pipeline - use container with horizontal scrolling for better spacing
            st.markdown("### 🔧 Main Pipeline")
            
            # Use 2 rows of 2 columns for better spacing
            for row in range(0, len(main_phases), 2):
                cols = st.columns(2)
                for col_idx in range(2):
                    phase_idx = row + col_idx
                    if phase_idx < len(main_phases):
                        phase = main_phases[phase_idx]
                        with cols[col_idx]:
                            result = self.phase_executor.render_phase_card(
                                phase,
                                workflow_state,
                                f"main_phase_{phase}"
                            )
                            
                            # Update artifacts
                            if result.artifacts:
                                st.session_state.workflow_ui_v2['artifacts'][phase] = result.artifacts
                            
                            # Auto-execute if in automatic mode
                            if (st.session_state.workflow_ui_v2['execution_mode'] == 'automatic' and
                                result.status == PhaseStatus.PENDING and
                                self.phase_executor._can_execute_phase(phase, workflow_state)):
                                
                                # Start phase automatically
                                config = self.phase_executor.phase_configs[phase]
                                self.phase_executor._start_phase_execution(phase, config, workflow_state)
                                log_phase_start(phase, config.description)
            
            # Test pipeline (if enabled)
            if test_phases:
                st.markdown("### 🧪 Test Pipeline")
                
                # Use 2 columns for better spacing (3 test phases -> 2 cols then 1)
                for row in range(0, len(test_phases), 2):
                    cols = st.columns(2)
                    for col_idx in range(2):
                        phase_idx = row + col_idx
                        if phase_idx < len(test_phases):
                            phase = test_phases[phase_idx]
                            with cols[col_idx]:
                                result = self.phase_executor.render_phase_card(
                                    phase,
                                    workflow_state,
                                    f"test_phase_{phase}"
                                )
                                
                                # Update artifacts
                                if result.artifacts:
                                    st.session_state.workflow_ui_v2['artifacts'][phase] = result.artifacts
                                
                                # Auto-execute if in automatic mode
                                if (st.session_state.workflow_ui_v2['execution_mode'] == 'automatic' and
                                    result.status == PhaseStatus.PENDING and
                                    self.phase_executor._can_execute_phase(phase, workflow_state)):
                                    
                                    config = self.phase_executor.phase_configs[phase]
                                    self.phase_executor._start_phase_execution(phase, config, workflow_state)
                                    log_phase_start(phase, config.description)
            
            # Check if workflow is complete
            self._check_workflow_completion(workflow_state, phases)
    
    def _render_artifacts_tab(self):
        """Render the artifacts tab."""
        # Prefer artifacts tracked by the phase executor
        artifacts = st.session_state.phase_executor_v2.get('artifacts_by_phase', {})
        
        if artifacts:
            self.artifact_viewer.render(artifacts, "artifacts_tab")
        else:
            st.info("📦 No artifacts yet. Artifacts will appear here as phases complete.")
    
    def _render_visualization_tab(self):
        """Render the visualization tab."""
        workflow_state = st.session_state.workflow_ui_v2.get('workflow_state')
        
        if workflow_state:
            phase_results = st.session_state.phase_executor_v2.get('phase_results', {})
            # Use artifacts collected during phase execution
            artifacts = st.session_state.phase_executor_v2.get('artifacts_by_phase', {})
            
            self.workflow_viewer.render(
                workflow_state,
                phase_results,
                artifacts,
                "visualization_tab"
            )
        else:
            st.info("📊 No workflow active. Start a workflow to see visualizations.")
    
    def _render_logs_tab(self):
        """Render the logs tab with full activity history."""
        st.markdown("### 📝 Activity Logs")
        
        # Export button
        if st.button("💾 Export Logs", key="export_logs"):
            log_data = self.live_feed.export_feed()
            st.download_button(
                "Download JSON",
                data=log_data,
                file_name=f"workflow_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Full feed view
        self.live_feed.render("logs_tab", height=800)
    
    def _render_footer(self):
        """Render footer with metrics."""
        if st.session_state.workflow_ui_v2.get('workflow_state'):
            metrics = st.session_state.phase_executor_v2.get('metrics', {})
            
            # Don't use managed containers for the footer to avoid widget ID changes
            with st.container():
                st.divider()
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Phases Completed", metrics.get('phases_completed', 0))
                
                with col2:
                    st.metric("Phases Failed", metrics.get('phases_failed', 0))
                
                with col3:
                    st.metric("Artifacts Created", metrics.get('artifacts_created', 0))
                
                with col4:
                    total_duration = metrics.get('total_duration', 0)
                    st.metric("Total Duration", f"{total_duration:.1f}s")
                
                with col5:
                    if st.session_state.workflow_ui_v2.get('workflow_completed'):
                        st.metric("Status", "✅ Complete")
                    elif st.session_state.workflow_ui_v2.get('workflow_active'):
                        st.metric("Status", "⚡ Running")
                    else:
                        st.metric("Status", "⏸️ Idle")
    
    def _start_workflow(self, task_description: str, model: str, 
                       generate_tests: bool, execution_mode: str, phase_models: Dict[str, str] = None):
        """Start a new workflow."""
        # Create workflow state
        workflow_id = f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        workflow_state = WorkflowState(
            workflow_id=workflow_id,
            task_description=task_description,
            model=model,
            phase_models=phase_models or {},
            generate_tests=generate_tests,
            metadata=WorkflowMetadata(
                workflow_id=workflow_id,
                started_at=datetime.now().isoformat(),
                current_phase="initialization",
                status="running"
            )
        )
        
        # Update session state
        st.session_state.workflow_ui_v2['workflow_state'] = workflow_state
        st.session_state.workflow_ui_v2['workflow_active'] = True
        st.session_state.workflow_ui_v2['workflow_completed'] = False
        st.session_state.workflow_ui_v2['execution_mode'] = execution_mode
        st.session_state.workflow_ui_v2['start_time'] = datetime.now()
        st.session_state.workflow_ui_v2['artifacts'] = {}
        
        # Reset phase executor state
        st.session_state.phase_executor_v2 = {
            'phase_results': {},
            'current_phase': None,
            'phase_queue': [],
            'execution_paused': False,
            'artifacts_by_phase': {},
            'logs': [],
            'metrics': {
                'total_duration': 0,
                'phases_completed': 0,
                'phases_failed': 0,
                'artifacts_created': 0
            }
        }
        
        # Log start
        log_info(f"Starting workflow: {workflow_id}")
        log_info(f"Task: {task_description}")
        log_info(f"Model: {model}")
        log_info(f"Phase models: {phase_models}")
        log_info(f"Execution mode: {execution_mode}")
        
        # Switch to execution tab
        st.session_state.workflow_ui_v2['current_tab'] = 'execution'
        
        # Rerun to show workflow
        st.rerun()
    
    def _stop_workflow(self):
        """Stop the current workflow."""
        st.session_state.workflow_ui_v2['workflow_active'] = False
        st.session_state.workflow_ui_v2['end_time'] = datetime.now()
        st.session_state.phase_executor_v2['current_phase'] = None
        
        log_info("Workflow stopped by user")
        st.success("Workflow stopped")
    
    def _check_workflow_completion(self, workflow_state: WorkflowState, phases: List[str]):
        """Check if workflow is complete."""
        if st.session_state.workflow_ui_v2['workflow_completed']:
            return
        
        phase_results = st.session_state.phase_executor_v2.get('phase_results', {})
        
        # Check if all phases are complete or skipped
        all_done = all(
            phase in phase_results and 
            phase_results[phase].status in [PhaseStatus.COMPLETED, PhaseStatus.SKIPPED]
            for phase in phases
        )
        
        if all_done:
            st.session_state.workflow_ui_v2['workflow_completed'] = True
            st.session_state.workflow_ui_v2['workflow_active'] = False
            st.session_state.workflow_ui_v2['end_time'] = datetime.now()
            
            # Calculate total duration
            duration = (st.session_state.workflow_ui_v2['end_time'] - 
                       st.session_state.workflow_ui_v2['start_time']).total_seconds()
            
            # Log completion
            log_info(f"Workflow completed in {duration:.1f}s")
            
            # Show success message
            st.balloons()
            st.success(f"🎉 Workflow completed successfully in {duration:.1f}s!")
            
            # Show download option for artifacts
            if st.session_state.workflow_ui_v2['artifacts']:
                export_data = self.workflow_viewer.export_workflow_data(
                    workflow_state,
                    phase_results,
                    st.session_state.workflow_ui_v2['artifacts']
                )
                
                st.download_button(
                    "📥 Download Workflow Results",
                    data=export_data,
                    file_name=f"{workflow_state.workflow_id}_results.json",
                    mime="application/json",
                    key="download_results"
                )


def main():
    """Main entry point for the workflow UI."""
    # Note: Page config must be set in app_v2.py before importing this module
    
    # Create and render the UI
    ui = WorkflowUIV2(debug_mode=False)
    ui.render_content()


if __name__ == "__main__":
    # This should only be used for direct testing
    # Normally app_v2.py should be used as the entry point
    st.set_page_config(
        page_title="AgenTool Workflow Engine V2",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    main()
