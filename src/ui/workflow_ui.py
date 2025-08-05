"""
Main Streamlit UI for AgenTool Workflow.

This is the entry point for the workflow visualization and control interface.
Run with: streamlit run src/ui/workflow_ui.py
"""

import streamlit as st
import asyncio
from typing import Dict, Any, Optional
import json
from datetime import datetime
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from workflow_runner import WorkflowRunner, WorkflowState
from components.progress_tracker import ProgressTracker
from components.artifact_viewer import ArtifactViewer
from components.code_editor import CodeEditor
from components.metrics_dashboard import MetricsDashboard
from utils.formatting import format_timestamp, format_duration

# Page configuration
st.set_page_config(
    page_title="AgenTool Workflow",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
    }
    .workflow-phase {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .phase-active {
        background-color: #e3f2fd;
        border: 2px solid #2196f3;
    }
    .phase-completed {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
    }
    .phase-error {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'workflow_state' not in st.session_state:
    st.session_state.workflow_state = WorkflowState()

if 'runner' not in st.session_state:
    st.session_state.runner = None

if 'artifacts' not in st.session_state:
    st.session_state.artifacts = {}

if 'metrics' not in st.session_state:
    st.session_state.metrics = {
        'start_time': None,
        'end_time': None,
        'phase_durations': {},
        'token_usage': {},
        'errors': []
    }

# Title and header
st.title("🤖 AgenTool Workflow Generator")
st.markdown("Create production-ready AgenTools with AI-powered code generation")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    model_provider = st.selectbox(
        "Model Provider",
        ["OpenAI", "Anthropic", "Google", "Groq"],
        help="Select the AI model provider"
    )
    
    model_map = {
        "OpenAI": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        "Anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
        "Google": ["gemini-1.5-pro", "gemini-1.5-flash"],
        "Groq": ["mixtral-8x7b", "llama-3-70b"]
    }
    
    model = st.selectbox(
        "Model",
        model_map.get(model_provider, []),
        help="Select the specific model to use"
    )
    
    # Task input
    st.header("📝 Task Description")
    task_description = st.text_area(
        "What AgenTool do you want to create?",
        value=st.session_state.get('last_task', ''),
        height=150,
        placeholder="Example: Create a session management AgenTool that handles user sessions with TTL support..."
    )
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        debug_mode = st.checkbox("Debug Mode", value=False)
        auto_save = st.checkbox("Auto-save artifacts", value=True)
        stream_updates = st.checkbox("Stream updates", value=True)
        
    # Control buttons
    st.header("🎮 Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button(
            "▶️ Start Workflow",
            disabled=not task_description or st.session_state.workflow_state.is_running
        )
    
    with col2:
        stop_button = st.button(
            "⏹️ Stop Workflow",
            disabled=not st.session_state.workflow_state.is_running
        )
    
    if st.session_state.workflow_state.completed:
        export_button = st.button("💾 Export Results")

# Main content area
main_container = st.container()

# Initialize phase containers outside tabs
phases = ["Analyzer", "Specifier", "Crafter", "Evaluator"]
phase_containers = {}

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📊 Progress", "🗂️ Artifacts", "💻 Generated Code", "📈 Metrics"])

with tab1:
    progress_container = st.container()
    
    # Overall progress
    if st.session_state.workflow_state.is_running or st.session_state.workflow_state.completed:
        progress = st.session_state.workflow_state.get_progress()
        st.progress(progress, text=f"Overall Progress: {int(progress * 100)}%")
    
    # Phase status
    st.subheader("Workflow Phases")
    
    for phase in phases:
        with st.container():
            phase_containers[phase.lower()] = ProgressTracker(phase)

with tab2:
    artifact_viewer = ArtifactViewer()
    artifact_viewer.render(st.session_state.artifacts)

with tab3:
    code_editor = CodeEditor()
    
    if st.session_state.workflow_state.current_phase == "crafter" or st.session_state.workflow_state.completed:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📋 Specification")
            if 'specification' in st.session_state.artifacts:
                code_editor.render_specification(st.session_state.artifacts['specification'])
        
        with col2:
            st.subheader("💻 Implementation")
            if 'implementation' in st.session_state.artifacts:
                code_editor.render_code(st.session_state.artifacts['implementation'])

with tab4:
    metrics_dashboard = MetricsDashboard()
    metrics_dashboard.render(st.session_state.metrics)

# Handle workflow start
async def run_workflow():
    """Run the workflow asynchronously."""
    try:
        # Store task for later
        st.session_state.last_task = task_description
        
        # Initialize runner
        runner = WorkflowRunner(
            task=task_description,
            model=f"{model_provider.lower()}:{model}",
            debug=debug_mode,
            stream_updates=stream_updates
        )
        st.session_state.runner = runner
        
        # Set up progress callbacks
        def on_phase_start(phase: str):
            if phase in phase_containers:
                phase_containers[phase].start()
        
        def on_phase_complete(phase: str, duration: float):
            if phase in phase_containers:
                phase_containers[phase].complete(duration)
            st.session_state.metrics['phase_durations'][phase] = duration
        
        def on_phase_error(phase: str, error: str):
            if phase in phase_containers:
                phase_containers[phase].error(error)
            st.session_state.metrics['errors'].append({
                'phase': phase,
                'error': error,
                'timestamp': datetime.now()
            })
        
        def on_artifact_created(name: str, data: Any):
            st.session_state.artifacts[name] = data
            if auto_save:
                # Save artifact to file system
                pass
        
        def on_stream_update(text: str):
            # Handle streaming updates
            if stream_updates:
                # Update the appropriate container
                pass
        
        # Register callbacks
        runner.on_phase_start = on_phase_start
        runner.on_phase_complete = on_phase_complete
        runner.on_phase_error = on_phase_error
        runner.on_artifact_created = on_artifact_created
        runner.on_stream_update = on_stream_update
        
        # Start metrics
        st.session_state.metrics['start_time'] = datetime.now()
        
        # Run the workflow
        await runner.run()
        
        # Complete metrics
        st.session_state.metrics['end_time'] = datetime.now()
        
    except Exception as e:
        st.error(f"Workflow failed: {str(e)}")
        st.session_state.workflow_state.error = str(e)

# Handle button clicks
if start_button:
    st.session_state.workflow_state.start()
    # Run workflow in event loop
    with st.spinner("Running workflow..."):
        asyncio.run(run_workflow())

if stop_button:
    if st.session_state.runner:
        st.session_state.runner.stop()
    st.session_state.workflow_state.stop()

# Export functionality
if 'export_button' in locals() and export_button:
    # Create export package
    export_data = {
        'task': task_description,
        'model': f"{model_provider.lower()}:{model}",
        'timestamp': datetime.now().isoformat(),
        'artifacts': st.session_state.artifacts,
        'metrics': st.session_state.metrics
    }
    
    st.download_button(
        label="Download Export Package",
        data=json.dumps(export_data, indent=2),
        file_name=f"agentool_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

# Auto-refresh for running workflows
if st.session_state.workflow_state.is_running:
    # Only refresh if we haven't refreshed recently
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    current_time = time.time()
    if current_time - st.session_state.last_refresh > 2:  # Refresh every 2 seconds
        st.session_state.last_refresh = current_time
        st.rerun()