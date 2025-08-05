# -*- coding: utf-8 -*-
"""
AgenTool Workflow UI - Streamlit Application for Workflow Orchestration.

This is the main entry point for the workflow visualization and execution interface.
It orchestrates the workflow directly, embracing Streamlit's execution model.

Run with: streamlit run src/ui/workflow_ui.py
"""

import streamlit as st
import asyncio
import json
import uuid
import os
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import workflow components
from agents.workflow import WorkflowState, WorkflowMetadata
from agents.models import (
    AnalyzerOutput,
    SpecificationOutput,
    CodeOutput,
    ValidationOutput,
    TestAnalysisOutput,
    TestStubOutput,
    TestImplementationOutput
)
from agentool.core.injector import get_injector, AgenToolInjector

# Import UI components
from components.artifact_viewer import ArtifactViewer
from components.code_editor import CodeEditor
from components.phase_executor import PhaseExecutor, PhaseConfig, PhaseResult
from components.workflow_viewer import WorkflowViewer

# Page configuration
st.set_page_config(
    page_title="AgenTool Workflow Generator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .phase-container {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 2px solid transparent;
    }
    .phase-pending {
        background-color: #f0f2f6;
        border-color: #d0d2d6;
    }
    .phase-running {
        background-color: #e3f2fd;
        border-color: #2196f3;
        animation: pulse 2s infinite;
    }
    .phase-completed {
        background-color: #e8f5e9;
        border-color: #4caf50;
    }
    .phase-error {
        background-color: #ffebee;
        border-color: #f44336;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.8; }
        100% { opacity: 1; }
    }
    .artifact-ref {
        font-family: monospace;
        background-color: #f5f5f5;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)


@dataclass
class UIState:
    """Extended state for UI-specific needs."""
    workflow_state: Optional[WorkflowState] = None
    workflow_running: bool = False
    current_phase: str = "idle"
    phase_results: Dict[str, PhaseResult] = field(default_factory=dict)
    artifacts: Dict[str, List[str]] = field(default_factory=dict)
    error: Optional[str] = None
    injector: Optional[AgenToolInjector] = None


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'ui_state' not in st.session_state:
        st.session_state.ui_state = UIState()
    
    if 'task_description' not in st.session_state:
        st.session_state.task_description = ""
    
    if 'model' not in st.session_state:
        st.session_state.model = "openai:gpt-4o"
    
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False


def initialize_agentoolkits() -> AgenToolInjector:
    """Initialize all AgenToolkits and return the injector."""
    injector = get_injector()
    
    try:
        import agentoolkit
        
        # Initialize all create_* functions
        for name in dir(agentoolkit):
            if name.startswith('create_'):
                create_func = getattr(agentoolkit, name)
                try:
                    if name == 'create_templates_agent':
                        # Special handling for templates
                        templates_path = os.path.abspath(
                            os.path.join(os.path.dirname(__file__), '../templates')
                        )
                        agent = create_func(templates_dir=templates_path)
                    else:
                        agent = create_func()
                except Exception as e:
                    if st.session_state.debug_mode:
                        st.warning(f"Could not initialize {name}: {e}")
        
        # Initialize workflow agents
        from agentoolkit.workflows import (
            create_workflow_analyzer_agent,
            create_workflow_specifier_agent,
            create_workflow_crafter_agent,
            create_workflow_evaluator_agent,
            create_workflow_test_analyzer_agent,
            create_workflow_test_stubber_agent,
            create_workflow_test_crafter_agent
        )
        
        create_workflow_analyzer_agent()
        create_workflow_specifier_agent()
        create_workflow_crafter_agent()
        create_workflow_evaluator_agent()
        create_workflow_test_analyzer_agent()
        create_workflow_test_stubber_agent()
        create_workflow_test_crafter_agent()
        
    except Exception as e:
        st.error(f"Failed to initialize AgenToolkits: {e}")
        if st.session_state.debug_mode:
            st.exception(e)
    
    return injector


def render_sidebar():
    """Render the sidebar configuration."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Task input
        st.subheader("📝 Task Description")
        task_description = st.text_area(
            "What AgenTool do you want to create?",
            value=st.session_state.task_description,
            height=150,
            placeholder="Example: Create a session management AgenTool that handles user sessions with TTL support...",
            key="task_input"
        )
        st.session_state.task_description = task_description
        
        # Model selection
        st.subheader("🤖 Model Selection")
        model_provider = st.selectbox(
            "Provider",
            ["OpenAI", "Anthropic", "Google", "Groq"],
            index=0
        )
        
        model_map = {
            "OpenAI": ["gpt-4o", "o4-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            "Anthropic": ["claude-4-opus", "claude-4-sonnet", "claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
            "Google": ["gemini-2.5-pro", "gemini-1.5-pro", "gemini-1.5-flash"],
            "Groq": ["mixtral-8x7b", "llama-3-70b"]
        }
        
        model_name = st.selectbox(
            "Model",
            model_map.get(model_provider, []),
            index=0
        )
        
        st.session_state.model = f"{model_provider.lower()}:{model_name}"
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            st.session_state.debug_mode = st.checkbox(
                "Debug Mode",
                value=st.session_state.debug_mode,
                help="Show detailed logs and error traces"
            )
            
            generate_tests = st.checkbox(
                "Generate Tests",
                value=True,
                help="Generate comprehensive test suites for each AgenTool"
            )
        
        # Control buttons
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(
                "🚀 Start Workflow",
                type="primary",
                disabled=not task_description or st.session_state.ui_state.workflow_running,
                use_container_width=True
            ):
                start_workflow()
        
        with col2:
            if st.button(
                "⏹️ Stop",
                type="secondary",
                disabled=not st.session_state.ui_state.workflow_running,
                use_container_width=True
            ):
                stop_workflow()
        
        # Reset button
        if st.session_state.ui_state.workflow_state and not st.session_state.ui_state.workflow_running:
            if st.button("🔄 Reset", use_container_width=True):
                reset_workflow()
        
        # Workflow info
        if st.session_state.ui_state.workflow_state:
            st.divider()
            st.subheader("ℹ️ Workflow Info")
            st.text(f"ID: {st.session_state.ui_state.workflow_state.workflow_id[:8]}...")
            st.text(f"Status: {st.session_state.ui_state.current_phase}")
            
            if st.session_state.ui_state.workflow_state.metadata:
                meta = st.session_state.ui_state.workflow_state.metadata
                if meta.started_at:
                    st.text(f"Started: {meta.started_at[:19]}")
                if meta.total_duration_seconds:
                    st.text(f"Duration: {meta.total_duration_seconds:.1f}s")


def start_workflow():
    """Start the workflow execution."""
    # Initialize workflow state
    workflow_id = str(uuid.uuid4())
    st.session_state.ui_state.workflow_state = WorkflowState(
        task_description=st.session_state.task_description,
        model=st.session_state.model,
        generate_tests=True,
        workflow_id=workflow_id,
        metadata=WorkflowMetadata(
            workflow_id=workflow_id,
            started_at=datetime.now().isoformat(),
            current_phase="initialization",
            status="running"
        )
    )
    
    st.session_state.ui_state.workflow_running = True
    st.session_state.ui_state.current_phase = "initialization"
    st.session_state.ui_state.error = None
    
    # Initialize injector if needed
    if not st.session_state.ui_state.injector:
        st.session_state.ui_state.injector = initialize_agentoolkits()
    
    st.rerun()


def stop_workflow():
    """Stop the workflow execution."""
    st.session_state.ui_state.workflow_running = False
    if st.session_state.ui_state.workflow_state and st.session_state.ui_state.workflow_state.metadata:
        st.session_state.ui_state.workflow_state.metadata.status = "stopped"
    st.rerun()


def reset_workflow():
    """Reset the workflow to initial state."""
    st.session_state.ui_state = UIState()
    st.session_state.task_description = ""
    st.rerun()


def render_workflow_phases():
    """Render the workflow phase execution interface."""
    if not st.session_state.ui_state.workflow_state:
        st.info("ℹ️ Configure your task and click 'Start Workflow' to begin")
        return
    
    # Create phase executor
    executor = PhaseExecutor(
        injector=st.session_state.ui_state.injector,
        debug_mode=st.session_state.debug_mode
    )
    
    # Define workflow phases
    phases = [
        ("analyzer", "Analyzer", "🔍 Analyzing task requirements..."),
        ("specification", "Specification", "📋 Creating specifications..."),
        ("crafter", "Crafter", "💻 Generating implementation..."),
        ("evaluator", "Evaluator", "✅ Validating code..."),
    ]
    
    if st.session_state.ui_state.workflow_state.generate_tests:
        phases.extend([
            ("test_analyzer", "Test Analyzer", "🧪 Analyzing test requirements..."),
            ("test_stubber", "Test Stubber", "🏗️ Creating test structure..."),
            ("test_crafter", "Test Crafter", "🔨 Implementing tests..."),
        ])
    
    # Execute phases progressively
    for phase_key, phase_name, phase_desc in phases:
        phase_status = get_phase_status(phase_key)
        
        # Create phase container with appropriate styling
        with st.container():
            st.markdown(
                f'<div class="phase-container phase-{phase_status}">',
                unsafe_allow_html=True
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(f"{phase_name}")
                if phase_status == "running":
                    st.write(phase_desc)
            
            with col2:
                if phase_status == "completed":
                    st.success("✅ Complete")
                elif phase_status == "running":
                    st.info("⏳ Running...")
                elif phase_status == "error":
                    st.error("❌ Failed")
                elif phase_status == "pending":
                    st.text("⏳ Pending")
            
            # Execute phase if it's the current one
            if (st.session_state.ui_state.workflow_running and 
                phase_status == "pending" and 
                is_previous_phase_complete(phase_key, phases)):
                
                st.session_state.ui_state.current_phase = phase_key
                
                # Execute the phase
                try:
                    with st.spinner(phase_desc):
                        result = executor.execute_phase(
                            phase_key,
                            st.session_state.ui_state.workflow_state
                        )
                        
                        # Store result
                        st.session_state.ui_state.phase_results[phase_key] = result
                        
                        # Update workflow state
                        setattr(
                            st.session_state.ui_state.workflow_state,
                            f"{phase_key}_completed",
                            True
                        )
                        
                        # Capture artifacts
                        if result.artifacts:
                            st.session_state.ui_state.artifacts[phase_key] = result.artifacts
                        
                        st.rerun()
                        
                except Exception as e:
                    st.session_state.ui_state.error = str(e)
                    st.session_state.ui_state.workflow_running = False
                    if st.session_state.debug_mode:
                        st.exception(e)
                    else:
                        st.error(f"Phase failed: {e}")
            
            # Show phase details if completed
            if phase_status == "completed" and phase_key in st.session_state.ui_state.phase_results:
                result = st.session_state.ui_state.phase_results[phase_key]
                
                with st.expander("View Details", expanded=False):
                    # Show execution time
                    st.metric("Execution Time", f"{result.duration:.2f}s")
                    
                    # Show artifacts
                    if result.artifacts:
                        st.write("**Artifacts Created:**")
                        for artifact in result.artifacts[:5]:  # Show first 5
                            st.code(artifact, language=None)
                        if len(result.artifacts) > 5:
                            st.text(f"... and {len(result.artifacts) - 5} more")
                    
                    # Show summary
                    if result.summary:
                        st.write("**Summary:**")
                        st.json(result.summary)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Check if workflow is complete
    all_complete = all(
        getattr(st.session_state.ui_state.workflow_state, f"{phase[0]}_completed", False)
        for phase in phases
    )
    
    if all_complete:
        st.session_state.ui_state.workflow_running = False
        if st.session_state.ui_state.workflow_state.metadata:
            st.session_state.ui_state.workflow_state.metadata.status = "completed"
            st.session_state.ui_state.workflow_state.metadata.completed_at = datetime.now().isoformat()


def get_phase_status(phase_key: str) -> str:
    """Get the status of a workflow phase."""
    if not st.session_state.ui_state.workflow_state:
        return "pending"
    
    if st.session_state.ui_state.error and st.session_state.ui_state.current_phase == phase_key:
        return "error"
    
    if getattr(st.session_state.ui_state.workflow_state, f"{phase_key}_completed", False):
        return "completed"
    
    if st.session_state.ui_state.current_phase == phase_key and st.session_state.ui_state.workflow_running:
        return "running"
    
    return "pending"


def is_previous_phase_complete(phase_key: str, phases: List[tuple]) -> bool:
    """Check if all previous phases are complete."""
    for i, (key, _, _) in enumerate(phases):
        if key == phase_key:
            # Check all previous phases
            for j in range(i):
                prev_key = phases[j][0]
                if not getattr(st.session_state.ui_state.workflow_state, f"{prev_key}_completed", False):
                    return False
            return True
    return False


def render_results_section():
    """Render the results section with tabs."""
    if not st.session_state.ui_state.workflow_state:
        return
    
    st.divider()
    
    # Check if tests were generated
    test_phases = ["test_analyzer", "test_stubber", "test_crafter"]
    has_tests = any(phase in st.session_state.ui_state.phase_results for phase in test_phases)
    
    # Create tabs dynamically based on whether tests were generated
    if has_tests:
        tab_names = ["📊 Workflow Graph", "🗄️ Artifacts", "💻 Generated Code", "🧪 Test Code", "📈 Summary"]
        tabs = st.tabs(tab_names)
        tab1, tab2, tab3, tab4, tab5 = tabs
    else:
        tab_names = ["📊 Workflow Graph", "🗄️ Artifacts", "💻 Generated Code", "📈 Summary"]
        tabs = st.tabs(tab_names)
        tab1, tab2, tab3, tab5 = tabs
        tab4 = None  # No test tab
    
    with tab1:
        # Workflow visualization
        if st.session_state.ui_state.artifacts:
            viewer = WorkflowViewer()
            viewer.render(
                st.session_state.ui_state.workflow_state,
                st.session_state.ui_state.phase_results,
                st.session_state.ui_state.artifacts
            )
    
    with tab2:
        # Artifact viewer
        if st.session_state.ui_state.artifacts:
            artifact_viewer = ArtifactViewer()
            # Flatten artifacts for viewer
            all_artifacts = {}
            for phase, artifacts in st.session_state.ui_state.artifacts.items():
                for artifact in artifacts:
                    # Extract key from storage reference
                    if artifact.startswith("storage_kv:"):
                        key = artifact.replace("storage_kv:", "")
                        all_artifacts[key] = {"type": "storage_kv", "phase": phase}
                    elif artifact.startswith("storage_fs:"):
                        key = artifact.replace("storage_fs:", "")
                        all_artifacts[key] = {"type": "storage_fs", "phase": phase}
            
            artifact_viewer.render(all_artifacts)
    
    with tab3:
        # Code display - handle multiple generated tools
        if "crafter" in st.session_state.ui_state.phase_results:
            code_editor = CodeEditor()
            result = st.session_state.ui_state.phase_results["crafter"]
            
            if result.data:
                # Check if we have multiple implementations
                if "implementations" in result.data:
                    # Multiple tools generated
                    st.subheader(f"💻 Generated Code ({len(result.data['implementations'])} tools)")
                    
                    # Create tabs for each tool
                    tool_names = [impl.get('tool_name', f'Tool {i+1}') for i, impl in enumerate(result.data['implementations'])]
                    tool_tabs = st.tabs(tool_names)
                    
                    for i, (tab, impl_name) in enumerate(zip(tool_tabs, tool_names)):
                        with tab:
                            # Fetch the actual code from storage
                            impl_key = f"workflow/{st.session_state.ui_state.workflow_state.workflow_id}/implementations/{impl_name}"
                            
                            # Get code from artifacts or fetch from storage
                            code_artifact = None
                            for phase_artifacts in st.session_state.ui_state.artifacts.values():
                                for artifact in phase_artifacts:
                                    if impl_name in artifact and 'implementations' in artifact:
                                        code_artifact = artifact
                                        break
                            
                            if code_artifact:
                                # Fetch from storage_kv
                                try:
                                    import asyncio
                                    injector = st.session_state.ui_state.injector
                                    result_data = asyncio.run(injector.run('storage_kv', {
                                        'operation': 'get',
                                        'key': impl_key
                                    }))
                                    
                                    if hasattr(result_data, 'output'):
                                        data = json.loads(result_data.output)
                                    else:
                                        data = result_data.data if hasattr(result_data, 'data') else result_data
                                    
                                    if data.get('data', {}).get('exists', False):
                                        code_data = json.loads(data['data']['value'])
                                        if 'code' in code_data:
                                            code_editor.render_code(code_data['code'], key_suffix=f"tool_{impl_name}")
                                            
                                            # Download button for this tool
                                            st.download_button(
                                                f"💾 Download {impl_name}.py",
                                                data=code_data['code'],
                                                file_name=f"{impl_name}.py",
                                                mime="text/plain",
                                                key=f"download_{impl_name}"
                                            )
                                        else:
                                            st.warning(f"No code found for {impl_name}")
                                    else:
                                        st.error(f"Could not fetch code for {impl_name}")
                                except Exception as e:
                                    st.error(f"Error loading code: {e}")
                            else:
                                st.warning(f"No artifact found for {impl_name}")
                    
                    # Download all button
                    if st.button("📦 Download All Code", key="download_all_code"):
                        st.info("Feature coming soon: Download all generated code as a zip file")
                        
                elif "code" in result.data:
                    # Single tool generated (backward compatibility)
                    st.subheader("💻 Generated Code")
                    code_editor.render_code(result.data["code"], key_suffix="single_tool")
                    
                    # Download button
                    st.download_button(
                        "💾 Download Code",
                        data=result.data["code"],
                        file_name=result.data.get("file_path", "generated_agentool.py"),
                        mime="text/plain"
                    )
                else:
                    st.info("No code generated yet")
        else:
            st.info("Code will be displayed here once the Crafter phase completes")
    
    # Test code tab (if tests were generated)
    if tab4 is not None:
        with tab4:
            render_test_code_section()
    
    # Summary tab
    with tab5:
        # Summary and metrics
        if st.session_state.ui_state.workflow_state.metadata:
            render_workflow_summary()


def render_test_code_section():
    """Render the test code section."""
    if not st.session_state.ui_state.phase_results:
        st.info("Test code will be displayed here once test generation completes")
        return
    
    # Check if we have test results
    test_phases = ["test_analyzer", "test_stubber", "test_crafter"]
    has_tests = any(phase in st.session_state.ui_state.phase_results for phase in test_phases)
    
    if not has_tests:
        st.info("No test code generated yet")
        return
    
    st.subheader("🧪 Generated Test Code")
    
    # Get test artifacts
    test_files = {}
    
    # Look for test files in artifacts
    for phase, artifacts in st.session_state.ui_state.artifacts.items():
        if "test" in phase:
            for artifact in artifacts:
                if "storage_fs:" in artifact and "/tests/" in artifact and artifact.endswith(".py"):
                    # Extract filename
                    filename = artifact.split("/")[-1]
                    # Fetch content
                    try:
                        import asyncio
                        injector = st.session_state.ui_state.injector
                        path = artifact.replace("storage_fs:", "")
                        
                        result = asyncio.run(injector.run('storage_fs', {
                            'operation': 'read',
                            'path': path
                        }))
                        
                        if hasattr(result, 'output'):
                            data = json.loads(result.output)
                        else:
                            data = result.data if hasattr(result, 'data') else result
                        
                        if data.get('data', {}).get('content'):
                            test_files[filename] = data['data']['content']
                    except Exception as e:
                        st.error(f"Error loading test file {filename}: {e}")
    
    if test_files:
        # Use CodeEditor to display test files
        code_editor = CodeEditor()
        code_editor.create_file_viewer(test_files, show_metrics=True)
    else:
        st.warning("Test files were generated but could not be loaded")


def render_workflow_summary():
    """Render workflow execution summary."""
    meta = st.session_state.ui_state.workflow_state.metadata
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Duration", f"{meta.total_duration_seconds or 0:.1f}s")
        st.metric("Phases Completed", 
                  sum(1 for _ in st.session_state.ui_state.phase_results.items()))
    
    with col2:
        st.metric("Artifacts Created", 
                  sum(len(arts) for arts in st.session_state.ui_state.artifacts.values()))
        st.metric("Status", meta.status.title())
    
    with col3:
        if "evaluator" in st.session_state.ui_state.phase_results:
            result = st.session_state.ui_state.phase_results["evaluator"]
            if result.data:
                st.metric("Ready for Deployment", 
                         "✅ Yes" if result.data.get("ready_for_deployment") else "❌ No")
                st.metric("Issues Found", result.data.get("issues_count", 0))
    
    # Phase durations
    if meta.phase_durations:
        st.subheader("Phase Durations")
        for phase, duration in meta.phase_durations.items():
            st.text(f"{phase.title()}: {duration:.2f}s")
    
    # Export results
    if st.button("📊 Export Workflow Results"):
        export_data = {
            "workflow_id": st.session_state.ui_state.workflow_state.workflow_id,
            "task": st.session_state.ui_state.workflow_state.task_description,
            "model": st.session_state.ui_state.workflow_state.model,
            "metadata": meta.model_dump() if meta else None,
            "artifacts": st.session_state.ui_state.artifacts,
            "results": {
                phase: result.model_dump() if hasattr(result, 'model_dump') else result.__dict__
                for phase, result in st.session_state.ui_state.phase_results.items()
            }
        }
        
        st.download_button(
            "Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name=f"workflow_{st.session_state.ui_state.workflow_state.workflow_id}.json",
            mime="application/json"
        )


def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()
    
    # Title and description
    st.title("🚀 AgenTool Workflow Generator")
    st.markdown("Create production-ready AgenTools with AI-powered code generation")
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    render_workflow_phases()
    
    # Results section
    render_results_section()
    
    # Debug information
    if st.session_state.debug_mode:
        with st.expander("🐛 Debug Information"):
            st.json({
                "ui_state": {
                    "workflow_running": st.session_state.ui_state.workflow_running,
                    "current_phase": st.session_state.ui_state.current_phase,
                    "phases_completed": [
                        k for k in dir(st.session_state.ui_state.workflow_state)
                        if k.endswith("_completed") and getattr(st.session_state.ui_state.workflow_state, k)
                    ] if st.session_state.ui_state.workflow_state else [],
                    "artifacts_count": sum(len(arts) for arts in st.session_state.ui_state.artifacts.values()),
                    "has_error": bool(st.session_state.ui_state.error)
                }
            })


if __name__ == "__main__":
    main()