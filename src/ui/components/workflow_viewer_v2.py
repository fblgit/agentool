# -*- coding: utf-8 -*-
"""
Workflow Viewer V2 - Interactive workflow visualization with real-time updates.

This component provides modern, interactive visualizations of workflow execution
using Streamlit Elements for draggable cards and live progress tracking.
"""

import streamlit as st
import json
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
from dataclasses import dataclass
import graphlib
try:
    import streamlit_mermaid as stmd
    MERMAID_AVAILABLE = True
except ImportError:
    MERMAID_AVAILABLE = False

try:
    from streamlit_elements import elements, mui, dashboard, nivo
    ELEMENTS_AVAILABLE = True
except ImportError:
    ELEMENTS_AVAILABLE = False

from agents.workflow import WorkflowState
from .phase_executor_v2 import PhaseResult, PhaseStatus
from .container_manager import get_container_manager
from .theme_manager import get_theme_manager


@dataclass
class PhaseNode:
    """Represents a phase node in the workflow graph."""
    id: str
    name: str
    icon: str
    status: PhaseStatus
    dependencies: List[str]
    position: Tuple[int, int]
    duration: Optional[float] = None
    progress: float = 0.0
    artifacts_count: int = 0


class WorkflowViewerV2:
    """
    Interactive workflow viewer with real-time updates and modern visualizations.
    
    This component provides multiple view modes for workflow visualization
    including graph, timeline, and dashboard views.
    """
    
    def __init__(self):
        """Initialize the workflow viewer."""
        self.container_manager = get_container_manager()
        self.theme_manager = get_theme_manager()
        
        # Phase configurations
        self.phase_info = {
            'analyzer': {
                'name': 'Analyzer',
                'icon': '🔍',
                'color': '#4CAF50',
                'dependencies': [],
                'position': (0, 0)
            },
            'specification': {
                'name': 'Specification',
                'icon': '📋',
                'color': '#2196F3',
                'dependencies': ['analyzer'],
                'position': (1, 0)
            },
            'crafter': {
                'name': 'Crafter',
                'icon': '💻',
                'color': '#FF9800',
                'dependencies': ['specification'],
                'position': (2, 0)
            },
            'evaluator': {
                'name': 'Evaluator',
                'icon': '✅',
                'color': '#9C27B0',
                'dependencies': ['crafter'],
                'position': (3, 0)
            },
            'test_analyzer': {
                'name': 'Test Analyzer',
                'icon': '🧪',
                'color': '#00BCD4',
                'dependencies': ['specification'],
                'position': (1, 1)
            },
            'test_stubber': {
                'name': 'Test Stubber',
                'icon': '🏗️',
                'color': '#CDDC39',
                'dependencies': ['test_analyzer'],
                'position': (2, 1)
            },
            'test_crafter': {
                'name': 'Test Crafter',
                'icon': '🔨',
                'color': '#795548',
                'dependencies': ['test_stubber'],
                'position': (3, 1)
            }
        }
        
        # Session state is already initialized in app_v2.py
    
    def render(self, workflow_state: WorkflowState,
              phase_results: Dict[str, PhaseResult],
              artifacts: Dict[str, List[str]],
              container_key: str = "workflow_viewer"):
        """
        Render the workflow viewer.
        
        Args:
            workflow_state: Current workflow state
            phase_results: Results from completed phases
            artifacts: Artifacts created by phases
            container_key: Key for the main container
        """
        # Apply theme
        self.theme_manager.apply_theme()
        
        # Don't use managed containers for main UI to avoid widget ID changes
        with st.container():
            # Header with view controls
            self._render_header()
            
            # Render based on view mode
            view_mode = st.session_state.workflow_viewer_v2['view_mode']
            
            if view_mode == 'graph':
                self._render_graph_view(workflow_state, phase_results, artifacts)
            elif view_mode == 'timeline':
                self._render_timeline_view(workflow_state, phase_results, artifacts)
            elif view_mode == 'dashboard':
                self._render_dashboard_view(workflow_state, phase_results, artifacts)
            else:  # metrics
                self._render_metrics_view(workflow_state, phase_results, artifacts)
    
    def _render_header(self):
        """Render the header with view mode selector and controls."""
        col1, col2, col3, col4 = st.columns([2, 3, 2, 2])
        
        with col1:
            st.markdown("### 📊 Workflow Visualization")
        
        with col2:
            # View mode selector
            view_modes = {
                'graph': '🔀 Graph',
                'timeline': '📅 Timeline',
                'dashboard': '📊 Dashboard',
                'metrics': '📈 Metrics'
            }
            
            selected = st.radio(
                "View Mode",
                list(view_modes.keys()),
                index=list(view_modes.keys()).index(st.session_state.workflow_viewer_v2['view_mode']),
                format_func=lambda x: view_modes[x],
                horizontal=True,
                key="workflow_view_mode_selector",
                label_visibility="collapsed",
                on_change=lambda: st.session_state.workflow_viewer_v2.update({'view_mode': st.session_state.workflow_view_mode_selector})
            )
        
        with col3:
            # Toggle options
            col3_1, col3_2 = st.columns(2)
            with col3_1:
                st.session_state.workflow_viewer_v2['show_dependencies'] = st.checkbox(
                    "Dependencies",
                    value=st.session_state.workflow_viewer_v2['show_dependencies'],
                    key="show_deps"
                )
            with col3_2:
                st.session_state.workflow_viewer_v2['show_artifacts'] = st.checkbox(
                    "Artifacts",
                    value=st.session_state.workflow_viewer_v2['show_artifacts'],
                    key="show_arts"
                )
        
        with col4:
            # Auto-refresh toggle
            st.session_state.workflow_viewer_v2['auto_refresh'] = st.checkbox(
                "🔄 Auto-refresh",
                value=st.session_state.workflow_viewer_v2['auto_refresh'],
                key="auto_refresh_workflow"
            )
    
    def _render_graph_view(self, workflow_state: WorkflowState,
                          phase_results: Dict[str, PhaseResult],
                          artifacts: Dict[str, List[str]]):
        """Render interactive graph view of the workflow."""
        container = st.container(height=600, border=True)
        
        with container:
            if ELEMENTS_AVAILABLE:
                self._render_interactive_graph(workflow_state, phase_results, artifacts)
            else:
                self._render_mermaid_graph(workflow_state, phase_results)
    
    def _render_interactive_graph(self, workflow_state: WorkflowState,
                                 phase_results: Dict[str, PhaseResult],
                                 artifacts: Dict[str, List[str]]):
        """Render interactive graph using Streamlit Elements."""
        with elements("workflow_graph"):
            # Create dashboard layout for phases
            layout = []
            nodes_data = []
            
            # Determine which phases to include
            phases_to_include = list(self.phase_info.keys())
            if not workflow_state.generate_tests:
                phases_to_include = [p for p in phases_to_include if not p.startswith('test_')]
            
            # Create phase nodes
            for phase_key in phases_to_include:
                info = self.phase_info[phase_key]
                result = phase_results.get(phase_key)
                
                # Determine status
                if result:
                    status = result.status
                    duration = result.duration
                    progress = result.progress
                    artifacts_count = len(artifacts.get(phase_key, []))
                else:
                    status = PhaseStatus.PENDING
                    duration = None
                    progress = 0.0
                    artifacts_count = 0
                
                # Create node
                node = PhaseNode(
                    id=phase_key,
                    name=info['name'],
                    icon=info['icon'],
                    status=status,
                    dependencies=info['dependencies'],
                    position=info['position'],
                    duration=duration,
                    progress=progress,
                    artifacts_count=artifacts_count
                )
                
                nodes_data.append(node)
                
                # Add to dashboard layout
                x, y = info['position']
                layout.append(
                    dashboard.Item(
                        phase_key,
                        x * 3,  # Scale position
                        y * 3,
                        2,  # Width
                        2,  # Height
                        isDraggable=True,
                        isResizable=False
                    )
                )
            
            # Handle layout changes
            def handle_layout_change(updated_layout):
                # Store updated positions
                for item in updated_layout:
                    phase_key = item['i']
                    if phase_key in self.phase_info:
                        self.phase_info[phase_key]['position'] = (
                            item['x'] // 3,
                            item['y'] // 3
                        )
            
            # Render dashboard with phase cards
            with dashboard.Grid(layout, onLayoutChange=handle_layout_change):
                for node in nodes_data:
                    self._render_phase_card_element(node, phase_results.get(node.id))
    
    def _render_phase_card_element(self, node: PhaseNode, result: Optional[PhaseResult]):
        """Render a phase card using Material UI."""
        # Determine colors based on status
        status_colors = {
            PhaseStatus.PENDING: "#9E9E9E",
            PhaseStatus.RUNNING: "#2196F3",
            PhaseStatus.COMPLETED: "#4CAF50",
            PhaseStatus.FAILED: "#F44336",
            PhaseStatus.SKIPPED: "#FFC107"
        }
        
        bg_color = status_colors.get(node.status, "#9E9E9E")
        
        with mui.Card(
            key=node.id,
            sx={
                "backgroundColor": bg_color,
                "color": "white",
                "height": "100%",
                "display": "flex",
                "flexDirection": "column",
                "padding": 2,
                "cursor": "pointer",
                "transition": "all 0.3s",
                "&:hover": {
                    "transform": "scale(1.05)",
                    "boxShadow": 3
                }
            }
        ):
            # Header
            mui.Typography(f"{node.icon} {node.name}", variant="h6", gutterBottom=True)
            
            # Status
            mui.Typography(f"Status: {node.status.value}", variant="body2")
            
            # Progress bar for running phases
            if node.status == PhaseStatus.RUNNING:
                mui.LinearProgress(
                    variant="determinate",
                    value=node.progress * 100,
                    sx={"marginY": 1}
                )
            
            # Metrics
            if node.duration:
                mui.Typography(f"Duration: {node.duration:.1f}s", variant="body2")
            
            if node.artifacts_count > 0:
                mui.Typography(f"Artifacts: {node.artifacts_count}", variant="body2")
            
            # Dependencies
            if st.session_state.workflow_viewer_v2['show_dependencies'] and node.dependencies:
                mui.Typography(
                    f"Depends on: {', '.join(node.dependencies)}",
                    variant="caption",
                    sx={"marginTop": 1}
                )
    
    def _render_mermaid_graph(self, workflow_state: WorkflowState,
                             phase_results: Dict[str, PhaseResult]):
        """Render workflow as Mermaid diagram (fallback)."""
        st.info("Install streamlit-elements for interactive graph. Using Mermaid diagram as fallback.")
        
        # Build Mermaid diagram
        mermaid_code = self._generate_mermaid_diagram(workflow_state, phase_results)
        
        if MERMAID_AVAILABLE:
            stmd.st_mermaid(mermaid_code)
        else:
            st.code(mermaid_code, language="mermaid")
            st.info("Install streamlit-mermaid for rendered diagrams: `pip install streamlit-mermaid`")
    
    def _generate_mermaid_diagram(self, workflow_state: WorkflowState,
                                  phase_results: Dict[str, PhaseResult]) -> str:
        """Generate Mermaid diagram code."""
        lines = ["graph TD"]
        
        # Determine phases to include
        phases_to_include = list(self.phase_info.keys())
        if not workflow_state.generate_tests:
            phases_to_include = [p for p in phases_to_include if not p.startswith('test_')]
        
        # Add nodes
        for phase_key in phases_to_include:
            info = self.phase_info[phase_key]
            result = phase_results.get(phase_key)
            
            # Determine status and styling
            if result:
                if result.status == PhaseStatus.COMPLETED:
                    style = f"style {phase_key} fill:#4CAF50,stroke:#2E7D32,color:#fff"
                elif result.status == PhaseStatus.RUNNING:
                    style = f"style {phase_key} fill:#2196F3,stroke:#1565C0,color:#fff"
                elif result.status == PhaseStatus.FAILED:
                    style = f"style {phase_key} fill:#F44336,stroke:#C62828,color:#fff"
                else:
                    style = f"style {phase_key} fill:#E0E0E0,stroke:#9E9E9E"
            else:
                style = f"style {phase_key} fill:#E0E0E0,stroke:#9E9E9E"
            
            # Node label
            label = f"{info['icon']} {info['name']}"
            if result and result.duration:
                label += f"<br/>({result.duration:.1f}s)"
            
            lines.append(f"    {phase_key}[\"{label}\"]")
            lines.append(f"    {style}")
        
        # Add edges
        if st.session_state.workflow_viewer_v2['show_dependencies']:
            for phase_key in phases_to_include:
                info = self.phase_info[phase_key]
                for dep in info['dependencies']:
                    if dep in phases_to_include:
                        lines.append(f"    {dep} --> {phase_key}")
        
        return "\n".join(lines)
    
    def _render_timeline_view(self, workflow_state: WorkflowState,
                             phase_results: Dict[str, PhaseResult],
                             artifacts: Dict[str, List[str]]):
        """Render timeline view of workflow execution."""
        container = st.container(height=500, border=True)
        
        with container:
            st.markdown("#### ⏱️ Execution Timeline")
            
            # Calculate timeline data
            timeline_events = []
            
            for phase_key, result in phase_results.items():
                if phase_key in self.phase_info:
                    info = self.phase_info[phase_key]
                    
                    # Start event
                    timeline_events.append({
                        'time': result.started_at,
                        'phase': phase_key,
                        'event': 'start',
                        'icon': info['icon'],
                        'name': info['name'],
                        'status': result.status
                    })
                    
                    # End event
                    if result.completed_at:
                        timeline_events.append({
                            'time': result.completed_at,
                            'phase': phase_key,
                            'event': 'end',
                            'icon': info['icon'],
                            'name': info['name'],
                            'status': result.status,
                            'duration': result.duration
                        })
            
            # Sort by time
            timeline_events.sort(key=lambda x: x['time'])
            
            # Render timeline
            if timeline_events:
                for event in timeline_events:
                    col1, col2, col3 = st.columns([2, 1, 5])
                    
                    with col1:
                        st.caption(event['time'].strftime('%H:%M:%S.%f')[:-3])
                    
                    with col2:
                        if event['event'] == 'start':
                            st.markdown(f"{event['icon']} ▶️")
                        else:
                            if event['status'] == PhaseStatus.COMPLETED:
                                st.markdown(f"{event['icon']} ✅")
                            elif event['status'] == PhaseStatus.FAILED:
                                st.markdown(f"{event['icon']} ❌")
                            else:
                                st.markdown(f"{event['icon']} ⏹️")
                    
                    with col3:
                        if event['event'] == 'start':
                            st.markdown(f"**Started {event['name']}**")
                        else:
                            duration_text = f" in {event.get('duration', 0):.1f}s" if 'duration' in event else ""
                            st.markdown(f"**Completed {event['name']}**{duration_text}")
                    
                    # Show artifacts if enabled
                    if (st.session_state.workflow_viewer_v2['show_artifacts'] and 
                        event['event'] == 'end' and 
                        event['phase'] in artifacts):
                        
                        phase_artifacts = artifacts[event['phase']]
                        if phase_artifacts:
                            with st.expander(f"📦 {len(phase_artifacts)} artifacts"):
                                for artifact in phase_artifacts[:5]:
                                    st.caption(f"• {artifact.split('/')[-1]}")
                                if len(phase_artifacts) > 5:
                                    st.caption(f"... and {len(phase_artifacts) - 5} more")
            else:
                st.info("No timeline events yet. Events will appear as phases execute.")
    
    def _render_dashboard_view(self, workflow_state: WorkflowState,
                              phase_results: Dict[str, PhaseResult],
                              artifacts: Dict[str, List[str]]):
        """Render dashboard view with key metrics and charts."""
        # Create metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate metrics
        total_phases = len([p for p in self.phase_info.keys() 
                           if not p.startswith('test_') or workflow_state.generate_tests])
        completed_phases = len([r for r in phase_results.values() 
                               if r.status == PhaseStatus.COMPLETED])
        failed_phases = len([r for r in phase_results.values() 
                           if r.status == PhaseStatus.FAILED])
        total_artifacts = sum(len(arts) for arts in artifacts.values())
        
        with col1:
            st.metric(
                "Progress",
                f"{completed_phases}/{total_phases}",
                delta=f"{(completed_phases/total_phases*100):.0f}%" if total_phases > 0 else "0%"
            )
        
        with col2:
            st.metric("Completed", completed_phases, delta=None if completed_phases == 0 else "✅")
        
        with col3:
            st.metric("Failed", failed_phases, delta=None if failed_phases == 0 else "⚠️")
        
        with col4:
            st.metric("Artifacts", total_artifacts)
        
        st.divider()
        
        # Charts row
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            self._render_phase_duration_chart(phase_results)
        
        with chart_col2:
            self._render_artifact_distribution_chart(artifacts)
    
    def _render_phase_duration_chart(self, phase_results: Dict[str, PhaseResult]):
        """Render phase duration bar chart."""
        st.markdown("#### ⏱️ Phase Durations")
        
        if phase_results:
            # Prepare data
            chart_data = []
            for phase_key, result in phase_results.items():
                if result.duration and phase_key in self.phase_info:
                    chart_data.append({
                        'Phase': self.phase_info[phase_key]['name'],
                        'Duration (s)': result.duration
                    })
            
            if chart_data:
                import pandas as pd
                df = pd.DataFrame(chart_data)
                st.bar_chart(df.set_index('Phase'))
            else:
                st.info("No duration data available yet")
        else:
            st.info("No phases completed yet")
    
    def _render_artifact_distribution_chart(self, artifacts: Dict[str, List[str]]):
        """Render artifact distribution pie chart."""
        st.markdown("#### 📦 Artifact Distribution")
        
        if artifacts:
            # Prepare data
            distribution = {}
            for phase, arts in artifacts.items():
                if phase in self.phase_info and arts:
                    distribution[self.phase_info[phase]['name']] = len(arts)
            
            if distribution:
                # Simple display as metrics since pie charts need additional libraries
                for name, count in distribution.items():
                    st.metric(name, count)
            else:
                st.info("No artifacts created yet")
        else:
            st.info("No artifacts created yet")
    
    def _render_metrics_view(self, workflow_state: WorkflowState,
                            phase_results: Dict[str, PhaseResult],
                            artifacts: Dict[str, List[str]]):
        """Render detailed metrics view."""
        container = st.container()
        
        with container:
            st.markdown("#### 📊 Workflow Metrics")
            
            # Overall metrics
            st.markdown("##### Overall Progress")
            
            # Calculate overall metrics
            total_duration = sum(r.duration for r in phase_results.values() if r.duration)
            avg_duration = total_duration / len(phase_results) if phase_results else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Duration", f"{total_duration:.1f}s")
            with col2:
                st.metric("Average Phase Duration", f"{avg_duration:.1f}s")
            with col3:
                success_rate = (
                    len([r for r in phase_results.values() if r.status == PhaseStatus.COMPLETED]) /
                    len(phase_results) * 100
                ) if phase_results else 0
                st.metric("Success Rate", f"{success_rate:.0f}%")
            
            st.divider()
            
            # Phase-specific metrics
            st.markdown("##### Phase Details")
            
            for phase_key, result in phase_results.items():
                if phase_key not in self.phase_info:
                    continue
                
                info = self.phase_info[phase_key]
                
                with st.expander(f"{info['icon']} {info['name']}", expanded=False):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        status_icon = {
                            PhaseStatus.COMPLETED: "✅",
                            PhaseStatus.FAILED: "❌",
                            PhaseStatus.RUNNING: "⚡",
                            PhaseStatus.SKIPPED: "⏭️"
                        }.get(result.status, "⏳")
                        st.metric("Status", f"{status_icon} {result.status.value}")
                    
                    with col2:
                        st.metric("Duration", f"{result.duration:.2f}s" if result.duration else "N/A")
                    
                    with col3:
                        phase_artifacts = artifacts.get(phase_key, [])
                        st.metric("Artifacts", len(phase_artifacts))
                    
                    with col4:
                        if result.summary:
                            # Show first metric from summary
                            first_key = list(result.summary.keys())[0] if result.summary else None
                            if first_key:
                                value = result.summary[first_key]
                                if isinstance(value, (int, float)):
                                    st.metric(first_key.replace('_', ' ').title(), value)
                    
                    # Additional details
                    if result.summary:
                        st.markdown("**Summary:**")
                        for key, value in result.summary.items():
                            if not isinstance(value, (list, dict)):
                                st.write(f"• {key.replace('_', ' ').title()}: {value}")
                    
                    if result.logs and len(result.logs) > 0:
                        st.markdown("**Recent Logs:**")
                        for log in result.logs[-3:]:
                            st.caption(f"• {log}")
    
    def export_workflow_data(self, workflow_state: WorkflowState,
                            phase_results: Dict[str, PhaseResult],
                            artifacts: Dict[str, List[str]]) -> str:
        """
        Export workflow data as JSON.
        
        Args:
            workflow_state: Current workflow state
            phase_results: Phase execution results
            artifacts: Created artifacts
            
        Returns:
            JSON string of workflow data
        """
        export_data = {
            'workflow_id': workflow_state.workflow_id,
            'task': workflow_state.task_description,
            'model': workflow_state.model,
            'timestamp': datetime.now().isoformat(),
            'phases': {},
            'artifacts': artifacts,
            'graph': {
                'nodes': [],
                'edges': []
            }
        }
        
        # Add phase data
        for phase_key, result in phase_results.items():
            if phase_key in self.phase_info:
                info = self.phase_info[phase_key]
                export_data['phases'][phase_key] = {
                    'name': info['name'],
                    'status': result.status.value,
                    'duration': result.duration,
                    'started_at': result.started_at.isoformat() if result.started_at else None,
                    'completed_at': result.completed_at.isoformat() if result.completed_at else None,
                    'summary': result.summary
                }
                
                # Add to graph
                export_data['graph']['nodes'].append({
                    'id': phase_key,
                    'label': info['name'],
                    'icon': info['icon'],
                    'status': result.status.value
                })
        
        # Add edges
        for phase_key in export_data['phases']:
            if phase_key in self.phase_info:
                for dep in self.phase_info[phase_key]['dependencies']:
                    export_data['graph']['edges'].append({
                        'source': dep,
                        'target': phase_key
                    })
        
        return json.dumps(export_data, indent=2)