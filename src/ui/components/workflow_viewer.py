# -*- coding: utf-8 -*-
"""
Workflow Viewer Component - Displays workflow graph and execution state.

This component provides visual representation of the workflow execution,
including Mermaid diagrams, phase dependencies, and real-time artifact tracking.
"""

import streamlit as st
import json
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import graphlib

from agents.workflow import WorkflowState
from .phase_executor import PhaseResult


class WorkflowViewer:
    """
    Display workflow execution graph and track artifacts in real-time.
    
    This component creates visual representations of the workflow,
    showing phase dependencies, execution status, and artifact creation.
    """
    
    def __init__(self):
        """Initialize the workflow viewer."""
        self.phase_info = {
            'analyzer': {
                'name': 'Analyzer',
                'icon': '🔍',
                'color': '#4CAF50',
                'dependencies': []
            },
            'specification': {
                'name': 'Specification',
                'icon': '📋',
                'color': '#2196F3',
                'dependencies': ['analyzer']
            },
            'crafter': {
                'name': 'Crafter',
                'icon': '💻',
                'color': '#FF9800',
                'dependencies': ['specification']
            },
            'evaluator': {
                'name': 'Evaluator',
                'icon': '✅',
                'color': '#9C27B0',
                'dependencies': ['crafter']
            },
            'test_analyzer': {
                'name': 'Test Analyzer',
                'icon': '🧪',
                'color': '#00BCD4',
                'dependencies': ['specification']
            },
            'test_stubber': {
                'name': 'Test Stubber',
                'icon': '🏗️',
                'color': '#CDDC39',
                'dependencies': ['test_analyzer']
            },
            'test_crafter': {
                'name': 'Test Crafter',
                'icon': '🔨',
                'color': '#795548',
                'dependencies': ['test_stubber']
            }
        }
    
    def render(self, workflow_state: WorkflowState, 
               phase_results: Dict[str, PhaseResult],
               artifacts: Dict[str, List[str]]):
        """
        Render the workflow viewer.
        
        Args:
            workflow_state: Current workflow state
            phase_results: Results from completed phases
            artifacts: Artifacts created by phases
        """
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Graph View", "🔄 Phase Flow", "📈 Artifact Timeline"])
        
        with tab1:
            self._render_mermaid_graph(workflow_state, phase_results)
        
        with tab2:
            self._render_phase_flow(workflow_state, phase_results)
        
        with tab3:
            self._render_artifact_timeline(artifacts, phase_results)
    
    def _render_mermaid_graph(self, workflow_state: WorkflowState, 
                              phase_results: Dict[str, PhaseResult]):
        """Render workflow as Mermaid diagram."""
        st.markdown("### Workflow Execution Graph")
        
        # Build Mermaid diagram
        mermaid_code = self._generate_mermaid_diagram(workflow_state, phase_results)
        
        # Display using st.graphviz_chart alternative (Mermaid)
        st.code(mermaid_code, language="mermaid")
        
        # Add legend
        st.markdown("#### Legend")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("✅ **Completed**")
        with col2:
            st.markdown("🔵 **Running**")
        with col3:
            st.markdown("⏳ **Pending**")
        with col4:
            st.markdown("🔴 **Failed**")
    
    def _generate_mermaid_diagram(self, workflow_state: WorkflowState,
                                  phase_results: Dict[str, PhaseResult]) -> str:
        """Generate Mermaid diagram code for the workflow."""
        lines = ["graph TD"]
        
        # Determine which phases to include
        phases_to_include = list(self.phase_info.keys())
        if not workflow_state.generate_tests:
            phases_to_include = [p for p in phases_to_include if not p.startswith('test_')]
        
        # Add nodes
        for phase_key in phases_to_include:
            info = self.phase_info[phase_key]
            status = self._get_phase_status(phase_key, workflow_state, phase_results)
            
            # Node styling based on status
            if status == 'completed':
                style = f"style {phase_key} fill:#4CAF50,stroke:#2E7D32,color:#fff"
            elif status == 'running':
                style = f"style {phase_key} fill:#2196F3,stroke:#1565C0,color:#fff"
            elif status == 'failed':
                style = f"style {phase_key} fill:#F44336,stroke:#C62828,color:#fff"
            else:  # pending
                style = f"style {phase_key} fill:#E0E0E0,stroke:#9E9E9E"
            
            # Add node
            label = f"{info['icon']} {info['name']}"
            if phase_key in phase_results:
                result = phase_results[phase_key]
                if result.duration:
                    label += f"<br/>({result.duration:.1f}s)"
            
            lines.append(f"    {phase_key}[\"{label}\"]")
            lines.append(f"    {style}")
        
        # Add edges (dependencies)
        for phase_key in phases_to_include:
            info = self.phase_info[phase_key]
            for dep in info['dependencies']:
                if dep in phases_to_include:
                    lines.append(f"    {dep} --> {phase_key}")
        
        return "\n".join(lines)
    
    def _render_phase_flow(self, workflow_state: WorkflowState,
                          phase_results: Dict[str, PhaseResult]):
        """Render detailed phase flow with metrics."""
        st.markdown("### Phase Execution Flow")
        
        # Calculate phase order using topological sort
        phases_to_include = list(self.phase_info.keys())
        if not workflow_state.generate_tests:
            phases_to_include = [p for p in phases_to_include if not p.startswith('test_')]
        
        # Create dependency graph
        graph = {}
        for phase in phases_to_include:
            deps = self.phase_info[phase]['dependencies']
            graph[phase] = [d for d in deps if d in phases_to_include]
        
        # Topological sort
        try:
            phase_order = list(graphlib.TopologicalSorter(graph).static_order())
        except graphlib.CycleError:
            st.error("Cycle detected in phase dependencies!")
            return
        
        # Display phases in order
        for i, phase_key in enumerate(phase_order):
            info = self.phase_info[phase_key]
            status = self._get_phase_status(phase_key, workflow_state, phase_results)
            
            # Create columns for phase display
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                # Status indicator
                if status == 'completed':
                    st.success(f"{info['icon']} ✅")
                elif status == 'running':
                    st.info(f"{info['icon']} 🔵")
                elif status == 'failed':
                    st.error(f"{info['icon']} ❌")
                else:
                    st.text(f"{info['icon']} ⏳")
            
            with col2:
                st.markdown(f"**{info['name']}**")
                
                # Show phase details if completed
                if phase_key in phase_results:
                    result = phase_results[phase_key]
                    
                    # Metrics
                    metric_cols = st.columns(3)
                    with metric_cols[0]:
                        st.metric("Duration", f"{result.duration:.2f}s")
                    
                    if result.summary:
                        # Display summary metrics
                        summary_items = list(result.summary.items())[:2]
                        for idx, (key, value) in enumerate(summary_items):
                            with metric_cols[idx + 1]:
                                # Convert list values to string or count
                                if isinstance(value, list):
                                    display_value = len(value) if value else 0
                                elif isinstance(value, (dict, set)):
                                    display_value = len(value) if value else 0
                                else:
                                    display_value = value
                                st.metric(key.replace('_', ' ').title(), display_value)
                    
                    # Artifacts created
                    if result.artifacts:
                        with st.expander(f"Artifacts ({len(result.artifacts)})"):
                            for artifact in result.artifacts[:3]:
                                st.code(artifact, language=None)
                            if len(result.artifacts) > 3:
                                st.text(f"... and {len(result.artifacts) - 3} more")
            
            with col3:
                # Dependencies
                if info['dependencies']:
                    deps_text = ", ".join([
                        self.phase_info[d]['icon'] for d in info['dependencies']
                        if d in phases_to_include
                    ])
                    st.caption(f"Depends on: {deps_text}")
            
            # Add separator between phases
            if i < len(phase_order) - 1:
                st.divider()
    
    def _render_artifact_timeline(self, artifacts: Dict[str, List[str]],
                                 phase_results: Dict[str, PhaseResult]):
        """Render timeline of artifact creation."""
        st.markdown("### Artifact Creation Timeline")
        
        # Create timeline data
        timeline_data = []
        
        for phase_key, artifact_list in artifacts.items():
            if phase_key in phase_results:
                result = phase_results[phase_key]
                phase_info = self.phase_info.get(phase_key, {})
                
                for artifact in artifact_list:
                    # Parse artifact type and path
                    if artifact.startswith('storage_kv:'):
                        artifact_type = 'KV Store'
                        path = artifact.replace('storage_kv:', '')
                    elif artifact.startswith('storage_fs:'):
                        artifact_type = 'File System'
                        path = artifact.replace('storage_fs:', '')
                    else:
                        artifact_type = 'Unknown'
                        path = artifact
                    
                    timeline_data.append({
                        'phase': phase_info.get('name', phase_key),
                        'icon': phase_info.get('icon', '📊'),
                        'type': artifact_type,
                        'path': path,
                        'duration': result.duration
                    })
        
        # Display timeline
        if timeline_data:
            # Group by phase
            current_phase = None
            for item in timeline_data:
                if item['phase'] != current_phase:
                    current_phase = item['phase']
                    st.markdown(f"#### {item['icon']} {current_phase}")
                
                # Display artifact
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.caption(item['type'])
                with col2:
                    st.code(item['path'], language=None)
        else:
            st.info("No artifacts created yet")
        
        # Summary statistics
        if artifacts:
            st.divider()
            st.markdown("#### Summary")
            
            total_artifacts = sum(len(arts) for arts in artifacts.values())
            kv_count = sum(1 for arts in artifacts.values() for a in arts if 'storage_kv:' in a)
            fs_count = sum(1 for arts in artifacts.values() for a in arts if 'storage_fs:' in a)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Artifacts", total_artifacts)
            with col2:
                st.metric("KV Store", kv_count)
            with col3:
                st.metric("File System", fs_count)
    
    def _get_phase_status(self, phase_key: str, workflow_state: WorkflowState,
                         phase_results: Dict[str, PhaseResult]) -> str:
        """Determine the status of a phase."""
        if phase_key in phase_results:
            result = phase_results[phase_key]
            return 'failed' if not result.success else 'completed'
        
        # Check if phase is currently running
        if hasattr(workflow_state, 'metadata') and workflow_state.metadata:
            if workflow_state.metadata.current_phase == phase_key:
                return 'running'
        
        return 'pending'
    
    def export_workflow_graph(self, workflow_state: WorkflowState,
                            phase_results: Dict[str, PhaseResult]) -> Dict[str, Any]:
        """
        Export workflow graph data for external visualization.
        
        Args:
            workflow_state: Current workflow state
            phase_results: Results from completed phases
            
        Returns:
            Dictionary with graph data
        """
        nodes = []
        edges = []
        
        phases_to_include = list(self.phase_info.keys())
        if not workflow_state.generate_tests:
            phases_to_include = [p for p in phases_to_include if not p.startswith('test_')]
        
        # Build nodes
        for phase_key in phases_to_include:
            info = self.phase_info[phase_key]
            status = self._get_phase_status(phase_key, workflow_state, phase_results)
            
            node_data = {
                'id': phase_key,
                'label': info['name'],
                'icon': info['icon'],
                'status': status,
                'color': info['color']
            }
            
            if phase_key in phase_results:
                result = phase_results[phase_key]
                node_data['duration'] = result.duration
                node_data['artifacts_count'] = len(result.artifacts)
                if result.summary:
                    node_data['summary'] = result.summary
            
            nodes.append(node_data)
        
        # Build edges
        for phase_key in phases_to_include:
            info = self.phase_info[phase_key]
            for dep in info['dependencies']:
                if dep in phases_to_include:
                    edges.append({
                        'source': dep,
                        'target': phase_key
                    })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'workflow_id': workflow_state.workflow_id,
            'task': workflow_state.task_description,
            'timestamp': datetime.now().isoformat()
        }