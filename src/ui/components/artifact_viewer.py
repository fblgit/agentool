# -*- coding: utf-8 -*-
"""
Artifact viewer component for displaying workflow artifacts.

This component provides a tree-based navigation interface for viewing
the various artifacts generated during the workflow execution.
"""

import streamlit as st
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import pandas as pd
from agentool.core.injector import get_injector


class ArtifactViewer:
    """
    Display workflow artifacts in an organized, navigable format.
    
    This component creates a hierarchical view of artifacts with
    expandable sections and formatted content display.
    """
    
    def __init__(self):
        """Initialize the artifact viewer."""
        self.artifact_types = {
            'analysis': {'icon': '🔍', 'name': 'Analysis'},
            'catalog': {'icon': '📚', 'name': 'Tool Catalog'},
            'specifications': {'icon': '📋', 'name': 'Specifications'},
            'implementation': {'icon': '💻', 'name': 'Implementation'},
            'evaluation': {'icon': '✅', 'name': 'Evaluation'},
            'validation': {'icon': '✅', 'name': 'Validation'},
            'existing_tools': {'icon': '🔧', 'name': 'Existing Tools'},
            'missing_tools': {'icon': '❓', 'name': 'Missing Tools'},
            'skeleton': {'icon': '🏗️', 'name': 'Code Skeleton'},
            'test_analysis': {'icon': '🧪', 'name': 'Test Analysis'},
            'test_stub': {'icon': '🏗️', 'name': 'Test Stubs'},
            'test_implementation': {'icon': '🔨', 'name': 'Test Implementation'},
            'final': {'icon': '🎯', 'name': 'Final Output'},
            'summary': {'icon': '📊', 'name': 'Summary'}
        }
        self.injector = get_injector()
    
    def render(self, artifacts: Dict[str, Any]):
        """
        Render the artifact viewer.
        
        Args:
            artifacts: Dictionary of artifact name to data
        """
        if not artifacts:
            st.info("No artifacts generated yet. Run the workflow to see results.")
            return
        
        # Group artifacts by type
        grouped = self._group_artifacts(artifacts)
        
        # Search functionality
        search_term = st.text_input("🔍 Search artifacts", placeholder="Type to filter...")
        
        # Display artifacts
        for artifact_type, items in grouped.items():
            if artifact_type in self.artifact_types:
                type_info = self.artifact_types[artifact_type]
                
                # Filter based on search
                if search_term:
                    filtered_items = self._filter_artifacts(items, search_term)
                    if not filtered_items:
                        continue
                    items = filtered_items
                
                # Create expandable section
                with st.expander(f"{type_info['icon']} {type_info['name']} ({len(items)} items)", 
                               expanded=artifact_type in ['analysis', 'implementation']):
                    
                    if artifact_type == 'analysis':
                        self._render_analysis(items)
                    elif artifact_type == 'catalog':
                        self._render_catalog(items)
                    elif artifact_type == 'specifications':
                        self._render_specifications(items)
                    elif artifact_type == 'implementation':
                        self._render_implementation(items)
                    elif artifact_type == 'evaluation':
                        self._render_evaluation(items)
                    elif artifact_type == 'validation':
                        self._render_validation(items)
                    elif artifact_type == 'existing_tools':
                        self._render_existing_tools(items)
                    elif artifact_type == 'missing_tools':
                        self._render_missing_tools(items)
                    elif artifact_type == 'skeleton':
                        self._render_skeleton(items)
                    elif artifact_type == 'test_analysis':
                        self._render_test_analysis(items)
                    elif artifact_type == 'test_stub':
                        self._render_test_stub(items)
                    elif artifact_type == 'test_implementation':
                        self._render_test_implementation(items)
                    elif artifact_type == 'final':
                        self._render_final_output(items)
                    elif artifact_type == 'summary':
                        self._render_summary(items)
                    else:
                        self._render_generic(items)
    
    def _group_artifacts(self, artifacts: Dict[str, Any]) -> Dict[str, List[Any]]:
        """Group artifacts by type."""
        grouped = {}
        
        for key, value in artifacts.items():
            # Determine artifact type from key
            if 'analysis' in key:
                artifact_type = 'analysis'
            elif 'catalog' in key:
                artifact_type = 'catalog'
            elif 'specification' in key:
                artifact_type = 'specifications'
            elif 'implementation' in key:
                artifact_type = 'implementation'
            elif 'validation' in key:
                artifact_type = 'validation'
            elif 'evaluation' in key:
                artifact_type = 'evaluation'
            elif 'existing_tools' in key:
                artifact_type = 'existing_tools'
            elif 'missing_tools' in key:
                artifact_type = 'missing_tools'
            elif 'skeleton' in key:
                artifact_type = 'skeleton'
            elif 'test_analysis' in key:
                artifact_type = 'test_analysis'
            elif 'test_stub' in key:
                artifact_type = 'test_stub'
            elif 'test_implementation' in key:
                artifact_type = 'test_implementation'
            elif 'final' in key:
                artifact_type = 'final'
            elif 'SUMMARY' in key:
                artifact_type = 'summary'
            else:
                artifact_type = 'other'
            
            if artifact_type not in grouped:
                grouped[artifact_type] = []
            
            grouped[artifact_type].append({
                'key': key,
                'data': value
            })
        
        return grouped
    
    def _filter_artifacts(self, items: List[Dict[str, Any]], search_term: str) -> List[Dict[str, Any]]:
        """Filter artifacts based on search term."""
        search_lower = search_term.lower()
        filtered = []
        
        for item in items:
            # Check in key
            if search_lower in item['key'].lower():
                filtered.append(item)
                continue
            
            # Check in data (convert to string for searching)
            if search_lower in json.dumps(item['data']).lower():
                filtered.append(item)
        
        return filtered
    
    def _fetch_artifact_content(self, key: str, storage_type: str) -> Tuple[bool, Any]:
        """Fetch artifact content from storage."""
        try:
            if storage_type == 'storage_kv':
                result = asyncio.run(self.injector.run('storage_kv', {
                    'operation': 'get',
                    'key': key
                }))
                
                if hasattr(result, 'output'):
                    data = json.loads(result.output)
                else:
                    data = result.data if hasattr(result, 'data') else result
                
                if data.get('data', {}).get('exists', False):
                    return True, json.loads(data['data']['value'])
                    
            elif storage_type == 'storage_fs':
                result = asyncio.run(self.injector.run('storage_fs', {
                    'operation': 'read',
                    'path': key
                }))
                
                if hasattr(result, 'output'):
                    data = json.loads(result.output)
                else:
                    data = result.data if hasattr(result, 'data') else result
                
                if data.get('data', {}).get('content'):
                    return True, data['data']['content']
                    
        except Exception as e:
            st.error(f"Error fetching artifact: {e}")
            
        return False, None
    
    def _render_analysis(self, items: List[Dict[str, Any]]):
        """Render analysis artifacts."""
        for item in items:
            # Fetch actual content from storage
            storage_type = item['data'].get('type', 'storage_kv')
            success, content = self._fetch_artifact_content(item['key'], storage_type)
            
            if not success:
                st.warning(f"Could not fetch content for {item['key']}")
                continue
            
            data = content
            
            st.markdown(f"### {data.get('name', 'Analysis')}")
            st.markdown(f"**Description:** {data.get('description', 'N/A')}")
            
            if 'system_design' in data:
                st.markdown("**System Design:**")
                st.text_area("", value=data['system_design'], height=200, disabled=True)
            
            if 'missing_tools' in data:
                st.markdown("**Missing Tools:**")
                for tool in data['missing_tools']:
                    st.markdown(f"- {tool}")
            
            if 'integration_points' in data:
                st.markdown("**Integration Points:**")
                for point in data['integration_points']:
                    st.markdown(f"- {point}")
    
    def _render_catalog(self, items: List[Dict[str, Any]]):
        """Render tool catalog artifacts."""
        for item in items:
            # Fetch actual content
            storage_type = item['data'].get('type', 'storage_kv')
            success, content = self._fetch_artifact_content(item['key'], storage_type)
            
            if not success:
                st.warning(f"Could not fetch catalog for {item['key']}")
                continue
            
            data = content
            
            if isinstance(data, list):
                # Create DataFrame for better display
                df_data = []
                for tool in data:
                    df_data.append({
                        'Name': tool.get('name', 'Unknown'),
                        'Type': tool.get('type', 'Unknown'),
                        'Description': tool.get('description', 'N/A')[:100] + '...',
                        'Has Schema': '✅' if tool.get('input_schema') else '❌'
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.json(data)
    
    def _render_specifications(self, items: List[Dict[str, Any]]):
        """Render specification artifacts."""
        for item in items:
            # Fetch actual content
            storage_type = item['data'].get('type', 'storage_kv')
            success, content = self._fetch_artifact_content(item['key'], storage_type)
            
            if not success:
                st.warning(f"Could not fetch specification for {item['key']}")
                continue
            
            data = content
            spec_name = item['key'].split('/')[-1]
            
            with st.container():
                st.markdown(f"#### 📋 {spec_name}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Description:**")
                    st.info(data.get('description', 'No description'))
                    
                    if 'dependencies' in data:
                        st.markdown("**Dependencies:**")
                        for dep in data.get('dependencies', []):
                            st.markdown(f"- {dep}")
                
                with col2:
                    if 'input_schema' in data:
                        st.markdown("**Input Schema:**")
                        st.json(data['input_schema'])
                    
                    if 'output_schema' in data:
                        st.markdown("**Output Schema:**")
                        st.json(data['output_schema'])
                
                st.divider()
    
    def _render_implementation(self, items: List[Dict[str, Any]]):
        """Render implementation artifacts."""
        for item in items:
            # Fetch actual content
            storage_type = item['data'].get('type', 'storage_kv')
            success, content = self._fetch_artifact_content(item['key'], storage_type)
            
            if not success:
                st.warning(f"Could not fetch implementation for {item['key']}")
                continue
            
            data = content
            
            if isinstance(data, dict) and 'code' in data:
                # Display code with syntax highlighting
                st.code(data['code'], language='python')
                
                if 'metadata' in data:
                    with st.expander("Implementation Metadata"):
                        st.json(data['metadata'])
            else:
                # Fallback display
                st.code(str(data), language='python')
    
    def _render_evaluation(self, items: List[Dict[str, Any]]):
        """Render evaluation artifacts."""
        for item in items:
            # Fetch actual content
            storage_type = item['data'].get('type', 'storage_kv')
            success, content = self._fetch_artifact_content(item['key'], storage_type)
            
            if not success:
                st.warning(f"Could not fetch evaluation for {item['key']}")
                continue
            
            data = content
            
            if 'passed' in data:
                if data['passed']:
                    st.success("✅ Evaluation Passed")
                else:
                    st.error("❌ Evaluation Failed")
            
            if 'issues' in data and data['issues']:
                st.warning(f"Found {len(data['issues'])} issues:")
                for issue in data['issues']:
                    st.markdown(f"- {issue}")
            
            if 'suggestions' in data and data['suggestions']:
                st.info("Suggestions for improvement:")
                for suggestion in data['suggestions']:
                    st.markdown(f"- {suggestion}")
            
            if 'metrics' in data:
                st.markdown("**Quality Metrics:**")
                metrics_df = pd.DataFrame([data['metrics']])
                st.dataframe(metrics_df, use_container_width=True)
    
    def _render_existing_tools(self, items: List[Dict[str, Any]]):
        """Render existing tools artifacts."""
        for item in items:
            # Fetch actual content
            storage_type = item['data'].get('type', 'storage_kv')
            success, content = self._fetch_artifact_content(item['key'], storage_type)
            
            if not success:
                continue  # Existing tools might not always be stored
            
            data = content
            
            if isinstance(data, list):
                st.markdown("**Available Tools:**")
                for tool in data:
                    with st.expander(f"🔧 {tool.get('name', 'Unknown Tool')}"):
                        st.markdown(f"**Type:** {tool.get('type', 'Unknown')}")
                        st.markdown(f"**Description:** {tool.get('description', 'No description')}")
                        
                        if 'input_schema' in tool:
                            st.markdown("**Input Schema:**")
                            st.json(tool['input_schema'])
    
    def _render_missing_tools(self, items: List[Dict[str, Any]]):
        """Render missing tools artifacts."""
        for item in items:
            # Fetch actual content
            storage_type = item['data'].get('type', 'storage_kv')
            success, content = self._fetch_artifact_content(item['key'], storage_type)
            
            if not success:
                st.warning(f"Could not fetch missing tool info for {item['key']}")
                continue
            
            data = content
            
            if isinstance(data, list):
                for tool_name in data:
                    st.markdown(f"- ❓ {tool_name}")
            else:
                st.json(data)
    
    def _render_skeleton(self, items: List[Dict[str, Any]]):
        """Render code skeleton artifacts."""
        for item in items:
            # Fetch actual content
            storage_type = item['data'].get('type', 'storage_kv')
            success, content = self._fetch_artifact_content(item['key'], storage_type)
            
            if not success:
                continue
            
            data = content
            
            if isinstance(data, str):
                st.code(data, language='python')
            else:
                st.json(data)
    
    def _render_validation(self, items: List[Dict[str, Any]]):
        """Render validation artifacts."""
        for item in items:
            # Fetch actual content
            storage_type = item['data'].get('type', 'storage_kv')
            success, content = self._fetch_artifact_content(item['key'], storage_type)
            
            if not success:
                st.warning(f"Could not fetch validation for {item['key']}")
                continue
            
            data = content
            tool_name = item['key'].split('/')[-1]
            
            with st.container():
                st.markdown(f"#### ✅ {tool_name} Validation")
                
                col1, col2 = st.columns(2)
                with col1:
                    if data.get('syntax_valid'):
                        st.success("Syntax Valid")
                    else:
                        st.error("Syntax Invalid")
                
                with col2:
                    if data.get('ready_for_deployment'):
                        st.success("Ready for Deployment")
                    else:
                        st.warning("Needs Attention")
                
                if data.get('issues'):
                    st.warning(f"Issues Found ({len(data['issues'])})")
                    for issue in data['issues'][:3]:
                        st.markdown(f"- {issue}")
                    if len(data['issues']) > 3:
                        st.text(f"... and {len(data['issues']) - 3} more")
                
                if data.get('fixes_applied'):
                    st.info(f"Fixes Applied ({len(data['fixes_applied'])})")
                    for fix in data['fixes_applied'][:3]:
                        st.markdown(f"- {fix}")
                
                st.divider()
    
    def _render_test_analysis(self, items: List[Dict[str, Any]]):
        """Render test analysis artifacts."""
        for item in items:
            # Fetch actual content
            storage_type = item['data'].get('type', 'storage_kv')
            success, content = self._fetch_artifact_content(item['key'], storage_type)
            
            if not success:
                continue
            
            data = content
            tool_name = item['key'].split('/')[-1]
            
            st.markdown(f"#### 🧪 Test Analysis: {tool_name}")
            
            if data.get('test_strategy'):
                st.markdown("**Test Strategy:**")
                st.info(data['test_strategy'])
            
            if data.get('test_cases'):
                st.markdown("**Planned Test Cases:**")
                for test_case in data.get('test_cases', [])[:5]:
                    st.markdown(f"- {test_case}")
    
    def _render_test_stub(self, items: List[Dict[str, Any]]):
        """Render test stub artifacts."""
        for item in items:
            # Fetch actual content
            storage_type = item['data'].get('type', 'storage_kv')
            success, content = self._fetch_artifact_content(item['key'], storage_type)
            
            if not success:
                continue
            
            data = content
            if storage_type == 'storage_fs':
                # This is the actual test stub code
                st.markdown(f"#### Test Stub: {item['key'].split('/')[-1]}")
                st.code(data, language='python')
            else:
                # This is metadata about the stub
                tool_name = item['key'].split('/')[-1]
                st.markdown(f"#### 🏗️ Test Stub: {tool_name}")
                if data.get('test_count'):
                    st.metric("Test Methods", data['test_count'])
    
    def _render_test_implementation(self, items: List[Dict[str, Any]]):
        """Render test implementation artifacts."""
        for item in items:
            # Fetch actual content
            storage_type = item['data'].get('type', 'storage_kv')
            success, content = self._fetch_artifact_content(item['key'], storage_type)
            
            if not success:
                continue
            
            data = content
            if storage_type == 'storage_fs':
                # This is the actual test implementation
                st.markdown(f"#### Test Implementation: {item['key'].split('/')[-1]}")
                st.code(data, language='python')
            else:
                # This is metadata about the implementation
                tool_name = item['key'].split('/')[-1]
                st.markdown(f"#### 🔨 Test Implementation: {tool_name}")
                
                if data.get('coverage_achieved'):
                    st.markdown("**Coverage:**")
                    coverage = data['coverage_achieved']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall", f"{coverage.get('overall', 0)}%")
                    with col2:
                        st.metric("Functions", f"{coverage.get('functions', 0)}%")
                    with col3:
                        st.metric("Edge Cases", f"{coverage.get('edge_cases', 0)}%")
    
    def _render_final_output(self, items: List[Dict[str, Any]]):
        """Render final output artifacts."""
        for item in items:
            # This is typically a directory, so list files
            st.markdown(f"#### 🎯 Final Output: {item['key']}")
            st.info("Final validated and improved code is stored here")
    
    def _render_summary(self, items: List[Dict[str, Any]]):
        """Render summary artifacts."""
        for item in items:
            # Fetch actual content
            storage_type = item['data'].get('type', 'storage_fs')
            success, content = self._fetch_artifact_content(item['key'], storage_type)
            
            if not success:
                continue
            
            st.markdown(f"#### 📊 Summary")
            if item['key'].endswith('.md'):
                # Render markdown
                st.markdown(content)
            else:
                # Render as text
                st.text(content)
    
    def _render_generic(self, items: List[Dict[str, Any]]):
        """Generic renderer for unknown artifact types."""
        for item in items:
            st.markdown(f"**{item['key']}**")
            
            # Try to fetch content
            storage_type = item['data'].get('type', 'storage_kv')
            success, content = self._fetch_artifact_content(item['key'], storage_type)
            
            if success:
                if isinstance(content, str):
                    st.text_area("", value=content, height=200, disabled=True)
                else:
                    st.json(content)
            else:
                st.json(item['data'])
    
    def export_artifact(self, artifact_key: str, artifact_data: Any) -> str:
        """
        Export a single artifact to JSON format.
        
        Args:
            artifact_key: Key/name of the artifact
            artifact_data: Artifact data
            
        Returns:
            JSON string of the artifact
        """
        export_data = {
            'key': artifact_key,
            'timestamp': datetime.now().isoformat(),
            'data': artifact_data
        }
        
        return json.dumps(export_data, indent=2)
    
    def export_all_artifacts(self, artifacts: Dict[str, Any]) -> str:
        """
        Export all artifacts to JSON format.
        
        Args:
            artifacts: All artifacts
            
        Returns:
            JSON string of all artifacts
        """
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'artifact_count': len(artifacts),
            'artifacts': artifacts
        }
        
        return json.dumps(export_data, indent=2)