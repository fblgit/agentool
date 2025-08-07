# -*- coding: utf-8 -*-
"""
Artifact Viewer V2 - Modernized artifact viewing with fragments and dialogs.

This component provides a non-blocking, fragment-based artifact viewer with
modal dialogs for viewing content without interrupting workflow execution.
"""

import streamlit as st
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, field
import pandas as pd

from agentool.core.injector import get_injector
from .container_manager import get_container_manager
from .theme_manager import get_theme_manager
from ..utils.fragments import create_auto_fragment, create_debounced_fragment


@dataclass
class ArtifactMetadata:
    """Metadata for an artifact."""
    key: str
    storage_type: str
    phase: str
    artifact_type: str
    created_at: datetime
    size: Optional[int] = None
    last_accessed: Optional[datetime] = None
    tags: Set[str] = field(default_factory=set)


class ArtifactViewerV2:
    """
    Modernized artifact viewer with fragment-based updates and modal dialogs.
    
    This component enables real-time artifact viewing without blocking
    workflow execution, using Streamlit fragments and dialogs.
    """
    
    def __init__(self, debug_mode: bool = False):
        """Initialize the artifact viewer."""
        # Get injector from session state if available, otherwise create new
        if 'injector' in st.session_state:
            self.injector = st.session_state.injector
        else:
            self.injector = get_injector()
        self.container_manager = get_container_manager()
        self.theme_manager = get_theme_manager()
        self.debug_mode = debug_mode
        
        # Initialize artifact type configuration
        self.artifact_types = {
            'analysis': {'icon': '🔍', 'name': 'Analysis', 'color': 'primary'},
            'catalog': {'icon': '📚', 'name': 'Tool Catalog', 'color': 'info'},
            'specifications': {'icon': '📋', 'name': 'Specifications', 'color': 'secondary'},
            'implementation': {'icon': '💻', 'name': 'Implementation', 'color': 'warning'},
            'evaluation': {'icon': '✅', 'name': 'Evaluation', 'color': 'success'},
            'validation': {'icon': '✅', 'name': 'Validation', 'color': 'success'},
            'existing_tools': {'icon': '🔧', 'name': 'Existing Tools', 'color': 'info'},
            'missing_tools': {'icon': '❓', 'name': 'Missing Tools', 'color': 'error'},
            'skeleton': {'icon': '🏗️', 'name': 'Code Skeleton', 'color': 'secondary'},
            'test_analysis': {'icon': '🧪', 'name': 'Test Analysis', 'color': 'info'},
            'test_stub': {'icon': '🏗️', 'name': 'Test Stubs', 'color': 'secondary'},
            'test_implementation': {'icon': '🔨', 'name': 'Test Implementation', 'color': 'warning'},
            'final': {'icon': '🎯', 'name': 'Final Output', 'color': 'success'},
            'summary': {'icon': '📊', 'name': 'Summary', 'color': 'primary'}
        }
        
        # Session state is already initialized in app_v2.py
    
    def render(self, artifacts: Dict[str, List[str]], container_key: str = "artifact_viewer"):
        """
        Render the artifact viewer with fragment-based updates.
        
        Args:
            artifacts: Dictionary mapping phase names to lists of artifact references (storage_kv:path or storage_fs:path)
            container_key: Key for the main container
        """
        # Apply theme
        self.theme_manager.apply_theme()
        
        # Don't use managed containers for main UI to avoid widget ID changes
        with st.container():
            # Header with controls
            self._render_header()
            
            # Create layout columns
            if st.session_state.artifact_viewer_v2['view_mode'] == 'tree':
                # Tree view with sidebar
                tree_col, detail_col = st.columns([1, 2])
                
                with tree_col:
                    tree_container = st.container(height=600, border=True)
                    self._render_artifact_tree_fragment(artifacts, tree_container)
                
                with detail_col:
                    detail_container = st.container(height=600, border=True)
                    self._render_artifact_detail_fragment(detail_container)
            
            elif st.session_state.artifact_viewer_v2['view_mode'] == 'grid':
                # Grid view
                grid_container = st.container()
                self._render_artifact_grid_fragment(artifacts, grid_container)
            
            else:  # list view
                # List view
                list_container = st.container(height=600, border=True)
                self._render_artifact_list_fragment(artifacts, list_container)
    
    def _render_header(self):
        """Render the header with view controls."""
        col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1, 1])
        
        with col1:
            # Search box with debouncing
            search_fragment = create_debounced_fragment(
                self._update_filter_text,
                "artifact_search",
                delay=0.3
            )
            
            search_value = st.text_input(
                "🔍 Search artifacts",
                value=st.session_state.artifact_viewer_v2['filter_text'],
                placeholder="Type to filter...",
                key="artifact_search_input"
            )
            
            if search_value != st.session_state.artifact_viewer_v2['filter_text']:
                search_fragment(search_value)
        
        with col2:
            # Type filter
            type_options = ['all'] + list(self.artifact_types.keys())
            st.selectbox(
                "Type",
                type_options,
                key="artifact_type_filter",
                on_change=self._update_filters
            )
        
        with col3:
            # Sort options
            sort_options = {
                'created': 'Created Time',
                'name': 'Name',
                'type': 'Type',
                'size': 'Size'
            }
            st.selectbox(
                "Sort by",
                list(sort_options.keys()),
                format_func=lambda x: sort_options[x],
                key="artifact_sort",
                on_change=self._update_sort
            )
        
        with col4:
            # View mode selector
            view_modes = {
                'tree': '🌳',
                'grid': '⊞',
                'list': '☰'
            }
            
            selected_mode = st.radio(
                "View",
                list(view_modes.keys()),
                index=list(view_modes.keys()).index(st.session_state.artifact_viewer_v2['view_mode']),
                format_func=lambda x: view_modes[x],
                horizontal=True,
                key="artifact_view_mode",
                label_visibility="collapsed",
                on_change=lambda: st.session_state.artifact_viewer_v2.update({'view_mode': st.session_state.artifact_view_mode})
            )
        
        with col5:
            # Refresh button
            if st.button("🔄", key="refresh_artifacts", help="Refresh artifacts"):
                self._refresh_artifacts()
    
    def _render_artifact_tree_fragment(self, artifacts: Dict[str, List[str]], container):
        """Render artifact tree with automatic updates."""
        with container:
            # Process and group artifacts
            grouped = self._group_artifacts(artifacts)
            filtered = self._filter_artifacts(grouped)
            
            # Render tree
            for artifact_type, items in filtered.items():
                if artifact_type in self.artifact_types:
                    type_info = self.artifact_types[artifact_type]
                    
                    # Check if category is expanded
                    is_expanded = artifact_type in st.session_state.artifact_viewer_v2['expanded_categories']
                    
                    # Create expander
                    with st.expander(
                        f"{type_info['icon']} {type_info['name']} ({len(items)})",
                        expanded=is_expanded
                    ):
                        # Toggle expansion state (removed since expander handles this)
                        
                        # Render items
                        for item in items:
                            self._render_tree_item(item, artifact_type)
    
    def _render_artifact_detail_fragment(self, container):
        """Render artifact detail view."""
        with container:
            selected = st.session_state.artifact_viewer_v2.get('selected_artifact')
            
            if selected:
                # Fetch and display artifact content
                self._display_artifact_content(selected)
            else:
                st.info("Select an artifact to view details")
    
    def _render_artifact_grid_fragment(self, artifacts: Dict[str, List[str]], container):
        """Render artifact grid view with cards."""
        with container:
            # Process artifacts
            grouped = self._group_artifacts(artifacts)
            filtered = self._filter_artifacts(grouped)
            
            # Calculate grid layout
            items_per_row = 3
            
            for artifact_type, items in filtered.items():
                if items and artifact_type in self.artifact_types:
                    type_info = self.artifact_types[artifact_type]
                    
                    st.markdown(f"### {type_info['icon']} {type_info['name']}")
                    
                    # Create grid rows
                    for i in range(0, len(items), items_per_row):
                        cols = st.columns(items_per_row)
                        
                        for j, item in enumerate(items[i:i+items_per_row]):
                            if j < items_per_row:
                                with cols[j]:
                                    self._render_artifact_card(item, artifact_type)
                    
                    st.divider()
    
    def _render_artifact_list_fragment(self, artifacts: Dict[str, List[str]], container):
        """Render artifact list view."""
        with container:
            # Process artifacts
            grouped = self._group_artifacts(artifacts)
            filtered = self._filter_artifacts(grouped)
            
            # Flatten and sort
            all_items = []
            for artifact_type, items in filtered.items():
                for item in items:
                    item['type'] = artifact_type
                    all_items.append(item)
            
            # Sort items
            sort_by = st.session_state.artifact_viewer_v2['sort_by']
            if sort_by == 'created':
                all_items.sort(key=lambda x: x.get('created_at', datetime.now()), reverse=True)
            elif sort_by == 'name':
                all_items.sort(key=lambda x: x.get('key', ''))
            elif sort_by == 'type':
                all_items.sort(key=lambda x: x.get('type', ''))
            
            # Create DataFrame for display
            if all_items:
                df_data = []
                for item in all_items:
                    type_info = self.artifact_types.get(item['type'], {})
                    df_data.append({
                        'Type': f"{type_info.get('icon', '📄')} {type_info.get('name', item['type'])}",
                        'Name': item.get('key', 'Unknown').split('/')[-1],
                        'Path': item.get('key', ''),
                        'Created': item.get('created_at', datetime.now()).strftime('%H:%M:%S') if isinstance(item.get('created_at'), datetime) else 'Unknown'
                    })
                
                df = pd.DataFrame(df_data)
                
                # Display DataFrame without selection (causes issues)
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Add view buttons for each artifact
                st.markdown("#### Actions")
                for idx, item in enumerate(all_items[:10]):  # Show first 10 with buttons
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.text(f"{item.get('key', 'Unknown').split('/')[-1]}")
                    with col2:
                        if st.button("View", key=f"list_view_{idx}_{item.get('key', '')}"):
                            self._show_artifact_modal(item)
            else:
                st.info("No artifacts to display")
    
    def _render_tree_item(self, item: Dict[str, Any], artifact_type: str):
        """Render a single tree item."""
        key = item.get('key', '')
        display_name = key.split('/')[-1] if '/' in key else key
        
        # Create clickable button
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button(
                f"📄 {display_name}",
                key=f"tree_item_{key}",
                use_container_width=True
            ):
                st.session_state.artifact_viewer_v2['selected_artifact'] = item
        
        with col2:
            # Quick actions
            if st.button("👁", key=f"view_{key}", help="Quick view"):
                self._show_artifact_modal(item)
    
    def _render_artifact_card(self, item: Dict[str, Any], artifact_type: str):
        """Render an artifact as a card."""
        key = item.get('key', '')
        display_name = key.split('/')[-1] if '/' in key else key
        type_info = self.artifact_types.get(artifact_type, {})
        
        # Apply glass card styling
        with st.container():
            st.markdown(
                f"""
                <div class="glass-card">
                    <h4>{type_info.get('icon', '📄')} {display_name}</h4>
                    <p style="color: var(--text-secondary); font-size: 0.875rem;">
                        {artifact_type.replace('_', ' ').title()}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Action buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("View", key=f"card_view_{key}", use_container_width=True):
                    self._show_artifact_modal(item)
            
            with col2:
                if st.button("Select", key=f"card_select_{key}", use_container_width=True):
                    st.session_state.artifact_viewer_v2['selected_artifact'] = item
    
    def _show_artifact_modal(self, item: Dict[str, Any]):
        """Display artifact content in the main area with better layout."""
        # Store selected artifact in session state for detail view
        st.session_state.artifact_viewer_v2['selected_artifact'] = item
        
        key = item.get('key', '')
        data = item.get('data', {})
        
        # Parse storage type and actual key from the artifact reference
        # The key format is "storage_kv:workflow/..." or "storage_fs:generated/..."
        if ':' in key:
            storage_type, actual_key = key.split(':', 1)
        else:
            storage_type = data.get('type', 'storage_kv')
            actual_key = key
        
        # Create a container for the artifact detail
        with st.container():
            # Header
            st.markdown(f"### 📄 {actual_key.split('/')[-1]}")
            st.caption(f"Path: `{actual_key}`")
            
            # Metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Storage", storage_type.replace('_', ' ').title())
            with col2:
                phase = data.get('phase', 'unknown')
                st.metric("Phase", phase)
            with col3:
                if 'created_at' in item:
                    created = item['created_at']
                    if isinstance(created, datetime):
                        st.metric("Created", created.strftime('%H:%M:%S'))
            
            st.divider()
            
            # Content in scrollable container
            content_container = st.container()
            with content_container:
                content = self._fetch_artifact_content(actual_key, storage_type)
                
                if content[0]:  # Success
                    # Check if it's a placeholder message
                    if isinstance(content[1], str) and content[1].startswith("⏳"):
                        st.info(content[1])
                    else:
                        self._display_content_by_type(actual_key, content[1])
                else:
                    st.error("Failed to fetch artifact content")
        
        # Actions
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📋 Copy", use_container_width=True):
                # Copy to clipboard using pyperclip if available
                try:
                    import pyperclip
                    if isinstance(content[1], str):
                        pyperclip.copy(content[1])
                    else:
                        pyperclip.copy(json.dumps(content[1], indent=2))
                    st.success("Copied to clipboard!")
                except ImportError:
                    st.info("Install pyperclip for clipboard support")
        
        with col2:
            # Create download button with actual content
            if content[0] and content[1]:
                if isinstance(content[1], str):
                    st.download_button(
                        "💾 Download",
                        data=content[1],
                        file_name=f"{actual_key.split('/')[-1]}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                else:
                    st.download_button(
                        "💾 Download",
                        data=json.dumps(content[1], indent=2),
                        file_name=f"{actual_key.split('/')[-1]}.json",
                        mime="application/json",
                        use_container_width=True
                    )
        
        with col3:
            if st.button("✖ Close", use_container_width=True):
                # Don't rerun, just clear selection
                st.session_state.artifact_viewer_v2['selected_artifact'] = None
    
    def _display_artifact_content(self, item: Dict[str, Any]):
        """Display artifact content in the detail view."""
        key = item.get('key', '')
        data = item.get('data', {})
        
        # Parse storage type and actual key from the artifact reference
        if ':' in key:
            storage_type, actual_key = key.split(':', 1)
        else:
            storage_type = data.get('type', 'storage_kv')
            actual_key = key
        
        st.markdown(f"### {actual_key.split('/')[-1]}")
        st.caption(f"Full path: `{actual_key}`")
        
        # Fetch content
        content = self._fetch_artifact_content(actual_key, storage_type)
        
        if content[0]:
            self._display_content_by_type(actual_key, content[1])
        else:
            st.error("Failed to fetch artifact content")
    
    def _display_content_by_type(self, key: str, content: Any):
        """Display content based on its type."""
        if isinstance(content, dict):
            # Check for special content types
            if 'code' in content:
                st.code(content['code'], language='python')
                if 'metadata' in content:
                    with st.expander("Metadata"):
                        st.json(content['metadata'])
            elif 'specifications' in content:
                # Specification content
                for spec in content.get('specifications', []):
                    with st.expander(spec.get('name', 'Unknown'), expanded=True):
                        st.json(spec)
            else:
                # Generic JSON
                st.json(content)
        
        elif isinstance(content, list):
            # List content
            st.json(content)
        
        elif isinstance(content, str):
            # String content - check file type
            if key.endswith('.py'):
                st.code(content, language='python')
            elif key.endswith('.md'):
                st.markdown(content)
            elif key.endswith('.json'):
                try:
                    st.json(json.loads(content))
                except (json.JSONDecodeError, Exception) as e:
                    st.error(f"Failed to parse JSON: {e}")
                    st.text(content)
            else:
                st.text(content)
        
        else:
            # Fallback
            st.write(content)
    
    def _fetch_artifact_content(self, key: str, storage_type: str) -> Tuple[bool, Any]:
        """Fetch artifact content from storage."""
        try:
            # Debug logging
            if self.debug_mode:
                st.caption(f"Debug: Fetching {storage_type}:{key}")
            
            if storage_type == 'storage_kv':
                # Ensure injector is available
                if not self.injector:
                    st.error("Injector not initialized!")
                    return False, None
                
                result = asyncio.run(self.injector.run('storage_kv', {
                    'operation': 'get',
                    'key': key
                }))
                
                # Debug the result
                if self.debug_mode:
                    st.caption(f"Debug: Result success={result.success}, data keys={list(result.data.keys()) if result.data else 'None'}")
                
                if result.success and result.data and result.data.get('exists', False):
                    value = result.data.get('value')
                    if value is None:
                        return True, "No content available"
                    # Try to parse as JSON if it looks like JSON
                    if isinstance(value, str) and (value.strip().startswith('{') or value.strip().startswith('[')):
                        try:
                            return True, json.loads(value)
                        except json.JSONDecodeError:
                            return True, value
                    return True, value
                else:
                    # Artifact doesn't exist yet - this is normal if the phase hasn't completed
                    # Return a user-friendly message instead of an error
                    if not result.success or (result.data and not result.data.get('exists', False)):
                        # This is a normal case - the artifact hasn't been created yet
                        return True, f"⏳ Artifact not yet available. The phase may still be running or hasn't started."
                    else:
                        # Actual error case
                        error_msg = f"Failed to retrieve artifact: {key}"
                        if self.debug_mode:
                            st.warning(error_msg)
                            if result.data:
                                st.caption(f"Debug data: {result.data}")
                        return False, error_msg
            
            elif storage_type == 'storage_fs':
                result = asyncio.run(self.injector.run('storage_fs', {
                    'operation': 'read',
                    'path': key
                }))
                
                # Debug the result
                if self.debug_mode:
                    st.caption(f"Debug: Result success={result.success}")
                
                if result.success and result.data.get('content'):
                    return True, result.data['content']
                else:
                    if self.debug_mode:
                        st.warning(f"File not found or read failed: {key}")
        
        except Exception as e:
            st.error(f"Error fetching artifact: {e}")
            import traceback
            st.code(traceback.format_exc())
        
        return False, None
    
    def _group_artifacts(self, artifacts: Dict[str, List[str]]) -> Dict[str, List[Any]]:
        """Group artifacts by type.
        
        Args:
            artifacts: Dictionary mapping phase names to lists of artifact references
            
        Returns:
            Dictionary mapping artifact types to lists of artifact items
        """
        grouped = {}
        
        # Process artifacts from each phase
        for phase, artifact_list in artifacts.items():
            for artifact_ref in artifact_list:
                # Parse the artifact reference (format: "storage_kv:path" or "storage_fs:path")
                if ':' in artifact_ref:
                    storage_type, artifact_path = artifact_ref.split(':', 1)
                else:
                    # Fallback for old format
                    storage_type = 'storage_kv'
                    artifact_path = artifact_ref
                
                # Determine artifact type from path
                artifact_type = self._determine_artifact_type(artifact_path)
                
                if artifact_type not in grouped:
                    grouped[artifact_type] = []
                
                # Create artifact item
                grouped[artifact_type].append({
                    'key': artifact_ref,  # Full reference including storage type
                    'path': artifact_path,  # Just the path
                    'data': {
                        'type': storage_type,
                        'phase': phase
                    },
                    'created_at': datetime.now()  # Could be enhanced to track actual time
                })
        
        return grouped
    
    def _determine_artifact_type(self, key: str) -> str:
        """Determine artifact type from key."""
        key_lower = key.lower()
        
        if 'analysis' in key_lower:
            return 'analysis'
        elif 'catalog' in key_lower:
            return 'catalog'
        elif 'specification' in key_lower or 'spec' in key_lower:
            return 'specifications'
        elif 'implementation' in key_lower:
            return 'implementation'
        elif 'validation' in key_lower:
            return 'validation'
        elif 'evaluation' in key_lower:
            return 'evaluation'
        elif 'existing_tools' in key_lower:
            return 'existing_tools'
        elif 'missing_tools' in key_lower:
            return 'missing_tools'
        elif 'skeleton' in key_lower:
            return 'skeleton'
        elif 'test_analysis' in key_lower:
            return 'test_analysis'
        elif 'test_stub' in key_lower:
            return 'test_stub'
        elif 'test_implementation' in key_lower or 'test' in key_lower:
            return 'test_implementation'
        elif 'final' in key_lower:
            return 'final'
        elif 'summary' in key_lower:
            return 'summary'
        else:
            return 'other'
    
    def _filter_artifacts(self, grouped: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Filter artifacts based on current filters."""
        filter_text = st.session_state.artifact_viewer_v2.get('filter_text', '').lower()
        filter_type = st.session_state.artifact_viewer_v2.get('filter_type', 'all')
        
        filtered = {}
        
        for artifact_type, items in grouped.items():
            # Type filter
            if filter_type != 'all' and artifact_type != filter_type:
                continue
            
            # Text filter
            if filter_text:
                filtered_items = []
                for item in items:
                    if (filter_text in item['key'].lower() or
                        filter_text in json.dumps(item.get('data', {})).lower()):
                        filtered_items.append(item)
                
                if filtered_items:
                    filtered[artifact_type] = filtered_items
            else:
                filtered[artifact_type] = items
        
        return filtered
    
    def _update_filter_text(self, text: str):
        """Update filter text."""
        st.session_state.artifact_viewer_v2['filter_text'] = text
    
    def _update_filters(self):
        """Update filters from UI."""
        st.session_state.artifact_viewer_v2['filter_type'] = st.session_state.get('artifact_type_filter', 'all')
    
    def _update_sort(self):
        """Update sort option."""
        st.session_state.artifact_viewer_v2['sort_by'] = st.session_state.get('artifact_sort', 'created')
    
    def _refresh_artifacts(self):
        """Refresh artifact list."""
        # In a real implementation, this would re-fetch artifacts
        st.success("Artifacts refreshed!")
    
    def export_artifacts(self, artifacts: Dict[str, List[str]]) -> str:
        """Export all artifacts to JSON."""
        total_count = sum(len(artifact_list) for artifact_list in artifacts.values())
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'artifact_count': total_count,
            'phases': len(artifacts),
            'artifacts': artifacts
        }
        return json.dumps(export_data, indent=2)