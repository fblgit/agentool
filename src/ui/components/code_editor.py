# -*- coding: utf-8 -*-
"""
Code editor component for displaying generated code with syntax highlighting.

This component provides a rich code editing and viewing experience
for the generated AgenTool implementations.
"""

import streamlit as st
from typing import Dict, Any, Optional, List
import json
import re


class CodeEditor:
    """
    Display and edit generated code with syntax highlighting.
    
    This component provides code viewing with proper formatting,
    line numbers, and export capabilities.
    """
    
    def __init__(self):
        """Initialize the code editor."""
        self.language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.md': 'markdown'
        }
    
    def render_code(self, code_data: Any, language: str = 'python', 
                   editable: bool = False, show_line_numbers: bool = True, key_suffix: str = ""):
        """
        Render code with syntax highlighting.
        
        Args:
            code_data: Code string or dict with code and metadata
            language: Programming language for syntax highlighting
            editable: Whether the code should be editable
            show_line_numbers: Whether to show line numbers
        """
        # Extract code from data structure
        if isinstance(code_data, dict):
            code = code_data.get('code', '')
            metadata = code_data.get('metadata', {})
        else:
            code = str(code_data)
            metadata = {}
        
        # Display metadata if available
        if metadata:
            with st.expander("📊 Code Metadata"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Lines of Code", metadata.get('loc', self._count_lines(code)))
                
                with col2:
                    st.metric("Functions", metadata.get('functions', self._count_functions(code)))
                
                with col3:
                    st.metric("Classes", metadata.get('classes', self._count_classes(code)))
        
        # Add line numbers if requested
        if show_line_numbers:
            code_with_numbers = self._add_line_numbers(code)
            display_code = code_with_numbers
        else:
            display_code = code
        
        # Display code
        if editable:
            edited_code = st.text_area(
                "Edit code:",
                value=code,
                height=500,
                key=f"code_editor_{key_suffix}" if key_suffix else "code_editor"
            )
            
            # Show diff if changed
            if edited_code != code:
                with st.expander("📝 Show Changes"):
                    self._show_diff(code, edited_code)
            
            return edited_code
        else:
            st.code(code, language=language)
            
            # Add copy button
            st.button("📋 Copy to Clipboard", 
                     on_click=lambda: st.write("Code copied!"),
                     key=f"copy_code_{key_suffix}" if key_suffix else "copy_code")
        
        return code
    
    def render_specification(self, spec_data: Dict[str, Any]):
        """
        Render a tool specification with proper formatting.
        
        Args:
            spec_data: Specification data dictionary
        """
        st.markdown("### Tool Specification")
        
        # Basic info
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Name:** `{spec_data.get('name', 'Unknown')}`")
            st.markdown(f"**Type:** {spec_data.get('type', 'Unknown')}")
            st.markdown(f"**Category:** {spec_data.get('category', 'General')}")
        
        with col2:
            dependencies = spec_data.get('dependencies', [])
            st.markdown(f"**Dependencies:** {len(dependencies)}")
            if dependencies:
                with st.expander("View Dependencies"):
                    for dep in dependencies:
                        st.markdown(f"- `{dep}`")
        
        # Description
        st.markdown("**Description:**")
        st.info(spec_data.get('description', 'No description provided'))
        
        # Schemas
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Input Schema:**")
            if 'input_schema' in spec_data:
                self._render_schema(spec_data['input_schema'])
        
        with col2:
            st.markdown("**Output Schema:**")
            if 'output_schema' in spec_data:
                self._render_schema(spec_data['output_schema'])
        
        # Implementation notes
        if 'implementation_notes' in spec_data:
            with st.expander("📝 Implementation Notes"):
                for note in spec_data['implementation_notes']:
                    st.markdown(f"- {note}")
    
    def _render_schema(self, schema: Dict[str, Any]):
        """Render a JSON schema in a readable format."""
        # Create a more readable schema display
        if 'properties' in schema:
            for prop_name, prop_def in schema['properties'].items():
                required = prop_name in schema.get('required', [])
                prop_type = prop_def.get('type', 'any')
                
                # Format property display
                if required:
                    st.markdown(f"- **{prop_name}** ({prop_type}) *required*")
                else:
                    st.markdown(f"- {prop_name} ({prop_type})")
                
                # Show description if available
                if 'description' in prop_def:
                    st.caption(f"  {prop_def['description']}")
                
                # Show enum values if available
                if 'enum' in prop_def:
                    st.caption(f"  Options: {', '.join(map(str, prop_def['enum']))}")
        else:
            # Fallback to JSON display
            st.json(schema)
    
    def _count_lines(self, code: str) -> int:
        """Count non-empty lines of code."""
        lines = code.strip().split('\n')
        return len([line for line in lines if line.strip()])
    
    def _count_functions(self, code: str) -> int:
        """Count function definitions in Python code."""
        # Simple regex for function definitions
        func_pattern = r'^\s*(?:async\s+)?def\s+\w+\s*\('
        return len(re.findall(func_pattern, code, re.MULTILINE))
    
    def _count_classes(self, code: str) -> int:
        """Count class definitions in Python code."""
        # Simple regex for class definitions
        class_pattern = r'^\s*class\s+\w+\s*(?:\(.*?\))?:'
        return len(re.findall(class_pattern, code, re.MULTILINE))
    
    def _add_line_numbers(self, code: str) -> str:
        """Add line numbers to code."""
        lines = code.split('\n')
        max_line_num = len(lines)
        line_num_width = len(str(max_line_num))
        
        numbered_lines = []
        for i, line in enumerate(lines, 1):
            line_num = str(i).rjust(line_num_width)
            numbered_lines.append(f"{line_num} | {line}")
        
        return '\n'.join(numbered_lines)
    
    def _show_diff(self, original: str, modified: str):
        """Show differences between original and modified code."""
        # Simple line-by-line diff
        original_lines = original.split('\n')
        modified_lines = modified.split('\n')
        
        max_lines = max(len(original_lines), len(modified_lines))
        
        for i in range(max_lines):
            orig_line = original_lines[i] if i < len(original_lines) else ''
            mod_line = modified_lines[i] if i < len(modified_lines) else ''
            
            if orig_line != mod_line:
                if orig_line and not mod_line:
                    st.markdown(f"🔴 Line {i+1}: `{orig_line}`")
                elif not orig_line and mod_line:
                    st.markdown(f"🟢 Line {i+1}: `{mod_line}`")
                else:
                    st.markdown(f"🟡 Line {i+1}:")
                    st.markdown(f"  - Old: `{orig_line}`")
                    st.markdown(f"  + New: `{mod_line}`")
    
    def create_file_viewer(self, files: Dict[str, str], show_metrics: bool = True):
        """
        Create a file viewer for multiple code files.
        
        Args:
            files: Dictionary of filename to code content
            show_metrics: Whether to show code metrics
        """
        if not files:
            st.info("No files to display")
            return
        
        # Show overall metrics if requested
        if show_metrics and len(files) > 1:
            col1, col2, col3 = st.columns(3)
            with col1:
                total_files = len(files)
                st.metric("Total Files", total_files)
            with col2:
                total_loc = sum(content.count('\n') for content in files.values())
                st.metric("Total Lines", total_loc)
            with col3:
                total_size = sum(len(content) for content in files.values())
                st.metric("Total Size", f"{total_size:,} bytes")
        
        # File selector
        selected_file = st.selectbox(
            "Select file:",
            options=list(files.keys()),
            format_func=lambda x: f"📄 {x}"
        )
        
        if selected_file:
            # Determine language from file extension
            ext = '.' + selected_file.split('.')[-1] if '.' in selected_file else ''
            language = self.language_map.get(ext, 'text')
            
            # Display file content
            self.render_code(
                files[selected_file],
                language=language,
                show_line_numbers=True,
                key_suffix=selected_file.replace('.', '_').replace('/', '_')
            )
            
            # Export buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label=f"💾 Download {selected_file}",
                    data=files[selected_file],
                    file_name=selected_file,
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col2:
                if len(files) > 1:
                    # TODO: Implement zip download
                    if st.button("📦 Download All Files", use_container_width=True):
                        st.info("Feature coming soon: Download all files as zip")
    
    def create_code_comparison(self, code1: str, code2: str, 
                             label1: str = "Original", label2: str = "Modified"):
        """
        Create a side-by-side code comparison.
        
        Args:
            code1: First code block
            code2: Second code block
            label1: Label for first code
            label2: Label for second code
        """
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### {label1}")
            st.code(code1, language='python')
        
        with col2:
            st.markdown(f"### {label2}")
            st.code(code2, language='python')
        
        # Show statistics
        with st.expander("📊 Comparison Statistics"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                lines1 = self._count_lines(code1)
                lines2 = self._count_lines(code2)
                diff_lines = lines2 - lines1
                st.metric("Lines", lines2, delta=diff_lines)
            
            with col2:
                funcs1 = self._count_functions(code1)
                funcs2 = self._count_functions(code2)
                diff_funcs = funcs2 - funcs1
                st.metric("Functions", funcs2, delta=diff_funcs)
            
            with col3:
                classes1 = self._count_classes(code1)
                classes2 = self._count_classes(code2)
                diff_classes = classes2 - classes1
                st.metric("Classes", classes2, delta=diff_classes)