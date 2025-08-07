# -*- coding: utf-8 -*-
"""
Theme Manager - Comprehensive visual design system for the workflow UI.

This component provides a centralized theming system with color palettes,
typography, spacing, animations, and component styles.
"""

import streamlit as st
import threading
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ThemeMode(Enum):
    """Available theme modes."""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"


class ColorScheme(Enum):
    """Available color schemes."""
    DEFAULT = "default"
    OCEAN = "ocean"
    FOREST = "forest"
    SUNSET = "sunset"
    MONOCHROME = "monochrome"


@dataclass
class ColorPalette:
    """Color palette definition."""
    # Primary colors
    primary: str
    primary_light: str
    primary_dark: str
    
    # Secondary colors
    secondary: str
    secondary_light: str
    secondary_dark: str
    
    # Semantic colors
    success: str
    success_light: str
    warning: str
    warning_light: str
    error: str
    error_light: str
    info: str
    info_light: str
    
    # Backgrounds
    bg_primary: str
    bg_secondary: str
    bg_tertiary: str
    bg_card: str
    bg_hover: str
    
    # Borders and dividers
    border: str
    border_light: str
    divider: str
    
    # Text colors
    text_primary: str
    text_secondary: str
    text_disabled: str
    text_inverse: str
    
    # Phase-specific colors
    phase_pending: str
    phase_running: str
    phase_complete: str
    phase_error: str
    phase_skipped: str
    
    # Artifact type colors
    artifact_kv: str
    artifact_fs: str
    artifact_spec: str
    artifact_code: str
    artifact_test: str
    artifact_docs: str


@dataclass
class Typography:
    """Typography settings."""
    font_family: str = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    font_family_mono: str = "'Fira Code', 'Monaco', 'Courier New', monospace"
    
    # Font sizes
    size_xs: str = "0.75rem"
    size_sm: str = "0.875rem"
    size_base: str = "1rem"
    size_lg: str = "1.125rem"
    size_xl: str = "1.25rem"
    size_2xl: str = "1.5rem"
    size_3xl: str = "1.875rem"
    size_4xl: str = "2.25rem"
    
    # Font weights
    weight_light: int = 300
    weight_normal: int = 400
    weight_medium: int = 500
    weight_semibold: int = 600
    weight_bold: int = 700
    
    # Line heights
    leading_none: float = 1.0
    leading_tight: float = 1.25
    leading_normal: float = 1.5
    leading_relaxed: float = 1.75
    leading_loose: float = 2.0


@dataclass
class Spacing:
    """Spacing system."""
    xs: str = "0.25rem"
    sm: str = "0.5rem"
    md: str = "1rem"
    lg: str = "1.5rem"
    xl: str = "2rem"
    xxl: str = "3rem"
    xxxl: str = "4rem"


@dataclass
class Animation:
    """Animation settings."""
    duration_fast: str = "150ms"
    duration_normal: str = "300ms"
    duration_slow: str = "500ms"
    duration_slower: str = "1000ms"
    
    easing_default: str = "cubic-bezier(0.4, 0, 0.2, 1)"
    easing_ease_in: str = "cubic-bezier(0.4, 0, 1, 1)"
    easing_ease_out: str = "cubic-bezier(0, 0, 0.2, 1)"
    easing_ease_in_out: str = "cubic-bezier(0.4, 0, 0.2, 1)"
    easing_spring: str = "cubic-bezier(0.175, 0.885, 0.32, 1.275)"


class ThemeManager:
    """
    Manages the visual theme for the workflow UI.
    
    This class provides a comprehensive theming system including colors,
    typography, spacing, and component styles.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    # Predefined color palettes
    PALETTES = {
        ThemeMode.DARK: {
            ColorScheme.DEFAULT: ColorPalette(
                # Primary colors
                primary="#6366F1",
                primary_light="#818CF8",
                primary_dark="#4F46E5",
                
                # Secondary colors
                secondary="#8B5CF6",
                secondary_light="#A78BFA",
                secondary_dark="#7C3AED",
                
                # Semantic colors
                success="#10B981",
                success_light="#34D399",
                warning="#F59E0B",
                warning_light="#FCD34D",
                error="#EF4444",
                error_light="#F87171",
                info="#3B82F6",
                info_light="#60A5FA",
                
                # Backgrounds
                bg_primary="#0F172A",
                bg_secondary="#1E293B",
                bg_tertiary="#334155",
                bg_card="rgba(30, 41, 59, 0.5)",
                bg_hover="rgba(99, 102, 241, 0.1)",
                
                # Borders and dividers
                border="#475569",
                border_light="#64748B",
                divider="#334155",
                
                # Text colors
                text_primary="#F1F5F9",
                text_secondary="#CBD5E1",
                text_disabled="#64748B",
                text_inverse="#0F172A",
                
                # Phase-specific colors
                phase_pending="#64748B",
                phase_running="#6366F1",
                phase_complete="#10B981",
                phase_error="#EF4444",
                phase_skipped="#94A3B8",
                
                # Artifact type colors
                artifact_kv="#EC4899",
                artifact_fs="#14B8A6",
                artifact_spec="#8B5CF6",
                artifact_code="#F59E0B",
                artifact_test="#06B6D4",
                artifact_docs="#10B981"
            ),
            ColorScheme.OCEAN: ColorPalette(
                primary="#0EA5E9",
                primary_light="#38BDF8",
                primary_dark="#0284C7",
                secondary="#06B6D4",
                secondary_light="#22D3EE",
                secondary_dark="#0891B2",
                success="#10B981",
                success_light="#34D399",
                warning="#F59E0B",
                warning_light="#FCD34D",
                error="#EF4444",
                error_light="#F87171",
                info="#3B82F6",
                info_light="#60A5FA",
                bg_primary="#082F49",
                bg_secondary="#0C4A6E",
                bg_tertiary="#075985",
                bg_card="rgba(12, 74, 110, 0.5)",
                bg_hover="rgba(14, 165, 233, 0.1)",
                border="#0284C7",
                border_light="#0EA5E9",
                divider="#075985",
                text_primary="#F0F9FF",
                text_secondary="#BAE6FD",
                text_disabled="#7DD3FC",
                text_inverse="#082F49",
                phase_pending="#64748B",
                phase_running="#0EA5E9",
                phase_complete="#10B981",
                phase_error="#EF4444",
                phase_skipped="#94A3B8",
                artifact_kv="#EC4899",
                artifact_fs="#14B8A6",
                artifact_spec="#8B5CF6",
                artifact_code="#F59E0B",
                artifact_test="#06B6D4",
                artifact_docs="#10B981"
            )
        },
        ThemeMode.LIGHT: {
            ColorScheme.DEFAULT: ColorPalette(
                primary="#4F46E5",
                primary_light="#6366F1",
                primary_dark="#4338CA",
                secondary="#7C3AED",
                secondary_light="#8B5CF6",
                secondary_dark="#6D28D9",
                success="#059669",
                success_light="#10B981",
                warning="#D97706",
                warning_light="#F59E0B",
                error="#DC2626",
                error_light="#EF4444",
                info="#2563EB",
                info_light="#3B82F6",
                bg_primary="#FFFFFF",
                bg_secondary="#F9FAFB",
                bg_tertiary="#F3F4F6",
                bg_card="rgba(255, 255, 255, 0.8)",
                bg_hover="rgba(79, 70, 229, 0.05)",
                border="#E5E7EB",
                border_light="#D1D5DB",
                divider="#E5E7EB",
                text_primary="#111827",
                text_secondary="#4B5563",
                text_disabled="#9CA3AF",
                text_inverse="#FFFFFF",
                phase_pending="#9CA3AF",
                phase_running="#4F46E5",
                phase_complete="#059669",
                phase_error="#DC2626",
                phase_skipped="#D1D5DB",
                artifact_kv="#DB2777",
                artifact_fs="#0D9488",
                artifact_spec="#7C3AED",
                artifact_code="#D97706",
                artifact_test="#0891B2",
                artifact_docs="#059669"
            )
        }
    }
    
    def __new__(cls, *args, **kwargs):
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, mode: ThemeMode = ThemeMode.DARK, 
                 scheme: ColorScheme = ColorScheme.DEFAULT):
        """
        Initialize the theme manager.
        
        Args:
            mode: Theme mode (light/dark/auto)
            scheme: Color scheme to use
        """
        # Only initialize once
        if not hasattr(self, '_initialized_singleton'):
            self.mode = mode
            self.scheme = scheme
            self.palette = self._get_palette()
            self.typography = Typography()
            self.spacing = Spacing()
            self.animation = Animation()
            self._initialized_singleton = True
            
            # Initialize in session state
            if 'theme_manager' not in st.session_state:
                st.session_state.theme_manager = {
                    'mode': mode.value,
                    'scheme': scheme.value,
                    'custom_css_applied': False
            }
    
    def _get_palette(self) -> ColorPalette:
        """Get the appropriate color palette based on mode and scheme."""
        # For auto mode, detect system preference (default to dark for now)
        if self.mode == ThemeMode.AUTO:
            # In a real implementation, this would detect system preference
            effective_mode = ThemeMode.DARK
        else:
            effective_mode = self.mode
        
        return self.PALETTES[effective_mode].get(
            self.scheme, 
            self.PALETTES[effective_mode][ColorScheme.DEFAULT]
        )
    
    def apply_theme(self):
        """Apply the theme to the Streamlit app."""
        # Ensure session state is initialized
        if 'theme_manager' not in st.session_state:
            st.session_state.theme_manager = {
                'mode': self.mode.value,
                'scheme': self.scheme.value,
                'custom_css_applied': False
            }
        
        if not st.session_state.theme_manager.get('custom_css_applied', False):
            st.markdown(self.generate_css(), unsafe_allow_html=True)
            st.session_state.theme_manager['custom_css_applied'] = True
    
    def generate_css(self) -> str:
        """Generate CSS styles based on the current theme."""
        return f"""
        <style>
        /* Root variables */
        :root {{
            /* Colors */
            --primary: {self.palette.primary};
            --primary-light: {self.palette.primary_light};
            --primary-dark: {self.palette.primary_dark};
            --secondary: {self.palette.secondary};
            --secondary-light: {self.palette.secondary_light};
            --secondary-dark: {self.palette.secondary_dark};
            --success: {self.palette.success};
            --success-light: {self.palette.success_light};
            --warning: {self.palette.warning};
            --warning-light: {self.palette.warning_light};
            --error: {self.palette.error};
            --error-light: {self.palette.error_light};
            --info: {self.palette.info};
            --info-light: {self.palette.info_light};
            
            /* Backgrounds */
            --bg-primary: {self.palette.bg_primary};
            --bg-secondary: {self.palette.bg_secondary};
            --bg-tertiary: {self.palette.bg_tertiary};
            --bg-card: {self.palette.bg_card};
            --bg-hover: {self.palette.bg_hover};
            
            /* Borders */
            --border: {self.palette.border};
            --border-light: {self.palette.border_light};
            --divider: {self.palette.divider};
            
            /* Text */
            --text-primary: {self.palette.text_primary};
            --text-secondary: {self.palette.text_secondary};
            --text-disabled: {self.palette.text_disabled};
            --text-inverse: {self.palette.text_inverse};
            
            /* Typography */
            --font-family: {self.typography.font_family};
            --font-family-mono: {self.typography.font_family_mono};
            
            /* Spacing */
            --spacing-xs: {self.spacing.xs};
            --spacing-sm: {self.spacing.sm};
            --spacing-md: {self.spacing.md};
            --spacing-lg: {self.spacing.lg};
            --spacing-xl: {self.spacing.xl};
            
            /* Animation */
            --duration-fast: {self.animation.duration_fast};
            --duration-normal: {self.animation.duration_normal};
            --duration-slow: {self.animation.duration_slow};
            --easing-default: {self.animation.easing_default};
        }}
        
        /* Glass morphism cards */
        .glass-card {{
            background: {self.palette.bg_card};
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid {self.palette.border};
            border-radius: 16px;
            padding: 1.5rem;
            transition: all var(--duration-normal) var(--easing-default);
        }}
        
        .glass-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 40px rgba(99, 102, 241, 0.2);
            border-color: {self.palette.primary};
        }}
        
        /* Phase cards */
        .phase-card {{
            background: {self.palette.bg_card};
            backdrop-filter: blur(8px);
            border: 2px solid transparent;
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            transition: all var(--duration-normal) var(--easing-default);
        }}
        
        .phase-pending {{
            background: linear-gradient(135deg, {self.palette.bg_secondary}, {self.palette.bg_tertiary});
            border-color: {self.palette.phase_pending};
        }}
        
        .phase-running {{
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1));
            border-color: {self.palette.phase_running};
            animation: pulse 2s ease-in-out infinite;
        }}
        
        .phase-complete {{
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(34, 197, 94, 0.1));
            border-color: {self.palette.phase_complete};
        }}
        
        .phase-error {{
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(248, 113, 113, 0.1));
            border-color: {self.palette.phase_error};
        }}
        
        /* Animated progress bars */
        .progress-bar {{
            background: linear-gradient(90deg, {self.palette.primary}, {self.palette.secondary});
            border-radius: 4px;
            height: 8px;
            position: relative;
            overflow: hidden;
        }}
        
        .progress-bar::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
            animation: shimmer 2s infinite;
        }}
        
        /* Live feed styling */
        .live-feed-item {{
            padding: 0.75rem;
            margin: 0.5rem 0;
            background: {self.palette.bg_hover};
            border-left: 3px solid {self.palette.primary};
            border-radius: 4px;
            animation: slideIn var(--duration-normal) var(--easing-default);
        }}
        
        /* Artifact badges */
        .artifact-badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 600;
            margin: 0.25rem;
        }}
        
        .artifact-kv {{
            background: linear-gradient(135deg, {self.palette.artifact_kv}, {self.palette.artifact_kv}88);
            color: white;
        }}
        
        .artifact-fs {{
            background: linear-gradient(135deg, {self.palette.artifact_fs}, {self.palette.artifact_fs}88);
            color: white;
        }}
        
        .artifact-spec {{
            background: linear-gradient(135deg, {self.palette.artifact_spec}, {self.palette.artifact_spec}88);
            color: white;
        }}
        
        .artifact-code {{
            background: linear-gradient(135deg, {self.palette.artifact_code}, {self.palette.artifact_code}88);
            color: white;
        }}
        
        /* Status badges */
        .status-badge {{
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 600;
            display: inline-block;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        .status-running {{
            background: linear-gradient(90deg, {self.palette.primary}, {self.palette.secondary});
            color: white;
            animation: pulse 2s infinite;
        }}
        
        .status-complete {{
            background: {self.palette.success};
            color: white;
        }}
        
        .status-error {{
            background: {self.palette.error};
            color: white;
        }}
        
        /* Animations */
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.7; }}
        }}
        
        @keyframes slideIn {{
            from {{
                transform: translateX(-20px);
                opacity: 0;
            }}
            to {{
                transform: translateX(0);
                opacity: 1;
            }}
        }}
        
        @keyframes shimmer {{
            100% {{ left: 100%; }}
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: {self.palette.bg_secondary};
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: {self.palette.border};
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: {self.palette.border_light};
        }}
        
        /* Modal styling */
        .modal-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            backdrop-filter: blur(5px);
            z-index: 1000;
            animation: fadeIn var(--duration-fast) var(--easing-default);
        }}
        
        .modal-content {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: {self.palette.bg_secondary};
            border: 1px solid {self.palette.border};
            border-radius: 16px;
            padding: 2rem;
            max-width: 80%;
            max-height: 80%;
            overflow-y: auto;
            z-index: 1001;
            animation: slideIn var(--duration-normal) var(--easing-spring);
        }}
        
        /* Code blocks */
        .code-block {{
            background: {self.palette.bg_primary};
            border: 1px solid {self.palette.border};
            border-radius: 8px;
            padding: 1rem;
            font-family: var(--font-family-mono);
            overflow-x: auto;
        }}
        
        /* Tooltips */
        .tooltip {{
            position: relative;
            display: inline-block;
        }}
        
        .tooltip-text {{
            visibility: hidden;
            background: {self.palette.bg_primary};
            color: {self.palette.text_primary};
            text-align: center;
            border: 1px solid {self.palette.border};
            border-radius: 6px;
            padding: 0.5rem;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity var(--duration-fast) var(--easing-default);
        }}
        
        .tooltip:hover .tooltip-text {{
            visibility: visible;
            opacity: 1;
        }}
        </style>
        """
    
    def get_component_style(self, component: str) -> Dict[str, Any]:
        """
        Get style dictionary for a specific component.
        
        Args:
            component: Component name
            
        Returns:
            Style dictionary
        """
        styles = {
            "phase_card": {
                "background": self.palette.bg_card,
                "border": f"2px solid {self.palette.border}",
                "borderRadius": "12px",
                "padding": "1.25rem",
                "marginBottom": "1rem"
            },
            "artifact_viewer": {
                "background": self.palette.bg_secondary,
                "border": f"1px solid {self.palette.border}",
                "borderRadius": "8px",
                "padding": "1rem"
            },
            "live_feed": {
                "background": self.palette.bg_primary,
                "border": f"1px solid {self.palette.border}",
                "borderRadius": "8px",
                "padding": "0.5rem",
                "maxHeight": "400px",
                "overflowY": "auto"
            },
            "button_primary": {
                "background": f"linear-gradient(90deg, {self.palette.primary}, {self.palette.secondary})",
                "color": "white",
                "border": "none",
                "borderRadius": "8px",
                "padding": "0.5rem 1rem",
                "fontWeight": "600",
                "cursor": "pointer"
            },
            "button_secondary": {
                "background": self.palette.bg_tertiary,
                "color": self.palette.text_primary,
                "border": f"1px solid {self.palette.border}",
                "borderRadius": "8px",
                "padding": "0.5rem 1rem",
                "fontWeight": "600",
                "cursor": "pointer"
            }
        }
        
        return styles.get(component, {})
    
    def get_phase_color(self, status: str) -> str:
        """Get color for a phase status."""
        status_colors = {
            "pending": self.palette.phase_pending,
            "running": self.palette.phase_running,
            "complete": self.palette.phase_complete,
            "completed": self.palette.phase_complete,
            "error": self.palette.phase_error,
            "failed": self.palette.phase_error,
            "skipped": self.palette.phase_skipped
        }
        return status_colors.get(status, self.palette.text_secondary)
    
    def get_artifact_color(self, artifact_type: str) -> str:
        """Get color for an artifact type."""
        type_colors = {
            "kv": self.palette.artifact_kv,
            "storage_kv": self.palette.artifact_kv,
            "fs": self.palette.artifact_fs,
            "storage_fs": self.palette.artifact_fs,
            "spec": self.palette.artifact_spec,
            "specification": self.palette.artifact_spec,
            "code": self.palette.artifact_code,
            "implementation": self.palette.artifact_code,
            "test": self.palette.artifact_test,
            "docs": self.palette.artifact_docs,
            "documentation": self.palette.artifact_docs
        }
        return type_colors.get(artifact_type, self.palette.text_secondary)


def get_theme_manager(mode: Optional[ThemeMode] = None, 
                     scheme: Optional[ColorScheme] = None) -> ThemeManager:
    """
    Get the theme manager instance for the current session.
    
    Args:
        mode: Optional theme mode override
        scheme: Optional color scheme override
        
    Returns:
        The ThemeManager instance
    """
    # Store instance in session state to ensure it's session-specific
    if 'theme_manager_instance' not in st.session_state or mode or scheme:
        st.session_state.theme_manager_instance = ThemeManager(
            mode or ThemeMode.DARK,
            scheme or ColorScheme.DEFAULT
        )
    
    return st.session_state.theme_manager_instance