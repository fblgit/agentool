# -*- coding: utf-8 -*-
"""
Styling Utilities - CSS and styling helpers for the workflow UI.

This module provides utility functions for applying custom styles,
animations, and visual effects to Streamlit components.
"""

import streamlit as st
from typing import Dict, Any, Optional, List, Tuple


def apply_custom_css():
    """Apply comprehensive custom CSS to the Streamlit app."""
    css = """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500;600&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Code blocks */
    pre, code {
        font-family: 'Fira Code', 'Monaco', monospace !important;
    }
    
    /* Fix button text overflow and sizing */
    .stButton > button {
        white-space: normal !important;
        word-wrap: break-word !important;
        height: auto !important;
        min-height: 38px !important;
        padding: 0.5rem 1rem !important;
        font-size: 0.875rem !important;
    }
    
    /* Fix selectbox text */
    .stSelectbox label {
        font-size: 0.875rem !important;
    }
    
    .stSelectbox > div > div {
        font-size: 0.875rem !important;
    }
    
    /* Fix radio button text */
    .stRadio label {
        font-size: 0.875rem !important;
    }
    
    .stRadio > div {
        gap: 0.5rem !important;
    }
    
    /* Fix checkbox text */
    .stCheckbox label {
        font-size: 0.875rem !important;
    }
    
    /* Fix text area */
    .stTextArea textarea {
        font-size: 0.875rem !important;
        min-height: 150px !important;
    }
    
    /* Fix columns spacing for pipeline visualization */
    .stColumns {
        gap: 1rem !important;
    }
    
    /* Fix metric cards in pipeline */
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem !important;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stMetric label {
        font-size: 0.75rem !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    
    .stMetric > div[data-testid="metric-container"] > div {
        font-size: 1.25rem !important;
    }
    
    /* Fix expander text */
    .stExpander summary {
        font-size: 0.875rem !important;
    }
    
    .stExpander div[data-testid="stExpanderContent"] {
        font-size: 0.875rem !important;
    }
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.1);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(99, 102, 241, 0.5);
        border-radius: 5px;
        transition: background 0.3s;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(99, 102, 241, 0.8);
    }
    
    /* Animated gradient background */
    .gradient-bg {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Pulse animation */
    @keyframes pulse {
        0%, 100% { 
            opacity: 1;
            transform: scale(1);
        }
        50% { 
            opacity: 0.8;
            transform: scale(0.98);
        }
    }
    
    /* Slide animations */
    @keyframes slideInLeft {
        from {
            transform: translateX(-100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideInUp {
        from {
            transform: translateY(100%);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    /* Fade animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes fadeOut {
        from { opacity: 1; }
        to { opacity: 0; }
    }
    
    /* Bounce animation */
    @keyframes bounce {
        0%, 100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-10px);
        }
    }
    
    /* Rotate animation */
    @keyframes rotate {
        from {
            transform: rotate(0deg);
        }
        to {
            transform: rotate(360deg);
        }
    }
    
    /* Loading spinner */
    .spinner {
        width: 40px;
        height: 40px;
        border: 4px solid rgba(99, 102, 241, 0.2);
        border-top-color: #6366F1;
        border-radius: 50%;
        animation: rotate 1s linear infinite;
    }
    
    /* Glow effect */
    .glow {
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from {
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.5);
        }
        to {
            box-shadow: 0 0 30px rgba(99, 102, 241, 0.8);
        }
    }
    
    /* Success animation */
    .success-animation {
        animation: successPulse 0.5s ease;
    }
    
    @keyframes successPulse {
        0% {
            transform: scale(0.8);
            opacity: 0;
        }
        50% {
            transform: scale(1.1);
        }
        100% {
            transform: scale(1);
            opacity: 1;
        }
    }
    
    /* Error shake */
    .error-shake {
        animation: shake 0.5s;
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
        20%, 40%, 60%, 80% { transform: translateX(5px); }
    }
    
    /* Card hover effects */
    .hover-lift {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .hover-lift:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
    }
    
    /* Button styles */
    .custom-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .custom-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Tooltip styles */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: rgba(0, 0, 0, 0.9);
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.875rem;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Progress bar styles */
    .custom-progress {
        width: 100%;
        height: 8px;
        background: rgba(99, 102, 241, 0.1);
        border-radius: 4px;
        overflow: hidden;
        position: relative;
    }
    
    .custom-progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 4px;
        position: relative;
        transition: width 0.3s ease;
    }
    
    .custom-progress-bar::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255, 255, 255, 0.3),
            transparent
        );
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    /* Notification styles */
    .notification {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        animation: slideInRight 0.3s ease;
    }
    
    .notification-success {
        background: linear-gradient(135deg, #10B981 0%, #34D399 100%);
        color: white;
    }
    
    .notification-error {
        background: linear-gradient(135deg, #EF4444 0%, #F87171 100%);
        color: white;
    }
    
    .notification-warning {
        background: linear-gradient(135deg, #F59E0B 0%, #FCD34D 100%);
        color: white;
    }
    
    .notification-info {
        background: linear-gradient(135deg, #3B82F6 0%, #60A5FA 100%);
        color: white;
    }
    
    /* Responsive design helpers */
    @media (max-width: 768px) {
        .hide-mobile {
            display: none !important;
        }
    }
    
    @media (min-width: 769px) {
        .hide-desktop {
            display: none !important;
        }
    }
    
    /* Focus styles */
    *:focus {
        outline: 2px solid rgba(99, 102, 241, 0.5);
        outline-offset: 2px;
    }
    
    /* Selection styles */
    ::selection {
        background: rgba(99, 102, 241, 0.3);
        color: inherit;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def create_animated_header(text: str, subtitle: Optional[str] = None):
    """
    Create an animated header with optional subtitle.
    
    Args:
        text: Main header text
        subtitle: Optional subtitle text
    """
    html = f"""<div style="animation: fadeIn 0.5s ease;">
    <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">{text}</h1>"""
    
    if subtitle:
        html += f"""
    <p style="color: rgba(99, 102, 241, 0.8); font-size: 1.125rem; margin-top: 0;">{subtitle}</p>"""
    
    html += """
</div>"""
    
    st.markdown(html, unsafe_allow_html=True)


def create_metric_card(label: str, value: Any, delta: Optional[Any] = None,
                       color: str = "primary", icon: Optional[str] = None):
    """
    Create a styled metric card.
    
    Args:
        label: Metric label
        value: Metric value
        delta: Optional delta value
        color: Color theme (primary, success, warning, error)
        icon: Optional icon emoji
    """
    color_map = {
        "primary": "#6366F1",
        "success": "#10B981",
        "warning": "#F59E0B",
        "error": "#EF4444",
        "info": "#3B82F6"
    }
    
    bg_color = color_map.get(color, color_map["primary"])
    
    html = f"""
    <div class="hover-lift" style="
        background: linear-gradient(135deg, {bg_color}22 0%, {bg_color}11 100%);
        border: 1px solid {bg_color}44;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.5rem 0;
    ">
    """
    
    if icon:
        html += f'<div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>'
    
    html += f"""
        <div style="color: #64748B; font-size: 0.875rem; margin-bottom: 0.25rem;">
            {label}
        </div>
        <div style="font-size: 1.75rem; font-weight: 600; color: {bg_color};">
            {value}
        </div>
    """
    
    if delta is not None:
        delta_color = "#10B981" if str(delta).startswith("+") else "#EF4444"
        html += f"""
        <div style="color: {delta_color}; font-size: 0.875rem; margin-top: 0.25rem;">
            {delta}
        </div>
        """
    
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def create_progress_ring(percentage: float, size: int = 120, 
                         stroke_width: int = 8, color: str = "#6366F1"):
    """
    Create a circular progress ring.
    
    Args:
        percentage: Progress percentage (0-100)
        size: Size of the ring in pixels
        stroke_width: Width of the ring stroke
        color: Ring color
    """
    radius = (size - stroke_width) / 2
    circumference = radius * 2 * 3.14159
    offset = circumference - (percentage / 100 * circumference)
    
    svg = f"""
    <svg width="{size}" height="{size}" style="transform: rotate(-90deg);">
        <circle
            cx="{size/2}"
            cy="{size/2}"
            r="{radius}"
            stroke="rgba(99, 102, 241, 0.1)"
            stroke-width="{stroke_width}"
            fill="none"
        />
        <circle
            cx="{size/2}"
            cy="{size/2}"
            r="{radius}"
            stroke="{color}"
            stroke-width="{stroke_width}"
            fill="none"
            stroke-dasharray="{circumference}"
            stroke-dashoffset="{offset}"
            stroke-linecap="round"
            style="transition: stroke-dashoffset 0.3s ease;"
        />
    </svg>
    <div style="
        position: relative;
        top: -{size - 20}px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
        color: {color};
    ">{percentage:.0f}%</div>
    """
    
    st.markdown(f'<div style="width: {size}px; margin: 0 auto;">{svg}</div>', 
                unsafe_allow_html=True)


def create_badge(text: str, color: str = "primary", icon: Optional[str] = None):
    """
    Create a styled badge.
    
    Args:
        text: Badge text
        color: Color theme
        icon: Optional icon
    """
    color_map = {
        "primary": "#6366F1",
        "success": "#10B981",
        "warning": "#F59E0B",
        "error": "#EF4444",
        "info": "#3B82F6",
        "secondary": "#8B5CF6"
    }
    
    bg_color = color_map.get(color, color_map["primary"])
    
    icon_html = f"{icon} " if icon else ""
    
    html = f"""
    <span style="
        display: inline-block;
        background: {bg_color};
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.125rem;
    ">{icon_html}{text}</span>
    """
    
    st.markdown(html, unsafe_allow_html=True)


def create_divider(style: str = "solid", color: str = "rgba(99, 102, 241, 0.2)",
                  margin: str = "1rem 0"):
    """
    Create a styled divider.
    
    Args:
        style: Line style (solid, dashed, dotted)
        color: Line color
        margin: Margin spacing
    """
    html = f"""
    <hr style="
        border: none;
        border-top: 2px {style} {color};
        margin: {margin};
    ">
    """
    st.markdown(html, unsafe_allow_html=True)


def create_alert(message: str, alert_type: str = "info", 
                icon: Optional[str] = None, dismissible: bool = False):
    """
    Create a styled alert box.
    
    Args:
        message: Alert message
        alert_type: Type of alert (info, success, warning, error)
        icon: Optional icon
        dismissible: Whether alert can be dismissed
    """
    type_config = {
        "info": {"bg": "#3B82F6", "icon": "ℹ️"},
        "success": {"bg": "#10B981", "icon": "✅"},
        "warning": {"bg": "#F59E0B", "icon": "⚠️"},
        "error": {"bg": "#EF4444", "icon": "❌"}
    }
    
    config = type_config.get(alert_type, type_config["info"])
    icon = icon or config["icon"]
    
    dismiss_button = """
        <button onclick="this.parentElement.style.display='none'" style="
            background: none;
            border: none;
            color: white;
            font-size: 1.25rem;
            cursor: pointer;
            float: right;
            margin-left: 1rem;
        ">×</button>
    """ if dismissible else ""
    
    html = f"""
    <div class="notification notification-{alert_type}" style="
        background: linear-gradient(135deg, {config['bg']} 0%, {config['bg']}dd 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        animation: slideInRight 0.3s ease;
    ">
        <span style="font-size: 1.5rem; margin-right: 0.75rem;">{icon}</span>
        <span style="flex: 1;">{message}</span>
        {dismiss_button}
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)


def create_loading_animation(text: str = "Loading..."):
    """
    Create a custom loading animation.
    
    Args:
        text: Loading text
    """
    html = f"""
    <div style="text-align: center; padding: 2rem;">
        <div class="spinner" style="margin: 0 auto;"></div>
        <p style="margin-top: 1rem; color: #6366F1; font-weight: 500;">
            {text}
        </p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def apply_responsive_layout():
    """Apply responsive layout styles."""
    css = """
    <style>
    /* Responsive containers */
    @media (max-width: 768px) {
        .stColumns > div {
            flex: 100% !important;
            max-width: 100% !important;
        }
        
        .stButton > button {
            width: 100% !important;
        }
        
        h1 { font-size: 1.75rem !important; }
        h2 { font-size: 1.5rem !important; }
        h3 { font-size: 1.25rem !important; }
    }
    
    /* Tablet layout */
    @media (min-width: 769px) and (max-width: 1024px) {
        .stColumns > div {
            flex: 50% !important;
        }
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def create_step_indicator(steps: List[str], current_step: int):
    """
    Create a step indicator.
    
    Args:
        steps: List of step names
        current_step: Current step index (0-based)
    """
    html = '<div style="display: flex; justify-content: space-between; margin: 2rem 0;">'
    
    for i, step in enumerate(steps):
        is_active = i == current_step
        is_completed = i < current_step
        
        if is_completed:
            bg_color = "#10B981"
            icon = "✓"
        elif is_active:
            bg_color = "#6366F1"
            icon = str(i + 1)
        else:
            bg_color = "#E5E7EB"
            icon = str(i + 1)
        
        html += f"""
        <div style="text-align: center; flex: 1;">
            <div style="
                width: 40px;
                height: 40px;
                background: {bg_color};
                color: white;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 0.5rem;
                font-weight: 600;
                {('animation: pulse 2s infinite;' if is_active else '')}
            ">{icon}</div>
            <div style="
                font-size: 0.875rem;
                color: {'#6366F1' if is_active else '#64748B'};
                font-weight: {'600' if is_active else '400'};
            ">{step}</div>
        </div>
        """
        
        # Add connector line
        if i < len(steps) - 1:
            html += f"""
            <div style="
                flex: 1;
                height: 2px;
                background: {'#10B981' if is_completed else '#E5E7EB'};
                margin-top: 20px;
            "></div>
            """
    
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)