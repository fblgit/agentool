# -*- coding: utf-8 -*-
"""
Live Feed Component - Real-time activity monitoring for workflow execution.

This component provides a live, auto-updating feed of workflow activities,
phase updates, and artifact creation events using Streamlit fragments.
"""

import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import time
import collections

from .container_manager import get_container_manager
from .theme_manager import get_theme_manager


class ActivityType(Enum):
    """Types of activities in the feed."""
    PHASE_START = "phase_start"
    PHASE_COMPLETE = "phase_complete"
    PHASE_ERROR = "phase_error"
    ARTIFACT_CREATED = "artifact_created"
    TOOL_PROCESSING = "tool_processing"
    VALIDATION = "validation"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


@dataclass
class ActivityItem:
    """Single activity item in the feed."""
    timestamp: datetime
    activity_type: ActivityType
    phase: Optional[str]
    title: str
    description: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    icon: Optional[str] = None
    color: Optional[str] = None


class LiveFeed:
    """
    Real-time activity feed for workflow execution.
    
    This component displays a continuously updating feed of workflow
    activities with automatic scrolling and filtering capabilities.
    """
    
    # Activity type configurations
    ACTIVITY_CONFIG = {
        ActivityType.PHASE_START: {
            'icon': '▶️',
            'color': 'primary',
            'prefix': 'Starting'
        },
        ActivityType.PHASE_COMPLETE: {
            'icon': '✅',
            'color': 'success',
            'prefix': 'Completed'
        },
        ActivityType.PHASE_ERROR: {
            'icon': '❌',
            'color': 'error',
            'prefix': 'Failed'
        },
        ActivityType.ARTIFACT_CREATED: {
            'icon': '📦',
            'color': 'info',
            'prefix': 'Created'
        },
        ActivityType.TOOL_PROCESSING: {
            'icon': '🔧',
            'color': 'warning',
            'prefix': 'Processing'
        },
        ActivityType.VALIDATION: {
            'icon': '✓',
            'color': 'success',
            'prefix': 'Validated'
        },
        ActivityType.INFO: {
            'icon': 'ℹ️',
            'color': 'info',
            'prefix': 'Info'
        },
        ActivityType.WARNING: {
            'icon': '⚠️',
            'color': 'warning',
            'prefix': 'Warning'
        },
        ActivityType.ERROR: {
            'icon': '🔴',
            'color': 'error',
            'prefix': 'Error'
        },
        ActivityType.SUCCESS: {
            'icon': '🎉',
            'color': 'success',
            'prefix': 'Success'
        }
    }
    
    def __init__(self, max_items: int = 100, auto_scroll: bool = True):
        """
        Initialize the live feed.
        
        Args:
            max_items: Maximum number of items to keep in the feed
            auto_scroll: Whether to auto-scroll to latest items
        """
        self.max_items = max_items
        self.auto_scroll = auto_scroll
        self.container_manager = get_container_manager()
        self.theme_manager = get_theme_manager()
        
        # Session state is already initialized in app_v2.py
    
    def add_activity(self, activity_type: ActivityType, title: str, 
                    phase: Optional[str] = None, description: Optional[str] = None,
                    details: Optional[Dict[str, Any]] = None):
        """
        Add a new activity to the feed.
        
        Args:
            activity_type: Type of activity
            title: Activity title
            phase: Associated phase name (optional)
            description: Detailed description (optional)
            details: Additional details dictionary (optional)
        """
        config = self.ACTIVITY_CONFIG.get(activity_type, {})
        
        activity = ActivityItem(
            timestamp=datetime.now(),
            activity_type=activity_type,
            phase=phase,
            title=title,
            description=description,
            details=details or {},
            icon=config.get('icon', '•'),
            color=config.get('color', 'info')
        )
        
        # Add to feed
        st.session_state.live_feed['activities'].append(activity)
        st.session_state.live_feed['last_update'] = datetime.now()
        
        # Update statistics
        stats = st.session_state.live_feed['stats']
        stats['total_activities'] += 1
        
        if activity_type == ActivityType.PHASE_START:
            stats['phases_started'] += 1
        elif activity_type == ActivityType.PHASE_COMPLETE:
            stats['phases_completed'] += 1
        elif activity_type == ActivityType.ARTIFACT_CREATED:
            stats['artifacts_created'] += 1
        elif activity_type in [ActivityType.ERROR, ActivityType.PHASE_ERROR]:
            stats['errors'] += 1
    
    def render(self, container_key: str = "live_feed", height: int = 400):
        """
        Render the live feed component.
        
        Args:
            container_key: Key for the container
            height: Height of the feed in pixels
        """
        # Apply theme
        self.theme_manager.apply_theme()
        
        # Don't use managed containers for main UI to avoid widget ID changes
        with st.container():
            # Header with controls
            self._render_header()
            
            # Feed container
            feed_container = st.container(height=height, border=True)
            
            # Render feed with fragment
            self._render_feed_fragment(feed_container)
            
            # Statistics
            if st.checkbox("Show Statistics", key="live_feed_show_stats"):
                self._render_statistics()
    
    def _render_header(self):
        """Render the feed header with controls."""
        col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 1, 1])
        
        with col1:
            st.markdown("### 🔴 Live Activity Feed")
        
        with col2:
            # Filter by type
            activity_types = ['all'] + [t.value for t in ActivityType]
            filter_type = st.selectbox(
                "Filter",
                activity_types,
                format_func=lambda x: x.replace('_', ' ').title(),
                key="live_feed_filter_type",
                label_visibility="collapsed"
            )
            st.session_state.live_feed['filter_type'] = filter_type
        
        with col3:
            # Pause/Resume toggle
            st.session_state.live_feed['paused'] = st.checkbox(
                "⏸" if st.session_state.live_feed['paused'] else "▶️",
                value=st.session_state.live_feed['paused'],
                key="live_feed_pause_toggle",
                help="Pause/Resume feed",
                label_visibility="collapsed"
            )
        
        with col4:
            # Clear button
            if st.button("🗑️", key="live_feed_clear", help="Clear feed"):
                self._clear_feed()
        
        with col5:
            # Auto-scroll toggle
            self.auto_scroll = st.checkbox(
                "📜",
                value=self.auto_scroll,
                key="live_feed_autoscroll",
                help="Auto-scroll"
            )
    
    def _render_feed_fragment(self, container):
        """Render the feed content with automatic updates."""
        with container:
            activities = list(st.session_state.live_feed['activities'])
            filter_type = st.session_state.live_feed['filter_type']
            
            # Filter activities
            if filter_type != 'all':
                activities = [a for a in activities 
                            if a.activity_type.value == filter_type]
            
            # Sort by timestamp (newest first)
            activities.sort(key=lambda x: x.timestamp, reverse=True)
            
            if activities:
                # Render activities
                for activity in activities[:50]:  # Show last 50 items
                    self._render_activity_item(activity)
                
                # Auto-scroll indicator
                if self.auto_scroll and len(activities) > 10:
                    st.markdown(
                        '<div id="feed-bottom"></div>',
                        unsafe_allow_html=True
                    )
                    # JavaScript to scroll to bottom
                    st.markdown(
                        """
                        <script>
                        const feedBottom = document.getElementById('feed-bottom');
                        if (feedBottom) {
                            feedBottom.scrollIntoView({ behavior: 'smooth' });
                        }
                        </script>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.info("No activities yet. Activities will appear here as the workflow progresses.")
    
    def _render_activity_item(self, activity: ActivityItem):
        """Render a single activity item."""
        # Calculate time ago
        time_ago = self._format_time_ago(activity.timestamp)
        
        # Get configuration
        config = self.ACTIVITY_CONFIG.get(activity.activity_type, {})
        
        # Create activity item with proper styling
        item_class = f"live-feed-item feed-{activity.color}"
        
        with st.container():
            # Use columns for layout
            col1, col2, col3 = st.columns([1, 8, 2])
            
            with col1:
                # Icon
                st.markdown(
                    f'<div style="font-size: 1.5em; text-align: center;">{activity.icon}</div>',
                    unsafe_allow_html=True
                )
            
            with col2:
                # Title and description
                title_html = f"**{activity.title}**"
                if activity.phase:
                    title_html = f"[{activity.phase}] {title_html}"
                
                st.markdown(title_html)
                
                if activity.description:
                    st.caption(activity.description)
                
                # Details (if any)
                if activity.details:
                    with st.expander("Details", expanded=False):
                        for key, value in activity.details.items():
                            st.text(f"{key}: {value}")
            
            with col3:
                # Timestamp
                st.caption(time_ago)
    
    def _render_statistics(self):
        """Render feed statistics."""
        stats = st.session_state.live_feed['stats']
        
        st.divider()
        st.markdown("#### 📊 Feed Statistics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Activities", stats['total_activities'])
        
        with col2:
            st.metric("Phases Started", stats['phases_started'])
        
        with col3:
            st.metric("Phases Completed", stats['phases_completed'])
        
        with col4:
            st.metric("Artifacts Created", stats['artifacts_created'])
        
        with col5:
            st.metric("Errors", stats['errors'])
        
        # Last update time
        last_update = st.session_state.live_feed['last_update']
        st.caption(f"Last update: {self._format_time_ago(last_update)}")
    
    def _format_time_ago(self, timestamp: datetime) -> str:
        """Format timestamp as time ago string."""
        now = datetime.now()
        delta = now - timestamp
        
        if delta < timedelta(seconds=5):
            return "just now"
        elif delta < timedelta(minutes=1):
            return f"{int(delta.total_seconds())}s ago"
        elif delta < timedelta(hours=1):
            minutes = int(delta.total_seconds() / 60)
            return f"{minutes}m ago"
        elif delta < timedelta(days=1):
            hours = int(delta.total_seconds() / 3600)
            return f"{hours}h ago"
        else:
            return timestamp.strftime('%Y-%m-%d %H:%M')
    
    def _clear_feed(self):
        """Clear the feed."""
        st.session_state.live_feed['activities'].clear()
        st.session_state.live_feed['stats'] = {
            'total_activities': 0,
            'phases_started': 0,
            'phases_completed': 0,
            'artifacts_created': 0,
            'errors': 0
        }
        st.rerun()
    
    def get_recent_activities(self, count: int = 10, 
                            activity_type: Optional[ActivityType] = None) -> List[ActivityItem]:
        """
        Get recent activities from the feed.
        
        Args:
            count: Number of activities to retrieve
            activity_type: Filter by specific type (optional)
            
        Returns:
            List of recent activities
        """
        activities = list(st.session_state.live_feed['activities'])
        
        if activity_type:
            activities = [a for a in activities if a.activity_type == activity_type]
        
        # Sort by timestamp (newest first) and return requested count
        activities.sort(key=lambda x: x.timestamp, reverse=True)
        return activities[:count]
    
    def export_feed(self) -> str:
        """
        Export the feed as JSON.
        
        Returns:
            JSON string of feed data
        """
        import json
        
        activities = list(st.session_state.live_feed['activities'])
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'total_activities': len(activities),
            'statistics': st.session_state.live_feed['stats'],
            'activities': [
                {
                    'timestamp': a.timestamp.isoformat(),
                    'type': a.activity_type.value,
                    'phase': a.phase,
                    'title': a.title,
                    'description': a.description,
                    'details': a.details
                }
                for a in activities
            ]
        }
        
        return json.dumps(export_data, indent=2)


# Convenience functions for adding specific activity types
def log_phase_start(phase: str, description: Optional[str] = None):
    """Log a phase start event."""
    feed = LiveFeed()
    feed.add_activity(
        ActivityType.PHASE_START,
        f"Starting {phase.replace('_', ' ').title()}",
        phase=phase,
        description=description
    )


def log_phase_complete(phase: str, duration: float, artifacts: int = 0):
    """Log a phase completion event."""
    feed = LiveFeed()
    feed.add_activity(
        ActivityType.PHASE_COMPLETE,
        f"Completed {phase.replace('_', ' ').title()}",
        phase=phase,
        description=f"Duration: {duration:.1f}s",
        details={'duration': duration, 'artifacts': artifacts}
    )


def log_phase_error(phase: str, error: str):
    """Log a phase error event."""
    feed = LiveFeed()
    feed.add_activity(
        ActivityType.PHASE_ERROR,
        f"Error in {phase.replace('_', ' ').title()}",
        phase=phase,
        description=str(error)[:200]  # Limit error message length
    )


def log_artifact_created(artifact_path: str, phase: Optional[str] = None):
    """Log an artifact creation event."""
    feed = LiveFeed()
    artifact_name = artifact_path.split('/')[-1] if '/' in artifact_path else artifact_path
    feed.add_activity(
        ActivityType.ARTIFACT_CREATED,
        f"Created {artifact_name}",
        phase=phase,
        description=f"Path: {artifact_path}"
    )


def log_tool_processing(tool_name: str, phase: str, action: str = "Processing"):
    """Log a tool processing event."""
    feed = LiveFeed()
    feed.add_activity(
        ActivityType.TOOL_PROCESSING,
        f"{action} {tool_name}",
        phase=phase
    )


def log_validation(item: str, passed: bool, phase: Optional[str] = None):
    """Log a validation event."""
    feed = LiveFeed()
    status = "passed" if passed else "failed"
    feed.add_activity(
        ActivityType.VALIDATION,
        f"Validation {status}: {item}",
        phase=phase
    )


def log_info(message: str, phase: Optional[str] = None):
    """Log an info message."""
    feed = LiveFeed()
    feed.add_activity(
        ActivityType.INFO,
        message,
        phase=phase
    )


def log_warning(message: str, phase: Optional[str] = None):
    """Log a warning message."""
    feed = LiveFeed()
    feed.add_activity(
        ActivityType.WARNING,
        message,
        phase=phase
    )


def log_error(message: str, phase: Optional[str] = None):
    """Log an error message."""
    feed = LiveFeed()
    feed.add_activity(
        ActivityType.ERROR,
        message,
        phase=phase
    )


def log_success(message: str, phase: Optional[str] = None):
    """Log a success message."""
    feed = LiveFeed()
    feed.add_activity(
        ActivityType.SUCCESS,
        message,
        phase=phase
    )