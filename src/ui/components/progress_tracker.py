"""
Progress tracking component for workflow phases.

This component provides visual progress tracking for each phase
of the workflow, including status indicators and timing information.
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import time


class ProgressTracker:
    """
    Track and display progress for a workflow phase.
    
    This component creates a visual representation of a phase's progress,
    including status indicators, timing, and error information.
    """
    
    def __init__(self, phase_name: str, container: st.container = None):
        """
        Initialize the progress tracker.
        
        Args:
            phase_name: Name of the phase to track
            container: Streamlit container (creates new if None)
        """
        self.phase_name = phase_name
        self.container = container or st.container()
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.status = "pending"  # pending, running, completed, error
        self.error_message: Optional[str] = None
        self.sub_tasks: Dict[str, bool] = {}
        
        # Create UI elements
        self._create_ui()
    
    def _create_ui(self):
        """Create the initial UI structure."""
        with self.container:
            # Main phase container
            self.phase_container = st.container()
            
            with self.phase_container:
                # Header row
                col1, col2, col3 = st.columns([1, 3, 1])
                
                with col1:
                    self.status_icon = st.empty()
                    self._update_status_icon()
                
                with col2:
                    self.phase_label = st.empty()
                    self._update_phase_label()
                
                with col3:
                    self.duration_label = st.empty()
                    self._update_duration()
                
                # Progress bar
                self.progress_bar = st.progress(0, text="")
                
                # Details expander
                self.details_expander = st.expander("Details", expanded=False)
                
                with self.details_expander:
                    self.details_container = st.container()
    
    def _update_status_icon(self):
        """Update the status icon based on current status."""
        icons = {
            "pending": "⏳",
            "running": "🔄",
            "completed": "✅",
            "error": "❌"
        }
        self.status_icon.markdown(f"## {icons.get(self.status, '❓')}")
    
    def _update_phase_label(self):
        """Update the phase label."""
        status_text = {
            "pending": "Waiting",
            "running": "Processing",
            "completed": "Completed",
            "error": "Failed"
        }
        
        label = f"**{self.phase_name}** - {status_text.get(self.status, 'Unknown')}"
        
        if self.status == "error" and self.error_message:
            label += f"\n\n🚨 {self.error_message}"
        
        self.phase_label.markdown(label)
    
    def _update_duration(self):
        """Update the duration display."""
        if self.start_time:
            if self.end_time:
                duration = (self.end_time - self.start_time).total_seconds()
                self.duration_label.markdown(f"**{duration:.1f}s**")
            else:
                # Still running
                duration = (datetime.now() - self.start_time).total_seconds()
                self.duration_label.markdown(f"**{duration:.1f}s** ⏱️")
    
    def _update_progress_bar(self):
        """Update the progress bar."""
        if self.status == "pending":
            self.progress_bar.progress(0, text="Waiting...")
        elif self.status == "running":
            # Indeterminate progress
            progress = int((time.time() * 100) % 100) / 100
            self.progress_bar.progress(progress, text="Processing...")
        elif self.status == "completed":
            self.progress_bar.progress(1.0, text="Completed")
        elif self.status == "error":
            self.progress_bar.progress(1.0, text="Failed")
    
    def _apply_styling(self):
        """Apply CSS styling based on status."""
        styles = {
            "pending": "",
            "running": "background-color: #e3f2fd; border: 2px solid #2196f3;",
            "completed": "background-color: #e8f5e9; border: 2px solid #4caf50;",
            "error": "background-color: #ffebee; border: 2px solid #f44336;"
        }
        
        style = styles.get(self.status, "")
        if style:
            self.phase_container.markdown(
                f'<div style="{style} padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">',
                unsafe_allow_html=True
            )
    
    def start(self):
        """Mark the phase as started."""
        self.status = "running"
        self.start_time = datetime.now()
        self.error_message = None
        
        self._update_status_icon()
        self._update_phase_label()
        self._update_progress_bar()
        self._apply_styling()
        
        # Add start message to details
        with self.details_container:
            st.info(f"Started at {self.start_time.strftime('%H:%M:%S')}")
    
    def complete(self, duration: Optional[float] = None):
        """
        Mark the phase as completed.
        
        Args:
            duration: Optional duration in seconds
        """
        self.status = "completed"
        self.end_time = datetime.now()
        
        if duration is not None:
            # Use provided duration
            self.start_time = self.end_time - timedelta(seconds=duration)
        
        self._update_status_icon()
        self._update_phase_label()
        self._update_duration()
        self._update_progress_bar()
        self._apply_styling()
        
        # Add completion message to details
        with self.details_container:
            st.success(f"Completed at {self.end_time.strftime('%H:%M:%S')}")
    
    def error(self, error_message: str):
        """
        Mark the phase as failed.
        
        Args:
            error_message: Error description
        """
        self.status = "error"
        self.end_time = datetime.now()
        self.error_message = error_message
        
        self._update_status_icon()
        self._update_phase_label()
        self._update_duration()
        self._update_progress_bar()
        self._apply_styling()
        
        # Add error details
        with self.details_container:
            st.error(f"Failed at {self.end_time.strftime('%H:%M:%S')}")
            st.exception(error_message)
    
    def add_sub_task(self, task_name: str, completed: bool = False):
        """
        Add a sub-task to track.
        
        Args:
            task_name: Name of the sub-task
            completed: Whether the task is completed
        """
        self.sub_tasks[task_name] = completed
        
        # Update details
        with self.details_container:
            for task, is_complete in self.sub_tasks.items():
                if is_complete:
                    st.markdown(f"✅ {task}")
                else:
                    st.markdown(f"⏳ {task}")
    
    def complete_sub_task(self, task_name: str):
        """Mark a sub-task as completed."""
        if task_name in self.sub_tasks:
            self.sub_tasks[task_name] = True
            self.add_sub_task(task_name, True)  # Refresh display
    
    def add_message(self, message: str, message_type: str = "info"):
        """
        Add a message to the details.
        
        Args:
            message: Message text
            message_type: Type of message (info, success, warning, error)
        """
        with self.details_container:
            if message_type == "info":
                st.info(message)
            elif message_type == "success":
                st.success(message)
            elif message_type == "warning":
                st.warning(message)
            elif message_type == "error":
                st.error(message)
            else:
                st.write(message)
    
    def update_streaming_output(self, text: str):
        """
        Update with streaming output.
        
        Args:
            text: Streaming text to display
        """
        if not hasattr(self, '_stream_container'):
            with self.details_container:
                self._stream_container = st.empty()
        
        self._stream_container.markdown(text)


class WorkflowProgressTracker:
    """
    Track progress for the entire workflow.
    
    This component manages multiple phase trackers and provides
    an overall progress view.
    """
    
    def __init__(self, phases: list[str]):
        """
        Initialize the workflow tracker.
        
        Args:
            phases: List of phase names
        """
        self.phases = phases
        self.phase_trackers: Dict[str, ProgressTracker] = {}
        self.container = st.container()
    
    def create_ui(self):
        """Create the UI for all phases."""
        with self.container:
            st.subheader("Workflow Progress")
            
            # Overall progress
            self.overall_progress = st.progress(0, text="Initializing...")
            self.progress_text = st.empty()
            
            # Phase containers
            for phase in self.phases:
                phase_container = st.container()
                tracker = ProgressTracker(phase, phase_container)
                self.phase_trackers[phase] = tracker
    
    def update_overall_progress(self):
        """Update the overall progress bar."""
        completed = sum(
            1 for tracker in self.phase_trackers.values()
            if tracker.status == "completed"
        )
        progress = completed / len(self.phases)
        
        self.overall_progress.progress(progress, text=f"{int(progress * 100)}% Complete")
        self.progress_text.markdown(
            f"**Overall Progress**: {completed}/{len(self.phases)} phases completed ({int(progress * 100)}%)"
        )
    
    def start_phase(self, phase_name: str):
        """Start a specific phase."""
        if phase_name in self.phase_trackers:
            self.phase_trackers[phase_name].start()
    
    def complete_phase(self, phase_name: str, duration: Optional[float] = None):
        """Complete a specific phase."""
        if phase_name in self.phase_trackers:
            self.phase_trackers[phase_name].complete(duration)
            self.update_overall_progress()
    
    def error_phase(self, phase_name: str, error_message: str):
        """Mark a phase as failed."""
        if phase_name in self.phase_trackers:
            self.phase_trackers[phase_name].error(error_message)
            self.update_overall_progress()
    
    def get_tracker(self, phase_name: str) -> Optional[ProgressTracker]:
        """Get the tracker for a specific phase."""
        return self.phase_trackers.get(phase_name)