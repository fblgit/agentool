# -*- coding: utf-8 -*-
"""
Fragment Utilities - Helper functions for Streamlit fragment management.

This module provides utilities for creating and managing Streamlit fragments
to enable partial UI updates without full page reruns.
"""

import streamlit as st
from typing import Any, Callable, Optional, Dict, List, Tuple
from functools import wraps
import time
import asyncio
from datetime import datetime, timedelta
import threading
from dataclasses import dataclass, field


@dataclass
class FragmentState:
    """State information for a fragment."""
    key: str
    last_run: Optional[datetime] = None
    run_count: int = 0
    is_running: bool = False
    error_count: int = 0
    last_error: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)


class FragmentManager:
    """
    Manages fragment execution and state.
    
    This class provides utilities for creating, tracking, and
    coordinating fragment execution in a Streamlit app.
    """
    
    def __init__(self):
        """Initialize the fragment manager."""
        if 'fragment_manager' not in st.session_state:
            st.session_state.fragment_manager = {
                'fragments': {},
                'run_queue': [],
                'global_pause': False
            }
        
        self._lock = threading.Lock()
    
    def register_fragment(self, key: str) -> FragmentState:
        """
        Register a new fragment.
        
        Args:
            key: Unique identifier for the fragment
            
        Returns:
            FragmentState object
        """
        with self._lock:
            if key not in st.session_state.fragment_manager['fragments']:
                state = FragmentState(key=key)
                st.session_state.fragment_manager['fragments'][key] = state
            
            return st.session_state.fragment_manager['fragments'][key]
    
    def get_fragment_state(self, key: str) -> Optional[FragmentState]:
        """
        Get the state of a fragment.
        
        Args:
            key: Fragment identifier
            
        Returns:
            FragmentState or None if not found
        """
        return st.session_state.fragment_manager['fragments'].get(key)
    
    def update_fragment_state(self, key: str, **updates):
        """
        Update fragment state.
        
        Args:
            key: Fragment identifier
            **updates: State updates to apply
        """
        with self._lock:
            if key in st.session_state.fragment_manager['fragments']:
                state = st.session_state.fragment_manager['fragments'][key]
                for k, v in updates.items():
                    if hasattr(state, k):
                        setattr(state, k, v)
    
    def queue_fragment_run(self, key: str):
        """
        Queue a fragment for execution.
        
        Args:
            key: Fragment identifier
        """
        with self._lock:
            if key not in st.session_state.fragment_manager['run_queue']:
                st.session_state.fragment_manager['run_queue'].append(key)
    
    def get_next_fragment(self) -> Optional[str]:
        """
        Get the next fragment to run from the queue.
        
        Returns:
            Fragment key or None if queue is empty
        """
        with self._lock:
            if st.session_state.fragment_manager['run_queue']:
                return st.session_state.fragment_manager['run_queue'].pop(0)
            return None
    
    def pause_all_fragments(self):
        """Pause all fragment execution."""
        st.session_state.fragment_manager['global_pause'] = True
    
    def resume_all_fragments(self):
        """Resume all fragment execution."""
        st.session_state.fragment_manager['global_pause'] = False
    
    def is_paused(self) -> bool:
        """Check if fragment execution is paused."""
        return st.session_state.fragment_manager['global_pause']
    
    def clear_fragment_errors(self, key: Optional[str] = None):
        """
        Clear error state for fragments.
        
        Args:
            key: Specific fragment to clear, or None for all
        """
        with self._lock:
            if key:
                if key in st.session_state.fragment_manager['fragments']:
                    state = st.session_state.fragment_manager['fragments'][key]
                    state.error_count = 0
                    state.last_error = None
            else:
                for state in st.session_state.fragment_manager['fragments'].values():
                    state.error_count = 0
                    state.last_error = None


# Global fragment manager instance
_fragment_manager = FragmentManager()


def create_auto_fragment(
    func: Callable,
    key: str,
    run_every: Optional[float] = None,
    container: Optional[Any] = None,
    show_spinner: bool = False,
    spinner_text: str = "Updating...",
    error_handler: Optional[Callable] = None
):
    """
    Create a fragment with automatic execution and error handling.
    
    Args:
        func: Function to execute in the fragment
        key: Unique identifier for the fragment
        run_every: Auto-run interval in seconds (optional)
        container: Container to render in (optional)
        show_spinner: Whether to show a spinner during execution
        spinner_text: Text to display in spinner
        error_handler: Custom error handler function
    
    Returns:
        Decorated fragment function
    """
    # Register the fragment
    _fragment_manager.register_fragment(key)
    
    @st.fragment(run_every=run_every)
    def fragment_wrapper(*args, **kwargs):
        # Check if globally paused
        if _fragment_manager.is_paused():
            return
        
        # Get fragment state
        state = _fragment_manager.get_fragment_state(key)
        if not state:
            return
        
        # Update running state
        _fragment_manager.update_fragment_state(key, is_running=True, last_run=datetime.now())
        
        try:
            # Execute with optional spinner
            if show_spinner and container:
                with container:
                    with st.spinner(spinner_text):
                        result = func(*args, **kwargs)
            elif container:
                with container:
                    result = func(*args, **kwargs)
            else:
                if show_spinner:
                    with st.spinner(spinner_text):
                        result = func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
            
            # Update state on success
            _fragment_manager.update_fragment_state(
                key,
                is_running=False,
                run_count=state.run_count + 1
            )
            
            return result
            
        except Exception as e:
            # Update error state
            _fragment_manager.update_fragment_state(
                key,
                is_running=False,
                error_count=state.error_count + 1,
                last_error=str(e)
            )
            
            # Call error handler if provided
            if error_handler:
                error_handler(e, key)
            else:
                # Default error handling
                if container:
                    with container:
                        st.error(f"Error in fragment '{key}': {e}")
                else:
                    st.error(f"Error in fragment '{key}': {e}")
    
    return fragment_wrapper


def create_conditional_fragment(
    func: Callable,
    key: str,
    condition: Callable[[], bool],
    container: Optional[Any] = None,
    run_every: Optional[float] = None
):
    """
    Create a fragment that only runs when a condition is met.
    
    Args:
        func: Function to execute in the fragment
        key: Unique identifier for the fragment
        condition: Function that returns True when fragment should run
        container: Container to render in (optional)
        run_every: Auto-run interval for checking condition (optional)
    
    Returns:
        Decorated fragment function
    """
    @st.fragment(run_every=run_every)
    def conditional_wrapper(*args, **kwargs):
        if condition():
            if container:
                with container:
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
    
    return conditional_wrapper


def create_debounced_fragment(
    func: Callable,
    key: str,
    delay: float = 0.5,
    container: Optional[Any] = None
):
    """
    Create a fragment with debounced execution.
    
    This fragment will only execute after a delay period has passed
    without any new calls.
    
    Args:
        func: Function to execute in the fragment
        key: Unique identifier for the fragment
        delay: Debounce delay in seconds
        container: Container to render in (optional)
    
    Returns:
        Decorated fragment function
    """
    debounce_key = f"debounce_{key}"
    
    @st.fragment
    def debounced_wrapper(*args, **kwargs):
        current_time = time.time()
        
        # Initialize or get last call time
        if debounce_key not in st.session_state:
            st.session_state[debounce_key] = current_time
        
        last_call = st.session_state[debounce_key]
        st.session_state[debounce_key] = current_time
        
        # Only execute if enough time has passed
        if current_time - last_call >= delay:
            if container:
                with container:
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
    
    return debounced_wrapper


def create_throttled_fragment(
    func: Callable,
    key: str,
    interval: float = 1.0,
    container: Optional[Any] = None
):
    """
    Create a fragment with throttled execution.
    
    This fragment will execute at most once per interval period.
    
    Args:
        func: Function to execute in the fragment
        key: Unique identifier for the fragment
        interval: Minimum interval between executions in seconds
        container: Container to render in (optional)
    
    Returns:
        Decorated fragment function
    """
    throttle_key = f"throttle_{key}"
    
    @st.fragment
    def throttled_wrapper(*args, **kwargs):
        current_time = time.time()
        
        # Initialize or get last execution time
        if throttle_key not in st.session_state:
            st.session_state[throttle_key] = 0
        
        last_exec = st.session_state[throttle_key]
        
        # Only execute if enough time has passed
        if current_time - last_exec >= interval:
            st.session_state[throttle_key] = current_time
            
            if container:
                with container:
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
    
    return throttled_wrapper


def create_lazy_fragment(
    func: Callable,
    key: str,
    trigger_key: str,
    container: Optional[Any] = None
):
    """
    Create a fragment that only executes when triggered.
    
    Args:
        func: Function to execute in the fragment
        key: Unique identifier for the fragment
        trigger_key: Session state key that triggers execution
        container: Container to render in (optional)
    
    Returns:
        Decorated fragment function
    """
    @st.fragment
    def lazy_wrapper(*args, **kwargs):
        # Check if trigger is set
        if st.session_state.get(trigger_key, False):
            # Reset trigger
            st.session_state[trigger_key] = False
            
            if container:
                with container:
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
    
    return lazy_wrapper


def batch_fragment_updates(fragments: List[Tuple[str, Callable, Dict]]):
    """
    Execute multiple fragment updates in a batch.
    
    Args:
        fragments: List of (key, func, kwargs) tuples
    """
    for key, func, kwargs in fragments:
        # Queue the fragment for execution
        _fragment_manager.queue_fragment_run(key)
        
        # Execute the fragment
        @st.fragment
        def batch_wrapper():
            return func(**kwargs)
        
        batch_wrapper()


def create_synchronized_fragments(
    fragments: Dict[str, Callable],
    sync_key: str,
    container_map: Optional[Dict[str, Any]] = None
):
    """
    Create a group of synchronized fragments that update together.
    
    Args:
        fragments: Dictionary mapping keys to functions
        sync_key: Shared synchronization key
        container_map: Optional mapping of keys to containers
    
    Returns:
        Dictionary of decorated fragment functions
    """
    synchronized = {}
    
    for key, func in fragments.items():
        container = container_map.get(key) if container_map else None
        
        @st.fragment
        def sync_wrapper(f=func, c=container, k=key):
            # Check synchronization state
            if st.session_state.get(f"{sync_key}_update", False):
                if c:
                    with c:
                        return f()
                else:
                    return f()
        
        synchronized[key] = sync_wrapper
    
    return synchronized


def fragment_with_progress(
    func: Callable,
    key: str,
    total_steps: int,
    container: Optional[Any] = None
):
    """
    Create a fragment that displays progress during execution.
    
    Args:
        func: Function to execute (should yield progress updates)
        key: Unique identifier for the fragment
        total_steps: Total number of steps
        container: Container to render in (optional)
    
    Returns:
        Decorated fragment function
    """
    @st.fragment
    def progress_wrapper(*args, **kwargs):
        if container:
            with container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, status in enumerate(func(*args, **kwargs)):
                    progress = min((i + 1) / total_steps, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(status)
                
                progress_bar.progress(1.0)
                status_text.text("Complete!")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, status in enumerate(func(*args, **kwargs)):
                progress = min((i + 1) / total_steps, 1.0)
                progress_bar.progress(progress)
                status_text.text(status)
            
            progress_bar.progress(1.0)
            status_text.text("Complete!")
    
    return progress_wrapper


def create_async_fragment(
    async_func: Callable,
    key: str,
    container: Optional[Any] = None,
    run_every: Optional[float] = None
):
    """
    Create a fragment from an async function.
    
    Args:
        async_func: Async function to execute
        key: Unique identifier for the fragment
        container: Container to render in (optional)
        run_every: Auto-run interval in seconds (optional)
    
    Returns:
        Decorated fragment function
    """
    @st.fragment(run_every=run_every)
    def async_wrapper(*args, **kwargs):
        # Run async function in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(async_func(*args, **kwargs))
            
            if container:
                with container:
                    return result
            else:
                return result
        finally:
            loop.close()
    
    return async_wrapper