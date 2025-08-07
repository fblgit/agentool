# -*- coding: utf-8 -*-
"""
Container Manager - Orchestrates persistent Streamlit containers for non-blocking updates.

This component manages st.empty() containers and provides a centralized system
for creating, storing, and updating UI elements without triggering full page reruns.
"""

import streamlit as st
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
import threading
from contextlib import contextmanager


@dataclass
class ContainerConfig:
    """Configuration for a managed container."""
    key: str
    container_type: str  # 'empty', 'container', 'columns', 'sidebar', 'expander'
    parent: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: Optional[datetime] = None


class UIContainerManager:
    """
    Manages persistent UI containers for fragment-based updates.
    
    This class provides a centralized system for creating and managing
    Streamlit containers that persist across reruns and can be updated
    independently through fragments.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the container manager."""
        # Only initialize once
        if not hasattr(self, '_initialized_singleton'):
            self._containers: Dict[str, Any] = {}
            self._configs: Dict[str, ContainerConfig] = {}
            self._container_hierarchy: Dict[str, List[str]] = {}
            self._initialized_singleton = True
    
    def _ensure_initialized(self):
        """Ensure session state is initialized for container manager."""
        # Always check session state, not just once
        if 'container_manager' not in st.session_state:
            st.session_state.container_manager = {
                'configs': {},
                'hierarchy': {}
            }
        # Note: We don't cache containers as they become invalid between runs
    
    def get_or_create_empty(self, key: str, parent: Optional[Any] = None) -> Any:
        """
        Get or create an st.empty() container.
        NOTE: Containers cannot be cached - they must be recreated each run.
        
        Args:
            key: Unique identifier for the container
            parent: Parent container to create within (optional)
            
        Returns:
            The st.empty() container
        """
        self._ensure_initialized()
        with self._lock:
            # Always create new empty container - DO NOT cache
            if parent:
                container = parent.empty()
            else:
                container = st.empty()
            
            # Store config only (not the container itself)
            st.session_state.container_manager['configs'][key] = ContainerConfig(
                key=key,
                container_type='empty',
                parent=getattr(parent, '_key', None) if parent else None
            )
            
            return container
    
    def get_or_create_container(self, key: str, parent: Optional[Any] = None, 
                               height: Optional[int] = None, border: bool = True) -> Any:
        """
        Get or create an st.container().
        NOTE: Containers cannot be cached - they must be recreated each run.
        
        Args:
            key: Unique identifier for the container
            parent: Parent container to create within (optional)
            height: Fixed height in pixels (optional)
            border: Whether to show border (default True)
            
        Returns:
            The st.container()
        """
        # Ensure initialization OUTSIDE the lock
        self._ensure_initialized()
        
        with self._lock:
            # Always create new container - DO NOT cache
            if parent:
                with parent:
                    container = st.container(height=height, border=border)
            else:
                container = st.container(height=height, border=border)
            
            # Store config only (not the container itself)
            st.session_state.container_manager['configs'][key] = ContainerConfig(
                key=key,
                container_type='container',
                parent=getattr(parent, '_key', None) if parent else None,
                properties={'height': height, 'border': border}
            )
            
            return container
    
    def create_column_set(self, key: str, spec: List[float], gap: str = "small") -> List[Any]:
        """
        Create a set of columns.
        
        Args:
            key: Unique identifier for the column set
            spec: Column width specifications
            gap: Gap size between columns
            
        Returns:
            List of column containers
        """
        self._ensure_initialized()
        with self._lock:
            # Always create new columns - DO NOT cache
            columns = st.columns(spec, gap=gap)
            
            # Don't store columns - they become invalid
            st.session_state.container_manager['configs'][key] = ContainerConfig(
                key=key,
                container_type='columns',
                properties={'spec': spec, 'gap': gap}
            )
            
            # Store hierarchy info only
            for i, col in enumerate(columns):
                col_key = f"{key}_col_{i}"
                st.session_state.container_manager['hierarchy'].setdefault(key, []).append(col_key)
            
            return columns
    
    def create_expander(self, key: str, label: str, expanded: bool = False,
                       icon: Optional[str] = None) -> Any:
        """
        Create an expander container.
        
        Args:
            key: Unique identifier for the expander
            label: Label text for the expander
            expanded: Whether to start expanded
            icon: Optional icon emoji
            
        Returns:
            The expander container
        """
        self._ensure_initialized()
        with self._lock:
            # Check if expander already exists
            if key in st.session_state.container_manager['containers']:
                return st.session_state.container_manager['containers'][key]
            
            # Create expander
            expander = st.expander(label, expanded=expanded, icon=icon)
            
            # Store in session state
            st.session_state.container_manager['containers'][key] = expander
            st.session_state.container_manager['configs'][key] = ContainerConfig(
                key=key,
                container_type='expander',
                properties={'label': label, 'expanded': expanded, 'icon': icon}
            )
            
            return expander
    
    def create_sidebar_container(self, key: str) -> Any:
        """
        Create a container in the sidebar.
        
        Args:
            key: Unique identifier for the sidebar container
            
        Returns:
            The sidebar container
        """
        self._ensure_initialized()
        with self._lock:
            # Check if sidebar container already exists
            if key in st.session_state.container_manager['containers']:
                return st.session_state.container_manager['containers'][key]
            
            # Create sidebar container
            with st.sidebar:
                container = st.container()
            
            # Store in session state
            st.session_state.container_manager['containers'][key] = container
            st.session_state.container_manager['configs'][key] = ContainerConfig(
                key=key,
                container_type='sidebar',
                properties={}
            )
            
            return container
    
    def update_container(self, key: str, update_func: Callable, *args, **kwargs):
        """
        Update a container's content without triggering a full rerun.
        
        Args:
            key: Container identifier
            update_func: Function to call with the container
            *args, **kwargs: Arguments to pass to update_func
        """
        self._ensure_initialized()
        if key not in st.session_state.container_manager['containers']:
            raise KeyError(f"Container '{key}' not found")
        
        container = st.session_state.container_manager['containers'][key]
        
        # Update the container
        with container:
            update_func(*args, **kwargs)
        
        # Update timestamp
        if key in st.session_state.container_manager['configs']:
            st.session_state.container_manager['configs'][key].last_updated = datetime.now()
    
    def clear_container(self, key: str):
        """
        Clear a container's content.
        
        Args:
            key: Container identifier
        """
        self._ensure_initialized()
        if key not in st.session_state.container_manager['containers']:
            return
        
        container = st.session_state.container_manager['containers'][key]
        
        # For empty containers, clear by calling empty again
        if st.session_state.container_manager['configs'][key].container_type == 'empty':
            container.empty()
        else:
            # For other containers, we need to recreate them
            # This is a limitation of Streamlit's current API
            pass
    
    def remove_container(self, key: str):
        """
        Remove a container from management.
        
        Args:
            key: Container identifier
        """
        self._ensure_initialized()
        with self._lock:
            # Remove from session state
            if key in st.session_state.container_manager['containers']:
                del st.session_state.container_manager['containers'][key]
            
            if key in st.session_state.container_manager['configs']:
                del st.session_state.container_manager['configs'][key]
            
            # Remove from hierarchy
            if key in st.session_state.container_manager['hierarchy']:
                # Also remove child containers
                for child_key in st.session_state.container_manager['hierarchy'][key]:
                    self.remove_container(child_key)
                del st.session_state.container_manager['hierarchy'][key]
    
    def get_container(self, key: str) -> Optional[Any]:
        """
        Get a container by key.
        
        Args:
            key: Container identifier
            
        Returns:
            The container or None if not found
        """
        self._ensure_initialized()
        return st.session_state.container_manager['containers'].get(key)
    
    def list_containers(self) -> List[str]:
        """
        List all managed container keys.
        
        Returns:
            List of container keys
        """
        self._ensure_initialized()
        return list(st.session_state.container_manager['containers'].keys())
    
    def get_container_info(self, key: str) -> Optional[ContainerConfig]:
        """
        Get configuration info for a container.
        
        Args:
            key: Container identifier
            
        Returns:
            ContainerConfig or None if not found
        """
        return st.session_state.container_manager['configs'].get(key)
    
    @contextmanager
    def managed_container(self, key: str, container_type: str = 'container', **kwargs):
        """
        Context manager for working with managed containers.
        
        Args:
            key: Container identifier
            container_type: Type of container to create
            **kwargs: Additional arguments for container creation
            
        Yields:
            The container
        """
        # Create or get container based on type
        if container_type == 'empty':
            container = self.get_or_create_empty(key, **kwargs)
        elif container_type == 'container':
            container = self.get_or_create_container(key, **kwargs)
        elif container_type == 'expander':
            container = self.create_expander(key, **kwargs)
        elif container_type == 'sidebar':
            container = self.create_sidebar_container(key)
        else:
            raise ValueError(f"Unknown container type: {container_type}")
        
        try:
            yield container
        finally:
            # Update last accessed time
            if key in st.session_state.container_manager['configs']:
                st.session_state.container_manager['configs'][key].last_updated = datetime.now()
    
    def create_grid_layout(self, key: str, rows: int, cols: int, 
                          heights: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Create a grid layout of containers.
        
        Args:
            key: Unique identifier for the grid
            rows: Number of rows
            cols: Number of columns
            heights: Optional list of row heights
            
        Returns:
            Dictionary mapping (row, col) tuples to containers
        """
        grid = {}
        
        for row in range(rows):
            row_key = f"{key}_row_{row}"
            row_cols = self.create_column_set(row_key, [1] * cols)
            
            for col in range(cols):
                cell_key = f"{key}_cell_{row}_{col}"
                height = heights[row] if heights and row < len(heights) else None
                
                with row_cols[col]:
                    container = st.container(height=height)
                    st.session_state.container_manager['containers'][cell_key] = container
                    grid[(row, col)] = container
        
        return grid
    
    def batch_update(self, updates: List[tuple]):
        """
        Perform multiple container updates in a single operation.
        
        Args:
            updates: List of (key, update_func, args, kwargs) tuples
        """
        for update in updates:
            if len(update) == 2:
                key, update_func = update
                args, kwargs = (), {}
            elif len(update) == 3:
                key, update_func, args = update
                kwargs = {}
            else:
                key, update_func, args, kwargs = update
            
            self.update_container(key, update_func, *args, **kwargs)


# Global singleton instance
_container_manager_singleton = None


def get_container_manager() -> UIContainerManager:
    """
    Get the singleton container manager instance.
    
    Returns:
        The UIContainerManager instance
    """
    global _container_manager_singleton
    if _container_manager_singleton is None:
        _container_manager_singleton = UIContainerManager()
    return _container_manager_singleton