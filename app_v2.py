#!/usr/bin/env python
"""
Streamlit app for AgenTool Workflow Engine V2.

Run with: streamlit run app_v2.py
"""

import streamlit as st
import sys
import os
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Page config MUST be the first Streamlit command
st.set_page_config(
    page_title="AgenTool Workflow Engine V2",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add emergency error handler
try:
    from datetime import datetime
except Exception as e:
    st.error(f"Failed to import datetime: {e}")
    st.stop()

# Initialize agentoolkits BEFORE importing UI
from agentool.core.injector import get_injector

def initialize_agentoolkits():
    """Initialize all AgenToolkits (similar to v1)."""
    injector = get_injector()
    errors = []
    
    try:
        import agentoolkit
        
        # Initialize all create_* functions
        for name in dir(agentoolkit):
            if name.startswith('create_'):
                create_func = getattr(agentoolkit, name)
                try:
                    if name == 'create_templates_agent':
                        # Special handling for templates
                        templates_path = os.path.abspath(
                            os.path.join(os.path.dirname(__file__), 'src/templates')
                        )
                        agent = create_func(templates_dir=templates_path)
                    else:
                        agent = create_func()
                except Exception as e:
                    errors.append(f"Could not initialize {name}: {e}")
        
        # Initialize workflow agents
        from agentoolkit.workflows import (
            create_workflow_analyzer_agent,
            create_workflow_specifier_agent,
            create_workflow_crafter_agent,
            create_workflow_evaluator_agent,
            create_workflow_test_analyzer_agent,
            create_workflow_test_stubber_agent,
            create_workflow_test_crafter_agent
        )
        
        create_workflow_analyzer_agent()
        create_workflow_specifier_agent()
        create_workflow_crafter_agent()
        create_workflow_evaluator_agent()
        create_workflow_test_analyzer_agent()
        create_workflow_test_stubber_agent()
        create_workflow_test_crafter_agent()
        
    except Exception as e:
        import traceback
        errors.append(f"Failed to initialize AgenToolkits: {e}\n{traceback.format_exc()}")
    
    # Store errors in session state to display later
    if errors:
        return injector, errors
    return injector, None

# Initialize ALL session state here to prevent conflicts
try:
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = True
        st.session_state.container_refs = {}
        st.session_state.fragment_refs = {}
        
        # Initialize agentoolkits and capture any errors
        injector, init_errors = initialize_agentoolkits()
        st.session_state.injector = injector
        st.session_state.initialization_errors = init_errors
        
        # Initialize workflow UI state
        st.session_state.workflow_ui_v2 = {
            'workflow_state': None,
            'workflow_active': False,
            'workflow_completed': False,
            'execution_mode': 'automatic',
            'show_sidebar': True,
            'show_live_feed': True,
            'show_visualizations': True,
            'current_tab': 'execution',
            'phase_results': {},
            'artifacts': {},
            'start_time': None,
            'end_time': None,
            'error': None
        }
        
        # Initialize phase executor state
        st.session_state.phase_executor_v2 = {
            'phase_results': {},
            'current_phase': None,
            'phase_queue': [],
            'execution_paused': False,
            'artifacts_by_phase': {},
            'logs': [],
            'metrics': {
                'total_duration': 0,
                'phases_completed': 0,
                'phases_failed': 0,
                'artifacts_created': 0
            }
        }
        
        # Initialize live feed state
        import collections
        st.session_state.live_feed = {
            'activities': collections.deque(maxlen=200),
            'filter_type': 'all',
            'filter_phase': 'all',
            'paused': False,
            'last_update': datetime.now(),
            'stats': {
                'total_activities': 0,
                'phases_started': 0,
                'phases_completed': 0,
                'artifacts_created': 0,
                'errors': 0
            }
        }
        
        # Initialize artifact viewer state
        st.session_state.artifact_viewer_v2 = {
            'artifacts': {},
            'metadata': {},
            'selected_artifact': None,
            'filter_text': '',
            'filter_type': 'all',
            'filter_phase': 'all',
            'sort_by': 'created',
            'view_mode': 'tree',
            'expanded_categories': set(['analysis'])
        }
        
        # Initialize workflow viewer state
        st.session_state.workflow_viewer_v2 = {
            'view_mode': 'graph',
            'selected_phase': None,
            'zoom_level': 1.0,
            'auto_refresh': True,
            'show_dependencies': True,
            'show_artifacts': True,
            'expanded_phases': set()
        }
        
        # Initialize container manager state
        st.session_state.container_manager = {
            'configs': {},
            'hierarchy': {}
        }
        
        # Initialize theme manager state
        st.session_state.theme_manager = {
            'mode': 'dark',
            'scheme': 'default',
            'custom_css_applied': False
        }

except Exception as e:
    st.error(f"Failed during session state initialization: {e}")
    import traceback
    st.code(traceback.format_exc())
    st.stop()

# Now import and run
from ui.workflow_ui_v2 import WorkflowUIV2

if __name__ == "__main__":
    try:
        # Display any initialization errors
        if st.session_state.get('initialization_errors'):
            st.error("Some components failed to initialize:")
            for error in st.session_state.initialization_errors:
                st.warning(error)
        
        # Create UI instance (singleton managers will be initialized)
        ui = WorkflowUIV2(debug_mode=True)  # Enable debug mode to see errors
        ui.render_content()  # Use render_content which doesn't set page config again
    except Exception as e:
        st.error(f"Fatal error in UI: {e}")
        import traceback
        st.code(traceback.format_exc())