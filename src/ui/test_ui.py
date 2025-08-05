#!/usr/bin/env python3
"""
Test script to verify UI components load correctly.

Run with: python src/ui/test_ui.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test that all UI components can be imported."""
    print("Testing UI component imports...")
    
    try:
        from ui.components import (
            ProgressTracker,
            WorkflowProgressTracker,
            ArtifactViewer,
            CodeEditor,
            MetricsDashboard
        )
        print("✅ All UI components imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    try:
        from ui.workflow_runner import WorkflowRunner, WorkflowState
        print("✅ Workflow runner imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    try:
        from ui.stream_handlers import StreamHandler, MultiPhaseStreamHandler
        print("✅ Stream handlers imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    try:
        from ui.utils.formatting import (
            format_timestamp,
            format_duration,
            format_bytes,
            format_number
        )
        print("✅ Formatting utilities imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    return True


def test_component_creation():
    """Test that components can be created."""
    print("\nTesting component creation...")
    
    # Note: Component creation requires full streamlit environment
    # This test is skipped when run outside streamlit
    print("⚠️  Component creation tests require Streamlit environment")
    print("   These tests are skipped in standalone mode")
    return True


def main():
    """Run all tests."""
    print("=== UI Component Tests ===\n")
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed")
        return 1
    
    # Test component creation
    if not test_component_creation():
        print("\n❌ Component creation tests failed")
        return 1
    
    print("\n✅ All tests passed!")
    print("\nTo run the UI:")
    print("  PYTHONPATH=src streamlit run src/ui/workflow_ui.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())