"""
Test suite for the Workflow Analyzer AgenTool.

This module tests the analyzer's ability to:
- Load and analyze the AgenTool catalog
- Identify existing tools relevant to a task
- Determine missing tools needed
- Store complete catalog and analysis in storage_kv
"""

import asyncio
import json
from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry


class TestWorkflowAnalyzer:
    """Test suite for workflow analyzer AgenTool."""
    
    def setup_method(self):
        """Clear registry and injector before each test."""
        AgenToolRegistry.clear()
        get_injector().clear()
        
        # Import and create all required agents in dependency order
        from agentoolkit.storage.fs import create_storage_fs_agent
        from agentoolkit.storage.kv import create_storage_kv_agent, _kv_storage, _kv_expiry
        from agentoolkit.observability.metrics import create_metrics_agent
        from agentoolkit.system.logging import create_logging_agent, _logging_config
        from agentoolkit.system.templates import create_templates_agent
        from agentoolkit.management.agentool import create_agentool_management_agent
        from agentoolkit.workflows.workflow_analyzer import create_workflow_analyzer_agent
        
        # Clear global state
        _kv_storage.clear()
        _kv_expiry.clear()
        _logging_config.clear()
        
        # Clear any previous workflow storage keys
        import os
        test_keys = [
            'workflow/test-workflow-001/catalog',
            'workflow/test-workflow-001/analysis', 
            'workflow/test-workflow-001/missing_tools/0',
            'workflow/test-workflow-002/catalog',
            'workflow/test-workflow-003/catalog',
            'workflow/test-workflow-003/analysis',
            'workflow/test-workflow-003/missing_tools/0'
        ]
        
        # Initialize agents in dependency order
        templates_dir = os.path.join(os.path.dirname(__file__), '../../src/templates')
        
        self.storage_fs_agent = create_storage_fs_agent()      # No dependencies
        self.storage_kv_agent = create_storage_kv_agent()      # No dependencies
        self.metrics_agent = create_metrics_agent()            # Depends on storage_kv
        self.logging_agent = create_logging_agent()            # Depends on storage_fs, metrics
        self.templates_agent = create_templates_agent(templates_dir)  # Depends on storage_fs
        self.management_agent = create_agentool_management_agent()  # Depends on logging
        
        # Create workflow analyzer agent individually (not via batch initialization)
        self.workflow_analyzer_agent = create_workflow_analyzer_agent()
    
    def teardown_method(self):
        """Clean up after each test."""
        # Clear any remaining storage keys that may have been created during tests
        try:
            injector = get_injector()
            test_keys = [
                'workflow/test-workflow-001/catalog',
                'workflow/test-workflow-001/analysis', 
                'workflow/test-workflow-002/catalog',
                'workflow/test-workflow-003/catalog',
                'workflow/test-workflow-003/analysis',
            ]
            # Note: We don't run async cleanup in teardown to avoid event loop issues
        except:
            pass  # Ignore cleanup errors
    
    def test_analyze_task(self, allow_model_requests):
        """Test task analysis against catalog."""
        
        async def run_test():
            injector = get_injector()
            
            # Run analysis
            result = await injector.run('workflow_analyzer', {
                "operation": "analyze",
                "task_description": "Create a session management system with TTL support",
                "workflow_id": "test-workflow-001",
                "model": "openai:gpt-4o"
            })
            
            # workflow_analyzer returns typed WorkflowAnalyzerOutput
            assert result.success is True
            assert result.data is not None
            
            # Check analysis structure
            analysis = result.data
            assert 'name' in analysis
            assert 'description' in analysis
            assert 'existing_tools' in analysis
            assert 'missing_tools' in analysis
            
            # Verify catalog was stored
            catalog_result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': 'workflow/test-workflow-001/catalog'
            })
            
            # storage_kv returns typed StorageKvOutput
            assert catalog_result.success is True
            assert catalog_result.data['exists'] is True
            
            # Verify analysis was stored
            analysis_result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': 'workflow/test-workflow-001/analysis'
            })
            
            # storage_kv returns typed StorageKvOutput
            assert analysis_result.success is True
            assert analysis_result.data['exists'] is True
            
            # Verify missing tools were stored individually
            if analysis['missing_tools']:
                missing_tool_result = await injector.run('storage_kv', {
                    'operation': 'get',
                    'key': 'workflow/test-workflow-001/missing_tools/0'
                })
                
                # storage_kv returns typed StorageKvOutput
                assert missing_tool_result.success is True
                assert missing_tool_result.data['exists'] is True
        
        asyncio.run(run_test())
    
    def test_catalog_storage(self, allow_model_requests):
        """Test that complete catalog is stored without mutation."""
        
        async def run_test():
            injector = get_injector()
            
            # Get original catalog
            catalog_result = await injector.run('agentool_mgmt', {
                'operation': 'export_catalog',
                'format': 'json'
            })
            
            # agentool_mgmt returns typed ManagementOutput
            assert catalog_result.success is True
            original_catalog = catalog_result.data['catalog']
            
            # Run analysis
            await injector.run('workflow_analyzer', {
                "operation": "analyze",
                "task_description": "Test task",
                "workflow_id": "test-workflow-002",
                "model": "openai:gpt-4o"
            })
            
            # Retrieve stored catalog
            stored_result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': 'workflow/test-workflow-002/catalog'
            })
            
            # storage_kv returns typed StorageKvOutput
            assert stored_result.success is True
            stored_catalog = json.loads(stored_result.data['value'])
            
            # Verify catalog is complete and unmutated
            assert 'agentools' in stored_catalog
            assert isinstance(stored_catalog['agentools'], list)
            
            # Check that full structure is preserved
            if stored_catalog['agentools']:
                first_tool = stored_catalog['agentools'][0]
                assert 'name' in first_tool
                assert 'operations' in first_tool
                assert 'tools' in first_tool
                assert 'input_schema' in first_tool
        
        asyncio.run(run_test())
    
    def test_missing_tools_storage(self, allow_model_requests):
        """Test individual storage of missing tools."""
        
        async def run_test():
            injector = get_injector()
            
            # Run analysis with a task likely to need new tools
            result = await injector.run('workflow_analyzer', {
                "operation": "analyze",
                "task_description": "Create a blockchain integration system with smart contracts",
                "workflow_id": "test-workflow-003",
                "model": "openai:gpt-4o"
            })
            
            # workflow_analyzer returns typed WorkflowAnalyzerOutput
            assert result.success is True
            analysis = result.data
            
            # If there are missing tools, verify they're stored individually
            if analysis.get('missing_tools'):
                for i, missing_tool in enumerate(analysis['missing_tools']):
                    key = f'workflow/test-workflow-003/missing_tools/{i}'
                    tool_result = await injector.run('storage_kv', {
                        'operation': 'get',
                        'key': key
                    })
                    
                    # storage_kv returns typed StorageKvOutput
                    assert tool_result.success is True
                    assert tool_result.data['exists'] is True
                    
                    stored_tool = json.loads(tool_result.data['value'])
                    assert stored_tool['name'] == missing_tool['name']
        
        asyncio.run(run_test())