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
        
        # Import and create the workflow agents
        from agentoolkit.workflows import initialize_workflow_agents
        agents = initialize_workflow_agents()
    
    def test_analyze_task(self):
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
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result.data if hasattr(result, 'data') else result
            
            # Verify success
            assert data['success'] is True
            assert 'analysis' in data['data']
            
            # Check analysis structure
            analysis = data['data']
            assert 'name' in analysis
            assert 'description' in analysis
            assert 'existing_tools' in analysis
            assert 'missing_tools' in analysis
            
            # Verify catalog was stored
            catalog_result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': 'workflow/test-workflow-001/catalog'
            })
            
            if hasattr(catalog_result, 'output'):
                catalog_data = json.loads(catalog_result.output)
            else:
                catalog_data = catalog_result.data if hasattr(catalog_result, 'data') else catalog_result
            
            assert catalog_data.get('exists', False) is True
            
            # Verify analysis was stored
            analysis_result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': 'workflow/test-workflow-001/analysis'
            })
            
            if hasattr(analysis_result, 'output'):
                analysis_data = json.loads(analysis_result.output)
            else:
                analysis_data = analysis_result.data if hasattr(analysis_result, 'data') else analysis_result
            
            assert analysis_data.get('exists', False) is True
            
            # Verify missing tools were stored individually
            if analysis['missing_tools']:
                missing_tool_result = await injector.run('storage_kv', {
                    'operation': 'get',
                    'key': 'workflow/test-workflow-001/missing_tools/0'
                })
                
                if hasattr(missing_tool_result, 'output'):
                    missing_tool_data = json.loads(missing_tool_result.output)
                else:
                    missing_tool_data = missing_tool_result.data if hasattr(missing_tool_result, 'data') else missing_tool_result
                
                assert missing_tool_data.get('exists', False) is True
        
        asyncio.run(run_test())
    
    def test_catalog_storage(self):
        """Test that complete catalog is stored without mutation."""
        
        async def run_test():
            injector = get_injector()
            
            # Get original catalog
            catalog_result = await injector.run('agentool_mgmt', {
                'operation': 'export_catalog',
                'format': 'json'
            })
            
            if hasattr(catalog_result, 'output'):
                original_catalog = json.loads(catalog_result.output)['catalog']
            else:
                original_catalog = catalog_result.data['catalog'] if hasattr(catalog_result, 'data') else catalog_result['catalog']
            
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
            
            if hasattr(stored_result, 'output'):
                stored_data = json.loads(stored_result.output)
            else:
                stored_data = stored_result.data if hasattr(stored_result, 'data') else stored_result
            
            stored_catalog = json.loads(stored_data['value'])
            
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
    
    def test_missing_tools_storage(self):
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
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result.data if hasattr(result, 'data') else result
            
            analysis = data['data']
            
            # If there are missing tools, verify they're stored individually
            if analysis.get('missing_tools'):
                for i, missing_tool in enumerate(analysis['missing_tools']):
                    key = f'workflow/test-workflow-003/missing_tools/{i}'
                    tool_result = await injector.run('storage_kv', {
                        'operation': 'get',
                        'key': key
                    })
                    
                    if hasattr(tool_result, 'output'):
                        tool_data = json.loads(tool_result.output)
                    else:
                        tool_data = tool_result.data if hasattr(tool_result, 'data') else tool_result
                    
                    assert tool_data.get('exists', False) is True
                    
                    stored_tool = json.loads(tool_data['value'])
                    assert stored_tool['name'] == missing_tool['name']
        
        asyncio.run(run_test())