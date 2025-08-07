"""AgenTool Analyzer Agent - Phase 1 of the workflow.

This agent analyzes the AgenTool catalog to understand available tools,
identify patterns, and provide recommendations for implementation.
"""

import json
from typing import Dict, Any, List
from agentool.core.injector import get_injector
from .base import BaseAgenToolAgent
from .models import AnalyzerOutput


class AnalyzerAgent(BaseAgenToolAgent[AnalyzerOutput]):
    """Agent for analyzing the AgenTool catalog and identifying patterns."""
    
    def __init__(self, model: str = "openai:gpt-4o"):
        """Initialize the analyzer agent.
        
        Args:
            model: LLM model to use for analysis
        """
        super().__init__(
            model=model,
            system_template="analyzer",
            output_type=AnalyzerOutput,
            name="AgenTool Analyzer"
        )
        
    async def analyze_catalog(
        self,
        task_description: str,
        workflow_id: str
    ) -> AnalyzerOutput:
        """Analyze the AgenTool catalog for the given task.
        
        Args:
            task_description: Description of the task to implement
            workflow_id: Unique identifier for this workflow run
            
        Returns:
            AnalyzerOutput with catalog analysis and recommendations
        """
        injector = get_injector()
        
        # Get the full catalog from agentool_mgmt
        catalog_result = await injector.run('agentool_mgmt', {
            'operation': 'export_catalog',
            'format': 'json'
        })
        
        # Management agent has use_typed_output=True, so we get typed result directly
        catalog = catalog_result.data.get('catalog', {})
        
        # Get detailed information about all tools
        detailed_tools = []
        for tool_name in catalog.get('agentools', []):
            try:
                tool_info_result = await injector.run('agentool_mgmt', {
                    'operation': 'get_agentool_info',
                    'agentool_name': tool_name,
                    'detailed': True
                })
                
                # Management agent has use_typed_output=True, so we get typed result directly
                if tool_info_result.success:
                    detailed_tools.append(tool_info_result.data.get('agentool', {}))
            except Exception as e:
                print(f"Warning: Could not get details for {tool_name}: {e}")
        
        # Prepare variables for the prompt template
        template_variables = {
            'task_description': task_description,
            'catalog': {
                'agentools': detailed_tools,
                'total_count': len(detailed_tools),
                'categories': self._categorize_tools(detailed_tools)
            }
        }
        
        # Generate analysis using the LLM
        analysis = await self.generate(
            user_prompt_template='analyze_catalog',
            variables=template_variables
        )
        
        # Ensure task_description is set in the output
        analysis.task_description = task_description
        
        # Save to state and get reference
        state_ref = await self.save_to_state(
            output=analysis,
            phase='analyzer',
            workflow_id=workflow_id
        )
        
        # Log the analysis completion
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'workflow',
            'message': 'Catalog analysis completed',
            'data': {
                'workflow_id': workflow_id,
                'tools_analyzed': len(detailed_tools),
                'relevant_tools': len(analysis.relevant_tools),
                'state_ref': state_ref
            }
        })
        
        return analysis
    
    def _categorize_tools(self, tools: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Categorize tools by their tags.
        
        Args:
            tools: List of tool information dictionaries
            
        Returns:
            Dictionary mapping categories to tool names
        """
        categories = {}
        
        for tool in tools:
            tool_name = tool.get('name', '')
            tags = tool.get('tags', [])
            
            for tag in tags:
                if tag not in categories:
                    categories[tag] = []
                categories[tag].append(tool_name)
        
        return categories


async def create_analyzer_agent(model: str = "openai:gpt-4o") -> AnalyzerAgent:
    """Create and initialize an analyzer agent.
    
    Args:
        model: LLM model to use
        
    Returns:
        Initialized AnalyzerAgent
    """
    agent = AnalyzerAgent(model)
    await agent.initialize()
    return agent