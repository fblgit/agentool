"""
Workflow Analyzer AgenTool - Analyzes AgenTool catalog for task requirements.

This AgenTool examines the registry to determine what tools exist
and what needs to be created for a given task. It provides the foundation
for the AI code generation workflow by identifying gaps and opportunities.
"""

import json
from typing import Dict, Any, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import RunContext, Agent

from agentool import create_agentool
from agentool.core.registry import RoutingConfig
from agentool.core.injector import get_injector

# Import the data models
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from agents.models import AnalyzerOutput, MissingToolSpec


class WorkflowAnalyzerInput(BaseModel):
    """Input schema for workflow analyzer operations."""
    operation: Literal['analyze'] = Field(
        description="Operation to perform"
    )
    task_description: str = Field(
        description="Description of the task/capability to build"
    )
    workflow_id: str = Field(
        description="Unique workflow identifier for state management"
    )
    model: str = Field(
        default="openai:gpt-4o",
        description="LLM model to use for analysis"
    )
    
    @field_validator('task_description')
    def validate_task_description(cls, v, info):
        """Validate task_description is provided for analyze operation."""
        operation = info.data.get('operation')
        if operation == 'analyze' and not v:
            raise ValueError("task_description is required for analyze operation")
        return v
    
    @field_validator('workflow_id')
    def validate_workflow_id(cls, v, info):
        """Validate workflow_id is provided for analyze operation."""
        operation = info.data.get('operation')
        if operation == 'analyze' and not v:
            raise ValueError("workflow_id is required for analyze operation")
        return v


class WorkflowAnalyzerOutput(BaseModel):
    """Output from workflow analysis."""
    success: bool = Field(description="Whether analysis succeeded")
    message: str = Field(description="Status message")
    data: Dict[str, Any] = Field(description="Analysis results")
    state_ref: str = Field(description="Reference to stored state in storage_kv")


async def analyze_task(
    ctx: RunContext[Any],
    task_description: str,
    workflow_id: str,
    model: str
) -> WorkflowAnalyzerOutput:
    """
    Analyze task requirements against the AgenTool catalog.
    
    This function:
    1. Retrieves the full AgenTool catalog
    2. Uses an LLM to analyze what exists vs what's needed
    3. Stores the analysis in storage_kv for later phases
    4. Returns structured analysis results
    
    Args:
        ctx: Runtime context
        task_description: What the user wants to build
        workflow_id: Unique identifier for this workflow
        model: LLM model to use
        
    Returns:
        Analysis output with existing/missing tools identified
        
    Raises:
        RuntimeError: If catalog retrieval or analysis fails
    """
    injector = get_injector()
    
    try:
        # Get catalog from agentool_mgmt
        catalog_result = await injector.run('agentool_mgmt', {
            'operation': 'export_catalog',
            'format': 'json'
        })
        
        # Extract catalog data
        if hasattr(catalog_result, 'output'):
            catalog_data = json.loads(catalog_result.output)
        else:
            catalog_data = catalog_result.data if hasattr(catalog_result, 'data') else catalog_result
        
        catalog = catalog_data.get('catalog', {})
        
        # Load system prompt from template with schema
        template_result = await injector.run('templates', {
            'operation': 'render',
            'template_name': 'system/analyzer',
            'variables': {
                'schema_json': json.dumps(AnalyzerOutput.model_json_schema(), indent=2)
            }
        })
        
        if hasattr(template_result, 'output'):
            template_data = json.loads(template_result.output)
        else:
            template_data = template_result.data if hasattr(template_result, 'data') else template_result
        
        # Extract rendered content from the data field
        if isinstance(template_data, dict) and 'data' in template_data:
            system_prompt = template_data['data'].get('rendered', 'You are an expert AgenTool analyzer.')
        else:
            system_prompt = template_data.get('rendered', 'You are an expert AgenTool analyzer.')
        
        # Create LLM agent for analysis
        agent = Agent(
            model,
            output_type=AnalyzerOutput,
            system_prompt=system_prompt
        )
        
        # Store complete catalog in storage_kv
        catalog_key = f'workflow/{workflow_id}/catalog'
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': catalog_key,
            'value': json.dumps(catalog)
        })
        
        # Prepare user prompt with reference to catalog
        user_prompt_result = await injector.run('templates', {
            'operation': 'render',
            'template_name': 'prompts/analyze_catalog',
            'variables': {
                'task_description': task_description,
                'catalog': f'!ref:storage_kv:{catalog_key}'
            }
        })
        
        if hasattr(user_prompt_result, 'output'):
            prompt_data = json.loads(user_prompt_result.output)
        else:
            prompt_data = user_prompt_result.data if hasattr(user_prompt_result, 'data') else user_prompt_result
        
        # Extract rendered content from the data field
        if isinstance(prompt_data, dict) and 'data' in prompt_data:
            user_prompt = prompt_data['data'].get('rendered', f"Analyze: {task_description}")
        else:
            user_prompt = prompt_data.get('rendered', f"Analyze: {task_description}")
        
        # Generate analysis using LLM
        result = await agent.run(user_prompt)
        analysis = result.output
        
        # Store analysis in storage_kv
        state_key = f'workflow/{workflow_id}/analysis'
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': state_key,
            'value': json.dumps(analysis.model_dump())
        })
        
        # Store each missing tool individually for easy reference
        for i, missing_tool in enumerate(analysis.missing_tools):
            missing_tool_key = f'workflow/{workflow_id}/missing_tools/{i}'
            await injector.run('storage_kv', {
                'operation': 'set',
                'key': missing_tool_key,
                'value': json.dumps(missing_tool.model_dump())
            })
        
        # Log the analysis
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'workflow',
            'message': 'Analysis phase completed',
            'data': {
                'workflow_id': workflow_id,
                'solution_name': analysis.name,
                'existing_tools_count': len(analysis.existing_tools),
                'missing_tools_count': len(analysis.missing_tools)
            }
        })
        
        return WorkflowAnalyzerOutput(
            success=True,
            message=f"Analysis complete: {len(analysis.existing_tools)} existing tools, {len(analysis.missing_tools)} tools to create",
            data=analysis.model_dump(),
            state_ref=state_key
        )
        
    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        
        # Log error
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'workflow',
            'message': 'Analysis phase failed',
            'data': {
                'workflow_id': workflow_id,
                'error': str(e)
            }
        })
        
        raise RuntimeError(error_msg) from e


# Routing configuration
routing = RoutingConfig(
    operation_field='operation',
    operation_map={
        'analyze': ('analyze_task', lambda x: {
            'task_description': x.task_description,
            'workflow_id': x.workflow_id,
            'model': x.model
        })
    }
)


def create_workflow_analyzer_agent():
    """Create and return the workflow analyzer AgenTool."""
    
    return create_agentool(
        name='workflow_analyzer',
        input_schema=WorkflowAnalyzerInput,
        routing_config=routing,
        tools=[analyze_task],
        output_type=WorkflowAnalyzerOutput,
        use_typed_output=True,  # Enable typed output for workflow_analyzer
        system_prompt="Analyze AgenTool catalog to identify existing tools and gaps for new capabilities.",
        description="Analyzes the AgenTool ecosystem to determine what exists and what needs to be created for a given task",
        version="1.0.0",
        tags=["workflow", "analysis", "catalog", "code-generation"],
        dependencies=["agentool_mgmt", "storage_kv", "templates", "logging", "metrics"],
        examples=[
            {
                "input": {
                    "operation": "analyze",
                    "task_description": "Create a session management system",
                    "workflow_id": "workflow-123"
                },
                "output": {
                    "success": True,
                    "message": "Analysis complete: 3 existing tools, 2 tools to create",
                    "data": {
                        "name": "session_management_system",
                        "existing_tools": ["storage_kv", "auth", "cache"],
                        "missing_tools": [
                            {
                                "name": "session_manager",
                                "description": "Manages user sessions with TTL"
                            }
                        ]
                    }
                }
            }
        ]
    )


# Create the agent instance
agent = create_workflow_analyzer_agent()