"""
Workflow Specifier AgenTool - Creates detailed specifications for missing tools.

This AgenTool takes the analysis output and generates complete specifications
for each tool that needs to be created, including schemas, operations, and examples.
"""

import json
from typing import Dict, Any, List, Literal, Optional
from pydantic import BaseModel, Field
from pydantic_ai import RunContext, Agent

from agentool import create_agentool
from agentool.core.registry import RoutingConfig
from agentool.core.injector import get_injector

# Import the data models
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from agents.models import SpecificationOutput, ToolSpecification, AnalyzerOutput


class WorkflowSpecifierInput(BaseModel):
    """Input schema for workflow specifier operations."""
    operation: Literal['specify'] = Field(
        description="Operation to perform"
    )
    workflow_id: str = Field(
        description="Workflow identifier to retrieve analysis from"
    )
    model: str = Field(
        default="openai:gpt-4o",
        description="LLM model to use for specification"
    )


class WorkflowSpecifierOutput(BaseModel):
    """Output from workflow specification."""
    success: bool = Field(description="Whether specification succeeded")
    message: str = Field(description="Status message")
    data: Dict[str, Any] = Field(description="Specification results")
    state_ref: str = Field(description="Reference to stored state in storage_kv")


async def create_specifications(
    ctx: RunContext[Any],
    workflow_id: str,
    model: str
) -> WorkflowSpecifierOutput:
    """
    Create detailed specifications for missing tools identified in analysis.
    
    This function:
    1. Loads the analysis from storage_kv
    2. For each missing tool, generates a complete specification
    3. Stores specifications for the crafter phase
    4. Returns structured specification results
    
    Args:
        ctx: Runtime context
        workflow_id: Workflow identifier
        model: LLM model to use
        
    Returns:
        Specifications for all missing tools
        
    Raises:
        RuntimeError: If analysis loading or specification fails
    """
    injector = get_injector()
    
    try:
        # Load analysis from storage_kv
        analysis_key = f'workflow/{workflow_id}/analysis'
        analysis_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': analysis_key
        })
        
        if hasattr(analysis_result, 'output'):
            analysis_data = json.loads(analysis_result.output)
        else:
            analysis_data = analysis_result.data if hasattr(analysis_result, 'data') else analysis_result
        
        # Check if the key exists in the data structure
        if not analysis_data.get('data', {}).get('exists', False):
            raise ValueError(f"No analysis found for workflow {workflow_id}")
        
        # Parse analysis - the value is in data.value
        analysis = AnalyzerOutput(**json.loads(analysis_data['data']['value']))
        
        # If no missing tools, return early
        if not analysis.missing_tools:
            return WorkflowSpecifierOutput(
                success=True,
                message="No tools to specify - all required tools exist",
                data={'specifications': []},
                state_ref=f'workflow/{workflow_id}/specs'
            )
        
        # Load system prompt with schema
        template_result = await injector.run('templates', {
            'operation': 'render',
            'template_name': 'system/specification',
            'variables': {
                'schema_json': json.dumps(ToolSpecification.model_json_schema(), indent=2)
            }
        })
        
        if hasattr(template_result, 'output'):
            template_data = json.loads(template_result.output)
        else:
            template_data = template_result.data if hasattr(template_result, 'data') else template_result
        
        # Extract rendered content from the data field
        if isinstance(template_data, dict) and 'data' in template_data:
            system_prompt = template_data['data'].get('rendered', 'You are an expert AgenTool specification designer.')
        else:
            system_prompt = template_data.get('rendered', 'You are an expert AgenTool specification designer.')
        
        # Create LLM agent for specification
        agent = Agent(
            model,
            output_type=ToolSpecification,
            system_prompt=system_prompt
        )
        
        # Get and store COMPLETE registry records for existing tools
        existing_tools_refs = []
        existing_tools_data = {}  # Collect all tools data
        
        for tool_name in analysis.existing_tools:
            try:
                # Get FULL registry config
                info_result = await injector.run('agentool_mgmt', {
                    'operation': 'get_agentool_info',
                    'agentool_name': tool_name,
                    'detailed': True
                })
                
                if hasattr(info_result, 'output'):
                    info_data = json.loads(info_result.output)
                else:
                    info_data = info_result.data if hasattr(info_result, 'data') else info_result
                
                if info_data.get('success'):
                    # Store COMPLETE config as-is - no mutation!
                    tool_key = f'workflow/{workflow_id}/existing_tools/{tool_name}'
                    await injector.run('storage_kv', {
                        'operation': 'set',
                        'key': tool_key,
                        'value': json.dumps(info_data['agentool'])  # Full registry record
                    })
                    existing_tools_refs.append(f'!ref:storage_kv:{tool_key}')
                    existing_tools_data[tool_name] = info_data['agentool']
                    
            except Exception as e:
                # Log but continue
                await injector.run('logging', {
                    'operation': 'log',
                    'level': 'WARN',
                    'logger_name': 'workflow',
                    'message': f'Could not get info for {tool_name}',
                    'data': {'error': str(e)}
                })
        
        # Store consolidated existing tools data for template reference
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': f'workflow/{workflow_id}/existing_tools',
            'value': json.dumps(existing_tools_data)
        })
        
        # Generate specification for each missing tool
        specifications = []
        
        for missing_tool in analysis.missing_tools:
            # Store the missing tool data for template reference
            missing_tool_key = f'workflow/{workflow_id}/current_missing_tool'
            await injector.run('storage_kv', {
                'operation': 'set',
                'key': missing_tool_key,
                'value': json.dumps(missing_tool.model_dump())
            })
            
            # Prepare prompt with references to complete data
            prompt_result = await injector.run('templates', {
                'operation': 'render',
                'template_name': 'prompts/create_specification',
                'variables': {
                    'agentool_to_implement': f'!ref:storage_kv:{missing_tool_key}',
                    'analysis_output': f'!ref:storage_kv:workflow/{workflow_id}/analysis',
                    'existing_tools_schemas': f'!ref:storage_kv:workflow/{workflow_id}/existing_tools'
                }
            })
            
            if hasattr(prompt_result, 'output'):
                prompt_data = json.loads(prompt_result.output)
            else:
                prompt_data = prompt_result.data if hasattr(prompt_result, 'data') else prompt_result
            
            # Extract rendered content from the data field
            if isinstance(prompt_data, dict) and 'data' in prompt_data:
                user_prompt = prompt_data['data'].get('rendered', f"Create specification for {missing_tool.name}")
            else:
                user_prompt = prompt_data.get('rendered', f"Create specification for {missing_tool.name}")
            
            # Generate specification
            result = await agent.run(user_prompt)
            spec = result.data
            
            # Ensure consistency with analysis
            spec.name = missing_tool.name
            spec.required_tools = missing_tool.required_tools
            spec.dependencies = missing_tool.dependencies
            
            specifications.append(spec)
            
            # Store individual specification for later reference
            spec_key = f'workflow/{workflow_id}/specifications/{spec.name}'
            await injector.run('storage_kv', {
                'operation': 'set',
                'key': spec_key,
                'value': json.dumps(spec.model_dump())
            })
            
            # Log progress
            await injector.run('logging', {
                'operation': 'log',
                'level': 'INFO',
                'logger_name': 'workflow',
                'message': f'Specification created for {spec.name}',
                'data': {
                    'workflow_id': workflow_id,
                    'tool_name': spec.name,
                    'operations': len(spec.extended_intents)
                }
            })
        
        # Create output
        spec_output = SpecificationOutput(specifications=specifications)
        
        # Store specifications in storage_kv
        state_key = f'workflow/{workflow_id}/specs'
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': state_key,
            'value': json.dumps(spec_output.model_dump())
        })
        
        # Log completion
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'workflow',
            'message': 'Specification phase completed',
            'data': {
                'workflow_id': workflow_id,
                'specifications_count': len(specifications),
                'tools': [s.name for s in specifications]
            }
        })
        
        return WorkflowSpecifierOutput(
            success=True,
            message=f"Created {len(specifications)} tool specifications",
            data=spec_output.model_dump(),
            state_ref=state_key
        )
        
    except Exception as e:
        error_msg = f"Specification failed: {str(e)}"
        
        # Log error
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'workflow',
            'message': 'Specification phase failed',
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
        'specify': ('create_specifications', lambda x: {
            'workflow_id': x.workflow_id,
            'model': x.model
        })
    }
)


def create_workflow_specifier_agent():
    """Create and return the workflow specifier AgenTool."""
    
    return create_agentool(
        name='workflow_specifier',
        input_schema=WorkflowSpecifierInput,
        routing_config=routing,
        tools=[create_specifications],
        output_type=WorkflowSpecifierOutput,
        system_prompt="Create detailed specifications for AgenTools based on analysis.",
        description="Generates complete specifications for missing tools including schemas, operations, and examples",
        version="1.0.0",
        tags=["workflow", "specification", "design", "code-generation"],
        dependencies=["storage_kv", "templates", "logging", "metrics", "agentool_mgmt"],
        examples=[
            {
                "input": {
                    "operation": "specify",
                    "workflow_id": "workflow-123"
                },
                "output": {
                    "success": True,
                    "message": "Created 2 tool specifications",
                    "data": {
                        "specifications": [
                            {
                                "name": "session_manager",
                                "description": "Manages user sessions with TTL",
                                "input_schema": {},
                                "output_schema": {},
                                "examples": []
                            }
                        ]
                    }
                }
            }
        ]
    )


# Create the agent instance
agent = create_workflow_specifier_agent()