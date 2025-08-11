"""
Workflow Specifier AgenTool - Creates detailed specifications for missing tools.

This AgenTool takes the analysis output and generates complete specifications
for each tool that needs to be created, including schemas, operations, and examples.
"""

import json
from typing import Dict, Any, List, Literal, Optional
from pydantic import BaseModel, Field
from pydantic_ai import RunContext, Agent

from agentool import create_agentool, BaseOperationInput
from agentool.core.registry import RoutingConfig
from agentool.core.injector import get_injector, AgenToolInjector

# Import the data models
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from agents.models import SpecificationOutput, ToolSpecification, ToolSpecificationLLM, AnalyzerOutput


class WorkflowSpecifierInput(BaseOperationInput):
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
    operation: str = Field(description="Operation that was performed")
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
    injector: AgenToolInjector = get_injector()
    
    try:
        # Log specification phase start
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'workflow',
            'message': 'Specification phase started',
            'data': {
                'workflow_id': workflow_id,
                'operation': 'specify',
                'model': model
            }
        })
        # Load analysis from storage_kv
        analysis_key = f'workflow/{workflow_id}/analysis'
        analysis_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': analysis_key
        })
        
        # storage_kv returns typed StorageKvOutput
        assert analysis_result.success is True
        if not analysis_result.data.get('exists', False):
            raise ValueError(f"No analysis found for workflow {workflow_id}")
        
        # Parse analysis - the value is in data.value
        analysis = AnalyzerOutput(**json.loads(analysis_result.data['value']))
        
        # Log analysis loaded
        await injector.run('logging', {
            'operation': 'log',
            'level': 'DEBUG',
            'logger_name': 'workflow',
            'message': 'Analysis loaded for specification',
            'data': {
                'workflow_id': workflow_id,
                'missing_tools_count': len(analysis.missing_tools),
                'existing_tools_count': len(analysis.existing_tools)
            }
        })
        
        # If no missing tools, return early
        if not analysis.missing_tools:
            return WorkflowSpecifierOutput(
                success=True,
                operation="specify",
                message="No tools to specify - all required tools exist",
                data={'specifications': []},
                state_ref=f'workflow/{workflow_id}/specs'
            )
        
        # Load system prompt with schema - use LLM version with renamed fields
        template_result = await injector.run('templates', {
            'operation': 'render',
            'template_name': 'system/specification',
            'variables': {
                'schema_json': json.dumps(ToolSpecificationLLM.model_json_schema(), indent=2)
            }
        })
        
        # templates returns typed TemplatesOutput
        assert template_result.success is True
        system_prompt = template_result.data.get('rendered', 'You are an expert AgenTool specification designer.')
        
        # Create LLM agent for specification - use LLM version to avoid OpenAI field conflicts
        agent = Agent(
            model,
            output_type=ToolSpecificationLLM,
            system_prompt=system_prompt,
            retries=3  # Increased retries for better reliability with structured output
        )
        
        # Get and store COMPLETE registry records for existing tools
        existing_tools_refs: List[str] = []
        existing_tools_data: Dict[str, Any] = {}  # Collect all tools data
        
        for tool_name in analysis.existing_tools:
            try:
                # Get FULL registry config
                info_result = await injector.run('agentool_mgmt', {
                    'operation': 'get_agentool_info',
                    'agentool_name': tool_name,
                    'detailed': True
                })
                
                # agentool_mgmt returns typed ManagementOutput
                assert info_result.success is True
                # Store COMPLETE config as-is - no mutation!
                tool_key = f'workflow/{workflow_id}/existing_tools/{tool_name}'
                await injector.run('storage_kv', {
                    'operation': 'set',
                    'key': tool_key,
                    'value': json.dumps(info_result.data['agentool'])  # Full registry record
                })
                existing_tools_refs.append(f'!ref:storage_kv:{tool_key}')
                existing_tools_data[tool_name] = info_result.data['agentool']
                    
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
        
        # Log existing tools collection
        await injector.run('logging', {
            'operation': 'log',
            'level': 'DEBUG',
            'logger_name': 'workflow',
            'message': 'Existing tools data collected',
            'data': {
                'workflow_id': workflow_id,
                'tools_collected': len(existing_tools_data),
                'tool_names': list(existing_tools_data.keys())
            }
        })
        
        # Log LLM specification generation start
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'workflow',
            'message': 'Starting LLM specification generation',
            'data': {
                'workflow_id': workflow_id,
                'model': model,
                'tools_to_specify': len(analysis.missing_tools),
                'tool_names': [tool.name for tool in analysis.missing_tools]
            }
        })
        
        # Generate specification for each missing tool
        specifications: List[ToolSpecification] = []
        
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
            
            # templates returns typed TemplatesOutput
            assert prompt_result.success is True
            user_prompt = prompt_result.data.get('rendered', f"Create specification for {missing_tool.name}")
            
            # Generate specification (returns ToolSpecificationLLM)
            result = await agent.run(user_prompt)
            spec_llm = result.output
            
            # Transform from LLM format to internal format
            spec = spec_llm.to_tool_specification()
            
            # Capture and record token usage
            usage = result.usage()
            
            # Store token metrics with tool label
            await injector.run('metrics', {
                'operation': 'increment',
                'name': 'agentool.workflow.tokens.request',
                'value': usage.request_tokens,
                'labels': {
                    'workflow_id': workflow_id,
                    'agent': 'workflow_specifier',
                    'tool': missing_tool.name,
                    'model': model
                }
            })
            
            await injector.run('metrics', {
                'operation': 'increment',
                'name': 'agentool.workflow.tokens.response',
                'value': usage.response_tokens,
                'labels': {
                    'workflow_id': workflow_id,
                    'agent': 'workflow_specifier',
                    'tool': missing_tool.name,
                    'model': model
                }
            })
            
            await injector.run('metrics', {
                'operation': 'increment',
                'name': 'agentool.workflow.tokens.total',
                'value': usage.total_tokens,
                'labels': {
                    'workflow_id': workflow_id,
                    'agent': 'workflow_specifier',
                    'tool': missing_tool.name,
                    'model': model
                }
            })
            
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
        
        # Log state storage completion
        await injector.run('logging', {
            'operation': 'log',
            'level': 'DEBUG',
            'logger_name': 'workflow',
            'message': 'Specification state stored successfully',
            'data': {
                'workflow_id': workflow_id,
                'state_key': state_key,
                'specifications_stored': len(specifications)
            }
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
            operation="specify",
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
                    "operation": "specify",
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