"""
Workflow Crafter AgenTool - Generates implementation code for AgenTools.

This AgenTool takes the specifications and generates complete, production-ready
AgenTool implementations following all best practices and patterns.
"""

import json
from typing import Dict, Any, List, Literal
from pydantic import BaseModel, Field
from pydantic_ai import RunContext, Agent

from agentool import create_agentool
from agentool.core.registry import RoutingConfig
from agentool.core.injector import get_injector

# Import the data models
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from agents.models import CodeOutput, AnalyzerOutput, SpecificationOutput, ExistingToolInfo


class WorkflowCrafterInput(BaseModel):
    """Input schema for workflow crafter operations."""
    operation: Literal['craft', 'craft_multi'] = Field(
        description="Operation to perform (craft single or multiple tools)"
    )
    workflow_id: str = Field(
        description="Workflow identifier to retrieve specs from"
    )
    model: str = Field(
        default="openai:gpt-4o",
        description="LLM model to use for code generation"
    )


class WorkflowCrafterOutput(BaseModel):
    """Output from workflow code generation."""
    success: bool = Field(description="Whether code generation succeeded")
    message: str = Field(description="Status message")
    data: Dict[str, Any] = Field(description="Generated code and metadata")
    state_ref: str = Field(description="Reference to stored state in storage_kv")


# Removed load_existing_tools_schemas function - we use references now


async def update_tool_transition(injector: Any, workflow_id: str, tool_name: str, analysis: AnalyzerOutput) -> None:
    """Update the analysis to transition a tool from missing to existing.
    
    This updates both:
    1. The analysis data to move the tool from missing_tools to existing_tools
    2. The existing_tools storage with the new tool's schema
    """
    # Create a mock registry entry for the newly crafted tool
    new_tool_registry = {
        "name": tool_name,
        "description": f"Generated {tool_name} AgenTool",
        "version": "1.0.0",
        "input_schema": {},  # Will be populated from spec
        "output_schema": {},  # Will be populated from spec
        "examples": [],
        "tags": ["generated", "workflow"],
        "dependencies": []
    }
    
    # Load the specification for this tool to get schemas
    spec_key = f'workflow/{workflow_id}/specifications/{tool_name}'
    spec_result = await injector.run('storage_kv', {
        'operation': 'get',
        'key': spec_key
    })
    
    if hasattr(spec_result, 'output'):
        spec_data = json.loads(spec_result.output)
    else:
        spec_data = spec_result.data if hasattr(spec_result, 'data') else spec_result
    
    if spec_data.get('data', {}).get('exists', False):
        spec = json.loads(spec_data['data']['value'])
        new_tool_registry['description'] = spec.get('description', new_tool_registry['description'])
        new_tool_registry['input_schema'] = spec.get('input_schema', {})
        new_tool_registry['output_schema'] = spec.get('output_schema', {})
        new_tool_registry['examples'] = spec.get('examples', [])
        new_tool_registry['dependencies'] = spec.get('dependencies', [])
    
    # Store the new tool in existing tools
    tool_key = f'workflow/{workflow_id}/existing_tools/{tool_name}'
    await injector.run('storage_kv', {
        'operation': 'set',
        'key': tool_key,
        'value': json.dumps(new_tool_registry)
    })
    
    # Update the existing_tools consolidated data
    existing_tools_key = f'workflow/{workflow_id}/existing_tools'
    existing_tools_result = await injector.run('storage_kv', {
        'operation': 'get',
        'key': existing_tools_key
    })
    
    if hasattr(existing_tools_result, 'output'):
        existing_tools_data = json.loads(existing_tools_result.output)
    else:
        existing_tools_data = existing_tools_result.data if hasattr(existing_tools_result, 'data') else existing_tools_result
    
    existing_tools = {}
    if existing_tools_data.get('data', {}).get('exists', False):
        existing_tools = json.loads(existing_tools_data['data']['value'])
    
    # Add the new tool
    existing_tools[tool_name] = new_tool_registry
    
    # Store updated existing tools
    await injector.run('storage_kv', {
        'operation': 'set',
        'key': existing_tools_key,
        'value': json.dumps(existing_tools)
    })
    
    # Log the transition
    await injector.run('logging', {
        'operation': 'log',
        'level': 'INFO',
        'logger_name': 'workflow',
        'message': f'Tool {tool_name} transitioned from missing to existing',
        'data': {
            'workflow_id': workflow_id,
            'tool_name': tool_name,
            'existing_tools_count': len(existing_tools)
        }
    })


async def craft_single_tool(
    injector: Any,
    workflow_id: str,
    model: str,
    tool_spec: Any,
    analysis: AnalyzerOutput,
    spec_output: SpecificationOutput
) -> Dict[str, Any]:
    """Helper function to craft a single tool implementation."""
    # Find corresponding missing tool
    missing_tool = None
    for mt in analysis.missing_tools:
        if mt.name == tool_spec.name:
            missing_tool = mt
            break
    
    if not missing_tool:
        missing_tool = analysis.missing_tools[0] if analysis.missing_tools else None
    
    # Load comprehensive skeleton template
    skeleton_result = await injector.run('templates', {
        'operation': 'render',
        'template_name': 'skeletons/agentool_comprehensive',
        'variables': {
            'tool_name': tool_spec.name
        }
    })
    
    if hasattr(skeleton_result, 'output'):
        skeleton_data = json.loads(skeleton_result.output)
    else:
        skeleton_data = skeleton_result.data if hasattr(skeleton_result, 'data') else skeleton_result
    
    # Extract rendered content from the data field
    if isinstance(skeleton_data, dict) and 'data' in skeleton_data:
        skeleton = skeleton_data['data'].get('rendered', '')
    else:
        skeleton = skeleton_data.get('rendered', '')
    
    # Load system prompt
    system_result = await injector.run('templates', {
        'operation': 'render',
        'template_name': 'system/crafter',
        'variables': {}
    })
    
    if hasattr(system_result, 'output'):
        system_data = json.loads(system_result.output)
    else:
        system_data = system_result.data if hasattr(system_result, 'data') else system_result
    
    # Extract rendered content from the data field
    if isinstance(system_data, dict) and 'data' in system_data:
        system_prompt = system_data['data'].get('rendered', 'You are an expert AgenTool implementation crafter.')
    else:
        system_prompt = system_data.get('rendered', 'You are an expert AgenTool implementation crafter.')
    
    # Create LLM agent for code generation (returns string)
    agent = Agent(
        model,
        output_type=str,  # Raw string output for code
        system_prompt=system_prompt
    )
    
    # Store the missing tool data for template reference
    missing_tool_key = f'workflow/{workflow_id}/current_missing_tool_for_craft/{tool_spec.name}'
    await injector.run('storage_kv', {
        'operation': 'set',
        'key': missing_tool_key,
        'value': json.dumps(missing_tool.model_dump() if missing_tool else {'name': tool_spec.name})
    })
    
    # Store the skeleton for template reference
    skeleton_key = f'workflow/{workflow_id}/skeleton/{tool_spec.name}'
    await injector.run('storage_kv', {
        'operation': 'set',
        'key': skeleton_key,
        'value': skeleton
    })
    
    # Prepare prompt with pure references including ALL specifications
    prompt_result = await injector.run('templates', {
        'operation': 'render',
        'template_name': 'prompts/craft_implementation',
        'variables': {
            'agentool_to_implement': f'!ref:storage_kv:{missing_tool_key}',
            'existing_tools_schemas': f'!ref:storage_kv:workflow/{workflow_id}/existing_tools',
            'spec_output': f'!ref:storage_kv:workflow/{workflow_id}/specifications/{tool_spec.name}',
            'all_specifications': f'!ref:storage_kv:workflow/{workflow_id}/specs',  # Pass ALL specs
            'analysis_output': f'!ref:storage_kv:workflow/{workflow_id}/analysis',
            'skeleton': f'!ref:storage_kv:{skeleton_key}'
        }
    })
    
    if hasattr(prompt_result, 'output'):
        prompt_data = json.loads(prompt_result.output)
    else:
        prompt_data = prompt_result.data if hasattr(prompt_result, 'data') else prompt_result
    
    # Extract rendered content from the data field
    if isinstance(prompt_data, dict) and 'data' in prompt_data:
        user_prompt = prompt_data['data'].get('rendered', 'Generate AgenTool implementation')
    else:
        user_prompt = prompt_data.get('rendered', 'Generate AgenTool implementation')
    
    # Generate implementation
    result = await agent.run(user_prompt)
    raw_output = result.output
    
    # Extract code from markdown code block
    import re
    code_match = re.search(r'```python\n(.*?)```', raw_output, re.DOTALL)
    if code_match:
        generated_code = code_match.group(1).strip()
    else:
        # Fallback if no code block found
        generated_code = raw_output.strip()
    
    # Create CodeOutput object with simplified path
    code_output = CodeOutput(
        code=generated_code,
        file_path=f"{tool_spec.name}.py"
    )
    
    # Store generated code in storage_kv
    state_key = f'workflow/{workflow_id}/implementations/{tool_spec.name}'
    await injector.run('storage_kv', {
        'operation': 'set',
        'key': state_key,
        'value': json.dumps(code_output.model_dump())
    })
    
    # Also save to file system with cleaner path structure
    file_path = f"generated/{workflow_id}/src/{code_output.file_path}"
    await injector.run('storage_fs', {
        'operation': 'write',
        'path': file_path,
        'content': code_output.code,
        'create_parents': True
    })
    
    # Log completion
    await injector.run('logging', {
        'operation': 'log',
        'level': 'INFO',
        'logger_name': 'workflow',
        'message': f'Code generation completed for {tool_spec.name}',
        'data': {
            'workflow_id': workflow_id,
            'tool_name': tool_spec.name,
            'file_path': file_path,
            'code_length': len(code_output.code),
            'lines': code_output.code.count('\n')
        }
    })
    
    # Update the analysis to transition this tool from missing to existing
    await update_tool_transition(injector, workflow_id, tool_spec.name, analysis)
    
    return {
        'tool_name': tool_spec.name,
        'file_path': file_path,
        'state_key': state_key,
        'code_output': code_output
    }


async def craft_implementation(
    ctx: RunContext[Any],
    workflow_id: str,
    model: str
) -> WorkflowCrafterOutput:
    """
    Generate complete AgenTool implementation from specifications.
    
    This function:
    1. Loads analysis and specifications from storage_kv
    2. Gets full schemas for existing tools
    3. Loads comprehensive skeleton template
    4. Uses LLM with full context to generate code
    5. Stores generated code
    
    Args:
        ctx: Runtime context
        workflow_id: Workflow identifier
        model: LLM model to use
        
    Returns:
        Generated implementation code
        
    Raises:
        RuntimeError: If loading data or generation fails
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
        
        if not analysis_data.get('data', {}).get('exists', False):
            raise ValueError(f"No analysis found for workflow {workflow_id}")
        
        analysis = AnalyzerOutput(**json.loads(analysis_data['data']['value']))
        
        # Load specifications
        specs_key = f'workflow/{workflow_id}/specs'
        specs_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': specs_key
        })
        
        if hasattr(specs_result, 'output'):
            specs_data = json.loads(specs_result.output)
        else:
            specs_data = specs_result.data if hasattr(specs_result, 'data') else specs_result
        
        if not specs_data.get('data', {}).get('exists', False):
            raise ValueError(f"No specifications found for workflow {workflow_id}")
        
        spec_output = SpecificationOutput(**json.loads(specs_data['data']['value']))
        
        # For each tool to implement, generate code
        if not spec_output.specifications:
            raise ValueError("No specifications to implement")
        
        # Iterate through all specifications and craft implementations
        implementations = []
        for tool_spec in spec_output.specifications:
            result = await craft_single_tool(
                injector=injector,
                workflow_id=workflow_id,
                model=model,
                tool_spec=tool_spec,
                analysis=analysis,
                spec_output=spec_output
            )
            implementations.append(result)
            
            # Log progress
            await injector.run('logging', {
                'operation': 'log',
                'level': 'INFO',
                'logger_name': 'workflow',
                'message': f'Implementation crafted for {tool_spec.name}',
                'data': {
                    'workflow_id': workflow_id,
                    'tool_name': tool_spec.name,
                    'progress': f'{len(implementations)}/{len(spec_output.specifications)}'
                }
            })
        
        # Prepare summary data
        summary_data = {
            'implementations': [{
                'tool_name': impl['tool_name'],
                'file_path': impl['file_path'],
                'state_key': impl['state_key'],
                'lines': impl['code_output'].code.count('\n')
            } for impl in implementations],
            'total_tools': len(implementations),
            'total_lines': sum(impl['code_output'].code.count('\n') for impl in implementations),
            'files': [impl['file_path'] for impl in implementations]
        }
        
        # Store summary in storage_kv
        summary_key = f'workflow/{workflow_id}/implementations_summary'
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': summary_key,
            'value': json.dumps(summary_data)
        })
        
        # Log overall completion
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'workflow',
            'message': 'All implementations completed',
            'data': {
                'workflow_id': workflow_id,
                'total_tools': len(implementations),
                'tools': [impl['tool_name'] for impl in implementations]
            }
        })
        
        return WorkflowCrafterOutput(
            success=True,
            message=f"Generated {len(implementations)} implementations",
            data=summary_data,
            state_ref=summary_key
        )
        
    except Exception as e:
        error_msg = f"Code generation failed: {str(e)}"
        
        # Log error
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'workflow',
            'message': 'Code generation phase failed',
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
        'craft': ('craft_implementation', lambda x: {
            'workflow_id': x.workflow_id,
            'model': x.model
        }),
        'craft_multi': ('craft_implementation', lambda x: {
            'workflow_id': x.workflow_id,
            'model': x.model
        })
    }
)


def create_workflow_crafter_agent():
    """Create and return the workflow crafter AgenTool."""
    
    return create_agentool(
        name='workflow_crafter',
        input_schema=WorkflowCrafterInput,
        routing_config=routing,
        tools=[craft_implementation],
        output_type=WorkflowCrafterOutput,
        system_prompt="Generate production-ready AgenTool implementations from specifications.",
        description="Crafts complete AgenTool code following best practices and patterns",
        version="1.0.0",
        tags=["workflow", "code-generation", "implementation", "crafter"],
        dependencies=["storage_kv", "storage_fs", "templates", "logging", "agentool_mgmt", "metrics"],
        examples=[
            {
                "input": {
                    "operation": "craft",
                    "workflow_id": "workflow-123"
                },
                "output": {
                    "success": True,
                    "message": "Generated implementation: src/agentoolkit/generated/session_manager.py",
                    "data": {
                        "code": "# Complete implementation...",
                        "file_path": "src/agentoolkit/generated/session_manager.py"
                    }
                }
            }
        ]
    )


# Create the agent instance
agent = create_workflow_crafter_agent()