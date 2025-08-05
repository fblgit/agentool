"""
Workflow Test Stubber AgenTool - Creates structured test skeletons.

This AgenTool takes the test analysis and generates well-structured test
skeletons with all setup, real agent creation, imports, and placeholder test methods
ready for implementation following the No-Mocks policy.
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
from agents.models import TestStubOutput, TestAnalysisOutput, SpecificationOutput


class WorkflowTestStubberInput(BaseModel):
    """Input schema for workflow test stubber operations."""
    operation: Literal['stub'] = Field(
        description="Operation to perform"
    )
    workflow_id: str = Field(
        description="Workflow identifier to retrieve test analysis from"
    )
    tool_name: str = Field(
        description="Name of the tool to create test stub for"
    )
    model: str = Field(
        default="openai:gpt-4o",
        description="LLM model to use for stub generation"
    )


class WorkflowTestStubberOutput(BaseModel):
    """Output from workflow test stubbing."""
    success: bool = Field(description="Whether stubbing succeeded")
    message: str = Field(description="Status message")
    data: Dict[str, Any] = Field(description="Test stub results")
    state_ref: str = Field(description="Reference to stored state in storage_kv")


async def create_test_stub(
    ctx: RunContext[Any],
    workflow_id: str,
    tool_name: str,
    model: str
) -> WorkflowTestStubberOutput:
    """
    Create structured test skeleton with all setup and placeholders.
    
    This function:
    1. Loads test analysis from storage_kv
    2. Loads final code and specifications
    3. Uses skeleton template and LLM to generate structured test stub
    4. Stores test stub for implementation phase
    
    Args:
        ctx: Runtime context
        workflow_id: Workflow identifier
        tool_name: Tool to create test stub for
        model: LLM model to use
        
    Returns:
        Test stub with complete structure
        
    Raises:
        RuntimeError: If analysis loading or stub generation fails
    """
    injector = get_injector()
    
    try:
        # Load test analysis
        analysis_key = f'workflow/{workflow_id}/test_analysis/{tool_name}'
        analysis_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': analysis_key
        })
        
        if hasattr(analysis_result, 'output'):
            analysis_data = json.loads(analysis_result.output)
        else:
            analysis_data = analysis_result.data if hasattr(analysis_result, 'data') else analysis_result
        
        if not analysis_data.get('data', {}).get('exists', False):
            raise ValueError(f"No test analysis found for tool {tool_name}")
        
        test_analysis = TestAnalysisOutput(**json.loads(analysis_data['data']['value']))
        
        # Load final code for reference
        validation_key = f'workflow/{workflow_id}/validations/{tool_name}'
        validation_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': validation_key
        })
        
        if hasattr(validation_result, 'output'):
            validation_data = json.loads(validation_result.output)
        else:
            validation_data = validation_result.data if hasattr(validation_result, 'data') else validation_result
        
        final_code = ""
        if validation_data.get('data', {}).get('exists', False):
            validation = json.loads(validation_data['data']['value'])
            final_code = validation.get('final_code', '')
        
        # Load specifications
        spec_key = f'workflow/{workflow_id}/specifications/{tool_name}'
        spec_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': spec_key
        })
        
        if hasattr(spec_result, 'output'):
            spec_data = json.loads(spec_result.output)
        else:
            spec_data = spec_result.data if hasattr(spec_result, 'data') else spec_result
        
        specification = None
        if spec_data.get('data', {}).get('exists', False):
            specification = json.loads(spec_data['data']['value'])
        
        # Load all specifications for context
        all_specs_key = f'workflow/{workflow_id}/specs'
        all_specs_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': all_specs_key
        })
        
        if hasattr(all_specs_result, 'output'):
            all_specs_data = json.loads(all_specs_result.output)
        else:
            all_specs_data = all_specs_result.data if hasattr(all_specs_result, 'data') else all_specs_result
        
        all_specifications = None
        if all_specs_data.get('data', {}).get('exists', False):
            all_specifications = json.loads(all_specs_data['data']['value'])
        
        # Load existing tools
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
        
        # Load test skeleton template
        skeleton_result = await injector.run('templates', {
            'operation': 'render',
            'template_name': 'skeletons/test_comprehensive',
            'variables': {
                'tool_name': tool_name
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
            'template_name': 'system/test_stubber',
            'variables': {}
        })
        
        if hasattr(system_result, 'output'):
            system_data = json.loads(system_result.output)
        else:
            system_data = system_result.data if hasattr(system_result, 'data') else system_result
        
        # Extract rendered content from the data field
        if isinstance(system_data, dict) and 'data' in system_data:
            system_prompt = system_data['data'].get('rendered', 'You are an expert test stub creator.')
        else:
            system_prompt = system_data.get('rendered', 'You are an expert test stub creator.')
        
        # Create LLM agent for stub generation (returns string)
        agent = Agent(
            model,
            output_type=str,  # Raw string output for code
            system_prompt=system_prompt
        )
        
        # Store data for template references
        final_code_key = f'workflow/{workflow_id}/test_stub/final_code/{tool_name}'
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': final_code_key,
            'value': final_code
        })
        
        skeleton_key = f'workflow/{workflow_id}/test_stub/skeleton/{tool_name}'
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': skeleton_key,
            'value': skeleton
        })
        
        # Get reference test structure from test_session.py
        ref_test_result = await injector.run('storage_fs', {
            'operation': 'read',
            'path': 'tests/agentoolkit/test_session.py'
        })
        
        if hasattr(ref_test_result, 'output'):
            ref_test_data = json.loads(ref_test_result.output)
        else:
            ref_test_data = ref_test_result.data if hasattr(ref_test_result, 'data') else ref_test_result
        
        ref_test_key = f'workflow/{workflow_id}/test_stub/reference_test'
        ref_test_content = ref_test_data.get('data', {}).get('content', '')
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': ref_test_key,
            'value': ref_test_content
        })
        
        # Prepare prompt with references
        prompt_result = await injector.run('templates', {
            'operation': 'render',
            'template_name': 'prompts/create_test_stub',
            'variables': {
                'tool_name': tool_name,
                'test_analysis': f'!ref:storage_kv:{analysis_key}',
                'final_code': f'!ref:storage_kv:{final_code_key}',
                'specification': f'!ref:storage_kv:{spec_key}',
                'all_specifications': f'!ref:storage_kv:{all_specs_key}',
                'existing_tools': f'!ref:storage_kv:{existing_tools_key}',
                'skeleton': f'!ref:storage_kv:{skeleton_key}',
                'reference_test': f'!ref:storage_kv:{ref_test_key}'
            }
        })
        
        if hasattr(prompt_result, 'output'):
            prompt_data = json.loads(prompt_result.output)
        else:
            prompt_data = prompt_result.data if hasattr(prompt_result, 'data') else prompt_result
        
        # Extract rendered content from the data field
        if isinstance(prompt_data, dict) and 'data' in prompt_data:
            user_prompt = prompt_data['data'].get('rendered', f'Create test stub for {tool_name}')
        else:
            user_prompt = prompt_data.get('rendered', f'Create test stub for {tool_name}')
        
        # Generate test stub
        result = await agent.run(user_prompt)
        raw_output = result.output
        
        # Extract code from markdown code block
        import re
        code_match = re.search(r'```python\n(.*?)```', raw_output, re.DOTALL)
        if code_match:
            stub_code = code_match.group(1).strip()
        else:
            # Fallback if no code block found
            stub_code = raw_output.strip()
        
        # Count placeholders
        placeholder_count = stub_code.count('# TODO:') + stub_code.count('pass  # TODO')
        
        # Check structure elements
        structure_elements = {
            "imports": "import" in stub_code and "from agentool" in stub_code,
            "agent_creation": "create_" in stub_code and "agent" in stub_code,
            "setup_teardown": "setup_method" in stub_code and "clear()" in stub_code,
            "test_placeholders": "def test_" in stub_code,
            "real_dependencies": "injector.run" in stub_code or "await injector.run" in stub_code
        }
        
        # Create TestStubOutput
        test_stub = TestStubOutput(
            code=stub_code,
            file_path=f"test_{tool_name}.py",
            placeholders_count=placeholder_count,
            structure_elements=structure_elements
        )
        
        # Store test stub in storage_kv
        state_key = f'workflow/{workflow_id}/test_stub/{tool_name}'
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': state_key,
            'value': json.dumps(test_stub.model_dump())
        })
        
        # Save to file system
        file_path = f"generated/{workflow_id}/test_stubs/{test_stub.file_path}"
        await injector.run('storage_fs', {
            'operation': 'write',
            'path': file_path,
            'content': test_stub.code,
            'create_parents': True
        })
        
        # Log completion
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'workflow',
            'message': f'Test stub created for {tool_name}',
            'data': {
                'workflow_id': workflow_id,
                'tool_name': tool_name,
                'file_path': file_path,
                'placeholders_count': placeholder_count,
                'structure_complete': all(structure_elements.values()),
                'test_cases': len(test_analysis.test_cases)
            }
        })
        
        return WorkflowTestStubberOutput(
            success=True,
            message=f"Test stub created with {placeholder_count} placeholders for {len(test_analysis.test_cases)} test cases",
            data=test_stub.model_dump(),
            state_ref=state_key
        )
        
    except Exception as e:
        error_msg = f"Test stub generation failed: {str(e)}"
        
        # Log error
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'workflow',
            'message': 'Test stub generation phase failed',
            'data': {
                'workflow_id': workflow_id,
                'tool_name': tool_name,
                'error': str(e)
            }
        })
        
        raise RuntimeError(error_msg) from e


# Routing configuration
routing = RoutingConfig(
    operation_field='operation',
    operation_map={
        'stub': ('create_test_stub', lambda x: {
            'workflow_id': x.workflow_id,
            'tool_name': x.tool_name,
            'model': x.model
        })
    }
)


def create_workflow_test_stubber_agent():
    """Create and return the workflow test stubber AgenTool."""
    
    return create_agentool(
        name='workflow_test_stubber',
        input_schema=WorkflowTestStubberInput,
        routing_config=routing,
        tools=[create_test_stub],
        output_type=WorkflowTestStubberOutput,
        system_prompt="Create well-structured test skeletons with proper setup and placeholders.",
        description="Generates structured test files with imports, fixtures, setup, and test method placeholders",
        version="1.0.0",
        tags=["workflow", "testing", "stub", "skeleton", "code-generation"],
        dependencies=["storage_kv", "storage_fs", "templates", "logging", "metrics"],
        examples=[
            {
                "input": {
                    "operation": "stub",
                    "workflow_id": "workflow-123",
                    "tool_name": "session_manager"
                },
                "output": {
                    "success": True,
                    "message": "Test stub created with 15 placeholders for 15 test cases",
                    "data": {
                        "code": "# Test implementation stub...",
                        "file_path": "test_session_manager.py",
                        "placeholders_count": 15,
                        "structure_elements": {
                            "imports": True,
                            "agent_creation": True,
                            "setup_teardown": True,
                            "test_placeholders": True,
                            "real_dependencies": True
                        }
                    }
                }
            }
        ]
    )


# Create the agent instance
agent = create_workflow_test_stubber_agent()