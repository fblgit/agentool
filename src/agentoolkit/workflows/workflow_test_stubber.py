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

from agentool import create_agentool, BaseOperationInput
from agentool.core.registry import RoutingConfig
from agentool.core.injector import get_injector

# Import the data models
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from agents.models import TestStubOutput, TestAnalysisOutput, SpecificationOutput


class WorkflowTestStubberInput(BaseOperationInput):
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
    operation: str = Field(description="Operation that was performed")
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
        # Log test stubbing phase start
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'workflow',
            'message': 'Test stubbing phase started',
            'data': {
                'workflow_id': workflow_id,
                'operation': 'stub',
                'tool_name': tool_name,
                'model': model
            }
        })
        # Load test analysis
        analysis_key = f'workflow/{workflow_id}/test_analysis/{tool_name}'
        analysis_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': analysis_key
        })
        
        # storage_kv returns typed StorageKvOutput
        assert analysis_result.success is True
        if not analysis_result.data.get('exists', False):
            raise ValueError(f"No test analysis found for tool {tool_name}")
        
        test_analysis = TestAnalysisOutput(**json.loads(analysis_result.data['value']))
        
        # Load final code for reference
        validation_key = f'workflow/{workflow_id}/validations/{tool_name}'
        validation_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': validation_key
        })
        
        # storage_kv returns typed StorageKvOutput
        assert validation_result.success is True
        final_code = ""
        if validation_result.data.get('exists', False):
            validation = json.loads(validation_result.data['value'])
            final_code = validation.get('final_code', '')
        
        # Load specifications
        spec_key = f'workflow/{workflow_id}/specifications/{tool_name}'
        spec_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': spec_key
        })
        
        # storage_kv returns typed StorageKvOutput
        assert spec_result.success is True
        specification = None
        if spec_result.data.get('exists', False):
            specification = json.loads(spec_result.data['value'])
        
        # Load all specifications for context
        all_specs_key = f'workflow/{workflow_id}/specs'
        all_specs_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': all_specs_key
        })
        
        # storage_kv returns typed StorageKvOutput
        assert all_specs_result.success is True
        all_specifications = None
        if all_specs_result.data.get('exists', False):
            all_specifications = json.loads(all_specs_result.data['value'])
        
        # Load existing tools
        existing_tools_key = f'workflow/{workflow_id}/existing_tools'
        existing_tools_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': existing_tools_key
        })
        
        # storage_kv returns typed StorageKvOutput
        assert existing_tools_result.success is True
        existing_tools = {}
        if existing_tools_result.data.get('exists', False):
            existing_tools = json.loads(existing_tools_result.data['value'])
        
        # Load test skeleton template
        skeleton_result = await injector.run('templates', {
            'operation': 'render',
            'template_name': 'skeletons/test_comprehensive',
            'variables': {
                'tool_name': tool_name
            }
        })
        
        # templates returns typed TemplatesOutput
        assert skeleton_result.success is True
        skeleton = skeleton_result.data.get('rendered', '')
        
        # Load system prompt
        system_result = await injector.run('templates', {
            'operation': 'render',
            'template_name': 'system/test_stubber',
            'variables': {}
        })
        
        # templates returns typed TemplatesOutput
        assert system_result.success is True
        system_prompt = system_result.data.get('rendered', 'You are an expert test stub creator.')
        
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
        
        # storage_fs returns typed StorageFsOutput
        assert ref_test_result.success is True
        ref_test_key = f'workflow/{workflow_id}/test_stub/reference_test'
        ref_test_content = ref_test_result.data.get('content', '')
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
        
        # templates returns typed TemplatesOutput
        assert prompt_result.success is True
        user_prompt = prompt_result.data.get('rendered', f'Create test stub for {tool_name}')
        
        # Log LLM generation start
        await injector.run('logging', {
            'operation': 'log',
            'level': 'DEBUG',
            'logger_name': 'workflow',
            'message': 'Starting LLM test stub generation',
            'data': {
                'workflow_id': workflow_id,
                'tool_name': tool_name,
                'model': model,
                'prompt_length': len(user_prompt)
            }
        })
        
        # Generate test stub
        result = await agent.run(user_prompt)
        raw_output = result.output
        
        # Capture and record token usage
        usage = result.usage()
        
        # Store token metrics with tool label
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.workflow.tokens.request',
            'value': usage.request_tokens,
            'labels': {
                'workflow_id': workflow_id,
                'agent': 'workflow_test_stubber',
                'tool': tool_name,
                'model': model
            }
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.workflow.tokens.response',
            'value': usage.response_tokens,
            'labels': {
                'workflow_id': workflow_id,
                'agent': 'workflow_test_stubber',
                'tool': tool_name,
                'model': model
            }
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.workflow.tokens.total',
            'value': usage.total_tokens,
            'labels': {
                'workflow_id': workflow_id,
                'agent': 'workflow_test_stubber',
                'tool': tool_name,
                'model': model
            }
        })
        
        # Log LLM generation complete
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'workflow',
            'message': 'LLM test stub generation completed',
            'data': {
                'workflow_id': workflow_id,
                'tool_name': tool_name,
                'output_length': len(raw_output),
                'has_code_block': '```python' in raw_output
            }
        })
        
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
            operation="stub",
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
                    "operation": "stub",
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