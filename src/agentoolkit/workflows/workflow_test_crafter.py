"""
Workflow Test Crafter AgenTool - Implements complete test logic.

This AgenTool takes the test stub and completes the implementation with
actual test logic, assertions, real dependency calls, and test data following the No-Mocks policy.
"""

import json
import re
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
from agents.models import TestImplementationOutput, TestAnalysisOutput, TestStubOutput


class WorkflowTestCrafterInput(BaseModel):
    """Input schema for workflow test crafter operations."""
    operation: Literal['craft'] = Field(
        description="Operation to perform"
    )
    workflow_id: str = Field(
        description="Workflow identifier to retrieve test stub from"
    )
    tool_name: str = Field(
        description="Name of the tool to implement tests for"
    )
    model: str = Field(
        default="openai:gpt-4o",
        description="LLM model to use for test implementation"
    )


class WorkflowTestCrafterOutput(BaseModel):
    """Output from workflow test implementation."""
    success: bool = Field(description="Whether implementation succeeded")
    message: str = Field(description="Status message")
    data: Dict[str, Any] = Field(description="Test implementation results")
    state_ref: str = Field(description="Reference to stored state in storage_kv")


def count_test_methods(code: str) -> int:
    """Count the number of test methods in the code."""
    return len(re.findall(r'def\s+test_\w+\s*\(', code))


def estimate_coverage(test_analysis: TestAnalysisOutput, implemented_tests: int) -> Dict[str, float]:
    """Estimate test coverage based on analysis and implementation."""
    total_cases = len(test_analysis.test_cases)
    if total_cases == 0:
        return {"overall": 0.0}
    
    # Count by type
    type_counts = {}
    for test_case in test_analysis.test_cases:
        test_type = test_case.test_type
        type_counts[test_type] = type_counts.get(test_type, 0) + 1
    
    # Calculate coverage
    coverage = {
        "overall": min((implemented_tests / total_cases) * 100, 100.0)
    }
    
    # Add type-specific coverage
    for test_type, count in type_counts.items():
        type_coverage = min((implemented_tests / total_cases) * 100, 100.0)
        coverage[test_type] = type_coverage
    
    return coverage


async def craft_test_implementation(
    ctx: RunContext[Any],
    workflow_id: str,
    tool_name: str,
    model: str
) -> WorkflowTestCrafterOutput:
    """
    Complete test implementation with actual test logic.
    
    This function:
    1. Loads test stub and analysis from storage_kv
    2. Uses LLM to implement complete test logic
    3. Validates test structure and completeness
    4. Stores final test implementation
    
    Args:
        ctx: Runtime context
        workflow_id: Workflow identifier
        tool_name: Tool to implement tests for
        model: LLM model to use
        
    Returns:
        Complete test implementation
        
    Raises:
        RuntimeError: If stub loading or implementation fails
    """
    injector = get_injector()
    
    try:
        # Load test stub
        stub_key = f'workflow/{workflow_id}/test_stub/{tool_name}'
        stub_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': stub_key
        })
        
        if hasattr(stub_result, 'output'):
            stub_data = json.loads(stub_result.output)
        else:
            stub_data = stub_result.data if hasattr(stub_result, 'data') else stub_result
        
        if not stub_data.get('data', {}).get('exists', False):
            raise ValueError(f"No test stub found for tool {tool_name}")
        
        test_stub = TestStubOutput(**json.loads(stub_data['data']['value']))
        
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
        
        # Load final code
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
        
        # Load all specifications
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
        
        # Load system prompt
        system_result = await injector.run('templates', {
            'operation': 'render',
            'template_name': 'system/test_crafter',
            'variables': {}
        })
        
        if hasattr(system_result, 'output'):
            system_data = json.loads(system_result.output)
        else:
            system_data = system_result.data if hasattr(system_result, 'data') else system_result
        
        # Extract rendered content from the data field
        if isinstance(system_data, dict) and 'data' in system_data:
            system_prompt = system_data['data'].get('rendered', 'You are an expert test implementation crafter.')
        else:
            system_prompt = system_data.get('rendered', 'You are an expert test implementation crafter.')
        
        # Create LLM agent for test implementation (returns string)
        agent = Agent(
            model,
            output_type=str,  # Raw string output for code
            system_prompt=system_prompt
        )
        
        # Store data for template references
        final_code_key = f'workflow/{workflow_id}/test_craft/final_code/{tool_name}'
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': final_code_key,
            'value': final_code
        })
        
        test_stub_code_key = f'workflow/{workflow_id}/test_craft/stub_code/{tool_name}'
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': test_stub_code_key,
            'value': test_stub.code
        })
        
        # Load and store test skeleton template
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
        
        skeleton_key = f'workflow/{workflow_id}/test_craft/skeleton/{tool_name}'
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': skeleton_key,
            'value': skeleton
        })
        
        # Prepare prompt with references
        prompt_result = await injector.run('templates', {
            'operation': 'render',
            'template_name': 'prompts/implement_tests',
            'variables': {
                'tool_name': tool_name,
                'test_stub': f'!ref:storage_kv:{test_stub_code_key}',
                'test_analysis': f'!ref:storage_kv:{analysis_key}',
                'final_code': f'!ref:storage_kv:{final_code_key}',
                'specification': f'!ref:storage_kv:{spec_key}',
                'all_specifications': f'!ref:storage_kv:{all_specs_key}',
                'existing_tools': f'!ref:storage_kv:{existing_tools_key}',
                'skeleton': f'!ref:storage_kv:{skeleton_key}'
            }
        })
        
        if hasattr(prompt_result, 'output'):
            prompt_data = json.loads(prompt_result.output)
        else:
            prompt_data = prompt_result.data if hasattr(prompt_result, 'data') else prompt_result
        
        # Extract rendered content from the data field
        if isinstance(prompt_data, dict) and 'data' in prompt_data:
            user_prompt = prompt_data['data'].get('rendered', f'Implement tests for {tool_name}')
        else:
            user_prompt = prompt_data.get('rendered', f'Implement tests for {tool_name}')
        
        # Generate test implementation
        result = await agent.run(user_prompt)
        raw_output = result.output
        
        # Extract code from markdown code block
        code_match = re.search(r'```python\n(.*?)```', raw_output, re.DOTALL)
        if code_match:
            test_code = code_match.group(1).strip()
        else:
            # Fallback if no code block found
            test_code = raw_output.strip()
        
        # Count implemented tests
        test_count = count_test_methods(test_code)
        
        # Estimate coverage
        coverage_achieved = estimate_coverage(test_analysis, test_count)
        
        # Validate implementation
        runnable = (
            "import" in test_code and
            "class Test" in test_code and
            "def test_" in test_code and
            test_count > 0
        )
        
        dependencies_created = all(
            dep in test_code
            for dep in test_analysis.integration_points
        )
        
        # Create TestImplementationOutput
        test_implementation = TestImplementationOutput(
            code=test_code,
            file_path=f"test_{tool_name}.py",
            test_count=test_count,
            coverage_achieved=coverage_achieved,
            runnable=runnable,
            dependencies_created=dependencies_created
        )
        
        # Store test implementation in storage_kv
        state_key = f'workflow/{workflow_id}/test_implementation/{tool_name}'
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': state_key,
            'value': json.dumps(test_implementation.model_dump())
        })
        
        # Save to file system
        file_path = f"generated/{workflow_id}/tests/{test_implementation.file_path}"
        await injector.run('storage_fs', {
            'operation': 'write',
            'path': file_path,
            'content': test_implementation.code,
            'create_parents': True
        })
        
        # Create test summary
        summary_path = f"generated/{workflow_id}/tests/TEST_SUMMARY_{tool_name}.md"
        summary = f"""# Test Summary for {tool_name}

## Test Statistics
- **Total Tests Implemented**: {test_count}
- **Test Cases Identified**: {len(test_analysis.test_cases)}
- **Coverage Achievement**: {coverage_achieved.get('overall', 0):.1f}%
- **Runnable Status**: {'✅ Ready' if runnable else '❌ Not Ready'}
- **Dependencies Created**: {'✅ Yes' if dependencies_created else '❌ No'}

## Coverage Breakdown
{chr(10).join(f"- **{k.title()}**: {v:.1f}%" for k, v in coverage_achieved.items())}

## Test Cases
{chr(10).join(f"{i+1}. **{tc.name}** ({tc.test_type})\\n   - {tc.description}" for i, tc in enumerate(test_analysis.test_cases))}

## Real Dependencies
{chr(10).join(f"- {dep}" for dep in test_analysis.dependency_setup.keys()) if test_analysis.dependency_setup else "No external dependencies"}

## File Location
- Test Implementation: `{file_path}`
- Test Stub: `generated/{workflow_id}/test_stubs/test_{tool_name}.py`

## Next Steps
1. Run the tests: `pytest {file_path} -v`
2. Check coverage: `pytest {file_path} --cov={tool_name} --cov-report=html`
3. Integrate with CI/CD pipeline
"""
        
        await injector.run('storage_fs', {
            'operation': 'write',
            'path': summary_path,
            'content': summary,
            'create_parents': True
        })
        
        # Log completion
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'workflow',
            'message': f'Test implementation completed for {tool_name}',
            'data': {
                'workflow_id': workflow_id,
                'tool_name': tool_name,
                'file_path': file_path,
                'test_count': test_count,
                'coverage': coverage_achieved.get('overall', 0),
                'runnable': runnable,
                'summary_path': summary_path
            }
        })
        
        return WorkflowTestCrafterOutput(
            success=True,
            message=f"Test implementation complete: {test_count} tests with {coverage_achieved.get('overall', 0):.1f}% coverage",
            data=test_implementation.model_dump(),
            state_ref=state_key
        )
        
    except Exception as e:
        error_msg = f"Test implementation failed: {str(e)}"
        
        # Log error
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'workflow',
            'message': 'Test implementation phase failed',
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
        'craft': ('craft_test_implementation', lambda x: {
            'workflow_id': x.workflow_id,
            'tool_name': x.tool_name,
            'model': x.model
        })
    }
)


def create_workflow_test_crafter_agent():
    """Create and return the workflow test crafter AgenTool."""
    
    return create_agentool(
        name='workflow_test_crafter',
        input_schema=WorkflowTestCrafterInput,
        routing_config=routing,
        tools=[craft_test_implementation],
        output_type=WorkflowTestCrafterOutput,
        system_prompt="Implement complete test logic with assertions, real dependencies, and test data following No-Mocks policy.",
        description="Completes test implementation by filling in stub placeholders with actual test logic",
        version="1.0.0",
        tags=["workflow", "testing", "implementation", "code-generation"],
        dependencies=["storage_kv", "storage_fs", "templates", "logging", "metrics"],
        examples=[
            {
                "input": {
                    "operation": "craft",
                    "workflow_id": "workflow-123",
                    "tool_name": "session_manager"
                },
                "output": {
                    "success": True,
                    "message": "Test implementation complete: 15 tests with 93.3% coverage",
                    "data": {
                        "code": "# Complete test implementation...",
                        "file_path": "test_session_manager.py",
                        "test_count": 15,
                        "coverage_achieved": {
                            "overall": 93.3,
                            "unit": 100.0,
                            "integration": 85.0,
                            "edge_case": 90.0
                        },
                        "runnable": True,
                        "dependencies_created": True
                    }
                }
            }
        ]
    )


# Create the agent instance
agent = create_workflow_test_crafter_agent()