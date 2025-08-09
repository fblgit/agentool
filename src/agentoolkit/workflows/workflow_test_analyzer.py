"""
Workflow Test Analyzer AgenTool - Analyzes generated code to determine test requirements.

This AgenTool examines the final validated code from the evaluator phase
and creates a comprehensive test analysis including test cases, coverage needs,
and real dependency requirements.
"""

import json
import ast
from typing import Dict, Any, List, Literal, Optional
from pydantic import BaseModel, Field
from pydantic_ai import RunContext, Agent

from agentool import create_agentool, BaseOperationInput
from agentool.core.registry import RoutingConfig
from agentool.core.injector import get_injector

# Import the data models
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from agents.models import TestAnalysisOutput, TestCaseSpec, CodeOutput, SpecificationOutput


class WorkflowTestAnalyzerInput(BaseOperationInput):
    """Input schema for workflow test analyzer operations."""
    operation: Literal['analyze'] = Field(
        description="Operation to perform"
    )
    workflow_id: str = Field(
        description="Workflow identifier to retrieve code from"
    )
    tool_name: str = Field(
        description="Name of the specific tool to analyze for testing"
    )
    model: str = Field(
        default="openai:gpt-4o",
        description="LLM model to use for test analysis"
    )


class WorkflowTestAnalyzerOutput(BaseModel):
    """Output from workflow test analysis."""
    success: bool = Field(description="Whether analysis succeeded")
    operation: str = Field(description="Operation that was performed")
    message: str = Field(description="Status message")
    data: Dict[str, Any] = Field(description="Test analysis results")
    state_ref: str = Field(description="Reference to stored state in storage_kv")


def extract_code_structure(code: str) -> Dict[str, Any]:
    """
    Extract structural information from Python code.
    
    Args:
        code: Python source code
        
    Returns:
        Dictionary with functions, classes, imports, and dependencies
    """
    structure = {
        "functions": [],
        "classes": [],
        "imports": [],
        "async_functions": [],
        "dependencies": []
    }
    
    try:
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                structure["functions"].append({
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "decorators": [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list],
                    "is_tool": any("tool" in str(d) for d in node.decorator_list)
                })
            elif isinstance(node, ast.AsyncFunctionDef):
                structure["async_functions"].append({
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "decorators": [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list],
                    "is_tool": any("tool" in str(d) for d in node.decorator_list)
                })
            elif isinstance(node, ast.ClassDef):
                structure["classes"].append({
                    "name": node.name,
                    "bases": [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases]
                })
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    structure["imports"].append(alias.name)
                    if "injector" in alias.name or "agentool" in alias.name:
                        structure["dependencies"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    structure["imports"].append(node.module)
                    if "injector" in node.module or "agentool" in node.module:
                        structure["dependencies"].append(node.module)
    except:
        pass
    
    return structure


async def analyze_test_requirements(
    ctx: RunContext[Any],
    workflow_id: str,
    tool_name: str,
    model: str
) -> WorkflowTestAnalyzerOutput:
    """
    Analyze generated code to determine comprehensive test requirements.
    
    This function:
    1. Loads the final validated code from storage_kv
    2. Extracts code structure and operations
    3. Uses LLM to analyze test requirements
    4. Stores test analysis for later phases
    
    Args:
        ctx: Runtime context
        workflow_id: Workflow identifier
        tool_name: Specific tool to analyze
        model: LLM model to use
        
    Returns:
        Test analysis with cases, mocks, and coverage needs
        
    Raises:
        RuntimeError: If code loading or analysis fails
    """
    injector = get_injector()
    
    try:
        # Log test analysis phase start
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'workflow',
            'message': 'Test analysis phase started',
            'data': {
                'workflow_id': workflow_id,
                'operation': 'analyze',
                'tool_name': tool_name,
                'model': model
            }
        })
        # Load the final validated code
        validation_key = f'workflow/{workflow_id}/validations/{tool_name}'
        validation_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': validation_key
        })
        
        # storage_kv returns typed StorageKvOutput
        assert validation_result.success is True
        if not validation_result.data.get('exists', False):
            raise ValueError(f"No validation found for tool {tool_name}")
        
        validation = json.loads(validation_result.data['value'])
        final_code = validation.get('final_code', '')
        
        if not final_code:
            raise ValueError(f"No final code found for tool {tool_name}")
        
        # Extract code structure
        code_structure = extract_code_structure(final_code)
        
        # Log code analysis
        await injector.run('logging', {
            'operation': 'log',
            'level': 'DEBUG',
            'logger_name': 'workflow',
            'message': 'Code structure extracted for test analysis',
            'data': {
                'workflow_id': workflow_id,
                'tool_name': tool_name,
                'functions_count': len(code_structure.get('functions', [])),
                'async_functions_count': len(code_structure.get('async_functions', [])),
                'classes_count': len(code_structure.get('classes', [])),
                'imports_count': len(code_structure.get('imports', []))
            }
        })
        
        # Load specifications for this tool
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
        
        # Load existing tools for dependency analysis
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
        
        # Load system prompt template
        template_result = await injector.run('templates', {
            'operation': 'render',
            'template_name': 'system/test_analyzer',
            'variables': {
                'schema_json': json.dumps(TestAnalysisOutput.model_json_schema(), indent=2)
            }
        })
        
        # templates returns typed TemplatesOutput
        assert template_result.success is True
        system_prompt = template_result.data.get('rendered', 'You are an expert test analyzer.')
        
        # Create LLM agent for test analysis
        agent = Agent(
            model,
            output_type=TestAnalysisOutput,
            system_prompt=system_prompt
        )
        
        # Store data for template references
        code_key = f'workflow/{workflow_id}/test_analysis/current_code'
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': code_key,
            'value': final_code
        })
        
        structure_key = f'workflow/{workflow_id}/test_analysis/code_structure'
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': structure_key,
            'value': json.dumps(code_structure)
        })
        
        # Prepare user prompt
        prompt_result = await injector.run('templates', {
            'operation': 'render',
            'template_name': 'prompts/analyze_tests',
            'variables': {
                'tool_name': tool_name,
                'final_code': f'!ref:storage_kv:{code_key}',
                'code_structure': f'!ref:storage_kv:{structure_key}',
                'specification': f'!ref:storage_kv:{spec_key}',
                'all_specifications': f'!ref:storage_kv:{all_specs_key}',
                'existing_tools': f'!ref:storage_kv:{existing_tools_key}'
            }
        })
        
        # templates returns typed TemplatesOutput
        assert prompt_result.success is True
        user_prompt = prompt_result.data.get('rendered', f"Analyze test requirements for {tool_name}")
        
        # Log LLM test analysis start
        await injector.run('logging', {
            'operation': 'log',
            'level': 'DEBUG',
            'logger_name': 'workflow',
            'message': 'Starting LLM test analysis',
            'data': {
                'workflow_id': workflow_id,
                'tool_name': tool_name,
                'model': model,
                'prompt_length': len(user_prompt)
            }
        })
        
        # Generate test analysis using LLM
        result = await agent.run(user_prompt)
        test_analysis = result.output
        
        # Capture and record token usage
        usage = result.usage()
        
        # Store token metrics with tool label
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.workflow.tokens.request',
            'value': usage.request_tokens,
            'labels': {
                'workflow_id': workflow_id,
                'agent': 'workflow_test_analyzer',
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
                'agent': 'workflow_test_analyzer',
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
                'agent': 'workflow_test_analyzer',
                'tool': tool_name,
                'model': model
            }
        })
        
        # Log LLM test analysis completion
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'workflow',
            'message': 'LLM test analysis completed',
            'data': {
                'workflow_id': workflow_id,
                'tool_name': tool_name,
                'test_cases_identified': len(test_analysis.test_cases),
                'fixtures_needed': len(test_analysis.fixtures_needed),
                'dependencies_required': len(test_analysis.dependency_setup)
            }
        })
        
        # Ensure tool name consistency
        test_analysis.tool_name = tool_name
        
        # Store test analysis in storage_kv
        state_key = f'workflow/{workflow_id}/test_analysis/{tool_name}'
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': state_key,
            'value': json.dumps(test_analysis.model_dump())
        })
        
        # Store individual test cases for easy reference
        for i, test_case in enumerate(test_analysis.test_cases):
            test_case_key = f'workflow/{workflow_id}/test_cases/{tool_name}/{i}'
            await injector.run('storage_kv', {
                'operation': 'set',
                'key': test_case_key,
                'value': json.dumps(test_case.model_dump())
            })
        
        # Log state storage completion
        await injector.run('logging', {
            'operation': 'log',
            'level': 'DEBUG',
            'logger_name': 'workflow',
            'message': 'Test analysis state stored successfully',
            'data': {
                'workflow_id': workflow_id,
                'tool_name': tool_name,
                'state_key': state_key,
                'test_cases_stored': len(test_analysis.test_cases)
            }
        })
        
        # Log the analysis
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'workflow',
            'message': 'Test analysis phase completed',
            'data': {
                'workflow_id': workflow_id,
                'tool_name': tool_name,
                'test_cases_count': len(test_analysis.test_cases),
                'dependencies_count': len(test_analysis.dependency_setup),
                'fixtures_count': len(test_analysis.fixtures_needed)
            }
        })
        
        return WorkflowTestAnalyzerOutput(
            success=True,
            operation="analyze",
            message=f"Test analysis complete: {len(test_analysis.test_cases)} test cases identified",
            data=test_analysis.model_dump(),
            state_ref=state_key
        )
        
    except Exception as e:
        error_msg = f"Test analysis failed: {str(e)}"
        
        # Log error
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'workflow',
            'message': 'Test analysis phase failed',
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
        'analyze': ('analyze_test_requirements', lambda x: {
            'workflow_id': x.workflow_id,
            'tool_name': x.tool_name,
            'model': x.model
        })
    }
)


def create_workflow_test_analyzer_agent():
    """Create and return the workflow test analyzer AgenTool."""
    
    return create_agentool(
        name='workflow_test_analyzer',
        input_schema=WorkflowTestAnalyzerInput,
        routing_config=routing,
        tools=[analyze_test_requirements],
        output_type=WorkflowTestAnalyzerOutput,
        system_prompt="Analyze generated AgenTool code to identify comprehensive test requirements.",
        description="Analyzes validated code to determine test cases, coverage needs, and real dependency requirements",
        version="1.0.0",
        tags=["workflow", "testing", "analysis", "code-generation"],
        dependencies=["storage_kv", "templates", "logging", "metrics"],
        examples=[
            {
                "input": {
                    "operation": "analyze",
                    "workflow_id": "workflow-123",
                    "tool_name": "session_manager"
                },
                "output": {
                    "success": True,
                    "operation": "analyze",
                    "message": "Test analysis complete: 15 test cases identified",
                    "data": {
                        "tool_name": "session_manager",
                        "test_cases": [
                            {
                                "name": "test_create_session_success",
                                "description": "Test successful session creation",
                                "test_type": "unit",
                                "operation": "create"
                            }
                        ],
                        "coverage_requirements": {
                            "operations": 100,
                            "error_handling": 80,
                            "edge_cases": 70
                        }
                    }
                }
            }
        ]
    )


# Create the agent instance
agent = create_workflow_test_analyzer_agent()