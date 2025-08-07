"""
Workflow Evaluator AgenTool - Validates and improves generated code.

This AgenTool takes the generated implementation and validates it for
syntax, imports, patterns, and quality, providing the final production-ready code.
"""

import ast
import json
from typing import Dict, Any, List, Literal
from pydantic import BaseModel, Field
from pydantic_ai import RunContext, Agent
from pydantic_ai.exceptions import ModelRetry

from agentool import create_agentool, BaseOperationInput
from agentool.core.registry import RoutingConfig
from agentool.core.injector import get_injector

# Import the data models
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from agents.models import ValidationOutput, CodeOutput, SpecificationOutput


class WorkflowEvaluatorInput(BaseOperationInput):
    """Input schema for workflow evaluator operations."""
    operation: Literal['evaluate', 'validate'] = Field(
        description="Operation to perform"
    )
    workflow_id: str = Field(
        description="Workflow identifier to retrieve code from"
    )
    model: str = Field(
        default="openai:gpt-4o",
        description="LLM model to use for evaluation"
    )
    auto_fix: bool = Field(
        default=True,
        description="Whether to automatically fix issues found"
    )


class WorkflowEvaluatorOutput(BaseModel):
    """Output from workflow evaluation."""
    success: bool = Field(description="Whether evaluation succeeded")
    operation: str = Field(description="Operation that was performed")
    message: str = Field(description="Status message")
    data: Dict[str, Any] = Field(description="Validation results and final code")
    state_ref: str = Field(description="Reference to stored state in storage_kv")


def validate_python_syntax(code: str) -> tuple[bool, List[str]]:
    """
    Validate Python syntax using AST parser.
    
    Args:
        code: Python code to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    try:
        ast.parse(code)
        return True, []
    except SyntaxError as e:
        errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        return False, errors
    except Exception as e:
        errors.append(f"Unexpected error during parsing: {str(e)}")
        return False, errors


def check_imports(code: str) -> List[str]:
    """
    Check for problematic imports in the code.
    
    Args:
        code: Python code to check
        
    Returns:
        List of warnings about imports
    """
    warnings = []
    standard_libs = {
        'json', 'os', 'sys', 'typing', 'datetime', 'asyncio',
        'collections', 're', 'functools', 'itertools'
    }
    allowed_packages = {
        'pydantic', 'pydantic_ai', 'agentool', 'logfire'
    }
    
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split('.')[0]
                    if module not in standard_libs and module not in allowed_packages:
                        warnings.append(f"Non-standard import '{module}' may not be available")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split('.')[0]
                    if module not in standard_libs and module not in allowed_packages:
                        warnings.append(f"Non-standard import from '{module}' may not be available")
    except:
        pass
    
    return warnings


async def evaluate_code(
    ctx: RunContext[Any],
    workflow_id: str,
    model: str,
    auto_fix: bool
) -> WorkflowEvaluatorOutput:
    """
    Evaluate and validate generated AgenTool implementation.
    
    This function:
    1. Loads generated code from storage_kv
    2. Performs syntax validation
    3. Checks imports
    4. Uses LLM to evaluate quality and fix issues
    5. Stores final validated code
    
    Args:
        ctx: Runtime context
        workflow_id: Workflow identifier
        model: LLM model to use
        auto_fix: Whether to fix issues
        
    Returns:
        Validation results and final code
        
    Raises:
        RuntimeError: If code loading or validation fails
    """
    injector = get_injector()
    
    try:
        # Log evaluation phase start
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'workflow',
            'message': 'Code evaluation phase started',
            'data': {
                'workflow_id': workflow_id,
                'operation': 'evaluate',
                'model': model,
                'auto_fix': auto_fix
            }
        })
        # First get specifications to know which tools to evaluate
        specs_key = f'workflow/{workflow_id}/specs'
        specs_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': specs_key
        })
        
        # storage_kv returns typed StorageKvOutput
        assert specs_result.success is True
        if not specs_result.data.get('exists', False):
            raise ValueError(f"No specifications found for workflow {workflow_id}")
        
        spec_output = SpecificationOutput(**json.loads(specs_result.data['value']))
        
        # Log specifications loaded
        await injector.run('logging', {
            'operation': 'log',
            'level': 'DEBUG',
            'logger_name': 'workflow',
            'message': 'Specifications loaded for evaluation',
            'data': {
                'workflow_id': workflow_id,
                'tools_to_evaluate': len(spec_output.specifications),
                'tool_names': [spec.name for spec in spec_output.specifications]
            }
        })
        
        # Evaluate all tools that were generated
        if not spec_output.specifications:
            raise ValueError("No specifications to evaluate")
        
        all_validations = []
        all_summaries = []
        
        # Iterate through all specifications
        for tool_spec in spec_output.specifications:
            try:
                # Load generated code from storage_kv
                code_key = f'workflow/{workflow_id}/implementations/{tool_spec.name}'
                code_result = await injector.run('storage_kv', {
                    'operation': 'get',
                    'key': code_key
                })
                
                # storage_kv returns typed StorageKvOutput
                assert code_result.success is True
                if not code_result.data.get('exists', False):
                    # Log warning but continue with other tools
                    await injector.run('logging', {
                        'operation': 'log',
                        'level': 'WARN',
                        'logger_name': 'workflow',
                        'message': f'No implementation found for {tool_spec.name}, skipping evaluation',
                        'data': {
                            'workflow_id': workflow_id,
                            'tool_name': tool_spec.name
                        }
                    })
                    continue
            
                code_output = CodeOutput(**json.loads(code_result.data['value']))
            
                # Perform syntax validation
                syntax_valid, syntax_errors = validate_python_syntax(code_output.code)
                
                # Check imports
                import_warnings = check_imports(code_output.code)
                
                # Initial validation results
                issues = syntax_errors + import_warnings
                
                # If no issues and syntax is valid, we might be done
                if syntax_valid and not import_warnings and not auto_fix:
                    validation = ValidationOutput(
                        syntax_valid=True,
                        imports_valid=True,
                        tests_passed=False,  # Would need actual test execution
                        issues=[],
                        fixes_applied=[],
                        improvements=[],
                        final_code=code_output.code,
                        ready_for_deployment=True
                    )
                else:
                    # Use LLM to evaluate and potentially fix the code
                    
                    # Load system prompt with schema
                    system_result = await injector.run('templates', {
                        'operation': 'render',
                        'template_name': 'system/evaluator',
                        'variables': {
                            'schema_json': json.dumps(ValidationOutput.model_json_schema(), indent=2)
                        }
                    })
                    
                    # templates returns typed TemplatesOutput
                    assert system_result.success is True
                    system_prompt = system_result.data.get('rendered', 'You are an expert code evaluator.')
                    
                    # Create LLM agent for evaluation
                    agent = Agent(
                        model,
                        output_type=ValidationOutput,
                        system_prompt=system_prompt
                    )
                    
                    # Store implementation code for template reference
                    impl_code_key = f'workflow/{workflow_id}/current_implementation_code'
                    await injector.run('storage_kv', {
                        'operation': 'set',
                        'key': impl_code_key,
                        'value': code_output.code
                    })
                    
                    # Read a reference implementation source code (storage_kv is a good comprehensive example)
                    ref_impl_result = await injector.run('storage_fs', {
                        'operation': 'read',
                        'path': 'src/agentoolkit/storage/kv.py'
                    })
                    
                    # storage_fs returns typed StorageFsOutput
                    assert ref_impl_result.success is True
                    
                    # Store reference implementation source code for template
                    ref_impl_key = f'workflow/{workflow_id}/reference_implementation'
                    ref_impl_content = ref_impl_result.data.get('content', '')
                    await injector.run('storage_kv', {
                        'operation': 'set',
                        'key': ref_impl_key,
                        'value': ref_impl_content
                    })
                    
                    # Prepare evaluation prompt with references
                    prompt_result = await injector.run('templates', {
                        'operation': 'render',
                        'template_name': 'prompts/evaluate_code',
                        'variables': {
                            'implementation_code': f'!ref:storage_kv:{impl_code_key}',
                            'spec_output': f'!ref:storage_kv:workflow/{workflow_id}/specifications/{tool_spec.name}',
                            'analysis_output': f'!ref:storage_kv:workflow/{workflow_id}/analysis',
                            'reference_implementation': f'!ref:storage_kv:{ref_impl_key}'
                        }
                    })
                    
                    # templates returns typed TemplatesOutput
                    assert prompt_result.success is True
                    user_prompt = prompt_result.data.get('rendered', 'Evaluate this code')
                    
                    # Get validation from LLM
                    result = await agent.run(user_prompt)
                    validation = result.output
                    
                    # Capture and record token usage
                    usage = result.usage()
                    
                    # Store token metrics with tool label
                    await injector.run('metrics', {
                        'operation': 'increment',
                        'name': 'agentool.workflow.tokens.request',
                        'value': usage.request_tokens,
                        'labels': {
                            'workflow_id': workflow_id,
                            'agent': 'workflow_evaluator',
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
                            'agent': 'workflow_evaluator',
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
                            'agent': 'workflow_evaluator',
                            'tool': tool_name,
                            'model': model
                        }
                    })
                    
                    # Override with our syntax check results
                    validation.syntax_valid = syntax_valid
                    validation.imports_valid = len(import_warnings) == 0
            
                # Store validation results in storage_kv
                state_key = f'workflow/{workflow_id}/validations/{tool_spec.name}'
                await injector.run('storage_kv', {
                    'operation': 'set',
                    'key': state_key,
                    'value': json.dumps(validation.model_dump())
                })
                
                # Collect validation for summary
                all_validations.append({
                    'tool_name': tool_spec.name,
                    'validation': validation,
                    'code_output': code_output
                })
                
                # Save final code to file system
                if validation.final_code != code_output.code:
                    # Code was improved, save new version
                    final_path = f"generated/{workflow_id}/final/{os.path.basename(code_output.file_path)}"
                    await injector.run('storage_fs', {
                        'operation': 'write',
                        'path': final_path,
                        'content': validation.final_code,
                        'create_parents': True
                    })
                    
                    # Log that we saved an improved version
                    await injector.run('logging', {
                        'operation': 'log',
                        'level': 'INFO',
                        'logger_name': 'workflow',
                        'message': f'Saved improved code version for {tool_spec.name}',
                        'data': {
                            'workflow_id': workflow_id,
                            'tool_name': tool_spec.name,
                            'original_path': code_output.file_path,
                            'final_path': final_path,
                            'improvements': len(validation.improvements),
                            'fixes': len(validation.fixes_applied)
                        }
                    })
            
            except Exception as e:
                # Log the error but continue processing other tools
                await injector.run('logging', {
                    'operation': 'log',
                    'level': 'ERROR',
                    'logger_name': 'workflow',
                    'message': f'Failed to evaluate {tool_spec.name}, continuing with other tools',
                    'data': {
                        'workflow_id': workflow_id,
                        'tool_name': tool_spec.name,
                        'error': str(e)
                    }
                })
                
                # Track this as a failed validation
                validation = ValidationOutput(
                    syntax_valid=False,
                    imports_valid=False,
                    tests_passed=False,
                    issues=[f"Evaluation failed: {str(e)}"],
                    fixes_applied=[],
                    improvements=[],
                    final_code="",
                    ready_for_deployment=False
                )
                
                all_validations.append({
                    'tool_name': tool_spec.name,
                    'validation': validation,
                    'code_output': CodeOutput(code="", file_path=f"{tool_spec.name}.py")
                })
                
                raise RuntimeError(f"Failed to evaluate tool {tool_spec.name}: {str(e)}") from e
        
        # Load analysis and specification for the summary
        analysis_key = f'workflow/{workflow_id}/analysis'
        analysis_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': analysis_key
        })
        
        # storage_kv returns typed StorageKvOutput
        assert analysis_result.success is True
        analysis = None
        if analysis_result.data.get('exists', False):
            analysis = json.loads(analysis_result.data['value'])
        
        # Create comprehensive summary for all tools
        summary_path = f"generated/{workflow_id}/SUMMARY.md"
        summary = f"""# AgenTool Generation Summary

## Workflow ID: {workflow_id}

## Generated Tools: {len(all_validations)} tools

## Analysis Phase Output
{f'''### Solution Name
{analysis.get('name', 'N/A')}

### Description
{analysis.get('description', 'N/A')}

### System Design
{analysis.get('system_design', 'N/A')}

### Guidelines
{chr(10).join(f"- {guideline}" for guideline in analysis.get('guidelines', [])) if analysis.get('guidelines') else "No guidelines"}

### Existing Tools Used
{chr(10).join(f"- {tool}" for tool in analysis.get('existing_tools', [])) if analysis.get('existing_tools') else "No existing tools"}

### Missing Tools Identified
{chr(10).join(f"- **{tool['name']}**: {tool['description']}" for tool in analysis.get('missing_tools', [])) if analysis.get('missing_tools') else "No missing tools"}
''' if analysis else "Analysis data not available"}

## Generated Tools Summary

{chr(10).join(f'''### Tool {i+1}: {val['tool_name']}

#### Specification
- **Name**: {val['tool_name']}
- **Description**: {next((spec.description for spec in spec_output.specifications if spec.name == val['tool_name']), 'N/A')}
- **Syntax Valid**: {val['validation'].syntax_valid}
- **Ready for Deployment**: {val['validation'].ready_for_deployment}

#### Issues Found
{chr(10).join(f"- {issue}" for issue in val['validation'].issues) if val['validation'].issues else "No issues found"}

#### Fixes Applied
{chr(10).join(f"- {fix}" for fix in val['validation'].fixes_applied) if val['validation'].fixes_applied else "No fixes needed"}

#### File Locations
- Original: {val['code_output'].file_path}
- Final: generated/{workflow_id}/final/{os.path.basename(val['code_output'].file_path)}
''' for i, val in enumerate(all_validations))}

## Overall Results
- **Total Tools Generated**: {len(all_validations)}
- **Tools Ready for Deployment**: {sum(1 for val in all_validations if val['validation'].ready_for_deployment)}
- **Tools with Issues**: {sum(1 for val in all_validations if val['validation'].issues)}
- **Total Fixes Applied**: {sum(len(val['validation'].fixes_applied) for val in all_validations)}

## Artifact References
- Analysis: `storage_kv:workflow/{workflow_id}/analysis`
- Specifications: `storage_kv:workflow/{workflow_id}/specs`
- Implementations Summary: `storage_kv:workflow/{workflow_id}/implementations_summary`
{chr(10).join(f"- {val['tool_name']} Implementation: `storage_kv:workflow/{workflow_id}/implementations/{val['tool_name']}`" for val in all_validations)}
{chr(10).join(f"- {val['tool_name']} Validation: `storage_kv:workflow/{workflow_id}/validations/{val['tool_name']}`" for val in all_validations)}
"""
        
        await injector.run('storage_fs', {
            'operation': 'write',
            'path': summary_path,
            'content': summary,
            'create_parents': True
        })
        
        # Prepare summary data
        summary_data = {
            'total_tools': len(all_validations),
            'tools_ready': sum(1 for val in all_validations if val['validation'].ready_for_deployment),
            'total_issues': sum(len(val['validation'].issues) for val in all_validations),
            'total_fixes': sum(len(val['validation'].fixes_applied) for val in all_validations),
            'validations': [{
                'tool_name': val['tool_name'],
                'ready': val['validation'].ready_for_deployment,
                'syntax_valid': val['validation'].syntax_valid,
                'issues': val['validation'].issues
            } for val in all_validations]
        }
        
        # Store summary in storage_kv
        summary_key = f'workflow/{workflow_id}/validations_summary'
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': summary_key,
            'value': json.dumps(summary_data)
        })
        
        # Log evaluation state storage completion
        await injector.run('logging', {
            'operation': 'log',
            'level': 'DEBUG',
            'logger_name': 'workflow',
            'message': 'Code evaluation state stored successfully',
            'data': {
                'workflow_id': workflow_id,
                'summary_key': summary_key,
                'validations_stored': len(all_validations)
            }
        })
        
        # Log completion
        all_ready = all(val['validation'].ready_for_deployment for val in all_validations)
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO' if all_ready else 'WARN',
            'logger_name': 'workflow',
            'message': 'Evaluation phase completed for all tools',
            'data': {
                'workflow_id': workflow_id,
                'total_tools': len(all_validations),
                'tools_ready': summary_data['tools_ready'],
                'total_issues': summary_data['total_issues'],
                'total_fixes': summary_data['total_fixes']
            }
        })
        
        return WorkflowEvaluatorOutput(
            success=True,
            operation="evaluate",
            message=f"Evaluated {len(all_validations)} tools - {summary_data['tools_ready']} ready for deployment",
            data=summary_data,
            state_ref=summary_key
        )
        
    except Exception as e:
        error_msg = f"Evaluation failed: {str(e)}"
        
        # Log error
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'workflow',
            'message': 'Evaluation phase failed',
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
        'evaluate': ('evaluate_code', lambda x: {
            'workflow_id': x.workflow_id,
            'model': x.model,
            'auto_fix': x.auto_fix
        }),
        'validate': ('evaluate_code', lambda x: {
            'workflow_id': x.workflow_id,
            'model': x.model,
            'auto_fix': False  # validate doesn't auto-fix
        })
    }
)


def create_workflow_evaluator_agent():
    """Create and return the workflow evaluator AgenTool."""
    
    return create_agentool(
        name='workflow_evaluator',
        input_schema=WorkflowEvaluatorInput,
        routing_config=routing,
        tools=[evaluate_code],
        output_type=WorkflowEvaluatorOutput,
        use_typed_output=True,  # Enable typed output for workflow_evaluator
        system_prompt="Evaluate and validate AgenTool implementations for quality and correctness.",
        description="Validates generated code for syntax, patterns, and quality, providing production-ready output",
        version="1.0.0",
        tags=["workflow", "validation", "evaluation", "code-quality"],
        dependencies=["storage_kv", "storage_fs", "templates", "logging", "metrics"],
        examples=[
            {
                "input": {
                    "operation": "evaluate",
                    "workflow_id": "workflow-123",
                    "auto_fix": True
                },
                "output": {
                    "success": True,
                    "operation": "evaluate",
                    "message": "Code evaluation complete - ready for deployment",
                    "data": {
                        "syntax_valid": True,
                        "imports_valid": True,
                        "ready_for_deployment": True,
                        "issues": [],
                        "fixes_applied": ["Fixed indentation", "Added missing docstrings"]
                    }
                }
            }
        ]
    )


# Create the agent instance
agent = create_workflow_evaluator_agent()