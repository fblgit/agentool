"""Workflow graph orchestration for AI Code Generation.

This module defines the graph structure and nodes that orchestrate
the multi-phase AgenTool generation workflow using pydantic_graph.
"""

import uuid
import json
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from pydantic_graph import BaseNode, End, Graph, GraphRunContext
from pydantic_ai.messages import ModelMessage
from agentool.core.injector import get_injector
import logfire

# configure logfire
def scrubbing_callback(m: logfire.ScrubMatch):
    return m.value
logfire.configure(token=os.environ.get('LOGFIRE_API_KEY'), scrubbing=logfire.ScrubbingOptions(callback=scrubbing_callback))
logfire.instrument_pydantic_ai()


try:
    from .models import WorkflowMetadata
except ImportError:
    from agents.models import WorkflowMetadata


@dataclass
class WorkflowState:
    """State management for the AgenTool generation workflow.
    
    This state is passed between nodes and accumulates references
    to data stored in storage_kv throughout the workflow execution.
    """
    
    # Input
    task_description: str
    model: str = "openai:gpt-4o"
    generate_tests: bool = True  # Test generation is always enabled
    
    # Workflow metadata
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Optional[WorkflowMetadata] = None
    
    # Phase completion flags
    analyzer_completed: bool = False
    specification_completed: bool = False
    crafter_completed: bool = False
    evaluator_completed: bool = False
    
    # Test phase completion flags
    test_analyzer_completed: bool = False
    test_stubber_completed: bool = False
    test_crafter_completed: bool = False
    
    # Message history for context
    messages: List[ModelMessage] = field(default_factory=list)
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    current_phase: str = "initialization"


@dataclass
class AnalyzerNode(BaseNode[WorkflowState]):
    """Node for the Analyzer phase of the workflow."""
    
    async def run(self, ctx: GraphRunContext[WorkflowState]) -> "SpecificationNode":
        """Run the analyzer phase using workflow_analyzer AgenTool.
        
        Args:
            ctx: Graph execution context with workflow state
            
        Returns:
            Next node (SpecificationNode)
        """
        ctx.state.current_phase = "analyzer"
        start_time = datetime.now()
        
        try:
            injector = get_injector()
            
            # Run analyzer AgenTool
            result = await injector.run('workflow_analyzer', {
                'operation': 'analyze',
                'task_description': ctx.state.task_description,
                'workflow_id': ctx.state.workflow_id,
                'model': ctx.state.model
            })
            
            # AgenTools now return typed outputs
            data = result.data
            
            if not result.success:
                raise RuntimeError(f"Analyzer failed: {result.message}")
            
            ctx.state.analyzer_completed = True
            
            # Update metadata
            if ctx.state.metadata:
                duration = (datetime.now() - start_time).total_seconds()
                ctx.state.metadata.phase_durations["analyzer"] = duration
                ctx.state.metadata.models_used["analyzer"] = ctx.state.model
            
            # Log progress
            analysis = data['data']
            await injector.run('logging', {
                'operation': 'log',
                'level': 'INFO',
                'logger_name': 'workflow',
                'message': 'Analyzer phase completed',
                'data': {
                    'workflow_id': ctx.state.workflow_id,
                    'existing_tools': len(analysis.get('existing_tools', [])),
                    'missing_tools': len(analysis.get('missing_tools', []))
                }
            })
            
            # Log artifacts created
            artifacts = []
            artifacts.append(f"storage_kv:workflow/{ctx.state.workflow_id}/catalog")
            artifacts.append(f"storage_kv:workflow/{ctx.state.workflow_id}/analysis")
            for tool in analysis.get('missing_tools', []):
                if isinstance(tool, dict) and 'name' in tool:
                    artifacts.append(f"storage_kv:workflow/{ctx.state.workflow_id}/missing_tools/{tool['name']}")
            
            await injector.run('logging', {
                'operation': 'log',
                'level': 'DEBUG',
                'logger_name': 'workflow.artifacts',
                'message': 'Analyzer artifacts created',
                'data': {
                    'workflow_id': ctx.state.workflow_id,
                    'phase': 'analyzer',
                    'artifacts': artifacts,
                    'artifact_count': len(artifacts)
                }
            })
            
        except Exception as e:
            ctx.state.errors.append(f"Analyzer error: {str(e)}")
            raise
        
        return SpecificationNode()


@dataclass
class SpecificationNode(BaseNode[WorkflowState]):
    """Node for the Specification phase of the workflow."""
    
    async def run(self, ctx: GraphRunContext[WorkflowState]) -> "CrafterNode":
        """Run the specification phase using workflow_specifier AgenTool.
        
        Args:
            ctx: Graph execution context with workflow state
            
        Returns:
            Next node (CrafterNode)
        """
        ctx.state.current_phase = "specification"
        start_time = datetime.now()
        
        try:
            injector = get_injector()
            
            # Run specifier AgenTool
            result = await injector.run('workflow_specifier', {
                'operation': 'specify',
                'workflow_id': ctx.state.workflow_id,
                'model': ctx.state.model
            })
            
            # AgenTools now return typed outputs
            data = result.data
            
            if not result.success:
                raise RuntimeError(f"Specifier failed: {result.message}")
            
            ctx.state.specification_completed = True
            
            # Update metadata
            if ctx.state.metadata:
                duration = (datetime.now() - start_time).total_seconds()
                ctx.state.metadata.phase_durations["specification"] = duration
                ctx.state.metadata.models_used["specification"] = ctx.state.model
            
            # Log progress
            spec_data = data['data']
            await injector.run('logging', {
                'operation': 'log',
                'level': 'INFO',
                'logger_name': 'workflow',
                'message': 'Specification phase completed',
                'data': {
                    'workflow_id': ctx.state.workflow_id,
                    'specifications_count': len(spec_data.get('specifications', []))
                }
            })
            
            # Log artifacts created
            artifacts = []
            artifacts.append(f"storage_kv:workflow/{ctx.state.workflow_id}/specs")
            for spec in spec_data.get('specifications', []):
                if isinstance(spec, dict) and 'name' in spec:
                    artifacts.append(f"storage_kv:workflow/{ctx.state.workflow_id}/specifications/{spec['name']}")
                    artifacts.append(f"storage_kv:workflow/{ctx.state.workflow_id}/existing_tools/{spec['name']}")
            
            await injector.run('logging', {
                'operation': 'log',
                'level': 'DEBUG',
                'logger_name': 'workflow.artifacts',
                'message': 'Specification artifacts created',
                'data': {
                    'workflow_id': ctx.state.workflow_id,
                    'phase': 'specification',
                    'artifacts': artifacts,
                    'artifact_count': len(artifacts)
                }
            })
            
        except Exception as e:
            ctx.state.errors.append(f"Specification error: {str(e)}")
            raise
        
        return CrafterNode()


@dataclass
class CrafterNode(BaseNode[WorkflowState]):
    """Node for the Crafter phase of the workflow."""
    
    async def run(self, ctx: GraphRunContext[WorkflowState]) -> "EvaluatorNode":
        """Run the crafter phase using workflow_crafter AgenTool.
        
        Args:
            ctx: Graph execution context with workflow state
            
        Returns:
            Next node (EvaluatorNode)
        """
        ctx.state.current_phase = "crafter"
        start_time = datetime.now()
        
        try:
            injector = get_injector()
            
            # Run crafter AgenTool
            result = await injector.run('workflow_crafter', {
                'operation': 'craft',
                'workflow_id': ctx.state.workflow_id,
                'model': ctx.state.model
            })
            
            # AgenTools now return typed outputs
            data = result.data
            
            if not result.success:
                raise RuntimeError(f"Crafter failed: {result.message}")
            
            ctx.state.crafter_completed = True
            
            # Update metadata
            if ctx.state.metadata:
                duration = (datetime.now() - start_time).total_seconds()
                ctx.state.metadata.phase_durations["crafter"] = duration
                ctx.state.metadata.models_used["crafter"] = ctx.state.model
            
            # Log progress
            code_data = data['data']
            await injector.run('logging', {
                'operation': 'log',
                'level': 'INFO',
                'logger_name': 'workflow',
                'message': 'Crafter phase completed',
                'data': {
                    'workflow_id': ctx.state.workflow_id,
                    'lines_of_code': code_data.get('code', '').count('\n'),
                    'file_path': code_data.get('file_path', '')
                }
            })
            
            # Log artifacts created
            artifacts = []
            # Extract tool name from file path or use default
            tool_name = 'unknown'
            if code_data.get('file_path'):
                import os
                tool_name = os.path.basename(code_data['file_path']).replace('.py', '')
            artifacts.append(f"storage_kv:workflow/{ctx.state.workflow_id}/implementations/{tool_name}")
            artifacts.append(f"storage_fs:generated/{ctx.state.workflow_id}/{code_data.get('file_path', '')}")
            
            await injector.run('logging', {
                'operation': 'log',
                'level': 'DEBUG',
                'logger_name': 'workflow.artifacts',
                'message': 'Crafter artifacts created',
                'data': {
                    'workflow_id': ctx.state.workflow_id,
                    'phase': 'crafter',
                    'artifacts': artifacts,
                    'artifact_count': len(artifacts),
                    'code_size_bytes': len(code_data.get('code', ''))
                }
            })
            
        except Exception as e:
            ctx.state.errors.append(f"Crafter error: {str(e)}")
            raise
        
        return EvaluatorNode()


@dataclass
class EvaluatorNode(BaseNode[WorkflowState]):
    """Node for the Evaluator phase of the workflow."""
    
    async def run(self, ctx: GraphRunContext[WorkflowState]) -> Union[End[Dict[str, Any]], "TestAnalyzerNode"]:
        """Run the evaluator phase using workflow_evaluator AgenTool.
        
        Args:
            ctx: Graph execution context with workflow state
            
        Returns:
            End node with final results
        """
        ctx.state.current_phase = "evaluator"
        start_time = datetime.now()
        
        try:
            injector = get_injector()
            
            # Run evaluator AgenTool
            result = await injector.run('workflow_evaluator', {
                'operation': 'evaluate',
                'workflow_id': ctx.state.workflow_id,
                'model': ctx.state.model,
                'auto_fix': True
            })
            
            # AgenTools now return typed outputs
            data = result.data
            
            if not result.success:
                raise RuntimeError(f"Evaluator failed: {result.message}")
            
            ctx.state.evaluator_completed = True
            
            # Update metadata
            validation_data = data['data']
            if ctx.state.metadata:
                duration = (datetime.now() - start_time).total_seconds()
                ctx.state.metadata.phase_durations["evaluator"] = duration
                ctx.state.metadata.models_used["evaluator"] = ctx.state.model
                ctx.state.metadata.completed_at = datetime.now().isoformat()
                ctx.state.metadata.status = "completed" if validation_data.get('ready_for_deployment', False) else "needs_attention"
                
                # Calculate total duration
                if ctx.state.metadata.started_at:
                    start = datetime.fromisoformat(ctx.state.metadata.started_at)
                    ctx.state.metadata.total_duration_seconds = (
                        datetime.now() - start
                    ).total_seconds()
            
            # Log completion
            await injector.run('logging', {
                'operation': 'log',
                'level': 'INFO' if validation_data.get('ready_for_deployment', False) else 'WARN',
                'logger_name': 'workflow',
                'message': 'Workflow completed',
                'data': {
                    'workflow_id': ctx.state.workflow_id,
                    'success': validation_data.get('syntax_valid', False),
                    'ready_for_deployment': validation_data.get('ready_for_deployment', False),
                    'issues_count': len(validation_data.get('issues', []))
                }
            })
            
            # Log artifacts created
            artifacts = []
            # Get tool name from state ref or validation data
            state_ref = data.get('state_ref', '')
            if '/' in state_ref:
                tool_name = state_ref.split('/')[-1]
            else:
                tool_name = 'unknown'
            
            artifacts.append(f"storage_kv:workflow/{ctx.state.workflow_id}/validations/{tool_name}")
            artifacts.append(f"storage_fs:generated/{ctx.state.workflow_id}/final/")
            artifacts.append(f"storage_fs:generated/{ctx.state.workflow_id}/SUMMARY.md")
            
            await injector.run('logging', {
                'operation': 'log',
                'level': 'DEBUG',
                'logger_name': 'workflow.artifacts',
                'message': 'Evaluator artifacts created',
                'data': {
                    'workflow_id': ctx.state.workflow_id,
                    'phase': 'evaluator',
                    'artifacts': artifacts,
                    'artifact_count': len(artifacts),
                    'validation_passed': validation_data.get('ready_for_deployment', False)
                }
            })
            
        except Exception as e:
            ctx.state.errors.append(f"Evaluator error: {str(e)}")
            if ctx.state.metadata:
                ctx.state.metadata.status = "error"
            raise
        
        # Check if we should generate tests
        if ctx.state.generate_tests:
            # Continue to test generation phase
            return TestAnalyzerNode()
        
        # Return final results
        return End({
            'workflow_id': ctx.state.workflow_id,
            'success': validation_data.get('ready_for_deployment', False) if 'validation_data' in locals() else False,
            'final_code': validation_data.get('final_code', '') if 'validation_data' in locals() else None,
            'syntax_valid': validation_data.get('syntax_valid', False) if 'validation_data' in locals() else False,
            'ready_for_deployment': validation_data.get('ready_for_deployment', False) if 'validation_data' in locals() else False,
            'issues': validation_data.get('issues', []) if 'validation_data' in locals() else [],
            'fixes_applied': validation_data.get('fixes_applied', []) if 'validation_data' in locals() else [],
            'improvements': validation_data.get('improvements', []) if 'validation_data' in locals() else [],
            'errors': ctx.state.errors,
            'metadata': ctx.state.metadata.model_dump() if ctx.state.metadata else None,
            'graph_path': f"generated/{ctx.state.workflow_id}/graph.mmd"
        })


@dataclass
class TestAnalyzerNode(BaseNode[WorkflowState]):
    """Node for the Test Analyzer phase of the workflow."""
    
    async def run(self, ctx: GraphRunContext[WorkflowState]) -> "TestStubberNode":
        """Run the test analyzer phase for all generated tools.
        
        Args:
            ctx: Graph execution context with workflow state
            
        Returns:
            Next node (TestStubberNode)
        """
        ctx.state.current_phase = "test_analyzer"
        start_time = datetime.now()
        
        try:
            injector = get_injector()
            
            # Load specifications to get list of tools
            specs_key = f'workflow/{ctx.state.workflow_id}/specs'
            specs_result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': specs_key
            })
            
            # AgenTools now return typed outputs
            specs_data = specs_result.data
            
            if not specs_result.success or not specs_data.get('exists', False):
                raise RuntimeError("No specifications found for test analysis")
            
            spec_output = json.loads(specs_data['value'])
            tools_analyzed = 0
            
            # Analyze tests for each tool
            for spec in spec_output.get('specifications', []):
                tool_name = spec.get('name')
                if not tool_name:
                    continue
                
                # Run test analyzer for this tool
                result = await injector.run('workflow_test_analyzer', {
                    'operation': 'analyze',
                    'workflow_id': ctx.state.workflow_id,
                    'tool_name': tool_name,
                    'model': ctx.state.model
                })
                
                # AgenTools now return typed outputs
                data = result.data
                
                if result.success:
                    tools_analyzed += 1
            
            ctx.state.test_analyzer_completed = True
            
            # Update metadata
            if ctx.state.metadata:
                duration = (datetime.now() - start_time).total_seconds()
                ctx.state.metadata.phase_durations["test_analyzer"] = duration
                ctx.state.metadata.models_used["test_analyzer"] = ctx.state.model
            
            # Log progress
            await injector.run('logging', {
                'operation': 'log',
                'level': 'INFO',
                'logger_name': 'workflow',
                'message': 'Test analyzer phase completed',
                'data': {
                    'workflow_id': ctx.state.workflow_id,
                    'tools_analyzed': tools_analyzed
                }
            })
            
            # Log artifacts created
            artifacts = []
            for spec in spec_output.get('specifications', []):
                tool_name = spec.get('name')
                if tool_name:
                    artifacts.append(f"storage_kv:workflow/{ctx.state.workflow_id}/test_analysis/{tool_name}")
            
            await injector.run('logging', {
                'operation': 'log',
                'level': 'DEBUG',
                'logger_name': 'workflow.artifacts',
                'message': 'Test analyzer artifacts created',
                'data': {
                    'workflow_id': ctx.state.workflow_id,
                    'phase': 'test_analyzer',
                    'artifacts': artifacts,
                    'artifact_count': len(artifacts)
                }
            })
            
        except Exception as e:
            ctx.state.errors.append(f"Test analyzer error: {str(e)}")
            raise
        
        return TestStubberNode()


@dataclass
class TestStubberNode(BaseNode[WorkflowState]):
    """Node for the Test Stubber phase of the workflow."""
    
    async def run(self, ctx: GraphRunContext[WorkflowState]) -> "TestCrafterNode":
        """Run the test stubber phase for all analyzed tools.
        
        Args:
            ctx: Graph execution context with workflow state
            
        Returns:
            Next node (TestCrafterNode)
        """
        ctx.state.current_phase = "test_stubber"
        start_time = datetime.now()
        
        try:
            injector = get_injector()
            
            # Load specifications to get list of tools
            specs_key = f'workflow/{ctx.state.workflow_id}/specs'
            specs_result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': specs_key
            })
            
            # AgenTools now return typed outputs
            specs_data = specs_result.data
            
            spec_output = json.loads(specs_data['value'])
            tools_stubbed = 0
            
            # Create stubs for each tool
            for spec in spec_output.get('specifications', []):
                tool_name = spec.get('name')
                if not tool_name:
                    continue
                
                # Check if test analysis exists for this tool
                analysis_key = f'workflow/{ctx.state.workflow_id}/test_analysis/{tool_name}'
                check_result = await injector.run('storage_kv', {
                    'operation': 'exists',
                    'key': analysis_key
                })
                
                # AgenTools now return typed outputs
                check_data = check_result.data
                
                if not check_result.success or not check_data.get('exists', False):
                    continue
                
                # Run test stubber for this tool
                result = await injector.run('workflow_test_stubber', {
                    'operation': 'stub',
                    'workflow_id': ctx.state.workflow_id,
                    'tool_name': tool_name,
                    'model': ctx.state.model
                })
                
                # AgenTools now return typed outputs
                data = result.data
                
                if result.success:
                    tools_stubbed += 1
            
            ctx.state.test_stubber_completed = True
            
            # Update metadata
            if ctx.state.metadata:
                duration = (datetime.now() - start_time).total_seconds()
                ctx.state.metadata.phase_durations["test_stubber"] = duration
                ctx.state.metadata.models_used["test_stubber"] = ctx.state.model
            
            # Log progress
            await injector.run('logging', {
                'operation': 'log',
                'level': 'INFO',
                'logger_name': 'workflow',
                'message': 'Test stubber phase completed',
                'data': {
                    'workflow_id': ctx.state.workflow_id,
                    'tools_stubbed': tools_stubbed
                }
            })
            
            # Log artifacts created
            artifacts = []
            for spec in spec_output.get('specifications', []):
                tool_name = spec.get('name')
                if tool_name:
                    artifacts.append(f"storage_kv:workflow/{ctx.state.workflow_id}/test_stub/{tool_name}")
                    artifacts.append(f"storage_fs:generated/{ctx.state.workflow_id}/test_stubs/test_{tool_name}.py")
            
            await injector.run('logging', {
                'operation': 'log',
                'level': 'DEBUG',
                'logger_name': 'workflow.artifacts',
                'message': 'Test stubber artifacts created',
                'data': {
                    'workflow_id': ctx.state.workflow_id,
                    'phase': 'test_stubber',
                    'artifacts': artifacts,
                    'artifact_count': len(artifacts)
                }
            })
            
        except Exception as e:
            ctx.state.errors.append(f"Test stubber error: {str(e)}")
            raise
        
        return TestCrafterNode()


@dataclass
class TestCrafterNode(BaseNode[WorkflowState, None, Dict[str, Any]]):
    """Node for the Test Crafter phase of the workflow."""
    
    async def run(self, ctx: GraphRunContext[WorkflowState]) -> End[Dict[str, Any]]:
        """Run the test crafter phase for all stubbed tools.
        
        Args:
            ctx: Graph execution context with workflow state
            
        Returns:
            End node with final results including test information
        """
        ctx.state.current_phase = "test_crafter"
        start_time = datetime.now()
        
        try:
            injector = get_injector()
            
            # Load specifications to get list of tools
            specs_key = f'workflow/{ctx.state.workflow_id}/specs'
            specs_result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': specs_key
            })
            
            # AgenTools now return typed outputs
            specs_data = specs_result.data
            
            spec_output = json.loads(specs_data['value'])
            tools_tested = 0
            test_files = []
            test_coverage = {}
            
            # Implement tests for each tool
            for spec in spec_output.get('specifications', []):
                tool_name = spec.get('name')
                if not tool_name:
                    continue
                
                # Check if test stub exists for this tool
                stub_key = f'workflow/{ctx.state.workflow_id}/test_stub/{tool_name}'
                check_result = await injector.run('storage_kv', {
                    'operation': 'exists',
                    'key': stub_key
                })
                
                # AgenTools now return typed outputs
                check_data = check_result.data
                
                if not check_result.success or not check_data.get('exists', False):
                    continue
                
                # Run test crafter for this tool
                result = await injector.run('workflow_test_crafter', {
                    'operation': 'craft',
                    'workflow_id': ctx.state.workflow_id,
                    'tool_name': tool_name,
                    'model': ctx.state.model
                })
                
                # AgenTools now return typed outputs
                data = result.data
                
                if result.success:
                    tools_tested += 1
                    impl_data = data.get('data', {})
                    test_files.append(f"generated/{ctx.state.workflow_id}/tests/test_{tool_name}.py")
                    test_coverage[tool_name] = impl_data.get('coverage_achieved', {})
            
            ctx.state.test_crafter_completed = True
            
            # Update metadata
            if ctx.state.metadata:
                duration = (datetime.now() - start_time).total_seconds()
                ctx.state.metadata.phase_durations["test_crafter"] = duration
                ctx.state.metadata.models_used["test_crafter"] = ctx.state.model
                ctx.state.metadata.completed_at = datetime.now().isoformat()
                
                # Update total duration
                if ctx.state.metadata.started_at:
                    start = datetime.fromisoformat(ctx.state.metadata.started_at)
                    ctx.state.metadata.total_duration_seconds = (
                        datetime.now() - start
                    ).total_seconds()
            
            # Log completion
            await injector.run('logging', {
                'operation': 'log',
                'level': 'INFO',
                'logger_name': 'workflow',
                'message': 'Test crafter phase completed',
                'data': {
                    'workflow_id': ctx.state.workflow_id,
                    'tools_tested': tools_tested,
                    'test_files_count': len(test_files)
                }
            })
            
            # Log artifacts created
            artifacts = []
            for spec in spec_output.get('specifications', []):
                tool_name = spec.get('name')
                if tool_name:
                    artifacts.append(f"storage_kv:workflow/{ctx.state.workflow_id}/test_implementation/{tool_name}")
                    artifacts.append(f"storage_fs:generated/{ctx.state.workflow_id}/tests/test_{tool_name}.py")
                    artifacts.append(f"storage_fs:generated/{ctx.state.workflow_id}/tests/TEST_SUMMARY_{tool_name}.md")
            
            await injector.run('logging', {
                'operation': 'log',
                'level': 'DEBUG',
                'logger_name': 'workflow.artifacts',
                'message': 'Test crafter artifacts created',
                'data': {
                    'workflow_id': ctx.state.workflow_id,
                    'phase': 'test_crafter',
                    'artifacts': artifacts,
                    'artifact_count': len(artifacts)
                }
            })
            
            # Load validation data from evaluator phase
            validation_summary_key = f'workflow/{ctx.state.workflow_id}/validations_summary'
            val_result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': validation_summary_key
            })
            
            validation_data = {}
            if hasattr(val_result, 'output'):
                val_data = json.loads(val_result.output)
                if val_data.get('data', {}).get('exists', False):
                    validation_data = json.loads(val_data['data']['value'])
            
        except Exception as e:
            ctx.state.errors.append(f"Test crafter error: {str(e)}")
            if ctx.state.metadata:
                ctx.state.metadata.status = "error"
            raise
        
        # Return enhanced final results with test information
        return End({
            'workflow_id': ctx.state.workflow_id,
            'success': validation_data.get('tools_ready', 0) > 0 if validation_data else False,
            'ready_for_deployment': validation_data.get('tools_ready', 0) == validation_data.get('total_tools', 0) if validation_data else False,
            'tools_generated': validation_data.get('total_tools', 0) if validation_data else 0,
            'tools_ready': validation_data.get('tools_ready', 0) if validation_data else 0,
            'issues_count': validation_data.get('total_issues', 0) if validation_data else 0,
            'errors': ctx.state.errors,
            'metadata': ctx.state.metadata.model_dump() if ctx.state.metadata else None,
            # Test-specific results
            'tests_generated': True,
            'test_files': test_files,
            'test_coverage': test_coverage,
            'test_summary_path': f"generated/{ctx.state.workflow_id}/tests/",
            'graph_path': f"generated/{ctx.state.workflow_id}/graph.mmd"
        })


# Create the workflow graph
agentool_generation_graph = Graph(
    nodes=[AnalyzerNode, SpecificationNode, CrafterNode, EvaluatorNode, TestAnalyzerNode, TestStubberNode, TestCrafterNode],
    state_type=WorkflowState
)


async def run_agentool_generation_workflow(
    task_description: str,
    model: str = "openai:gpt-4o",
    generate_tests: bool = True
) -> Dict[str, Any]:
    """Run the complete AgenTool generation workflow.
    
    This orchestrates the workflow AgenTools through pydantic_graph,
    managing state transitions and error handling.
    
    Args:
        task_description: Description of the AgenTool to generate
        model: LLM model to use for all phases
        generate_tests: Whether to generate test files (default: True)
        
    Returns:
        Dictionary with workflow results including generated code
    """
    # Initialize workflow state
    state = WorkflowState(
        task_description=task_description,
        model=model,
        generate_tests=generate_tests,
        metadata=WorkflowMetadata(
            workflow_id=str(uuid.uuid4()),
            started_at=datetime.now().isoformat(),
            current_phase="initialization",
            status="running"
        )
    )
    
    # Log workflow start
    injector = get_injector()
    await injector.run('logging', {
        'operation': 'log',
        'level': 'INFO',
        'logger_name': 'workflow',
        'message': 'Starting AgenTool generation workflow',
        'data': {
            'workflow_id': state.workflow_id,
            'task': task_description,
            'model': model
        }
    })
    
    # Ensure workflow AgenTools are registered
    try:
        from agentoolkit.workflows import initialize_workflow_agents
        initialize_workflow_agents()
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'WARN',
            'logger_name': 'workflow',
            'message': 'Could not initialize workflow agents',
            'data': {'error': str(e)}
        })
    
    # Run the workflow graph
    result = await agentool_generation_graph.run(
        AnalyzerNode(),
        state=state
    )
    
    # Dump the graph visualization after completion
    try:
        # Generate Mermaid diagram code
        mermaid_code = agentool_generation_graph.mermaid_code(start_node=AnalyzerNode)
        
        # Save as .mmd file (Mermaid format)
        graph_path = f"generated/{state.workflow_id}/graph.mmd"
        await injector.run('storage_fs', {
            'operation': 'write',
            'path': graph_path,
            'content': mermaid_code,
            'create_parents': True
        })
        
        # Also try to generate the image if possible
        try:
            mermaid_image = agentool_generation_graph.mermaid_image(start_node=AnalyzerNode)
            image_path = f"generated/{state.workflow_id}/graph.png"
            
            # Use regular Python file operations for binary data
            import os
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            with open(image_path, 'wb') as f:
                f.write(mermaid_image)
            
            await injector.run('logging', {
                'operation': 'log',
                'level': 'INFO',
                'logger_name': 'workflow',
                'message': 'Graph visualization saved',
                'data': {
                    'workflow_id': state.workflow_id,
                    'mermaid_path': graph_path,
                    'image_path': image_path
                }
            })
        except Exception as img_error:
            # Image generation might fail if dependencies aren't installed
            await injector.run('logging', {
                'operation': 'log',
                'level': 'WARN',
                'logger_name': 'workflow',
                'message': 'Could not generate graph image',
                'data': {
                    'workflow_id': state.workflow_id,
                    'error': str(img_error),
                    'mermaid_path': graph_path
                }
            })
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'workflow',
            'message': 'Failed to save graph visualization',
            'data': {
                'workflow_id': state.workflow_id,
                'error': str(e)
            }
        })
    
    # Calculate total phases based on whether tests are generated
    total_phases = 7 if generate_tests else 4
    phases_completed = sum([
        state.analyzer_completed,
        state.specification_completed,
        state.crafter_completed,
        state.evaluator_completed,
        state.test_analyzer_completed if generate_tests else False,
        state.test_stubber_completed if generate_tests else False,
        state.test_crafter_completed if generate_tests else False
    ])
    
    # Build artifact categories
    storage_kv_artifacts = [
        'catalog', 'analysis', 'missing_tools',
        'specifications', 'existing_tools',
        'implementations', 'validations'
    ]
    storage_fs_artifacts = [
        'generated_code', 'final_code', 'summary', 'graph'
    ]
    
    if generate_tests:
        storage_kv_artifacts.extend(['test_analysis', 'test_stub', 'test_implementation', 'test_cases'])
        storage_fs_artifacts.extend(['test_stubs', 'tests', 'test_summaries'])
    
    # Log workflow artifacts summary
    await injector.run('logging', {
        'operation': 'log',
        'level': 'INFO',
        'logger_name': 'workflow.artifacts',
        'message': 'Workflow artifacts summary',
        'data': {
            'workflow_id': state.workflow_id,
            'total_phases': total_phases,
            'phases_completed': phases_completed,
            'tests_enabled': generate_tests,
            'artifact_categories': {
                'storage_kv': storage_kv_artifacts,
                'storage_fs': storage_fs_artifacts
            },
            'success': result.output.get('success', False) if result.output else False
        }
    })
    
    return result.output


if __name__ == "__main__":
    import asyncio
    import sys
    
    async def main():
        """Test the workflow with a sample task."""
        
        # Initialize all AgenTools from agentoolkit
        print("Initializing AgenToolkits...")
        injector = get_injector()
        
        # Import all create functions and agent instances
        import agentoolkit
        
        # Call all create_* functions to register the agents
        for name in dir(agentoolkit):
            if name.startswith('create_'):
                print(f"  - Initializing {name}...")
                create_func = getattr(agentoolkit, name)
                try:
                    # Special case for templates agent - provide absolute path
                    if name == 'create_templates_agent':
                        import os
                        templates_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../templates'))
                        print(f"    Using templates path: {templates_path}")
                        agent = create_func(templates_dir=templates_path)
                    else:
                        agent = create_func()
                    # The create_agentool automatically registers with injector
                except Exception as e:
                    print(f"    Warning: Could not initialize {name}: {e}")
            elif name.endswith('_agent') and not name.startswith('create_'):
                # These are direct agent instances (like embedder_agent)
                print(f"  - Registering {name}...")
                agent_instance = getattr(agentoolkit, name)
                # These should auto-register when imported
        
        # Also initialize workflow agents
        try:
            from agentoolkit.workflows import (
                create_workflow_analyzer_agent,
                create_workflow_specifier_agent,
                create_workflow_crafter_agent,
                create_workflow_evaluator_agent,
                create_workflow_test_analyzer_agent,
                create_workflow_test_stubber_agent,
                create_workflow_test_crafter_agent
            )
            
            print("  - Initializing workflow agents...")
            create_workflow_analyzer_agent()
            create_workflow_specifier_agent()
            create_workflow_crafter_agent()
            create_workflow_evaluator_agent()
            
            print("  - Initializing test workflow agents...")
            create_workflow_test_analyzer_agent()
            create_workflow_test_stubber_agent()
            create_workflow_test_crafter_agent()
        except ImportError as e:
            print(f"    Warning: Could not import workflow agents: {e}")
        
        # List registered agents
        registry = injector._instances  # Access private instances dict
        print(f"\nRegistered AgenTools ({len(registry)}):")
        for agent_name in sorted(registry.keys()):
            print(f"  - {agent_name}")
        
        # Sample task description
        task = """
        Create a session management AgenTool that handles user sessions with TTL support.
        It should allow creating sessions, retrieving session data, updating sessions,
        and removing expired sessions. Use storage_kv for persistence.
        """
        #task = "Create a first class TODO agentoolkit with all required feaatures to enable a new Tracking toolkit with all the features for creaang, starting, pausing, completing, blocking, deleting a task powered by the TODO aand Trcking toolkits in synergy. The Tracking should allow to reference a TODO for a task, so checkbox based progress can be supported."
        task = 'Create a TODO agentoolkit for create, update, status change, delete, list capabilities integrated with the ecosystem.'
        
        # Use command line argument if provided
        if len(sys.argv) > 1:
            task = sys.argv[1]
        
        print(f"\nStarting workflow with task: {task[:100]}...")
        
        try:
            result = await run_agentool_generation_workflow(
                task_description=task,
                model="openai:gpt-4o"
            )
            
            print("\n=== Workflow Results ===")
            print(f"Success: {result.get('success', False)}")
            print(f"Workflow ID: {result.get('workflow_id', 'N/A')}")
            print(f"Syntax Valid: {result.get('syntax_valid', False)}")
            print(f"Ready for Deployment: {result.get('ready_for_deployment', False)}")
            
            # Handle both formats - list from EvaluatorNode, count from TestCrafterNode
            issues = result.get('issues', [])
            issues_count = result.get('issues_count', 0)
            
            if issues and isinstance(issues, list):
                print(f"\nIssues Found ({len(issues)}):")
                for issue in issues[:5]:  # Show first 5 issues
                    print(f"  - {issue}")
            elif issues_count:
                print(f"\nIssues Found: {issues_count}")
            
            if result.get('improvements'):
                print(f"\nImprovements Made ({len(result['improvements'])}):")
                for imp in result['improvements'][:5]:  # Show first 5 improvements
                    print(f"  - {imp}")
            
            if result.get('errors'):
                print(f"\nErrors ({len(result['errors'])}):")
                for error in result['errors']:
                    print(f"  - {error}")
            
            if result.get('metadata'):
                meta = result['metadata']
                print(f"\nMetadata:")
                print(f"  Status: {meta.get('status', 'N/A')}")
                print(f"  Total Duration: {meta.get('total_duration_seconds', 0):.2f}s")
                print(f"  Phases: {meta.get('phase_durations', {})}")
            
            if result.get('graph_path'):
                print(f"\nWorkflow Graph: {result['graph_path']}")
            
            if result.get('tests_generated'):
                print(f"\n=== Test Generation Results ===")
                test_files = result.get('test_files', [])
                if isinstance(test_files, list):
                    print(f"Test Files Generated: {len(test_files)}")
                    if test_files:
                        print("Test Files:")
                        for test_file in test_files:
                            print(f"  - {test_file}")
                if result.get('test_coverage'):
                    print("\nTest Coverage by Tool:")
                    for tool, coverage in result['test_coverage'].items():
                        overall = coverage.get('overall', 0)
                        print(f"  - {tool}: {overall:.1f}% overall coverage")
                print(f"\nTest Summary Path: {result.get('test_summary_path', 'N/A')}")
            
        except Exception as e:
            print(f"Workflow failed with error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Run the async main function
    asyncio.run(main())
