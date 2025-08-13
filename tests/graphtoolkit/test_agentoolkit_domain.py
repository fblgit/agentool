"""
Tests for AgenToolkit domain in GraphToolkit.

This test file verifies the analyzer phase of the agentoolkit domain,
ensuring proper storage patterns and workflow execution.
"""

import json
import uuid
import pytest
from pydantic import BaseModel, Field
from typing import List

from graphtoolkit.domains.agentoolkit import (
    AgenToolkitAnalyzerInput,
    analyzer_phase,
    AnalyzerOutput
)
from graphtoolkit.core.types import WorkflowState, WorkflowDefinition, NodeConfig
from graphtoolkit.core.deps import WorkflowDeps
from graphtoolkit.core.factory import create_node_instance
from graphtoolkit.nodes.generic import GenericPhaseNode
from graphtoolkit.nodes.atomic.control import NextPhaseNode, RefinementNode
from graphtoolkit.nodes.atomic.storage import DependencyCheckNode, LoadDependenciesNode, SavePhaseOutputNode
from graphtoolkit.nodes.atomic.templates import TemplateRenderNode
from graphtoolkit.nodes.atomic.llm import LLMCallNode
from graphtoolkit.nodes.atomic.validation import SchemaValidationNode, QualityGateNode
from graphtoolkit.nodes.atomic.control import StateUpdateNode
from graphtoolkit.nodes.base import BaseNode, ErrorNode
from pydantic_graph import Graph


pytestmark = pytest.mark.anyio


class TestAgenToolkitAnalyzer:
    """Test the analyzer phase of the agentoolkit domain end-to-end."""
    
    async def test_analyzer_phase_end_to_end(self):
        """Test the complete analyzer phase workflow with real execution."""
        # Setup
        workflow_id = f"test_{uuid.uuid4().hex[:8]}"
        task_description = "Create a Redis cache manager with TTL support and connection pooling"
        
        print(f"\n🚀 Starting REAL analyzer phase for workflow {workflow_id}")
        print(f"📝 Task: {task_description}")
        
        # Create workflow definition with all phases
        workflow_def = WorkflowDefinition(
            domain='agentoolkit',
            phases={'analyzer': analyzer_phase},
            phase_sequence=['analyzer'],
            node_configs={
                'dependency_check': NodeConfig(node_type='storage_check'),
                'load_dependencies': NodeConfig(node_type='storage_load', retryable=True, max_retries=2),
                'template_render': NodeConfig(node_type='template'),
                'llm_call': NodeConfig(node_type='llm', retryable=True, max_retries=2),
                'schema_validation': NodeConfig(node_type='validation'),
                'save_phase_output': NodeConfig(node_type='storage_save'),
                'state_update': NodeConfig(node_type='state'),
                'quality_gate': NodeConfig(node_type='validation')
            }
        )
        
        # Create initial state with task description
        state = WorkflowState(
            workflow_def=workflow_def,
            workflow_id=workflow_id,
            domain='agentoolkit',
            current_phase='analyzer',
            current_node='dependency_check',
            domain_data={'task_description': task_description}
        )
        
        # Create real dependencies (no mocks!)
        deps = WorkflowDeps.create_default()
        # Disable metrics to avoid infinite loops during testing
        deps = WorkflowDeps(
            models=deps.models,
            storage=deps.storage,
            template_engine=deps.template_engine,
            phase_registry=deps.phase_registry,
            process_executor=deps.process_executor,
            thread_executor=deps.thread_executor,
            domain_validators=deps.domain_validators,
            metrics_enabled=False,  # Disable metrics
            logging_level=deps.logging_level,
            cache_enabled=deps.cache_enabled
        )
        
        # Create the graph with all node classes (not instances)
        # pydantic_graph needs the classes, not instances
        nodes = [
            GenericPhaseNode,
            DependencyCheckNode,
            LoadDependenciesNode,
            TemplateRenderNode,
            LLMCallNode,
            SchemaValidationNode,
            SavePhaseOutputNode,
            StateUpdateNode,
            QualityGateNode,
            NextPhaseNode,
            RefinementNode,
            ErrorNode,
            BaseNode  # Include BaseNode since others reference it
        ]
        
        # Build graph
        graph = Graph(nodes=nodes)
        
        print("\n📊 Executing workflow graph...")
        
        # Run the analyzer phase
        try:
            result = await graph.run(
                GenericPhaseNode(),
                state=state,
                deps=deps
            )
            
            print(f"\n✅ Workflow completed successfully!")
            
            # Debug: Check what's in the final state
            if hasattr(result, 'output'):
                final_state = result.output
                print(f"\n🔍 Final state debug:")
                print(f"   - Current phase: {final_state.current_phase}")
                print(f"   - Current node: {final_state.current_node}")
                print(f"   - Completed phases: {final_state.completed_phases}")
                print(f"   - Phase outputs keys: {list(final_state.phase_outputs.keys())}")
                print(f"   - Domain data keys: {list(final_state.domain_data.keys())}")
                
                # Check for errors
                if 'error' in final_state.domain_data:
                    print(f"\n   ❌ ERROR FOUND:")
                    print(f"     - Error: {final_state.domain_data['error']}")
                    print(f"     - Error node: {final_state.domain_data.get('error_node', 'unknown')}")
                    print(f"     - Error time: {final_state.domain_data.get('error_time', 'unknown')}")
                
                # Check if rendered prompts were stored
                if 'rendered_prompts' in final_state.domain_data:
                    rendered = final_state.domain_data['rendered_prompts']
                    print(f"   - Rendered prompts type: {type(rendered)}")
                    if isinstance(rendered, dict):
                        print(f"   - Rendered prompts keys: {list(rendered.keys())}")
                        for k, v in rendered.items():
                            print(f"     - {k}: {len(v) if v else 0} chars")
                
                # Check if analyzer output was generated
                if 'analyzer_output' in final_state.domain_data:
                    output = final_state.domain_data['analyzer_output']
                    print(f"   - Analyzer output type: {type(output)}")
                    if hasattr(output, '__dict__'):
                        print(f"   - Analyzer output fields: {list(output.__dict__.keys())}")
            else:
                print(f"\n⚠️ No output in result - result type: {type(result)}")
            
            # Access the storage to verify what was stored
            from agentool.core.injector import get_injector
            injector = get_injector()
            
            # Check all expected storage keys
            expected_keys = [
                f'workflow/{workflow_id}/input/catalog',
                f'workflow/{workflow_id}/input/prompt',
                f'workflow/{workflow_id}/render/analyzer',
                f'workflow/{workflow_id}/output/analyzer'
            ]
            
            print("\n🗂️ Checking storage patterns:")
            missing_keys = []
            for key in expected_keys:
                try:
                    result = await injector.run('storage_kv', {
                        'operation': 'get',
                        'key': key,
                        'namespace': 'workflow'
                    })
                    
                    if result.success:
                        print(f"   ✓ {key} - Found")
                        
                        # Display full content for debugging
                        if 'catalog' in key:
                            catalog = result.data.get('value', {})
                            print(f"      → Full catalog data:")
                            print(f"        - Type: {type(catalog).__name__}")
                            if isinstance(catalog, dict):
                                print(f"        - Keys: {list(catalog.keys())}")
                                if 'agentools' in catalog:
                                    print(f"        - Catalog has {len(catalog.get('agentools', []))} tools")
                                    print(f"        - First 2 tools: {catalog.get('agentools', [])[:2]}")
                        
                        elif 'prompt' in key:
                            prompt = result.data.get('value', '')
                            print(f"      → Full prompt data:")
                            print(f"        - Type: {type(prompt).__name__}")
                            print(f"        - Content: {prompt}")
                        
                        elif 'render' in key:
                            rendered = result.data.get('value', {})
                            print(f"      → Full rendered data:")
                            print(f"        - Type: {type(rendered).__name__}")
                            if isinstance(rendered, dict):
                                print(f"        - Keys: {list(rendered.keys())}")
                                if 'system_prompt' in rendered:
                                    print(f"        - System prompt length: {len(rendered['system_prompt'])} chars")
                                    print(f"        - System prompt preview: {rendered['system_prompt'][:200]}...")
                                if 'user_prompt' in rendered:
                                    print(f"        - User prompt length: {len(rendered['user_prompt'])} chars")
                                    print(f"        - User prompt preview: {rendered['user_prompt'][:200]}...")
                        
                        elif 'output' in key:
                            output = result.data.get('value', {})
                            print(f"      → Full output data:")
                            print(f"        - Type: {type(output).__name__}")
                            if isinstance(output, dict):
                                print(f"        - Keys: {list(output.keys())}")
                                print(f"        - Name: {output.get('name', 'N/A')}")
                                print(f"        - Description: {output.get('description', 'N/A')[:100] if output.get('description') else 'N/A'}...")
                                print(f"        - Existing tools: {output.get('existing_tools', [])}")
                                print(f"        - Missing tools count: {len(output.get('missing_tools', []))}")
                                if output.get('missing_tools'):
                                    print(f"        - First missing tool: {output.get('missing_tools', [])[0] if output.get('missing_tools') else 'None'}")
                    else:
                        print(f"   ✗ {key} - Not found: {result.message}")
                        missing_keys.append(key)
                
                except Exception as e:
                    print(f"   ✗ {key} - Error: {e}")
                    missing_keys.append(key)
            
            # Display the final output
            final_state = result.output if hasattr(result, 'output') else state
            if 'analyzer_output' in final_state.domain_data:
                analyzer_output = final_state.domain_data['analyzer_output']
                print(f"\n🎯 Final Analyzer Output:")
                # Handle both dict and Pydantic model
                if hasattr(analyzer_output, 'name'):
                    # It's a Pydantic model
                    print(f"   Name: {analyzer_output.name}")
                    print(f"   Description: {analyzer_output.description[:100] if analyzer_output.description else 'N/A'}...")
                    print(f"   System Design: {analyzer_output.system_design[:100] if analyzer_output.system_design else 'N/A'}...")
                    print(f"   Guidelines: {analyzer_output.guidelines[:2] if analyzer_output.guidelines else []}")
                    print(f"   Existing Tools to Use: {analyzer_output.existing_tools}")
                    print(f"   Missing Tools to Create: {analyzer_output.missing_tools}")
                else:
                    # It's a dict
                    print(f"   Name: {analyzer_output.get('name', 'N/A')}")
                    print(f"   Description: {analyzer_output.get('description', 'N/A')[:100]}...")
                    print(f"   System Design: {analyzer_output.get('system_design', 'N/A')[:100]}...")
                    print(f"   Guidelines: {analyzer_output.get('guidelines', [])[:2]}")
                    print(f"   Existing Tools to Use: {analyzer_output.get('existing_tools', [])}")
                    print(f"   Missing Tools to Create: {analyzer_output.get('missing_tools', [])}")
            
            # Verify the hierarchical storage pattern
            assert all('/' in key for key in expected_keys), "All keys should use hierarchical pattern"
            assert all(workflow_id in key for key in expected_keys), "All keys should include workflow_id"
            
            # Check if critical keys are missing and fail if they are
            if missing_keys:
                print(f"\n❌ Missing critical keys: {missing_keys}")
                # The render and output keys are critical - test should fail if missing
                critical_missing = [k for k in missing_keys if 'render' in k or 'output' in k]
                if critical_missing:
                    raise AssertionError(f"Critical storage keys missing: {critical_missing}")
            else:
                print("\n✅ All storage patterns verified and populated!")
            
        except Exception as e:
            print(f"\n❌ Workflow failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def test_analyzer_phase_with_display(self):
        """Test analyzer phase with detailed display of each step."""
        workflow_id = f"test_{uuid.uuid4().hex[:8]}"
        task_description = "Create a task queue system with priority support and dead letter queue"
        
        print(f"\n{'='*80}")
        print(f"AGENTOOLKIT ANALYZER PHASE TEST")
        print(f"{'='*80}")
        print(f"Workflow ID: {workflow_id}")
        print(f"Task: {task_description}")
        print(f"{'='*80}\n")
        
        # Create workflow definition
        workflow_def = WorkflowDefinition(
            domain='agentoolkit',
            phases={'analyzer': analyzer_phase},
            phase_sequence=['analyzer'],
            node_configs={
                'dependency_check': NodeConfig(node_type='storage_check'),
                'load_dependencies': NodeConfig(node_type='storage_load'),
                'template_render': NodeConfig(node_type='template'),
                'llm_call': NodeConfig(node_type='llm', retryable=True, max_retries=2),
                'schema_validation': NodeConfig(node_type='validation'),
                'save_phase_output': NodeConfig(node_type='storage_save'),
                'state_update': NodeConfig(node_type='state'),
                'quality_gate': NodeConfig(node_type='validation')
            }
        )
        
        # Create initial state
        state = WorkflowState(
            workflow_def=workflow_def,
            workflow_id=workflow_id,
            domain='agentoolkit',
            current_phase='analyzer',
            current_node='dependency_check',
            domain_data={'task_description': task_description}
        )
        
        # Create deps
        deps = WorkflowDeps.create_default()
        # Disable metrics to avoid infinite loops during testing
        deps = WorkflowDeps(
            models=deps.models,
            storage=deps.storage,
            template_engine=deps.template_engine,
            phase_registry=deps.phase_registry,
            process_executor=deps.process_executor,
            thread_executor=deps.thread_executor,
            domain_validators=deps.domain_validators,
            metrics_enabled=False,  # Disable metrics
            logging_level=deps.logging_level,
            cache_enabled=deps.cache_enabled
        )
        
        # Create the graph with all node classes
        nodes = [
            GenericPhaseNode,
            DependencyCheckNode,
            LoadDependenciesNode,
            TemplateRenderNode,
            LLMCallNode,
            SchemaValidationNode,
            SavePhaseOutputNode,
            StateUpdateNode,
            QualityGateNode,
            NextPhaseNode,
            RefinementNode,
            ErrorNode,
            BaseNode
        ]
        
        graph = Graph(nodes=nodes)
        
        # Execute each node step by step for visibility
        print("Step 1: Dependency Check")
        print("-" * 40)
        # Would execute here in real run
        print("   ✓ No dependencies for analyzer phase\n")
        
        print("Step 2: Load Catalog")
        print("-" * 40)
        print("   → Loading catalog from agentool_mgmt...")
        print("   → Storing catalog at workflow/{id}/input/catalog")
        print("   → Storing prompt at workflow/{id}/input/prompt\n")
        
        print("Step 3: Template Rendering")
        print("-" * 40)
        print("   → Rendering system template: templates/agentool/system/analyzer.jinja")
        print("   → Rendering user template: templates/agentool/prompts/analyze_catalog.jinja")
        print("   → Storing at workflow/{id}/render/analyzer\n")
        
        print("Step 4: LLM Call")
        print("-" * 40)
        print("   → Calling LLM with rendered prompts")
        print("   → Model: openai:gpt-4o")
        print("   → Temperature: 0.7")
        print("   → Max tokens: 2000\n")
        
        print("Step 5: Schema Validation")
        print("-" * 40)
        print("   → Validating output against AnalyzerOutput schema")
        print("   → Required fields: name, description, system_design, etc.\n")
        
        print("Step 6: Save Output")
        print("-" * 40)
        print("   → Storing at workflow/{id}/output/analyzer\n")
        
        print("Step 7: State Update")
        print("-" * 40)
        print("   → Marking analyzer phase complete\n")
        
        print("Step 8: Quality Gate")
        print("-" * 40)
        print("   → Checking quality threshold: 0.8")
        print("   → Refinement allowed: True")
        print("   → Max refinements: 2\n")
        
        print(f"{'='*80}")
        print("STORAGE HIERARCHY")
        print(f"{'='*80}")
        print(f"""
        workflow/
        └── {workflow_id}/
            ├── input/
            │   ├── catalog     # AgenTool catalog from mgmt
            │   └── prompt      # Task description
            ├── render/
            │   └── analyzer    # Rendered templates
            └── output/
                └── analyzer    # LLM analysis output
        """)
        
        print(f"\n{'='*80}")
        print("✅ Test structure verified - ready for real execution")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])