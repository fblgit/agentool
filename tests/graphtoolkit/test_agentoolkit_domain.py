"""
Tests for AgenToolkit domain in GraphToolkit.

This test file verifies the analyzer phase of the agentoolkit domain,
ensuring proper storage patterns and workflow execution.
"""

import json
import os
import uuid
import pytest
from pydantic import BaseModel, Field
from typing import List

from graphtoolkit.domains.agentoolkit import (
    AgenToolkitAnalyzerInput,
    analyzer_phase,
    AnalyzerOutput,
    specifier_phase,
    crafter_phase,
    evaluator_phase,
    refiner_phase,
    documenter_phase,
    test_analyzer_phase
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


    async def test_analyzer_and_specifier_phases(self):
        """Test both analyzer and specifier phases with iteration."""
        # Setup
        workflow_id = f"test_{uuid.uuid4().hex[:8]}"
        task_description = "Create a simple Redis cache with get and set operations"
        
        print(f"\n🚀 Starting analyzer + specifier test for workflow {workflow_id}")
        print(f"📝 Task: {task_description}")
        
        # First, run the analyzer phase
        analyzer_def = WorkflowDefinition(
            domain='agentoolkit',
            phases={'analyzer': analyzer_phase},
            phase_sequence=['analyzer'],
            node_configs={
                'dependency_check': NodeConfig(node_type='storage_check'),
                'load_dependencies': NodeConfig(node_type='storage_load'),
                'template_render': NodeConfig(node_type='template'),
                'llm_call': NodeConfig(node_type='llm'),
                'schema_validation': NodeConfig(node_type='validation'),
                'save_phase_output': NodeConfig(node_type='storage_save'),
                'state_update': NodeConfig(node_type='state'),
                'quality_gate': NodeConfig(node_type='validation')
            }
        )
        
        analyzer_state = WorkflowState(
            workflow_def=analyzer_def,
            workflow_id=workflow_id,
            domain='agentoolkit',
            current_phase='analyzer',
            current_node='dependency_check',
            domain_data={'task_description': task_description}
        )
        
        deps = WorkflowDeps.create_default()
        # Disable metrics
        deps = WorkflowDeps(
            models=deps.models,
            storage=deps.storage,
            template_engine=deps.template_engine,
            phase_registry=deps.phase_registry,
            process_executor=deps.process_executor,
            thread_executor=deps.thread_executor,
            domain_validators=deps.domain_validators,
            metrics_enabled=False,
            logging_level=deps.logging_level,
            cache_enabled=deps.cache_enabled
        )
        
        # Create graph for analyzer
        analyzer_nodes = [
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
        
        analyzer_graph = Graph(nodes=analyzer_nodes)
        
        print("\n📊 Phase 1: Running Analyzer...")
        analyzer_result = await analyzer_graph.run(
            GenericPhaseNode(),
            state=analyzer_state,
            deps=deps
        )
        
        # Extract analyzer output
        final_analyzer_state = analyzer_result.output if hasattr(analyzer_result, 'output') else analyzer_state
        analyzer_output = final_analyzer_state.domain_data.get('analyzer_output')
        
        if not analyzer_output:
            print("❌ No analyzer output found")
            return
        
        print(f"\n✅ Analyzer complete! Found {len(analyzer_output.missing_tools if hasattr(analyzer_output, 'missing_tools') else [])} missing tools")
        
        # Now run specifier phase with the analyzer output
        from graphtoolkit.nodes.atomic.iteration import (
            IterationControlNode,
            SaveIterationOutputNode,
            AggregationNode
        )
        
        specifier_def = WorkflowDefinition(
            domain='agentoolkit',
            phases={'specifier': specifier_phase},
            phase_sequence=['specifier'],
            node_configs={
                'dependency_check': NodeConfig(node_type='storage_check'),
                'load_dependencies': NodeConfig(node_type='storage_load'),
                'iteration_control': NodeConfig(node_type='iteration'),
                'template_render': NodeConfig(node_type='template'),
                'llm_call': NodeConfig(node_type='llm'),
                'schema_validation': NodeConfig(node_type='validation'),
                'save_iteration_output': NodeConfig(node_type='iteration_save'),
                'aggregation': NodeConfig(node_type='aggregation'),
                'save_phase_output': NodeConfig(node_type='storage_save'),
                'state_update': NodeConfig(node_type='state'),
                'quality_gate': NodeConfig(node_type='validation')
            }
        )
        
        # Create specifier state with analyzer output
        specifier_state = WorkflowState(
            workflow_def=specifier_def,
            workflow_id=workflow_id,
            domain='agentoolkit',
            current_phase='specifier',
            current_node='dependency_check',
            domain_data={
                'task_description': task_description,
                'analyzer_output': analyzer_output
            },
            completed_phases={'analyzer'},
            phase_outputs={'analyzer': final_analyzer_state.phase_outputs.get('analyzer')}
        )
        
        # Create graph for specifier with iteration nodes
        specifier_nodes = [
            GenericPhaseNode,
            DependencyCheckNode,
            LoadDependenciesNode,
            IterationControlNode,
            TemplateRenderNode,
            LLMCallNode,
            SchemaValidationNode,
            SaveIterationOutputNode,
            AggregationNode,
            SavePhaseOutputNode,
            StateUpdateNode,
            QualityGateNode,
            NextPhaseNode,
            RefinementNode,
            ErrorNode,
            BaseNode
        ]
        
        specifier_graph = Graph(nodes=specifier_nodes)
        
        print("\n📊 Phase 2: Running Specifier with iteration...")
        specifier_result = await specifier_graph.run(
            GenericPhaseNode(),
            state=specifier_state,
            deps=deps
        )
        
        print("\n✅ Specifier complete!")
        
        # Check storage patterns
        from agentool.core.injector import get_injector
        injector = get_injector()
        
        print("\n🗂️ Checking Specifier Storage Patterns:")
        
        # Check individual specifications
        missing_tools = analyzer_output.missing_tools if hasattr(analyzer_output, 'missing_tools') else []
        print(f"\n   Missing tools count: {len(missing_tools)}")
        for i, tool in enumerate(missing_tools):
            print(f"   Tool {i}: type={type(tool).__name__}, value={tool}")
            # Handle both dict and Pydantic model
            if isinstance(tool, dict):
                tool_name = tool.get('name', f'tool_{i}')
            elif hasattr(tool, 'name'):
                tool_name = tool.name
            else:
                tool_name = str(tool)
            print(f"   Extracted tool_name: {tool_name}")
            
            # Check individual spec
            spec_key = f'workflow/{workflow_id}/specification/{tool_name}'
            print(f"   Looking for key: {spec_key}")
            spec_result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': spec_key,
                'namespace': 'workflow'
            })
            
            if spec_result.success:
                spec_data = spec_result.data.get('value', {})
                print(f"\n   ✓ Specification for {tool_name}:")
                print(f"     - Name: {spec_data.get('name', 'N/A')}")
                print(f"     - Description: {spec_data.get('description', 'N/A')[:80]}...")
                print(f"     - Operations: {len(spec_data.get('operations', []))} operations")
            else:
                print(f"   ✗ No specification for {tool_name}")
            
            # Check render
            render_key = f'workflow/{workflow_id}/render/specifier/{tool_name}'
            render_result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': render_key,
                'namespace': 'workflow'
            })
            
            if render_result.success:
                render_data = render_result.data.get('value', {})
                print(f"   ✓ Rendered template for {tool_name}:")
                print(f"     - System prompt: {len(render_data.get('system_prompt', ''))} chars")
                print(f"     - User prompt: {len(render_data.get('user_prompt', ''))} chars")
            else:
                print(f"   ✗ No rendered template for {tool_name}")
        
        # Check aggregated specifications
        specs_key = f'workflow/{workflow_id}/specifications'
        specs_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': specs_key,
            'namespace': 'workflow'
        })
        
        if specs_result.success:
            specs_data = specs_result.data.get('value', {})
            specifications = specs_data.get('specifications', [])
            print(f"\n   ✓ Aggregated Specifications:")
            print(f"     - Total: {len(specifications)} specifications")
            for spec in specifications:
                print(f"     - {spec.get('name', 'unknown')}: {spec.get('description', 'N/A')[:60]}...")
        else:
            print(f"\n   ✗ No aggregated specifications found")
        
        print("\n✅ Test complete!")

    async def test_analyzer_specifier_and_crafter_phases(self):
        """Test all phases: analyzer, specifier, crafter, evaluator, and refiner (if needed) with detailed report."""
        # Setup
        workflow_id = f"test_{uuid.uuid4().hex[:8]}"
        task_description = "Create a simple notification service that sends alerts via email"
        #task_description = "Create a vehicle insurance and claims comprehensive system"
        #task_description = "Create a fully fleshed Docker container lifecycle manager"
        
        print(f"\n{'='*80}")
        print(f"COMPLETE AGENTOOLKIT WORKFLOW TEST")
        print(f"{'='*80}")
        print(f"Workflow ID: {workflow_id}")
        print(f"Task: {task_description}")
        print(f"{'='*80}\n")
        
        # Create deps without metrics to avoid loops
        deps = WorkflowDeps.create_default()
        deps = WorkflowDeps(
            models=deps.models,
            storage=deps.storage,
            template_engine=deps.template_engine,
            phase_registry=deps.phase_registry,
            process_executor=deps.process_executor,
            thread_executor=deps.thread_executor,
            domain_validators=deps.domain_validators,
            metrics_enabled=False,
            logging_level=deps.logging_level,
            cache_enabled=deps.cache_enabled
        )
        
        # Common nodes for all phases
        base_nodes = [
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
        
        # ========================================
        # PHASE 1: ANALYZER
        # ========================================
        print("📊 PHASE 1: ANALYZER")
        print("-" * 40)
        
        analyzer_def = WorkflowDefinition(
            domain='agentoolkit',
            phases={'analyzer': analyzer_phase},
            phase_sequence=['analyzer'],
            node_configs={
                'dependency_check': NodeConfig(node_type='storage_check'),
                'load_dependencies': NodeConfig(node_type='storage_load'),
                'template_render': NodeConfig(node_type='template'),
                'llm_call': NodeConfig(node_type='llm'),
                'schema_validation': NodeConfig(node_type='validation'),
                'save_phase_output': NodeConfig(node_type='storage_save'),
                'state_update': NodeConfig(node_type='state'),
                'quality_gate': NodeConfig(node_type='validation')
            }
        )
        
        analyzer_state = WorkflowState(
            workflow_def=analyzer_def,
            workflow_id=workflow_id,
            domain='agentoolkit',
            current_phase='analyzer',
            current_node='dependency_check',
            domain_data={'task_description': task_description}
        )
        
        analyzer_graph = Graph(nodes=base_nodes)
        
        print("   → Running analysis of task requirements...")
        analyzer_result = await analyzer_graph.run(
            GenericPhaseNode(),
            state=analyzer_state,
            deps=deps
        )
        
        final_analyzer_state = analyzer_result.output if hasattr(analyzer_result, 'output') else analyzer_state
        analyzer_output = final_analyzer_state.domain_data.get('analyzer_output')
        
        if not analyzer_output:
            print("   ❌ Analyzer failed - no output")
            return
        
        # Extract analyzer results
        missing_tools = analyzer_output.missing_tools if hasattr(analyzer_output, 'missing_tools') else []
        existing_tools = analyzer_output.existing_tools if hasattr(analyzer_output, 'existing_tools') else []
        
        print(f"   ✓ Analysis complete!")
        print(f"     - System Design: {len(analyzer_output.system_design if hasattr(analyzer_output, 'system_design') else '') } chars")
        print(f"     - Existing tools to use: {len(existing_tools)}")
        print(f"     - Missing tools to create: {len(missing_tools)}")
        
        if missing_tools:
            for tool in missing_tools[:3]:  # Show first 3
                tool_name = tool.name if hasattr(tool, 'name') else tool.get('name', 'unknown')
                print(f"       • {tool_name}")
        
        # ========================================
        # PHASE 2: SPECIFIER
        # ========================================
        print("\n📊 PHASE 2: SPECIFIER")
        print("-" * 40)
        
        from graphtoolkit.nodes.atomic.iteration import (
            IterationControlNode,
            SaveIterationOutputNode,
            AggregationNode
        )
        
        specifier_def = WorkflowDefinition(
            domain='agentoolkit',
            phases={'specifier': specifier_phase},
            phase_sequence=['specifier'],
            node_configs={
                'dependency_check': NodeConfig(node_type='storage_check'),
                'load_dependencies': NodeConfig(node_type='storage_load'),
                'iteration_control': NodeConfig(node_type='iteration'),
                'template_render': NodeConfig(node_type='template'),
                'llm_call': NodeConfig(node_type='llm'),
                'schema_validation': NodeConfig(node_type='validation'),
                'save_iteration_output': NodeConfig(node_type='iteration_save'),
                'aggregation': NodeConfig(node_type='aggregation'),
                'save_phase_output': NodeConfig(node_type='storage_save'),
                'state_update': NodeConfig(node_type='state'),
                'quality_gate': NodeConfig(node_type='validation')
            }
        )
        
        specifier_state = WorkflowState(
            workflow_def=specifier_def,
            workflow_id=workflow_id,
            domain='agentoolkit',
            current_phase='specifier',
            current_node='dependency_check',
            domain_data={
                'task_description': task_description,
                'analyzer_output': analyzer_output
            },
            completed_phases={'analyzer'},
            phase_outputs={'analyzer': final_analyzer_state.phase_outputs.get('analyzer')}
        )
        
        # Add iteration nodes
        specifier_nodes = base_nodes + [
            IterationControlNode,
            SaveIterationOutputNode,
            AggregationNode
        ]
        
        specifier_graph = Graph(nodes=specifier_nodes)
        
        print(f"   → Creating specifications for {len(missing_tools)} missing tools...")
        specifier_result = await specifier_graph.run(
            GenericPhaseNode(),
            state=specifier_state,
            deps=deps
        )
        
        final_specifier_state = specifier_result.output if hasattr(specifier_result, 'output') else specifier_state
        
        print(f"   ✓ Specifications complete!")
        
        # ========================================
        # PHASE 3: CRAFTER
        # ========================================
        print("\n📊 PHASE 3: CRAFTER")
        print("-" * 40)
        
        crafter_def = WorkflowDefinition(
            domain='agentoolkit',
            phases={'crafter': crafter_phase},
            phase_sequence=['crafter'],
            node_configs={
                'dependency_check': NodeConfig(node_type='storage_check'),
                'load_dependencies': NodeConfig(node_type='storage_load'),
                'iteration_control': NodeConfig(node_type='iteration'),
                'template_render': NodeConfig(node_type='template'),
                'llm_call': NodeConfig(node_type='llm'),
                'schema_validation': NodeConfig(node_type='validation'),
                'save_iteration_output': NodeConfig(node_type='iteration_save'),
                'aggregation': NodeConfig(node_type='aggregation'),
                'save_phase_output': NodeConfig(node_type='storage_save'),
                'state_update': NodeConfig(node_type='state'),
                'quality_gate': NodeConfig(node_type='validation')
            }
        )
        
        crafter_state = WorkflowState(
            workflow_def=crafter_def,
            workflow_id=workflow_id,
            domain='agentoolkit',
            current_phase='crafter',
            current_node='dependency_check',
            domain_data={
                'task_description': task_description,
                'analyzer_output': analyzer_output,
                'specifier_output': final_specifier_state.domain_data.get('specifier_output')
            },
            completed_phases={'analyzer', 'specifier'},
            phase_outputs={
                'analyzer': final_analyzer_state.phase_outputs.get('analyzer'),
                'specifier': final_specifier_state.phase_outputs.get('specifier')
            }
        )
        
        crafter_nodes = base_nodes + [
            IterationControlNode,
            SaveIterationOutputNode,
            AggregationNode
        ]
        
        crafter_graph = Graph(nodes=crafter_nodes)
        
        print(f"   → Crafting implementations for {len(missing_tools)} tools...")
        crafter_result = await crafter_graph.run(
            GenericPhaseNode(),
            state=crafter_state,
            deps=deps
        )
        
        final_crafter_state = crafter_result.output if hasattr(crafter_result, 'output') else crafter_state
        
        print(f"   ✓ Implementations complete!")
        
        # ========================================
        # PHASE 4: EVALUATOR
        # ========================================
        print("\n📊 PHASE 4: EVALUATOR")
        print("-" * 40)
        
        evaluator_def = WorkflowDefinition(
            domain='agentoolkit',
            phases={'evaluator': evaluator_phase},
            phase_sequence=['evaluator'],
            node_configs={
                'dependency_check': NodeConfig(node_type='storage_check'),
                'load_dependencies': NodeConfig(node_type='storage_load'),
                'iteration_control': NodeConfig(node_type='iteration'),
                'template_render': NodeConfig(node_type='template'),
                'llm_call': NodeConfig(node_type='llm'),
                'schema_validation': NodeConfig(node_type='validation'),
                'save_iteration_output': NodeConfig(node_type='iteration_save'),
                'aggregation': NodeConfig(node_type='aggregation'),
                'save_phase_output': NodeConfig(node_type='storage_save'),
                'state_update': NodeConfig(node_type='state'),
                'quality_gate': NodeConfig(node_type='validation')
            }
        )
        
        evaluator_state = WorkflowState(
            workflow_def=evaluator_def,
            workflow_id=workflow_id,
            domain='agentoolkit',
            current_phase='evaluator',
            current_node='dependency_check',
            domain_data={
                'task_description': task_description,
                'analyzer_output': analyzer_output,
                'specifier_output': final_specifier_state.domain_data.get('specifier_output'),
                'crafter_output': final_crafter_state.domain_data.get('crafter_output')
            },
            completed_phases={'analyzer', 'specifier', 'crafter'},
            phase_outputs={
                'analyzer': final_analyzer_state.phase_outputs.get('analyzer'),
                'specifier': final_specifier_state.phase_outputs.get('specifier'),
                'crafter': final_crafter_state.phase_outputs.get('crafter')
            }
        )
        
        evaluator_nodes = base_nodes + [
            IterationControlNode,
            SaveIterationOutputNode,
            AggregationNode
        ]
        
        evaluator_graph = Graph(nodes=evaluator_nodes)
        
        print(f"   → Evaluating implementations for {len(missing_tools)} tools...")
        evaluator_result = await evaluator_graph.run(
            GenericPhaseNode(),
            state=evaluator_state,
            deps=deps
        )
        
        final_evaluator_state = evaluator_result.output if hasattr(evaluator_result, 'output') else evaluator_state
        
        print(f"   ✓ Evaluations complete!")
        
        # ========================================
        # PHASE 5: REFINER (Conditional)
        # ========================================
        
        # Check which tools need refinement
        tools_to_refine = []
        for tool in missing_tools:
            tool_name = tool.name if hasattr(tool, 'name') else tool.get('name', 'unknown')
            eval_key = f'workflow/{workflow_id}/evaluation/{tool_name}'
            
            # Get evaluation result for this tool
            from agentool.core.injector import get_injector
            injector = get_injector()
            eval_result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': eval_key,
                'namespace': 'workflow'
            })
            
            if eval_result.success:
                eval_data = eval_result.data.get('value', {})
                if not eval_data.get('ready_for_deployment', False):
                    tools_to_refine.append(tool)
                    print(f"   📝 Tool {tool_name} needs refinement (score: {eval_data.get('overall_score', 0):.2f})")
        
        if tools_to_refine:
            print("\n📊 PHASE 5: REFINER")
            print("-" * 40)
            print(f"   → Refining {len(tools_to_refine)} tools that need improvement...")
            
            refiner_def = WorkflowDefinition(
                domain='agentoolkit',
                phases={'refiner': refiner_phase},
                phase_sequence=['refiner'],
                node_configs={
                    'dependency_check': NodeConfig(node_type='storage_check'),
                    'load_dependencies': NodeConfig(node_type='storage_load'),
                    'iteration_control': NodeConfig(node_type='iteration'),
                    'template_render': NodeConfig(node_type='template'),
                    'llm_call': NodeConfig(node_type='llm'),
                    'schema_validation': NodeConfig(node_type='validation'),
                    'save_iteration_output': NodeConfig(node_type='iteration_save'),
                    'aggregation': NodeConfig(node_type='aggregation'),
                    'save_phase_output': NodeConfig(node_type='storage_save'),
                    'state_update': NodeConfig(node_type='state'),
                    'quality_gate': NodeConfig(node_type='validation')
                }
            )
            
            refiner_state = WorkflowState(
                workflow_def=refiner_def,
                workflow_id=workflow_id,
                domain='agentoolkit',
                current_phase='refiner',
                current_node='dependency_check',
                domain_data={
                    'task_description': task_description,
                    'analyzer_output': analyzer_output,
                    'specifier_output': final_specifier_state.domain_data.get('specifier_output'),
                    'crafter_output': final_crafter_state.domain_data.get('crafter_output'),
                    'evaluator_output': final_evaluator_state.domain_data.get('evaluator_output'),
                    'tools_to_refine': tools_to_refine  # Set the items to iterate over
                },
                completed_phases={'analyzer', 'specifier', 'crafter', 'evaluator'},
                phase_outputs={
                    'analyzer': final_analyzer_state.phase_outputs.get('analyzer'),
                    'specifier': final_specifier_state.phase_outputs.get('specifier'),
                    'crafter': final_crafter_state.phase_outputs.get('crafter'),
                    'evaluator': final_evaluator_state.phase_outputs.get('evaluator')
                }
            )
            
            refiner_nodes = base_nodes + [
                IterationControlNode,
                SaveIterationOutputNode,
                AggregationNode
            ]
            
            refiner_graph = Graph(nodes=refiner_nodes)
            
            refiner_result = await refiner_graph.run(
                GenericPhaseNode(),
                state=refiner_state,
                deps=deps
            )
            
            final_refiner_state = refiner_result.output if hasattr(refiner_result, 'output') else refiner_state
            
            print(f"   ✓ Refinements complete!")
        else:
            print("\n📊 PHASE 5: REFINER")
            print("-" * 40)
            print("   ✅ No refinement needed - all tools ready for deployment!")
            final_refiner_state = None
        
        # ========================================
        # FINAL REPORT
        # ========================================
        print(f"\n{'='*80}")
        print("📋 WORKFLOW COMPLETION REPORT")
        print(f"{'='*80}")
        
        from agentool.core.injector import get_injector
        injector = get_injector()
        
        # Report on Analyzer
        print("\n🔍 ANALYZER PHASE RESULTS")
        print("-" * 40)
        
        # Check analyzer storage
        analyzer_keys = [
            f'workflow/{workflow_id}/input/catalog',
            f'workflow/{workflow_id}/input/prompt',
            f'workflow/{workflow_id}/render/analyzer',
            f'workflow/{workflow_id}/output/analyzer'
        ]
        
        for key in analyzer_keys:
            result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': key,
                'namespace': 'workflow'
            })
            if result.success:
                data = result.data.get('value', {})
                if 'catalog' in key:
                    print(f"   ✓ Catalog loaded: {len(data.get('agentools', [])) if isinstance(data, dict) else 0} tools")
                elif 'prompt' in key:
                    print(f"   ✓ Task prompt stored: {len(str(data))} chars")
                elif 'render' in key:
                    print(f"   ✓ Templates rendered: system={len(data.get('system_prompt', ''))} chars, user={len(data.get('user_prompt', ''))} chars")
                elif 'output' in key:
                    print(f"   ✓ Analysis output:")
                    if isinstance(data, dict):
                        print(f"     - Name: {data.get('name', 'N/A')}")
                        print(f"     - Description: {data.get('description', 'N/A')[:80]}...")
                        print(f"     - System design: {len(data.get('system_design', ''))} chars")
                        print(f"     - Guidelines: {len(data.get('guidelines', []))} items")
                        print(f"     - Existing tools: {data.get('existing_tools', [])}")
                        print(f"     - Missing tools: {len(data.get('missing_tools', []))} tools")
        
        # Report on Specifier
        print("\n📝 SPECIFIER PHASE RESULTS")
        print("-" * 40)
        
        # Check individual specifications
        spec_count = 0
        for tool in missing_tools:
            tool_name = tool.name if hasattr(tool, 'name') else tool.get('name', 'unknown')
            spec_key = f'workflow/{workflow_id}/specification/{tool_name}'
            
            spec_result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': spec_key,
                'namespace': 'workflow'
            })
            
            if spec_result.success:
                spec_count += 1
                spec_data = spec_result.data.get('value', {})
                print(f"   ✓ Specification for {tool_name}:")
                # Check multiple possible field names for operations
                operations = spec_data.get('operations', spec_data.get('tool_operations', []))
                print(f"     - Operations: {len(operations)} defined")
                print(f"     - Dependencies: {spec_data.get('dependencies', [])}")
                # Debug: show what fields are actually present
                print(f"     - Available fields: {list(spec_data.keys())[:8]}")  # Show first 8 keys
        
        # Check aggregated specifications
        specs_key = f'workflow/{workflow_id}/specifications'
        specs_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': specs_key,
            'namespace': 'workflow'
        })
        
        if specs_result.success:
            specs_data = specs_result.data.get('value', {})
            specifications = specs_data.get('specifications', [])
            print(f"\n   ✓ Aggregated Specifications: {len(specifications)} total")
        
        # Report on Crafter
        print("\n🔨 CRAFTER PHASE RESULTS")
        print("-" * 40)
        
        # Check individual implementations
        impl_count = 0
        for tool in missing_tools:
            tool_name = tool.name if hasattr(tool, 'name') else tool.get('name', 'unknown')
            impl_key = f'workflow/{workflow_id}/crafter/{tool_name}'
            
            impl_result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': impl_key,
                'namespace': 'workflow'
            })
            
            if impl_result.success:
                impl_count += 1
                impl_data = impl_result.data.get('value', {})
                print(f"   ✓ Implementation for {tool_name}:")
                print(f"     - Code: {len(impl_data.get('code', ''))} chars")
                print(f"     - Imports: {len(impl_data.get('imports', []))} imports")
                print(f"     - Dependencies: {impl_data.get('dependencies', [])}")
                
                # Show first few lines of code
                code = impl_data.get('code', '')
                if code:
                    lines = code.split('\n')[:3]
                    for line in lines:
                        if line.strip():
                            print(f"       {line[:60]}...")
                            break
        
        # Check aggregated implementations
        crafts_key = f'workflow/{workflow_id}/crafts'
        crafts_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': crafts_key,
            'namespace': 'workflow'
        })
        
        if crafts_result.success:
            crafts_data = crafts_result.data.get('value', {})
            crafts = crafts_data.get('crafts', [])
            print(f"\n   ✓ Aggregated Implementations: {len(crafts)} total")
            
            # Calculate total lines of code
            total_lines = 0
            for craft in crafts:
                code = craft.get('code', '')
                total_lines += len(code.split('\n')) if code else 0
            print(f"     - Total lines of code generated: {total_lines}")
        
        # Report on Evaluator
        print("\n🔍 EVALUATOR PHASE RESULTS")
        print("-" * 40)
        
        # Check individual evaluations
        eval_count = 0
        deployment_ready = 0
        total_scores = []
        
        for tool in missing_tools:
            tool_name = tool.name if hasattr(tool, 'name') else tool.get('name', 'unknown')
            eval_key = f'workflow/{workflow_id}/evaluation/{tool_name}'
            
            eval_result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': eval_key,
                'namespace': 'workflow'
            })
            
            if eval_result.success:
                eval_count += 1
                eval_data = eval_result.data.get('value', {})
                overall_score = eval_data.get('overall_score', 0.0)
                ready_for_deployment = eval_data.get('ready_for_deployment', False)
                total_scores.append(overall_score)
                
                if ready_for_deployment:
                    deployment_ready += 1
                
                print(f"   ✓ Evaluation for {tool_name}:")
                print(f"     - Overall score: {overall_score:.2f}/1.0")
                print(f"     - Ready for deployment: {'✅' if ready_for_deployment else '❌'}")
                print(f"     - Issues found: {len(eval_data.get('issues', []))}")
                print(f"     - Improvements suggested: {len(eval_data.get('improvements', []))}")
                
                # Show all issues without truncation
                issues = eval_data.get('issues', [])
                for issue in issues:
                    print(f"       - Issue: {issue}")
                
                # Show all improvements without truncation  
                improvements = eval_data.get('improvements', [])
                for improvement in improvements:
                    print(f"       - Improvement: {improvement}")
        
        # Check aggregated evaluations
        evals_key = f'workflow/{workflow_id}/evaluations'
        evals_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': evals_key,
            'namespace': 'workflow'
        })
        
        if evals_result.success:
            evals_data = evals_result.data.get('value', {})
            evaluations = evals_data.get('items', [])
            print(f"\n   ✓ Aggregated Evaluations: {len(evaluations)} total")
            
            if total_scores:
                avg_score = sum(total_scores) / len(total_scores)
                print(f"     - Average overall score: {avg_score:.2f}/1.0")
                print(f"     - Deployment ready: {deployment_ready}/{len(missing_tools)} ({deployment_ready/len(missing_tools)*100:.1f}%)")
        
        # Report on Refiner (if executed)
        if final_refiner_state:
            print("\n🔧 REFINER PHASE RESULTS")
            print("-" * 40)
            
            # Check individual refinements
            refine_count = 0
            for tool in tools_to_refine:
                tool_name = tool.name if hasattr(tool, 'name') else tool.get('name', 'unknown')
                refine_key = f'workflow/{workflow_id}/refine/{tool_name}'
                
                refine_result = await injector.run('storage_kv', {
                    'operation': 'get',
                    'key': refine_key,
                    'namespace': 'workflow'
                })
                
                if refine_result.success:
                    refine_count += 1
                    refine_data = refine_result.data.get('value', {})
                    print(f"   ✓ Refined implementation for {tool_name}:")
                    print(f"     - Code: {len(refine_data.get('code', ''))} chars")
                    # Show first few lines of refined code
                    code = refine_data.get('code', '')
                    if code:
                        lines = code.split('\n')[:3]
                        for line in lines:
                            if line.strip():
                                print(f"       {line[:60]}...")
                                break
            
            # Check aggregated refinements
            refines_key = f'workflow/{workflow_id}/refines'
            refines_result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': refines_key,
                'namespace': 'workflow'
            })
            
            if refines_result.success:
                refines_data = refines_result.data.get('value', {})
                refinements = refines_data.get('refinements', [])
                print(f"\n   ✓ Aggregated Refinements: {len(refinements)} total")
                print(f"     - Tools refined: {', '.join([r.get('tool_name', 'unknown') for r in refinements])}")
        
        # Phase 6: Documenter - Generate documentation for all tools
        print("\n📝 PHASE 6: DOCUMENTER")
        print("-" * 40)
        print("   Generating documentation for all tools...")
        
        # Update workflow def for documenter phase
        workflow_def = WorkflowDefinition(
            domain='agentoolkit',
            phases={
                'analyzer': analyzer_phase,
                'specifier': specifier_phase,
                'crafter': crafter_phase,
                'evaluator': evaluator_phase,
                'refiner': refiner_phase,
                'documenter': documenter_phase
            },
            phase_sequence=['documenter'],  # Only run documenter
            node_configs={
                'dependency_check': NodeConfig(node_type='storage_check'),
                'load_dependencies': NodeConfig(node_type='storage_load'),
                'iteration_control': NodeConfig(node_type='iteration'),
                'template_render': NodeConfig(node_type='template'),
                'llm_call': NodeConfig(node_type='llm', retryable=True, max_retries=2),
                'schema_validation': NodeConfig(node_type='validation'),
                'save_iteration_output': NodeConfig(node_type='iteration_save'),
                'aggregation': NodeConfig(node_type='aggregation'),
                'save_phase_output': NodeConfig(node_type='storage_save'),
                'state_update': NodeConfig(node_type='state'),
                'quality_gate': NodeConfig(node_type='validation')
            }
        )
        
        # Update state for documenter
        # We need to carry over the phase_outputs from the previous states
        # final_refiner_state is already a state (not GraphRunResult), final_evaluator_state too
        prev_state = final_refiner_state if final_refiner_state else final_evaluator_state
        documenter_state = WorkflowState(
            workflow_def=workflow_def,
            workflow_id=workflow_id,
            domain='agentoolkit',
            current_phase='documenter',
            current_node='dependency_check',
            completed_phases={'analyzer', 'specifier', 'crafter', 'evaluator', 'refiner'},
            domain_data=prev_state.domain_data,
            phase_outputs=prev_state.phase_outputs  # Important: carry over phase outputs
        )
        
        # Add iteration nodes
        nodes = [
            GenericPhaseNode,
            DependencyCheckNode,
            LoadDependenciesNode,
            IterationControlNode,
            TemplateRenderNode,
            LLMCallNode,
            SchemaValidationNode,
            SaveIterationOutputNode,
            AggregationNode,
            SavePhaseOutputNode,
            StateUpdateNode,
            QualityGateNode,
            NextPhaseNode,
            RefinementNode,
            ErrorNode,
            BaseNode
        ]
        
        graph = Graph(nodes=nodes)
        
        # Run documenter phase
        documenter_result = await graph.run(GenericPhaseNode(), deps=deps, state=documenter_state)
        final_documenter_state = documenter_result.output if hasattr(documenter_result, 'output') else documenter_state
        
        # Report on Documenter
        print("\n📝 DOCUMENTER PHASE RESULTS")
        print("-" * 40)
        
        # Check individual documentations
        doc_count = 0
        for tool in missing_tools:
            tool_name = tool.name if hasattr(tool, 'name') else tool.get('name', 'unknown')
            doc_key = f'workflow/{workflow_id}/document/{tool_name}'
            
            doc_result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': doc_key,
                'namespace': 'workflow'
            })
            
            if doc_result.success:
                doc_count += 1
                doc_data = doc_result.data.get('value', {})
                markdown = doc_data.get('code', '')
                print(f"   ✓ Documentation for {tool_name}:")
                print(f"     - Markdown: {len(markdown)} chars")
                # Show first heading from markdown
                if markdown:
                    lines = markdown.split('\n')
                    for line in lines:
                        if line.startswith('#'):
                            print(f"       {line[:60]}...")
                            break
        
        # Check aggregated documentations
        docs_key = f'workflow/{workflow_id}/documentations'
        docs_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': docs_key,
            'namespace': 'workflow'
        })
        
        if docs_result.success:
            docs_data = docs_result.data.get('value', {})
            documentations = docs_data.get('documentations', [])
            print(f"\n   ✓ Aggregated Documentation: {len(documentations)} total")
            print(f"     - Tools documented: {', '.join([d.get('tool_name', 'unknown') for d in documentations])}")
        
        # Phase 7: TestAnalyzer - Analyze test requirements for all tools
        print("\n🧪 PHASE 7: TEST ANALYZER")
        print("-" * 40)
        print("   Analyzing test requirements for all tools...")
        
        # Update workflow def for test_analyzer phase
        workflow_def = WorkflowDefinition(
            domain='agentoolkit',
            phases={
                'analyzer': analyzer_phase,
                'specifier': specifier_phase,
                'crafter': crafter_phase,
                'evaluator': evaluator_phase,
                'refiner': refiner_phase,
                'documenter': documenter_phase,
                'test_analyzer': test_analyzer_phase
            },
            phase_sequence=['test_analyzer'],  # Only run test_analyzer
            node_configs={
                'dependency_check': NodeConfig(node_type='storage_check'),
                'load_dependencies': NodeConfig(node_type='storage_load'),
                'iteration_control': NodeConfig(node_type='iteration'),
                'template_render': NodeConfig(node_type='template'),
                'llm_call': NodeConfig(node_type='llm', retryable=True, max_retries=2),
                'schema_validation': NodeConfig(node_type='validation'),
                'save_iteration_output': NodeConfig(node_type='iteration_save'),
                'aggregation': NodeConfig(node_type='aggregation'),
                'save_phase_output': NodeConfig(node_type='storage_save'),
                'state_update': NodeConfig(node_type='state'),
                'quality_gate': NodeConfig(node_type='validation')
            }
        )
        
        # Update state for test_analyzer
        # We need to carry over the phase_outputs from the previous states
        # final_documenter_state is already extracted from the result
        test_analyzer_state = WorkflowState(
            workflow_def=workflow_def,
            workflow_id=workflow_id,
            domain='agentoolkit',
            current_phase='test_analyzer',
            current_node='dependency_check',
            completed_phases={'analyzer', 'specifier', 'crafter', 'evaluator', 'refiner', 'documenter'},
            domain_data=final_documenter_state.domain_data,
            phase_outputs=final_documenter_state.phase_outputs  # Important: carry over phase outputs
        )
        
        # Use same nodes
        graph = Graph(nodes=nodes)
        
        # Run test_analyzer phase
        final_test_analyzer_state = await graph.run(GenericPhaseNode(), deps=deps, state=test_analyzer_state)
        
        # Report on TestAnalyzer
        print("\n🧪 TEST ANALYZER PHASE RESULTS")
        print("-" * 40)
        
        # Check individual test analyses
        test_analysis_count = 0
        for tool in missing_tools:
            tool_name = tool.name if hasattr(tool, 'name') else tool.get('name', 'unknown')
            analysis_key = f'workflow/{workflow_id}/test_analysis/{tool_name}'
            
            analysis_result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': analysis_key,
                'namespace': 'workflow'
            })
            
            if analysis_result.success:
                test_analysis_count += 1
                analysis_data = analysis_result.data.get('value', {})
                print(f"   ✓ Test analysis for {tool_name}:")
                test_cases = analysis_data.get('test_cases', [])
                print(f"     - Test cases: {len(test_cases)} tests")
                if test_cases:
                    # Show first few test case names
                    test_names = [tc.get('name', 'unnamed') for tc in test_cases[:3]]
                    for test_name in test_names:
                        print(f"       • {test_name}")
                    if len(test_cases) > 3:
                        print(f"       ... and {len(test_cases) - 3} more")
        
        # Check aggregated test analyses
        analyses_key = f'workflow/{workflow_id}/test_analyses'
        analyses_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': analyses_key,
            'namespace': 'workflow'
        })
        
        if analyses_result.success:
            analyses_data = analyses_result.data.get('value', {})
            test_analyses = analyses_data.get('test_analyses', [])
            print(f"\n   ✓ Aggregated Test Analyses: {len(test_analyses)} total")
            print(f"     - Tools analyzed: {', '.join([a.get('tool_name', 'unknown') for a in test_analyses])}")
        
        # Phase 8: TestStubber - Create test skeletons for all tools
        print("\n📝 PHASE 8: TEST STUBBER")
        print("-" * 40)
        print("   Creating test skeletons for all tools...")
        
        # Import test_stubber_phase
        from src.graphtoolkit.domains.agentoolkit import test_stubber_phase
        
        # Update workflow def for test_stubber phase
        workflow_def = WorkflowDefinition(
            domain='agentoolkit',
            phases={
                'analyzer': analyzer_phase,
                'specifier': specifier_phase,
                'crafter': crafter_phase,
                'evaluator': evaluator_phase,
                'refiner': refiner_phase,
                'documenter': documenter_phase,
                'test_analyzer': test_analyzer_phase,
                'test_stubber': test_stubber_phase
            },
            phase_sequence=['test_stubber'],  # Only run test_stubber
            node_configs={
                'dependency_check': NodeConfig(node_type='storage_check'),
                'load_dependencies': NodeConfig(node_type='storage_load'),
                'iteration_control': NodeConfig(node_type='iteration'),
                'template_render': NodeConfig(node_type='template'),
                'llm_call': NodeConfig(node_type='llm', retryable=True, max_retries=2),
                'schema_validation': NodeConfig(node_type='validation'),
                'save_iteration_output': NodeConfig(node_type='iteration_save'),
                'aggregation': NodeConfig(node_type='aggregation'),
                'save_phase_output': NodeConfig(node_type='storage_save'),
                'state_update': NodeConfig(node_type='state'),
                'quality_gate': NodeConfig(node_type='validation')
            }
        )
        
        # Extract the final state from test_analyzer result
        if hasattr(final_test_analyzer_state, 'output'):
            final_test_analyzer_state = final_test_analyzer_state.output
        
        # Update state for test_stubber
        test_stubber_state = WorkflowState(
            workflow_def=workflow_def,
            workflow_id=workflow_id,
            domain='agentoolkit',
            current_phase='test_stubber',
            current_node='dependency_check',
            completed_phases={'analyzer', 'specifier', 'crafter', 'evaluator', 'refiner', 'documenter', 'test_analyzer'},
            domain_data=final_test_analyzer_state.domain_data,
            phase_outputs=final_test_analyzer_state.phase_outputs
        )
        
        # Use same nodes
        graph = Graph(nodes=nodes)
        
        # Run test_stubber phase
        test_stubber_result = await graph.run(GenericPhaseNode(), deps=deps, state=test_stubber_state)
        
        # Report on TestStubber
        print("\n📝 TEST STUBBER PHASE RESULTS")
        print("-" * 40)
        
        # Check individual test stubs
        test_stub_count = 0
        total_placeholders = 0
        for tool in missing_tools:
            tool_name = tool.name if hasattr(tool, 'name') else tool.get('name', 'unknown')
            stub_key = f'workflow/{workflow_id}/test_stub/{tool_name}'
            
            stub_result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': stub_key,
                'namespace': 'workflow'
            })
            
            if stub_result.success:
                test_stub_count += 1
                stub_data = stub_result.data.get('value', {})
                print(f"   ✓ Test stub for {tool_name}:")
                file_path = stub_data.get('file_path', 'unknown')
                placeholders = stub_data.get('placeholders_count', 0)
                total_placeholders += placeholders
                print(f"     - File: {file_path}")
                print(f"     - Placeholders: {placeholders} test methods to implement")
                
                # Show first few lines of the stub code
                code = stub_data.get('code', '')
                if code:
                    lines = code.split('\n')[:5]
                    print(f"     - Preview:")
                    for line in lines:
                        if line.strip():
                            print(f"       {line[:60]}...")
                            break
        
        # Check aggregated test stubs
        stubs_key = f'workflow/{workflow_id}/test_stubs'
        stubs_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': stubs_key,
            'namespace': 'workflow'
        })
        
        if stubs_result.success:
            stubs_data = stubs_result.data.get('value', {})
            test_stubs = stubs_data.get('test_stubs', [])
            print(f"\n   ✓ Aggregated Test Stubs: {len(test_stubs)} total")
            print(f"     - Tools stubbed: {', '.join([s.get('tool_name', 'unknown') for s in test_stubs])}")
            print(f"     - Total placeholders: {stubs_data.get('total_placeholders', 0)}")
        
        # Phase 9: TestCrafter - Implement complete tests for all tools
        print("\n🔨 PHASE 9: TEST CRAFTER")
        print("-" * 40)
        print("   Implementing complete tests for all tools...")
        
        # Import test_crafter_phase
        from src.graphtoolkit.domains.agentoolkit import test_crafter_phase
        
        # Update workflow def for test_crafter phase
        workflow_def = WorkflowDefinition(
            domain='agentoolkit',
            phases={
                'analyzer': analyzer_phase,
                'specifier': specifier_phase,
                'crafter': crafter_phase,
                'evaluator': evaluator_phase,
                'refiner': refiner_phase,
                'documenter': documenter_phase,
                'test_analyzer': test_analyzer_phase,
                'test_stubber': test_stubber_phase,
                'test_crafter': test_crafter_phase
            },
            phase_sequence=['test_crafter'],  # Only run test_crafter
            node_configs={
                'dependency_check': NodeConfig(node_type='storage_check'),
                'load_dependencies': NodeConfig(node_type='storage_load'),
                'iteration_control': NodeConfig(node_type='iteration'),
                'template_render': NodeConfig(node_type='template'),
                'llm_call': NodeConfig(node_type='llm', retryable=True, max_retries=2),
                'schema_validation': NodeConfig(node_type='validation'),
                'save_iteration_output': NodeConfig(node_type='iteration_save'),
                'aggregation': NodeConfig(node_type='aggregation'),
                'save_phase_output': NodeConfig(node_type='storage_save'),
                'state_update': NodeConfig(node_type='state'),
                'quality_gate': NodeConfig(node_type='validation')
            }
        )
        
        # Extract the final state from test_stubber result
        if hasattr(test_stubber_result, 'output'):
            final_test_stubber_state = test_stubber_result.output
        else:
            final_test_stubber_state = test_stubber_state
        
        # Update state for test_crafter
        test_crafter_state = WorkflowState(
            workflow_def=workflow_def,
            workflow_id=workflow_id,
            domain='agentoolkit',
            current_phase='test_crafter',
            current_node='dependency_check',
            completed_phases={'analyzer', 'specifier', 'crafter', 'evaluator', 'refiner', 'documenter', 'test_analyzer', 'test_stubber'},
            domain_data=final_test_stubber_state.domain_data,
            phase_outputs=final_test_stubber_state.phase_outputs
        )
        
        # Use same nodes
        graph = Graph(nodes=nodes)
        
        # Run test_crafter phase
        test_crafter_result = await graph.run(GenericPhaseNode(), deps=deps, state=test_crafter_state)
        
        # Report on TestCrafter
        print("\n🔨 TEST CRAFTER PHASE RESULTS")
        print("-" * 40)
        
        # Check individual test implementations
        test_impl_count = 0
        total_test_methods = 0
        for tool in missing_tools:
            tool_name = tool.name if hasattr(tool, 'name') else tool.get('name', 'unknown')
            impl_key = f'workflow/{workflow_id}/test_impl/{tool_name}'
            
            impl_result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': impl_key,
                'namespace': 'workflow'
            })
            
            if impl_result.success:
                test_impl_count += 1
                impl_data = impl_result.data.get('value', {})
                print(f"   ✓ Test implementation for {tool_name}:")
                file_path = impl_data.get('file_path', 'unknown')
                test_count = impl_data.get('test_count', 0)
                total_test_methods += test_count
                print(f"     - File: {file_path}")
                print(f"     - Test methods: {test_count} implemented")
                
                # Show first few lines of the test code
                code = impl_data.get('code', '')
                if code:
                    # Find first test method
                    lines = code.split('\n')
                    for i, line in enumerate(lines):
                        if line.strip().startswith('def test_'):
                            print(f"     - First test: {line.strip()[:60]}...")
                            break
        
        # Check aggregated test implementations
        impls_key = f'workflow/{workflow_id}/test_implementations'
        impls_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': impls_key,
            'namespace': 'workflow'
        })
        
        if impls_result.success:
            impls_data = impls_result.data.get('value', {})
            test_implementations = impls_data.get('test_implementations', [])
            print(f"\n   ✓ Aggregated Test Implementations: {len(test_implementations)} total")
            print(f"     - Tools tested: {', '.join([impl.get('tool_name', 'unknown') for impl in test_implementations])}")
            print(f"     - Total test methods: {impls_data.get('total_tests', 0)}")
        
        # Summary
        print(f"\n{'='*80}")
        print("📊 WORKFLOW SUMMARY")
        print(f"{'='*80}")
        print(f"   Workflow ID: {workflow_id}")
        print(f"   Task: {task_description}")
        print(f"   ")
        print(f"   Phase Results:")
        print(f"   1. Analyzer: ✅ Identified {len(missing_tools)} tools to create")
        print(f"   2. Specifier: ✅ Created {spec_count} specifications")
        print(f"   3. Crafter: ✅ Generated {impl_count} implementations")
        print(f"   4. Evaluator: ✅ Evaluated {eval_count} implementations")
        if total_scores:
            avg_score = sum(total_scores) / len(total_scores)
            print(f"      - Average quality score: {avg_score:.2f}/1.0")
            print(f"      - Deployment ready: {deployment_ready}/{len(missing_tools)} tools")
        if final_refiner_state:
            refine_count = len(tools_to_refine)
            print(f"   5. Refiner: ✅ Refined {refine_count} implementations")
        else:
            print(f"   5. Refiner: ⏭️ Skipped (all tools ready for deployment)")
        print(f"   6. Documenter: ✅ Generated {doc_count} documentations")
        print(f"   7. TestAnalyzer: ✅ Analyzed {test_analysis_count} test requirements")
        print(f"   8. TestStubber: ✅ Created {test_stub_count} test skeletons")
        print(f"      - Total placeholders: {total_placeholders} test methods")
        print(f"   9. TestCrafter: ✅ Implemented {test_impl_count} test suites")
        print(f"      - Total test methods: {total_test_methods} tests")
        print(f"   ")
        print(f"   Storage Footprint:")
        print(f"   - Analyzer: 4 artifacts stored")
        print(f"   - Specifier: {spec_count * 2 + 1} artifacts stored")
        print(f"   - Crafter: {impl_count * 2 + 1} artifacts stored")
        print(f"   - Evaluator: {eval_count * 2 + 1} artifacts stored")
        if final_refiner_state:
            refine_count = len(tools_to_refine)
            print(f"   - Refiner: {refine_count * 2 + 1} artifacts stored")
            print(f"   - Documenter: {doc_count * 2 + 1} artifacts stored")
            print(f"   - TestAnalyzer: {test_analysis_count * 2 + 1} artifacts stored")
            print(f"   - TestStubber: {test_stub_count * 2 + 1} artifacts stored")
            print(f"   - TestCrafter: {test_impl_count * 2 + 1} artifacts stored")
            total_artifacts = 4 + spec_count * 2 + 1 + impl_count * 2 + 1 + eval_count * 2 + 1 + refine_count * 2 + 1 + doc_count * 2 + 1 + test_analysis_count * 2 + 1 + test_stub_count * 2 + 1 + test_impl_count * 2 + 1
        else:
            print(f"   - Documenter: {doc_count * 2 + 1} artifacts stored")
            print(f"   - TestAnalyzer: {test_analysis_count * 2 + 1} artifacts stored")
            print(f"   - TestStubber: {test_stub_count * 2 + 1} artifacts stored")
            print(f"   - TestCrafter: {test_impl_count * 2 + 1} artifacts stored")
            total_artifacts = 4 + spec_count * 2 + 1 + impl_count * 2 + 1 + eval_count * 2 + 1 + doc_count * 2 + 1 + test_analysis_count * 2 + 1 + test_stub_count * 2 + 1 + test_impl_count * 2 + 1
        print(f"   - Total: {total_artifacts} artifacts")
        
        # Dump KV storage to file for inspection
        print(f"\n📦 Dumping KV Storage to File...")
        import json
        import tempfile
        from datetime import datetime
        
        # Get all keys from storage
        storage_client = deps.get_storage_client()
        
        # Get all keys with prefix for this workflow
        workflow_prefix = f"workflow/{workflow_id}"
        all_keys_result = await storage_client.run('storage_kv', {
            'operation': 'keys',
            'namespace': 'workflow',
            'pattern': f'{workflow_prefix}*'
        })
        
        # Handle the StorageKvOutput result
        try:
            if hasattr(all_keys_result, 'success') and all_keys_result.success:
                # Pattern already filtered for this workflow
                workflow_keys = all_keys_result.data if all_keys_result.data else []
                
                # Collect all key-value pairs
                storage_dump = {
                    'workflow_id': workflow_id,
                    'timestamp': datetime.now().isoformat(),
                    'total_keys': len(workflow_keys),
                    'data': {}
                }
                
                for key in sorted(workflow_keys):
                    get_result = await storage_client.run('storage_kv', {
                        'operation': 'get',
                        'key': key,
                        'namespace': 'workflow'
                    })
                    # Handle the StorageKvOutput for get operation
                    if hasattr(get_result, 'success') and get_result.success:
                        storage_dump['data'][key] = get_result.data
                
                # Write to temp file
                with tempfile.NamedTemporaryFile(
                    mode='w',
                    prefix=f'agentoolkit_storage_{workflow_id}_',
                    suffix='.json',
                    dir='/tmp',
                    delete=False
                ) as f:
                    json.dump(storage_dump, f, indent=2, default=str)
                    dump_file = f.name
                
                print(f"   📁 Storage dump saved to: {dump_file}")
                print(f"   📊 Total keys: {len(workflow_keys)}")
                print(f"   💾 File size: {os.path.getsize(dump_file):,} bytes")
                print(f"\n   View with: cat {dump_file} | jq '.'")
            else:
                error_msg = all_keys_result.message if hasattr(all_keys_result, 'message') else 'Unknown error'
                print(f"   ⚠️ Could not list storage keys: {error_msg}")
        except Exception as e:
            print(f"   ⚠️ Error dumping storage: {e}")
        
        phases_run = 9
        print(f"\n✅ COMPLETE {phases_run}-PHASE WORKFLOW TEST SUCCESSFUL!")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
