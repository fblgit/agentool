# GraphToolkit Meta-Framework

> **State-driven, domain-agnostic workflow execution engine built on pydantic-graph**

The GraphToolkit is a meta-framework that transforms workflow development from writing custom phase-specific code to configuring universal, reusable patterns. It extends the existing AgenTool ecosystem with atomic node decomposition, enabling any domain to define multi-phase workflows through declarative configuration.

## 🚀 Quick Start

```python
from graphtoolkit import create_domain_workflow, PHASE_REGISTRY

# Create an AgenTool workflow 
workflow = create_domain_workflow(
    domain='agentool',
    phases=['analyzer', 'specifier', 'crafter', 'evaluator']
)

# Run with state-driven configuration
result = await workflow.run(
    state=WorkflowState(
        task_description="Create a TODO management system",
        domain="agentool"
    )
)
```

## 📁 Complete File Tree

```
src/graphtoolkit/
├── README.md                          # This comprehensive guide
├── PLAN.md                           # Implementation plan & evolution strategy
├── __init__.py                       # Main exports & public API
│
├── core/                             # 🏗️ Foundation & Registry
│   ├── __init__.py                   # Core exports
│   ├── types.py                      # Universal type definitions (WorkflowState, PhaseDefinition, etc.)
│   ├── registry.py                   # PhaseRegistry & WorkflowDefinition management  
│   ├── factory.py                    # Graph construction from phase definitions
│   └── (uses existing src/agentoolkit/system/templates.py)
│
├── nodes/                            # 🔗 Node Implementations
│   ├── __init__.py                   # Node exports
│   ├── base.py                       # BaseNode patterns & atomic node base classes
│   ├── generic.py                    # GenericPhaseNode - orchestrates atomic chains
│   │
│   ├── atomic/                       # ⚛️ Atomic Node Implementations
│   │   ├── __init__.py               # Atomic node exports
│   │   ├── storage.py                # LoadDependenciesNode, SavePhaseOutputNode
│   │   ├── templates.py              # TemplateRenderNode (deterministic, cacheable)
│   │   ├── llm.py                    # LLMCallNode (expensive, non-retryable at node level)  
│   │   ├── validation.py             # SchemaValidationNode, QualityGateNode
│   │   └── control.py                # StateUpdateNode, NextPhaseNode, ConditionalNode
│   │
│   └── iteration/                    # 🔄 Iteration & Parallel Processing
│       ├── __init__.py               # Iteration exports
│       └── iterables.py              # IterableNode with self-return pattern
│
├── state/                            # 🔄 State Management
│   ├── __init__.py                   # State exports
│   ├── mutations.py                  # Immutable state mutation patterns
│   └── persistence.py               # State recovery & checkpointing
│
├── domains/                          # 🌐 Domain-Specific Configurations
│   ├── __init__.py                   # Domain exports
│   ├── agentool.py                   # AgenTool workflow phases (analyzer, specifier, crafter, evaluator)
│   ├── api.py                        # API design workflow phases (analyzer, designer, generator, validator)  
│   └── workflow.py                   # Workflow orchestration phases (analyzer, orchestrator, validator)
│
└── (uses existing src/templates/ directory structure)
    ├── system/                       # System prompt templates by domain
    │   ├── agentool/                 # AgenTool system prompts
    │   │   ├── analyzer.jinja        # Analysis system prompt
    │   │   ├── specifier.jinja       # Specification system prompt  
    │   │   ├── crafter.jinja         # Code generation system prompt
    │   │   └── evaluator.jinja       # Evaluation system prompt
    │   ├── api/                      # API design system prompts
    │   │   ├── analyzer.jinja        # API analysis system prompt
    │   │   ├── designer.jinja        # API design system prompt
    │   │   └── generator.jinja       # API generation system prompt
    │   └── workflow/                 # Workflow orchestration system prompts
    │       ├── analyzer.jinja        # Process analysis system prompt
    │       └── orchestrator.jinja    # Orchestration system prompt
    │
    ├── prompts/                      # User prompt templates by domain  
    │   ├── agentool/                 # AgenTool user prompts
    │   │   ├── analyze_catalog.jinja # Catalog analysis prompt
    │   │   ├── create_specification.jinja # Specification creation prompt
    │   │   ├── craft_implementation.jinja # Code generation prompt  
    │   │   └── evaluate_code.jinja   # Code evaluation prompt
    │   ├── api/                      # API design user prompts
    │   │   ├── analyze_requirements.jinja # API requirements analysis
    │   │   ├── design_schema.jinja   # API schema design
    │   │   └── generate_implementation.jinja # API implementation
    │   └── workflow/                 # Workflow orchestration user prompts
    │       ├── analyze_process.jinja # Process analysis
    │       └── design_orchestration.jinja # Orchestration design
    │
    └── fragments/                    # Reusable template components
        ├── error_handling.jinja      # Common error handling patterns
        ├── quality_criteria.jinja    # Quality assessment criteria
        ├── refinement_feedback.jinja # Refinement feedback patterns
        └── context_setup.jinja       # Common context setup
```

## 📚 Reference Documentation  

### 🔴 Mandatory Reading (Critical for Understanding)

**Before working with GraphToolkit, you MUST read these documents:**

1. **[../docs/workflow-graph-system.md](../docs/workflow-graph-system.md)**
   - Master architecture overview & design philosophy
   - State-driven execution patterns
   - Universal phase flow patterns

2. **[../docs/GRAPH_TYPE_DEFINITIONS.md](../docs/GRAPH_TYPE_DEFINITIONS.md)**  
   - Canonical type definitions (WorkflowDefinition, PhaseDefinition, etc.)
   - Type dependency hierarchy
   - State-based configuration patterns

3. **[../docs/NODE_CATALOG.md](../docs/NODE_CATALOG.md)**
   - Complete atomic node specifications
   - Node chaining patterns  
   - Error handling strategies by node type

4. **[../docs/DATA_FLOW_REQUIREMENTS.md](../docs/DATA_FLOW_REQUIREMENTS.md)**
   - Universal data flow patterns
   - State transformation requirements
   - Storage reference patterns

5. **[../docs/STATE_MUTATIONS.md](../docs/STATE_MUTATIONS.md)**
   - Immutable state mutation patterns
   - Atomic node state responsibilities
   - Error state handling

6. **[../docs/GRAPH_ARCHITECTURE.md](../docs/GRAPH_ARCHITECTURE.md)**
   - Detailed architectural patterns
   - State-driven configuration examples
   - Integration with pydantic_graph

### 📋 Supporting Documentation

7. **[../docs/DIAGRAMS.md](../docs/DIAGRAMS.md)** - Visual architecture diagrams
8. **[../CLAUDE.md](../CLAUDE.md)** - Project instructions, testing commands, architecture overview

### 🔗 Existing Infrastructure (Build Upon)

**Template System (✅ Available):**
- `src/agentoolkit/system/templates.py` - Complete Jinja2 template system with storage integration
- `src/templates/` - Existing template directory with system/prompts/skeletons
- `docs/agentoolkits/templates.md` - Full documentation

**Storage Systems (✅ Available):**
- `src/agentoolkit/storage/kv.py` - Key-value storage with memory/redis/dynamodb backends  
- `src/agentoolkit/storage/fs.py` - File system storage with local/s3/gcs backends
- `docs/agentoolkits/storage-kv.md`, `docs/agentoolkits/storage-fs.md` - Documentation

**Core Framework (✅ Available):**
- `src/agentool/` - Complete AgenTool framework (factory, registry, injector, model)
- `src/agentool/core/registry.py` - AgenTool registry system
- `src/agentool/core/injector.py` - Dependency injection system
- `src/agentool/factory.py` - AgenTool creation factory

**Workflow Components (✅ Available):**
- `src/agentoolkit/workflows/` - All existing workflow AgenTools:
  - `workflow_analyzer.py` - Analysis workflow
  - `workflow_specifier.py` - Specification workflow  
  - `workflow_crafter.py` - Code generation workflow
  - `workflow_evaluator.py` - Evaluation workflow
  - `workflow_test_*.py` - Test generation workflows

**Other Infrastructure (✅ Available):**
- `src/agentoolkit/observability/metrics.py` - Metrics collection
- `src/agentoolkit/system/logging.py` - Logging system
- `src/agentoolkit/auth/` - Authentication system
- `src/agentoolkit/network/http.py` - HTTP client

**Current V1 System (Backward Compatibility Required):**
- `src/agents/workflow.py` - Current workflow implementation using injector pattern

## 🏗️ Architecture Overview

### Universal Execution Pattern

**Every domain follows the same atomic execution pattern:**

```
InputSchema → Variables → TemplateRender → LLM Call → OutputSchema → Storage
```

### From Monolithic to Atomic

**Before (V1 - Current System):**
```python
# Each phase is a monolithic node (~500 lines each)
class AnalyzerNode(BaseNode[WorkflowState]):
    async def run(self, ctx) -> SpecificationNode:
        # Load dependencies (50 lines)
        # Render templates (50 lines)  
        # Call LLM (100 lines)
        # Validate output (75 lines)
        # Save results (100 lines)
        # Handle errors (125 lines)
        return SpecificationNode()
```

**After (V2 - Meta-Framework):**
```python  
# GenericPhaseNode orchestrates atomic nodes (~100 lines total)
class GenericPhaseNode(BaseNode[WorkflowState, WorkflowDeps, WorkflowState]):
    async def run(self, ctx) -> BaseNode:
        # Read phase definition from state
        phase_def = ctx.state.workflow_def.phases[ctx.state.current_phase]
        # Return first atomic node - it chains to the rest  
        return create_node_instance(phase_def.atomic_nodes[0])

# Each atomic node is focused and reusable (~50 lines each)
# DependencyCheck → LoadDependencies → TemplateRender → 
# LLMCall → SchemaValidation → SaveOutput → StateUpdate → QualityGate
```

### Key Benefits

- **95% Code Reduction**: From 2000+ lines per domain to 600 lines total
- **Atomic Resilience**: Storage failures don't invalidate expensive LLM calls
- **Universal Reuse**: Same nodes work across all domains
- **Fine-grained Observability**: Track success/failure per atomic operation  
- **Selective Retry**: Only retry failed operations, not entire phases

## 🔧 Core Components

### State-Driven Configuration

```python
# Everything is defined in state, not code
workflow_def = WorkflowDefinition(
    domain="agentool",
    phases={
        "analyzer": PhaseDefinition(
            phase_name="analyzer",
            atomic_nodes=["dependency_check", "load_dependencies", "template_render", 
                         "llm_call", "schema_validation", "save_output", "state_update", "quality_gate"],
            input_schema=AnalyzerInput,
            output_schema=AnalyzerOutput,
            templates=TemplateConfig(
                system_template="templates/system/agentool/analyzer.jinja",
                user_template="templates/prompts/agentool/analyze_catalog.jinja"
            ),
            storage_pattern="workflow/{workflow_id}/analysis",
            quality_threshold=0.8
        )
    },
    node_configs={
        "llm_call": NodeConfig(retryable=True, max_retries=3),
        "save_output": NodeConfig(retryable=False, max_retries=0)
    }
)
```

### Atomic Node Chaining

```python
# Nodes chain by returning next node (no sub-graphs)
class LoadDependenciesNode(BaseNode[WorkflowState, WorkflowDeps, WorkflowState]):
    async def run(self, ctx) -> BaseNode:
        # Get retry configuration from state
        config = ctx.state.workflow_def.node_configs["load_dependencies"]
        
        try:
            # Load dependencies from storage
            dependencies = await self.load_from_storage(ctx)
            
            # Update state with loaded data
            new_state = replace(ctx.state, domain_data={
                **ctx.state.domain_data,
                'loaded_dependencies': dependencies
            })
            
            # Chain to next atomic node
            return TemplateRenderNode()
            
        except StorageError as e:
            if config.retryable and self.should_retry(ctx.state):
                return LoadDependenciesNode()  # Retry by returning self
            else:
                return ErrorNode(error=str(e))
```

### Universal State Pattern

```python
@dataclass(frozen=True)  # Immutable
class WorkflowState:
    # Universal fields work for any domain
    workflow_id: str
    domain: str  # 'agentool', 'api', 'workflow', etc.
    
    # Configuration-driven execution
    workflow_def: WorkflowDefinition
    
    # Phase tracking  
    completed_phases: Set[str]
    current_phase: str
    
    # Storage references (not data itself)
    phase_outputs: Dict[str, StorageRef]
    
    # Domain-flexible data
    domain_data: Dict[str, Any]
    
    # Quality & refinement tracking
    quality_scores: Dict[str, float] 
    refinement_count: Dict[str, int]
```

## 🌐 Domain Examples

### AgenTool Domain (Default)

```python
# Complete AgenTool workflow definition
AGENTOOL_WORKFLOW = WorkflowDefinition(
    domain="agentool",
    phases={
        "analyzer": PhaseDefinition(
            atomic_nodes=["dependency_check", "load_dependencies", "template_render", "llm_call", "schema_validation", "save_output", "state_update", "quality_gate"],
            input_schema=AnalyzerInput,
            output_schema=AnalyzerOutput,
            dependencies=[],
            templates=TemplateConfig(
                system_template="templates/system/agentool/analyzer.jinja",
                user_template="templates/prompts/agentool/analyze_catalog.jinja"
            )
        ),
        "specifier": PhaseDefinition(
            dependencies=["analyzer"],
            # ... similar structure
        ),
        "crafter": PhaseDefinition(
            dependencies=["specifier"],  
            # ... similar structure
        ),
        "evaluator": PhaseDefinition(
            dependencies=["crafter"],
            # ... similar structure
        )
    },
    phase_sequence=["analyzer", "specifier", "crafter", "evaluator"]
)
```

### API Design Domain

```python
API_WORKFLOW = WorkflowDefinition(
    domain="api",
    phases={
        "analyzer": PhaseDefinition(
            templates=TemplateConfig(
                system_template="templates/system/api/analyzer.jinja",
                user_template="templates/prompts/api/analyze_requirements.jinja"
            ),
            # ... same atomic nodes, different templates & schemas
        ),
        "designer": PhaseDefinition(
            dependencies=["analyzer"],
            # ... API-specific configuration
        ),
        "generator": PhaseDefinition(
            dependencies=["designer"],
            # ... API generation configuration
        )
    },
    phase_sequence=["analyzer", "designer", "generator"]
)
```

### Cross-Domain Workflows

```python
# Mix phases from different domains
HYBRID_WORKFLOW = WorkflowDefinition(
    domain="hybrid",
    phases={
        "agentool_analysis": PHASE_REGISTRY["agentool.analyzer"],
        "api_design": PHASE_REGISTRY["api.designer"], 
        "workflow_orchestration": PHASE_REGISTRY["workflow.orchestrator"]
    },
    phase_sequence=["agentool_analysis", "api_design", "workflow_orchestration"]
)
```

## 🔄 V1 System Evolution & Compatibility

### Current V1 System Support

The GraphToolkit **extends** rather than replaces the existing system. All V1 workflows continue to work unchanged:

**V1 Workflow Nodes (Fully Supported):**
- `src/agents/workflow.py`:
  - `AnalyzerNode` → Uses `workflow_analyzer` AgenTool
  - `SpecificationNode` → Uses `workflow_specifier` AgenTool  
  - `CrafterNode` → Uses `workflow_crafter` AgenTool
  - `EvaluatorNode` → Uses `workflow_evaluator` AgenTool
  - `TestAnalyzerNode`, `TestStubberNode`, `TestCrafterNode`

**V1 AgenTool Integration (Preserved):**
- `src/agentoolkit/workflows/`:
  - `workflow_analyzer.py`, `workflow_specifier.py`, `workflow_crafter.py`, `workflow_evaluator.py`
  - `workflow_test_analyzer.py`, `workflow_test_stubber.py`, `workflow_test_crafter.py`

**V1 Templates (Reused):**
- `src/templates/system/` - System prompt templates
- `src/templates/prompts/` - User prompt templates  
- `src/templates/skeletons/` - Code skeleton templates

### Migration Strategy

**Phase 1: Parallel Operation**
```python
# V1 workflows continue unchanged
result = await run_agentool_generation_workflow(
    task_description="Create auth system",
    model="openai:gpt-4o"
)

# V2 workflows available for new use cases
result = await run_meta_workflow(
    domain="agentool", 
    phases=["analyzer", "specifier", "crafter", "evaluator"],
    state=WorkflowState(task_description="Create auth system")
)
```

**Phase 2: Gradual Feature Adoption**
- New domains (API, documentation) use V2 only
- Existing AgenTool workflows can opt-in to V2 benefits
- Template system unified across V1 and V2

**Phase 3: Full Integration (Future)**
- V1 nodes become thin wrappers over V2 atomic nodes
- Complete migration when all features proven stable
- Single codebase with full backward compatibility

### Template Evolution

**Current Templates (V1):**
```
src/templates/
├── system/
│   ├── analyzer.jinja       # → templates/system/agentool/analyzer.jinja
│   ├── specification.jinja  # → templates/system/agentool/specifier.jinja
│   └── crafter.jinja       # → templates/system/agentool/crafter.jinja
└── prompts/
    ├── analyze_catalog.jinja     # → templates/prompts/agentool/analyze_catalog.jinja
    └── create_specification.jinja # → templates/prompts/agentool/create_specification.jinja
```

**Extended Templates (V2):**
```
src/graphtoolkit/templates/
├── system/
│   ├── agentool/           # Existing templates + domain organization
│   ├── api/               # New API design domain  
│   └── workflow/          # New workflow orchestration domain
├── prompts/
│   ├── agentool/          # Existing prompts + domain organization
│   ├── api/               # New API design prompts
│   └── workflow/          # New workflow prompts  
└── fragments/             # New: reusable components
    ├── error_handling.jinja
    ├── quality_criteria.jinja  
    └── refinement_feedback.jinja
```

## 🧪 Testing & Development

### Testing Commands (from CLAUDE.md)

```bash
# Run all tests
pytest

# Run specific test categories  
pytest tests/graphtoolkit/  # GraphToolkit-specific tests
pytest tests/test_agentool.py  # Core AgenTool tests
pytest tests/test_agent_integration.py  # Integration tests

# Run with coverage
pytest --cov=src --cov-report=html

# Run async tests
pytest -v tests/test_agentool_async_sync.py
```

### Development Setup

```bash
# Install development dependencies
pip install -e .

# Initialize AgenTools (required for workflows)
python -c "import agentoolkit; agentoolkit.initialize_all()"
```

### V1 Compatibility Testing

```python
# Ensure V1 workflows still work
async def test_v1_compatibility():
    # Test existing workflow execution
    result = await run_agentool_generation_workflow(
        task_description="Create test AgenTool",
        model="openai:gpt-4o"
    )
    assert result['success'] == True
    
    # Test that all V1 storage patterns work  
    assert 'workflow_id' in result
    assert 'final_code' in result
```

## 🚀 Quick Start Examples

### Basic AgenTool Workflow

```python
from graphtoolkit import create_domain_workflow, WorkflowState

# Create workflow
workflow = create_domain_workflow(
    domain='agentool',
    phases=['analyzer', 'specifier', 'crafter', 'evaluator']
)

# Execute with configuration
result = await workflow.run(
    state=WorkflowState(
        task_description="Create a user authentication system",
        domain="agentool",
        workflow_def=AGENTOOL_WORKFLOW
    )
)

print(f"Generated code: {result.phase_outputs['crafter']}")
```

### Custom Domain Workflow  

```python
from graphtoolkit import PhaseRegistry, PhaseDefinition

# Register new domain phases
PhaseRegistry.register_domain("documentation", {
    "analyzer": PhaseDefinition(
        phase_name="analyzer",
        atomic_nodes=["dependency_check", "load_dependencies", "template_render", "llm_call", "schema_validation", "save_output"],
        input_schema=DocAnalyzerInput,
        output_schema=DocAnalyzerOutput,
        templates=TemplateConfig(
            system_template="templates/system/documentation/analyzer.jinja",
            user_template="templates/prompts/documentation/analyze_content.jinja"
        )
    ),
    "writer": PhaseDefinition(
        dependencies=["analyzer"],
        # ... writer configuration
    )
})

# Use new domain
workflow = create_domain_workflow(
    domain='documentation', 
    phases=['analyzer', 'writer']
)
```

### State-Driven Conditional Logic

```python
# Configure conditional branching in workflow definition
workflow_def = WorkflowDefinition(
    domain="agentool",
    phases=standard_phases,
    conditions={
        "complexity_check": ConditionConfig(
            condition_type="state_path",
            state_path="domain_data.complexity",
            operator="==",
            expected_value="high"
        )
    }
)

# Nodes use state-driven conditions
class GeneratorRoutingNode(BaseNode):
    async def run(self, ctx) -> BaseNode:
        complexity_condition = ctx.state.workflow_def.conditions["complexity_check"]
        
        if complexity_condition.evaluate(ctx.state):
            return AdvancedGeneratorNode()
        else:
            return SimpleGeneratorNode()
```

## 📝 Contributing & Style Guide

### Code Style Considerations

**From CLAUDE.md project instructions:**
- Write clear, concise code without unnecessary comments
- Follow existing patterns in `src/agentool/` and `src/agentoolkit/`
- Maintain compatibility with pydantic-ai and pydantic-graph
- Use dataclasses with `frozen=True` for immutable state
- Follow atomic node patterns from NODE_CATALOG.md

### Adding New Domains

1. **Create domain phase definitions** in `src/graphtoolkit/domains/{domain}.py`
2. **Add templates** in `src/graphtoolkit/templates/{system,prompts}/{domain}/`
3. **Register with PhaseRegistry** in domain module
4. **Add tests** in `tests/graphtoolkit/domains/`
5. **Update documentation** in this README.md

### Adding New Atomic Nodes

1. **Follow specifications** in docs/NODE_CATALOG.md
2. **Implement state-based retry** if applicable
3. **Add to appropriate category** in `src/graphtoolkit/nodes/atomic/`
4. **Register with node factory** in `src/graphtoolkit/core/factory.py`
5. **Add comprehensive tests**

## 🔗 Integration Points

### With Existing Systems

- **AgenTool Core** (`src/agentool/`): Builds upon factory patterns and registry system
- **AgenToolkits** (`src/agentoolkit/`): Reuses existing tools and maintains injector integration
- **Templates** (`src/templates/`): Extends existing template structure with domain organization  
- **Storage Systems**: Compatible with existing storage_kv and storage_fs patterns

### With pydantic-ai & pydantic-graph

- **Type Safety**: Full integration with pydantic model validation
- **Graph Execution**: Built on pydantic-graph BaseNode patterns
- **State Persistence**: Compatible with pydantic-graph persistence mechanisms
- **Dependency Injection**: Follows pydantic-graph GraphDeps patterns

## 📈 Roadmap

### Current Phase: Foundation (V2.0)
- ✅ Type definitions and registry system
- ✅ Atomic node architecture 
- ✅ AgenTool domain implementation
- ✅ Template system integration
- ✅ V1 compatibility layer

### Next Phase: Domain Expansion (V2.1)
- 🔄 API design domain workflows
- 🔄 Documentation generation domain  
- 🔄 Cross-domain workflow patterns
- 🔄 Enhanced error handling and recovery

### Future Phase: Advanced Features (V2.2+)
- 🔮 Visual workflow editor
- 🔮 Real-time workflow monitoring  
- 🔮 Workflow version control and rollback
- 🔮 Advanced parallel execution patterns

## 🆘 Getting Help

### Quick Reference
- **Architecture Questions**: Read docs/workflow-graph-system.md
- **Type Issues**: Check docs/GRAPH_TYPE_DEFINITIONS.md  
- **Node Implementation**: Reference docs/NODE_CATALOG.md
- **State Management**: Review docs/STATE_MUTATIONS.md

### Common Issues

**Q: V1 workflow not working?**  
A: V1 workflows should work unchanged. Check that workflow AgenTools are properly registered.

**Q: How to add a new domain?**
A: Create PhaseDefinitions in `domains/{domain}.py`, add templates, register with PhaseRegistry.

**Q: State mutations not working?**
A: Ensure you're using `replace()` for immutable updates, not direct assignment.

**Q: Atomic nodes not chaining?**
A: Verify each node returns the next node instance, not just node name.

---

**The GraphToolkit represents the evolution of workflow orchestration from hardcoded phases to universal, configurable patterns. It maintains full backward compatibility while enabling unprecedented flexibility and reusability across domains.**