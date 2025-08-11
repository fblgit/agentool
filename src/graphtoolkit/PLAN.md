# GraphToolkit Meta-Framework Implementation Plan

## Overview

The GraphToolkit is a **state-driven meta-framework** that evolves the existing AgenTool workflow system from monolithic phase-specific nodes to a universal, domain-agnostic architecture. This represents a natural evolution from the current `src/agents/workflow.py` system to a more flexible, reusable, and maintainable framework.

## Design Philosophy

### From Monolithic to Atomic
**Before (V1 - Current System):**
```python
# Each phase is a standalone node with 500+ lines
class AnalyzerNode(BaseNode[WorkflowState]):
    async def run(self, ctx) -> SpecificationNode:
        # Load dependencies
        # Render templates  
        # Call LLM
        # Validate output
        # Save results
        # All in one monolithic operation
        return SpecificationNode()
```

**After (V2 - Meta-Framework):**
```python
# GenericPhaseNode orchestrates atomic nodes
class GenericPhaseNode(BaseNode[WorkflowState, WorkflowDeps, WorkflowState]):
    async def run(self, ctx) -> BaseNode:
        # Read phase definition from state
        phase_def = ctx.state.workflow_def.phases[ctx.state.current_phase]
        # Return first atomic node - it chains to the rest
        return create_node_instance(phase_def.atomic_nodes[0])
```

### Universal Pattern
**All domains follow the same execution pattern:**
`InputSchema → Variables → TemplateRender → LLM Call → OutputSchema → Storage`

This pattern works for:
- **AgenTool workflows**: Analyzer → Specifier → Crafter → Evaluator
- **API design workflows**: API Analyzer → Schema Designer → Implementation → Validator  
- **Documentation workflows**: Content Analyzer → Structure Designer → Writer → Reviewer
- **Any custom domain**: Define phases as configuration, not code

## Reference Architecture

### Mandatory Reading Material

**Primary Documentation (MUST READ):**
1. [docs/workflow-graph-system.md](../docs/workflow-graph-system.md) - Master architecture overview
2. [docs/GRAPH_TYPE_DEFINITIONS.md](../docs/GRAPH_TYPE_DEFINITIONS.md) - Canonical type definitions
3. [docs/NODE_CATALOG.md](../docs/NODE_CATALOG.md) - Complete node specifications  
4. [docs/DATA_FLOW_REQUIREMENTS.md](../docs/DATA_FLOW_REQUIREMENTS.md) - Data flow patterns
5. [docs/STATE_MUTATIONS.md](../docs/STATE_MUTATIONS.md) - Immutable state patterns
6. [docs/GRAPH_ARCHITECTURE.md](../docs/GRAPH_ARCHITECTURE.md) - Architectural details

**Supporting Documentation:**
7. [docs/DIAGRAMS.md](../docs/DIAGRAMS.md) - Visual architecture diagrams
8. [CLAUDE.md](../CLAUDE.md) - Project instructions and testing commands

### Current V1 System Analysis

**Existing Workflow System (`src/agents/workflow.py`):**
- **Nodes**: `AnalyzerNode`, `SpecificationNode`, `CrafterNode`, `EvaluatorNode`
- **Test Nodes**: `TestAnalyzerNode`, `TestStubberNode`, `TestCrafterNode`
- **State Management**: `WorkflowState` with phase completion flags
- **Integration**: Uses existing AgenToolkit workflows via injector system

**Key V1 Characteristics We Must Support:**
```python
# V1 State Structure (backward compatibility required)
@dataclass
class WorkflowState:
    task_description: str
    model: str = "openai:gpt-4o"
    generate_tests: bool = True
    phase_models: Dict[str, str] = field(default_factory=dict)
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Phase completion flags (must continue to work)
    analyzer_completed: bool = False
    specification_completed: bool = False
    crafter_completed: bool = False
    evaluator_completed: bool = False
```

**V1 AgenTool Integration Pattern:**
```python
# Current pattern using injector
injector = get_injector()
result = await injector.run('workflow_analyzer', {
    'operation': 'analyze',
    'task_description': ctx.state.task_description,
    'workflow_id': ctx.state.workflow_id,
    'model': phase_model
})
```

## Implementation Strategy

### Phase 1: Foundation Types
**Files created (✅ Complete):**
- `src/graphtoolkit/core/types.py` - All type definitions from GRAPH_TYPE_DEFINITIONS.md
- `src/graphtoolkit/core/registry.py` - PhaseRegistry system

**Existing Infrastructure Used:**
- Template system: `src/agentoolkit/system/templates.py` (no new file needed)
- Storage systems: `src/agentoolkit/storage/{kv,fs}.py`
- Core framework: `src/agentool/` (factory, registry, injector)

**Key types to implement:**
- `WorkflowDefinition` - Complete workflow in state
- `PhaseDefinition` - Single phase configuration  
- `NodeConfig` - Node behavior configuration
- `ConditionConfig` - State-driven conditional logic
- `WorkflowState` - Universal immutable state
- `WorkflowDeps` - Dependency injection

### Phase 2: Atomic Node Architecture  
**Files to create (🔄 In Progress):**
- `src/graphtoolkit/nodes/base.py` - Base atomic node patterns
- `src/graphtoolkit/nodes/generic.py` - GenericPhaseNode orchestrator  
- `src/graphtoolkit/nodes/atomic/` - All atomic node implementations

**Integration with Existing Systems:**
- Use `src/agentool/core/injector.py` for dependency injection
- Integrate with `src/agentoolkit/storage/` for LoadDependenciesNode/SavePhaseOutputNode
- Use `src/agentoolkit/system/templates.py` for TemplateRenderNode

**Atomic Node Categories:**
- **Storage Nodes**: `LoadDependenciesNode`, `SavePhaseOutputNode`
- **LLM Nodes**: `TemplateRenderNode`, `LLMCallNode` 
- **Validation Nodes**: `SchemaValidationNode`, `QualityGateNode`
- **Control Nodes**: `StateUpdateNode`, `NextPhaseNode`, `ConditionalNode`
- **Iteration Nodes**: `IterableNode` with self-return pattern

### Phase 3: Domain Definitions
**Files to create:**
- `src/graphtoolkit/domains/agentool.py` - AgenTool workflow phases
- `src/graphtoolkit/domains/api.py` - API design workflow phases  
- `src/graphtoolkit/domains/workflow.py` - Workflow orchestration phases

**Building Upon Existing Workflows:**
- Analyze existing `src/agentoolkit/workflows/workflow_*.py` patterns
- Extract common patterns into meta-framework PhaseDefinitions
- Maintain compatibility with existing `src/agents/workflow.py`

**AgenTool Domain Phases:**
```python
AGENTOOL_PHASES = {
    'agentool.analyzer': PhaseDefinition(
        phase_name='analyzer',
        atomic_nodes=['dependency_check', 'load_dependencies', 'template_render', 'llm_call', 'schema_validation', 'save_output', 'state_update', 'quality_gate'],
        input_schema=AnalyzerInput,
        output_schema=AnalyzerOutput,
        system_template='templates/system/agentool/analyzer.jinja',
        user_template='templates/prompts/agentool/analyze_catalog.jinja',
        storage_pattern='workflow/{workflow_id}/analysis'
    ),
    'agentool.specifier': PhaseDefinition(...),
    'agentool.crafter': PhaseDefinition(...),
    'agentool.evaluator': PhaseDefinition(...)
}
```

### Phase 4: Migration & Integration
**Evolution Strategy:**
1. **Parallel Operation**: V1 and V2 systems work side-by-side
2. **Gradual Migration**: Existing workflows continue unchanged
3. **Opt-in Upgrade**: New workflows can use meta-framework  
4. **Template Reuse**: Existing templates work with new system
5. **Storage Compatibility**: Same storage patterns and keys

## Backward Compatibility Requirements

### V1 Workflow Support
The existing `src/agents/workflow.py` system must continue to work unchanged:

```python
# This must continue to work exactly as before
result = await run_agentool_generation_workflow(
    task_description=task,
    model="openai:gpt-4o",
    generate_tests=True
)
```

### V1 State Structure
All existing state fields and patterns must be preserved:
- Phase completion flags (`analyzer_completed`, etc.)  
- Workflow metadata structure
- Storage reference patterns
- Error tracking and logging
- Token usage tracking

### V1 AgenTool Integration
The injector pattern and existing workflow AgenTools must work unchanged:
- `workflow_analyzer`, `workflow_specifier`, `workflow_crafter`, `workflow_evaluator`
- `workflow_test_analyzer`, `workflow_test_stubber`, `workflow_test_crafter`  
- Storage keys and artifact patterns
- Logging and monitoring integration

## Success Metrics

### Technical Metrics
- **Code Reduction**: 95% less duplicated code across domains
- **Reusability**: Same atomic nodes work for all domains
- **Error Resilience**: Storage failures don't invalidate LLM responses
- **Observability**: Fine-grained execution tracking per atomic operation

### Compatibility Metrics  
- **V1 Workflows**: 100% backward compatibility
- **Template Reuse**: All existing templates work with V2
- **Storage Patterns**: Same storage keys and structures
- **Performance**: No regression in execution time

### Framework Metrics
- **Domain Support**: AgenTool, API, Workflow domains implemented
- **Extension**: Easy to add new domains via configuration
- **Type Safety**: Full type checking through all workflows
- **Testing**: Comprehensive test coverage for atomic operations

## Implementation Timeline

### Week 1: Foundation & Types
- Core type definitions
- PhaseRegistry system
- Template engine
- Basic atomic node structure

### Week 2: Atomic Nodes & State
- All atomic node implementations
- State mutation patterns  
- Error handling and retry logic
- GenericPhaseNode orchestrator

### Week 3: Domain Integration
- AgenTool domain definitions
- Template structure migration
- V1 compatibility layer
- Integration testing

### Week 4: Documentation & Polish
- Comprehensive README.md
- Migration guides
- Performance optimization
- Production readiness

## Risk Mitigation

### Compatibility Risks
- **Risk**: Breaking existing workflows
- **Mitigation**: Parallel operation, extensive testing, gradual migration

### Performance Risks  
- **Risk**: Overhead from atomic decomposition
- **Mitigation**: Benchmark against V1, optimize hot paths

### Complexity Risks
- **Risk**: Over-engineering the meta-framework
- **Mitigation**: Focus on current use cases, gradual feature addition

### Adoption Risks
- **Risk**: Team resistance to new patterns  
- **Mitigation**: Clear documentation, migration examples, training

## File Structure Reference

```
src/graphtoolkit/                    # Meta-framework root
├── README.md                        # Complete documentation 
├── PLAN.md                          # This file
├── core/                           # Foundation types & registry
├── nodes/                          # Node implementations  
│   ├── base.py                     # Base atomic node patterns
│   ├── generic.py                  # GenericPhaseNode orchestrator
│   └── atomic/                     # Atomic node implementations
├── state/                          # State management
├── domains/                        # Domain-specific configurations  
└── templates/                      # Template structure
```

## Next Steps

1. **Read mandatory documentation** - Complete understanding of architecture
2. **Create type definitions** - Foundation for all other components  
3. **Implement atomic nodes** - Core execution primitives
4. **Build domain definitions** - AgenTool workflow configuration
5. **Test compatibility** - Ensure V1 workflows continue working
6. **Document thoroughly** - Enable team adoption and contribution

This plan ensures we build a robust, backward-compatible meta-framework that evolves the existing system while maintaining all current functionality.