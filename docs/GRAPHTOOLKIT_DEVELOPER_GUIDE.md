# GraphToolkit Developer Guide

## Table of Contents
1. [Understanding the Core Concepts](#understanding-the-core-concepts)
2. [How Immutable State Works](#how-immutable-state-works)
3. [Data Flow Through the Graph](#data-flow-through-the-graph)
4. [How to Add a New Domain](#how-to-add-a-new-domain)
5. [Node Chaining Pattern](#node-chaining-pattern)
6. [Practical Examples](#practical-examples)

## Understanding the Core Concepts

GraphToolkit is a **meta-framework** where workflows are defined as data (configuration) rather than code. This means:

- **Phases are configurations**, not hardcoded implementations
- **One generic node type** executes all phases by reading configuration from state
- **Atomic nodes chain together** by returning the next node
- **State flows through the graph** accumulating data, never losing it

### Key Architecture Components

```
┌─────────────────────────────────────────────────────┐
│                  WorkflowDefinition                  │
│  (Complete workflow configuration stored in state)   │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│                  GenericPhaseNode                    │
│     (Reads phase config from state, returns          │
│      first atomic node to start the chain)           │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│                Atomic Node Chain                     │
│  DependencyCheck → LoadDeps → Template → LLM →       │
│  Validate → Save → StateUpdate → QualityGate         │
└─────────────────────────────────────────────────────┘
```

## How Immutable State Works

### What is Immutable State?

Immutable state means that once created, a state object **cannot be changed**. Instead of modifying existing state, we create new state objects with updated values. This ensures:

1. **Thread safety** - Multiple nodes can read state without conflicts
2. **Debugging clarity** - Every state transformation is traceable
3. **Rollback capability** - Previous states remain intact
4. **Predictability** - State changes are explicit and controlled

### The Replace Pattern

In Python, we use `dataclasses.replace()` to create new state:

```python
from dataclasses import dataclass, replace

@dataclass(frozen=True)  # frozen=True makes it immutable
class WorkflowState:
    workflow_id: str
    current_phase: str
    completed_phases: Set[str]
    domain_data: Dict[str, Any]

# WRONG - This will raise an error due to frozen=True
state.current_phase = "new_phase"  # ❌ AttributeError!

# RIGHT - Create new state with updated values
new_state = replace(
    state,
    current_phase="new_phase",
    completed_phases=state.completed_phases | {"old_phase"}
)
```

### State Accumulation Pattern

State accumulates data as it flows through nodes:

```python
# Node 1: Load dependencies
new_state = replace(
    state,
    domain_data={
        **state.domain_data,  # Keep existing data
        'dependencies_loaded': dependency_data  # Add new data
    }
)

# Node 2: Render templates (state has dependencies from Node 1)
new_state = replace(
    state,
    domain_data={
        **state.domain_data,  # Keep dependencies + existing
        'rendered_prompts': prompts  # Add prompts
    }
)

# Node 3: LLM call (state has dependencies + prompts)
new_state = replace(
    state,
    domain_data={
        **state.domain_data,  # Keep everything
        'llm_response': response  # Add response
    }
)
```

## Data Flow Through the Graph

### How Nodes Pass Data

Data flows through the graph via **state passing** and **node chaining**:

```python
class AtomicNode(BaseNode):
    async def run(self, ctx: GraphRunContext) -> BaseNode:
        # 1. Read from current state
        input_data = ctx.state.domain_data.get('some_input')
        
        # 2. Process the data
        result = await self.process(input_data)
        
        # 3. Create new state with result
        new_state = replace(
            ctx.state,
            domain_data={
                **ctx.state.domain_data,
                'my_result': result
            }
        )
        
        # 4. Return next node (it will receive new_state)
        return NextNodeInChain()
```

### State as the Universal Container

```
Initial State
    ↓
[Node A] reads state → processes → creates State v2 with result A
    ↓
[Node B] reads State v2 → sees result A → creates State v3 with result B
    ↓
[Node C] reads State v3 → sees results A & B → creates State v4 with result C
    ↓
Final State (contains results A, B, and C)
```

### Storage References Pattern

Large data is stored externally, only references in state:

```python
# Instead of storing large data in state
new_state = replace(state, domain_data={'huge_file': file_contents})  # ❌ BAD

# Store reference to external storage
storage_ref = await storage.save('workflow/123/output', file_contents)
new_state = replace(
    state,
    phase_outputs={
        **state.phase_outputs,
        'my_phase': StorageRef(
            storage_type='kv',
            key='workflow/123/output',
            created_at=datetime.now()
        )
    }
)  # ✅ GOOD
```

## How to Add a New Domain

### Step 1: Define Your Domain's Phases

Create a new file `src/graphtoolkit/domains/yourdomain.py`:

```python
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from ..core.types import PhaseDefinition, WorkflowDefinition, NodeConfig
from ..core.registry import register_phase

# Step 1.1: Define input/output schemas for each phase

class Phase1Input(BaseModel):
    """Input schema for phase 1."""
    input_data: str
    config: Dict[str, Any]

class Phase1Output(BaseModel):
    """Output from phase 1."""
    analysis_result: Dict[str, Any]
    identified_items: List[str]

class Phase2Input(BaseModel):
    """Input schema for phase 2."""
    analysis_result: Dict[str, Any]  # From phase 1
    target_format: str

class Phase2Output(BaseModel):
    """Output from phase 2."""
    generated_content: str
    metadata: Dict[str, Any]
```

### Step 2: Create Phase Definitions

```python
# Step 2.1: Define each phase as configuration

def create_phase1_definition() -> PhaseDefinition:
    """Phase 1: Analyze input."""
    return PhaseDefinition(
        phase_name='analyzer',
        domain='yourdomain',
        
        # Atomic nodes that will execute in sequence
        atomic_nodes=[
            'dependency_check',    # Verify dependencies
            'load_dependencies',   # Load any required data
            'template_render',     # Render prompts
            'llm_call',           # Call LLM
            'schema_validation',   # Validate output
            'save_phase_output',   # Save results
            'state_update',       # Update state
            'quality_gate'        # Check quality
        ],
        
        # Schemas for type safety
        input_schema=Phase1Input,
        output_schema=Phase1Output,
        
        # Templates for LLM interaction
        system_template='templates/system/yourdomain/analyzer.jinja',
        user_template='templates/prompts/yourdomain/analyze.jinja',
        template_variables=['input_data', 'config'],
        
        # Storage configuration
        storage_pattern='{domain}/{workflow_id}/phase1_analysis',
        storage_type='kv',
        
        # Dependencies (none for first phase)
        dependencies=[],
        
        # Quality control
        quality_threshold=0.8,
        allow_refinement=True,
        max_refinements=3
    )

def create_phase2_definition() -> PhaseDefinition:
    """Phase 2: Generate output based on analysis."""
    return PhaseDefinition(
        phase_name='generator',
        domain='yourdomain',
        
        atomic_nodes=[
            'dependency_check',
            'load_dependencies',   # Will load phase1 output
            'template_render',
            'llm_call',
            'schema_validation',
            'save_phase_output',
            'state_update',
            'quality_gate'
        ],
        
        input_schema=Phase2Input,
        output_schema=Phase2Output,
        
        system_template='templates/system/yourdomain/generator.jinja',
        user_template='templates/prompts/yourdomain/generate.jinja',
        template_variables=['analysis_result', 'target_format'],
        
        storage_pattern='{domain}/{workflow_id}/phase2_output',
        
        # This phase depends on phase 1
        dependencies=['analyzer'],
        
        quality_threshold=0.85
    )
```

### Step 3: Register Your Domain

```python
# Step 3.1: Register phases with the global registry

def register_yourdomain_phases():
    """Register all phases for yourdomain."""
    from ..core.registry import PHASE_REGISTRY
    
    phases = {
        'analyzer': create_phase1_definition(),
        'generator': create_phase2_definition(),
    }
    
    for phase_name, phase_def in phases.items():
        PHASE_REGISTRY.register_phase(
            f'yourdomain.{phase_name}',
            phase_def
        )

# Step 3.2: Create workflow factory function

def create_yourdomain_workflow(
    input_data: str,
    config: Optional[Dict[str, Any]] = None,
    workflow_id: Optional[str] = None
) -> tuple[WorkflowDefinition, WorkflowState]:
    """Create a workflow for yourdomain."""
    from ..core.factory import build_domain_workflow, create_workflow_state
    import uuid
    
    # Ensure phases are registered
    register_yourdomain_phases()
    
    # Build workflow definition
    workflow_def = build_domain_workflow(
        domain='yourdomain',
        phases=['analyzer', 'generator'],
        enable_refinement=True
    )
    
    # Create initial state
    initial_state = create_workflow_state(
        workflow_def=workflow_def,
        workflow_id=workflow_id or str(uuid.uuid4()),
        initial_data={
            'input_data': input_data,
            'config': config or {}
        }
    )
    
    return workflow_def, initial_state
```

### Step 4: Create Templates

Create Jinja2 templates for your domain:

`templates/system/yourdomain/analyzer.jinja`:
```jinja
You are an expert analyzer. Your task is to analyze the provided input
and identify key patterns, structures, and important elements.

Provide your analysis in the specified JSON format.
```

`templates/prompts/yourdomain/analyze.jinja`:
```jinja
Please analyze the following input:

{{ input_data }}

Configuration:
{% for key, value in config.items() %}
- {{ key }}: {{ value }}
{% endfor %}

Provide a comprehensive analysis including:
1. Main patterns identified
2. Key items found
3. Recommendations for next steps
```

### Step 5: Run Your Domain Workflow

```python
async def run_yourdomain_workflow():
    """Example of running the new domain workflow."""
    from graphtoolkit.core.executor import WorkflowExecutor
    from graphtoolkit.core.deps import WorkflowDeps
    
    # Create workflow
    workflow_def, initial_state = create_yourdomain_workflow(
        input_data="Your input data here",
        config={'option1': 'value1'}
    )
    
    # Create dependencies
    deps = WorkflowDeps(
        models=ModelConfig(provider='openai', model='gpt-4o-mini'),
        storage=StorageConfig(kv_backend='memory')
    )
    
    # Execute workflow
    executor = WorkflowExecutor(deps)
    result = await executor.run(initial_state)
    
    return result
```

## Node Chaining Pattern

### How Nodes Chain Together

Each atomic node returns the next node in the chain:

```python
class DependencyCheckNode(BaseNode):
    async def run(self, ctx: GraphRunContext) -> BaseNode:
        # Check dependencies
        phase_def = ctx.state.workflow_def.phases[ctx.state.current_phase]
        for dep in phase_def.dependencies:
            if dep not in ctx.state.completed_phases:
                raise Error(f"Missing dependency: {dep}")
        
        # Return next node in chain
        return LoadDependenciesNode()

class LoadDependenciesNode(BaseNode):
    async def run(self, ctx: GraphRunContext) -> BaseNode:
        # Load data from previous phases
        loaded_data = {}
        for dep in ctx.state.workflow_def.phases[ctx.state.current_phase].dependencies:
            storage_ref = ctx.state.phase_outputs.get(dep)
            if storage_ref:
                data = await ctx.deps.storage.load(storage_ref.key)
                loaded_data[dep] = data
        
        # Update state with loaded data
        new_state = replace(
            ctx.state,
            domain_data={
                **ctx.state.domain_data,
                'loaded_dependencies': loaded_data
            }
        )
        
        # Return next node
        return TemplateRenderNode()
```

### The Complete Chain

```
GenericPhaseNode (starts the phase)
    ↓ returns
DependencyCheckNode (validates dependencies)
    ↓ returns
LoadDependenciesNode (loads previous phase outputs)
    ↓ returns
TemplateRenderNode (renders prompts with loaded data)
    ↓ returns
LLMCallNode (executes LLM with prompts)
    ↓ returns
SchemaValidationNode (validates LLM output)
    ↓ returns
SavePhaseOutputNode (saves validated output)
    ↓ returns
StateUpdateNode (marks phase complete)
    ↓ returns
QualityGateNode (checks quality)
    ↓ returns either
    → NextPhaseNode (if quality passed)
    → RefinementNode (if quality failed)
```

## Practical Examples

### Example 1: Accessing Previous Phase Data

```python
class MyCustomNode(BaseNode):
    async def run(self, ctx: GraphRunContext) -> BaseNode:
        # Access data from previous phases
        dependencies = ctx.state.domain_data.get('loaded_dependencies', {})
        
        # Get specific phase output
        phase1_data = dependencies.get('analyzer', {})
        
        # Use the data
        processed = self.process_data(phase1_data)
        
        # Add result to state
        new_state = replace(
            ctx.state,
            domain_data={
                **ctx.state.domain_data,
                'my_processed_data': processed
            }
        )
        
        return NextNode()
```

### Example 2: Conditional Node Routing

```python
class ConditionalNode(BaseNode):
    async def run(self, ctx: GraphRunContext) -> BaseNode:
        # Read some condition from state
        quality_score = ctx.state.quality_scores.get(ctx.state.current_phase, 0)
        
        # Route based on condition
        if quality_score >= 0.8:
            return SuccessNode()
        elif ctx.state.refinement_count.get(ctx.state.current_phase, 0) < 3:
            return RefinementNode()
        else:
            return AcceptLowerQualityNode()
```

### Example 3: Retry Pattern with Immutable State

```python
class RetryableNode(BaseNode):
    async def run(self, ctx: GraphRunContext) -> BaseNode:
        # Check retry count in state
        retry_key = f"{ctx.state.current_phase}_my_operation"
        retry_count = ctx.state.retry_counts.get(retry_key, 0)
        
        try:
            # Attempt operation
            result = await self.risky_operation()
            
            # Success - continue to next node
            return NextNode()
            
        except Exception as e:
            if retry_count < 3:
                # Update retry count and return self to retry
                new_state = replace(
                    ctx.state,
                    retry_counts={
                        **ctx.state.retry_counts,
                        retry_key: retry_count + 1
                    }
                )
                return RetryableNode()  # Return self to retry
            else:
                # Max retries exceeded
                return ErrorNode(error=str(e))
```

## Key Takeaways

1. **State is immutable** - Always create new state objects with `replace()`
2. **Data accumulates** - State grows as it flows through nodes
3. **Nodes chain by returning** - Each node returns the next node
4. **Configuration drives behavior** - WorkflowDefinition in state controls everything
5. **Storage uses references** - Large data stored externally, only refs in state
6. **Domains are just configuration** - Add new domains by defining phases and templates

## Common Pitfalls to Avoid

❌ **Don't mutate state directly**
```python
ctx.state.some_field = value  # Will raise error
```

❌ **Don't store large data in state**
```python
new_state = replace(state, domain_data={'huge_file': mb_of_data})
```

❌ **Don't skip nodes in the chain**
```python
# Each node should return the next, not skip ahead
return FinalNode()  # Bad if there are nodes in between
```

❌ **Don't lose existing state data**
```python
# This loses all existing domain_data
new_state = replace(state, domain_data={'only_my_data': value})

# Do this instead to preserve existing data
new_state = replace(
    state,
    domain_data={**state.domain_data, 'my_data': value}
)
```

✅ **Do follow the patterns**
- Use `replace()` for state updates
- Store references for large data
- Return the next node in sequence
- Accumulate data in domain_data
- Define phases as configuration