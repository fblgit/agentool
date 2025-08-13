# Pydantic-Graph State Management Patterns

This document provides a comprehensive overview of valid state management patterns in the pydantic-graph framework, based on analysis of the official documentation and real-world examples.

## Overview

Pydantic-graph offers multiple approaches to state management, each optimized for different use cases. The key insight is that there's no single "correct" pattern - the choice depends on your specific requirements for mutability, persistence, testability, and complexity.

## Core Concepts

### GraphRunContext
Every node receives a `GraphRunContext` object that provides access to:
- `ctx.state`: The current state object (if any)
- `ctx.deps`: Dependencies/services (if any)
- State modifications happen through this context

### State Type Parameters
Nodes are parameterized with three types:
```python
BaseNode[StateT, DepsT, RunEndT]
```
- `StateT`: Type of state object (can be None)
- `DepsT`: Type of dependencies (can be None)
- `RunEndT`: Return type when graph ends

## Pattern 1: Mutable State Pattern

**When to use**: Simple workflows where multiple nodes need to accumulate changes to shared state.

### Characteristics
- State is a regular (non-frozen) dataclass
- Nodes directly modify state via `ctx.state.field = value`
- Changes are immediately visible to subsequent nodes
- Simple and intuitive for basic workflows

### Example: Vending Machine
```python
from dataclasses import dataclass
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

@dataclass
class MachineState:
    """Mutable state - changes accumulate across nodes."""
    user_balance: float = 0.0
    product: str | None = None

@dataclass
class CoinsInserted(BaseNode[MachineState]):
    amount: float

    async def run(self, ctx: GraphRunContext[MachineState]) -> SelectProduct | Purchase:
        # Directly modify mutable state
        ctx.state.user_balance += self.amount
        if ctx.state.product is not None:
            return Purchase(ctx.state.product)
        else:
            return SelectProduct()

@dataclass
class Purchase(BaseNode[MachineState, None, None]):
    product: str

    async def run(self, ctx: GraphRunContext[MachineState]) -> End | InsertCoin:
        if price := PRODUCT_PRICES.get(self.product):
            ctx.state.product = self.product
            if ctx.state.user_balance >= price:
                # Modify state before ending
                ctx.state.user_balance -= price
                return End(None)
            else:
                diff = price - ctx.state.user_balance
                return InsertCoin()
        else:
            return SelectProduct()

# Usage
vending_machine_graph = Graph(nodes=[InsertCoin, CoinsInserted, SelectProduct, Purchase])
state = MachineState()
await vending_machine_graph.run(InsertCoin(), state=state)
# State is mutated: state.product and state.user_balance have changed
```

### Pros
- Simple and intuitive
- Direct state manipulation
- Good for accumulating changes
- Minimal boilerplate

### Cons
- Harder to test (state mutations)
- No built-in undo/rollback
- Potential race conditions in parallel execution
- Side effects make debugging harder

## Pattern 2: Immutable State Pattern (Functional)

**When to use**: When you need predictable, testable workflows with clear state transitions.

### Characteristics
- State is a frozen dataclass
- Nodes return new state instances using `replace()`
- Each node creates a new state snapshot
- Functional programming style

### Example: Configuration Pipeline
```python
from dataclasses import dataclass, replace, field
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

@dataclass(frozen=True)
class ConfigState:
    """Immutable state - must create new instances."""
    config_values: dict = field(default_factory=dict)
    validation_errors: list = field(default_factory=list)
    processed: bool = False

@dataclass
class ValidateConfig(BaseNode[ConfigState, None, ConfigState]):
    async def run(self, ctx: GraphRunContext[ConfigState]) -> ProcessConfig | End[ConfigState]:
        errors = []
        for key, value in ctx.state.config_values.items():
            if not self.is_valid(key, value):
                errors.append(f"Invalid {key}: {value}")
        
        if errors:
            # Create new state with errors
            new_state = replace(
                ctx.state,
                validation_errors=errors
            )
            return End(new_state)
        else:
            return ProcessConfig()
    
    def is_valid(self, key: str, value: any) -> bool:
        # Validation logic
        return True

@dataclass
class ProcessConfig(BaseNode[ConfigState, None, ConfigState]):
    async def run(self, ctx: GraphRunContext[ConfigState]) -> End[ConfigState]:
        # Process the config and return new state
        processed_values = {k: v.upper() if isinstance(v, str) else v 
                           for k, v in ctx.state.config_values.items()}
        
        new_state = replace(
            ctx.state,
            config_values=processed_values,
            processed=True
        )
        return End(new_state)

# Usage
config_graph = Graph(nodes=[ValidateConfig, ProcessConfig])
initial_state = ConfigState(config_values={'key': 'value'})
result = await config_graph.run(ValidateConfig(), state=initial_state)
# initial_state is unchanged, result.output contains new state
```

### Pros
- Predictable and testable
- No side effects
- Easy to debug (clear state transitions)
- Safe for parallel execution
- Built-in history via snapshots

### Cons
- More verbose (requires `replace()`)
- Memory overhead (copying state)
- Can be less intuitive initially

## Pattern 3: Stateless Pattern

**When to use**: Simple transformations where data flows through node parameters and return values.

### Characteristics
- No state object (`StateT = None`)
- Data passed via node constructor parameters
- Return values chain between nodes
- Pure functional approach

### Example: Data Transformation Pipeline
```python
from dataclasses import dataclass
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

@dataclass
class ParseData(BaseNode[None, None, dict]):
    raw_data: str
    
    async def run(self, ctx: GraphRunContext) -> TransformData:
        parsed = json.loads(self.raw_data)
        return TransformData(parsed)

@dataclass
class TransformData(BaseNode[None, None, dict]):
    data: dict
    
    async def run(self, ctx: GraphRunContext) -> ValidateData:
        transformed = {k.upper(): v for k, v in self.data.items()}
        return ValidateData(transformed)

@dataclass
class ValidateData(BaseNode[None, None, dict]):
    data: dict
    
    async def run(self, ctx: GraphRunContext) -> End[dict]:
        if self.is_valid(self.data):
            return End(self.data)
        else:
            return End({})
    
    def is_valid(self, data: dict) -> bool:
        return len(data) > 0

# Usage
pipeline_graph = Graph(nodes=[ParseData, TransformData, ValidateData])
result = await pipeline_graph.run(ParseData('{"key": "value"}'))
# Result flows through nodes via parameters
```

### Pros
- Simplest pattern
- No state management overhead
- Purely functional
- Excellent for simple pipelines

### Cons
- Limited to simple flows
- No shared state between branches
- Data must flow linearly

## Pattern 4: Hybrid Pattern (Frozen Container with Mutable Collections)

**When to use**: Complex workflows needing both immutability guarantees and efficient updates to collections.

### Characteristics
- Main state object is frozen
- Contains mutable collections (dict, list, set)
- Provides safety with flexibility
- Best of both worlds approach

### Example: Workflow State from GraphToolkit
```python
from dataclasses import dataclass, field, replace
from typing import Dict, Set, List, Any
from datetime import datetime

@dataclass(frozen=True)  # Main container is immutable
class WorkflowState:
    """Hybrid: immutable container with mutable collections."""
    
    # Immutable core identity
    workflow_id: str
    domain: str
    
    # Mutable collections for accumulating data
    completed_phases: Set[str] = field(default_factory=set)
    phase_outputs: Dict[str, StorageRef] = field(default_factory=dict)
    domain_data: Dict[str, Any] = field(default_factory=dict)
    quality_scores: Dict[str, float] = field(default_factory=dict)
    retry_counts: Dict[str, int] = field(default_factory=dict)
    
    # Immutable metadata
    created_at: datetime = field(default_factory=datetime.now)
    
    def with_new_phase(self, phase: str) -> 'WorkflowState':
        """Helper to create new state with updated phase."""
        # Can't modify workflow_id (frozen), but can modify collections
        self.completed_phases.add(phase)  # Mutable set
        return self  # Same instance, modified collections
    
    def with_storage_ref(self, phase: str, ref: StorageRef) -> 'WorkflowState':
        """Create truly new instance when needed."""
        return replace(
            self,
            phase_outputs={**self.phase_outputs, phase: ref},
            updated_at=datetime.now()
        )

@dataclass
class ProcessPhase(BaseNode[WorkflowState]):
    phase_name: str
    
    async def run(self, ctx: GraphRunContext[WorkflowState]) -> NextPhase | End:
        # Can modify mutable collections
        ctx.state.completed_phases.add(self.phase_name)
        ctx.state.quality_scores[self.phase_name] = 0.95
        
        # Can't modify frozen fields
        # ctx.state.workflow_id = 'new_id'  # This would error!
        
        if more_phases:
            return NextPhase()
        else:
            return End(ctx.state)
```

### Pros
- Flexibility for collections
- Safety for core identity
- Efficient updates (no full copies)
- Good balance of safety and performance

### Cons
- More complex mental model
- Need to understand which fields are mutable
- Potential confusion about mutability boundaries

## Pattern 5: State with Dependencies Pattern

**When to use**: Production systems needing separation between data (state) and services/configuration (deps).

### Characteristics
- State holds data
- Dependencies hold services, clients, configuration
- Clean separation of concerns
- Both passed through context

### Example: Multi-Agent System with External Services
```python
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import httpx

@dataclass
class AgentState:
    """Mutable state for agent execution."""
    messages: list[str] = field(default_factory=list)
    results: dict = field(default_factory=dict)
    error_count: int = 0

@dataclass
class AgentDeps:
    """Immutable dependencies/services."""
    http_client: httpx.AsyncClient
    executor: ProcessPoolExecutor
    api_key: str
    model_config: dict

@dataclass
class FetchData(BaseNode[AgentState, AgentDeps, dict]):
    url: str
    
    async def run(self, ctx: GraphRunContext[AgentState, AgentDeps]) -> ProcessData | HandleError:
        try:
            response = await ctx.deps.http_client.get(
                self.url,
                headers={'Authorization': f'Bearer {ctx.deps.api_key}'}
            )
            response.raise_for_status()
            ctx.state.results['raw_data'] = response.json()
            return ProcessData()
        except Exception as e:
            ctx.state.error_count += 1
            ctx.state.messages.append(f"Error fetching: {e}")
            return HandleError(str(e))

@dataclass
class ProcessData(BaseNode[AgentState, AgentDeps, dict]):
    async def run(self, ctx: GraphRunContext[AgentState, AgentDeps]) -> End[dict]:
        # Use executor from deps for CPU-intensive processing
        loop = asyncio.get_running_loop()
        processed = await loop.run_in_executor(
            ctx.deps.executor,
            self.process_intensive,
            ctx.state.results['raw_data']
        )
        ctx.state.results['processed'] = processed
        return End(ctx.state.results)
    
    def process_intensive(self, data: dict) -> dict:
        # CPU-intensive processing
        return {k: v * 2 for k, v in data.items()}

# Usage
async with httpx.AsyncClient() as client:
    with ProcessPoolExecutor() as executor:
        deps = AgentDeps(
            http_client=client,
            executor=executor,
            api_key='secret',
            model_config={'temperature': 0.7}
        )
        state = AgentState()
        
        graph = Graph(nodes=[FetchData, ProcessData, HandleError])
        result = await graph.run(
            FetchData('https://api.example.com/data'),
            state=state,
            deps=deps
        )
```

### Pros
- Clean separation of concerns
- Easy dependency injection for testing
- Reusable services across runs
- Type-safe service access

### Cons
- More complex setup
- Need to manage service lifecycle
- Additional type parameters

## Pattern 6: Persistent State Pattern

**When to use**: Long-running workflows that need resume capability or audit trails.

### Characteristics
- State serialized between nodes
- Can resume from any point
- Full execution history available
- Supports distributed execution

### Example: Resumable Workflow
```python
from pathlib import Path
from pydantic_graph import (
    Graph, BaseNode, End, GraphRunContext,
    FileStatePersistence, FullStatePersistence
)

@dataclass
class TaskState:
    """State that will be persisted."""
    task_id: str
    steps_completed: list[str] = field(default_factory=list)
    current_step: int = 0
    data: dict = field(default_factory=dict)

@dataclass
class StepNode(BaseNode[TaskState, None, dict]):
    step_name: str
    
    async def run(self, ctx: GraphRunContext[TaskState]) -> StepNode | End[dict]:
        # Perform step
        ctx.state.steps_completed.append(self.step_name)
        ctx.state.current_step += 1
        ctx.state.data[self.step_name] = f"Result of {self.step_name}"
        
        # Check if more steps
        if ctx.state.current_step < 3:
            return StepNode(f"step_{ctx.state.current_step + 1}")
        else:
            return End(ctx.state.data)

# Initial run with persistence
async def start_workflow():
    persistence = FileStatePersistence(Path('workflow_state.json'))
    state = TaskState(task_id='task_123')
    graph = Graph(nodes=[StepNode])
    
    # Initialize and run first step
    await graph.initialize(
        StepNode('step_1'),
        state=state,
        persistence=persistence
    )

# Resume from where it left off
async def resume_workflow():
    persistence = FileStatePersistence(Path('workflow_state.json'))
    graph = Graph(nodes=[StepNode])
    
    # Load and continue from saved state
    async with graph.iter_from_persistence(persistence) as run:
        node_or_end = await run.next()
        if isinstance(node_or_end, End):
            print("Workflow completed:", node_or_end.data)
        else:
            print("Next node:", node_or_end)

# Full history tracking
async def audit_workflow():
    persistence = FullStatePersistence()
    state = TaskState(task_id='audit_123')
    graph = Graph(nodes=[StepNode])
    
    result = await graph.run(
        StepNode('step_1'),
        state=state,
        persistence=persistence
    )
    
    # Access complete execution history
    for snapshot in persistence.history:
        print(f"Node: {snapshot.node}, State: {snapshot.state}")
```

### Pros
- Resume capability for long-running workflows
- Full audit trail
- Distributed execution support
- Failure recovery

### Cons
- I/O overhead
- Storage requirements
- Serialization constraints
- More complex setup

## Pattern Comparison Table

| Pattern | Mutability | Complexity | Performance | Testability | Use Case |
|---------|------------|------------|-------------|-------------|----------|
| **Mutable State** | Full | Low | High | Low | Simple workflows, prototypes |
| **Immutable State** | None | Medium | Medium | High | Production systems, testing |
| **Stateless** | N/A | Low | High | High | Simple pipelines, transformations |
| **Hybrid** | Partial | Medium | High | Medium | Complex workflows, GraphToolkit |
| **With Dependencies** | Variable | High | High | High | Production systems with services |
| **Persistent** | Variable | High | Low | Medium | Long-running, resumable workflows |

## Best Practices

### 1. Choose Based on Requirements
- **Need simplicity?** → Mutable State or Stateless
- **Need testability?** → Immutable State
- **Need performance?** → Hybrid or Mutable
- **Need resume?** → Persistent State
- **Need services?** → State with Dependencies

### 2. State Design Guidelines
```python
# DO: Keep state focused and minimal
@dataclass
class GoodState:
    user_id: str
    step: int
    results: dict

# DON'T: Mix concerns in state
@dataclass
class BadState:
    user_id: str
    http_client: httpx.Client  # Should be in deps!
    database_conn: Connection  # Should be in deps!
    results: dict
```

### 3. Immutability Helpers
```python
# Create helper methods for common updates
@dataclass(frozen=True)
class State:
    values: dict
    
    def with_value(self, key: str, value: any) -> 'State':
        """Helper for immutable updates."""
        return replace(self, values={**self.values, key: value})
    
    def increment_counter(self, key: str) -> 'State':
        """Domain-specific helper."""
        current = self.values.get(key, 0)
        return self.with_value(key, current + 1)
```

### 4. Type Safety
```python
# Always specify type parameters explicitly
class MyNode(BaseNode[MyState, MyDeps, MyOutput]):
    pass

# Use type hints for clarity
async def run(self, ctx: GraphRunContext[MyState, MyDeps]) -> NextNode | End[MyOutput]:
    pass
```

### 5. Testing Strategies

#### For Mutable State:
```python
def test_mutable_node():
    state = MutableState(value=0)
    original_value = state.value
    
    # Run node
    await node.run(ctx_with_state(state))
    
    # Test mutations
    assert state.value != original_value
```

#### For Immutable State:
```python
def test_immutable_node():
    initial_state = ImmutableState(value=0)
    
    # Run node
    result = await node.run(ctx_with_state(initial_state))
    
    # Test immutability
    assert initial_state.value == 0  # Unchanged
    assert result.state.value == 1  # New state
```

## Migration Between Patterns

### From Mutable to Immutable:
```python
# Before (Mutable)
ctx.state.value = new_value
ctx.state.items.append(item)

# After (Immutable)
new_state = replace(ctx.state, value=new_value)
new_state = replace(ctx.state, items=[*ctx.state.items, item])
```

### From Stateless to Stateful:
```python
# Before (Stateless)
@dataclass
class ProcessNode(BaseNode[None, None, dict]):
    data: dict
    
# After (Stateful)
@dataclass
class ProcessNode(BaseNode[ProcessState, None, dict]):
    # Data now in state instead of node
```

### Adding Persistence:
```python
# Before (No persistence)
result = await graph.run(StartNode(), state=state)

# After (With persistence)
persistence = FileStatePersistence(Path('state.json'))
result = await graph.run(StartNode(), state=state, persistence=persistence)
```

## Common Pitfalls and Solutions

### Pitfall 1: Modifying Frozen State
```python
# WRONG
@dataclass(frozen=True)
class State:
    values: list

# This seems to work but violates immutability concept!
ctx.state.values.append(item)  # Modifies list in frozen dataclass

# CORRECT
@dataclass(frozen=True)
class State:
    values: tuple  # Use immutable collections

# Or use replace
new_state = replace(ctx.state, values=[*ctx.state.values, item])
```

### Pitfall 2: Sharing Mutable State
```python
# WRONG
shared_state = MutableState()
result1 = await graph1.run(Node(), state=shared_state)
result2 = await graph2.run(Node(), state=shared_state)  # Unexpected mutations!

# CORRECT
state1 = MutableState()
state2 = MutableState()  # Separate instances
# Or use deepcopy if needed
```

### Pitfall 3: Heavy Objects in State
```python
# WRONG
@dataclass
class State:
    huge_dataframe: pd.DataFrame  # Expensive to copy
    database_connection: Connection  # Should be in deps

# CORRECT
@dataclass
class State:
    dataframe_id: str  # Reference to stored data
    
@dataclass
class Deps:
    database_connection: Connection
    dataframe_store: DataFrameStore
```

## Conclusion

Pydantic-graph's flexibility in state management is one of its key strengths. By understanding these patterns and their trade-offs, you can choose the right approach for your specific use case. Remember:

1. **Start simple** (mutable or stateless) and evolve as needed
2. **Consider your requirements** (testing, persistence, performance)
3. **Use type hints** for safety and clarity
4. **Separate concerns** (state vs. dependencies)
5. **Document your choice** for team understanding

The patterns can also be mixed within a single application - use immutable state for critical paths and mutable state for simple accumulation tasks. The key is understanding the trade-offs and choosing consciously based on your specific requirements.