# State Mutations and Transformations - Phase 2

## References

- [Workflow Graph System](workflow-graph-system.md)
- [Graph Architecture](GRAPH_ARCHITECTURE.md)
- [Node Catalog](NODE_CATALOG.md)
- [Data Flow Requirements](DATA_FLOW_REQUIREMENTS.md)
- [Graph Type Definitions](GRAPH_TYPE_DEFINITIONS.md)
- [State Mutations (this doc)](STATE_MUTATIONS.md)

## Overview

This document specifies how state is transformed as it flows through the workflow graph. All mutations follow the immutable state pattern where new state objects are created rather than modifying existing ones.

## Core Mutation Principles

### 1. Immutability
```python
# NEVER do this:
ctx.state.some_field = new_value  # ❌ Mutating state

# ALWAYS do this:
new_state = replace(ctx.state, some_field=new_value)  # ✅ Creating new state
```

### 2. Accumulation
```python
# State accumulates data, doesn't lose it
new_state = replace(
    ctx.state,
    new_field=computed_value,  # Add new
    # Existing fields preserved automatically
)
```

### 3. Reference Storage
```python
# Store references, not data
ref = StorageRef(storage_type='kv', key='workflow/123/data')  # Key without prefix
new_state = ctx.state.with_storage_ref('data_ref', ref)
```

## State Mutation Patterns

### Pattern 1: Simple Field Update

```python
@dataclass
class SimpleUpdateNode(BaseNode[WorkflowState, WorkflowDeps, WorkflowState]):
    """Update a single field in state."""
    
    async def run(self, ctx: GraphRunContext[WorkflowState, WorkflowDeps]) -> NextNode:
        # Compute new value
        new_value = await self.compute_value(ctx.state)
        
        # Create new state with updated field
        new_state = replace(
            ctx.state,
            some_field=new_value
        )
        
        return self.next  # Return next node in graph
```

### Pattern 2: Nested Field Update

```python
@dataclass
class NestedUpdateNode(BaseNode[WorkflowState, WorkflowDeps, WorkflowState]):
    """Update nested fields in state."""
    
    async def run(self, ctx: GraphRunContext[WorkflowState, WorkflowDeps]) -> NextNode:
        # Update nested field (phase_status.analysis_complete)
        new_phase_status = replace(
            ctx.state.phase_status,
            analysis_complete=True
        )
        
        # Update state with new nested object
        new_state = replace(
            ctx.state,
            phase_status=new_phase_status
        )
        
        return self.next  # Return next node in graph
```

### Pattern 3: Collection Updates

```python
@dataclass
class CollectionUpdateNode(BaseNode[WorkflowState, WorkflowDeps, WorkflowState]):
    """Update collections in state."""
    
    async def run(self, ctx: GraphRunContext[WorkflowState, WorkflowDeps]) -> NextNode:
        # Add to list
        new_list = ctx.state.missing_tools + ['new_tool']
        
        # Update dictionary
        new_dict = {
            **ctx.state.quality_metrics,
            'new_tool': QualityMetrics(...)
        }
        
        # Create new state
        new_state = replace(
            ctx.state,
            missing_tools=new_list,
            quality_metrics=new_dict
        )
        
        return self.next  # Return next node in graph
```

### Pattern 4: Storage Reference Updates

```python
@dataclass
class StorageRefUpdateNode(BaseNode[WorkflowState, WorkflowDeps, WorkflowState]):
    """Update storage references."""
    
    async def run(self, ctx: GraphRunContext[WorkflowState, WorkflowDeps]) -> NextNode:
        # Save data and get reference
        data = getattr(ctx.state, 'tool_specifications', [])  # Use actual field
        ref = await self.save_to_storage(data)
        
        # Use helper method for storage refs
        new_state = ctx.state.with_storage_ref(
            ref_type='analysis_ref',
            ref=ref
        )
        
        return self.next  # Return next node in graph
```

## Phase-Specific Mutations

### Analysis Phase Mutations

```python
def mutate_after_analysis(
    state: WorkflowState, 
    analysis: Dict[str, Any]  # Analysis results as dict
) -> WorkflowState:
    """Transform state after analysis phase."""
    
    # Create storage reference (key without prefix)
    analysis_ref = StorageRef(
        storage_type='kv',
        key=f'workflow/{state.metadata.workflow_id}/analysis',
        created_at=datetime.now()
    )
    
    # Update multiple fields atomically
    return replace(
        state,
        # Add analysis results from dict
        missing_tools=analysis['missing_tools'],
        existing_tools=analysis['existing_tools'],
        system_design=analysis['system_design'],
        guidelines=analysis['guidelines'],
        
        # Update storage references
        storage=replace(
            state.storage,
            analysis_ref=analysis_ref
        ),
        
        # Mark phase complete
        phase_status=replace(
            state.phase_status,
            analysis_complete=True
        )
    )
```

### Specification Phase Mutations

```python
def mutate_after_specification(
    state: WorkflowState,
    specs: List[ToolSpecification]
) -> WorkflowState:
    """Transform state after specification phase."""
    
    # Create storage refs for each spec
    spec_refs = {}
    for spec in specs:
        ref = StorageRef(
            storage_type='kv',
            key=f'workflow/{state.metadata.workflow_id}/specs/{spec.name}'
        )
        spec_refs[spec.name] = ref
    
    # Update state
    return replace(
        state,
        # Add specifications
        tool_specifications=specs,
        
        # Update storage
        storage=replace(
            state.storage,
            tool_specs=spec_refs
        ),
        
        # Update phase
        phase_status=replace(
            state.phase_status,
            specification_complete=True
        ),
        
        # Update processing state
        processing=replace(
            state.processing,
            tools_to_process=[spec.name for spec in specs]
        )
    )
```

### Crafting Phase Mutations

```python
def mutate_after_crafting(
    state: WorkflowState,
    tool_name: str,
    code: CodeBlock
) -> WorkflowState:
    """Transform state after crafting a tool."""
    
    # Save code and get reference
    code_ref = StorageRef(
        storage_type='fs',
        key=f'generated/{state.metadata.workflow_id}/{tool_name}.py'
    )
    
    # Update state
    return replace(
        state,
        # Add generated code
        generated_code={
            **state.generated_code,
            tool_name: code
        },
        
        # Update storage refs
        storage=replace(
            state.storage,
            code_files={
                **state.storage.code_files,
                tool_name: code_ref
            }
        ),
        
        # Update processing
        processing=replace(
            state.processing,
            tools_completed=state.processing.tools_completed + [tool_name]
        )
    )
```

### Evaluation Phase Mutations

```python
def mutate_after_evaluation(
    state: EvaluationState,
    tool_name: str,
    metrics: QualityMetrics
) -> EvaluationState:
    """Transform state after evaluation."""
    
    # Check if refinement needed
    needs_refinement = metrics.quality_score < state.deps.quality.min_quality_score
    
    # Create new state
    new_state = replace(
        state,
        # Add quality metrics
        quality_metrics={
            **state.quality_metrics,
            tool_name: metrics
        },
        
        # Track quality over iterations
        quality_trajectory={
            **state.quality_trajectory,
            tool_name: state.quality_trajectory.get(tool_name, []) + [metrics.quality_score]
        }
    )
    
    # Add to refinement list if needed
    if needs_refinement:
        new_state = replace(
            new_state,
            needs_refinement=state.needs_refinement + [tool_name]
        )
    
    return new_state
```

### Refinement Mutations

```python
def mutate_after_refinement(
    state: EvaluationState,
    tool_name: str,
    refined_code: CodeBlock,
    improvement_score: float
) -> EvaluationState:
    """Transform state after refinement."""
    
    # Create refinement record
    record = RefinementRecord(
        iteration=state.evaluation_iteration,
        timestamp=datetime.now(),
        previous_score=state.quality_metrics[tool_name].quality_score,
        new_score=improvement_score,
        feedback="Applied refinement based on quality issues",
        changes_made=["Fixed imports", "Added error handling"],
        code_before_ref=state.storage.code_files[tool_name],
        code_after_ref=StorageRef(
            storage_type='fs',
            key=f'generated/{state.metadata.workflow_id}/{tool_name}_v{state.evaluation_iteration}.py'
        )
    )
    
    # Update state with refinement
    return state.with_refinement(tool_name, record)
```

## Parallel Execution State Mutations

### Fork State for Parallel Processing

```python
def fork_state_for_parallel(
    state: WorkflowState,
    items: List[Any]
) -> List[WorkflowState]:
    """Create isolated state copies for parallel execution."""
    
    forked_states = []
    for i, item in enumerate(items):
        # Deep copy state for isolation
        item_state = replace(
            state,
            processing=replace(
                state.processing,
                current_item=item,
                parallel_index=i
            )
        )
        forked_states.append(item_state)
    
    return forked_states
```

### Merge Parallel Results

```python
def merge_parallel_results(
    base_state: WorkflowState,
    parallel_states: List[WorkflowState]
) -> WorkflowState:
    """Merge results from parallel executions."""
    
    # Aggregate results
    all_results = []
    all_errors = []
    
    for pstate in parallel_states:
        if pstate.processing.last_result:
            all_results.append(pstate.processing.last_result)
        if pstate.processing.last_error:
            all_errors.append(pstate.processing.last_error)
    
    # Merge token usage
    total_usage = base_state.total_token_usage
    for pstate in parallel_states:
        for phase, usage in pstate.total_token_usage.items():
            if phase in total_usage:
                total_usage[phase] = total_usage[phase] + usage
            else:
                total_usage[phase] = usage
    
    # Create merged state
    return replace(
        base_state,
        processing=replace(
            base_state.processing,
            parallel_results=all_results,
            parallel_errors=all_errors
        ),
        total_token_usage=total_usage
    )
```

## Conditional State Mutations

### Branching State Updates

```python
def update_state_for_branch(
    state: WorkflowState,
    condition_result: bool,
    branch_name: str
) -> WorkflowState:
    """Update state when taking a branch."""
    
    return replace(
        state,
        processing=replace(
            state.processing,
            last_condition=condition_result,
            branch_taken=branch_name,
            branch_history=state.processing.branch_history + [branch_name]
        )
    )
```

## State Validation During Mutations

### Pre-Mutation Validation

```python
def validate_before_mutation(state: WorkflowState) -> None:
    """Validate state before mutation."""
    
    # Check required fields
    if not state.metadata.workflow_id:
        raise ValueError("workflow_id is required")
    
    # Check phase progression
    if state.phase_status.specification_complete:
        if not state.phase_status.analysis_complete:
            raise ValueError("Cannot complete specification before analysis")
    
    # Check data consistency
    if state.storage.analysis_ref and not state.phase_status.analysis_complete:
        raise ValueError("Analysis ref exists but phase not marked complete")
```

### Post-Mutation Validation

```python
def validate_after_mutation(
    old_state: WorkflowState,
    new_state: WorkflowState
) -> None:
    """Validate state after mutation."""
    
    # Ensure workflow_id unchanged
    if old_state.metadata.workflow_id != new_state.metadata.workflow_id:
        raise ValueError("workflow_id cannot be changed")
    
    # Ensure no data loss (accumulation principle)
    if len(new_state.missing_tools) < len(old_state.missing_tools):
        raise ValueError("Data loss detected in missing_tools")
    
    # Validate new references using proper field access
    from dataclasses import asdict
    storage_dict = asdict(new_state.storage)
    for ref_name, ref in storage_dict.items():
        if ref and not isinstance(ref, (StorageRef, dict)):
            raise ValueError(f"Invalid storage reference: {ref_name}")
```

## Token Usage Accumulation

```python
def accumulate_token_usage(
    state: WorkflowState,
    phase: str,
    usage: TokenUsage
) -> WorkflowState:
    """Accumulate token usage for a phase."""
    
    current_usage = state.total_token_usage.get(
        phase,
        TokenUsage(0, 0, 0, usage.model)
    )
    
    new_usage = current_usage + usage
    
    return replace(
        state,
        total_token_usage={
            **state.total_token_usage,
            phase: new_usage
        }
    )
```

## Error State Mutations

```python
def mutate_for_error(
    state: WorkflowState,
    error: Exception,
    node_name: str
) -> WorkflowState:
    """Update state when error occurs."""
    
    return replace(
        state,
        processing=replace(
            state.processing,
            tools_failed=state.processing.tools_failed + [node_name],
            processing_errors={
                **state.processing.processing_errors,
                node_name: str(error)
            },
            last_error=str(error),
            error_count=state.processing.error_count + 1
        )
    )
```

## State Recovery Patterns

```python
def create_recovery_point(state: WorkflowState) -> StorageRef:
    """Create a recovery point for state."""
    
    # Serialize state
    state_json = json.dumps(asdict(state), default=str)
    
    # Save to storage
    ref = StorageRef(
        storage_type='kv',
        key=f'workflow/{state.metadata.workflow_id}/recovery/{datetime.now().isoformat()}',
        size_bytes=len(state_json)
    )
    
    return ref

def restore_from_recovery_point(ref: StorageRef, deps: WorkflowDeps) -> WorkflowState:
    """Restore state from recovery point."""
    
    # Load from storage
    state_json = deps.storage_client.load_kv(ref.key)
    
    # Deserialize to state
    state_dict = json.loads(state_json)
    
    # Reconstruct state object
    return WorkflowState(**state_dict)
```

## Best Practices

### 1. Always Use Helper Methods
```python
# Good - Use helper methods for common mutations
new_state = state.with_phase_complete('analysis')
new_state = state.with_storage_ref('data_ref', ref)

# Bad - Manual replacement prone to errors
new_state = replace(state, phase_status=replace(...))
```

### 2. Atomic Multi-Field Updates
```python
# Good - Update related fields together
new_state = replace(
    state,
    tool_specifications=specs,
    phase_status=replace(state.phase_status, specification_complete=True),
    storage=replace(state.storage, specs_refs=refs)
)

# Bad - Multiple separate updates
state1 = replace(state, tool_specifications=specs)
state2 = replace(state1, phase_status=...)
state3 = replace(state2, storage=...)
```

### 3. Validate Mutations
```python
# Good - Validate before and after
validate_before_mutation(state)
new_state = perform_mutation(state)
validate_after_mutation(state, new_state)

# Bad - No validation
new_state = perform_mutation(state)
```

### 4. Track Mutation History
```python
# Good - Track what changed
new_state = replace(
    state,
    some_field=new_value,
    mutation_log=state.mutation_log + [
        f"Updated some_field from {state.some_field} to {new_value}"
    ]
)
```

This completes the Phase 2 documentation of state mutations and transformations.