# Node Catalog - Phase 1 Specification

## Overview

This catalog enumerates all node types in the workflow graph system, providing technical specifications without implementation details. Each node is designed as an atomic operation following the single responsibility principle.

## Node Type Hierarchy

```mermaid
graph TD
    BaseNode[BaseNode - Abstract]
    BaseNode --> StorageNode[StorageNode - Abstract]
    BaseNode --> TransformNode[TransformNode - Abstract]
    BaseNode --> ValidationNode[ValidationNode - Abstract]
    BaseNode --> LLMNode[LLMNode - Abstract]
    BaseNode --> ControlNode[ControlNode - Abstract]
    
    StorageNode --> LoadKVNode
    StorageNode --> SaveKVNode
    StorageNode --> LoadFSNode
    StorageNode --> SaveFSNode
    StorageNode --> BatchLoadNode
    StorageNode --> BatchSaveNode
    StorageNode --> ExistsCheckNode
    StorageNode --> DeleteNode
    
    TransformNode --> JSONParseNode
    TransformNode --> JSONSerializeNode
    TransformNode --> TemplateRenderNode
    TransformNode --> CodeFormatNode
    TransformNode --> DataMergeNode
    TransformNode --> DataFilterNode
    TransformNode --> DataMapNode
    TransformNode --> DataReduceNode
    
    ValidationNode --> SyntaxValidationNode
    ValidationNode --> ImportValidationNode
    ValidationNode --> SchemaValidationNode
    ValidationNode --> QualityGateNode
    ValidationNode --> DependencyCheckNode
    ValidationNode --> TypeCheckNode
    
    LLMNode --> PromptBuilderNode
    LLMNode --> LLMCallNode
    LLMNode --> ResponseParserNode
    LLMNode --> RefinementNode
    LLMNode --> BatchLLMNode
    LLMNode --> StreamingLLMNode
    
    ControlNode --> ConditionalNode
    ControlNode --> ParallelMapNode
    ControlNode --> SequentialMapNode
    ControlNode --> AggregatorNode
    ControlNode --> RetryNode
    ControlNode --> LoopNode
    ControlNode --> ForkNode
    ControlNode --> JoinNode
```

## Base Node Types

### BaseNode (Abstract)
**Purpose**: Root class for all nodes in the system  
**Signature**: `BaseNode[StateT, DepsT, OutputT]`  
**Required Method**: `async def run(self, ctx: GraphRunContext[StateT, DepsT]) -> NextNode | End[OutputT]`

**Data Requirements**:
- Input: GraphRunContext with state and dependencies
- Output: Next node or End with result

### StorageNode (Abstract)
**Purpose**: Base for all storage operation nodes  
**Extends**: BaseNode  
**Common Pattern**: Key-based operations on storage systems

### TransformNode (Abstract)
**Purpose**: Base for data transformation operations  
**Extends**: BaseNode  
**Common Pattern**: Input data → Transform → Output data

### ValidationNode (Abstract)
**Purpose**: Base for validation and checking operations  
**Extends**: BaseNode  
**Common Pattern**: Input → Validate → Pass/Fail result

### LLMNode (Abstract)
**Purpose**: Base for LLM interaction nodes  
**Extends**: BaseNode  
**Common Pattern**: Prompt → LLM Call → Response

### ControlNode (Abstract)
**Purpose**: Base for flow control operations  
**Extends**: BaseNode  
**Common Pattern**: Evaluate condition → Route execution

## Storage Nodes

### LoadKVNode
**Purpose**: Load data from storage_kv  
**Operation**: Retrieve value by key from key-value store

**Data Requirements**:
```
Input State:
  - key: str (reference to storage location)
Output State:
  - loaded_data: Any (deserialized data)
  - load_timestamp: datetime
```

### SaveKVNode
**Purpose**: Save data to storage_kv  
**Operation**: Store value with key in key-value store

**Data Requirements**:
```
Input State:
  - key: str
  - data: Any (serializable)
Output State:
  - save_ref: str (storage reference)
  - save_timestamp: datetime
```

### LoadFSNode
**Purpose**: Load file from storage_fs  
**Operation**: Read file content from filesystem

**Data Requirements**:
```
Input State:
  - path: str (file path)
Output State:
  - content: str (file content)
  - metadata: FileMetadata
```

### SaveFSNode
**Purpose**: Save file to storage_fs  
**Operation**: Write content to filesystem

**Data Requirements**:
```
Input State:
  - path: str
  - content: str
  - create_parents: bool
Output State:
  - file_ref: str
  - bytes_written: int
```

### BatchLoadNode
**Purpose**: Load multiple items in parallel  
**Operation**: Parallel retrieval from storage

**Data Requirements**:
```
Input State:
  - keys: List[str]
  - storage_type: Literal['kv', 'fs']
Output State:
  - loaded_items: Dict[str, Any]
  - failed_keys: List[str]
```

### BatchSaveNode
**Purpose**: Save multiple items in parallel  
**Operation**: Parallel storage of multiple items

**Data Requirements**:
```
Input State:
  - items: Dict[str, Any]
  - storage_type: Literal['kv', 'fs']
Output State:
  - saved_refs: Dict[str, str]
  - failed_items: List[str]
```

### ExistsCheckNode
**Purpose**: Check if storage key/path exists  
**Operation**: Verify existence without loading

**Data Requirements**:
```
Input State:
  - key_or_path: str
  - storage_type: Literal['kv', 'fs']
Output State:
  - exists: bool
  - metadata: Optional[Dict]
```

### DeleteNode
**Purpose**: Remove item from storage  
**Operation**: Delete by key or path

**Data Requirements**:
```
Input State:
  - key_or_path: str
  - storage_type: Literal['kv', 'fs']
Output State:
  - deleted: bool
  - error: Optional[str]
```

## Transform Nodes

### JSONParseNode
**Purpose**: Parse JSON string to object  
**Operation**: Deserialize JSON with validation

**Data Requirements**:
```
Input State:
  - json_string: str
Output State:
  - parsed_data: Any
  - parse_errors: List[str]
```

### JSONSerializeNode
**Purpose**: Serialize object to JSON string  
**Operation**: Convert Python object to JSON

**Data Requirements**:
```
Input State:
  - data: Any
  - indent: Optional[int]
Output State:
  - json_string: str
  - serialization_metadata: Dict
```

### TemplateRenderNode
**Purpose**: Render Jinja2 templates  
**Operation**: Template + context → rendered output

**Data Requirements**:
```
Input State:
  - template_name: str
  - context: Dict[str, Any]
Output State:
  - rendered: str
  - template_metadata: Dict
```

### CodeFormatNode
**Purpose**: Format Python code  
**Operation**: Apply code formatting standards

**Data Requirements**:
```
Input State:
  - code: str
  - style: Literal['black', 'yapf', 'autopep8']
Output State:
  - formatted_code: str
  - changes_made: bool
```

### DataMergeNode
**Purpose**: Merge multiple data sources  
**Operation**: Combine dictionaries/lists with strategy

**Data Requirements**:
```
Input State:
  - sources: List[Dict]
  - merge_strategy: Literal['override', 'append', 'deep']
Output State:
  - merged_data: Dict
  - conflicts: List[str]
```

### DataFilterNode
**Purpose**: Filter collection by criteria  
**Operation**: Apply predicate to collection

**Data Requirements**:
```
Input State:
  - collection: List[Any]
  - filter_criteria: Dict
Output State:
  - filtered: List[Any]
  - removed_count: int
```

### DataMapNode
**Purpose**: Transform each item in collection  
**Operation**: Apply function to each element

**Data Requirements**:
```
Input State:
  - collection: List[Any]
  - map_operation: str (operation identifier)
Output State:
  - mapped: List[Any]
  - transform_errors: List[str]
```

### DataReduceNode
**Purpose**: Reduce collection to single value  
**Operation**: Aggregate collection elements

**Data Requirements**:
```
Input State:
  - collection: List[Any]
  - reduce_operation: str
  - initial_value: Any
Output State:
  - reduced_value: Any
  - reduction_metadata: Dict
```

## Validation Nodes

### SyntaxValidationNode
**Purpose**: Validate Python syntax  
**Operation**: AST parsing and syntax checking

**Data Requirements**:
```
Input State:
  - code: str
  - python_version: str
Output State:
  - syntax_valid: bool
  - syntax_errors: List[SyntaxError]
  - ast_metadata: Dict
```

### ImportValidationNode
**Purpose**: Check import availability  
**Operation**: Verify imports can be resolved

**Data Requirements**:
```
Input State:
  - code: str
  - allowed_packages: List[str]
Output State:
  - imports_valid: bool
  - missing_imports: List[str]
  - forbidden_imports: List[str]
```

### SchemaValidationNode
**Purpose**: Validate against Pydantic schema  
**Operation**: Type checking with Pydantic models

**Data Requirements**:
```
Input State:
  - data: Any
  - schema_ref: str (reference to schema)
Output State:
  - schema_valid: bool
  - validation_errors: List[ValidationError]
  - coerced_data: Any
```

### QualityGateNode
**Purpose**: Check quality thresholds  
**Operation**: Compare metrics against thresholds

**Data Requirements**:
```
Input State:
  - metrics: Dict[str, float]
  - thresholds: Dict[str, float]
Output State:
  - passed: bool
  - failed_metrics: List[str]
  - quality_report: Dict
```

### DependencyCheckNode
**Purpose**: Verify dependencies available  
**Operation**: Check required dependencies exist

**Data Requirements**:
```
Input State:
  - dependencies: List[str]
  - check_type: Literal['import', 'pip', 'system']
Output State:
  - dependencies_met: bool
  - missing: List[str]
  - versions: Dict[str, str]
```

### TypeCheckNode
**Purpose**: Static type checking  
**Operation**: Validate type annotations

**Data Requirements**:
```
Input State:
  - code: str
  - strict_mode: bool
Output State:
  - type_valid: bool
  - type_errors: List[TypeError]
  - type_coverage: float
```

## LLM Nodes

### PromptBuilderNode
**Purpose**: Construct prompts from templates  
**Operation**: Template + data → formatted prompt

**Data Requirements**:
```
Input State:
  - template_ref: str
  - prompt_data: Dict[str, Any]
  - system_prompt: Optional[str]
Output State:
  - prompt: str
  - token_estimate: int
  - prompt_metadata: Dict
```

### LLMCallNode
**Purpose**: Execute LLM API call  
**Operation**: Send prompt, receive response

**Data Requirements**:
```
Input State:
  - prompt: str
  - model: str (model identifier)
  - parameters: ModelParameters
Output State:
  - response: str
  - usage: TokenUsage
  - model_metadata: Dict
```

### ResponseParserNode
**Purpose**: Parse structured LLM response  
**Operation**: Extract structured data from text

**Data Requirements**:
```
Input State:
  - response: str
  - expected_format: str (schema reference)
Output State:
  - parsed_response: Any
  - parse_confidence: float
  - extraction_errors: List[str]
```

### RefinementNode
**Purpose**: Iterative improvement with LLM  
**Operation**: Apply feedback to improve output

**Data Requirements**:
```
Input State:
  - current_output: str
  - feedback: str
  - iteration: int
Output State:
  - refined_output: str
  - changes_made: List[str]
  - improvement_score: float
```

### BatchLLMNode
**Purpose**: Parallel LLM calls  
**Operation**: Process multiple prompts concurrently

**Data Requirements**:
```
Input State:
  - prompts: List[str]
  - model: str
  - max_concurrent: int
Output State:
  - responses: List[str]
  - total_usage: TokenUsage
  - failed_indices: List[int]
```

### StreamingLLMNode
**Purpose**: Stream LLM responses  
**Operation**: Progressive response generation

**Data Requirements**:
```
Input State:
  - prompt: str
  - model: str
  - stream_callback: str (callback reference)
Output State:
  - final_response: str
  - chunks_received: int
  - stream_metadata: Dict
```

## Control Flow Nodes

### ConditionalNode
**Purpose**: Branch execution based on condition  
**Operation**: Evaluate condition, choose path

**Data Requirements**:
```
Input State:
  - condition_data: Any
  - condition_type: str
Output State:
  - branch_taken: Literal['true', 'false']
  - condition_result: bool
```

**Flow Pattern**:
```mermaid
graph LR
    Cond{Condition} -->|True| TrueBranch
    Cond -->|False| FalseBranch
```

### ParallelMapNode
**Purpose**: Execute sub-graph for each item in parallel  
**Operation**: Parallel processing of collection

**Data Requirements**:
```
Input State:
  - items: List[Any]
  - max_workers: int
Output State:
  - results: List[Any]
  - execution_times: List[float]
  - failed_items: List[int]
```

**Flow Pattern**:
```mermaid
graph LR
    Items --> P1[Process 1]
    Items --> P2[Process 2]
    Items --> P3[Process N]
    P1 --> Results
    P2 --> Results
    P3 --> Results
```

### SequentialMapNode
**Purpose**: Process items one by one  
**Operation**: Sequential processing with state accumulation

**Data Requirements**:
```
Input State:
  - items: List[Any]
  - accumulate_state: bool
Output State:
  - results: List[Any]
  - accumulated_state: Any
  - processing_order: List[int]
```

### AggregatorNode
**Purpose**: Combine results from parallel operations  
**Operation**: Merge parallel execution results

**Data Requirements**:
```
Input State:
  - partial_results: List[Any]
  - aggregation_strategy: str
Output State:
  - aggregated_result: Any
  - aggregation_metadata: Dict
```

### RetryNode
**Purpose**: Retry failed operations  
**Operation**: Exponential backoff retry logic

**Data Requirements**:
```
Input State:
  - operation_state: Any
  - retry_count: int
  - max_retries: int
Output State:
  - result: Any
  - attempts_made: int
  - retry_errors: List[str]
```

### LoopNode
**Purpose**: Iterate until condition met  
**Operation**: Conditional loop execution

**Data Requirements**:
```
Input State:
  - loop_state: Any
  - iteration: int
  - max_iterations: int
Output State:
  - final_state: Any
  - iterations_completed: int
  - termination_reason: str
```

**Flow Pattern**:
```mermaid
graph LR
    Init --> Check{Continue?}
    Check -->|Yes| Process
    Process --> Update
    Update --> Check
    Check -->|No| End
```

### ForkNode
**Purpose**: Split execution into multiple paths  
**Operation**: Create parallel execution branches

**Data Requirements**:
```
Input State:
  - fork_data: Any
  - branch_count: int
Output State:
  - branch_states: List[Any]
  - fork_metadata: Dict
```

### JoinNode
**Purpose**: Synchronize parallel branches  
**Operation**: Wait for all branches to complete

**Data Requirements**:
```
Input State:
  - branch_results: List[Any]
  - join_strategy: str
Output State:
  - joined_result: Any
  - join_metadata: Dict
```

## Node Composition Patterns

### Sub-Graph Composition

Nodes can contain sub-graphs for complex operations:

```python
@dataclass
class ComplexOperationNode(BaseNode[...]):
    sub_graph: Graph  # Contains other nodes
    
    async def run(self, ctx):
        # Execute sub-graph with current state
        result = await self.sub_graph.run(...)
        return NextNode(result)
```

### Dynamic Node Selection

Nodes can dynamically choose next node based on state:

```python
async def run(self, ctx):
    if ctx.state.quality_score > 0.8:
        return ApprovalNode(...)
    elif ctx.state.iteration < 3:
        return RefinementNode(...)
    else:
        return FallbackNode(...)
```

### State Accumulation Pattern

Nodes add to state without removing:

```python
async def run(self, ctx):
    # Add new data to state
    new_state = dataclasses.replace(
        ctx.state,
        new_field=computed_value,
        # Existing fields preserved
    )
    # Pass accumulated state forward
    return NextNode(state=new_state)
```

## Data Flow Requirements

### Phase Transitions

```mermaid
graph TB
    subgraph "Analysis Phase"
        A[catalog_ref] --> AA[AnalyzerNodes]
        AA --> AB[analysis_ref, missing_tools]
    end
    
    subgraph "Specification Phase"
        AB --> S[SpecifierNodes]
        S --> SB[specs_refs, tool_specs]
    end
    
    subgraph "Crafting Phase"
        SB --> C[CrafterNodes]
        C --> CB[code_refs, implementation_refs]
    end
    
    subgraph "Evaluation Phase"
        CB --> E[EvaluatorNodes]
        E --> EB{Quality Check}
        EB -->|Pass| EF[final_code_refs]
        EB -->|Fail| ER[RefinementNodes]
        ER --> E
    end
```

### Storage Reference Pattern

All data stored externally, only references in state:

```
State Contains:
  - storage_kv:workflow/{id}/analysis → analysis_ref
  - storage_kv:workflow/{id}/specs/* → specs_refs
  - storage_fs:generated/{id}/*.py → code_refs
  
Benefits:
  - Small state size
  - Parallel access
  - Persistence friendly
  - Memory efficient
```

## Next Phase Specifications

Phase 2 will add:
- Detailed field definitions for each node
- State mutation specifications
- Inter-node contracts
- Dependency requirements

Phase 3 will add:
- Complete implementation examples
- Integration test scenarios
- Performance benchmarks
- Migration guides