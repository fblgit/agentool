# Meta-Framework Mermaid Diagrams

This document contains the complete set of architectural diagrams for the meta-framework workflow system.

## 1. Master State-Driven Architecture

```mermaid
graph TB
    subgraph "State-Driven Configuration"
        WD[WorkflowDefinition in State]
        WD --> PD[PhaseDefinitions]
        WD --> NC[NodeConfigs]
        WD --> CC[ConditionConfigs]
        
        PD --> AN[atomic_nodes: List]
        PD --> DEPS[dependencies: List]
        PD --> TMPL[templates: Paths]
        
        NC --> RETRY[Retry Settings]
        NC --> ITER[Iteration Settings]
        NC --> CACHE[Cache Settings]
    end
    
    subgraph "Execution Engine"
        GPN[GenericPhaseNode]
        GPN --> READ[Read Phase from State]
        READ --> FIRST[Return First Atomic Node]
        
        FIRST --> CHAIN[Atomic Node Chain]
        CHAIN --> DC[DependencyCheck]
        DC --> LD[LoadDependencies]
        LD --> TR[TemplateRender]
        TR --> LLM[LLMCall]
        LLM --> SV[SchemaValidation]
        SV --> SO[SaveOutput]
        SO --> SU[StateUpdate]
        SU --> QG[QualityGate]
    end
    
    subgraph "State Evolution"
        S0[Initial State]
        S0 --> S1[State + Dependencies]
        S1 --> S2[State + Templates]
        S2 --> S3[State + LLM Output]
        S3 --> S4[State + Validation]
        S4 --> S5[State + Storage Ref]
        S5 --> S6[State + Completion]
        S6 --> S7[State + Quality]
    end
    
    WD -.->|Drives| GPN
    CHAIN -.->|Mutates| S0
```

## 2. Atomic Node Chaining Pattern

```mermaid
graph LR
    subgraph "Phase Execution via Atomic Chaining"
        Start[GenericPhaseNode] -->|returns| N1[DependencyCheckNode]
        N1 -->|returns| N2[LoadDependenciesNode]
        N2 -->|returns| N3[TemplateRenderNode]
        N3 -->|returns| N4[LLMCallNode]
        N4 -->|returns| N5[SchemaValidationNode]
        N5 -->|returns| N6[SavePhaseOutputNode]
        N6 -->|returns| N7[StateUpdateNode]
        N7 -->|returns| N8[QualityGateNode]
        N8 -->|Pass returns| Next[NextPhaseNode]
        N8 -->|Fail returns| Refine[RefinementNode]
        
        %% Retry paths
        N2 -.->|retry returns self| N2
        N4 -.->|retry returns self| N4
        N6 -.->|retry returns self| N6
    end
```

## 3. State-Based Retry Mechanism

```mermaid
stateDiagram-v2
    [*] --> CheckConfig: Node.run()
    
    CheckConfig --> ReadRetryConfig: Read from state.workflow_def.node_configs
    ReadRetryConfig --> CheckRetryCount: Get retry_count from state
    
    CheckRetryCount --> ExecuteOperation: First attempt or retry
    
    ExecuteOperation --> Success: Operation succeeds
    ExecuteOperation --> Failure: Operation fails
    
    Success --> UpdateState: Create new state
    UpdateState --> ReturnNext: Return next node
    
    Failure --> CheckRetryable: Check config.retryable
    CheckRetryable --> CheckMaxRetries: retryable=true
    CheckRetryable --> ReturnError: retryable=false
    
    CheckMaxRetries --> IncrementRetry: count < max_retries
    CheckMaxRetries --> ReturnError: count >= max_retries
    
    IncrementRetry --> UpdateRetryState: state.retry_counts[node]++
    UpdateRetryState --> ReturnSelf: return self.__class__()
    
    ReturnNext --> [*]
    ReturnError --> [*]
    ReturnSelf --> CheckConfig: Re-execute same node
    
    note right of ReturnSelf
        Node returns new instance
        of itself for retry
    end note
```

## 4. Node Factory Pattern

```mermaid
graph TB
    REG[NODE_REGISTRY] --> FACTORY[create_node_instance]
    FACTORY --> LOOKUP[node_id lookup]
    LOOKUP --> CLASS[get NodeClass]
    CLASS --> INSTANCE[return new NodeClass]
    
    INSTANCE --> NODE[Node Instance]
    NODE --> READS[reads ctx.state]
    READS --> CONFIG[gets configuration]
    
    subgraph examples[Registry Examples]
        DC[dependency_check maps to DependencyCheckNode]
        LD[load_dependencies maps to LoadDependenciesNode]
        TR[template_render maps to TemplateRenderNode]
        LLM[llm_call maps to LLMCallNode]
    end
```

## 5. Parallel Execution with Graph.iter()

```mermaid
graph TB
    SEQ[IterableNode with items] --> CHECK{iter_enabled?}
    
    CHECK -->|No| SINGLE[Process all at once]
    CHECK -->|Yes| FORK[Fork Execution]
    
    FORK --> P1[Process Item A]
    FORK --> P2[Process Item B] 
    FORK --> P3[Process Item C]
    
    P1 --> AGG[AggregatorNode]
    P2 --> AGG
    P3 --> AGG
    
    AGG --> NEXT[Continue Chain]
    
    subgraph control[Graph.iter Control]
        ITER[graph.iter start] --> DETECT[detect IterableNode]
        DETECT --> SPAWN[spawn tasks]
        SPAWN --> GATHER[asyncio.gather]
    end
    
    FORK -.-> ITER
```

## 6. Complete AgenTool Workflow Example

```mermaid
graph TB
    START[User Request: Create auth_tool] --> INIT[Initialize WorkflowState]
    
    INIT --> PHASE1[Phase: Analyzer]
    PHASE1 --> A_OUT[Analysis Output]
    
    A_OUT --> PHASE2[Phase: Specifier]  
    PHASE2 --> S_OUT[Specification Output]
    
    S_OUT --> PHASE3[Phase: Crafter]
    PHASE3 --> C_OUT[Generated Code]
    
    C_OUT --> PHASE4[Phase: Evaluator]
    PHASE4 --> QG{Quality Gate}
    
    QG -->|Pass| DONE[Complete: auth_tool.py]
    QG -->|Fail| REFINE[Refinement]
    REFINE --> PHASE3
    
    subgraph states[State Evolution]
        S1[Empty] --> S2[Analysis]
        S2 --> S3[Specifications]  
        S3 --> S4[Code]
        S4 --> S5[Quality]
    end
```

## 7. Error Handling Strategies

```mermaid
graph TB
    ERROR[Exception Occurs] --> TYPE{Error Type?}
    
    TYPE --> RET[RetryableError]
    TYPE --> NONRET[NonRetryableError]  
    TYPE --> VAL[ValidationError]
    
    RET --> CHECK{Check Config}
    CHECK -->|retryable=true| COUNT{count < max?}
    CHECK -->|retryable=false| FAIL[Return ErrorNode]
    
    COUNT -->|Yes| RETRY[Return self]
    COUNT -->|No| FAIL
    
    NONRET --> BUBBLE[Bubble to Orchestrator]
    VAL --> REFINE[Return RefinementNode]
    
    subgraph strategy[Retry Strategy by Node Type]
        LLM[LLM Nodes: Retryable]
        STORAGE[Storage Nodes: Configurable]
        TEMPLATE[Template Nodes: Not Retryable]
        VALIDATION[Validation Nodes: Trigger Refinement]
    end
```

## 8. Storage Reference Pattern

```mermaid
graph LR
    STATE[WorkflowState] --> REFS[phase_outputs]
    REFS --> SR1[StorageRef: analysis]
    REFS --> SR2[StorageRef: specs] 
    REFS --> SR3[StorageRef: code]
    
    SR1 -.-> DATA1[Analysis Results]
    SR2 -.-> DATA2[Specifications]
    SR3 -.-> DATA3[Generated Code]
    
    subgraph storage[Storage Layer]
        KV[(KV Store)]
        FS[(File System)]
        KV --> DATA1
        KV --> DATA2
        FS --> DATA3
    end
    
    subgraph benefits[Benefits]
        B1[Small State Size]
        B2[Parallel Access]
        B3[Version Control]
    end
```

## 9a. NodeConfig Structure

```mermaid
graph TB
    NC[NodeConfig in State] --> R[Retry Configuration]
    NC --> I[Iteration Configuration]
    NC --> C[Cache Configuration]
    
    R --> R1[retryable: bool]
    R --> R2[max_retries: int]
    R --> R3[retry_backoff: str]
    
    I --> I1[iter_enabled: bool]
    I --> I2[iter_in_type: Type]
    I --> I3[iter_out_type: Type]
    
    C --> C1[cacheable: bool]
    C --> C2[cache_ttl: int]
```

## 9b. Node Behavior Application

```mermaid
graph TB
    NODE[Atomic Node] --> READ[Read config from state]
    READ --> APPLY{Apply Configuration}
    
    APPLY --> RET_B{Retryable?}
    RET_B -->|Yes| RETRY[Self-return on error]
    RET_B -->|No| ERROR[Return ErrorNode]
    
    APPLY --> ITER_B{Iteration?}
    ITER_B -->|Yes| LOOP[Process one, return self]
    ITER_B -->|No| BATCH[Process all at once]
    
    APPLY --> CACHE_B{Cacheable?}
    CACHE_B -->|Yes| CHECK_C[Check cache first]
    CACHE_B -->|No| EXEC[Execute directly]
```

## 10. Quality Gate and Refinement Flow

```mermaid
graph TB
    QG[QualityGateNode] --> COMPARE{score >= threshold?}
    COMPARE -->|Pass| NEXT[NextPhaseNode]
    COMPARE -->|Fail| REFINE_CHECK{Can Refine?}
    
    REFINE_CHECK -->|Yes| FEEDBACK[Generate Feedback]
    REFINE_CHECK -->|No| ACCEPT[Accept Quality]
    
    FEEDBACK --> RERUN[Return RefinementNode]
    RERUN --> PHASE[Re-execute Phase]
    PHASE --> QG
    
    ACCEPT --> NEXT
```

## 11. Meta-Framework Value Proposition

```mermaid
graph LR
    subgraph traditional[Traditional Approach]
        T1[Analyzer Code 500 lines]
        T2[Specifier Code 500 lines]
        T3[Crafter Code 500 lines]
        T4[Evaluator Code 500 lines]
        
        T1 -.-> T2
        T2 -.-> T3
        T3 -.-> T4
    end
    
    subgraph meta[Meta-Framework Approach]
        PD[PhaseDefinitions 30 lines each]
        GPN[GenericPhaseNode 100 lines total]
        AN[Atomic Nodes 50 lines each]
        
        PD --> GPN
        GPN --> AN
    end
    
    CALC1[Total: 2000+ lines per domain]
    CALC2[Total: 600 lines for all domains]
    
    traditional --> CALC1
    meta --> CALC2
```

## 12. State-Driven Data Flow

```mermaid
graph TB
    INPUT[Input Data] --> VALIDATE[Schema Validation]
    VALIDATE --> STATE[WorkflowState]
    
    STATE --> PHASE[Current Phase]
    PHASE --> READ[Read PhaseDefinition]
    READ --> VARS[Extract Variables]
    VARS --> TEMPLATE[Render Templates]
    TEMPLATE --> LLM[LLM Call]
    LLM --> OUTPUT[Structured Output]
    OUTPUT --> STORE[Store Reference]
    STORE --> NEWSTATE[Updated State]
    
    NEWSTATE --> NEXT{More Phases?}
    NEXT -->|Yes| PHASE
    NEXT -->|No| COMPLETE[Workflow Complete]
```

## Diagram Legend

- **Solid Arrows**: Direct execution flow
- **Dashed Arrows**: Reference/data relationships
- **Dotted Arrows**: Optional/retry paths