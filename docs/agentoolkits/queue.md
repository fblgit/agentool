# Queue AgenToolkit

> **Note**: For guidelines on creating new AgenToolkits, see [CRAFTING_AGENTOOLS.md](../CRAFTING_AGENTOOLS.md). For testing patterns and examples, refer to [tests/agentoolkit/test_queue.py](../../../tests/agentoolkit/test_queue.py).

## Overview

The Queue AgenToolkit provides message queuing capabilities that serve as the data bus for the AgenTool ecosystem. It enables message passing between AgenTools, supports delayed message delivery via scheduler integration, and provides reactive workflow patterns through automatic message execution.

### Key Features
- FIFO message queuing with multiple named queues
- Delayed message delivery via scheduler integration
- Auto-execution of dequeued messages for reactive patterns
- Dead Letter Queue (DLQ) support for failed messages
- Message persistence and retry mechanisms
- Queue management and monitoring capabilities
- Integration with scheduler for time-based message delivery
- Configurable queue sizes and timeout handling

## Creation Method

```python
from agentoolkit.system.queue import create_queue_agent

# Create the agent
agent = create_queue_agent()
```

The creation function returns a fully configured AgenTool with name `'queue'`. Queues are created automatically when first accessed.

## Input Schema

### QueueInput

The input schema is a Pydantic model with the following fields:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `operation` | `Literal['enqueue', 'dequeue', 'peek', 'size', 'clear', 'create_queue', 'delete_queue', 'list_queues', 'get_message', 'move_to_dlq', 'get_dlq']` | Yes | - | The queue operation to perform |
| `queue_name` | `Optional[str]` | No | 'default' | Name of the queue |
| `message` | `Optional[Any]` | No | None | Message to enqueue (JSON serializable) |
| `message_id` | `Optional[str]` | No | None | Message identifier (auto-generated if not provided) |
| `delay` | `Optional[int]` | No | None | Delay in seconds before message is available |
| `schedule_at` | `Optional[str]` | No | None | ISO datetime when message should be available |
| `auto_execute` | `Optional[bool]` | No | False | Auto-execute message as AgenTool call |
| `target_agentool` | `Optional[str]` | No | None | Target AgenTool for auto-execution |
| `max_size` | `Optional[int]` | No | None | Maximum queue size |
| `timeout` | `Optional[float]` | No | None | Timeout for dequeue operation |
| `max_retries` | `Optional[int]` | No | 3 | Max retries before moving to DLQ |

## Operations Schema

The routing configuration maps operations to tool functions:

| Operation | Description | Required Parameters |
|-----------|-------------|-------------------|
| `enqueue` | Add message to queue | `queue_name`, `message` |
| `dequeue` | Remove and return message from queue | `queue_name` |
| `peek` | View next message without removing | `queue_name` |
| `size` | Get queue size and statistics | `queue_name` |
| `clear` | Clear all messages from queue | `queue_name` |
| `list_queues` | List all queues with metadata | - |
| `get_dlq` | Get messages from dead letter queue | `queue_name` |

## Output Schema

### QueueOutput

All operations return a `QueueOutput` model with:

| Field | Type | Description |
|-------|------|-------------|
| `operation` | `str` | The operation that was performed |
| `queue_name` | `str` | The queue that was operated on |
| `message` | `str` | Human-readable result message |
| `data` | `Optional[Dict[str, Any]]` | Operation-specific data |

### Operation-Specific Data Fields

- **enqueue**: `message_id`, `queue_size` (immediate) or `scheduled_at`, `job_id` (delayed)
- **dequeue**: `message_id`, `message`, `timestamp`, `executed` (if auto-execute), `execution_result`
- **peek**: `message_id`, `message`, `timestamp` or `empty`
- **size**: `size`, `max_size`, `total_enqueued`, `total_dequeued`, `dlq_size`
- **clear**: `cleared_count`
- **list_queues**: `queues`, `count`
- **get_dlq**: `messages`, `count`

## Message Structure

Messages are automatically wrapped with metadata:

```python
{
    'id': 'msg_abc12345',           # Auto-generated or provided
    'payload': original_message,     # Your message content
    'timestamp': '2024-01-01T12:00:00Z',
    'queue_name': 'my_queue',
    'retry_count': 0
}
```

## Dependencies

This AgenToolkit depends on:
- **scheduler**: Used for delayed message delivery and auto-execution

## Tools

### manage_queue
```python
async def manage_queue(ctx: RunContext[Any], operation: str, **kwargs) -> QueueOutput
```
Main routing function that dispatches to specific queue operation handlers.

**Core Queue Functions:**

### queue_enqueue
```python
async def queue_enqueue(ctx: RunContext[Any], queue_name: str, message: Any,
                       message_id: Optional[str], delay: Optional[int],
                       schedule_at: Optional[str], max_size: Optional[int]) -> QueueOutput
```
Add a message to the queue with optional delayed delivery.

**Key Features:**
- Auto-generates message ID if not provided
- Supports immediate and delayed enqueuing
- Delayed messages use scheduler integration
- Queue size limits (if specified)

**Raises:**
- `RuntimeError`: If enqueue operation fails or scheduling fails

### queue_dequeue
```python
async def queue_dequeue(ctx: RunContext[Any], queue_name: str, timeout: Optional[float],
                       auto_execute: bool, target_agentool: Optional[str],
                       max_retries: int) -> QueueOutput
```
Remove and return a message from the queue with optional auto-execution.

**Key Features:**
- Optional timeout for blocking dequeue
- Auto-execution for reactive patterns
- Retry mechanism with DLQ support
- Execution result tracking

**Raises:**
- `TimeoutError`: If timeout expires with no messages
- `RuntimeError`: If dequeue operation fails or execution fails

### queue_peek
```python
async def queue_peek(ctx: RunContext[Any], queue_name: str) -> QueueOutput
```
View the next message without removing it from the queue.

**Key Features:**
- Non-destructive message inspection
- Returns message metadata and payload

**Raises:**
- `RuntimeError`: If peek operation fails

### queue_size
```python
async def queue_size(ctx: RunContext[Any], queue_name: str) -> QueueOutput
```
Get queue size and statistics including DLQ information.

**Key Features:**
- Current queue size
- Total enqueued/dequeued counters
- DLQ size information
- Queue configuration details

**Raises:**
- `RuntimeError`: If size operation fails

### queue_clear
```python
async def queue_clear(ctx: RunContext[Any], queue_name: str) -> QueueOutput
```
Clear all messages from a queue.

**Key Features:**
- Removes all pending messages
- Returns count of cleared messages
- Preserves queue metadata

**Raises:**
- `RuntimeError`: If clear operation fails

### queue_list
```python
async def queue_list(ctx: RunContext[Any]) -> QueueOutput
```
List all queues with their metadata and statistics.

**Key Features:**
- Shows all existing queues
- Includes size and statistics for each queue
- DLQ information per queue

**Raises:**
- `RuntimeError`: If list operation fails

### queue_get_dlq
```python
async def queue_get_dlq(ctx: RunContext[Any], queue_name: str) -> QueueOutput
```
Get messages from the dead letter queue.

**Key Features:**
- Retrieves failed messages
- Includes retry count and failure information
- Limited to last 1000 DLQ messages per queue

**Raises:**
- `RuntimeError`: If DLQ retrieval fails

## Reactive Patterns

The queue supports reactive execution patterns through auto-execution:

### Auto-Execution Flow
1. Message is dequeued with `auto_execute=True`
2. Message payload is passed to specified `target_agentool`
3. Execution happens via scheduler for monitoring
4. Success: Returns execution result
5. Failure: Retries up to `max_retries`, then moves to DLQ

### Message Format for Auto-Execution
```python
# Option 1: Message contains agentool name
message = {
    'agentool_name': 'logging',
    'input_data': {
        'operation': 'log',
        'level': 'INFO',
        'message': 'Processed item'
    }
}

# Option 2: Target specified in dequeue
message = {
    'operation': 'get',
    'key': 'config_value'
}
# target_agentool='config' specified in dequeue call
```

## Exceptions

| Exception Type | Scenarios |
|---------------|-----------|
| `TimeoutError` | - Dequeue timeout expires with no messages available |
| `ValueError` | - Unknown operation<br>- No target AgenTool specified for auto-execution |
| `RuntimeError` | - Queue operation failures<br>- Message execution failures<br>- Scheduler integration errors<br>- DLQ operation failures |

## Usage Examples

### Basic Queue Operations
```python
from agentoolkit.system.queue import create_queue_agent
from agentool.core.injector import get_injector

# Create and register the agent
agent = create_queue_agent()
injector = get_injector()

# Enqueue a simple message
result = await injector.run('queue', {
    "operation": "enqueue",
    "queue_name": "tasks",
    "message": {"task": "process_data", "id": "123"}
})

# Dequeue a message
result = await injector.run('queue', {
    "operation": "dequeue",
    "queue_name": "tasks"
})

# Peek at next message without removing
result = await injector.run('queue', {
    "operation": "peek",
    "queue_name": "tasks"
})
```

### Delayed Message Delivery
```python
# Enqueue with delay (5 minutes)
result = await injector.run('queue', {
    "operation": "enqueue",
    "queue_name": "notifications",
    "message": {
        "type": "reminder",
        "user_id": "user_123",
        "content": "Meeting in 5 minutes"
    },
    "delay": 300  # 5 minutes in seconds
})

# Enqueue for specific time
result = await injector.run('queue', {
    "operation": "enqueue",
    "queue_name": "scheduled_reports",
    "message": {
        "report_type": "daily",
        "date": "2024-01-01"
    },
    "schedule_at": "2024-01-01T09:00:00"
})
```

### Reactive Execution Patterns
```python
# Enqueue message for auto-execution
result = await injector.run('queue', {
    "operation": "enqueue",
    "queue_name": "log_events", 
    "message": {
        "agentool_name": "logging",
        "input_data": {
            "operation": "log",
            "level": "INFO",
            "message": "User action completed",
            "data": {"user_id": "123", "action": "purchase"}
        }
    }
})

# Auto-execute on dequeue
result = await injector.run('queue', {
    "operation": "dequeue",
    "queue_name": "log_events",
    "auto_execute": True,
    "max_retries": 3
})

# Alternatively, specify target AgenTool in dequeue
result = await injector.run('queue', {
    "operation": "enqueue",
    "queue_name": "config_updates",
    "message": {
        "operation": "set",
        "key": "feature.enabled",
        "value": True
    }
})

result = await injector.run('queue', {
    "operation": "dequeue",
    "queue_name": "config_updates",
    "auto_execute": True,
    "target_agentool": "config",
    "max_retries": 5
})
```

### Queue Management
```python
# Get queue size and statistics
result = await injector.run('queue', {
    "operation": "size",
    "queue_name": "tasks"
})

# List all queues
result = await injector.run('queue', {
    "operation": "list_queues"
})

# Clear a queue
result = await injector.run('queue', {
    "operation": "clear",
    "queue_name": "temp_queue"
})

# Check dead letter queue
result = await injector.run('queue', {
    "operation": "get_dlq",
    "queue_name": "tasks"
})
```

### Advanced Usage with Timeouts
```python
# Dequeue with timeout
try:
    result = await injector.run('queue', {
        "operation": "dequeue",
        "queue_name": "priority_tasks",
        "timeout": 10.0  # Wait up to 10 seconds
    })
except TimeoutError:
    print("No messages available within timeout")

# Queue with size limit
result = await injector.run('queue', {
    "operation": "enqueue",
    "queue_name": "limited_queue",
    "message": {"data": "important"},
    "max_size": 100  # Limit queue to 100 messages
})
```

### Error Handling and DLQ
```python
# Enqueue message that will fail execution
result = await injector.run('queue', {
    "operation": "enqueue",
    "queue_name": "error_prone",
    "message": {
        "agentool_name": "nonexistent_tool",
        "input_data": {"operation": "invalid"}
    }
})

# Try to execute - will fail and move to DLQ after retries
try:
    result = await injector.run('queue', {
        "operation": "dequeue",
        "queue_name": "error_prone",
        "auto_execute": True,
        "max_retries": 2
    })
except RuntimeError as e:
    print(f"Execution failed: {e}")

# Check DLQ for failed messages
result = await injector.run('queue', {
    "operation": "get_dlq",
    "queue_name": "error_prone"
})
```

### Multi-Queue Workflows
```python
# Create a workflow with multiple queues
queues = ["input", "processing", "output", "errors"]

# Process workflow
input_data = {"item_id": "123", "data": "sample"}

# Stage 1: Input queue
await injector.run('queue', {
    "operation": "enqueue",
    "queue_name": "input",
    "message": input_data
})

# Stage 2: Processing (with auto-execution)
await injector.run('queue', {
    "operation": "dequeue", 
    "queue_name": "input",
    "auto_execute": True,
    "target_agentool": "templates",  # Process with templates
    "max_retries": 3
})

# Results automatically go to output queue via scheduler result_queue
```

## Testing

The test suite is located at `tests/agentoolkit/test_queue.py`. Tests cover:
- Basic queue operations (enqueue, dequeue, peek, size, clear)
- Delayed message delivery via scheduler integration
- Auto-execution and reactive patterns
- Dead letter queue functionality
- Error handling and retry mechanisms
- Timeout behavior for dequeue operations
- Multi-queue scenarios and workflow patterns

To run tests:
```bash
pytest tests/agentoolkit/test_queue.py -v
```

## Notes

- Queues are created automatically when first accessed
- Message IDs are auto-generated using UUID if not provided
- Delayed messages use the scheduler AgenToolkit for time-based delivery
- Auto-execution runs via scheduler for consistent monitoring and error handling
- DLQ is limited to 1000 messages per queue to prevent memory issues
- Queue operations are thread-safe using asyncio primitives
- Message payloads must be JSON-serializable for storage and retrieval
- The queue serves as the primary data bus for AgenTool ecosystem communication
- Integration with scheduler enables complex reactive workflow patterns
- Retry mechanisms ensure message reliability with configurable failure handling