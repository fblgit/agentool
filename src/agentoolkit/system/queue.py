"""
Queue AgenTool - Data bus for message passing between AgenTools.

This toolkit provides message queuing capabilities with scheduler integration,
serving as the data bus for the AgenTool ecosystem. Messages can trigger
scheduled executions, creating reactive workflows.

Features:
- FIFO message queuing
- Multiple named queues
- Delayed message delivery via scheduler
- Auto-execution of dequeued messages
- Dead letter queue support
- Message persistence

Example Usage:
    >>> from agentoolkit.system import create_queue_agent
    >>> from agentool.core.injector import get_injector
    >>> 
    >>> # Create and register the agent
    >>> agent = create_queue_agent()
    >>> 
    >>> # Enqueue a message
    >>> injector = get_injector()
    >>> result = await injector.run('queue', {
    ...     "operation": "enqueue",
    ...     "queue_name": "tasks",
    ...     "message": {"task": "process_data", "id": "123"}
    ... })
    >>> 
    >>> # Enqueue with delay (uses scheduler)
    >>> result = await injector.run('queue', {
    ...     "operation": "enqueue",
    ...     "queue_name": "notifications",
    ...     "message": {"notify": "user", "after": "5 minutes"},
    ...     "delay": 300  # 5 minutes
    ... })
"""

import asyncio
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Literal
from collections import deque
from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from agentool.base import BaseOperationInput
from agentool import create_agentool
from agentool.core.registry import RoutingConfig
from agentool.core.injector import get_injector


# Global queue storage
_queues: Dict[str, asyncio.Queue] = {}  # queue_name -> asyncio.Queue
_queue_metadata: Dict[str, Dict[str, Any]] = {}  # queue_name -> metadata
_message_registry: Dict[str, Dict[str, Any]] = {}  # message_id -> message info
_dlq: Dict[str, deque] = {}  # dead letter queues


class QueueInput(BaseOperationInput):
    """Input schema for queue operations."""
    operation: Literal[
        'enqueue', 'dequeue', 'peek', 'size', 'clear',
        'create_queue', 'delete_queue', 'list_queues',
        'get_message', 'move_to_dlq', 'get_dlq'
    ] = Field(description="The queue operation to perform")
    
    # Queue identification
    queue_name: Optional[str] = Field('default', description="Name of the queue")
    
    # Message data
    message: Optional[Any] = Field(None, description="Message to enqueue (JSON serializable)")
    message_id: Optional[str] = Field(None, description="Message identifier")
    
    # Scheduling options (integration with scheduler)
    delay: Optional[int] = Field(None, description="Delay in seconds before message is available")
    schedule_at: Optional[str] = Field(None, description="ISO datetime when message should be available")
    
    # Execution options (for reactive patterns)
    auto_execute: Optional[bool] = Field(False, description="Auto-execute message as AgenTool call")
    target_agentool: Optional[str] = Field(None, description="Target AgenTool for auto-execution")
    
    # Queue configuration
    max_size: Optional[int] = Field(None, description="Maximum queue size")
    timeout: Optional[float] = Field(None, description="Timeout for dequeue operation")
    
    # DLQ options
    max_retries: Optional[int] = Field(3, description="Max retries before moving to DLQ")


class QueueOutput(BaseModel):
    """Structured output for queue operations."""
    operation: str = Field(description="The operation that was performed")
    queue_name: str = Field(description="The queue that was operated on")
    message: str = Field(description="Human-readable result message")
    data: Optional[Dict[str, Any]] = Field(None, description="Operation-specific data")


def _get_or_create_queue(queue_name: str, max_size: Optional[int] = None) -> asyncio.Queue:
    """Get or create a queue."""
    if queue_name not in _queues:
        _queues[queue_name] = asyncio.Queue(maxsize=max_size or 0)
        _queue_metadata[queue_name] = {
            'created_at': datetime.now().isoformat(),
            'max_size': max_size,
            'total_enqueued': 0,
            'total_dequeued': 0
        }
        _dlq[queue_name] = deque(maxlen=1000)  # Keep last 1000 DLQ messages
    
    return _queues[queue_name]


async def _schedule_delayed_enqueue(queue_name: str, message: Any, delay: int = None, 
                                   schedule_at: str = None, message_id: str = None):
    """Schedule a delayed enqueue using the scheduler."""
    injector = get_injector()
    
    # Determine schedule
    if delay:
        schedule_time = (datetime.now() + timedelta(seconds=delay)).isoformat()
        schedule_type = 'once'
    elif schedule_at:
        schedule_time = schedule_at
        schedule_type = 'once'
    else:
        return None
    
    # Create scheduled job to enqueue message
    result = await injector.run('scheduler', {
        'operation': 'schedule',
        'job_name': f"delayed_enqueue_{message_id or uuid.uuid4().hex[:8]}",
        'schedule_type': schedule_type,
        'schedule': schedule_time,
        'agentool_name': 'queue',
        'input_data': {
            'operation': 'enqueue',
            'queue_name': queue_name,
            'message': message,
            'message_id': message_id
        }
    })
    
    return result


async def _execute_message_as_agentool(message: Dict[str, Any], target_agentool: str = None):
    """Execute a message as an AgenTool call (reactive pattern)."""
    injector = get_injector()
    
    # Determine target AgenTool
    agentool_name = target_agentool or message.get('agentool_name')
    if not agentool_name:
        raise ValueError("No target AgenTool specified for auto-execution")
    
    # Extract input data
    input_data = message.get('input_data', message)
    
    # Execute via scheduler for monitoring
    result = await injector.run('scheduler', {
        'operation': 'run_now',
        'agentool_name': agentool_name,
        'input_data': input_data
    })
    
    return result


# Tool implementations

async def queue_enqueue(ctx: RunContext[Any], queue_name: str, message: Any,
                       message_id: Optional[str], delay: Optional[int],
                       schedule_at: Optional[str], max_size: Optional[int]) -> QueueOutput:
    """Enqueue a message."""
    try:
        # Generate message ID if not provided
        if not message_id:
            message_id = f"msg_{uuid.uuid4().hex[:8]}"
        
        # Handle delayed enqueue via scheduler
        if delay or schedule_at:
            result = await _schedule_delayed_enqueue(
                queue_name, message, delay, schedule_at, message_id
            )
            
            # Check if scheduling was successful
            # The result is an AgentRunResult, parse the output
            if result and hasattr(result, 'output'):
                try:
                    schedule_data = json.loads(result.output)
                    # If we got here without exception, scheduling succeeded
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"Failed to parse scheduler response: {e}") from e
            else:
                raise RuntimeError("Scheduler returned no result")
            
            return QueueOutput(
                operation='enqueue',
                queue_name=queue_name,
                message=f"Message scheduled for delayed enqueue",
                data={
                    'message_id': message_id,
                    'scheduled_at': schedule_data.get('data', {}).get('next_run'),
                    'job_id': schedule_data.get('data', {}).get('job_id')
                }
            )
        
        # Regular immediate enqueue
        queue = _get_or_create_queue(queue_name, max_size)
        
        # Create message wrapper
        wrapped_message = {
            'id': message_id,
            'payload': message,
            'timestamp': datetime.now().isoformat(),
            'queue_name': queue_name,
            'retry_count': 0
        }
        
        # Store in registry
        _message_registry[message_id] = wrapped_message
        
        # Enqueue
        await queue.put(wrapped_message)
        
        # Update metadata
        _queue_metadata[queue_name]['total_enqueued'] += 1
        
        return QueueOutput(
            operation='enqueue',
            queue_name=queue_name,
            message=f"Message enqueued successfully",
            data={
                'message_id': message_id,
                'queue_size': queue.qsize()
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to enqueue message to queue '{queue_name}': {e}") from e


async def queue_dequeue(ctx: RunContext[Any], queue_name: str, timeout: Optional[float],
                       auto_execute: bool, target_agentool: Optional[str],
                       max_retries: int) -> QueueOutput:
    """Dequeue a message."""
    try:
        queue = _get_or_create_queue(queue_name)
        
        # Dequeue with optional timeout
        if timeout:
            wrapped_message = await asyncio.wait_for(queue.get(), timeout=timeout)
        else:
            wrapped_message = await queue.get()
        
        # Update metadata
        _queue_metadata[queue_name]['total_dequeued'] += 1
        
        # Handle auto-execution (reactive pattern)
        if auto_execute:
            try:
                execution_result = await _execute_message_as_agentool(
                    wrapped_message['payload'], target_agentool
                )
                
                # If we got here without exception, the execution succeeded
                # The scheduler would have raised an exception if the execution failed
                
                return QueueOutput(
                    operation='dequeue',
                    queue_name=queue_name,
                    message=f"Message dequeued and executed",
                    data={
                        'message_id': wrapped_message['id'],
                        'message': wrapped_message['payload'],
                        'executed': True,
                        'execution_result': execution_result.output if hasattr(execution_result, 'output') else None
                    }
                )
                
            except Exception as exec_error:
                # Handle execution failure
                wrapped_message['retry_count'] += 1
                
                if wrapped_message['retry_count'] >= max_retries:
                    # Move to DLQ
                    _dlq[queue_name].append(wrapped_message)
                    raise RuntimeError(
                        f"Message {wrapped_message['id']} execution failed after {max_retries} retries, moved to DLQ: {exec_error}"
                    ) from exec_error
                else:
                    # Re-enqueue for retry
                    await queue.put(wrapped_message)
                    raise RuntimeError(
                        f"Message {wrapped_message['id']} execution failed (retry {wrapped_message['retry_count']}/{max_retries}): {exec_error}"
                    ) from exec_error
        
        # Normal dequeue without execution
        return QueueOutput(
            operation='dequeue',
            queue_name=queue_name,
            message=f"Message dequeued successfully",
            data={
                'message_id': wrapped_message['id'],
                'message': wrapped_message['payload'],
                'timestamp': wrapped_message['timestamp']
            }
        )
        
    except asyncio.TimeoutError as e:
        raise TimeoutError(f"No messages available in queue '{queue_name}' within {timeout}s timeout") from e
    except RuntimeError:
        raise  # Re-raise our own RuntimeErrors
    except Exception as e:
        raise RuntimeError(f"Failed to dequeue message from queue '{queue_name}': {e}") from e


async def queue_peek(ctx: RunContext[Any], queue_name: str) -> QueueOutput:
    """Peek at the next message without removing it."""
    try:
        queue = _get_or_create_queue(queue_name)
        
        if queue.empty():
            return QueueOutput(
                operation='peek',
                queue_name=queue_name,
                message="Queue is empty",
                data={'empty': True}
            )
        
        # Get internal queue data (implementation specific)
        # This is a workaround since asyncio.Queue doesn't have peek
        items = list(queue._queue)  # Access internal deque
        
        if items:
            next_message = items[0]
            return QueueOutput(
                operation='peek',
                queue_name=queue_name,
                message="Next message retrieved",
                data={
                    'message_id': next_message['id'],
                    'message': next_message['payload'],
                    'timestamp': next_message['timestamp']
                }
            )
        else:
            return QueueOutput(
                operation='peek',
                queue_name=queue_name,
                message="Queue is empty",
                data={'empty': True}
            )
        
    except Exception as e:
        raise RuntimeError(f"Failed to peek message in queue '{queue_name}': {e}") from e


async def queue_size(ctx: RunContext[Any], queue_name: str) -> QueueOutput:
    """Get the size of a queue."""
    try:
        queue = _get_or_create_queue(queue_name)
        size = queue.qsize()
        
        metadata = _queue_metadata.get(queue_name, {})
        
        return QueueOutput(
            operation='size',
            queue_name=queue_name,
            message=f"Queue has {size} messages",
            data={
                'size': size,
                'max_size': metadata.get('max_size', 0),
                'total_enqueued': metadata.get('total_enqueued', 0),
                'total_dequeued': metadata.get('total_dequeued', 0),
                'dlq_size': len(_dlq.get(queue_name, []))
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to get size of queue '{queue_name}': {e}") from e


async def queue_clear(ctx: RunContext[Any], queue_name: str) -> QueueOutput:
    """Clear all messages from a queue."""
    try:
        queue = _get_or_create_queue(queue_name)
        
        # Count messages before clearing
        count = queue.qsize()
        
        # Clear queue
        while not queue.empty():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        return QueueOutput(
            operation='clear',
            queue_name=queue_name,
            message=f"Cleared {count} messages from queue",
            data={'cleared_count': count}
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to clear queue '{queue_name}': {e}") from e


async def queue_list(ctx: RunContext[Any]) -> QueueOutput:
    """List all queues."""
    try:
        queues_info = []
        
        for queue_name, queue in _queues.items():
            metadata = _queue_metadata.get(queue_name, {})
            queues_info.append({
                'name': queue_name,
                'size': queue.qsize(),
                'max_size': metadata.get('max_size', 0),
                'created_at': metadata.get('created_at'),
                'total_enqueued': metadata.get('total_enqueued', 0),
                'total_dequeued': metadata.get('total_dequeued', 0),
                'dlq_size': len(_dlq.get(queue_name, []))
            })
        
        return QueueOutput(
            operation='list_queues',
            queue_name='*',
            message=f"Found {len(queues_info)} queues",
            data={
                'queues': queues_info,
                'count': len(queues_info)
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to list queues: {e}") from e


async def queue_get_dlq(ctx: RunContext[Any], queue_name: str) -> QueueOutput:
    """Get messages from dead letter queue."""
    try:
        dlq_messages = list(_dlq.get(queue_name, []))
        
        return QueueOutput(
            operation='get_dlq',
            queue_name=queue_name,
            message=f"Retrieved {len(dlq_messages)} messages from DLQ",
            data={
                'messages': dlq_messages,
                'count': len(dlq_messages)
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to get DLQ messages for queue '{queue_name}': {e}") from e


# Main routing function
async def manage_queue(ctx: RunContext[Any], operation: str, **kwargs) -> QueueOutput:
    """Main routing function for queue operations."""
    
    queue_name = kwargs.get('queue_name', 'default')
    
    if operation == 'enqueue':
        return await queue_enqueue(
            ctx,
            queue_name,
            kwargs['message'],
            kwargs.get('message_id'),
            kwargs.get('delay'),
            kwargs.get('schedule_at'),
            kwargs.get('max_size')
        )
    
    elif operation == 'dequeue':
        return await queue_dequeue(
            ctx,
            queue_name,
            kwargs.get('timeout'),
            kwargs.get('auto_execute', False),
            kwargs.get('target_agentool'),
            kwargs.get('max_retries', 3)
        )
    
    elif operation == 'peek':
        return await queue_peek(ctx, queue_name)
    
    elif operation == 'size':
        return await queue_size(ctx, queue_name)
    
    elif operation == 'clear':
        return await queue_clear(ctx, queue_name)
    
    elif operation == 'list_queues':
        return await queue_list(ctx)
    
    elif operation == 'get_dlq':
        return await queue_get_dlq(ctx, queue_name)
    
    else:
        raise ValueError(f"Unknown queue operation: {operation}")


def create_queue_agent():
    """Create and return the Queue AgenTool agent."""
    
    # Define routing configuration
    routing_config = RoutingConfig(
        operation_field='operation',
        operation_map={
            'enqueue': ('manage_queue', lambda x: {
                'operation': x.operation,
                'queue_name': x.queue_name,
                'message': x.message,
                'message_id': x.message_id,
                'delay': x.delay,
                'schedule_at': x.schedule_at,
                'max_size': x.max_size
            }),
            'dequeue': ('manage_queue', lambda x: {
                'operation': x.operation,
                'queue_name': x.queue_name,
                'timeout': x.timeout,
                'auto_execute': x.auto_execute,
                'target_agentool': x.target_agentool,
                'max_retries': x.max_retries
            }),
            'peek': ('manage_queue', lambda x: {
                'operation': x.operation,
                'queue_name': x.queue_name
            }),
            'size': ('manage_queue', lambda x: {
                'operation': x.operation,
                'queue_name': x.queue_name
            }),
            'clear': ('manage_queue', lambda x: {
                'operation': x.operation,
                'queue_name': x.queue_name
            }),
            'list_queues': ('manage_queue', lambda x: {
                'operation': x.operation
            }),
            'get_dlq': ('manage_queue', lambda x: {
                'operation': x.operation,
                'queue_name': x.queue_name
            })
        }
    )
    
    # Create the AgenTool
    agent = create_agentool(
        name='queue',
        input_schema=QueueInput,
        output_type=QueueOutput,
        routing_config=routing_config,
        tools=[manage_queue],
        system_prompt="Message queue for data bus operations. Routes messages between AgenTools.",
        description="Queue for message passing and reactive workflows in the AgenTool ecosystem",
        version="1.0.0",
        tags=["queue", "messaging", "data-bus", "reactive"],
        dependencies=["scheduler"]  # Uses scheduler for delayed messages and auto-execution
    )
    
    return agent