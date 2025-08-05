"""
Tests for queue toolkit.

This module tests all functionality of the queue toolkit
including enqueue/dequeue operations, delayed messages, DLQ, and auto-execution.
"""

import json
import asyncio
from datetime import datetime, timedelta

from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry


class TestQueue:
    """Test suite for queue toolkit."""
    
    def setup_method(self):
        """Clear registry and injector before each test."""
        AgenToolRegistry.clear()
        get_injector().clear()
        
        # Import and create the agent, clear global state
        from agentoolkit.system.queue import (
            create_queue_agent,
            _queues,
            _queue_metadata,
            _message_registry,
            _dlq
        )
        
        # Clear global state
        _queues.clear()
        _queue_metadata.clear()
        _message_registry.clear()
        _dlq.clear()
        
        # Create both queue and scheduler agents due to cross-dependencies
        from agentoolkit.system.scheduler import create_scheduler_agent
        
        # Initialize both agents
        self.queue_agent = create_queue_agent()
        self.scheduler_agent = create_scheduler_agent()
        
        # Create a simple test AgenTool for auto-execution tests
        from agentool import create_agentool
        from agentool.base import BaseOperationInput
        from agentool.core.registry import RoutingConfig
        from pydantic import BaseModel, Field
        from pydantic_ai import RunContext
        from typing import Literal, Any
        
        class ProcessInput(BaseOperationInput):
            operation: Literal['process']
            data: Any
        
        async def process_data(ctx: RunContext[Any], data: Any):
            return {"processed": data, "timestamp": datetime.now().isoformat()}
        
        process_agent = create_agentool(
            name='processor',
            input_schema=ProcessInput,
            routing_config=RoutingConfig(
                operation_field='operation',
                operation_map={
                    'process': ('process_data', lambda x: {'data': x.data})
                }
            ),
            tools=[process_data]
        )
    
    def test_enqueue_and_dequeue(self):
        """Test basic enqueue and dequeue operations."""
        
        async def run_test():
            injector = get_injector()
            
            # Enqueue a message
            message = {"task": "test", "priority": 1}
            enqueue_result = await injector.run('queue', {
                "operation": "enqueue",
                "queue_name": "test_queue",
                "message": message
            })
            
            if hasattr(enqueue_result, 'output'):
                enqueue_data = json.loads(enqueue_result.output)
            else:
                enqueue_data = enqueue_result
            
            assert enqueue_data['operation'] == 'enqueue'
            assert 'message_id' in enqueue_data['data']
            assert enqueue_data['data']['queue_size'] == 1
            
            message_id = enqueue_data['data']['message_id']
            
            # Dequeue the message
            dequeue_result = await injector.run('queue', {
                "operation": "dequeue",
                "queue_name": "test_queue"
            })
            
            if hasattr(dequeue_result, 'output'):
                dequeue_data = json.loads(dequeue_result.output)
            else:
                dequeue_data = dequeue_result
            
            assert dequeue_data['operation'] == 'dequeue'
            assert dequeue_data['data']['message_id'] == message_id
            assert dequeue_data['data']['message'] == message
        
        asyncio.run(run_test())
    
    def test_multiple_queues(self):
        """Test working with multiple named queues."""
        
        async def run_test():
            injector = get_injector()
            
            # Create messages in different queues
            queues_data = [
                ("queue1", {"type": "A", "value": 1}),
                ("queue2", {"type": "B", "value": 2}),
                ("queue3", {"type": "C", "value": 3})
            ]
            
            for queue_name, message in queues_data:
                await injector.run('queue', {
                    "operation": "enqueue",
                    "queue_name": queue_name,
                    "message": message
                })
            
            # List all queues
            list_result = await injector.run('queue', {
                "operation": "list_queues"
            })
            
            if hasattr(list_result, 'output'):
                list_data = json.loads(list_result.output)
            else:
                list_data = list_result
            
            assert list_data['data']['count'] == 3
            
            queue_names = [q['name'] for q in list_data['data']['queues']]
            assert "queue1" in queue_names
            assert "queue2" in queue_names
            assert "queue3" in queue_names
            
            # Check each queue size
            for queue_name, _ in queues_data:
                size_result = await injector.run('queue', {
                    "operation": "size",
                    "queue_name": queue_name
                })
                
                if hasattr(size_result, 'output'):
                    size_data = json.loads(size_result.output)
                else:
                    size_data = size_result
                
                assert size_data['data']['size'] == 1
        
        asyncio.run(run_test())
    
    def test_peek_operation(self):
        """Test peeking at messages without removing them."""
        
        async def run_test():
            injector = get_injector()
            
            # Enqueue multiple messages
            messages = [
                {"order": 1, "data": "first"},
                {"order": 2, "data": "second"},
                {"order": 3, "data": "third"}
            ]
            
            for msg in messages:
                await injector.run('queue', {
                    "operation": "enqueue",
                    "queue_name": "peek_test",
                    "message": msg
                })
            
            # Peek at the first message
            peek_result = await injector.run('queue', {
                "operation": "peek",
                "queue_name": "peek_test"
            })
            
            if hasattr(peek_result, 'output'):
                peek_data = json.loads(peek_result.output)
            else:
                peek_data = peek_result
            
            assert peek_data['data']['message'] == messages[0]
            
            # Check queue size is unchanged
            size_result = await injector.run('queue', {
                "operation": "size",
                "queue_name": "peek_test"
            })
            
            if hasattr(size_result, 'output'):
                size_data = json.loads(size_result.output)
            else:
                size_data = size_result
            
            assert size_data['data']['size'] == 3
        
        asyncio.run(run_test())
    
    def test_clear_queue(self):
        """Test clearing all messages from a queue."""
        
        async def run_test():
            injector = get_injector()
            
            # Enqueue multiple messages
            for i in range(5):
                await injector.run('queue', {
                    "operation": "enqueue",
                    "queue_name": "clear_test",
                    "message": {"index": i}
                })
            
            # Clear the queue
            clear_result = await injector.run('queue', {
                "operation": "clear",
                "queue_name": "clear_test"
            })
            
            if hasattr(clear_result, 'output'):
                clear_data = json.loads(clear_result.output)
            else:
                clear_data = clear_result
            
            assert clear_data['data']['cleared_count'] == 5
            
            # Check queue is empty
            size_result = await injector.run('queue', {
                "operation": "size",
                "queue_name": "clear_test"
            })
            
            if hasattr(size_result, 'output'):
                size_data = json.loads(size_result.output)
            else:
                size_data = size_result
            
            assert size_data['data']['size'] == 0
        
        asyncio.run(run_test())
    
    def test_dequeue_timeout(self):
        """Test dequeue with timeout on empty queue."""
        
        async def run_test():
            injector = get_injector()
            
            # Try to dequeue from empty queue with timeout
            try:
                dequeue_result = await injector.run('queue', {
                    "operation": "dequeue",
                    "queue_name": "empty_queue",
                    "timeout": 0.5  # 500ms timeout
                })
                # Should raise TimeoutError
                assert False, "Expected TimeoutError for dequeue on empty queue"
            except TimeoutError as e:
                assert 'empty_queue' in str(e)
                assert 'timeout' in str(e).lower()
        
        asyncio.run(run_test())
    
    def test_fifo_ordering(self):
        """Test that queue maintains FIFO ordering."""
        
        async def run_test():
            injector = get_injector()
            
            # Enqueue messages in order
            messages = []
            for i in range(10):
                msg = {"sequence": i, "value": f"msg_{i}"}
                messages.append(msg)
                await injector.run('queue', {
                    "operation": "enqueue",
                    "queue_name": "fifo_test",
                    "message": msg
                })
            
            # Dequeue all messages and verify order
            for i in range(10):
                dequeue_result = await injector.run('queue', {
                    "operation": "dequeue",
                    "queue_name": "fifo_test"
                })
                
                if hasattr(dequeue_result, 'output'):
                    dequeue_data = json.loads(dequeue_result.output)
                else:
                    dequeue_data = dequeue_result
                
                assert dequeue_data['data']['message'] == messages[i]
        
        asyncio.run(run_test())
    
    def test_auto_execution(self):
        """Test auto-execution of messages as AgenTool calls."""
        
        async def run_test():
            injector = get_injector()
            
            # Scheduler agent already initialized in setup_method
            
            # Enqueue a message for auto-execution
            await injector.run('queue', {
                "operation": "enqueue",
                "queue_name": "auto_exec_test",
                "message": {
                    "operation": "process",
                    "data": {"test": "auto_exec"}
                }
            })
            
            # Dequeue with auto-execution
            dequeue_result = await injector.run('queue', {
                "operation": "dequeue",
                "queue_name": "auto_exec_test",
                "auto_execute": True,
                "target_agentool": "processor"
            })
            
            if hasattr(dequeue_result, 'output'):
                dequeue_data = json.loads(dequeue_result.output)
            else:
                dequeue_data = dequeue_result
            
            assert dequeue_data['data']['executed'] is True
        
        asyncio.run(run_test())
    
    def test_delayed_enqueue(self):
        """Test delayed message enqueue using scheduler integration."""
        
        async def run_test():
            injector = get_injector()
            
            # Need scheduler for delayed enqueue
            from agentoolkit.system.scheduler import create_scheduler_agent
            scheduler_agent = create_scheduler_agent()
            
            # Enqueue with delay
            enqueue_result = await injector.run('queue', {
                "operation": "enqueue",
                "queue_name": "delayed_test",
                "message": {"delayed": True},
                "delay": 2  # 2 seconds delay
            })
            
            if hasattr(enqueue_result, 'output'):
                enqueue_data = json.loads(enqueue_result.output)
            else:
                enqueue_data = enqueue_result
            
            assert 'scheduled_at' in enqueue_data['data']
            assert 'job_id' in enqueue_data['data']
        
        asyncio.run(run_test())
    
    def test_dlq_handling(self):
        """Test dead letter queue handling."""
        
        async def run_test():
            injector = get_injector()
            
            # Need scheduler for auto-execution
            from agentoolkit.system.scheduler import create_scheduler_agent
            scheduler_agent = create_scheduler_agent()
            
            # Create a failing processor
            from agentool import create_agentool
            from agentool.base import BaseOperationInput
            from agentool.core.registry import RoutingConfig
            from pydantic_ai import RunContext
            from typing import Literal, Any
            
            class FailInput(BaseOperationInput):
                operation: Literal['fail']
                data: Any
            
            async def fail_process(ctx: RunContext[Any], data: Any):
                raise ValueError("Intentional failure for DLQ test")
            
            fail_agent = create_agentool(
                name='fail_processor',
                input_schema=FailInput,
                routing_config=RoutingConfig(
                    operation_field='operation',
                    operation_map={
                        'fail': ('fail_process', lambda x: {'data': x.data})
                    }
                ),
                tools=[fail_process]
            )
            
            # Enqueue a message that will fail
            await injector.run('queue', {
                "operation": "enqueue",
                "queue_name": "dlq_test",
                "message": {
                    "operation": "fail",
                    "data": {"test": "dlq"}
                }
            })
            
            # First dequeue with auto-execution (will fail and move to DLQ after max_retries=1)
            try:
                dequeue_result = await injector.run('queue', {
                    "operation": "dequeue",
                    "queue_name": "dlq_test",
                    "auto_execute": True,
                    "target_agentool": "fail_processor",
                    "max_retries": 1  # Move to DLQ after 1 attempt (retry_count starts at 0)
                })
                # Should raise RuntimeError because message moved to DLQ
                assert False, "Expected RuntimeError for failed auto-execution"
            except RuntimeError as e:
                assert 'moved to DLQ' in str(e)
                assert 'Intentional failure' in str(e)
            
            # Check DLQ
            dlq_result = await injector.run('queue', {
                "operation": "get_dlq",
                "queue_name": "dlq_test"
            })
            
            if hasattr(dlq_result, 'output'):
                dlq_data = json.loads(dlq_result.output)
            else:
                dlq_data = dlq_result
            
            assert dlq_data['data']['count'] == 1
        
        asyncio.run(run_test())
    
    def test_queue_metadata(self):
        """Test queue metadata tracking."""
        
        async def run_test():
            injector = get_injector()
            
            # Enqueue and dequeue multiple messages
            for i in range(5):
                await injector.run('queue', {
                    "operation": "enqueue",
                    "queue_name": "metadata_test",
                    "message": {"index": i}
                })
            
            # Dequeue 3 messages
            for i in range(3):
                await injector.run('queue', {
                    "operation": "dequeue",
                    "queue_name": "metadata_test"
                })
            
            # Check queue size and metadata
            size_result = await injector.run('queue', {
                "operation": "size",
                "queue_name": "metadata_test"
            })
            
            if hasattr(size_result, 'output'):
                size_data = json.loads(size_result.output)
            else:
                size_data = size_result
            
            assert size_data['data']['size'] == 2
            assert size_data['data']['total_enqueued'] == 5
            assert size_data['data']['total_dequeued'] == 3
        
        asyncio.run(run_test())
    
    def test_empty_queue_peek(self):
        """Test peeking at an empty queue."""
        
        async def run_test():
            injector = get_injector()
            
            # Peek at empty queue
            peek_result = await injector.run('queue', {
                "operation": "peek",
                "queue_name": "empty_peek_test"
            })
            
            if hasattr(peek_result, 'output'):
                peek_data = json.loads(peek_result.output)
            else:
                peek_data = peek_result
            
            assert peek_data['data']['empty'] is True
        
        asyncio.run(run_test())