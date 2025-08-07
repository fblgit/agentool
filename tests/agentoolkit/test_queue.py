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
        
        # Create all required agents in dependency order
        from agentoolkit.storage.fs import create_storage_fs_agent
        from agentoolkit.storage.kv import create_storage_kv_agent, _kv_storage, _kv_expiry
        from agentoolkit.observability.metrics import create_metrics_agent
        from agentoolkit.system.logging import create_logging_agent, _logging_config
        from agentoolkit.system.scheduler import create_scheduler_agent
        
        # Clear additional global state
        _kv_storage.clear()
        _kv_expiry.clear()
        _logging_config.clear()
        
        # Initialize agents in dependency order
        self.storage_fs_agent = create_storage_fs_agent()  # No dependencies
        self.storage_kv_agent = create_storage_kv_agent()  # No dependencies
        self.metrics_agent = create_metrics_agent()        # Depends on storage_kv
        self.logging_agent = create_logging_agent()        # Depends on storage_fs, metrics
        self.scheduler_agent = create_scheduler_agent()    # Depends on logging
        self.queue_agent = create_queue_agent()            # Depends on scheduler
        
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
            
            # queue returns typed QueueOutput
            assert enqueue_result.success is True
            assert enqueue_result.operation == 'enqueue'
            assert 'message_id' in enqueue_result.data
            assert enqueue_result.data['queue_size'] == 1
            
            message_id = enqueue_result.data['message_id']
            
            # Dequeue the message
            dequeue_result = await injector.run('queue', {
                "operation": "dequeue",
                "queue_name": "test_queue"
            })
            
            # queue returns typed QueueOutput
            assert dequeue_result.success is True
            assert dequeue_result.operation == 'dequeue'
            assert dequeue_result.data['message_id'] == message_id
            assert dequeue_result.data['message'] == message
        
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
            
            # queue returns typed QueueOutput
            assert list_result.success is True
            assert list_result.data['count'] == 3
            
            queue_names = [q['name'] for q in list_result.data['queues']]
            assert "queue1" in queue_names
            assert "queue2" in queue_names
            assert "queue3" in queue_names
            
            # Check each queue size
            for queue_name, _ in queues_data:
                size_result = await injector.run('queue', {
                    "operation": "size",
                    "queue_name": queue_name
                })
                
                # queue returns typed QueueOutput
                assert size_result.success is True
                assert size_result.data['size'] == 1
        
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
            
            # queue returns typed QueueOutput
            assert peek_result.success is True
            assert peek_result.data['message'] == messages[0]
            
            # Check queue size is unchanged
            size_result = await injector.run('queue', {
                "operation": "size",
                "queue_name": "peek_test"
            })
            
            # queue returns typed QueueOutput
            assert size_result.success is True
            assert size_result.data['size'] == 3
        
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
            
            # queue returns typed QueueOutput
            assert clear_result.success is True
            assert clear_result.data['cleared_count'] == 5
            
            # Check queue is empty
            size_result = await injector.run('queue', {
                "operation": "size",
                "queue_name": "clear_test"
            })
            
            # queue returns typed QueueOutput
            assert size_result.success is True
            assert size_result.data['size'] == 0
        
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
                
                # queue returns typed QueueOutput
                assert dequeue_result.success is True
                assert dequeue_result.data['message'] == messages[i]
        
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
            
            # queue returns typed QueueOutput
            assert dequeue_result.success is True
            assert dequeue_result.data['executed'] is True
        
        asyncio.run(run_test())
    
    def test_delayed_enqueue(self):
        """Test delayed message enqueue using scheduler integration."""
        
        async def run_test():
            injector = get_injector()
            
            # Scheduler already initialized in setup_method
            
            # Enqueue with delay
            enqueue_result = await injector.run('queue', {
                "operation": "enqueue",
                "queue_name": "delayed_test",
                "message": {"delayed": True},
                "delay": 2  # 2 seconds delay
            })
            
            # queue returns typed QueueOutput
            assert enqueue_result.success is True
            assert 'scheduled_at' in enqueue_result.data
            assert 'job_id' in enqueue_result.data
        
        asyncio.run(run_test())
    
    def test_dlq_handling(self):
        """Test dead letter queue handling."""
        
        async def run_test():
            injector = get_injector()
            
            # Scheduler already initialized in setup_method
            
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
            
            # queue returns typed QueueOutput
            assert dlq_result.success is True
            assert dlq_result.data['count'] == 1
        
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
            
            # queue returns typed QueueOutput
            assert size_result.success is True
            assert size_result.data['size'] == 2
            assert size_result.data['total_enqueued'] == 5
            assert size_result.data['total_dequeued'] == 3
        
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
            
            # queue returns typed QueueOutput
            assert peek_result.success is True
            assert peek_result.data['empty'] is True
        
        asyncio.run(run_test())