"""
Tests for scheduler toolkit.

This module tests all functionality of the scheduler toolkit
including job scheduling, execution, management, and integration with AgenTools.
"""

import json
import asyncio
from datetime import datetime, timedelta

from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry


class TestScheduler:
    """Test suite for scheduler toolkit."""
    
    def setup_method(self):
        """Clear registry and injector before each test."""
        AgenToolRegistry.clear()
        get_injector().clear()
        
        # Import and clear global state
        import agentoolkit.system.scheduler as scheduler_module
        
        # Reset scheduler state
        scheduler_module._scheduler = None
        scheduler_module._scheduler_running = False
        scheduler_module._job_registry.clear()
        scheduler_module._execution_history.clear()
        
        # Create all required agents in dependency order
        from agentoolkit.storage.fs import create_storage_fs_agent
        from agentoolkit.storage.kv import create_storage_kv_agent, _kv_storage, _kv_expiry
        from agentoolkit.observability.metrics import create_metrics_agent
        from agentoolkit.system.logging import create_logging_agent, _logging_config
        from agentoolkit.system.scheduler import create_scheduler_agent
        from agentoolkit.system.queue import create_queue_agent
        
        # Clear additional global state
        _kv_storage.clear()
        _kv_expiry.clear()
        _logging_config.clear()
        
        # Initialize agents in dependency order
        self.storage_fs_agent = create_storage_fs_agent()  # No dependencies
        self.storage_kv_agent = create_storage_kv_agent()  # No dependencies
        self.metrics_agent = create_metrics_agent()        # Depends on storage_kv
        self.logging_agent = create_logging_agent()        # Depends on storage_fs, metrics
        self.scheduler_agent = create_scheduler_agent()    # Depends on queue, logging
        self.queue_agent = create_queue_agent()            # Depends on scheduler
        
        # Create a simple test AgenTool for execution
        from agentool import create_agentool
        from agentool.base import BaseOperationInput
        from agentool.core.registry import RoutingConfig
        from pydantic import BaseModel, Field
        from pydantic_ai import RunContext
        from typing import Literal, Any
        
        class TestInput(BaseOperationInput):
            operation: Literal['echo', 'error']
            message: str = Field(default="test")
        
        async def echo_message(ctx: RunContext[Any], message: str):
            return {"echo": message, "timestamp": datetime.now().isoformat()}
        
        async def raise_error(ctx: RunContext[Any], message: str):
            raise ValueError(f"Test error: {message}")
        
        test_agent = create_agentool(
            name='test_tool',
            input_schema=TestInput,
            routing_config=RoutingConfig(
                operation_field='operation',
                operation_map={
                    'echo': ('echo_message', lambda x: {'message': x.message}),
                    'error': ('raise_error', lambda x: {'message': x.message})
                }
            ),
            tools=[echo_message, raise_error]
        )
    
    def test_scheduler_status(self):
        """Test scheduler status operation."""
        
        async def run_test():
            injector = get_injector()
            
            # Check scheduler status
            result = await injector.run('scheduler', {
                "operation": "status"
            })
            
            # scheduler returns typed SchedulerOutput
            assert result.success is True
            assert result.operation == 'status'
            assert 'running' in result.data
            assert 'jobs_count' in result.data
        
        asyncio.run(run_test())
    
    def test_schedule_once_job(self):
        """Test scheduling a one-time job."""
        
        async def run_test():
            injector = get_injector()
            
            # Schedule a one-time job
            schedule_time = (datetime.now() + timedelta(seconds=2)).isoformat()
            
            result = await injector.run('scheduler', {
                "operation": "schedule",
                "job_name": "test_once_job",
                "schedule_type": "once",
                "schedule": schedule_time,
                "agentool_name": "test_tool",
                "input_data": {"operation": "echo", "message": "once job"}
            })
            
            # scheduler returns typed SchedulerOutput
            assert result.success is True
            assert result.operation == 'schedule'
            assert 'job_id' in result.data
            assert 'next_run' in result.data
            
            job_id = result.data['job_id']
            
            # Get job details
            job_result = await injector.run('scheduler', {
                "operation": "get_job",
                "job_id": job_id
            })
            
            # scheduler returns typed SchedulerOutput
            assert job_result.success is True
            assert job_result.operation == 'get_job'
            assert job_result.data['job_id'] == job_id
            assert job_result.data['job_name'] == "test_once_job"
            
            # Cancel the job
            cancel_result = await injector.run('scheduler', {
                "operation": "cancel",
                "job_id": job_id
            })
            
            # scheduler returns typed SchedulerOutput
            assert cancel_result.success is True
            assert cancel_result.operation == 'cancel'
        
        asyncio.run(run_test())
    
    def test_schedule_interval_job(self):
        """Test scheduling an interval job."""
        
        async def run_test():
            injector = get_injector()
            
            # Schedule an interval job
            result = await injector.run('scheduler', {
                "operation": "schedule",
                "job_name": "test_interval_job",
                "schedule_type": "interval",
                "schedule": "5 seconds",
                "agentool_name": "test_tool",
                "input_data": {"operation": "echo", "message": "interval job"}
            })
            
            # scheduler returns typed SchedulerOutput
            assert result.success is True
            assert result.operation == 'schedule'
            job_id = result.data['job_id']
            
            # List all jobs
            list_result = await injector.run('scheduler', {
                "operation": "list"
            })
            
            # scheduler returns typed SchedulerOutput
            assert list_result.success is True
            assert list_result.operation == 'list'
            assert list_result.data['count'] > 0
            job_ids = [job['job_id'] for job in list_result.data['jobs']]
            assert job_id in job_ids
            
            # Cancel the job
            await injector.run('scheduler', {
                "operation": "cancel",
                "job_id": job_id
            })
        
        asyncio.run(run_test())
    
    def test_schedule_cron_job(self):
        """Test scheduling a cron job."""
        
        async def run_test():
            injector = get_injector()
            
            # Schedule a cron job (every minute)
            result = await injector.run('scheduler', {
                "operation": "schedule",
                "job_name": "test_cron_job",
                "schedule_type": "cron",
                "schedule": "* * * * *",  # Every minute
                "agentool_name": "test_tool",
                "input_data": {"operation": "echo", "message": "cron job"}
            })
            
            # scheduler returns typed SchedulerOutput
            assert result.success is True
            assert result.operation == 'schedule'
            job_id = result.data['job_id']
            
            # Cancel the job
            await injector.run('scheduler', {
                "operation": "cancel",
                "job_id": job_id
            })
        
        asyncio.run(run_test())
    
    def test_run_now(self):
        """Test immediate execution of an AgenTool."""
        
        async def run_test():
            injector = get_injector()
            
            # Run immediately
            result = await injector.run('scheduler', {
                "operation": "run_now",
                "agentool_name": "test_tool",
                "input_data": {"operation": "echo", "message": "immediate execution"}
            })
            
            # scheduler returns typed SchedulerOutput
            assert result.success is True
            assert result.operation == 'run_now'
            assert 'job_id' in result.data
            assert 'result' in result.data
        
        asyncio.run(run_test())
    
    def test_pause_and_resume_job(self):
        """Test pausing and resuming a job."""
        
        async def run_test():
            injector = get_injector()
            
            # Schedule a job
            result = await injector.run('scheduler', {
                "operation": "schedule",
                "job_name": "test_pause_job",
                "schedule_type": "interval",
                "schedule": "10 seconds",
                "agentool_name": "test_tool",
                "input_data": {"operation": "echo", "message": "pause test"}
            })
            
            # scheduler returns typed SchedulerOutput
            assert result.success is True
            job_id = result.data['job_id']
            
            # Pause the job
            pause_result = await injector.run('scheduler', {
                "operation": "pause",
                "job_id": job_id
            })
            
            # scheduler returns typed SchedulerOutput
            assert pause_result.success is True
            assert pause_result.operation == 'pause'
            
            # Resume the job
            resume_result = await injector.run('scheduler', {
                "operation": "resume",
                "job_id": job_id
            })
            
            # scheduler returns typed SchedulerOutput
            assert resume_result.success is True
            assert resume_result.operation == 'resume'
            
            # Cancel the job
            await injector.run('scheduler', {
                "operation": "cancel",
                "job_id": job_id
            })
        
        asyncio.run(run_test())
    
    def test_error_handling(self):
        """Test error handling in scheduled jobs."""
        
        async def run_test():
            injector = get_injector()
            
            # Schedule a job that will error (should now raise exception)
            try:
                result = await injector.run('scheduler', {
                    "operation": "run_now",
                    "agentool_name": "test_tool",
                    "input_data": {"operation": "error", "message": "test error"}
                })
                # Should not reach here
                assert False, "Expected RuntimeError for failed AgenTool execution"
            except RuntimeError as e:
                assert "Failed to execute AgenTool" in str(e)
                print(f"\n   Expected exception caught: {e}")
        
        asyncio.run(run_test())
    
    def test_invalid_schedule(self):
        """Test invalid schedule handling."""
        
        async def run_test():
            injector = get_injector()
            
            # Try to schedule with invalid schedule type 
            # Pydantic validation error is caught and returned in output
            result = await injector.run('scheduler', {
                "operation": "schedule",
                "job_name": "invalid_job",
                "schedule_type": "invalid",
                "schedule": "invalid",
                "agentool_name": "test_tool",
                "input_data": {"operation": "echo", "message": "test"}
            })
            
            # Check that validation error is in the output
            assert hasattr(result, 'output')
            assert "validation error" in result.output.lower()
            assert "schedule_type" in result.output.lower()
            assert "Input should be 'cron', 'interval' or 'once'" in result.output
            print(f"\n   Expected validation error in output: {result.output}")
        
        asyncio.run(run_test())
    
    def test_job_with_queues(self):
        """Test job execution with result and error queues."""
        
        async def run_test():
            injector = get_injector()
            
            # Schedule a job with result queue
            result = await injector.run('scheduler', {
                "operation": "run_now",
                "agentool_name": "test_tool",
                "input_data": {"operation": "echo", "message": "queue test"},
                "result_queue": "test_results",
                "error_queue": "test_errors"
            })
            
            # scheduler returns typed SchedulerOutput
            assert result.success is True
            assert result.operation == 'run_now'
            
            # Check if result was sent to queue
            # (Would need queue to be available to fully test)
            assert 'job_id' in result.data
        
        asyncio.run(run_test())
    
    def test_get_job_not_found(self):
        """Test get_job operation when job doesn't exist."""
        
        async def run_test():
            injector = get_injector()
            
            # Try to get a non-existent job
            result = await injector.run('scheduler', {
                "operation": "get_job",
                "job_id": "non_existent_job_123"
            })
            
            # scheduler returns typed SchedulerOutput with success=False
            assert result.success is False
            assert result.operation == 'get_job'
            assert "not found" in result.message.lower()
            assert result.data is None
        
        asyncio.run(run_test())
    
    def test_logging_integration(self):
        """Test that scheduler logs events properly."""
        
        async def run_test():
            injector = get_injector()
            
            # Configure logging to capture scheduler logs
            config_result = await injector.run('logging', {
                "operation": "configure",
                "logger_name": "scheduler",
                "output": "both",
                "file_path": "/tmp/scheduler_test.log",
                "format": "json",
                "level": "DEBUG"  # Changed from min_level to level
            })
            
            assert config_result.success is True
            
            # Schedule a job with future time - this should trigger logging
            future_time = (datetime.now() + timedelta(seconds=10)).isoformat()
            schedule_result = await injector.run('scheduler', {
                "operation": "schedule",
                "job_name": "test_logging_job",
                "schedule_type": "once",
                "schedule": future_time,
                "agentool_name": "test_tool",
                "input_data": {"operation": "echo", "message": "logging test"}
            })
            
            assert schedule_result.success is True
            job_id = schedule_result.data['job_id']
            
            # Wait a moment for async logging to complete
            await asyncio.sleep(0.1)
            
            # Run the job immediately - this should trigger execution logging
            run_result = await injector.run('scheduler', {
                "operation": "run_now",
                "agentool_name": "test_tool",
                "input_data": {"operation": "echo", "message": "immediate logging test"}
            })
            
            assert run_result.success is True
            
            # Cancel the scheduled job - this should trigger cancellation logging
            cancel_result = await injector.run('scheduler', {
                "operation": "cancel",
                "job_id": job_id
            })
            
            assert cancel_result.success is True
            
            # Get logs from the scheduler logger to verify events were logged
            logs_result = await injector.run('logging', {
                "operation": "get_logs",
                "logger_name": "scheduler",
                "file_path": "/tmp/scheduler_test.log"
            })
            
            # Verify that logs were created (logging returns typed LoggingOutput)
            assert logs_result.success is True
            assert logs_result.data['count'] > 0
            
            # Check that specific events were logged
            log_entries = logs_result.data.get('entries', [])
            
            # Parse log messages - they should be in JSON format
            log_messages = []
            for entry in log_entries:
                if isinstance(entry, str):
                    # Try to parse JSON log entry
                    try:
                        import json
                        log_obj = json.loads(entry)
                        log_messages.append(log_obj.get('message', ''))
                    except:
                        log_messages.append(entry)
                elif isinstance(entry, dict):
                    log_messages.append(entry.get('message', ''))
                else:
                    log_messages.append(str(entry))
            
            # Print logs for debugging
            print(f"\nCaptured {len(log_messages)} log messages:")
            for msg in log_messages:
                print(f"  - {msg}")
            
            # Verify key log messages are present
            assert any('scheduled successfully' in msg for msg in log_messages), f"Should log job scheduling. Got messages: {log_messages}"
            assert any('Executing job' in msg for msg in log_messages), f"Should log job execution. Got messages: {log_messages}"
            assert any('cancelled' in msg for msg in log_messages), f"Should log job cancellation. Got messages: {log_messages}"
        
        asyncio.run(run_test())