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
        
        # Create both scheduler and queue agents due to cross-dependencies
        from agentoolkit.system.scheduler import create_scheduler_agent
        from agentoolkit.system.queue import create_queue_agent
        
        # Initialize both agents
        self.scheduler_agent = create_scheduler_agent()
        self.queue_agent = create_queue_agent()
        
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
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert 'operation' in data
            assert data['operation'] == 'status'
            assert 'running' in data['data']
            assert 'jobs_count' in data['data']
        
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
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert 'operation' in data
            assert data['operation'] == 'schedule'
            assert 'job_id' in data['data']
            assert 'next_run' in data['data']
            
            job_id = data['data']['job_id']
            
            # Get job details
            job_result = await injector.run('scheduler', {
                "operation": "get_job",
                "job_id": job_id
            })
            
            if hasattr(job_result, 'output'):
                job_data = json.loads(job_result.output)
            else:
                job_data = job_result
            
            assert 'operation' in job_data
            assert job_data['data']['job_id'] == job_id
            assert job_data['data']['job_name'] == "test_once_job"
            
            # Cancel the job
            cancel_result = await injector.run('scheduler', {
                "operation": "cancel",
                "job_id": job_id
            })
            
            if hasattr(cancel_result, 'output'):
                cancel_data = json.loads(cancel_result.output)
            else:
                cancel_data = cancel_result
            
            assert 'operation' in cancel_data
        
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
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert 'operation' in data
            job_id = data['data']['job_id']
            
            # List all jobs
            list_result = await injector.run('scheduler', {
                "operation": "list"
            })
            
            if hasattr(list_result, 'output'):
                list_data = json.loads(list_result.output)
            else:
                list_data = list_result
            
            assert 'operation' in list_data
            assert list_data['data']['count'] > 0
            job_ids = [job['job_id'] for job in list_data['data']['jobs']]
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
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert 'operation' in data
            job_id = data['data']['job_id']
            
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
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert 'operation' in data
            assert data['operation'] == 'run_now'
            assert 'job_id' in data['data']
            assert 'result' in data['data']
        
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
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            job_id = data['data']['job_id']
            
            # Pause the job
            pause_result = await injector.run('scheduler', {
                "operation": "pause",
                "job_id": job_id
            })
            
            if hasattr(pause_result, 'output'):
                pause_data = json.loads(pause_result.output)
            else:
                pause_data = pause_result
            
            assert 'operation' in pause_data
            assert pause_data['operation'] == 'pause'
            
            # Resume the job
            resume_result = await injector.run('scheduler', {
                "operation": "resume",
                "job_id": job_id
            })
            
            if hasattr(resume_result, 'output'):
                resume_data = json.loads(resume_result.output)
            else:
                resume_data = resume_result
            
            assert 'operation' in resume_data
            assert resume_data['operation'] == 'resume'
            
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
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            # Check if result was sent to queue
            # (Would need queue to be available to fully test)
            assert 'job_id' in data['data']
        
        asyncio.run(run_test())