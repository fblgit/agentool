"""
Scheduler AgenTool - Execution engine for scheduled AgenTool runs.

This toolkit provides scheduling capabilities using APScheduler, serving as the
execution engine for the AgenTool ecosystem. It only executes injector.run() calls,
creating a clean execution bus architecture.

Features:
- Cron, interval, and one-time scheduling
- Direct AgenTool execution via injector
- Job persistence and management
- Event monitoring and reactive execution
- Integration with queue for data bus

Example Usage:
    >>> from agentoolkit.system import create_scheduler_agent
    >>> from agentool.core.injector import get_injector
    >>> 
    >>> # Create and register the agent
    >>> agent = create_scheduler_agent()
    >>> 
    >>> # Schedule an AgenTool to run every hour
    >>> injector = get_injector()
    >>> result = await injector.run('scheduler', {
    ...     "operation": "schedule",
    ...     "job_name": "hourly_metrics",
    ...     "schedule_type": "interval",
    ...     "schedule": "1 hour",
    ...     "agentool_name": "metrics",
    ...     "input_data": {"operation": "export", "format": "json"}
    ... })
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import RunContext

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.triggers.date import DateTrigger
    from apscheduler.events import (
        EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED,
        EVENT_JOB_ADDED, EVENT_JOB_REMOVED
    )
    from apscheduler.job import Job
    APSCHEDULER_AVAILABLE = True
except ImportError as e:
    APSCHEDULER_AVAILABLE = False
    # Fallback to simple asyncio-based scheduler
    AsyncIOScheduler = None

from agentool.base import BaseOperationInput
from agentool import create_agentool
from agentool.core.registry import RoutingConfig
from agentool.core.injector import get_injector


# Global scheduler instance (singleton)
_scheduler: Optional[AsyncIOScheduler] = None
_scheduler_running: bool = False
_job_registry: Dict[str, Dict[str, Any]] = {}  # job_id -> job_info
_execution_history: Dict[str, List[Dict[str, Any]]] = {}  # job_id -> execution results


class SchedulerInput(BaseOperationInput):
    """Input schema for scheduler operations."""
    operation: Literal[
        'schedule', 'cancel', 'pause', 'resume',
        'run_now', 'list', 'get_job', 'get_history',
        'start', 'stop', 'status'
    ] = Field(description="The scheduler operation to perform")
    
    # Job identification
    job_id: Optional[str] = Field(None, description="Unique job identifier")
    job_name: Optional[str] = Field(None, description="Human-readable job name")
    
    # Schedule configuration
    schedule_type: Optional[Literal['cron', 'interval', 'once']] = Field(
        None, description="Type of schedule"
    )
    schedule: Optional[str] = Field(
        None, description="Schedule expression (cron pattern, interval, or datetime)"
    )
    
    # Execution configuration - The core of the execution bus
    agentool_name: Optional[str] = Field(
        None, description="Name of the AgenTool to execute"
    )
    input_data: Optional[Dict[str, Any]] = Field(
        None, description="Input data to pass to the AgenTool"
    )
    
    # Options
    max_instances: Optional[int] = Field(
        1, description="Maximum concurrent executions"
    )
    misfire_grace_time: Optional[int] = Field(
        30, description="Seconds to wait for missed jobs"
    )
    coalesce: Optional[bool] = Field(
        True, description="Combine missed executions"
    )
    
    # Integration options
    result_queue: Optional[str] = Field(
        None, description="Queue name to send execution results to"
    )
    error_queue: Optional[str] = Field(
        None, description="Queue name to send errors to"
    )
    
    @field_validator('schedule_type')
    def validate_schedule_type(cls, v, info):
        """Validate that schedule_type is provided for schedule operation."""
        operation = info.data.get('operation')
        if operation == 'schedule' and not v:
            raise ValueError("schedule_type is required for schedule operation")
        return v
    
    @field_validator('schedule')
    def validate_schedule(cls, v, info):
        """Validate that schedule is provided for schedule operation."""
        operation = info.data.get('operation')
        if operation == 'schedule' and not v:
            raise ValueError("schedule is required for schedule operation")
        return v
    
    @field_validator('agentool_name')
    def validate_agentool_name(cls, v, info):
        """Validate that agentool_name is provided for operations that require it."""
        operation = info.data.get('operation')
        if operation in ['schedule', 'run_now'] and not v:
            raise ValueError(f"agentool_name is required for {operation} operation")
        return v
    
    @field_validator('input_data')
    def validate_input_data(cls, v, info):
        """Validate that input_data is provided for operations that require it."""
        operation = info.data.get('operation')
        if operation in ['schedule', 'run_now'] and v is None:
            raise ValueError(f"input_data is required for {operation} operation")
        return v
    
    @field_validator('job_id')
    def validate_job_id(cls, v, info):
        """Validate that job_id is provided for operations that require it."""
        operation = info.data.get('operation')
        if operation in ['cancel', 'pause', 'resume', 'get_job'] and not v:
            raise ValueError(f"job_id is required for {operation} operation")
        return v


class SchedulerOutput(BaseModel):
    """Structured output for scheduler operations."""
    success: bool = Field(default=True, description="Whether the operation succeeded")
    operation: str = Field(description="The operation that was performed")
    message: str = Field(description="Human-readable result message")
    data: Optional[Dict[str, Any]] = Field(None, description="Operation-specific data")


def _get_or_create_scheduler() -> AsyncIOScheduler:
    """Get or create the global scheduler instance."""
    global _scheduler, _scheduler_running
    
    
    if not APSCHEDULER_AVAILABLE:
        raise ImportError("APScheduler is not installed. Install with: pip install apscheduler")
    
    if _scheduler is None:
        _scheduler = AsyncIOScheduler()
        # Add event listeners for monitoring
        _scheduler.add_listener(_job_event_listener, 
                               EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED)
    
    if not _scheduler_running:
        _scheduler.start()
        _scheduler_running = True
    
    return _scheduler


def _job_event_listener(event):
    """Listen to job events for monitoring and reactive execution."""
    job_id = event.job_id
    
    # Record execution history
    if job_id not in _execution_history:
        _execution_history[job_id] = []
    
    execution_record = {
        'timestamp': datetime.now().isoformat(),
        'event_type': event.__class__.__name__,
        'success': event.exception is None if hasattr(event, 'exception') else True
    }
    
    if hasattr(event, 'exception') and event.exception:
        execution_record['error'] = str(event.exception)
    
    _execution_history[job_id].append(execution_record)
    
    # Limit history to last 100 executions
    if len(_execution_history[job_id]) > 100:
        _execution_history[job_id] = _execution_history[job_id][-100:]
    
    # Log the event asynchronously
    asyncio.create_task(_log_job_event(job_id, event))


async def _log_job_event(job_id: str, event) -> None:
    """Log job events to the logging system."""
    injector = get_injector()
    
    # Determine log level and message based on event type
    event_type = event.__class__.__name__
    job_info = _job_registry.get(job_id, {})
    
    # Check if it's an error event first (has exception)
    if hasattr(event, 'exception') and event.exception:
        # Failed execution
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'scheduler',
            'message': f"Job {job_id} execution failed",
            'data': {
                'job_id': job_id,
                'job_name': job_info.get('job_name'),
                'agentool': job_info.get('agentool_name'),
                'error': str(event.exception),
                'event_type': event_type
            }
        })
    elif event_type == 'JobExecutionEvent':
        # Successful execution
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'scheduler',
            'message': f"Job {job_id} executed successfully",
            'data': {
                'job_id': job_id,
                'job_name': job_info.get('job_name'),
                'agentool': job_info.get('agentool_name'),
                'event_type': event_type
            }
        })
    elif event_type == 'JobMissedEvent':
        # Missed execution
        await injector.run('logging', {
            'operation': 'log',
            'level': 'WARN',
            'logger_name': 'scheduler',
            'message': f"Job {job_id} missed its scheduled execution",
            'data': {
                'job_id': job_id,
                'job_name': job_info.get('job_name'),
                'agentool': job_info.get('agentool_name'),
                'event_type': event_type
            }
        })


async def _execute_agentool(job_id: str, agentool_name: str, input_data: Dict[str, Any],
                           result_queue: Optional[str] = None, 
                           error_queue: Optional[str] = None):
    """
    Core execution function - This is what the scheduler actually runs.
    It ONLY executes injector.run() calls, making scheduler a pure execution engine.
    """
    injector = get_injector()
    
    # Log job start
    await injector.run('logging', {
        'operation': 'log',
        'level': 'DEBUG',
        'logger_name': 'scheduler',
        'message': f"Starting execution of job {job_id}",
        'data': {
            'job_id': job_id,
            'agentool': agentool_name,
            'input': input_data
        }
    })
    
    try:
        # THE CORE: Scheduler only executes AgenTool calls
        result = await injector.run(agentool_name, input_data)
        
        # If configured, send result to queue (data bus)
        if result_queue:
            await injector.run('logging', {
                'operation': 'log',
                'level': 'DEBUG',
                'logger_name': 'scheduler',
                'message': f"Sending job {job_id} result to queue {result_queue}",
                'data': {'job_id': job_id, 'queue': result_queue}
            })
            
            # Serialize result appropriately based on its type
            if hasattr(result, 'model_dump'):
                # Result is a Pydantic model (typed output)
                result_data = result.model_dump()
            elif hasattr(result, 'output'):
                # Result is an AgentRunResult
                result_data = json.loads(result.output)
            else:
                # Result is something else, convert to string
                result_data = str(result)
            
            await injector.run('queue', {
                'operation': 'enqueue',
                'queue_name': result_queue,
                'message': {
                    'job_id': job_id,
                    'agentool': agentool_name,
                    'result': result_data,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                }
            })
        
        return result
        
    except Exception as e:
        # Log the error
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'scheduler',
            'message': f"Job {job_id} failed with error",
            'data': {
                'job_id': job_id,
                'agentool': agentool_name,
                'error': str(e),
                'error_type': type(e).__name__
            }
        })
        
        # Send error to error queue if configured
        if error_queue:
            await injector.run('logging', {
                'operation': 'log',
                'level': 'DEBUG',
                'logger_name': 'scheduler',
                'message': f"Sending job {job_id} error to queue {error_queue}",
                'data': {'job_id': job_id, 'queue': error_queue}
            })
            
            await injector.run('queue', {
                'operation': 'enqueue',
                'queue_name': error_queue,
                'message': {
                    'job_id': job_id,
                    'agentool': agentool_name,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'error'
                }
            })
        raise


def _parse_schedule(schedule_type: str, schedule: str) -> Any:
    """Parse schedule string into APScheduler trigger."""
    if schedule_type == 'cron':
        # Parse cron expression
        return CronTrigger.from_crontab(schedule)
    
    elif schedule_type == 'interval':
        # Parse interval like "30 seconds", "5 minutes", "1 hour", "2 days"
        parts = schedule.lower().split()
        if len(parts) != 2:
            raise ValueError(f"Invalid interval format: {schedule}")
        
        value = int(parts[0])
        unit = parts[1].rstrip('s')  # Remove trailing 's'
        
        kwargs = {}
        if unit in ['second', 'seconds']:
            kwargs['seconds'] = value
        elif unit in ['minute', 'minutes']:
            kwargs['minutes'] = value
        elif unit in ['hour', 'hours']:
            kwargs['hours'] = value
        elif unit in ['day', 'days']:
            kwargs['days'] = value
        elif unit in ['week', 'weeks']:
            kwargs['weeks'] = value
        else:
            raise ValueError(f"Invalid interval unit: {unit}")
        
        return IntervalTrigger(**kwargs)
    
    elif schedule_type == 'once':
        # Parse datetime string
        if schedule == 'now':
            run_date = datetime.now()
        else:
            run_date = datetime.fromisoformat(schedule)
        return DateTrigger(run_date=run_date)
    
    else:
        raise ValueError(f"Invalid schedule type: {schedule_type}")


# Tool implementations

async def scheduler_schedule(ctx: RunContext[Any], job_id: Optional[str], job_name: Optional[str],
                           schedule_type: str, schedule: str, agentool_name: str,
                           input_data: Dict[str, Any], max_instances: int,
                           misfire_grace_time: int, coalesce: bool,
                           result_queue: Optional[str], error_queue: Optional[str]) -> SchedulerOutput:
    """Schedule an AgenTool for execution."""
    
    try:
        scheduler = _get_or_create_scheduler()
        
        # Generate job ID if not provided
        if not job_id:
            job_id = f"job_{uuid.uuid4().hex[:8]}"
        
        # Parse schedule
        trigger = _parse_schedule(schedule_type, schedule)
        
        # Add job to scheduler
        job = scheduler.add_job(
            func=_execute_agentool,
            trigger=trigger,
            args=[job_id, agentool_name, input_data, result_queue, error_queue],
            id=job_id,
            name=job_name or job_id,
            max_instances=max_instances,
            misfire_grace_time=misfire_grace_time,
            coalesce=coalesce,
            replace_existing=True
        )
        
        # Store job info
        _job_registry[job_id] = {
            'job_id': job_id,
            'job_name': job_name or job_id,
            'schedule_type': schedule_type,
            'schedule': schedule,
            'agentool_name': agentool_name,
            'input_data': input_data,
            'created_at': datetime.now().isoformat(),
            'next_run': job.next_run_time.isoformat() if job.next_run_time else None,
            'result_queue': result_queue,
            'error_queue': error_queue
        }
        
        # Log job scheduled
        injector = get_injector()
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'scheduler',
            'message': f"Job '{job_name or job_id}' scheduled successfully",
            'data': {
                'job_id': job_id,
                'job_name': job_name or job_id,
                'schedule_type': schedule_type,
                'schedule': schedule,
                'agentool': agentool_name,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None
            }
        })
        
        return SchedulerOutput(
            success=True,
            operation='schedule',
            message=f"Job '{job_name or job_id}' scheduled successfully",
            data={
                'job_id': job_id,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to schedule job: {str(e)}") from e


async def scheduler_cancel(ctx: RunContext[Any], job_id: str) -> SchedulerOutput:
    """Cancel a scheduled job."""
    try:
        scheduler = _get_or_create_scheduler()
        
        # Get job info before removing
        job_info = _job_registry.get(job_id, {})
        
        scheduler.remove_job(job_id)
        
        # Remove from registry
        if job_id in _job_registry:
            del _job_registry[job_id]
        
        # Log job cancellation
        injector = get_injector()
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'scheduler',
            'message': f"Job '{job_id}' cancelled",
            'data': {
                'job_id': job_id,
                'job_name': job_info.get('job_name'),
                'agentool': job_info.get('agentool_name')
            }
        })
        
        return SchedulerOutput(
            success=True,
            operation='cancel',
            message=f"Job '{job_id}' cancelled successfully",
            data={'job_id': job_id}
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to cancel job: {str(e)}") from e


async def scheduler_run_now(ctx: RunContext[Any], job_id: Optional[str], 
                           agentool_name: str, input_data: Dict[str, Any],
                           result_queue: Optional[str], error_queue: Optional[str]) -> SchedulerOutput:
    """Execute an AgenTool immediately."""
    try:
        # Generate job ID for tracking
        if not job_id:
            job_id = f"immediate_{uuid.uuid4().hex[:8]}"
        
        # Log immediate execution request
        injector = get_injector()
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'scheduler',
            'message': f"Executing job {job_id} immediately",
            'data': {
                'job_id': job_id,
                'agentool': agentool_name,
                'input': input_data
            }
        })
        
        # Execute directly
        result = await _execute_agentool(job_id, agentool_name, input_data, 
                                        result_queue, error_queue)
        
        # Serialize result appropriately based on its type
        if hasattr(result, 'model_dump'):
            # Result is a Pydantic model (typed output)
            result_data = result.model_dump()
        elif hasattr(result, 'output'):
            # Result is an AgentRunResult
            result_data = json.loads(result.output)
        else:
            # Result is something else, convert to string
            result_data = str(result)
        
        return SchedulerOutput(
            success=True,
            operation='run_now',
            message=f"AgenTool '{agentool_name}' executed successfully",
            data={
                'job_id': job_id,
                'result': result_data
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to execute AgenTool: {str(e)}") from e


async def scheduler_list(ctx: RunContext[Any]) -> SchedulerOutput:
    """List all scheduled jobs."""
    try:
        scheduler = _get_or_create_scheduler()
        
        jobs = []
        for job in scheduler.get_jobs():
            job_info = _job_registry.get(job.id, {})
            jobs.append({
                'job_id': job.id,
                'job_name': job.name,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None,
                'agentool_name': job_info.get('agentool_name'),
                'schedule_type': job_info.get('schedule_type'),
                'schedule': job_info.get('schedule')
            })
        
        return SchedulerOutput(
            success=True,
            operation='list',
            message=f"Found {len(jobs)} scheduled jobs",
            data={'jobs': jobs, 'count': len(jobs)}
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to list jobs: {str(e)}") from e


async def scheduler_get_job(ctx: RunContext[Any], job_id: str) -> SchedulerOutput:
    """Get details of a specific job."""
    try:
        scheduler = _get_or_create_scheduler()
        
        job = scheduler.get_job(job_id)
        if not job:
            # Discovery operation - return success=False instead of raising
            return SchedulerOutput(
                success=False,
                operation='get_job',
                message=f"Job '{job_id}' not found",
                data=None
            )
        
        job_info = _job_registry.get(job_id, {})
        history = _execution_history.get(job_id, [])
        
        return SchedulerOutput(
            success=True,
            operation='get_job',
            message=f"Job '{job_id}' details retrieved",
            data={
                'job_id': job.id,
                'job_name': job.name,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None,
                'pending': job.pending,
                'info': job_info,
                'recent_history': history[-10:] if history else []
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to get job details: {str(e)}") from e


async def scheduler_pause(ctx: RunContext[Any], job_id: str) -> SchedulerOutput:
    """Pause a scheduled job."""
    try:
        scheduler = _get_or_create_scheduler()
        
        scheduler.pause_job(job_id)
        
        return SchedulerOutput(
            success=True,
            operation='pause',
            message=f"Job '{job_id}' paused successfully",
            data={'job_id': job_id}
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to pause job: {str(e)}") from e


async def scheduler_resume(ctx: RunContext[Any], job_id: str) -> SchedulerOutput:
    """Resume a paused job."""
    try:
        scheduler = _get_or_create_scheduler()
        
        scheduler.resume_job(job_id)
        
        return SchedulerOutput(
            success=True,
            operation='resume',
            message=f"Job '{job_id}' resumed successfully",
            data={'job_id': job_id}
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to resume job: {str(e)}") from e


async def scheduler_status(ctx: RunContext[Any]) -> SchedulerOutput:
    """Get scheduler status."""
    try:
        global _scheduler_running
        
        status = {
            'running': _scheduler_running,
            'jobs_count': len(_job_registry),
            'apscheduler_available': APSCHEDULER_AVAILABLE
        }
        
        if _scheduler_running and _scheduler:
            status['scheduler_state'] = _scheduler.state
            status['jobs'] = len(_scheduler.get_jobs())
        
        return SchedulerOutput(
            success=True,
            operation='status',
            message="Scheduler status retrieved",
            data=status
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to get scheduler status: {str(e)}") from e


# Main routing function
async def manage_scheduler(ctx: RunContext[Any], operation: str, **kwargs) -> SchedulerOutput:
    """Main routing function for scheduler operations."""
    
    if operation == 'schedule':
        return await scheduler_schedule(
            ctx,
            kwargs.get('job_id'),
            kwargs.get('job_name'),
            kwargs['schedule_type'],
            kwargs['schedule'],
            kwargs['agentool_name'],
            kwargs['input_data'],
            kwargs.get('max_instances', 1),
            kwargs.get('misfire_grace_time', 30),
            kwargs.get('coalesce', True),
            kwargs.get('result_queue'),
            kwargs.get('error_queue')
        )
    
    elif operation == 'cancel':
        return await scheduler_cancel(ctx, kwargs['job_id'])
    
    elif operation == 'run_now':
        return await scheduler_run_now(
            ctx,
            kwargs.get('job_id'),
            kwargs['agentool_name'],
            kwargs['input_data'],
            kwargs.get('result_queue'),
            kwargs.get('error_queue')
        )
    
    elif operation == 'list':
        return await scheduler_list(ctx)
    
    elif operation == 'get_job':
        return await scheduler_get_job(ctx, kwargs['job_id'])
    
    elif operation == 'pause':
        return await scheduler_pause(ctx, kwargs['job_id'])
    
    elif operation == 'resume':
        return await scheduler_resume(ctx, kwargs['job_id'])
    
    elif operation == 'status':
        return await scheduler_status(ctx)
    
    else:
        raise ValueError(f"Unknown operation: {operation}")


def create_scheduler_agent():
    """Create and return the Scheduler AgenTool agent."""
    
    # Don't try to start scheduler here - it will be started when needed
    # This avoids issues with asyncio event loops not being available
    
    # Define routing configuration
    routing_config = RoutingConfig(
        operation_field='operation',
        operation_map={
            'schedule': ('manage_scheduler', lambda x: {
                'operation': x.operation,
                'job_id': x.job_id,
                'job_name': x.job_name,
                'schedule_type': x.schedule_type,
                'schedule': x.schedule,
                'agentool_name': x.agentool_name,
                'input_data': x.input_data,
                'max_instances': x.max_instances,
                'misfire_grace_time': x.misfire_grace_time,
                'coalesce': x.coalesce,
                'result_queue': x.result_queue,
                'error_queue': x.error_queue
            }),
            'cancel': ('manage_scheduler', lambda x: {
                'operation': x.operation,
                'job_id': x.job_id
            }),
            'run_now': ('manage_scheduler', lambda x: {
                'operation': x.operation,
                'job_id': x.job_id,
                'agentool_name': x.agentool_name,
                'input_data': x.input_data,
                'result_queue': x.result_queue,
                'error_queue': x.error_queue
            }),
            'list': ('manage_scheduler', lambda x: {
                'operation': x.operation
            }),
            'get_job': ('manage_scheduler', lambda x: {
                'operation': x.operation,
                'job_id': x.job_id
            }),
            'pause': ('manage_scheduler', lambda x: {
                'operation': x.operation,
                'job_id': x.job_id
            }),
            'resume': ('manage_scheduler', lambda x: {
                'operation': x.operation,
                'job_id': x.job_id
            }),
            'status': ('manage_scheduler', lambda x: {
                'operation': x.operation
            })
        }
    )
    
    # Create the AgenTool
    agent = create_agentool(
        name='scheduler',
        input_schema=SchedulerInput,
        output_type=SchedulerOutput,
        routing_config=routing_config,
        tools=[manage_scheduler],
        system_prompt="Execute AgenTools on schedules. The execution engine for the AgenTool ecosystem.",
        description="Scheduler for executing AgenTools at specified times or intervals",
        version="1.0.0",
        tags=["scheduler", "execution", "cron", "automation"],
        dependencies=["queue", "logging"]  # Uses queue for result/error queues and logging for events
    )
    
    return agent