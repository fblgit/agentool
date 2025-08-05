# Scheduler AgenToolkit

> **Note**: For guidelines on creating new AgenToolkits, see [CRAFTING_AGENTOOLS.md](../CRAFTING_AGENTOOLS.md). For testing patterns and examples, refer to [tests/agentoolkit/test_scheduler.py](../../../tests/agentoolkit/test_scheduler.py).

## Overview

The Scheduler AgenToolkit provides execution engine capabilities for scheduled AgenTool runs using APScheduler. It serves as the core execution bus for the AgenTool ecosystem, enabling time-based automation and reactive workflows. The scheduler only executes `injector.run()` calls, maintaining a clean and focused architecture.

### Key Features
- Multiple scheduling types (cron, interval, one-time)
- Direct AgenTool execution via injector system
- Job persistence and management (pause, resume, cancel)
- Event monitoring and execution history tracking
- Integration with queue system for result/error handling
- Reactive execution patterns
- Missed job handling with configurable grace periods
- Concurrent execution control with instance limits

## Creation Method

```python
from agentoolkit.system.scheduler import create_scheduler_agent

# Create the agent
agent = create_scheduler_agent()
```

The creation function returns a fully configured AgenTool with name `'scheduler'`. The APScheduler instance is created and started automatically when first used.

## Input Schema

### SchedulerInput

The input schema is a Pydantic model with the following fields:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `operation` | `Literal['schedule', 'cancel', 'pause', 'resume', 'run_now', 'list', 'get_job', 'get_history', 'start', 'stop', 'status']` | Yes | - | The scheduler operation to perform |
| `job_id` | `Optional[str]` | No | None | Unique job identifier (auto-generated if not provided) |
| `job_name` | `Optional[str]` | No | None | Human-readable job name |
| `schedule_type` | `Optional[Literal['cron', 'interval', 'once']]` | No | None | Type of schedule |
| `schedule` | `Optional[str]` | No | None | Schedule expression (cron pattern, interval, or datetime) |
| `agentool_name` | `Optional[str]` | No | None | Name of the AgenTool to execute |
| `input_data` | `Optional[Dict[str, Any]]` | No | None | Input data to pass to the AgenTool |
| `max_instances` | `Optional[int]` | No | 1 | Maximum concurrent executions |
| `misfire_grace_time` | `Optional[int]` | No | 30 | Seconds to wait for missed jobs |
| `coalesce` | `Optional[bool]` | No | True | Combine missed executions |
| `result_queue` | `Optional[str]` | No | None | Queue name to send execution results to |
| `error_queue` | `Optional[str]` | No | None | Queue name to send errors to |

## Operations Schema

The routing configuration maps operations to tool functions:

| Operation | Description | Required Parameters |
|-----------|-------------|-------------------|
| `schedule` | Schedule an AgenTool for execution | `schedule_type`, `schedule`, `agentool_name`, `input_data` |
| `cancel` | Cancel a scheduled job | `job_id` |
| `pause` | Pause a job (stops scheduling but keeps job) | `job_id` |
| `resume` | Resume a paused job | `job_id` |
| `run_now` | Execute an AgenTool immediately | `agentool_name`, `input_data` |
| `list` | List all scheduled jobs | - |
| `get_job` | Get details of a specific job | `job_id` |
| `status` | Get scheduler status | - |

## Output Schema

### SchedulerOutput

All operations return a `SchedulerOutput` model with:

| Field | Type | Description |
|-------|------|-------------|
| `operation` | `str` | The operation that was performed |
| `message` | `str` | Human-readable result message |
| `data` | `Optional[Dict[str, Any]]` | Operation-specific data |

### Operation-Specific Data Fields

- **schedule**: `job_id`, `next_run`
- **cancel**: `job_id`
- **pause**: `job_id`
- **resume**: `job_id`
- **run_now**: `job_id`, `result`
- **list**: `jobs`, `count`
- **get_job**: `job_id`, `job_name`, `next_run`, `pending`, `info`, `recent_history`
- **status**: `running`, `jobs_count`, `apscheduler_available`, `scheduler_state`, `jobs`

## Schedule Types

### Cron Schedules
Uses standard cron syntax:
- `"0 9 * * 1-5"` - Every weekday at 9 AM
- `"*/15 * * * *"` - Every 15 minutes
- `"0 0 1 * *"` - First day of every month at midnight

### Interval Schedules
Human-readable interval expressions:
- `"30 seconds"` - Every 30 seconds
- `"5 minutes"` - Every 5 minutes
- `"1 hour"` - Every hour
- `"2 days"` - Every 2 days
- `"1 week"` - Every week

### One-time Schedules
Single execution schedules:
- `"now"` - Execute immediately
- `"2024-12-25T09:00:00"` - Execute at specific datetime (ISO format)

## Dependencies

This AgenToolkit depends on:
- **queue**: Optional integration for result and error handling
- **APScheduler**: External dependency for scheduling functionality

## Tools

### manage_scheduler
```python
async def manage_scheduler(ctx: RunContext[Any], operation: str, **kwargs) -> SchedulerOutput
```
Main routing function that dispatches to specific scheduler operation handlers.

**Core Execution Functions:**

### scheduler_schedule
```python
async def scheduler_schedule(ctx: RunContext[Any], job_id: Optional[str], job_name: Optional[str],
                           schedule_type: str, schedule: str, agentool_name: str,
                           input_data: Dict[str, Any], max_instances: int,
                           misfire_grace_time: int, coalesce: bool,
                           result_queue: Optional[str], error_queue: Optional[str]) -> SchedulerOutput
```
Schedule an AgenTool for execution with specified schedule pattern.

**Key Features:**
- Auto-generates job ID if not provided
- Supports all schedule types (cron, interval, once)
- Configurable job execution parameters
- Optional result/error queue integration

**Raises:**
- `RuntimeError`: If scheduling fails or invalid schedule parameters

### scheduler_run_now
```python
async def scheduler_run_now(ctx: RunContext[Any], job_id: Optional[str], 
                           agentool_name: str, input_data: Dict[str, Any],
                           result_queue: Optional[str], error_queue: Optional[str]) -> SchedulerOutput
```
Execute an AgenTool immediately without scheduling.

**Key Features:**
- Immediate execution
- Result/error queue integration
- Execution history tracking

**Raises:**
- `RuntimeError`: If AgenTool execution fails

### scheduler_cancel
```python
async def scheduler_cancel(ctx: RunContext[Any], job_id: str) -> SchedulerOutput
```
Cancel a scheduled job permanently.

**Raises:**
- `RuntimeError`: If job cancellation fails

### scheduler_pause
```python
async def scheduler_pause(ctx: RunContext[Any], job_id: str) -> SchedulerOutput
```
Pause a job (prevents execution but keeps job definition).

**Raises:**
- `RuntimeError`: If job pause fails

### scheduler_resume
```python
async def scheduler_resume(ctx: RunContext[Any], job_id: str) -> SchedulerOutput
```
Resume a paused job.

**Raises:**
- `RuntimeError`: If job resume fails

### scheduler_list
```python
async def scheduler_list(ctx: RunContext[Any]) -> SchedulerOutput
```
List all scheduled jobs with basic information.

**Raises:**
- `RuntimeError`: If job listing fails

### scheduler_get_job
```python
async def scheduler_get_job(ctx: RunContext[Any], job_id: str) -> SchedulerOutput
```
Get detailed information about a specific job including execution history.

**Raises:**
- `KeyError`: If job not found
- `RuntimeError`: If job retrieval fails

### scheduler_status
```python
async def scheduler_status(ctx: RunContext[Any]) -> SchedulerOutput
```
Get scheduler status and statistics.

**Raises:**
- `RuntimeError`: If status retrieval fails

## Execution Architecture

The scheduler follows a clean execution bus pattern:

1. **Pure Execution Engine**: Only executes `injector.run(agentool_name, input_data)` calls
2. **Event Monitoring**: Tracks job execution events and maintains history
3. **Queue Integration**: Sends results/errors to configured queues for reactive patterns
4. **History Tracking**: Maintains last 100 execution records per job

### Core Execution Function
```python
async def _execute_agentool(job_id: str, agentool_name: str, input_data: Dict[str, Any],
                           result_queue: Optional[str] = None, 
                           error_queue: Optional[str] = None):
    """Core execution function - only executes injector.run() calls."""
    injector = get_injector()
    result = await injector.run(agentool_name, input_data)
    # Send result to queue if configured
    # Handle errors and send to error queue if configured
```

## Exceptions

| Exception Type | Scenarios |
|---------------|-----------|
| `ImportError` | - APScheduler not installed |
| `ValueError` | - Invalid schedule type or format<br>- Unknown operation<br>- Invalid interval unit |
| `KeyError` | - Job not found when getting job details |
| `RuntimeError` | - Scheduler operation failures<br>- Job execution errors<br>- APScheduler errors |

## Usage Examples

### Basic Job Scheduling
```python
from agentoolkit.system.scheduler import create_scheduler_agent
from agentool.core.injector import get_injector

# Create and register the agent
agent = create_scheduler_agent()
injector = get_injector()

# Schedule a job to run every hour
result = await injector.run('scheduler', {
    "operation": "schedule",
    "job_name": "hourly_metrics",
    "schedule_type": "interval",
    "schedule": "1 hour",
    "agentool_name": "metrics",
    "input_data": {
        "operation": "export",
        "format": "json"
    }
})

# Schedule a cron job
result = await injector.run('scheduler', {
    "operation": "schedule",
    "job_name": "daily_backup",
    "schedule_type": "cron",
    "schedule": "0 2 * * *",  # 2 AM daily
    "agentool_name": "storage_fs",
    "input_data": {
        "operation": "copy",
        "source": "/data",
        "destination": "/backup"
    }
})
```

### Queue Integration for Reactive Patterns
```python
# Schedule job with result and error queues
result = await injector.run('scheduler', {
    "operation": "schedule",
    "job_name": "data_processor",
    "schedule_type": "interval",
    "schedule": "5 minutes",
    "agentool_name": "templates",
    "input_data": {
        "operation": "render",
        "template_name": "report",
        "variables": {"timestamp": "!ref:storage_kv:current_time"}
    },
    "result_queue": "processed_reports",
    "error_queue": "processing_errors"
})

# Set up reactive handlers for results and errors
# Results will be automatically queued for further processing
# Errors will be queued for alerting/retry logic
```

### Job Management
```python
# List all scheduled jobs
result = await injector.run('scheduler', {
    "operation": "list"
})

# Get details of a specific job
result = await injector.run('scheduler', {
    "operation": "get_job",
    "job_id": "hourly_metrics"
})

# Pause a job temporarily
result = await injector.run('scheduler', {
    "operation": "pause",
    "job_id": "hourly_metrics"
})

# Resume a paused job
result = await injector.run('scheduler', {
    "operation": "resume",
    "job_id": "hourly_metrics"
})

# Cancel a job permanently
result = await injector.run('scheduler', {
    "operation": "cancel",
    "job_id": "hourly_metrics"
})
```

### Immediate Execution
```python
# Execute an AgenTool immediately
result = await injector.run('scheduler', {
    "operation": "run_now",
    "agentool_name": "logging",
    "input_data": {
        "operation": "log",
        "level": "INFO",
        "message": "Manual execution triggered",
        "data": {"trigger": "user_request"}
    }
})
```

### Advanced Scheduling with Execution Control
```python
# Schedule with advanced options
result = await injector.run('scheduler', {
    "operation": "schedule",
    "job_id": "custom_processor",
    "job_name": "Custom Data Processor",
    "schedule_type": "interval",
    "schedule": "30 seconds",
    "agentool_name": "config",
    "input_data": {
        "operation": "get",
        "key": "processing.enabled",
        "default": True
    },
    "max_instances": 2,  # Allow up to 2 concurrent executions
    "misfire_grace_time": 60,  # Wait 60 seconds for missed jobs
    "coalesce": False,  # Don't combine missed executions
    "result_queue": "config_results",
    "error_queue": "config_errors"
})
```

### One-time Scheduling
```python
# Schedule for specific time
result = await injector.run('scheduler', {
    "operation": "schedule",
    "job_name": "year_end_report",
    "schedule_type": "once",
    "schedule": "2024-12-31T23:59:59",
    "agentool_name": "templates",
    "input_data": {
        "operation": "render",
        "template_name": "annual_report",
        "variables": {"year": 2024}
    }
})

# Schedule for immediate execution
result = await injector.run('scheduler', {
    "operation": "schedule",
    "job_name": "urgent_task",
    "schedule_type": "once",
    "schedule": "now",
    "agentool_name": "logging",
    "input_data": {
        "operation": "log",
        "level": "CRITICAL",
        "message": "Urgent task executed"
    }
})
```

### Scheduler Status and Monitoring
```python
# Get scheduler status
result = await injector.run('scheduler', {
    "operation": "status"
})

# Example response:
# {
#     "data": {
#         "running": True,
#         "jobs_count": 5,
#         "apscheduler_available": True,
#         "scheduler_state": 1,  # APScheduler state
#         "jobs": 5
#     }
# }
```

## Testing

The test suite is located at `tests/agentoolkit/test_scheduler.py`. Tests cover:
- All schedule types (cron, interval, once)
- Job management operations (schedule, cancel, pause, resume)
- Immediate execution functionality
- Queue integration for results and errors
- Execution history tracking
- Error handling for invalid schedules and missing jobs
- Concurrent execution limits and misfire handling

To run tests:
```bash
pytest tests/agentoolkit/test_scheduler.py -v
```

## Notes

- APScheduler is required as an external dependency (`pip install apscheduler`)
- The scheduler automatically starts when first used and maintains a singleton instance
- Job execution history is limited to the last 100 executions per job to prevent memory issues
- Queue integration is optional but enables powerful reactive workflow patterns
- Cron expressions follow standard cron syntax with full feature support
- Interval expressions support human-readable formats for ease of use
- Job IDs are auto-generated if not provided, ensuring uniqueness
- The scheduler serves as a pure execution engine, only calling `injector.run()` methods
- Missed job handling is configurable per job with grace periods and coalescing options
- Event monitoring provides detailed execution tracking for debugging and analysis