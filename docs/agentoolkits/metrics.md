# Metrics AgenToolkit

> **Note**: For guidelines on creating new AgenToolkits, see [CRAFTING_AGENTOOLS.md](../CRAFTING_AGENTOOLS.md). For testing patterns and examples, refer to [tests/agentoolkit/test_metrics.py](../../../tests/agentoolkit/test_metrics.py).

## Overview

The Metrics AgenToolkit provides comprehensive metrics tracking and observability capabilities for AgenTools. It supports automatic execution tracking and manual metric creation with multiple metric types including counters, gauges, histograms, summaries, and timers. The toolkit integrates seamlessly with the injector system to automatically track AgenTool executions.

### Key Features
- Automatic execution tracking (success/failure/duration)
- Multiple metric types (counter, gauge, histogram, summary, timer)
- Time-series data support with observation limits
- Statistical aggregations (sum, avg, min, max, percentiles)
- Label-based filtering and categorization
- Auto-creation of metrics for AgenTool tracking
- Export capabilities (JSON, Prometheus, StatsD formats)
- Pattern-based metric querying and aggregation

## Creation Method

```python
from agentoolkit.observability.metrics import create_metrics_agent

# Create the agent
agent = create_metrics_agent()
```

The creation function returns a fully configured AgenTool with name `'metrics'`.

## Input Schema

### MetricsInput

The input schema is a Pydantic model with the following fields:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `operation` | `Literal['create', 'increment', 'decrement', 'set', 'observe', 'get', 'list', 'aggregate', 'reset', 'delete', 'export']` | Yes | - | The metrics operation to perform |
| `name` | `Optional[str]` | No | None | Metric name (e.g., 'http.requests.total') |
| `pattern` | `Optional[str]` | No | None | Pattern for list/aggregate operations |
| `type` | `Optional[MetricType]` | No | None | Type of metric (counter, gauge, histogram, summary, timer) |
| `description` | `Optional[str]` | No | None | Human-readable description |
| `unit` | `Optional[str]` | No | None | Unit of measurement (e.g., 'seconds', 'bytes') |
| `value` | `Optional[float]` | No | None | Value for increment/decrement/set/observe |
| `labels` | `Optional[Dict[str, str]]` | No | None | Labels for metric categorization |
| `time_range` | `Optional[int]` | No | None | Time range in seconds for queries |
| `aggregation` | `Optional[Literal['sum', 'avg', 'min', 'max', 'count', 'p50', 'p95', 'p99']]` | No | None | Aggregation function |
| `format` | `Optional[Literal['json', 'prometheus', 'statsd']]` | No | 'json' | Export format |

## Operations Schema

The routing configuration maps operations to tool functions:

| Operation | Tool Function | Required Parameters | Description |
|-----------|--------------|-------------------|-------------|
| `create` | `metrics_create` | `name`, `type` | Create a new metric with type and metadata |
| `increment` | `metrics_increment` | `name`, `value`, `labels` | Increment a counter or gauge metric |
| `decrement` | `metrics_decrement` | `name`, `value`, `labels` | Decrement a gauge metric |
| `set` | `metrics_set` | `name`, `value`, `labels` | Set a gauge metric to a specific value |
| `observe` | `metrics_observe` | `name`, `value`, `labels` | Record an observation for histogram/summary/timer |
| `get` | `metrics_get` | `name` | Get a metric's current value and metadata |
| `list` | `metrics_list` | `pattern` | List all metrics matching a pattern |
| `aggregate` | `metrics_aggregate` | `pattern`, `aggregation`, `time_range` | Aggregate metrics matching a pattern |
| `reset` | `metrics_reset` | `name` | Reset a metric to its initial value |
| `delete` | `metrics_delete` | `name` | Delete a metric |
| `export` | `metrics_export` | `pattern`, `format` | Export metrics in various formats |

## Output Schema

### MetricsOutput

All operations return a `MetricsOutput` model with:

| Field | Type | Description |
|-------|------|-------------|
| `operation` | `str` | The operation that was performed |
| `message` | `str` | Human-readable result message |
| `data` | `Optional[Any]` | Operation-specific data |

### Operation-Specific Data Fields

- **create**: `name`, `type`, `description`, `unit`
- **increment/decrement**: `name`, `new_value`, `labels`
- **set**: `name`, `value`, `labels`
- **observe**: `name`, `value`, `labels`, `observation_count`
- **get**: `name`, `type`, `description`, `unit`, `value`, `statistics`, `created_at`, `updated_at`
- **list**: `metrics`, `count`, `pattern`
- **aggregate**: `pattern`, `aggregation`, `result`, `metric_count`, `value_count`, `time_range`
- **reset**: `name`, `type`
- **delete**: `name`
- **export**: `format`, `content`, `metric_count`

## Dependencies

This AgenToolkit depends on:
- **storage_kv**: Used for persistent metric storage and metadata

## Tools

### metrics_create
```python
async def metrics_create(ctx: RunContext[Any], name: str, type: MetricType,
                        description: Optional[str], unit: Optional[str]) -> MetricsOutput
```
Create a new metric with specified type and metadata. Initializes appropriate data structures based on metric type.

**Raises:**
- `ValueError`: If metric already exists or invalid parameters
- `RuntimeError`: If storage operation fails

### metrics_increment
```python
async def metrics_increment(ctx: RunContext[Any], name: str, value: float,
                           labels: Optional[Dict[str, str]]) -> MetricsOutput
```
Increment a counter or gauge metric. Auto-creates counter metrics for AgenTool tracking (names starting with "agentool.").

**Raises:**
- `KeyError`: If metric doesn't exist (except auto-created ones)
- `ValueError`: If invalid operation for metric type
- `RuntimeError`: If storage operation fails

### metrics_decrement
```python
async def metrics_decrement(ctx: RunContext[Any], name: str, value: float,
                           labels: Optional[Dict[str, str]]) -> MetricsOutput
```
Decrement a gauge metric. Only works with gauge metric types.

**Raises:**
- `KeyError`: If metric doesn't exist
- `ValueError`: If not a gauge metric
- `RuntimeError`: If storage operation fails

### metrics_set
```python
async def metrics_set(ctx: RunContext[Any], name: str, value: float,
                     labels: Optional[Dict[str, str]]) -> MetricsOutput
```
Set a gauge metric to a specific value. Only works with gauge metric types.

**Raises:**
- `KeyError`: If metric doesn't exist
- `ValueError`: If not a gauge metric
- `RuntimeError`: If storage operation fails

### metrics_observe
```python
async def metrics_observe(ctx: RunContext[Any], name: str, value: float,
                         labels: Optional[Dict[str, str]]) -> MetricsOutput
```
Record an observation for histogram, summary, or timer metrics. Auto-creates timer metrics for duration tracking.

**Raises:**
- `KeyError`: If metric doesn't exist (except auto-created ones)
- `ValueError`: If invalid operation for metric type
- `RuntimeError`: If storage operation fails

### metrics_get
```python
async def metrics_get(ctx: RunContext[Any], name: str) -> MetricsOutput
```
Get a metric's current value and metadata. Calculates statistics for observation-based metrics.

**Raises:**
- `KeyError`: If metric doesn't exist
- `ValueError`: If invalid metric name
- `RuntimeError`: If storage operation fails

### metrics_list
```python
async def metrics_list(ctx: RunContext[Any], pattern: Optional[str]) -> MetricsOutput
```
List all metrics matching a pattern. Returns basic metadata for each metric.

**Raises:**
- `RuntimeError`: If storage operation fails

### metrics_aggregate
```python
async def metrics_aggregate(ctx: RunContext[Any], pattern: str, aggregation: str,
                           time_range: Optional[int]) -> MetricsOutput
```
Aggregate metrics matching a pattern using specified aggregation function. Supports time-based filtering.

**Raises:**
- `ValueError`: If invalid aggregation function or empty pattern
- `RuntimeError`: If storage operation fails

### metrics_reset
```python
async def metrics_reset(ctx: RunContext[Any], name: str) -> MetricsOutput
```
Reset a metric to its initial value. Counters/gauges reset to 0, observation metrics clear history.

**Raises:**
- `KeyError`: If metric doesn't exist
- `ValueError`: If invalid metric name
- `RuntimeError`: If storage operation fails

### metrics_delete
```python
async def metrics_delete(ctx: RunContext[Any], name: str) -> MetricsOutput
```
Delete a metric completely from storage.

**Raises:**
- `KeyError`: If metric doesn't exist
- `ValueError`: If invalid metric name
- `RuntimeError`: If storage operation fails

### metrics_export
```python
async def metrics_export(ctx: RunContext[Any], pattern: Optional[str],
                        format: str) -> MetricsOutput
```
Export metrics in various formats (JSON, Prometheus, StatsD).

**Raises:**
- `ValueError`: If invalid export format
- `RuntimeError`: If storage operation fails

## Exceptions

| Exception Type | Scenarios |
|---------------|-----------|
| `ValueError` | - Metric already exists during creation<br>- Invalid metric type for operation<br>- Empty metric name<br>- Invalid aggregation function<br>- Invalid export format |
| `KeyError` | - Getting non-existent metric<br>- Setting TTL on non-existent metric |
| `RuntimeError` | - Storage operation failures<br>- Serialization issues<br>- File I/O errors during export |

## Usage Examples

### Basic Metric Operations
```python
from agentoolkit.observability.metrics import create_metrics_agent
from agentool.core.injector import get_injector

# Create and register the agent
agent = create_metrics_agent()
injector = get_injector()

# Create a counter metric
result = await injector.run('metrics', {
    "operation": "create",
    "name": "api.requests.total",
    "type": "counter",
    "description": "Total API requests",
    "unit": "count"
})

# Increment with labels
result = await injector.run('metrics', {
    "operation": "increment",
    "name": "api.requests.total",
    "value": 1,
    "labels": {"endpoint": "/users", "method": "GET", "status": "200"}
})

# Create and use a gauge
result = await injector.run('metrics', {
    "operation": "create",
    "name": "memory.usage",
    "type": "gauge",
    "description": "Memory usage percentage",
    "unit": "percent"
})

result = await injector.run('metrics', {
    "operation": "set",
    "name": "memory.usage",
    "value": 75.5
})
```

### Timer and Histogram Metrics
```python
# Create a timer for response times
result = await injector.run('metrics', {
    "operation": "create",
    "name": "api.response.time",
    "type": "timer",
    "description": "API response times",
    "unit": "seconds"
})

# Record observations
result = await injector.run('metrics', {
    "operation": "observe",
    "name": "api.response.time",
    "value": 0.125,
    "labels": {"endpoint": "/users"}
})

result = await injector.run('metrics', {
    "operation": "observe",
    "name": "api.response.time",
    "value": 0.089,
    "labels": {"endpoint": "/users"}
})

# Get statistics
result = await injector.run('metrics', {
    "operation": "get",
    "name": "api.response.time"
})
# Returns statistics including avg, min, max, percentiles
```

### Aggregation and Export
```python
# Aggregate all API metrics
result = await injector.run('metrics', {
    "operation": "aggregate",
    "pattern": "api.*",
    "aggregation": "avg",
    "time_range": 3600  # Last hour
})

# Export metrics in Prometheus format
result = await injector.run('metrics', {
    "operation": "export",
    "pattern": "*",
    "format": "prometheus"
})

# List all metrics
result = await injector.run('metrics', {
    "operation": "list",
    "pattern": "api.*"
})
```

### Auto-Created Metrics
```python
# AgenTool execution metrics are auto-created
# These are created automatically when AgenTools are executed:
# - agentool.{name}.executions (counter)
# - agentool.{name}.duration (timer)
# - agentool.{name}.errors (counter)

# Get auto-created metrics
result = await injector.run('metrics', {
    "operation": "get",
    "name": "agentool.storage_kv.executions"
})
```

## Testing

The test suite is located at `tests/agentoolkit/test_metrics.py`. Tests cover:
- All metric type operations (counter, gauge, histogram, summary, timer)
- Label-based metric tracking
- Statistical calculations and aggregations
- Auto-creation of AgenTool tracking metrics
- Export format validation
- Error handling for invalid operations
- Pattern matching and filtering

To run tests:
```bash
pytest tests/agentoolkit/test_metrics.py -v
```

## Notes

- Observations are limited to 100 entries per metric to prevent memory issues
- Auto-creation occurs for metrics starting with "agentool." (counters) or containing "duration" (timers)
- Labels are stored as JSON-serialized keys for efficient storage and lookup
- Statistics calculations require at least 2 observations for standard deviation and percentiles
- Export formats support different metric representations optimized for each monitoring system
- Time-based aggregation filters observations by timestamp within the specified range
- The toolkit follows Redis-style conventions for consistency with other storage systems