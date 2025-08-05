"""
Metrics AgenTool - Comprehensive metrics tracking and observability.

This toolkit provides automatic and manual metrics tracking for AgenTools,
including counters, gauges, histograms, summaries, and timers. It integrates
seamlessly with the injector to automatically track executions.

Features:
- Automatic execution tracking (success/failure/duration)
- Multiple metric types (counter, gauge, histogram, summary, timer)
- Time-series data support
- Statistical aggregations (sum, avg, min, max, percentiles)
- Label-based filtering
- Export capabilities

Example Usage:
    >>> from agentoolkit.observability.metrics import create_metrics_agent
    >>> from agentool.core.injector import get_injector
    >>> 
    >>> # Create and register the agent
    >>> agent = create_metrics_agent()
    >>> 
    >>> # Use through injector (automatic tracking happens in background)
    >>> injector = get_injector()
    >>> 
    >>> # Create a custom metric
    >>> result = await injector.run('metrics', {
    ...     "operation": "create",
    ...     "name": "api.requests",
    ...     "type": "counter",
    ...     "description": "API request count"
    ... })
    >>> 
    >>> # Increment the metric
    >>> result = await injector.run('metrics', {
    ...     "operation": "increment",
    ...     "name": "api.requests",
    ...     "value": 1,
    ...     "labels": {"endpoint": "/users", "method": "GET"}
    ... })
"""

import time
import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Literal, Union
from enum import Enum
from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from agentool.base import BaseOperationInput
from agentool import create_agentool
from agentool.core.registry import RoutingConfig
from agentool.core.injector import get_injector


class MetricType(str, Enum):
    """Supported metric types."""
    COUNTER = "counter"        # Monotonic increasing value
    GAUGE = "gauge"            # Value that can go up/down  
    HISTOGRAM = "histogram"    # Distribution of values
    SUMMARY = "summary"        # Statistical summary
    TIMER = "timer"           # Duration measurements


class MetricDefinition(BaseModel):
    """Strict schema for defining a single metric."""
    type: MetricType = Field(description="Type of metric")
    description: str = Field(description="Human-readable description")
    unit: Optional[str] = Field(default="count", description="Unit of measurement")
    labels: Optional[List[str]] = Field(default=None, description="Expected label keys")


class MetricsSchema(BaseModel):
    """Schema for batch metric creation."""
    metrics: Dict[str, MetricDefinition] = Field(
        description="Map of metric names to their definitions"
    )


class MetricsInput(BaseOperationInput):
    """Input schema for metrics operations."""
    operation: Literal[
        'create', 'increment', 'decrement', 'set', 'observe',
        'get', 'list', 'aggregate', 'reset', 'delete', 'export',
        'create_agent_metrics', 'create_from_schema'
    ] = Field(description="The metrics operation to perform")
    
    # Metric identification
    name: Optional[str] = Field(None, description="Metric name (e.g., 'http.requests.total')")
    pattern: Optional[str] = Field(None, description="Pattern for list/aggregate operations")
    
    # Metric configuration
    type: Optional[MetricType] = Field(None, description="Type of metric")
    description: Optional[str] = Field(None, description="Human-readable description")
    unit: Optional[str] = Field(None, description="Unit of measurement (e.g., 'seconds', 'bytes')")
    
    # Metric data
    value: Optional[float] = Field(None, description="Value for increment/decrement/set/observe")
    labels: Optional[Dict[str, str]] = Field(None, description="Labels for metric categorization")
    
    # Query parameters
    time_range: Optional[int] = Field(None, description="Time range in seconds for queries")
    aggregation: Optional[Literal['sum', 'avg', 'min', 'max', 'count', 'p50', 'p95', 'p99']] = Field(
        None, description="Aggregation function"
    )
    
    # Export format
    format: Optional[Literal['json', 'prometheus', 'statsd']] = Field(
        'json', description="Export format"
    )
    
    # New fields for create_agent_metrics
    agent_name: Optional[str] = Field(None, description="Agent name for create_agent_metrics")
    routing_config: Optional[Dict[str, Any]] = Field(None, description="Routing config with operation_map")
    
    # New field for create_from_schema
    metrics_schema: Optional[MetricsSchema] = Field(None, description="Strict schema for custom metrics")


class MetricsOutput(BaseModel):
    """Structured output for metrics operations."""
    operation: str = Field(description="The operation that was performed")
    message: str = Field(description="Human-readable result message")
    data: Optional[Any] = Field(None, description="Operation-specific data")


async def metrics_create(ctx: RunContext[Any], name: str, type: MetricType,
                        description: Optional[str], unit: Optional[str]) -> MetricsOutput:
    """
    Create a new metric.
    
    Args:
        ctx: Runtime context
        name: Metric name
        type: Type of metric
        description: Description
        unit: Unit of measurement
        
    Returns:
        MetricsOutput with creation status
        
    Raises:
        ValueError: If metric already exists or invalid parameters
        RuntimeError: If storage operation fails
    """
    try:
        if not name or not name.strip():
            raise ValueError("Metric name cannot be empty")
            
        injector = get_injector()
        
        # Check if metric already exists
        meta_result = await injector.run('storage_kv', {
            "operation": "get_metric",
            "key": f"metric:{name}",
            "namespace": "metrics"
        })
        
        if hasattr(meta_result, 'output'):
            meta_data = json.loads(meta_result.output)
        else:
            meta_data = meta_result
        
        # If metric exists (data is not None), raise error
        if meta_data["data"] is not None:
            raise ValueError(f"Metric '{name}' already exists")
        
        # Create metric metadata
        metric_meta = {
            "name": name,
            "type": type.value if isinstance(type, MetricType) else type,
            "description": description or "",
            "unit": unit or "",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "value": 0 if type in [MetricType.COUNTER, MetricType.GAUGE] else [],
            "observations": [],
            "labels_seen": {}
        }
        
        # Store metric metadata
        await injector.run('storage_kv', {
            "operation": "set",
            "key": f"metric:{name}",
            "value": metric_meta,
            "namespace": "metrics"
        })
        
        return MetricsOutput(
            operation="create",
            message=f"Created {type.value if isinstance(type, MetricType) else type} metric '{name}'",
            data={
                "name": name,
                "type": type.value if isinstance(type, MetricType) else type,
                "description": description,
                "unit": unit
            }
        )
            
    except ValueError:
        raise
    except KeyError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to create metric '{name}': {e}") from e


async def metrics_increment(ctx: RunContext[Any], name: str, value: float,
                           labels: Optional[Dict[str, str]]) -> MetricsOutput:
    """
    Increment a counter or gauge metric.
    
    Args:
        ctx: Runtime context
        name: Metric name
        value: Value to increment by
        labels: Optional labels
        
    Returns:
        MetricsOutput with new value
        
    Raises:
        KeyError: If metric doesn't exist
        ValueError: If invalid operation for metric type
        RuntimeError: If storage operation fails
    """
    try:
        if not name or not name.strip():
            raise ValueError("Metric name cannot be empty")
            
        injector = get_injector()
        
        # Get metric metadata - use get_metric which returns None instead of raising
        meta_result = await injector.run('storage_kv', {
            "operation": "get_metric",
            "key": f"metric:{name}",
            "namespace": "metrics"
        })
        
        if hasattr(meta_result, 'output'):
            meta_data = json.loads(meta_result.output)
        else:
            meta_data = meta_result
        
        # Check if metric exists (data is None when not found)
        if meta_data["data"] is None:
            # Auto-create metrics for automatic tracking
            if name.startswith("agentool."):
                # Determine metric type
                if name == "agentool.templates.count":
                    metric_type = MetricType.GAUGE
                else:
                    metric_type = MetricType.COUNTER
                
                # Create the metric automatically
                await metrics_create(ctx, name, metric_type, 
                                   f"Auto-created metric for {name}", "count")
                
                # Get the newly created metric
                meta_result = await injector.run('storage_kv', {
                    "operation": "get_metric",
                    "key": f"metric:{name}",
                    "namespace": "metrics"
                })
                
                if hasattr(meta_result, 'output'):
                    meta_data = json.loads(meta_result.output)
                else:
                    meta_data = meta_result
                
                metric_meta = meta_data["data"]["value"]
            else:
                raise KeyError(f"Metric '{name}' does not exist")
        else:
            metric_meta = meta_data["data"]["value"]
        
        if metric_meta["type"] not in [MetricType.COUNTER.value, MetricType.GAUGE.value]:
            raise ValueError(f"Cannot increment {metric_meta['type']} metric")
        
        # Update value
        if labels:
            # Convert to dict structure if needed for labeled metrics
            if not isinstance(metric_meta["value"], dict):
                old_value = metric_meta["value"]
                metric_meta["value"] = {}
                # Preserve the old unlabeled value if it exists
                if old_value != 0:
                    metric_meta["value"]["{}"] = old_value
            
            # Handle labeled metrics
            label_key = json.dumps(labels, sort_keys=True)
            metric_meta["value"][label_key] = metric_meta["value"].get(label_key, 0) + value
        else:
            # Handle unlabeled metrics
            if isinstance(metric_meta["value"], dict):
                # Already using labels, update the unlabeled entry
                metric_meta["value"]["{}"] = metric_meta["value"].get("{}", 0) + value
            else:
                metric_meta["value"] += value
        
        # Track labels
        if labels:
            for key, val in labels.items():
                if key not in metric_meta["labels_seen"]:
                    metric_meta["labels_seen"][key] = set()
                if isinstance(metric_meta["labels_seen"][key], list):
                    metric_meta["labels_seen"][key] = set(metric_meta["labels_seen"][key])
                metric_meta["labels_seen"][key].add(val)
                # Convert back to list for JSON serialization
                metric_meta["labels_seen"][key] = list(metric_meta["labels_seen"][key])
        
        metric_meta["updated_at"] = datetime.now().isoformat()
        
        # Store updated metric
        await injector.run('storage_kv', {
            "operation": "set",
            "key": f"metric:{name}",
            "value": metric_meta,
            "namespace": "metrics"
        })
        
        return MetricsOutput(
            operation="increment",
            message=f"Incremented metric '{name}' by {value}",
            data={
                "name": name,
                "new_value": metric_meta["value"],
                "labels": labels
            }
        )
            
    except (KeyError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to increment metric '{name}': {e}") from e


async def metrics_decrement(ctx: RunContext[Any], name: str, value: float,
                           labels: Optional[Dict[str, str]]) -> MetricsOutput:
    """
    Decrement a gauge metric.
    
    Args:
        ctx: Runtime context
        name: Metric name
        value: Value to decrement by
        labels: Optional labels
        
    Returns:
        MetricsOutput with new value
        
    Raises:
        KeyError: If metric doesn't exist
        ValueError: If not a gauge metric
        RuntimeError: If storage operation fails
    """
    try:
        if not name or not name.strip():
            raise ValueError("Metric name cannot be empty")
            
        injector = get_injector()
        
        # Get metric metadata - use get_metric which returns None instead of raising
        meta_result = await injector.run('storage_kv', {
            "operation": "get_metric",
            "key": f"metric:{name}",
            "namespace": "metrics"
        })
        
        if hasattr(meta_result, 'output'):
            meta_data = json.loads(meta_result.output)
        else:
            meta_data = meta_result
        
        # Check if metric exists (data is None when not found)
        if meta_data["data"] is None:
            raise KeyError(f"Metric '{name}' does not exist")
        
        metric_meta = meta_data["data"]["value"]
        
        if metric_meta["type"] != MetricType.GAUGE.value:
            raise ValueError(f"Cannot decrement {metric_meta['type']} metric - only gauge metrics can be decremented")
        
        # Update value
        if isinstance(metric_meta["value"], dict):
            # Handle labeled metrics
            label_key = json.dumps(labels or {}, sort_keys=True)
            metric_meta["value"][label_key] = metric_meta["value"].get(label_key, 0) - value
        else:
            metric_meta["value"] -= value
        
        metric_meta["updated_at"] = datetime.now().isoformat()
        
        # Store updated metric
        await injector.run('storage_kv', {
            "operation": "set",
            "key": f"metric:{name}",
            "value": metric_meta,
            "namespace": "metrics"
        })
        
        return MetricsOutput(
            operation="decrement",
            message=f"Decremented metric '{name}' by {value}",
            data={
                "name": name,
                "new_value": metric_meta["value"],
                "labels": labels
            }
        )
            
    except (KeyError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to decrement metric '{name}': {e}") from e


async def metrics_set(ctx: RunContext[Any], name: str, value: float,
                     labels: Optional[Dict[str, str]]) -> MetricsOutput:
    """
    Set a gauge metric to a specific value.
    
    Args:
        ctx: Runtime context
        name: Metric name
        value: Value to set
        labels: Optional labels
        
    Returns:
        MetricsOutput with new value
        
    Raises:
        KeyError: If metric doesn't exist
        ValueError: If not a gauge metric
        RuntimeError: If storage operation fails
    """
    try:
        if not name or not name.strip():
            raise ValueError("Metric name cannot be empty")
            
        injector = get_injector()
        
        # Get metric metadata - use get_metric which returns None instead of raising
        meta_result = await injector.run('storage_kv', {
            "operation": "get_metric",
            "key": f"metric:{name}",
            "namespace": "metrics"
        })
        
        if hasattr(meta_result, 'output'):
            meta_data = json.loads(meta_result.output)
        else:
            meta_data = meta_result
        
        # Check if metric exists (data is None when not found)
        if meta_data["data"] is None:
            # Auto-create gauge metrics for automatic tracking
            if name.startswith("agentool.") and name == "agentool.templates.count":
                # Create the gauge metric automatically
                await metrics_create(ctx, name, MetricType.GAUGE, 
                                   f"Auto-created gauge metric for {name}", "count")
                
                # Get the newly created metric
                meta_result = await injector.run('storage_kv', {
                    "operation": "get_metric",
                    "key": f"metric:{name}",
                    "namespace": "metrics"
                })
                
                if hasattr(meta_result, 'output'):
                    meta_data = json.loads(meta_result.output)
                else:
                    meta_data = meta_result
                
                metric_meta = meta_data["data"]["value"]
            else:
                raise KeyError(f"Metric '{name}' does not exist")
        else:
            metric_meta = meta_data["data"]["value"]
        
        if metric_meta["type"] != MetricType.GAUGE.value:
            raise ValueError(f"Cannot set value for {metric_meta['type']} metric - only gauge metrics can be set")
        
        # Update value
        if labels:
            if not isinstance(metric_meta["value"], dict):
                metric_meta["value"] = {}
            label_key = json.dumps(labels, sort_keys=True)
            metric_meta["value"][label_key] = value
        else:
            metric_meta["value"] = value
        
        metric_meta["updated_at"] = datetime.now().isoformat()
        
        # Store updated metric
        await injector.run('storage_kv', {
            "operation": "set",
            "key": f"metric:{name}",
            "value": metric_meta,
            "namespace": "metrics"
        })
        
        return MetricsOutput(
            operation="set",
            message=f"Set metric '{name}' to {value}",
            data={
                "name": name,
                "value": value,
                "labels": labels
            }
        )
            
    except (KeyError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to set metric '{name}': {e}") from e


async def metrics_observe(ctx: RunContext[Any], name: str, value: float,
                         labels: Optional[Dict[str, str]]) -> MetricsOutput:
    """
    Record an observation for histogram, summary, or timer metrics.
    
    Args:
        ctx: Runtime context
        name: Metric name
        value: Observed value
        labels: Optional labels
        
    Returns:
        MetricsOutput with confirmation
        
    Raises:
        KeyError: If metric doesn't exist
        ValueError: If invalid operation for metric type
        RuntimeError: If storage operation fails
    """
    try:
        if not name or not name.strip():
            raise ValueError("Metric name cannot be empty")
            
        injector = get_injector()
        
        # Get metric metadata - use get_metric which returns None instead of raising
        meta_result = await injector.run('storage_kv', {
            "operation": "get_metric",
            "key": f"metric:{name}",
            "namespace": "metrics"
        })
        
        if hasattr(meta_result, 'output'):
            meta_data = json.loads(meta_result.output)
        else:
            meta_data = meta_result
        
        # Check if metric exists (data is None when not found)
        if meta_data["data"] is None:
            # Auto-create timer/histogram metrics for automatic tracking
            if name.startswith("agentool.") and ("duration" in name or name.endswith(".seconds")):
                # Create the metric automatically as a timer
                await metrics_create(ctx, name, MetricType.TIMER, 
                                   f"Auto-created timer for {name}", "seconds")
                
                # Get the newly created metric
                meta_result = await injector.run('storage_kv', {
                    "operation": "get_metric",
                    "key": f"metric:{name}",
                    "namespace": "metrics"
                })
                
                if hasattr(meta_result, 'output'):
                    meta_data = json.loads(meta_result.output)
                else:
                    meta_data = meta_result
                
                metric_meta = meta_data["data"]["value"]
            else:
                raise KeyError(f"Metric '{name}' does not exist")
        else:
            metric_meta = meta_data["data"]["value"]
        
        if metric_meta["type"] not in [MetricType.HISTOGRAM.value, MetricType.SUMMARY.value, MetricType.TIMER.value]:
            raise ValueError(f"Cannot observe value for {metric_meta['type']} metric")
        
        # Add observation with timestamp
        observation = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "labels": labels or {}
        }
        
        # Keep only last 1000 observations per metric (configurable)
        if "observations" not in metric_meta:
            metric_meta["observations"] = []
        
        metric_meta["observations"].append(observation)
        
        # Limit observations to prevent unbounded growth
        MAX_OBSERVATIONS = 100  # Reduced from 1000 to prevent memory issues
        if len(metric_meta["observations"]) > MAX_OBSERVATIONS:
            metric_meta["observations"] = metric_meta["observations"][-MAX_OBSERVATIONS:]
        
        metric_meta["updated_at"] = datetime.now().isoformat()
        
        # Store updated metric
        await injector.run('storage_kv', {
            "operation": "set",
            "key": f"metric:{name}",
            "value": metric_meta,
            "namespace": "metrics"
        })
        
        return MetricsOutput(
            operation="observe",
            message=f"Recorded observation {value} for metric '{name}'",
            data={
                "name": name,
                "value": value,
                "labels": labels,
                "observation_count": len(metric_meta["observations"])
            }
        )
            
    except (KeyError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to record observation for metric '{name}': {e}") from e


async def metrics_get(ctx: RunContext[Any], name: str) -> MetricsOutput:
    """
    Get a metric's current value and metadata.
    
    Args:
        ctx: Runtime context
        name: Metric name
        
    Returns:
        MetricsOutput with metric data
        
    Raises:
        KeyError: If metric doesn't exist
        ValueError: If invalid metric name
        RuntimeError: If storage operation fails
    """
    try:
        if not name or not name.strip():
            raise ValueError("Metric name cannot be empty")
            
        injector = get_injector()
        
        # Get metric metadata - use get_metric which returns None instead of raising
        meta_result = await injector.run('storage_kv', {
            "operation": "get_metric",
            "key": f"metric:{name}",
            "namespace": "metrics"
        })
        
        if hasattr(meta_result, 'output'):
            meta_data = json.loads(meta_result.output)
        else:
            meta_data = meta_result
        
        # Check if metric exists (data is None when not found)
        if meta_data["data"] is None:
            # Auto-create standard metrics if they don't exist
            if name.startswith("agentool."):
                # Determine metric type based on name pattern
                if name.endswith(".total") or name.endswith(".count"):
                    metric_type = MetricType.COUNTER
                elif name.endswith(".duration") or name.endswith(".seconds"):
                    metric_type = MetricType.TIMER
                elif name == "agentool.templates.count":
                    # Special case for templates count which uses set operation
                    metric_type = MetricType.GAUGE
                else:
                    metric_type = MetricType.COUNTER  # Default to counter
                
                # Create the metric
                await metrics_create(ctx, name, metric_type, 
                                   f"Auto-created metric for {name}", 
                                   "seconds" if metric_type == MetricType.TIMER else "count")
                
                # Get the newly created metric
                meta_result = await injector.run('storage_kv', {
                    "operation": "get_metric",
                    "key": f"metric:{name}",
                    "namespace": "metrics"
                })
                
                if hasattr(meta_result, 'output'):
                    meta_data = json.loads(meta_result.output)
                else:
                    meta_data = meta_result
                
                metric_meta = meta_data["data"]["value"]
            else:
                raise KeyError(f"Metric '{name}' does not exist")
        else:
            metric_meta = meta_data["data"]["value"]
        
        # Calculate statistics for histogram/summary/timer
        stats = None
        if metric_meta["type"] in [MetricType.HISTOGRAM.value, MetricType.SUMMARY.value, MetricType.TIMER.value]:
            if metric_meta.get("observations"):
                values = [obs["value"] for obs in metric_meta["observations"]]
                if values:
                    stats = {
                        "count": len(values),
                        "sum": sum(values),
                        "avg": statistics.mean(values),
                        "min": min(values),
                        "max": max(values),
                        "median": statistics.median(values),
                        "stdev": statistics.stdev(values) if len(values) > 1 else 0
                    }
                    
                    # Calculate percentiles
                    if len(values) >= 2:
                        sorted_values = sorted(values)
                        stats["p50"] = sorted_values[int(len(sorted_values) * 0.50)]
                        stats["p95"] = sorted_values[int(len(sorted_values) * 0.95)]
                        stats["p99"] = sorted_values[int(len(sorted_values) * 0.99)]
        
        return MetricsOutput(
            operation="get",
            message=f"Retrieved metric '{name}'",
            data={
                "name": metric_meta["name"],
                "type": metric_meta["type"],
                "description": metric_meta.get("description", ""),
                "unit": metric_meta.get("unit", ""),
                "value": metric_meta.get("value"),
                "statistics": stats,
                "created_at": metric_meta["created_at"],
                "updated_at": metric_meta["updated_at"]
            }
        )
        
    except (KeyError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to get metric '{name}': {e}") from e


async def metrics_list(ctx: RunContext[Any], pattern: Optional[str]) -> MetricsOutput:
    """
    List all metrics matching a pattern.
    
    Args:
        ctx: Runtime context
        pattern: Optional pattern to filter metrics
        
    Returns:
        MetricsOutput with list of metrics
        
    Raises:
        RuntimeError: If storage operation fails
    """
    try:
        injector = get_injector()
        
        # Get all metric keys
        # If pattern provided, need to prefix it with metric: for storage lookup
        storage_pattern = f"metric:{pattern}" if pattern else "metric:*"
        keys_result = await injector.run('storage_kv', {
            "operation": "keys",
            "pattern": storage_pattern,
            "namespace": "metrics"
        })
        
        if hasattr(keys_result, 'output'):
            keys_data = json.loads(keys_result.output)
        else:
            keys_data = keys_result
        
        # Extract keys from the response
        metric_keys = keys_data.get("data", {}).get("keys", [])
        metrics = []
        
        for key in metric_keys:
            try:
                # Get each metric's metadata
                meta_result = await injector.run('storage_kv', {
                    "operation": "get_metric",
                    "key": key,
                    "namespace": "metrics"
                })
                
                if hasattr(meta_result, 'output'):
                    meta_data = json.loads(meta_result.output)
                else:
                    meta_data = meta_result
                
                # Skip if metric doesn't exist (data is None)
                if meta_data["data"] is None:
                    continue
                
                metric_meta = meta_data["data"]["value"]
                metrics.append({
                    "name": metric_meta["name"],
                    "type": metric_meta["type"],
                    "description": metric_meta.get("description", ""),
                    "unit": metric_meta.get("unit", ""),
                    "updated_at": metric_meta["updated_at"]
                })
            except KeyError:
                # Skip metrics that don't exist (may have been deleted)
                continue
        
        return MetricsOutput(
            operation="list",
            message=f"Found {len(metrics)} metrics",
            data={
                "metrics": metrics,
                "count": len(metrics),
                "pattern": pattern
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to list metrics: {e}") from e


async def metrics_aggregate(ctx: RunContext[Any], pattern: str, aggregation: str,
                           time_range: Optional[int]) -> MetricsOutput:
    """
    Aggregate metrics matching a pattern.
    
    Args:
        ctx: Runtime context
        pattern: Pattern to match metrics
        aggregation: Aggregation function
        time_range: Optional time range in seconds
        
    Returns:
        MetricsOutput with aggregated data
        
    Raises:
        ValueError: If invalid aggregation function
        RuntimeError: If storage operation fails
    """
    try:
        if not pattern or not pattern.strip():
            raise ValueError("Pattern cannot be empty")
            
        if aggregation not in ['sum', 'avg', 'min', 'max', 'count', 'p50', 'p95', 'p99']:
            raise ValueError(f"Invalid aggregation function: {aggregation}")
            
        injector = get_injector()
        
        # Get all matching metrics
        keys_result = await injector.run('storage_kv', {
            "operation": "keys",
            "pattern": f"metric:{pattern}",
            "namespace": "metrics"
        })
        
        if hasattr(keys_result, 'output'):
            keys_data = json.loads(keys_result.output)
        else:
            keys_data = keys_result
        
        metric_keys = keys_data.get("data", {}).get("keys", [])
        aggregated_values = []
        cutoff_time = None
        
        if time_range:
            cutoff_time = datetime.now() - timedelta(seconds=time_range)
        
        for key in metric_keys:
            try:
                # Get metric data
                meta_result = await injector.run('storage_kv', {
                    "operation": "get_metric",
                    "key": key,
                    "namespace": "metrics"
                })
                
                if hasattr(meta_result, 'output'):
                    meta_data = json.loads(meta_result.output)
                else:
                    meta_data = meta_result
                
                # Skip if metric doesn't exist (data is None)
                if meta_data["data"] is None:
                    continue
                
                metric_meta = meta_data["data"]["value"]
                
                # Handle different metric types
                if metric_meta["type"] in [MetricType.COUNTER.value, MetricType.GAUGE.value]:
                    if isinstance(metric_meta["value"], dict):
                        # Labeled metrics
                        for label_values in metric_meta["value"].values():
                            aggregated_values.append(label_values)
                    else:
                        aggregated_values.append(metric_meta["value"])
                        
                elif metric_meta["type"] in [MetricType.HISTOGRAM.value, MetricType.SUMMARY.value, MetricType.TIMER.value]:
                    # Filter by time range if specified
                    observations = metric_meta.get("observations", [])
                    if cutoff_time:
                        observations = [
                            obs for obs in observations
                            if datetime.fromisoformat(obs["timestamp"]) > cutoff_time
                        ]
                    
                    values = [obs["value"] for obs in observations]
                    aggregated_values.extend(values)
            except KeyError:
                # Skip metrics that don't exist
                continue
        
        # Perform aggregation
        result = None
        if aggregated_values:
            if aggregation == "sum":
                result = sum(aggregated_values)
            elif aggregation == "avg":
                result = statistics.mean(aggregated_values)
            elif aggregation == "min":
                result = min(aggregated_values)
            elif aggregation == "max":
                result = max(aggregated_values)
            elif aggregation == "count":
                result = len(aggregated_values)
            elif aggregation == "p50" and len(aggregated_values) > 1:
                sorted_vals = sorted(aggregated_values)
                result = sorted_vals[int(len(sorted_vals) * 0.50)]
            elif aggregation == "p95" and len(aggregated_values) > 1:
                sorted_vals = sorted(aggregated_values)
                result = sorted_vals[int(len(sorted_vals) * 0.95)]
            elif aggregation == "p99" and len(aggregated_values) > 1:
                sorted_vals = sorted(aggregated_values)
                result = sorted_vals[int(len(sorted_vals) * 0.99)]
        
        return MetricsOutput(
            operation="aggregate",
            message=f"Aggregated {len(metric_keys)} metrics",
            data={
                "pattern": pattern,
                "aggregation": aggregation,
                "result": result,
                "metric_count": len(metric_keys),
                "value_count": len(aggregated_values),
                "time_range": time_range
            }
        )
        
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to aggregate metrics: {e}") from e


async def metrics_reset(ctx: RunContext[Any], name: str) -> MetricsOutput:
    """
    Reset a metric to its initial value.
    
    Args:
        ctx: Runtime context
        name: Metric name
        
    Returns:
        MetricsOutput with reset confirmation
        
    Raises:
        KeyError: If metric doesn't exist
        ValueError: If invalid metric name
        RuntimeError: If storage operation fails
    """
    try:
        if not name or not name.strip():
            raise ValueError("Metric name cannot be empty")
            
        injector = get_injector()
        
        # Get metric metadata - use get_metric which returns None instead of raising
        meta_result = await injector.run('storage_kv', {
            "operation": "get_metric",
            "key": f"metric:{name}",
            "namespace": "metrics"
        })
        
        if hasattr(meta_result, 'output'):
            meta_data = json.loads(meta_result.output)
        else:
            meta_data = meta_result
        
        # Check if metric exists (data is None when not found)
        if meta_data["data"] is None:
            raise KeyError(f"Metric '{name}' does not exist")
        
        metric_meta = meta_data["data"]["value"]
        
        # Reset based on type
        if metric_meta["type"] in [MetricType.COUNTER.value, MetricType.GAUGE.value]:
            metric_meta["value"] = 0
        else:
            metric_meta["observations"] = []
        
        metric_meta["updated_at"] = datetime.now().isoformat()
        
        # Store updated metric
        await injector.run('storage_kv', {
            "operation": "set",
            "key": f"metric:{name}",
            "value": metric_meta,
            "namespace": "metrics"
        })
        
        return MetricsOutput(
            operation="reset",
            message=f"Reset metric '{name}'",
            data={"name": name, "type": metric_meta["type"]}
        )
            
    except (KeyError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to reset metric '{name}': {e}") from e


async def metrics_delete(ctx: RunContext[Any], name: str) -> MetricsOutput:
    """
    Delete a metric.
    
    Args:
        ctx: Runtime context
        name: Metric name
        
    Returns:
        MetricsOutput with deletion confirmation
        
    Raises:
        KeyError: If metric doesn't exist
        ValueError: If invalid metric name
        RuntimeError: If storage operation fails
    """
    try:
        if not name or not name.strip():
            raise ValueError("Metric name cannot be empty")
            
        injector = get_injector()
        
        # Check if metric exists first
        try:
            await injector.run('storage_kv', {
                "operation": "get_metric",
                "key": f"metric:{name}",
                "namespace": "metrics"
            })
        except KeyError:
            raise KeyError(f"Metric '{name}' does not exist")
        
        # Delete the metric
        await injector.run('storage_kv', {
            "operation": "delete",
            "key": f"metric:{name}",
            "namespace": "metrics"
        })
        
        return MetricsOutput(
            operation="delete",
            message=f"Deleted metric '{name}'",
            data={"name": name}
        )
            
    except (KeyError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to delete metric '{name}': {e}") from e


async def metrics_export(ctx: RunContext[Any], pattern: Optional[str],
                        format: str) -> MetricsOutput:
    """
    Export metrics in various formats.
    
    Args:
        ctx: Runtime context
        pattern: Optional pattern to filter metrics
        format: Export format (json, prometheus, statsd)
        
    Returns:
        MetricsOutput with exported data
        
    Raises:
        ValueError: If invalid export format
        RuntimeError: If storage operation fails
    """
    try:
        if format not in ['json', 'prometheus', 'statsd']:
            raise ValueError(f"Invalid export format: {format}")
            
        injector = get_injector()
        
        # Get all matching metrics
        keys_result = await injector.run('storage_kv', {
            "operation": "keys",
            "pattern": f"metric:{pattern}" if pattern else "metric:*",
            "namespace": "metrics"
        })
        
        if hasattr(keys_result, 'output'):
            keys_data = json.loads(keys_result.output)
        else:
            keys_data = keys_result
        
        metric_keys = keys_data.get("data", {}).get("keys", [])
        metrics_data = []
        
        for key in metric_keys:
            try:
                # Get metric data
                meta_result = await injector.run('storage_kv', {
                    "operation": "get_metric",
                    "key": key,
                    "namespace": "metrics"
                })
                
                if hasattr(meta_result, 'output'):
                    meta_data = json.loads(meta_result.output)
                else:
                    meta_data = meta_result
                
                # Skip if metric doesn't exist (data is None)
                if meta_data["data"] is None:
                    continue
                
                metrics_data.append(meta_data["data"]["value"])
            except KeyError:
                # Skip metrics that don't exist
                continue
        
        # Format output based on requested format
        if format == "json":
            output = {
                "metrics": metrics_data,
                "exported_at": datetime.now().isoformat(),
                "count": len(metrics_data)
            }
            
        elif format == "prometheus":
            lines = []
            for metric in metrics_data:
                # Format as Prometheus text format
                if metric.get("description"):
                    lines.append(f"# HELP {metric['name']} {metric['description']}")
                lines.append(f"# TYPE {metric['name']} {metric['type']}")
                
                if metric["type"] in [MetricType.COUNTER.value, MetricType.GAUGE.value]:
                    if isinstance(metric["value"], dict):
                        # Labeled metrics
                        for label_json, value in metric["value"].items():
                            labels = json.loads(label_json) if label_json != "{}" else {}
                            label_str = ",".join([f'{k}="{v}"' for k, v in labels.items()])
                            if label_str:
                                lines.append(f"{metric['name']}{{{label_str}}} {value}")
                            else:
                                lines.append(f"{metric['name']} {value}")
                    else:
                        lines.append(f"{metric['name']} {metric['value']}")
            
            output = "\n".join(lines)
            
        elif format == "statsd":
            lines = []
            for metric in metrics_data:
                if metric["type"] == MetricType.COUNTER.value:
                    value = metric["value"] if not isinstance(metric["value"], dict) else sum(metric["value"].values())
                    # Counters should be integers in StatsD format
                    lines.append(f"{metric['name']}:{int(value)}|c")
                elif metric["type"] == MetricType.GAUGE.value:
                    value = metric["value"] if not isinstance(metric["value"], dict) else list(metric["value"].values())[0]
                    lines.append(f"{metric['name']}:{value}|g")
                elif metric["type"] == MetricType.TIMER.value:
                    if metric.get("observations"):
                        for obs in metric["observations"][-10:]:  # Last 10 observations
                            lines.append(f"{metric['name']}:{obs['value']}|ms")
            
            output = "\n".join(lines)
        
        else:
            output = metrics_data
        
        return MetricsOutput(
            operation="export",
            message=f"Exported {len(metrics_data)} metrics in {format} format",
            data={
                "format": format,
                "content": output,
                "metric_count": len(metrics_data)
            }
        )
        
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to export metrics: {e}") from e


async def metrics_create_agent_metrics(
    ctx: RunContext[Any], 
    agent_name: str, 
    routing_config: Dict[str, Any]
) -> MetricsOutput:
    """
    Create all standard metrics for an agent by analyzing its routing config.
    
    This function automatically creates the standard metrics that the injector
    will use when tracking agent executions. It extracts operations from the
    routing config and creates appropriate counter and timer metrics.
    
    Args:
        ctx: Runtime context
        agent_name: Name of the agent
        routing_config: Routing configuration with operation_map
        
    Returns:
        MetricsOutput with creation results
    """
    try:
        created_metrics = []
        skipped_metrics = []
        
        # Extract operations from routing_config
        operations = []
        if isinstance(routing_config, dict) and 'operation_map' in routing_config:
            operations = list(routing_config['operation_map'].keys())
        elif hasattr(routing_config, 'operation_map'):
            operations = list(routing_config.operation_map.keys())
        
        # Define standard agent metrics
        agent_metrics = {
            f'agentool.{agent_name}.executions.total': MetricDefinition(
                type=MetricType.COUNTER,
                description=f"Total executions for {agent_name}",
                unit="count"
            ),
            f'agentool.{agent_name}.executions.success': MetricDefinition(
                type=MetricType.COUNTER,
                description=f"Successful executions for {agent_name}",
                unit="count"
            ),
            f'agentool.{agent_name}.executions.failure': MetricDefinition(
                type=MetricType.COUNTER,
                description=f"Failed executions for {agent_name}",
                unit="count"
            ),
            f'agentool.{agent_name}.duration.seconds': MetricDefinition(
                type=MetricType.TIMER,
                description=f"Execution duration for {agent_name}",
                unit="seconds"
            ),
        }
        
        # Add operation-specific metrics
        for op in operations:
            agent_metrics[f'agentool.{agent_name}.{op}.count'] = MetricDefinition(
                type=MetricType.COUNTER,
                description=f"Count of {op} operations for {agent_name}",
                unit="count"
            )
            agent_metrics[f'agentool.{agent_name}.{op}.duration'] = MetricDefinition(
                type=MetricType.TIMER,
                description=f"Duration of {op} operations for {agent_name}",
                unit="seconds"
            )
        
        # Create all metrics
        for name, definition in agent_metrics.items():
            try:
                await metrics_create(ctx, name, definition.type, definition.description, definition.unit)
                created_metrics.append(name)
            except ValueError as e:
                # Metric already exists, that's fine
                if "already exists" in str(e):
                    skipped_metrics.append(name)
                else:
                    raise
        
        return MetricsOutput(
            operation="create_agent_metrics",
            message=f"Created {len(created_metrics)} metrics for agent '{agent_name}' ({len(skipped_metrics)} already existed)",
            data={
                "agent_name": agent_name,
                "operations": operations,
                "created_metrics": created_metrics,
                "skipped_metrics": skipped_metrics,
                "total_metrics": len(agent_metrics)
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to create agent metrics for '{agent_name}': {e}") from e


async def metrics_create_from_schema(
    ctx: RunContext[Any],
    metrics_schema: MetricsSchema
) -> MetricsOutput:
    """
    Create multiple metrics from a strict schema.
    
    This allows batch creation of custom metrics with strict type checking.
    Each metric in the schema is created with its specified type, description,
    and unit.
    
    Args:
        ctx: Runtime context
        metrics_schema: Strict schema defining metrics to create
        
    Returns:
        MetricsOutput with creation results
    """
    try:
        created_metrics = []
        failed_metrics = []
        skipped_metrics = []
        
        for name, definition in metrics_schema.metrics.items():
            try:
                await metrics_create(
                    ctx, 
                    name, 
                    definition.type, 
                    definition.description, 
                    definition.unit
                )
                created_metrics.append(name)
            except ValueError as e:
                if "already exists" in str(e):
                    skipped_metrics.append(name)
                else:
                    failed_metrics.append({"name": name, "error": str(e)})
            except Exception as e:
                failed_metrics.append({"name": name, "error": str(e)})
        
        return MetricsOutput(
            operation="create_from_schema",
            message=f"Created {len(created_metrics)} metrics, {len(skipped_metrics)} skipped, {len(failed_metrics)} failed",
            data={
                "created": created_metrics,
                "skipped": skipped_metrics,
                "failed": failed_metrics,
                "total_requested": len(metrics_schema.metrics)
            }
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to create metrics from schema: {e}") from e


def create_metrics_agent():
    """
    Create and return the Metrics AgenTool.
    
    Returns:
        Agent configured for metrics operations
    """
    metrics_routing = RoutingConfig(
        operation_field='operation',
        operation_map={
            'create': ('metrics_create', lambda x: {
                'name': x.name, 'type': x.type,
                'description': x.description, 'unit': x.unit
            }),
            'increment': ('metrics_increment', lambda x: {
                'name': x.name, 'value': x.value or 1,
                'labels': x.labels
            }),
            'decrement': ('metrics_decrement', lambda x: {
                'name': x.name, 'value': x.value or 1,
                'labels': x.labels
            }),
            'set': ('metrics_set', lambda x: {
                'name': x.name, 'value': x.value,
                'labels': x.labels
            }),
            'observe': ('metrics_observe', lambda x: {
                'name': x.name, 'value': x.value,
                'labels': x.labels
            }),
            'get': ('metrics_get', lambda x: {
                'name': x.name
            }),
            'list': ('metrics_list', lambda x: {
                'pattern': x.pattern
            }),
            'aggregate': ('metrics_aggregate', lambda x: {
                'pattern': x.pattern, 'aggregation': x.aggregation,
                'time_range': x.time_range
            }),
            'reset': ('metrics_reset', lambda x: {
                'name': x.name
            }),
            'delete': ('metrics_delete', lambda x: {
                'name': x.name
            }),
            'export': ('metrics_export', lambda x: {
                'pattern': x.pattern, 'format': x.format or 'json'
            }),
            'create_agent_metrics': ('metrics_create_agent_metrics', lambda x: {
                'agent_name': x.agent_name, 'routing_config': x.routing_config
            }),
            'create_from_schema': ('metrics_create_from_schema', lambda x: {
                'metrics_schema': x.metrics_schema
            }),
        }
    )
    
    return create_agentool(
        name='metrics',
        input_schema=MetricsInput,
        routing_config=metrics_routing,
        tools=[
            metrics_create, metrics_increment, metrics_decrement,
            metrics_set, metrics_observe, metrics_get,
            metrics_list, metrics_aggregate, metrics_reset,
            metrics_delete, metrics_export, metrics_create_agent_metrics,
            metrics_create_from_schema
        ],
        output_type=MetricsOutput,
        system_prompt="Track and analyze metrics for observability and monitoring.",
        description="Comprehensive metrics toolkit with automatic and manual tracking",
        version="1.0.0",
        tags=["metrics", "observability", "monitoring", "analytics"],
        dependencies=["storage_kv"],
        examples=[
            {
                "description": "Create a counter metric",
                "input": {
                    "operation": "create",
                    "name": "api.requests.total",
                    "type": "counter",
                    "description": "Total API requests"
                },
                "output": {
                    "operation": "create",
                    "message": "Created counter metric 'api.requests.total'"
                }
            },
            {
                "description": "Increment a counter",
                "input": {
                    "operation": "increment",
                    "name": "api.requests.total",
                    "value": 1,
                    "labels": {"endpoint": "/users", "method": "GET"}
                },
                "output": {
                    "operation": "increment",
                    "message": "Incremented metric 'api.requests.total' by 1"
                }
            },
            {
                "description": "Record a timer observation",
                "input": {
                    "operation": "observe",
                    "name": "api.latency",
                    "value": 0.125,
                    "labels": {"endpoint": "/users"}
                },
                "output": {
                    "operation": "observe",
                    "message": "Recorded observation 0.125 for metric 'api.latency'"
                }
            }
        ]
    )


# Create the agent instance
agent = create_metrics_agent()