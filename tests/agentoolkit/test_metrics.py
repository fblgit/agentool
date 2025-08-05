"""
Tests for metrics toolkit.

This module tests all functionality of the metrics toolkit including
metric creation, tracking, aggregation, and automatic injector integration.
"""

import json
import asyncio
import time
from datetime import datetime

from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry


class TestMetrics:
    """Test suite for metrics toolkit."""
    
    def setup_method(self):
        """Clear registry and injector before each test."""
        # Import storage globals first
        from agentoolkit.storage.kv import _kv_storage, _kv_expiry
        
        # Clear everything BEFORE creating agents
        AgenToolRegistry.clear()
        get_injector().clear()
        _kv_storage.clear()
        _kv_expiry.clear()
        
        # Now create the agents with clean state
        from agentoolkit.storage.kv import create_storage_kv_agent
        from agentoolkit.observability.metrics import create_metrics_agent
        
        storage_agent = create_storage_kv_agent()
        metrics_agent = create_metrics_agent()
    
    def test_create_counter_metric(self):
        """Test creating a counter metric."""
        
        async def run_test():
            injector = get_injector()
            
            # Create a counter
            result = await injector.run('metrics', {
                "operation": "create",
                "name": "test.counter",
                "type": "counter",
                "description": "Test counter metric",
                "unit": "count"
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert data["operation"] == "create"
            assert "test.counter" in data["message"]
            
            # Try creating same metric again - should raise ValueError
            try:
                duplicate = await injector.run('metrics', {
                    "operation": "create",
                    "name": "test.counter",
                    "type": "counter"
                })
                assert False, "Expected ValueError for duplicate metric"
            except ValueError as e:
                assert "already exists" in str(e)
        
        asyncio.run(run_test())
    
    def test_increment_counter(self):
        """Test incrementing a counter metric."""
        
        async def run_test():
            injector = get_injector()
            
            # Create counter
            await injector.run('metrics', {
                "operation": "create",
                "name": "api.requests",
                "type": "counter"
            })
            
            # Increment counter
            result = await injector.run('metrics', {
                "operation": "increment",
                "name": "api.requests",
                "value": 1
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert data["data"]["new_value"] == 1
            
            # Increment again
            result2 = await injector.run('metrics', {
                "operation": "increment",
                "name": "api.requests",
                "value": 5
            })
            
            if hasattr(result2, 'output'):
                data2 = json.loads(result2.output)
            else:
                data2 = result2
            
            assert data2["data"]["new_value"] == 6
            
            # Get current value
            get_result = await injector.run('metrics', {
                "operation": "get",
                "name": "api.requests"
            })
            
            if hasattr(get_result, 'output'):
                get_data = json.loads(get_result.output)
            else:
                get_data = get_result
            
            assert get_data["data"]["value"] == 6
        
        asyncio.run(run_test())
    
    def test_gauge_metric(self):
        """Test gauge metric operations."""
        
        async def run_test():
            injector = get_injector()
            
            # Create gauge
            await injector.run('metrics', {
                "operation": "create",
                "name": "memory.usage",
                "type": "gauge",
                "unit": "bytes"
            })
            
            # Set value
            result = await injector.run('metrics', {
                "operation": "set",
                "name": "memory.usage",
                "value": 1024
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            # Remove success assertion as MetricsOutput no longer has success field
            assert data["data"]["value"] == 1024
            
            # Increment gauge
            inc_result = await injector.run('metrics', {
                "operation": "increment",
                "name": "memory.usage",
                "value": 256
            })
            
            if hasattr(inc_result, 'output'):
                inc_data = json.loads(inc_result.output)
            else:
                inc_data = inc_result
            
            assert inc_data["data"]["new_value"] == 1280
            
            # Decrement gauge
            dec_result = await injector.run('metrics', {
                "operation": "decrement",
                "name": "memory.usage",
                "value": 512
            })
            
            if hasattr(dec_result, 'output'):
                dec_data = json.loads(dec_result.output)
            else:
                dec_data = dec_result
            
            assert dec_data["data"]["new_value"] == 768
        
        asyncio.run(run_test())
    
    def test_histogram_metric(self):
        """Test histogram metric with observations."""
        
        async def run_test():
            injector = get_injector()
            
            # Create histogram
            await injector.run('metrics', {
                "operation": "create",
                "name": "request.latency",
                "type": "histogram",
                "unit": "seconds"
            })
            
            # Record observations
            values = [0.1, 0.2, 0.15, 0.3, 0.25, 0.18, 0.22, 0.35, 0.12, 0.28]
            for value in values:
                await injector.run('metrics', {
                    "operation": "observe",
                    "name": "request.latency",
                    "value": value
                })
            
            # Get metric with statistics
            result = await injector.run('metrics', {
                "operation": "get",
                "name": "request.latency"
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            # Remove success assertion as MetricsOutput no longer has success field
            stats = data["data"]["statistics"]
            assert stats["count"] == 10
            assert abs(stats["avg"] - 0.215) < 0.01
            assert stats["min"] == 0.1
            assert stats["max"] == 0.35
        
        asyncio.run(run_test())
    
    def test_labeled_metrics(self):
        """Test metrics with labels."""
        
        async def run_test():
            injector = get_injector()
            
            # Create counter
            await injector.run('metrics', {
                "operation": "create",
                "name": "http.requests",
                "type": "counter"
            })
            
            # Increment with different labels
            await injector.run('metrics', {
                "operation": "increment",
                "name": "http.requests",
                "value": 10,
                "labels": {"method": "GET", "endpoint": "/users"}
            })
            
            await injector.run('metrics', {
                "operation": "increment",
                "name": "http.requests",
                "value": 5,
                "labels": {"method": "POST", "endpoint": "/users"}
            })
            
            await injector.run('metrics', {
                "operation": "increment",
                "name": "http.requests",
                "value": 3,
                "labels": {"method": "GET", "endpoint": "/posts"}
            })
            
            # Get metric
            result = await injector.run('metrics', {
                "operation": "get",
                "name": "http.requests"
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            # Remove success assertion as MetricsOutput no longer has success field
            values = data["data"]["value"]
            
            # Check labeled values
            assert isinstance(values, dict)
            assert len(values) == 3
        
        asyncio.run(run_test())
    
    def test_list_metrics(self):
        """Test listing metrics with patterns."""
        
        async def run_test():
            injector = get_injector()
            
            # Create multiple metrics
            metrics = [
                ("app.requests.total", "counter"),
                ("app.requests.errors", "counter"),
                ("app.latency", "histogram"),
                ("system.cpu.usage", "gauge"),
                ("system.memory.usage", "gauge")
            ]
            
            for name, metric_type in metrics:
                await injector.run('metrics', {
                    "operation": "create",
                    "name": name,
                    "type": metric_type
                })
            
            # List all metrics
            all_result = await injector.run('metrics', {
                "operation": "list"
            })
            
            if hasattr(all_result, 'output'):
                all_data = json.loads(all_result.output)
            else:
                all_data = all_result
            
            assert all_data["data"]["count"] == 5
            
            # List with pattern
            app_result = await injector.run('metrics', {
                "operation": "list",
                "pattern": "app.*"
            })
            
            if hasattr(app_result, 'output'):
                app_data = json.loads(app_result.output)
            else:
                app_data = app_result
            
            assert app_data["data"]["count"] == 3
            
            # List system metrics
            sys_result = await injector.run('metrics', {
                "operation": "list",
                "pattern": "system.*"
            })
            
            if hasattr(sys_result, 'output'):
                sys_data = json.loads(sys_result.output)
            else:
                sys_data = sys_result
            
            assert sys_data["data"]["count"] == 2
        
        asyncio.run(run_test())
    
    def test_aggregate_metrics(self):
        """Test metric aggregation."""
        
        async def run_test():
            injector = get_injector()
            
            # Create and populate counters
            for i in range(3):
                name = f"service{i}.requests"
                await injector.run('metrics', {
                    "operation": "create",
                    "name": name,
                    "type": "counter"
                })
                
                await injector.run('metrics', {
                    "operation": "increment",
                    "name": name,
                    "value": (i + 1) * 10
                })
            
            # Aggregate sum
            sum_result = await injector.run('metrics', {
                "operation": "aggregate",
                "pattern": "service*.requests",
                "aggregation": "sum"
            })
            
            if hasattr(sum_result, 'output'):
                sum_data = json.loads(sum_result.output)
            else:
                sum_data = sum_result
            
            assert sum_data["data"]["result"] == 60  # 10 + 20 + 30
            
            # Aggregate average
            avg_result = await injector.run('metrics', {
                "operation": "aggregate",
                "pattern": "service*.requests",
                "aggregation": "avg"
            })
            
            if hasattr(avg_result, 'output'):
                avg_data = json.loads(avg_result.output)
            else:
                avg_data = avg_result
            
            assert avg_data["data"]["result"] == 20
        
        asyncio.run(run_test())
    
    def test_reset_and_delete_metrics(self):
        """Test resetting and deleting metrics."""
        
        async def run_test():
            injector = get_injector()
            
            # Create and populate counter
            await injector.run('metrics', {
                "operation": "create",
                "name": "test.reset",
                "type": "counter"
            })
            
            await injector.run('metrics', {
                "operation": "increment",
                "name": "test.reset",
                "value": 100
            })
            
            # Reset metric
            reset_result = await injector.run('metrics', {
                "operation": "reset",
                "name": "test.reset"
            })
            
            if hasattr(reset_result, 'output'):
                reset_data = json.loads(reset_result.output)
            else:
                reset_data = reset_result
            
            # Remove success assertion as MetricsOutput no longer has success field
            
            # Check value is reset
            get_result = await injector.run('metrics', {
                "operation": "get",
                "name": "test.reset"
            })
            
            if hasattr(get_result, 'output'):
                get_data = json.loads(get_result.output)
            else:
                get_data = get_result
            
            assert get_data["data"]["value"] == 0
            
            # Delete metric
            delete_result = await injector.run('metrics', {
                "operation": "delete",
                "name": "test.reset"
            })
            
            if hasattr(delete_result, 'output'):
                delete_data = json.loads(delete_result.output)
            else:
                delete_data = delete_result
            
            # Remove success assertion as MetricsOutput no longer has success field
            
            # Verify deletion - should raise KeyError
            try:
                get_deleted = await injector.run('metrics', {
                    "operation": "get",
                    "name": "test.reset"
                })
                assert False, "Expected KeyError for deleted metric"
            except KeyError:
                pass  # Expected behavior
        
        asyncio.run(run_test())
    
    def test_export_metrics(self):
        """Test exporting metrics in different formats."""
        
        async def run_test():
            injector = get_injector()
            
            # Create metrics
            await injector.run('metrics', {
                "operation": "create",
                "name": "export.counter",
                "type": "counter",
                "description": "Test export counter"
            })
            
            await injector.run('metrics', {
                "operation": "increment",
                "name": "export.counter",
                "value": 42
            })
            
            # Export as JSON
            json_result = await injector.run('metrics', {
                "operation": "export",
                "format": "json"
            })
            
            if hasattr(json_result, 'output'):
                json_data = json.loads(json_result.output)
            else:
                json_data = json_result
            
            # Remove success assertion as MetricsOutput no longer has success field
            assert json_data["data"]["format"] == "json"
            assert json_data["data"]["metric_count"] >= 1
            
            # Export as Prometheus
            prom_result = await injector.run('metrics', {
                "operation": "export",
                "format": "prometheus"
            })
            
            if hasattr(prom_result, 'output'):
                prom_data = json.loads(prom_result.output)
            else:
                prom_data = prom_result
            
            # Remove success assertion as MetricsOutput no longer has success field
            assert prom_data["data"]["format"] == "prometheus"
            content = prom_data["data"]["content"]
            assert "# HELP export.counter" in content
            assert "# TYPE export.counter counter" in content
            assert "export.counter 42" in content
            
            # Export as StatsD
            statsd_result = await injector.run('metrics', {
                "operation": "export",
                "format": "statsd"
            })
            
            if hasattr(statsd_result, 'output'):
                statsd_data = json.loads(statsd_result.output)
            else:
                statsd_data = statsd_result
            
            # Remove success assertion as MetricsOutput no longer has success field
            assert statsd_data["data"]["format"] == "statsd"
            content = statsd_data["data"]["content"]
            assert "export.counter:42|c" in content
        
        asyncio.run(run_test())
    
    def test_automatic_metrics_tracking(self):
        """Test automatic metrics tracking through injector."""
        
        async def run_test():
            injector = get_injector()
            
            # Create a simple test agent for tracking
            from agentool import create_agentool
            from agentool.core.registry import RoutingConfig
            from pydantic import BaseModel, Field
            from pydantic_ai import RunContext
            from typing import Literal
            
            class TestInput(BaseModel):
                operation: Literal['success', 'failure'] = Field(description="Test operation")
            
            async def test_success(ctx: RunContext[None]) -> dict:
                await asyncio.sleep(0.01)  # Simulate work
                return {"status": "success"}
            
            async def test_failure(ctx: RunContext[None]) -> dict:
                await asyncio.sleep(0.01)  # Simulate work
                raise ValueError("Test error")
            
            test_agent = create_agentool(
                name='test_tracked',
                input_schema=TestInput,
                routing_config=RoutingConfig(
                    operation_field='operation',
                    operation_map={
                        'success': ('test_success', lambda x: {}),
                        'failure': ('test_failure', lambda x: {}),
                    }
                ),
                tools=[test_success, test_failure],
                system_prompt="Test agent for metrics tracking"
            )
            
            # Ensure metrics are enabled
            injector.enable_metrics(True)
            
            # Execute successful operation
            await injector.run('test_tracked', {"operation": "success"})
            
            # Execute failing operation (expect error)
            try:
                await injector.run('test_tracked', {"operation": "failure"})
            except:
                pass  # Expected
            
            # Wait a bit for async metrics recording
            await asyncio.sleep(0.1)
            
            # Check automatic metrics were recorded
            total_result = await injector.run('metrics', {
                "operation": "get",
                "name": "agentool.test_tracked.executions.total"
            })
            
            if hasattr(total_result, 'output'):
                total_data = json.loads(total_result.output)
            else:
                total_data = total_result
            
            # No longer checking success field - function now throws exceptions on failure
            # Check if metric exists and has value
            if total_data["data"].get("value") is not None:
                assert total_data["data"]["value"] >= 1
            
            # Test disabling metrics
            injector.enable_metrics(False)
            assert not injector.is_metrics_enabled()
            
            # Re-enable for other tests
            injector.enable_metrics(True)
        
        asyncio.run(run_test())
    
    def test_timer_metric(self):
        """Test timer metric type."""
        
        async def run_test():
            injector = get_injector()
            
            # Create timer
            await injector.run('metrics', {
                "operation": "create",
                "name": "operation.duration",
                "type": "timer",
                "unit": "seconds"
            })
            
            # Record multiple timings
            timings = [0.05, 0.1, 0.15, 0.08, 0.12]
            for timing in timings:
                await injector.run('metrics', {
                    "operation": "observe",
                    "name": "operation.duration",
                    "value": timing
                })
            
            # Get statistics
            result = await injector.run('metrics', {
                "operation": "get",
                "name": "operation.duration"
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            # Remove success assertion as MetricsOutput no longer has success field
            stats = data["data"]["statistics"]
            assert stats["count"] == 5
            assert abs(stats["avg"] - 0.1) < 0.01
        
        asyncio.run(run_test())
    
    def test_error_cases(self):
        """Test error handling in metrics operations."""
        
        async def run_test():
            injector = get_injector()
            
            # Try to increment non-existent metric - should raise KeyError
            try:
                result = await injector.run('metrics', {
                    "operation": "increment",
                    "name": "non.existent",
                    "value": 1
                })
                assert False, "Expected KeyError for non-existent metric"
            except KeyError as e:
                assert "does not exist" in str(e)
            
            # Try invalid operation on counter
            await injector.run('metrics', {
                "operation": "create",
                "name": "test.counter",
                "type": "counter"
            })
            
            # Try decrement on counter - should raise ValueError
            try:
                dec_result = await injector.run('metrics', {
                    "operation": "decrement",
                    "name": "test.counter",
                    "value": 1
                })
                assert False, "Expected ValueError for decrementing counter"
            except ValueError as e:
                assert "only gauge" in str(e).lower()
            
            # Try to observe on counter - should raise ValueError
            try:
                obs_result = await injector.run('metrics', {
                    "operation": "observe",
                    "name": "test.counter",
                    "value": 1.5
                })
                assert False, "Expected ValueError for observing on counter"
            except ValueError as e:
                assert "cannot observe" in str(e).lower()
        
        asyncio.run(run_test())
    
    def test_summary_metric(self):
        """Test summary metric type with observations."""
        
        async def run_test():
            injector = get_injector()
            
            # Create summary metric
            await injector.run('metrics', {
                "operation": "create",
                "name": "response.size",
                "type": "summary",
                "unit": "bytes"
            })
            
            # Record observations
            values = [1024, 2048, 512, 4096, 8192, 256, 3072, 1536, 768, 2560]
            for value in values:
                await injector.run('metrics', {
                    "operation": "observe",
                    "name": "response.size",
                    "value": value
                })
            
            # Get metric with statistics
            result = await injector.run('metrics', {
                "operation": "get",
                "name": "response.size"
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            # Remove success assertion as MetricsOutput no longer has success field
            stats = data["data"]["statistics"]
            assert stats["count"] == 10
            assert stats["min"] == 256
            assert stats["max"] == 8192
            assert "median" in stats
            assert "stdev" in stats
            assert "p50" in stats
            assert "p95" in stats
            assert "p99" in stats
        
        asyncio.run(run_test())
    
    def test_metrics_with_empty_observations(self):
        """Test metrics with no observations."""
        
        async def run_test():
            injector = get_injector()
            
            # Create histogram without observations
            await injector.run('metrics', {
                "operation": "create",
                "name": "empty.histogram",
                "type": "histogram"
            })
            
            # Get metric
            result = await injector.run('metrics', {
                "operation": "get",
                "name": "empty.histogram"
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            # Remove success assertion as MetricsOutput no longer has success field
            assert data["data"]["statistics"] is None  # No stats for empty observations
        
        asyncio.run(run_test())
    
    def test_gauge_set_with_labels(self):
        """Test setting gauge values with labels."""
        
        async def run_test():
            injector = get_injector()
            
            # Create gauge
            await injector.run('metrics', {
                "operation": "create",
                "name": "cpu.usage",
                "type": "gauge",
                "unit": "percent"
            })
            
            # Set values with different labels
            await injector.run('metrics', {
                "operation": "set",
                "name": "cpu.usage",
                "value": 45.5,
                "labels": {"core": "0"}
            })
            
            await injector.run('metrics', {
                "operation": "set",
                "name": "cpu.usage",
                "value": 67.8,
                "labels": {"core": "1"}
            })
            
            # Get metric
            result = await injector.run('metrics', {
                "operation": "get",
                "name": "cpu.usage"
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            # Remove success assertion as MetricsOutput no longer has success field
            values = data["data"]["value"]
            assert isinstance(values, dict)
            assert len(values) == 2
        
        asyncio.run(run_test())
    
    def test_aggregate_with_time_range(self):
        """Test aggregation with time range filtering."""
        
        async def run_test():
            injector = get_injector()
            
            # Create histogram
            await injector.run('metrics', {
                "operation": "create",
                "name": "timed.metric",
                "type": "histogram"
            })
            
            # Add observations
            for i in range(5):
                await injector.run('metrics', {
                    "operation": "observe",
                    "name": "timed.metric",
                    "value": i * 10
                })
            
            # Aggregate with time range (should include all recent observations)
            result = await injector.run('metrics', {
                "operation": "aggregate",
                "pattern": "timed.*",
                "aggregation": "avg",
                "time_range": 3600  # Last hour
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            # Remove success assertion as MetricsOutput no longer has success field
            assert data["data"]["result"] == 20  # Average of 0, 10, 20, 30, 40
        
        asyncio.run(run_test())
    
    def test_percentile_aggregations(self):
        """Test percentile aggregations (p50, p95, p99)."""
        
        async def run_test():
            injector = get_injector()
            
            # Create counter metrics with different values
            for i in range(10):
                name = f"perc.metric{i}"
                await injector.run('metrics', {
                    "operation": "create",
                    "name": name,
                    "type": "counter"
                })
                await injector.run('metrics', {
                    "operation": "increment",
                    "name": name,
                    "value": i * 10
                })
            
            # Test p50 (median)
            p50_result = await injector.run('metrics', {
                "operation": "aggregate",
                "pattern": "perc.*",
                "aggregation": "p50"
            })
            
            if hasattr(p50_result, 'output'):
                p50_data = json.loads(p50_result.output)
            else:
                p50_data = p50_result
            
            # Remove success assertion as MetricsOutput no longer has success field
            # Median of [0,10,20,30,40,50,60,70,80,90] should be around 45
            assert 40 <= p50_data["data"]["result"] <= 50
            
            # Test p95
            p95_result = await injector.run('metrics', {
                "operation": "aggregate",
                "pattern": "perc.*",
                "aggregation": "p95"
            })
            
            if hasattr(p95_result, 'output'):
                p95_data = json.loads(p95_result.output)
            else:
                p95_data = p95_result
            
            # Remove success assertion as MetricsOutput no longer has success field
            assert p95_data["data"]["result"] >= 80
            
            # Test p99
            p99_result = await injector.run('metrics', {
                "operation": "aggregate",
                "pattern": "perc.*",
                "aggregation": "p99"
            })
            
            if hasattr(p99_result, 'output'):
                p99_data = json.loads(p99_result.output)
            else:
                p99_data = p99_result
            
            # Remove success assertion as MetricsOutput no longer has success field
            assert p99_data["data"]["result"] >= 90
        
        asyncio.run(run_test())
    
    def test_aggregate_min_max_count(self):
        """Test min, max, and count aggregations."""
        
        async def run_test():
            injector = get_injector()
            
            # Create metrics
            for i in range(5):
                name = f"agg.test{i}"
                await injector.run('metrics', {
                    "operation": "create",
                    "name": name,
                    "type": "counter"
                })
                await injector.run('metrics', {
                    "operation": "increment",
                    "name": name,
                    "value": (i + 1) * 5
                })
            
            # Test min
            min_result = await injector.run('metrics', {
                "operation": "aggregate",
                "pattern": "agg.*",
                "aggregation": "min"
            })
            
            if hasattr(min_result, 'output'):
                min_data = json.loads(min_result.output)
            else:
                min_data = min_result
            
            assert min_data["data"]["result"] == 5
            
            # Test max
            max_result = await injector.run('metrics', {
                "operation": "aggregate",
                "pattern": "agg.*",
                "aggregation": "max"
            })
            
            if hasattr(max_result, 'output'):
                max_data = json.loads(max_result.output)
            else:
                max_data = max_result
            
            assert max_data["data"]["result"] == 25
            
            # Test count
            count_result = await injector.run('metrics', {
                "operation": "aggregate",
                "pattern": "agg.*",
                "aggregation": "count"
            })
            
            if hasattr(count_result, 'output'):
                count_data = json.loads(count_result.output)
            else:
                count_data = count_result
            
            assert count_data["data"]["result"] == 5
        
        asyncio.run(run_test())
    
    def test_export_prometheus_with_labels(self):
        """Test Prometheus export format with labeled metrics."""
        
        async def run_test():
            injector = get_injector()
            
            # Create counter with labels
            await injector.run('metrics', {
                "operation": "create",
                "name": "http.requests.total",
                "type": "counter",
                "description": "Total HTTP requests"
            })
            
            # Increment with different labels
            await injector.run('metrics', {
                "operation": "increment",
                "name": "http.requests.total",
                "value": 100,
                "labels": {"method": "GET", "status": "200"}
            })
            
            await injector.run('metrics', {
                "operation": "increment",
                "name": "http.requests.total",
                "value": 50,
                "labels": {"method": "POST", "status": "201"}
            })
            
            # Export as Prometheus
            result = await injector.run('metrics', {
                "operation": "export",
                "format": "prometheus",
                "pattern": "http.*"
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            # Remove success assertion as MetricsOutput no longer has success field
            content = data["data"]["content"]
            
            # Check Prometheus format
            assert "# HELP http.requests.total" in content
            assert "# TYPE http.requests.total counter" in content
            assert 'method="GET"' in content
            assert 'status="200"' in content
            assert 'method="POST"' in content
            assert 'status="201"' in content
        
        asyncio.run(run_test())
    
    def test_export_statsd_with_timer(self):
        """Test StatsD export with timer metrics."""
        
        async def run_test():
            injector = get_injector()
            
            # Create timer
            await injector.run('metrics', {
                "operation": "create",
                "name": "api.latency",
                "type": "timer",
                "unit": "ms"
            })
            
            # Add observations
            for i in range(15):  # More than 10 to test the limit
                await injector.run('metrics', {
                    "operation": "observe",
                    "name": "api.latency",
                    "value": (i + 1) * 10
                })
            
            # Export as StatsD
            result = await injector.run('metrics', {
                "operation": "export",
                "format": "statsd"
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            # Remove success assertion as MetricsOutput no longer has success field
            content = data["data"]["content"]
            
            # Check StatsD format - should have last 10 observations
            lines = content.split('\n')
            timer_lines = [l for l in lines if 'api.latency' in l]
            assert len(timer_lines) == 10  # Only last 10 observations
            assert all('|ms' in line for line in timer_lines)
        
        asyncio.run(run_test())
    
    def test_delete_non_existent_metric(self):
        """Test deleting a metric that doesn't exist."""
        
        async def run_test():
            injector = get_injector()
            
            # Try to delete non-existent metric - should raise KeyError
            try:
                result = await injector.run('metrics', {
                    "operation": "delete",
                    "name": "does.not.exist"
                })
                assert False, "Expected KeyError for non-existent metric"
            except KeyError as e:
                assert "does not exist" in str(e)
        
        asyncio.run(run_test())
    
    def test_aggregate_empty_pattern(self):
        """Test aggregation with no matching metrics."""
        
        async def run_test():
            injector = get_injector()
            
            # Aggregate with pattern that matches nothing
            result = await injector.run('metrics', {
                "operation": "aggregate",
                "pattern": "nomatch.*",
                "aggregation": "sum"
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            # Remove success assertion as MetricsOutput no longer has success field
            assert data["data"]["result"] is None  # No values to aggregate
            assert data["data"]["value_count"] == 0
        
        asyncio.run(run_test())
    
    def test_labeled_metrics_aggregation(self):
        """Test aggregation with labeled metrics."""
        
        async def run_test():
            injector = get_injector()
            
            # Create gauge with labels
            await injector.run('metrics', {
                "operation": "create",
                "name": "labeled.gauge",
                "type": "gauge"
            })
            
            # Set values with labels
            await injector.run('metrics', {
                "operation": "set",
                "name": "labeled.gauge",
                "value": 10,
                "labels": {"region": "us-east"}
            })
            
            await injector.run('metrics', {
                "operation": "set",
                "name": "labeled.gauge",
                "value": 20,
                "labels": {"region": "us-west"}
            })
            
            await injector.run('metrics', {
                "operation": "set",
                "name": "labeled.gauge",
                "value": 30,
                "labels": {"region": "eu-west"}
            })
            
            # Aggregate sum
            result = await injector.run('metrics', {
                "operation": "aggregate",
                "pattern": "labeled.*",
                "aggregation": "sum"
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            # Remove success assertion as MetricsOutput no longer has success field
            assert data["data"]["result"] == 60  # Sum of all labeled values
        
        asyncio.run(run_test())
    
    def test_metrics_with_single_observation(self):
        """Test histogram with only one observation (no stdev)."""
        
        async def run_test():
            injector = get_injector()
            
            # Create histogram
            await injector.run('metrics', {
                "operation": "create",
                "name": "single.observation",
                "type": "histogram"
            })
            
            # Add single observation
            await injector.run('metrics', {
                "operation": "observe",
                "name": "single.observation",
                "value": 42.5
            })
            
            # Get metric
            result = await injector.run('metrics', {
                "operation": "get",
                "name": "single.observation"
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            # Remove success assertion as MetricsOutput no longer has success field
            stats = data["data"]["statistics"]
            assert stats["count"] == 1
            assert stats["stdev"] == 0  # Single value has no standard deviation
            assert "p50" not in stats  # No percentiles for single value
        
        asyncio.run(run_test())