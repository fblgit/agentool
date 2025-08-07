"""
Dependency injection system for AgenTools.

This module provides a clean dependency injection system that integrates
with the AgenTool registry to manage inter-agent dependencies.
"""

from __future__ import annotations

import json
import time
import asyncio
import os
from datetime import datetime, date
from datetime import time as datetime_time
from decimal import Decimal
from typing import Dict, Any, Optional, Type, TypeVar, Generic, Union, Set, List
from dataclasses import dataclass, field
from contextlib import contextmanager
from pydantic_ai import Agent
from pydantic import BaseModel

from .registry import AgenToolRegistry


T = TypeVar('T')


# Templates for displaying available shortcuts
_TEMPLATE_SHORT_DEFAULT_START = """
================================================================================
                      AgenTool Injector Shortcuts
================================================================================

Available shortcuts for common operations:
"""

_TEMPLATE_SHORT_LOGGING = """
LOGGING
   await injector.log("User logged in", "INFO", {"user_id": 123})
   await injector.log("Error occurred", "ERROR", {"code": 500})
"""

_TEMPLATE_SHORT_STORAGE_KV = """
KEY-VALUE STORAGE
   await injector.kset("user:123", {"name": "Alice", "role": "admin"})
   user = await injector.kget("user:123", default={})
   success = await injector.kset("session:abc", data, ttl=3600)
"""

_TEMPLATE_SHORT_STORAGE_FS = """
FILE STORAGE
   content = await injector.fsread("/path/to/file")
   success = await injector.fswrite("/path/to/file", "content")
"""

_TEMPLATE_SHORT_CONFIG = """
CONFIGURATION
   value = await injector.config_get("database.host")
   success = await injector.config_set("database.host", "localhost")
"""

_TEMPLATE_SHORT_METRICS = """
METRICS
   await injector.metric_inc("api.requests")  # Increment by 1
   await injector.metric_dec("connections")  # Decrement by 1
"""

_TEMPLATE_SHORT_REFERENCES = """
REFERENCES
   # Store content and get reference
   ref = await injector.ref("storage_kv", "template_data", {"x": 1})
   
   # Resolve reference to content
   data = await injector.unref("!ref:storage_kv:template_data")
   file = await injector.unref("!ref:storage_fs:/path/to/file")
   conf = await injector.unref("!ref:config:database.host")
"""

_TEMPLATE_SHORT_DEFAULT_END = """
These shortcuts are equivalent to using injector.run() but with less boilerplate.
All original run() functionality remains available for complex operations.
"""

# Mapping of agent names to their shortcut templates
_SHORTCUTS_TEMPLATES = {
    'logging': _TEMPLATE_SHORT_LOGGING,
    'storage_kv': _TEMPLATE_SHORT_STORAGE_KV,
    'storage_fs': _TEMPLATE_SHORT_STORAGE_FS,
    'config': _TEMPLATE_SHORT_CONFIG,
    'metrics': _TEMPLATE_SHORT_METRICS,
}


def serialize_to_json_string(data: Any) -> str:
    """
    Serialize various Python types to JSON string.
    
    Handles:
    - Strings (validates if already valid JSON)
    - Dicts, lists, tuples
    - BaseModel instances
    - Basic types (int, float, bool, None)
    - Datetime objects
    - Decimal objects
    
    Args:
        data: The data to serialize
        
    Returns:
        A valid JSON string
        
    Raises:
        ValueError: If the data cannot be serialized to JSON
    """
    # If it's already a string, check if it's valid JSON
    if isinstance(data, str):
        try:
            # Validate it's proper JSON
            json.loads(data)
            return data
        except json.JSONDecodeError:
            # Not valid JSON, treat as a plain string value
            return json.dumps(data)
    
    # Handle Pydantic models
    if isinstance(data, BaseModel):
        return data.model_dump_json()
    
    # Handle datetime objects
    if isinstance(data, (datetime, date, datetime_time)):
        return json.dumps(data.isoformat())
    
    # Handle Decimal
    if isinstance(data, Decimal):
        return json.dumps(str(data))
    
    # Handle basic types and collections
    try:
        return json.dumps(data)
    except (TypeError, ValueError) as e:
        # Try with default string conversion for unknown types
        try:
            return json.dumps(str(data))
        except Exception:
            raise ValueError(f"Cannot serialize {type(data).__name__} to JSON: {e}")


def validate_json_string(data: str) -> bool:
    """
    Validate if a string is valid JSON.
    
    Args:
        data: The string to validate
        
    Returns:
        True if valid JSON, False otherwise
    """
    try:
        json.loads(data)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


class AgenToolInjector:
    """
    Manages dependency injection for AgenTools.
    
    This class acts as a container for agent instances and their dependencies,
    providing a clean way to wire up complex agent hierarchies.
    """
    
    def __init__(self):
        """Initialize the injector."""
        self._instances: Dict[str, Agent] = {}
        self._dependency_overrides: Dict[str, Agent] = {}
        # Example collection
        self._example_collection_enabled: bool = True
        self._completed_agents: Set[str] = set()  # Fast skip filter
        self._examples_dir: str = "logs/examples"
        # Caching for completeness checks
        self._completeness_cache: Dict[str, bool] = {}  # agent_name -> is_complete
        # Track known metrics to avoid KeyError when accessing non-existent metrics
        self._known_metrics: Set[str] = set()
    
    def register(self, name: str, agent: Agent) -> None:
        """
        Register an agent with the injector.
        
        Args:
            name: The name of the agent
            agent: The Agent instance
        """
        self._instances[name] = agent
    
    def get(self, name: str) -> Agent:
        """
        Get an AgenTool instance by name.
        
        This method handles:
        1. Instance caching (singleton pattern)
        2. Dependency resolution
        3. Automatic agent creation
        
        Args:
            name: The name of the AgenTool
            
        Returns:
            The Agent instance
            
        Raises:
            ValueError: If the AgenTool is not registered
        """
        # Check for override first
        if name in self._dependency_overrides:
            return self._dependency_overrides[name]
        
        # Return cached instance if available
        if name in self._instances:
            return self._instances[name]
        
        # Get configuration from registry
        config = AgenToolRegistry.get(name)
        if not config:
            raise ValueError(f"AgenTool '{name}' not found in registry")
        
        # Agent not found - this shouldn't happen if agents are properly registered
        raise ValueError(
            f"AgenTool '{name}' found in registry but no instance available. "
            f"Make sure the agent was created using create_agentool()."
        )
    
    def create_deps(self, agent_name: str) -> Optional[Any]:
        """
        Create a dependencies object for an agent based on its configuration.
        
        This automatically wires up all required dependencies.
        
        Args:
            agent_name: The name of the agent
            
        Returns:
            A deps object with all dependencies wired, or None if no deps needed
        """
        config = AgenToolRegistry.get(agent_name)
        if not config or not config.dependencies:
            return None
        
        # Build a deps object dynamically based on dependencies
        deps_dict = {}
        for dep_name in config.dependencies:
            deps_dict[dep_name] = self.get(dep_name)
        
        # Create a simple namespace object with the dependencies
        return type('Deps', (), deps_dict)()
    
    async def run(self, agent_name: str, input_data: Union[str, Dict, BaseModel, Any], **kwargs) -> Any:
        """
        Run an agent with automatic dependency injection and JSON serialization.
        
        Args:
            agent_name: The name of the agent to run
            input_data: The input data (can be string, dict, BaseModel, or any JSON-serializable type)
            **kwargs: Additional arguments for the agent
            
        Returns:
            The AgentRunResult from the agent
            
        Examples:
            >>> # With a dict
            >>> result = await injector.run('kv_storage', {"operation": "get", "key": "foo"})
            >>> 
            >>> # With a Pydantic model
            >>> input_model = KVInput(operation="set", key="foo", value="bar")
            >>> result = await injector.run('kv_storage', input_model)
            >>> 
            >>> # With a JSON string (unchanged behavior)
            >>> result = await injector.run('kv_storage', '{"operation": "get", "key": "foo"}')
            >>> 
            >>> # With basic types
            >>> result = await injector.run('echo', "hello world")
            >>> result = await injector.run('counter', 42)
        """
        # Serialize input data to JSON string
        json_input = serialize_to_json_string(input_data)
        
        # Get metrics configuration from registry
        metrics_config = AgenToolRegistry.get_metrics_config()
        
        # Fast skip if collection disabled or agent is complete
        should_collect = (
            self._example_collection_enabled and 
            agent_name not in self._completed_agents
        )
        
        # Track metrics if enabled and agent is not in the disabled list
        start_time = None
        operation = None
        if (metrics_config.enabled and agent_name not in metrics_config.disabled_agents) or should_collect:
            start_time = time.time()
            # Try to extract operation from input
            try:
                if isinstance(input_data, dict) and 'operation' in input_data:
                    operation = input_data['operation']
                elif isinstance(input_data, BaseModel) and hasattr(input_data, 'operation'):
                    operation = input_data.operation
                elif isinstance(input_data, str):
                    try:
                        parsed = json.loads(input_data)
                        if isinstance(parsed, dict) and 'operation' in parsed:
                            operation = parsed['operation']
                    except:
                        pass
            except:
                pass
        
        agent = self.get(agent_name)
        deps = self.create_deps(agent_name)
        
        try:
            if deps:
                result = await agent.run(json_input, deps=deps, **kwargs)
            else:
                result = await agent.run(json_input, **kwargs)
            
            # Collect success example
            if should_collect and operation:
                await self._add_example_to_registry(
                    agent_name=agent_name,
                    operation=operation,
                    input_data=input_data,
                    output=result.output if hasattr(result, 'output') else str(result),
                    is_error=False
                )
            
            # Record success metrics (skip for disabled agents)
            if metrics_config.enabled and agent_name not in metrics_config.disabled_agents:
                # Record metrics directly without creating tasks
                try:
                    await self._record_metrics(agent_name, operation, True, None, 
                                              time.time() - start_time if start_time else None,
                                              metrics_config.metrics_agent_name)
                except:
                    pass  # Ignore any metrics recording errors
            
            # =================================================================
            # TYPED OUTPUT HANDLING 
            # =================================================================
            # 
            # All AgenTools use typed outputs by default (use_typed_output=True).
            # This provides full type safety and excellent developer experience.
            #
            # How it works:
            # 1. Tool functions return typed Pydantic models (e.g., StorageKvOutput)
            # 2. The model gets serialized to JSON in AgentRunResult.output internally
            # 3. The injector automatically deserializes it back to the typed model
            # 4. Callers receive the typed object with full IDE support
            #
            # Benefits:
            # - Direct field access: result.success, result.data, etc.
            # - Full IDE autocomplete and type checking
            # - Automatic validation against output schemas
            # =================================================================
            
            # Get the AgenTool's configuration from the registry
            config = AgenToolRegistry.get(agent_name)
            
            # Process typed output (default behavior for all AgenTools)
            if config and config.use_typed_output and config.output_type and hasattr(result, 'output'):
                # Deserialize JSON back to the original typed Pydantic model
                try:
                    # Parse JSON and reconstruct the typed Pydantic model
                    output_data = json.loads(result.output)
                    typed_output = config.output_type(**output_data)
                    
                    # Return the typed object for direct field access
                    # Example: result.data['value'] with full type safety
                    return typed_output
                    
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    # If parsing fails, return the raw result
                    # This only happens if output doesn't match the expected schema
                    pass
            
            # Return raw result only in these edge cases:
            # - Tool doesn't define an output_type (should be rare)
            # - Result doesn't have .output attribute
            # - JSON parsing or validation failed
            # - use_typed_output is explicitly set to False (discouraged)
            return result
            
        except Exception as e:
            # Collect error example
            if should_collect and operation:
                await self._add_example_to_registry(
                    agent_name=agent_name,
                    operation=operation,
                    input_data=input_data,
                    error={"type": type(e).__name__, "message": str(e)},
                    is_error=True
                )
            
            # Record failure metrics (skip for disabled agents)
            if metrics_config.enabled and agent_name not in metrics_config.disabled_agents:
                try:
                    await self._record_metrics(agent_name, operation, False, str(type(e).__name__),
                                              time.time() - start_time if start_time else None,
                                              metrics_config.metrics_agent_name)
                except:
                    pass  # Ignore any metrics recording errors
            raise
    
    @contextmanager
    def override(self, **overrides: Agent):
        """
        Context manager to temporarily override dependencies.
        
        This is useful for testing or providing mock implementations.
        
        Args:
            **overrides: Mapping of agent names to override agents
            
        Example:
            >>> injector = AgenToolInjector()
            >>> mock_kv = create_mock_kv_agent()
            >>> with injector.override(kv_storage=mock_kv):
            ...     result = await injector.run('session', input_data)
        """
        # Store current overrides
        old_overrides = self._dependency_overrides.copy()
        
        # Apply new overrides
        self._dependency_overrides.update(overrides)
        
        try:
            yield
        finally:
            # Restore original overrides
            self._dependency_overrides = old_overrides
    
    def clear(self):
        """Clear all cached instances."""
        self._instances.clear()
        self._dependency_overrides.clear()
        # Clear completeness cache as well
        self._completeness_cache.clear()
        self._completed_agents.clear()
        # Clear known metrics
        self._known_metrics.clear()
    
    def initialize_examples(self, examples_dir: str = "logs/examples") -> None:
        """Load existing examples from JSON files into registry on startup."""
        self._examples_dir = examples_dir
        
        if not os.path.exists(examples_dir):
            return
        
        # Load each agent's examples
        for filename in os.listdir(examples_dir):
            if filename.endswith('.json'):
                agent_name = filename[:-5]  # Remove .json
                filepath = os.path.join(examples_dir, filename)
                
                try:
                    with open(filepath, 'r') as f:
                        examples = json.load(f)
                    
                    # Update registry with loaded examples
                    if AgenToolRegistry.exists(agent_name):
                        AgenToolRegistry.update(agent_name, examples=examples)
                        
                        # Check if this agent is complete
                        if self._is_agent_complete(agent_name):
                            self._completed_agents.add(agent_name)
                            print(f"✓ {agent_name}: examples complete")
                        else:
                            print(f"◐ {agent_name}: examples partial")
                except Exception as e:
                    print(f"Failed to load examples for {agent_name}: {e}")
    
    def _is_agent_complete(self, agent_name: str) -> bool:
        """Check if agent has both success and error examples for all operations.
        
        Uses caching to avoid repeated computations.
        """
        # Check cache first
        if agent_name in self._completeness_cache:
            return self._completeness_cache[agent_name]
        
        # Perform the actual check
        config = AgenToolRegistry.get(agent_name)
        if not config:
            return False
        
        operations = set(config.routing_config.operation_map.keys())
        if not operations:
            # No operations = complete
            self._completeness_cache[agent_name] = True
            return True
        
        # Track what we have
        has_success = set()
        has_error = set()
        
        for example in config.examples:
            operation = example.get("input", {}).get("operation")
            if not operation:
                continue
            
            if "error" in example:
                has_error.add(operation)
            else:
                has_success.add(operation)
        
        # Complete if every operation has both
        is_complete = has_success == operations and has_error == operations
        
        # Store in cache
        self._completeness_cache[agent_name] = is_complete
        
        return is_complete
    
    def _already_has_example(self, examples: List[Dict], operation: str, example_type: str) -> bool:
        """Check if we already have this type of example for this operation."""
        for example in examples:
            ex_operation = example.get("input", {}).get("operation")
            if ex_operation != operation:
                continue
            
            if example_type == "error" and "error" in example:
                return True
            if example_type == "success" and "error" not in example:
                return True
        
        return False
    
    async def _add_example_to_registry(
        self,
        agent_name: str,
        operation: str,
        input_data: Any,
        output: Any = None,
        error: Dict = None,
        is_error: bool = False
    ) -> None:
        """Add example to registry if needed and save immediately."""
        
        config = AgenToolRegistry.get(agent_name)
        if not config:
            return
        
        # Check if we need this example
        example_type = "error" if is_error else "success"
        if self._already_has_example(config.examples, operation, example_type):
            return
        
        # Prepare input
        input_dict = input_data if isinstance(input_data, dict) else json.loads(serialize_to_json_string(input_data))
        
        # Create example
        new_example = {"input": input_dict}
        if is_error:
            new_example["error"] = error
        else:
            new_example["output"] = output
        
        # Update registry (live update)
        updated_examples = list(config.examples)  # Copy existing
        updated_examples.append(new_example)
        AgenToolRegistry.update(agent_name, examples=updated_examples)
        
        # Invalidate cache for this agent since we added a new example
        if agent_name in self._completeness_cache:
            del self._completeness_cache[agent_name]
        
        # Save to JSON immediately
        await self._save_agent_examples(agent_name)
        
        # Check if agent is now complete (will trigger cache rebuild)
        if self._is_agent_complete(agent_name):
            self._completed_agents.add(agent_name)
            print(f"✓ {agent_name} examples complete!")
    
    async def _save_agent_examples(self, agent_name: str) -> None:
        """Save a single agent's examples to JSON."""
        config = AgenToolRegistry.get(agent_name)
        if not config:
            return
        
        # Create directory
        os.makedirs(self._examples_dir, exist_ok=True)
        
        # Save to agent-specific file
        filepath = os.path.join(self._examples_dir, f"{agent_name}.json")
        with open(filepath, 'w') as f:
            json.dump(config.examples, f, indent=2)
    
    
    async def _ensure_metric_exists(self, metric_name: str, metrics_agent_name: str) -> None:
        """
        Ensure a metric exists before trying to manipulate it.
        
        This method checks if a metric is in the known metrics set and creates it
        if it doesn't exist, preventing KeyError when accessing non-existent metrics.
        
        Args:
            metric_name: The name of the metric to ensure exists
            metrics_agent_name: Name of the metrics agent to use
        """
        if metric_name in self._known_metrics:
            return  # Metric already known to exist
        
        # For agentool.* metrics, we rely on the metrics system's auto-creation
        # The metrics system will auto-create these metrics when first accessed
        # We just add them to known metrics to avoid repeated checks
        if metric_name.startswith("agentool."):
            self._known_metrics.add(metric_name)
        # For non-agentool metrics, they must be created explicitly
        # We don't add them to known metrics since they might not exist
    
    async def _record_metrics(self, agent_name: str, operation: Optional[str],
                             success: bool, error_type: Optional[str],
                             duration: Optional[float], metrics_agent_name: str) -> None:
        """
        Record metrics for an agent execution.
        
        Args:
            agent_name: Name of the agent that was executed
            operation: Operation that was performed (if available)
            success: Whether the execution succeeded
            error_type: Type of error if failed
            duration: Duration of execution in seconds
            metrics_agent_name: Name of the metrics agent to use
        """
        try:
            # Only record if metrics agent is registered
            if metrics_agent_name not in self._instances:
                return
            
            # Record total executions
            metric_name = f"agentool.{agent_name}.executions.total"
            await self._ensure_metric_exists(metric_name, metrics_agent_name)
            await self.run(metrics_agent_name, {
                "operation": "increment",
                "name": metric_name,
                "value": 1
            })
            
            # Record success/failure
            if success:
                metric_name = f"agentool.{agent_name}.executions.success"
                await self._ensure_metric_exists(metric_name, metrics_agent_name)
                await self.run(metrics_agent_name, {
                    "operation": "increment",
                    "name": metric_name,
                    "value": 1
                })
            else:
                metric_name = f"agentool.{agent_name}.executions.failure"
                await self._ensure_metric_exists(metric_name, metrics_agent_name)
                await self.run(metrics_agent_name, {
                    "operation": "increment",
                    "name": metric_name,
                    "value": 1
                })
                
                # Record error type
                if error_type:
                    metric_name = f"agentool.{agent_name}.errors.{error_type}"
                    await self._ensure_metric_exists(metric_name, metrics_agent_name)
                    await self.run(metrics_agent_name, {
                        "operation": "increment",
                        "name": metric_name,
                        "value": 1
                    })
            
            # Record duration
            if duration is not None:
                metric_name = f"agentool.{agent_name}.duration.seconds"
                await self._ensure_metric_exists(metric_name, metrics_agent_name)
                await self.run(metrics_agent_name, {
                    "operation": "observe",
                    "name": metric_name,
                    "value": duration
                })
            
            # Record operation-specific metrics
            if operation:
                metric_name = f"agentool.{agent_name}.{operation}.count"
                await self._ensure_metric_exists(metric_name, metrics_agent_name)
                await self.run(metrics_agent_name, {
                    "operation": "increment",
                    "name": metric_name,
                    "value": 1
                })
                
                if duration is not None:
                    metric_name = f"agentool.{agent_name}.{operation}.duration"
                    await self._ensure_metric_exists(metric_name, metrics_agent_name)
                    await self.run(metrics_agent_name, {
                        "operation": "observe",
                        "name": metric_name,
                        "value": duration
                    })
                
                if not success and error_type:
                    metric_name = f"agentool.{agent_name}.{operation}.errors"
                    await self._ensure_metric_exists(metric_name, metrics_agent_name)
                    await self.run(metrics_agent_name, {
                        "operation": "increment",
                        "name": metric_name,
                        "value": 1,
                        "labels": {"error_type": error_type}
                    })
        except:
            # Silently ignore metrics recording errors to not affect main execution
            pass
    
    def enable_metrics(self, enabled: bool = True) -> None:
        """
        Enable or disable automatic metrics tracking.
        (Delegates to registry's metrics configuration)
        
        Args:
            enabled: Whether to enable metrics tracking
        """
        AgenToolRegistry.enable_metrics(enabled)
    
    def set_metrics_agent(self, agent_name: str) -> None:
        """
        Set the name of the metrics agent to use.
        (Delegates to registry's metrics configuration)
        
        Args:
            agent_name: Name of the metrics agent
        """
        AgenToolRegistry.update_metrics_config(metrics_agent_name=agent_name)
    
    def is_metrics_enabled(self) -> bool:
        """Check if metrics tracking is enabled."""
        return AgenToolRegistry.is_metrics_enabled()
    
    def enable_example_collection(self, enabled: bool = True, auto_load: bool = True) -> None:
        """Enable example collection with optional auto-loading."""
        self._example_collection_enabled = enabled
        
        if enabled and auto_load:
            self.initialize_examples()
        
        if enabled:
            incomplete = []
            for agent_name in AgenToolRegistry.list_names():
                if agent_name not in self._completed_agents:
                    incomplete.append(agent_name)
            
            if incomplete:
                print(f"Example collection enabled. Incomplete agents: {', '.join(incomplete)}")
            else:
                print("Example collection enabled. All agents have complete examples!")
    
    def get_example_status(self) -> Dict[str, Dict]:
        """Get detailed status of example collection."""
        status = {}
        
        for agent_name in AgenToolRegistry.list_names():
            config = AgenToolRegistry.get(agent_name)
            if not config:
                continue
            
            operations = list(config.routing_config.operation_map.keys())
            
            # Analyze current examples
            success_ops = set()
            error_ops = set()
            
            for example in config.examples:
                op = example.get("input", {}).get("operation")
                if op:
                    if "error" in example:
                        error_ops.add(op)
                    else:
                        success_ops.add(op)
            
            status[agent_name] = {
                "complete": agent_name in self._completed_agents,
                "operations": {
                    op: {
                        "success": op in success_ops,
                        "error": op in error_ops
                    }
                    for op in operations
                },
                "progress": f"{len(success_ops) + len(error_ops)}/{len(operations) * 2}"
            }
        
        return status
    
    def export_all_examples(self, output_file: str = "logs/all_examples.json") -> None:
        """Export all examples from registry to a single file."""
        all_examples = {}
        
        for agent_name in AgenToolRegistry.list_names():
            config = AgenToolRegistry.get(agent_name)
            if config and config.examples:
                all_examples[agent_name] = config.examples
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(all_examples, f, indent=2)
        
        print(f"Exported examples for {len(all_examples)} agents to {output_file}")
    
    def invalidate_completeness_cache(self, agent_name: str = None) -> None:
        """Invalidate completeness cache for a specific agent or all agents.
        
        Args:
            agent_name: Specific agent to invalidate, or None for all
        """
        if agent_name:
            # Invalidate specific agent
            if agent_name in self._completeness_cache:
                del self._completeness_cache[agent_name]
            # Also remove from completed set if present
            self._completed_agents.discard(agent_name)
        else:
            # Clear all caches
            self._completeness_cache.clear()
            self._completed_agents.clear()
    
    # ======================================
    # SHORTCUT METHODS FOR COMMON OPERATIONS
    # ======================================
    
    async def log(self, 
                  message: str, 
                  level: str = "INFO", 
                  data: Optional[Dict] = None, 
                  logger_name: str = "default") -> Any:
        """
        Shortcut for logging operations.
        
        Args:
            message: The log message
            level: Log level (DEBUG, INFO, WARN, ERROR, CRITICAL)
            data: Optional structured data to include in the log
            logger_name: Name of the logger to use
            
        Returns:
            LoggingOutput or AgentRunResult depending on typed output configuration
            
        Example:
            await injector.log("User authenticated", "INFO", {"user_id": 123})
        """
        return await self.run('logging', {
            "operation": "log",
            "level": level,
            "message": message,
            "data": data or {},
            "logger_name": logger_name
        })
    
    async def kget(self, 
                   key: str, 
                   default: Any = None, 
                   namespace: str = "default") -> Any:
        """
        Get a value from key-value storage with a default fallback.
        
        Args:
            key: The key to retrieve
            default: Default value if key doesn't exist
            namespace: Storage namespace
            
        Returns:
            The stored value or default if not found
            
        Example:
            user_data = await injector.kget("user:123", default={})
        """
        try:
            result = await self.run('storage_kv', {
                "operation": "get",
                "key": key,
                "namespace": namespace
            })
            
            # All AgenTools now use typed output
            if result.success and result.data and result.data.get("exists"):
                return result.data.get("value", default)
                    
            return default
        except:
            return default
    
    async def kset(self, 
                   key: str, 
                   value: Any, 
                   ttl: Optional[int] = None, 
                   namespace: str = "default") -> bool:
        """
        Set a value in key-value storage with optional TTL.
        
        Args:
            key: The key to set
            value: The value to store (must be JSON serializable)
            ttl: Optional time-to-live in seconds
            namespace: Storage namespace
            
        Returns:
            True if the value was successfully stored, False otherwise
            
        Example:
            success = await injector.kset("session:abc", session_data, ttl=3600)
        """
        try:
            result = await self.run('storage_kv', {
                "operation": "set",
                "key": key,
                "value": value,
                "ttl": ttl,
                "namespace": namespace
            })
            
            # All AgenTools now use typed output
            return result.success
        except:
            return False
    
    async def metric(self, 
                     name: str, 
                     value: float = 1.0, 
                     op: str = "increment",
                     labels: Optional[Dict[str, str]] = None) -> Any:
        """
        Internal metric tracking method (undocumented).
        Used by chain() and other internal operations.
        """
        return await self.run('metrics', {
            "operation": op,
            "name": name,
            "value": value,
            "labels": labels or {}
        })
    
    async def metric_inc(self, 
                         name: str, 
                         value: float = 1.0,
                         labels: Optional[Dict[str, str]] = None) -> Any:
        """
        Increment a metric counter.
        
        Args:
            name: Metric name (e.g., "api.requests.total")
            value: Amount to increment (default 1.0)
            labels: Optional metric labels for dimensionality
            
        Returns:
            MetricsOutput with the operation result
            
        Example:
            await injector.metric_inc("api.requests")
            await injector.metric_inc("processed.items", 5)
        """
        return await self.run('metrics', {
            "operation": "increment",
            "name": name,
            "value": value,
            "labels": labels or {}
        })
    
    async def metric_dec(self, 
                         name: str, 
                         value: float = 1.0,
                         labels: Optional[Dict[str, str]] = None) -> Any:
        """
        Decrement a metric counter.
        
        Args:
            name: Metric name (e.g., "active.connections")
            value: Amount to decrement (default 1.0)
            labels: Optional metric labels for dimensionality
            
        Returns:
            MetricsOutput with the operation result
            
        Example:
            await injector.metric_dec("active.connections")
            await injector.metric_dec("queue.size", 3)
        """
        return await self.run('metrics', {
            "operation": "decrement",
            "name": name,
            "value": value,
            "labels": labels or {}
        })
    
    async def fsread(self, path: str) -> Optional[str]:
        """
        Read a file from filesystem storage.
        
        Args:
            path: Path to the file to read
            
        Returns:
            File content as string or None if file doesn't exist
            
        Example:
            content = await injector.fsread("/path/to/config.json")
        """
        try:
            result = await self.run('storage_fs', {
                "operation": "read",
                "path": path
            })
            return result.data.get("content") if result.success else None
        except:
            return None
    
    async def fswrite(self, 
                      path: str, 
                      content: str,
                      create_parents: bool = True) -> bool:
        """
        Write content to a file in filesystem storage.
        
        Args:
            path: Path where the file will be written
            content: Content to write to the file
            create_parents: Whether to create parent directories if they don't exist
            
        Returns:
            True if file was successfully written, False otherwise
            
        Example:
            success = await injector.fswrite("/path/to/output.txt", "Hello World")
        """
        try:
            result = await self.run('storage_fs', {
                "operation": "write",
                "path": path,
                "content": content,
                "create_parents": create_parents
            })
            return result.success
        except:
            return False
    
    async def config_get(self, 
                         key: str, 
                         namespace: str = "app",
                         default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key (supports dot notation like 'database.host')
            namespace: Configuration namespace
            default: Default value if key doesn't exist
            
        Returns:
            The configuration value or default if not found
            
        Example:
            db_host = await injector.config_get("database.host", default="localhost")
        """
        try:
            result = await self.run('config', {
                "operation": "get",
                "key": key,
                "namespace": namespace,
                "default": default
            })
            return result.data.get("value", default) if result.success else default
        except:
            return default
    
    async def config_set(self, 
                         key: str, 
                         value: Any,
                         namespace: str = "app") -> bool:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key (supports dot notation like 'database.host')
            value: The value to set
            namespace: Configuration namespace
            
        Returns:
            True if the configuration was successfully set, False otherwise
            
        Example:
            success = await injector.config_set("database.host", "192.168.1.100")
        """
        try:
            result = await self.run('config', {
                "operation": "set",
                "key": key,
                "value": value,
                "namespace": namespace
            })
            return result.success
        except:
            return False
    
    async def unref(self, reference: str) -> Any:
        """
        Resolve a reference string to its actual content.
        
        Supports reference formats:
            - !ref:storage_kv:key_name
            - !ref:storage_fs:/path/to/file
            - !ref:config:config.key.path
            
        Args:
            reference: The reference string to resolve
            
        Returns:
            The resolved content or an error string if resolution fails
            
        Example:
            content = await injector.unref("!ref:storage_kv:user_profile")
        """
        # Return as-is if not a reference
        if not reference.startswith("!ref:"): 
            return reference
        
        try:
            # Parse reference format
            parts = reference[5:].split(":", 1)
            if len(parts) != 2: 
                return f"<invalid_ref:{reference}>"
            
            ref_type, ref_key = parts
            
            # Resolve based on storage type
            if ref_type == "storage_kv":
                value = await self.kget(ref_key)
                return value if value is not None else f"<undefined:{reference}>"
                
            elif ref_type == "storage_fs":
                try:
                    result = await self.run('storage_fs', {
                        "operation": "read",
                        "path": ref_key
                    })
                    
                    # All AgenTools now use typed output
                    return result.data.get("content") if result.success else f"<undefined:{reference}>"
                except:
                    return f"<undefined:{reference}>"
                    
            elif ref_type == "config":
                try:
                    result = await self.run('config', {
                        "operation": "get",
                        "key": ref_key,
                        "namespace": "app"
                    })
                    
                    # All AgenTools now use typed output
                    return result.data.get("value") if result.success else f"<undefined:{reference}>"
                except:
                    return f"<undefined:{reference}>"
                    
            else:
                return f"<unknown_type:{reference}>"
                
        except Exception as e:
            return f"<error:{reference}:{str(e)}>"
    
    async def ref(self, 
                  ref_type: str, 
                  key: str, 
                  content: Any) -> str:
        """
        Store content and return a reference string for later retrieval.
        
        Args:
            ref_type: Storage type (storage_kv, storage_fs, config)
            key: The key or path where content will be stored
            content: The content to store
            
        Returns:
            A reference string in the format !ref:type:key
            
        Example:
            ref = await injector.ref("storage_kv", "template_data", {"title": "Hello"})
            # Returns: "!ref:storage_kv:template_data"
        """
        if ref_type == "storage_kv":
            await self.kset(key, content)
            
        elif ref_type == "storage_fs":
            await self.run('storage_fs', {
                "operation": "write",
                "path": key,
                "content": json.dumps(content) if not isinstance(content, str) else content
            })
            
        elif ref_type == "config":
            await self.run('config', {
                "operation": "set",
                "key": key,
                "value": content,
                "namespace": "app"
            })
            
        return f"!ref:{ref_type}:{key}"
    
    def chain(self) -> 'FluentOps':
        """
        Internal method for chaining operations (undocumented).
        """
        return FluentOps(self)
    
    def show_shortcuts(self, dependencies: Optional[List[str]] = None) -> None:
        """
        Print a formatted guide of available shortcut methods.
        
        Args:
            dependencies: Optional list of agent names to show shortcuts for.
                         If None, shows all available shortcuts.
                         
        Example:
            injector.show_shortcuts()  # Show all shortcuts
            injector.show_shortcuts(['storage_kv', 'logging'])  # Show only these
        """
        output = _TEMPLATE_SHORT_DEFAULT_START
        
        # Determine which shortcuts to show
        if dependencies is None:
            # Show all except hidden ones (chain and base metric)
            agents_to_show = ['logging', 'storage_kv', 'storage_fs', 'config', 'metrics']
        else:
            # Show only requested dependencies that have shortcuts
            agents_to_show = [dep for dep in dependencies if dep in _SHORTCUTS_TEMPLATES]
        
        # Add templates for each agent
        for agent_name in agents_to_show:
            if agent_name in _SHORTCUTS_TEMPLATES:
                output += _SHORTCUTS_TEMPLATES[agent_name]
        
        # Always show references if storage_kv, storage_fs, or config is shown
        if any(agent in agents_to_show for agent in ['storage_kv', 'storage_fs', 'config']):
            output += _TEMPLATE_SHORT_REFERENCES
        
        output += _TEMPLATE_SHORT_DEFAULT_END
        print(output)


class FluentOps:
    """Fluent operations chain for batching multiple operations."""
    
    def __init__(self, injector: 'AgenToolInjector'):
        self._injector = injector
        self._ops: List[tuple] = []
    
    def log(self, message: str, level: str = "INFO", data: Optional[Dict] = None, 
            logger: str = "default") -> 'FluentOps':
        """Queue a log operation."""
        self._ops.append(('log', [], {'message': message, 'level': level, 'data': data, 'logger': logger}))
        return self
    
    def kget(self, key: str, default: Any = None, namespace: str = "default") -> 'FluentOps':
        """Queue a KV get operation."""
        self._ops.append(('kget', [], {'key': key, 'default': default, 'namespace': namespace}))
        return self
    
    def kset(self, key: str, value: Any, ttl: Optional[int] = None, 
             namespace: str = "default") -> 'FluentOps':
        """Queue a KV set operation."""
        self._ops.append(('kset', [], {'key': key, 'value': value, 'ttl': ttl, 'namespace': namespace}))
        return self
    
    def metric(self, name: str, value: float = 1, op: str = "increment",
               labels: Optional[Dict] = None) -> 'FluentOps':
        """Queue a metric operation."""
        self._ops.append(('metric', [], {'name': name, 'value': value, 'op': op, 'labels': labels}))
        return self
    
    async def execute(self) -> List[Any]:
        """Execute all queued operations and return results."""
        results = []
        for method_name, args, kwargs in self._ops:
            method = getattr(self._injector, method_name)
            result = await method(**kwargs)
            results.append(result)
        return results


# Global injector instance
_global_injector = AgenToolInjector()


def get_injector() -> AgenToolInjector:
    """Get the global AgenTool injector instance."""
    return _global_injector


@dataclass
class InjectedDeps(Generic[T]):
    """
    Base class for dependency injection with AgenTools.
    
    This provides a cleaner interface for defining dependencies.
    """
    _injector: AgenToolInjector = field(default_factory=get_injector, init=False)
    
    def get_agent(self, name: str) -> Agent:
        """Get an agent from the injector."""
        return self._injector.get(name)
    
    async def call_agent(self, name: str, input_data: Union[str, Dict, BaseModel, Any]) -> Any:
        """
        Call an agent with its dependencies automatically injected.
        
        Args:
            name: The name of the agent
            input_data: The input data (can be string, dict, BaseModel, or any JSON-serializable type)
            
        Returns:
            The AgentRunResult from the agent
        """
        result = await self._injector.run(name, input_data)
        # Return the AgentRunResult so the caller can access .output
        return result
