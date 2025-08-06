"""
AgenTool Registry for managing AgenTool configurations.

This module provides a global registry for AgenTool configurations,
enabling runtime lookup and model provider integration.

The registry is the central storage for all AgenTool configurations,
allowing the AgenToolModel to look up schemas and routing information
at runtime. This enables dynamic AgenTool creation and management.

Example:
    >>> from agentool import AgenToolRegistry, AgenToolConfig
    >>> 
    >>> config = AgenToolConfig(input_schema=MySchema, routing_config=routing)
    >>> AgenToolRegistry.register('my_tool', config)
    >>> 
    >>> # Later, retrieve the config
    >>> config = AgenToolRegistry.get('my_tool')
"""

from __future__ import annotations

from typing import Dict, Type, Optional, Callable, Any, Tuple, List, Set
from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel


@dataclass
class RoutingConfig:
    """Configuration for routing operations to tools.
    
    Attributes:
        operation_field: The field name in the input schema that contains the operation
        operation_map: Maps operation values to (tool_name, transform_func) tuples
                      where transform_func extracts args from input for the tool
    """
    operation_field: str = 'operation'
    operation_map: Dict[str, Tuple[str, Callable[[Any], Dict[str, Any]]]] = None
    
    def __post_init__(self):
        if self.operation_map is None:
            self.operation_map = {}


@dataclass
class ToolMetadata:
    """Metadata about a tool function.
    
    Attributes:
        name: The tool function name
        description: Description from the tool's docstring
        is_async: Whether the tool is async
        parameters: List of parameter names
        parameter_types: Dict of parameter names to types
        return_type: The return type annotation as string
        return_type_annotation: The actual return type object
    """
    name: str
    description: Optional[str] = None
    is_async: bool = False
    parameters: List[str] = field(default_factory=list)
    parameter_types: Dict[str, str] = field(default_factory=dict)
    return_type: Optional[str] = None
    return_type_annotation: Optional[Type[Any]] = None


@dataclass
class MetricsConfig:
    """Configuration for metrics tracking.
    
    Attributes:
        enabled: Whether metrics tracking is enabled globally
        metrics_agent_name: Name of the metrics agent to use
        disabled_agents: Set of agent names that should not be tracked
        auto_create_metrics: Whether to auto-create metrics for agentool.* namespace
        default_histogram_buckets: Default buckets for histogram metrics
        max_observations: Maximum observations to keep per metric
        export_interval: Interval in seconds for periodic metrics export
        export_format: Default export format (json, prometheus, statsd)
    """
    enabled: bool = True
    metrics_agent_name: str = 'metrics'
    disabled_agents: Set[str] = field(default_factory=lambda: {'metrics', 'storage_kv'})
    auto_create_metrics: bool = True
    default_histogram_buckets: List[float] = field(
        default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    max_observations: int = 100
    export_interval: Optional[int] = None  # Seconds, None means no periodic export
    export_format: str = 'json'


@dataclass
class AgenToolConfig:
    """Configuration for an AgenTool.
    
    Attributes:
        input_schema: The Pydantic model for input validation
        routing_config: Configuration for routing to tools
        output_type: Optional output type for structured responses
        description: Optional description of the AgenTool
        version: Version of the AgenTool
        tags: List of tags for categorization
        tools_metadata: Metadata about the tools used
        dependencies: List of required dependencies
        examples: List of example inputs/outputs
        created_at: Timestamp when the AgenTool was created
        updated_at: Timestamp when the AgenTool was last updated
        use_typed_output: Whether to return typed Pydantic models instead of AgentRunResult
    """
    input_schema: Type[BaseModel]
    routing_config: RoutingConfig
    output_type: Optional[Type[BaseModel]] = None
    description: Optional[str] = None
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    tools_metadata: List[ToolMetadata] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    use_typed_output: bool = False  # Default False for backward compatibility


class AgenToolRegistry:
    """Global registry for AgenTool configurations."""
    
    _configs: Dict[str, AgenToolConfig] = {}
    _metrics_config: MetricsConfig = MetricsConfig()
    
    @classmethod
    def register(cls, name: str, config: AgenToolConfig) -> None:
        """Register an AgenTool configuration.
        
        Args:
            name: The name of the AgenTool (e.g., 'storage', 'compute')
            config: The AgenTool configuration
        """
        cls._configs[name] = config
    
    @classmethod
    def get(cls, name: str) -> Optional[AgenToolConfig]:
        """Get an AgenTool configuration by name.
        
        Args:
            name: The name of the AgenTool
            
        Returns:
            The configuration if found, None otherwise
        """
        return cls._configs.get(name)
    
    @classmethod
    def list_names(cls) -> list[str]:
        """List all registered AgenTool names."""
        return list(cls._configs.keys())
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered configurations (useful for testing)."""
        cls._configs.clear()
        cls._metrics_config = MetricsConfig()  # Reset metrics config too
    
    @classmethod
    def list_detailed(cls) -> List[Dict[str, Any]]:
        """Return detailed information about all registered AgenTools.
        
        Returns:
            List of dictionaries containing detailed AgenTool information
        """
        detailed_list = []
        for name, config in cls._configs.items():
            details = {
                "name": name,
                "version": config.version,
                "description": config.description,
                "tags": config.tags,
                "operations": list(config.routing_config.operation_map.keys()),
                "tools": [
                    {
                        "name": tool.name,
                        "async": tool.is_async,
                        "params": tool.parameters,
                        "description": tool.description
                    }
                    for tool in config.tools_metadata
                ],
                "input_schema": config.input_schema.model_json_schema(),
                "created_at": config.created_at.isoformat(),
                "updated_at": config.updated_at.isoformat()
            }
            if config.examples:
                details["examples"] = config.examples
            detailed_list.append(details)
        return detailed_list
    
    @classmethod
    def search(cls, tags: Optional[List[str]] = None, name_pattern: Optional[str] = None) -> List[str]:
        """Search AgenTools by tags or name pattern.
        
        Args:
            tags: List of tags to filter by (matches if any tag is present)
            name_pattern: Substring to search for in names
            
        Returns:
            List of matching AgenTool names
        """
        results = []
        for name, config in cls._configs.items():
            # Check name pattern
            if name_pattern and name_pattern.lower() not in name.lower():
                continue
            
            # Check tags
            if tags and not any(tag in config.tags for tag in tags):
                continue
            
            results.append(name)
        return results
    
    @classmethod
    def get_schema(cls, name: str) -> Optional[Dict[str, Any]]:
        """Get the full JSON schema for an AgenTool.
        
        Args:
            name: The name of the AgenTool
            
        Returns:
            The JSON schema if found, None otherwise
        """
        config = cls.get(name)
        if config:
            return config.input_schema.model_json_schema()
        return None
    
    @classmethod
    def get_tools_info(cls, name: str) -> Optional[List[Dict[str, Any]]]:
        """Get information about tools used by an AgenTool.
        
        Args:
            name: The name of the AgenTool
            
        Returns:
            List of tool information if found, None otherwise
        """
        config = cls.get(name)
        if config:
            return [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "is_async": tool.is_async,
                    "parameters": tool.parameters,
                    "parameter_types": tool.parameter_types,
                    "return_type": tool.return_type
                }
                for tool in config.tools_metadata
            ]
        return None
    
    @classmethod
    def get_operations(cls, name: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """Get available operations for an AgenTool.
        
        Args:
            name: The name of the AgenTool
            
        Returns:
            Dictionary of operations with their details if found, None otherwise
        """
        config = cls.get(name)
        if config:
            operations = {}
            for op_name, (tool_name, transform_func) in config.routing_config.operation_map.items():
                # Find the tool metadata
                tool_meta = next((t for t in config.tools_metadata if t.name == tool_name), None)
                operations[op_name] = {
                    "tool": tool_name,
                    "description": tool_meta.description if tool_meta else None,
                    "parameters": tool_meta.parameters if tool_meta else []
                }
            return operations
        return None
    
    @classmethod
    def export_catalog(cls) -> Dict[str, Any]:
        """Export the full AgenTool catalog as a dictionary.
        
        Returns:
            Dictionary containing the full catalog with metadata
        """
        return {
            "version": "1.0.0",
            "generated_at": datetime.now().isoformat(),
            "total_agentools": len(cls._configs),
            "agentools": cls.list_detailed()
        }
    
    @classmethod
    def generate_markdown_docs(cls) -> str:
        """Generate markdown documentation for all registered AgenTools.
        
        Returns:
            Markdown formatted documentation string
        """
        lines = ["# AgenTool Registry Documentation", ""]
        lines.append(f"Generated at: {datetime.now().isoformat()}")
        lines.append(f"Total AgenTools: {len(cls._configs)}")
        lines.append("")
        
        for name, config in sorted(cls._configs.items()):
            lines.append(f"## {name}")
            lines.append("")
            
            if config.description:
                lines.append(f"**Description:** {config.description}")
                lines.append("")
            
            lines.append(f"**Version:** {config.version}")
            lines.append("")
            
            if config.tags:
                lines.append(f"**Tags:** {', '.join(config.tags)}")
                lines.append("")
            
            # Operations
            lines.append("### Operations")
            lines.append("")
            for op_name, (tool_name, _) in config.routing_config.operation_map.items():
                tool_meta = next((t for t in config.tools_metadata if t.name == tool_name), None)
                if tool_meta and tool_meta.description:
                    lines.append(f"- **{op_name}**: {tool_meta.description}")
                else:
                    lines.append(f"- **{op_name}**: Calls `{tool_name}`")
            lines.append("")
            
            # Tools
            if config.tools_metadata:
                lines.append("### Tools")
                lines.append("")
                for tool in config.tools_metadata:
                    async_marker = "(async)" if tool.is_async else "(sync)"
                    lines.append(f"#### {tool.name} {async_marker}")
                    if tool.description:
                        lines.append(f"{tool.description}")
                    if tool.parameters:
                        lines.append(f"- Parameters: {', '.join(tool.parameters)}")
                    if tool.return_type:
                        lines.append(f"- Returns: `{tool.return_type}`")
                    lines.append("")
            
            # Schema
            lines.append("### Input Schema")
            lines.append("```json")
            import json
            schema = config.input_schema.model_json_schema()
            lines.append(json.dumps(schema, indent=2))
            lines.append("```")
            lines.append("")
            
            # Examples
            if config.examples:
                lines.append("### Examples")
                lines.append("")
                for i, example in enumerate(config.examples, 1):
                    lines.append(f"#### Example {i}")
                    if "input" in example:
                        lines.append("**Input:**")
                        lines.append("```json")
                        lines.append(json.dumps(example["input"], indent=2))
                        lines.append("```")
                    if "output" in example:
                        lines.append("**Output:**")
                        lines.append("```json")
                        lines.append(json.dumps(example["output"], indent=2))
                        lines.append("```")
                    lines.append("")
            
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)
    
    @classmethod
    def generate_dependency_graph(cls, include_tools: bool = True) -> Dict[str, Dict[str, List[str]]]:
        """Generate a dependency graph showing relationships between AgenTools and their tools.
        
        Args:
            include_tools: Whether to include tool dependencies in the graph
            
        Returns:
            Dictionary with 'agentools' showing inter-AgenTool dependencies and 
            optionally 'tools' showing which tools each AgenTool uses
        """
        graph = {"agentools": {}, "tools": {}} if include_tools else {"agentools": {}}
        
        for name, config in cls._configs.items():
            # Direct dependencies from config (like libraries, other AgenTools)
            agentool_deps = list(config.dependencies)
            
            # Tool dependencies - the actual tool functions used
            tool_deps = []
            
            # Check if any tools reference other AgenTools
            for op_name, (tool_name, _) in config.routing_config.operation_map.items():
                # If tool_name matches another AgenTool, it's an AgenTool dependency
                if tool_name in cls._configs and tool_name != name:
                    if tool_name not in agentool_deps:
                        agentool_deps.append(tool_name)
                else:
                    # Otherwise it's a tool dependency
                    if tool_name not in tool_deps:
                        tool_deps.append(tool_name)
            
            graph["agentools"][name] = agentool_deps
            
            if include_tools:
                # Also include tools from tools_metadata for completeness
                for tool_meta in config.tools_metadata:
                    if tool_meta.name not in tool_deps:
                        tool_deps.append(tool_meta.name)
                graph["tools"][name] = sorted(tool_deps)
        
        return graph
    
    @classmethod
    def generate_api_spec(cls) -> Dict[str, Any]:
        """Generate an OpenAPI-like specification for all AgenTools.
        
        Returns:
            Dictionary containing API specification
        """
        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "AgenTool Registry API",
                "version": "1.0.0",
                "description": "Specification for all registered AgenTools"
            },
            "paths": {}
        }
        
        for name, config in cls._configs.items():
            path = f"/agentools/{name}"
            spec["paths"][path] = {
                "post": {
                    "summary": config.description or f"Execute {name} AgenTool",
                    "operationId": f"execute_{name}",
                    "tags": config.tags,
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": config.input_schema.model_json_schema()
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Successful operation",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "description": "Tool execution result"
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Invalid input"
                        }
                    }
                }
            }
            
            # Add operation-specific endpoints
            for op_name in config.routing_config.operation_map.keys():
                op_path = f"/agentools/{name}/{op_name}"
                spec["paths"][op_path] = {
                    "post": {
                        "summary": f"Execute {op_name} operation on {name}",
                        "operationId": f"{name}_{op_name}",
                        "tags": config.tags + [op_name],
                        "responses": {
                            "200": {"description": "Success"},
                            "400": {"description": "Invalid input"}
                        }
                    }
                }
        
        return spec
    
    # Metrics Configuration Methods
    
    @classmethod
    def get_metrics_config(cls) -> MetricsConfig:
        """Get the current metrics configuration.
        
        Returns:
            The current MetricsConfig instance
        """
        return cls._metrics_config
    
    @classmethod
    def set_metrics_config(cls, config: MetricsConfig) -> None:
        """Set a new metrics configuration.
        
        Args:
            config: The new MetricsConfig to use
        """
        cls._metrics_config = config
    
    @classmethod
    def update_metrics_config(cls, **kwargs) -> None:
        """Update specific metrics configuration settings.
        
        Args:
            **kwargs: Settings to update in the metrics config
            
        Example:
            >>> AgenToolRegistry.update_metrics_config(
            ...     enabled=True,
            ...     export_format='prometheus',
            ...     max_observations=200
            ... )
        """
        config = cls._metrics_config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Invalid metrics config setting: {key}")
    
    @classmethod
    def enable_metrics(cls, enabled: bool = True) -> None:
        """Enable or disable metrics tracking globally.
        
        Args:
            enabled: Whether to enable metrics tracking
        """
        cls._metrics_config.enabled = enabled
    
    @classmethod
    def add_disabled_agent(cls, agent_name: str) -> None:
        """Add an agent to the list of agents that should not be tracked.
        
        Args:
            agent_name: Name of the agent to disable tracking for
        """
        cls._metrics_config.disabled_agents.add(agent_name)
    
    @classmethod
    def remove_disabled_agent(cls, agent_name: str) -> None:
        """Remove an agent from the disabled tracking list.
        
        Args:
            agent_name: Name of the agent to re-enable tracking for
        """
        cls._metrics_config.disabled_agents.discard(agent_name)
    
    @classmethod
    def is_metrics_enabled(cls) -> bool:
        """Check if metrics tracking is enabled.
        
        Returns:
            True if metrics are enabled, False otherwise
        """
        return cls._metrics_config.enabled
    
    @classmethod
    def get_metrics_export_format(cls) -> str:
        """Get the current default metrics export format.
        
        Returns:
            The export format (json, prometheus, or statsd)
        """
        return cls._metrics_config.export_format
    
    # Registry Management Methods
    
    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister an AgenTool configuration.
        
        Args:
            name: The name of the AgenTool to unregister
            
        Returns:
            True if unregistered successfully, False if not found
        """
        if name in cls._configs:
            del cls._configs[name]
            return True
        return False
    
    @classmethod
    def update(cls, name: str, **updates) -> bool:
        """Update an existing AgenTool configuration.
        
        Args:
            name: The name of the AgenTool to update
            **updates: Fields to update in the configuration
            
        Returns:
            True if updated successfully, False if not found
            
        Raises:
            ValueError: If an invalid field is specified
        """
        config = cls._configs.get(name)
        if not config:
            return False
        
        # Update timestamp
        config.updated_at = datetime.now()
        
        # Valid fields that can be updated
        updatable_fields = {
            'description', 'version', 'tags', 'dependencies', 'examples'
        }
        
        for field, value in updates.items():
            if field not in updatable_fields:
                raise ValueError(f"Field '{field}' cannot be updated. Valid fields: {updatable_fields}")
            setattr(config, field, value)
        
        return True
    
    @classmethod
    def exists(cls, name: str) -> bool:
        """Check if an AgenTool is registered.
        
        Args:
            name: The name of the AgenTool
            
        Returns:
            True if registered, False otherwise
        """
        return name in cls._configs
    
    @classmethod
    def validate_config(cls, config_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate an AgenTool configuration dictionary.
        
        Args:
            config_data: Configuration data to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Required fields
        required_fields = ['name', 'input_schema', 'routing_config']
        for field in required_fields:
            if field not in config_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate name
        if 'name' in config_data:
            name = config_data['name']
            if not isinstance(name, str) or not name.strip():
                errors.append("Name must be a non-empty string")
            elif len(name) > 100:
                errors.append("Name must be 100 characters or less")
        
        # Validate version format
        if 'version' in config_data:
            version = config_data['version']
            if not isinstance(version, str):
                errors.append("Version must be a string")
        
        # Validate tags
        if 'tags' in config_data:
            tags = config_data['tags']
            if not isinstance(tags, list):
                errors.append("Tags must be a list")
            elif not all(isinstance(tag, str) for tag in tags):
                errors.append("All tags must be strings")
        
        # Validate dependencies
        if 'dependencies' in config_data:
            deps = config_data['dependencies']
            if not isinstance(deps, list):
                errors.append("Dependencies must be a list")
            elif not all(isinstance(dep, str) for dep in deps):
                errors.append("All dependencies must be strings")
        
        # Validate examples structure
        if 'examples' in config_data:
            examples = config_data['examples']
            if not isinstance(examples, list):
                errors.append("Examples must be a list")
            else:
                for i, example in enumerate(examples):
                    if not isinstance(example, dict):
                        errors.append(f"Example {i} must be a dictionary")
                    elif 'input' not in example:
                        errors.append(f"Example {i} must have an 'input' field")
        
        return len(errors) == 0, errors