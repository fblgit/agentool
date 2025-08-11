"""
AgenTool Management Toolkit.

Provides comprehensive management, introspection, and creation capabilities
for the AgenTool registry system.
"""

import json
import sys
import os
from typing import Any, Dict, List, Optional, Literal, Union
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import RunContext

# Add the parent directories to path for imports
#sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agentool import create_agentool, RoutingConfig
from agentool.core.registry import AgenToolRegistry, AgenToolConfig, ToolMetadata
from agentool.factory import extract_tool_metadata


class AgenToolManagementInput(BaseModel):
    """Input schema for AgenTool management operations."""
    
    operation: Literal[
        # Registry introspection operations
        'list_agentools', 'get_agentool_info', 'get_agentool_schema',
        'search_agentools', 'get_operations', 'get_tools_info', 'get_routing_config',
        
        # Registry management operations  
        'create_agentool_config', 'update_agentool_config',
        'register_agentool', 'unregister_agentool', 'validate_agentool_config',
        
        # Analysis and documentation operations
        'generate_dependency_graph', 'analyze_agentool_usage', 'validate_dependencies',
        'get_api_specification', 'generate_docs', 'get_examples', 'export_catalog',
        'generate_usage_guide',
        
        # Dynamic creation operations
        'create_simple_agentool', 'create_tool_function',
        'validate_tool_signature', 'infer_routing_config'
    ] = Field(description="The management operation to perform")
    
    # Common parameters
    agentool_name: Optional[str] = Field(None, description="Name of the AgenTool to operate on")
    
    # Search parameters
    tags: Optional[List[str]] = Field(None, description="Tags to filter by")
    name_pattern: Optional[str] = Field(None, description="Name pattern to search for")
    
    # Creation parameters
    description: Optional[str] = Field(None, description="Description for new AgenTool")
    version: Optional[str] = Field("1.0.0", description="Version for new AgenTool")
    tools: Optional[List[str]] = Field(None, description="Tool function names or definitions")
    
    # Configuration parameters
    config_data: Optional[Dict[str, Any]] = Field(None, description="Configuration data")
    
    # Documentation parameters
    format: Optional[Literal['json', 'markdown', 'yaml']] = Field('json', description="Output format")
    include_examples: Optional[bool] = Field(True, description="Include examples in output")
    include_schemas: Optional[bool] = Field(True, description="Include schemas in output")
    
    # Additional parameters for various operations
    detailed: Optional[bool] = Field(False, description="Include detailed information")
    include_tools: Optional[bool] = Field(True, description="Include tool information")
    
    @field_validator('agentool_name', mode='before')
    def validate_agentool_name(cls, v, info):
        """Validate agentool_name is provided for operations that require it."""
        operation = info.data.get('operation') if info.data else None
        if operation in ['get_agentool_info', 'get_agentool_schema', 'get_operations', 
                         'get_tools_info', 'get_routing_config', 'get_examples', 
                         'generate_usage_guide'] and (v is None or v == '' or not v):
            raise ValueError(f"agentool_name is required for {operation} operation")
        return v


class ManagementOutput(BaseModel):
    """Structured output for management operations."""
    success: bool = Field(description="Whether the operation succeeded")
    operation: str = Field(description="The operation that was performed")
    message: str = Field(description="Human-readable result message")
    data: Optional[Dict[str, Any]] = Field(None, description="Operation-specific data")


# Registry Introspection Operations

async def list_agentools(ctx: RunContext[Any], detailed: bool = False) -> ManagementOutput:
    """List all registered AgenTools with basic or detailed information."""
    if detailed:
        agentools = AgenToolRegistry.list_detailed()
        return ManagementOutput(
            success=True,
            operation="list_agentools",
            message=f"Found {len(agentools)} AgenTools",
            data={
                "count": len(agentools),
                "agentools": agentools
            }
        )
    else:
        names = AgenToolRegistry.list_names()
        basic_info = []
        for name in names:
            config = AgenToolRegistry.get(name)
            if config:
                basic_info.append({
                    "name": name,
                    "version": config.version,
                    "description": config.description,
                    "operations_count": len(config.routing_config.operation_map),
                    "tools_count": len(config.tools_metadata),
                    "tags": config.tags
                })
        
        return ManagementOutput(
            success=True,
            operation="list_agentools",
            message=f"Found {len(names)} AgenTools",
            data={
                "count": len(names),
                "agentools": basic_info
            }
        )


async def get_agentool_info(ctx: RunContext[Any], agentool_name: str, detailed: bool = True) -> ManagementOutput:
    """Get detailed information about a specific AgenTool."""
    config = AgenToolRegistry.get(agentool_name)
    if not config:
        return ManagementOutput(
            success=False,
            operation="get_agentool_info",
            message=f"AgenTool '{agentool_name}' not found in registry",
            data=None
        )
    
    info = {
        "name": agentool_name,
        "version": config.version,
        "description": config.description,
        "tags": config.tags,
        "dependencies": config.dependencies,
        "created_at": config.created_at.isoformat(),
        "updated_at": config.updated_at.isoformat()
    }
    
    if detailed:
        info.update({
            "operations": list(config.routing_config.operation_map.keys()),
            "operation_field": config.routing_config.operation_field,
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "is_async": tool.is_async,
                    "parameters": tool.parameters,
                    "parameter_types": tool.parameter_types,
                    "return_type": tool.return_type
                }
                for tool in config.tools_metadata
            ],
            "examples": config.examples,
            "output_type": str(config.output_type) if config.output_type else None
        })
    
    return ManagementOutput(
        success=True,
        operation="get_agentool_info",
        message=f"Retrieved info for AgenTool '{agentool_name}'",
        data={"agentool": info}
    )


async def get_agentool_schema(ctx: RunContext[Any], agentool_name: str) -> ManagementOutput:
    """Get the JSON schema for an AgenTool's input."""
    schema = AgenToolRegistry.get_schema(agentool_name)
    if not schema:
        return ManagementOutput(
            success=False,
            operation="get_agentool_schema",
            message=f"AgenTool '{agentool_name}' not found in registry",
            data=None
        )
    
    return ManagementOutput(
        success=True,
        operation="get_agentool_schema",
        message=f"Retrieved schema for AgenTool '{agentool_name}'",
        data={
            "agentool_name": agentool_name,
            "schema": schema
        }
    )


async def search_agentools(ctx: RunContext[Any], tags: Optional[List[str]] = None, 
                          name_pattern: Optional[str] = None) -> ManagementOutput:
    """Search AgenTools by tags or name pattern."""
    results = AgenToolRegistry.search(tags=tags, name_pattern=name_pattern)
    
    # Get detailed info for results
    detailed_results = []
    for name in results:
        config = AgenToolRegistry.get(name)
        if config:
            detailed_results.append({
                "name": name,
                "version": config.version,
                "description": config.description,
                "tags": config.tags,
                "operations": list(config.routing_config.operation_map.keys())
            })
    
    return ManagementOutput(
        success=True,
        operation="search_agentools",
        message=f"Found {len(results)} matching AgenTools",
        data={
            "search_criteria": {
                "tags": tags,
                "name_pattern": name_pattern
            },
            "count": len(results),
            "results": detailed_results
        }
    )


async def get_operations(ctx: RunContext[Any], agentool_name: str) -> ManagementOutput:
    """Get available operations for an AgenTool."""
    operations = AgenToolRegistry.get_operations(agentool_name)
    if operations is None:
        return ManagementOutput(
            success=False,
            operation="get_operations",
            message=f"AgenTool '{agentool_name}' not found in registry",
            data=None
        )
    
    return ManagementOutput(
        success=True,
        operation="get_operations",
        message=f"Retrieved operations for AgenTool '{agentool_name}'",
        data={
            "agentool_name": agentool_name,
            "operations": operations
        }
    )


async def get_tools_info(ctx: RunContext[Any], agentool_name: str) -> ManagementOutput:
    """Get information about tools used by an AgenTool."""
    tools_info = AgenToolRegistry.get_tools_info(agentool_name)
    if tools_info is None:
        return ManagementOutput(
            success=False,
            operation="get_tools_info",
            message=f"AgenTool '{agentool_name}' not found in registry",
            data=None
        )
    
    return ManagementOutput(
        success=True,
        operation="get_tools_info",
        message=f"Retrieved tools info for AgenTool '{agentool_name}'",
        data={
            "agentool_name": agentool_name,
            "tools": tools_info
        }
    )


async def get_routing_config(ctx: RunContext[Any], agentool_name: str) -> ManagementOutput:
    """Get the routing configuration for an AgenTool."""
    config = AgenToolRegistry.get(agentool_name)
    if not config:
        return ManagementOutput(
            success=False,
            operation="get_routing_config",
            message=f"AgenTool '{agentool_name}' not found in registry",
            data=None
        )
    
    # Extract routing configuration details
    routing_info = {
        "operation_field": config.routing_config.operation_field,
        "operations": {}
    }
    
    # For each operation, get the tool and mapper details
    for op_name, (tool_name, mapper_func) in config.routing_config.operation_map.items():
        # Get tool metadata
        tool_meta = next((t for t in config.tools_metadata if t.name == tool_name), None)
        
        # Try to extract mapper info (this is a lambda function, so we can't get much detail)
        # But we can indicate what parameters are being mapped
        routing_info["operations"][op_name] = {
            "tool_name": tool_name,
            "tool_description": tool_meta.description if tool_meta else None,
            "tool_parameters": tool_meta.parameters if tool_meta else [],
            "is_async": tool_meta.is_async if tool_meta else None,
            "has_mapper": mapper_func is not None
        }
    
    return ManagementOutput(
        success=True,
        operation="get_routing_config",
        message=f"Retrieved routing config for AgenTool '{agentool_name}'",
        data={
            "agentool_name": agentool_name,
            "routing_config": routing_info,
            "total_operations": len(routing_info["operations"])
        }
    )


# Analysis and Documentation Operations

async def generate_dependency_graph(ctx: RunContext[Any], include_tools: bool = True) -> ManagementOutput:
    """Generate a dependency graph showing relationships between AgenTools."""
    graph = AgenToolRegistry.generate_dependency_graph(include_tools=include_tools)
    
    return ManagementOutput(
        success=True,
        operation="generate_dependency_graph",
        message="Generated dependency graph",
        data={
            "generated_at": datetime.now().isoformat(),
            "include_tools": include_tools,
            "dependency_graph": graph
        }
    )


async def analyze_agentool_usage(ctx: RunContext[Any], agentool_name: Optional[str] = None) -> ManagementOutput:
    """Analyze usage patterns and relationships for AgenTools."""
    if agentool_name:
        # Analyze specific AgenTool
        config = AgenToolRegistry.get(agentool_name)
        if not config:
            return {
                "success": False,
                "error": f"AgenTool '{agentool_name}' not found in registry"
            }
        
        # Find dependents (AgenTools that depend on this one)
        dependents = []
        for name, other_config in AgenToolRegistry._configs.items():
            if agentool_name in other_config.dependencies:
                dependents.append(name)
        
        analysis = {
            "agentool_name": agentool_name,
            "dependencies": config.dependencies,
            "dependents": dependents,
            "operations_count": len(config.routing_config.operation_map),
            "tools_count": len(config.tools_metadata),
            "is_leaf": len(dependents) == 0,
            "is_root": len(config.dependencies) == 0
        }
    else:
        # Analyze all AgenTools
        all_names = AgenToolRegistry.list_names()
        analysis = {
            "total_agentools": len(all_names),
            "leaf_agentools": [],  # No dependents
            "root_agentools": [],  # No dependencies
            "most_depended_on": [],
            "most_dependencies": []
        }
        
        dependency_counts = {}
        dependent_counts = {}
        
        for name in all_names:
            config = AgenToolRegistry.get(name)
            if config:
                dependency_counts[name] = len(config.dependencies)
                
                # Count dependents
                dependents = 0
                for other_name, other_config in AgenToolRegistry._configs.items():
                    if name in other_config.dependencies:
                        dependents += 1
                dependent_counts[name] = dependents
                
                if dependents == 0:
                    analysis["leaf_agentools"].append(name)
                if len(config.dependencies) == 0:
                    analysis["root_agentools"].append(name)
        
        # Find most depended on
        most_depended = sorted(dependent_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        analysis["most_depended_on"] = [{"name": name, "dependent_count": count} for name, count in most_depended]
        
        # Find those with most dependencies
        most_deps = sorted(dependency_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        analysis["most_dependencies"] = [{"name": name, "dependency_count": count} for name, count in most_deps]
    
    return ManagementOutput(
        success=True,
        operation="analyze_agentool_usage",
        message="Usage analysis complete",
        data={"analysis": analysis}
    )


async def validate_dependencies(ctx: RunContext[Any], agentool_name: Optional[str] = None) -> ManagementOutput:
    """Validate that all dependencies for AgenTools are available."""
    results = []
    
    if agentool_name:
        # Validate specific AgenTool
        agentools_to_check = [agentool_name]
    else:
        # Validate all AgenTools
        agentools_to_check = AgenToolRegistry.list_names()
    
    for name in agentools_to_check:
        config = AgenToolRegistry.get(name)
        if not config:
            results.append({
                "agentool_name": name,
                "valid": False,
                "error": "AgenTool not found in registry"
            })
            continue
        
        missing_deps = []
        available_deps = []
        
        for dep in config.dependencies:
            if AgenToolRegistry.get(dep) is not None:
                available_deps.append(dep)
            else:
                missing_deps.append(dep)
        
        results.append({
            "agentool_name": name,
            "valid": len(missing_deps) == 0,
            "dependencies": config.dependencies,
            "available_dependencies": available_deps,
            "missing_dependencies": missing_deps
        })
    
    all_valid = all(result["valid"] for result in results)
    
    return ManagementOutput(
        success=True,
        operation="validate_dependencies",
        message="Dependencies validated" if all_valid else "Some dependencies are missing",
        data={
            "all_valid": all_valid,
            "validation_results": results
        }
    )


async def get_api_specification(ctx: RunContext[Any], format: str = 'json') -> ManagementOutput:
    """Generate an OpenAPI-like specification for all AgenTools."""
    spec = AgenToolRegistry.generate_api_spec()
    
    if format.lower() == 'yaml':
        try:
            import yaml
            spec_formatted = yaml.dump(spec, default_flow_style=False)
        except ImportError:
            return ManagementOutput(
                success=False,
                operation="get_api_specification",
                message="PyYAML not available for YAML format",
                data=None
            )
    else:
        spec_formatted = spec
    
    return ManagementOutput(
        success=True,
        operation="get_api_specification",
        message=f"Generated API specification in {format} format",
        data={
            "format": format,
            "specification": spec_formatted
        }
    )


async def generate_docs(ctx: RunContext[Any], format: str = 'markdown', 
                       agentool_name: Optional[str] = None) -> ManagementOutput:
    """Generate documentation for AgenTools."""
    if format.lower() == 'markdown':
        if agentool_name:
            # Generate docs for specific AgenTool
            config = AgenToolRegistry.get(agentool_name)
            if not config:
                return ManagementOutput(
                    success=False,
                    operation="generate_docs",
                    message=f"AgenTool '{agentool_name}' not found in registry",
                    data=None
                )
            
            # Create single AgenTool markdown doc
            lines = [f"# {agentool_name}", ""]
            
            if config.description:
                lines.extend([f"**Description:** {config.description}", ""])
            
            lines.extend([
                f"**Version:** {config.version}",
                f"**Created:** {config.created_at.isoformat()}",
                f"**Updated:** {config.updated_at.isoformat()}",
                ""
            ])
            
            if config.tags:
                lines.extend([f"**Tags:** {', '.join(config.tags)}", ""])
            
            # Operations
            lines.extend(["## Operations", ""])
            for op_name, (tool_name, _) in config.routing_config.operation_map.items():
                tool_meta = next((t for t in config.tools_metadata if t.name == tool_name), None)
                if tool_meta and tool_meta.description:
                    lines.append(f"- **{op_name}**: {tool_meta.description}")
                else:
                    lines.append(f"- **{op_name}**: Calls `{tool_name}`")
            lines.append("")
            
            # Schema
            lines.extend(["## Input Schema", "```json"])
            schema = config.input_schema.model_json_schema()
            lines.extend([json.dumps(schema, indent=2), "```", ""])
            
            docs = "\n".join(lines)
        else:
            # Generate docs for all AgenTools
            docs = AgenToolRegistry.generate_markdown_docs()
        
        return ManagementOutput(
            success=True,
            operation="generate_docs",
            message="Generated documentation in markdown format",
            data={
                "format": "markdown",
                "documentation": docs
            }
        )
    else:
        # JSON format
        if agentool_name:
            detailed = AgenToolRegistry.list_detailed()
            agentool_docs = next((a for a in detailed if a["name"] == agentool_name), None)
            if not agentool_docs:
                return ManagementOutput(
                    success=False,
                    operation="generate_docs",
                    message=f"AgenTool '{agentool_name}' not found in registry",
                    data=None
                )
            docs = agentool_docs
        else:
            docs = AgenToolRegistry.list_detailed()
        
        return ManagementOutput(
            success=True,
            operation="generate_docs",
            message="Generated documentation in JSON format",
            data={
                "format": "json",
                "documentation": docs
            }
        )


async def get_examples(ctx: RunContext[Any], agentool_name: str) -> ManagementOutput:
    """Get usage examples for a specific AgenTool."""
    config = AgenToolRegistry.get(agentool_name)
    if not config:
        return ManagementOutput(
            success=False,
            operation="get_examples",
            message=f"AgenTool '{agentool_name}' not found in registry",
            data=None
        )
    
    examples = config.examples if config.examples else []
    
    # If no stored examples, generate basic ones from operations
    if not examples:
        generated_examples = []
        for op_name in config.routing_config.operation_map.keys():
            example_input = {config.routing_config.operation_field: op_name}
            generated_examples.append({
                "title": f"Basic {op_name} operation",
                "input": example_input,
                "description": f"Example usage of the {op_name} operation"
            })
        examples = generated_examples
    
    return ManagementOutput(
        success=True,
        operation="get_examples",
        message=f"Retrieved examples for AgenTool '{agentool_name}'",
        data={
            "agentool_name": agentool_name,
            "examples": examples
        }
    )


async def export_catalog(ctx: RunContext[Any], format: str = 'json') -> ManagementOutput:
    """Export the full AgenTool catalog."""
    catalog = AgenToolRegistry.export_catalog()
    
    if format.lower() == 'yaml':
        try:
            import yaml
            catalog_formatted = yaml.dump(catalog, default_flow_style=False)
        except ImportError:
            return ManagementOutput(
                success=False,
                operation="export_catalog",
                message="PyYAML not available for YAML format",
                data=None
            )
    else:
        catalog_formatted = catalog
    
    return ManagementOutput(
        success=True,
        operation="export_catalog",
        message=f"Exported catalog in {format} format",
        data={
            "format": format,
            "catalog": catalog_formatted
        }
    )


async def generate_usage_guide(ctx: RunContext[Any], agentool_name: str) -> ManagementOutput:
    """Generate a usage guide for a specific AgenTool."""
    config = AgenToolRegistry.get(agentool_name)
    if not config:
        return ManagementOutput(
            success=False,
            operation="generate_usage_guide",
            message=f"AgenTool '{agentool_name}' not found in registry",
            data=None
        )
    
    guide = {
        "agentool_name": agentool_name,
        "description": config.description,
        "version": config.version,
        "getting_started": {
            "import_statement": f"from agentool.core.injector import get_injector",
            "usage_pattern": f"injector = get_injector()\nresult = await injector.run('{agentool_name}', input_data)"
        },
        "available_operations": [],
        "input_schema": config.input_schema.model_json_schema(),
        "examples": config.examples if config.examples else []
    }
    
    # Add operation details
    for op_name, (tool_name, _) in config.routing_config.operation_map.items():
        tool_meta = next((t for t in config.tools_metadata if t.name == tool_name), None)
        op_info = {
            "operation": op_name,
            "tool": tool_name,
            "description": tool_meta.description if tool_meta else None,
            "parameters": tool_meta.parameters if tool_meta else [],
            "example_input": {config.routing_config.operation_field: op_name}
        }
        guide["available_operations"].append(op_info)
    
    return ManagementOutput(
        success=True,
        operation="generate_usage_guide",
        message=f"Generated usage guide for AgenTool '{agentool_name}'",
        data={"usage_guide": guide}
    )


# Main routing function
async def manage_agentool(ctx: RunContext[Any], 
                         operation: str,
                         agentool_name: Optional[str] = None,
                         tags: Optional[List[str]] = None,
                         name_pattern: Optional[str] = None,
                         description: Optional[str] = None,
                         version: Optional[str] = "1.0.0",
                         tools: Optional[List[str]] = None,
                         config_data: Optional[Dict[str, Any]] = None,
                         format: Optional[str] = 'json',
                         include_examples: Optional[bool] = True,
                         include_schemas: Optional[bool] = True,
                         detailed: Optional[bool] = False,
                         include_tools: Optional[bool] = True) -> ManagementOutput:
    """Main routing function for AgenTool management operations."""
    
    # Get injector for logging
    from agentool.core.injector import get_injector
    injector = get_injector()
    
    # Log the operation start
    await injector.run('logging', {
        'operation': 'log',
        'level': 'INFO',
        'logger_name': 'management',
        'message': f"Management operation '{operation}' started",
        'data': {
            'operation': operation,
            'agentool_name': agentool_name,
            'tags': tags,
            'detailed': detailed,
            'format': format
        }
    })
    
    try:
        result = None
        
        # Registry introspection operations
        if operation == 'list_agentools':
            result = await list_agentools(ctx, detailed=detailed)
        
        elif operation == 'get_agentool_info':
            if not agentool_name:
                result = ManagementOutput(
                    success=False,
                    operation=operation,
                    message="agentool_name is required for this operation",
                    data=None
                )
            else:
                result = await get_agentool_info(ctx, agentool_name, detailed=detailed)
        
        elif operation == 'get_agentool_schema':
            if not agentool_name:
                result = ManagementOutput(
                    success=False,
                    operation=operation,
                    message="agentool_name is required for this operation",
                    data=None
                )
            else:
                result = await get_agentool_schema(ctx, agentool_name)
        
        elif operation == 'search_agentools':
            result = await search_agentools(ctx, tags=tags, name_pattern=name_pattern)
        
        elif operation == 'get_operations':
            if not agentool_name:
                result = ManagementOutput(
                    success=False,
                    operation=operation,
                    message="agentool_name is required for this operation",
                    data=None
                )
            else:
                result = await get_operations(ctx, agentool_name)
        
        elif operation == 'get_tools_info':
            if not agentool_name:
                result = ManagementOutput(
                    success=False,
                    operation=operation,
                    message="agentool_name is required for this operation",
                    data=None
                )
            else:
                result = await get_tools_info(ctx, agentool_name)
        
        elif operation == 'get_routing_config':
            if not agentool_name:
                result = ManagementOutput(
                    success=False,
                    operation=operation,
                    message="agentool_name is required for this operation",
                    data=None
                )
            else:
                result = await get_routing_config(ctx, agentool_name)
        
        # Analysis and documentation operations
        elif operation == 'generate_dependency_graph':
            result = await generate_dependency_graph(ctx, include_tools=include_tools)
        
        elif operation == 'analyze_agentool_usage':
            result = await analyze_agentool_usage(ctx, agentool_name=agentool_name)
        
        elif operation == 'validate_dependencies':
            result = await validate_dependencies(ctx, agentool_name=agentool_name)
        
        elif operation == 'get_api_specification':
            result = await get_api_specification(ctx, format=format)
        
        elif operation == 'generate_docs':
            result = await generate_docs(ctx, format=format, agentool_name=agentool_name)
        
        elif operation == 'get_examples':
            if not agentool_name:
                result = ManagementOutput(
                    success=False,
                    operation=operation,
                    message="agentool_name is required for this operation",
                    data=None
                )
            else:
                result = await get_examples(ctx, agentool_name)
        
        elif operation == 'export_catalog':
            result = await export_catalog(ctx, format=format)
        
        elif operation == 'generate_usage_guide':
            if not agentool_name:
                result = ManagementOutput(
                    success=False,
                    operation=operation,
                    message="agentool_name is required for this operation",
                    data=None
                )
            else:
                result = await generate_usage_guide(ctx, agentool_name)
        
        # Placeholder for future operations
        elif operation in ['create_agentool_config', 'update_agentool_config', 
                          'register_agentool', 'unregister_agentool', 'validate_agentool_config',
                          'create_simple_agentool', 'create_tool_function',
                          'validate_tool_signature', 'infer_routing_config']:
            result = ManagementOutput(
                success=False,
                operation=operation,
                message=f"Operation '{operation}' is not yet implemented",
                data=None
            )
        
        else:
            result = ManagementOutput(
                success=False,
                operation=operation,
                message=f"Unknown operation: {operation}",
                data=None
            )
        
        # Log successful completion
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'management',
            'message': f"Management operation '{operation}' completed successfully",
            'data': {
                'operation': operation,
                'agentool_name': agentool_name,
                'success': result.success,
                'message': result.message
            }
        })
        
        return result
    
    except Exception as e:
        # Log error
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'management',
            'message': f"Management operation '{operation}' failed with error",
            'data': {
                'operation': operation,
                'agentool_name': agentool_name,
                'error': str(e),
                'error_type': type(e).__name__
            }
        })
        
        return ManagementOutput(
            success=False,
            operation=operation,
            message=f"Error executing operation '{operation}': {str(e)}",
            data=None
        )


def create_agentool_management_agent():
    """Create and return the AgenTool Management agent."""
    
    # Define routing configuration
    routing_config = RoutingConfig(
        operation_field='operation',
        operation_map={
            # Registry introspection operations
            'list_agentools': ('manage_agentool', lambda x: {
                'operation': x.operation,
                'detailed': x.detailed
            }),
            'get_agentool_info': ('manage_agentool', lambda x: {
                'operation': x.operation,
                'agentool_name': x.agentool_name,
                'detailed': x.detailed
            }),
            'get_agentool_schema': ('manage_agentool', lambda x: {
                'operation': x.operation,
                'agentool_name': x.agentool_name
            }),
            'search_agentools': ('manage_agentool', lambda x: {
                'operation': x.operation,
                'tags': x.tags,
                'name_pattern': x.name_pattern
            }),
            'get_operations': ('manage_agentool', lambda x: {
                'operation': x.operation,
                'agentool_name': x.agentool_name
            }),
            'get_tools_info': ('manage_agentool', lambda x: {
                'operation': x.operation,
                'agentool_name': x.agentool_name
            }),
            'get_routing_config': ('manage_agentool', lambda x: {
                'operation': x.operation,
                'agentool_name': x.agentool_name
            }),
            
            # Analysis and documentation operations
            'generate_dependency_graph': ('manage_agentool', lambda x: {
                'operation': x.operation,
                'include_tools': x.include_tools
            }),
            'analyze_agentool_usage': ('manage_agentool', lambda x: {
                'operation': x.operation,
                'agentool_name': x.agentool_name
            }),
            'validate_dependencies': ('manage_agentool', lambda x: {
                'operation': x.operation,
                'agentool_name': x.agentool_name
            }),
            'get_api_specification': ('manage_agentool', lambda x: {
                'operation': x.operation,
                'format': x.format
            }),
            'generate_docs': ('manage_agentool', lambda x: {
                'operation': x.operation,
                'format': x.format,
                'agentool_name': x.agentool_name
            }),
            'get_examples': ('manage_agentool', lambda x: {
                'operation': x.operation,
                'agentool_name': x.agentool_name
            }),
            'export_catalog': ('manage_agentool', lambda x: {
                'operation': x.operation,
                'format': x.format
            }),
            'generate_usage_guide': ('manage_agentool', lambda x: {
                'operation': x.operation,
                'agentool_name': x.agentool_name
            }),
            
            # Placeholder for future operations
            'create_agentool_config': ('manage_agentool', lambda x: {
                'operation': x.operation,
                'agentool_name': x.agentool_name,
                'config_data': x.config_data
            }),
            'update_agentool_config': ('manage_agentool', lambda x: {
                'operation': x.operation,
                'agentool_name': x.agentool_name,
                'config_data': x.config_data
            }),
            'register_agentool': ('manage_agentool', lambda x: {
                'operation': x.operation,
                'agentool_name': x.agentool_name,
                'config_data': x.config_data
            }),
            'unregister_agentool': ('manage_agentool', lambda x: {
                'operation': x.operation,
                'agentool_name': x.agentool_name
            }),
            'validate_agentool_config': ('manage_agentool', lambda x: {
                'operation': x.operation,
                'config_data': x.config_data
            }),
            'create_simple_agentool': ('manage_agentool', lambda x: {
                'operation': x.operation,
                'agentool_name': x.agentool_name,
                'description': x.description,
                'version': x.version,
                'tools': x.tools
            }),
            'create_tool_function': ('manage_agentool', lambda x: {
                'operation': x.operation,
                'config_data': x.config_data
            }),
            'validate_tool_signature': ('manage_agentool', lambda x: {
                'operation': x.operation,
                'config_data': x.config_data
            }),
            'infer_routing_config': ('manage_agentool', lambda x: {
                'operation': x.operation,
                'tools': x.tools
            })
        }
    )
    
    # Create the AgenTool
    agent = create_agentool(
        name='agentool_mgmt',
        input_schema=AgenToolManagementInput,
        output_type=ManagementOutput,
        routing_config=routing_config,
        tools=[manage_agentool],
        system_prompt="Manage and introspect AgenTool registry operations with comprehensive capabilities.",
        description="Comprehensive management and introspection toolkit for AgenTool registry system",
        version="1.0.0",
        tags=["management", "introspection", "registry", "documentation"],
        dependencies=["logging"]  # Uses logging for operation tracking
    )
    
    return agent