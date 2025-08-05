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
from pydantic import BaseModel, Field
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


# Registry Introspection Operations

async def list_agentools(ctx: RunContext[Any], detailed: bool = False) -> Dict[str, Any]:
    """List all registered AgenTools with basic or detailed information."""
    if detailed:
        agentools = AgenToolRegistry.list_detailed()
        return {
            "success": True,
            "count": len(agentools),
            "agentools": agentools
        }
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
        
        return {
            "success": True,
            "count": len(names),
            "agentools": basic_info
        }


async def get_agentool_info(ctx: RunContext[Any], agentool_name: str, detailed: bool = True) -> Dict[str, Any]:
    """Get detailed information about a specific AgenTool."""
    config = AgenToolRegistry.get(agentool_name)
    if not config:
        return {
            "success": False,
            "error": f"AgenTool '{agentool_name}' not found in registry"
        }
    
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
    
    return {
        "success": True,
        "agentool": info
    }


async def get_agentool_schema(ctx: RunContext[Any], agentool_name: str) -> Dict[str, Any]:
    """Get the JSON schema for an AgenTool's input."""
    schema = AgenToolRegistry.get_schema(agentool_name)
    if not schema:
        return {
            "success": False,
            "error": f"AgenTool '{agentool_name}' not found in registry"
        }
    
    return {
        "success": True,
        "agentool_name": agentool_name,
        "schema": schema
    }


async def search_agentools(ctx: RunContext[Any], tags: Optional[List[str]] = None, 
                          name_pattern: Optional[str] = None) -> Dict[str, Any]:
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
    
    return {
        "success": True,
        "search_criteria": {
            "tags": tags,
            "name_pattern": name_pattern
        },
        "count": len(results),
        "results": detailed_results
    }


async def get_operations(ctx: RunContext[Any], agentool_name: str) -> Dict[str, Any]:
    """Get available operations for an AgenTool."""
    operations = AgenToolRegistry.get_operations(agentool_name)
    if operations is None:
        return {
            "success": False,
            "error": f"AgenTool '{agentool_name}' not found in registry"
        }
    
    return {
        "success": True,
        "agentool_name": agentool_name,
        "operations": operations
    }


async def get_tools_info(ctx: RunContext[Any], agentool_name: str) -> Dict[str, Any]:
    """Get information about tools used by an AgenTool."""
    tools_info = AgenToolRegistry.get_tools_info(agentool_name)
    if tools_info is None:
        return {
            "success": False,
            "error": f"AgenTool '{agentool_name}' not found in registry"
        }
    
    return {
        "success": True,
        "agentool_name": agentool_name,
        "tools": tools_info
    }


async def get_routing_config(ctx: RunContext[Any], agentool_name: str) -> Dict[str, Any]:
    """Get the routing configuration for an AgenTool."""
    config = AgenToolRegistry.get(agentool_name)
    if not config:
        return {
            "success": False,
            "error": f"AgenTool '{agentool_name}' not found in registry"
        }
    
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
    
    return {
        "success": True,
        "agentool_name": agentool_name,
        "routing_config": routing_info,
        "total_operations": len(routing_info["operations"])
    }


# Analysis and Documentation Operations

async def generate_dependency_graph(ctx: RunContext[Any], include_tools: bool = True) -> Dict[str, Any]:
    """Generate a dependency graph showing relationships between AgenTools."""
    graph = AgenToolRegistry.generate_dependency_graph(include_tools=include_tools)
    
    return {
        "success": True,
        "generated_at": datetime.now().isoformat(),
        "include_tools": include_tools,
        "dependency_graph": graph
    }


async def analyze_agentool_usage(ctx: RunContext[Any], agentool_name: Optional[str] = None) -> Dict[str, Any]:
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
    
    return {
        "success": True,
        "analysis": analysis
    }


async def validate_dependencies(ctx: RunContext[Any], agentool_name: Optional[str] = None) -> Dict[str, Any]:
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
    
    return {
        "success": True,
        "all_valid": all_valid,
        "validation_results": results
    }


async def get_api_specification(ctx: RunContext[Any], format: str = 'json') -> Dict[str, Any]:
    """Generate an OpenAPI-like specification for all AgenTools."""
    spec = AgenToolRegistry.generate_api_spec()
    
    if format.lower() == 'yaml':
        try:
            import yaml
            spec_formatted = yaml.dump(spec, default_flow_style=False)
        except ImportError:
            return {
                "success": False,
                "error": "PyYAML not available for YAML format"
            }
    else:
        spec_formatted = spec
    
    return {
        "success": True,
        "format": format,
        "specification": spec_formatted
    }


async def generate_docs(ctx: RunContext[Any], format: str = 'markdown', 
                       agentool_name: Optional[str] = None) -> Dict[str, Any]:
    """Generate documentation for AgenTools."""
    if format.lower() == 'markdown':
        if agentool_name:
            # Generate docs for specific AgenTool
            config = AgenToolRegistry.get(agentool_name)
            if not config:
                return {
                    "success": False,
                    "error": f"AgenTool '{agentool_name}' not found in registry"
                }
            
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
        
        return {
            "success": True,
            "format": "markdown",
            "documentation": docs
        }
    else:
        # JSON format
        if agentool_name:
            detailed = AgenToolRegistry.list_detailed()
            agentool_docs = next((a for a in detailed if a["name"] == agentool_name), None)
            if not agentool_docs:
                return {
                    "success": False,
                    "error": f"AgenTool '{agentool_name}' not found in registry"
                }
            docs = agentool_docs
        else:
            docs = AgenToolRegistry.list_detailed()
        
        return {
            "success": True,
            "format": "json",
            "documentation": docs
        }


async def get_examples(ctx: RunContext[Any], agentool_name: str) -> Dict[str, Any]:
    """Get usage examples for a specific AgenTool."""
    config = AgenToolRegistry.get(agentool_name)
    if not config:
        return {
            "success": False,
            "error": f"AgenTool '{agentool_name}' not found in registry"
        }
    
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
    
    return {
        "success": True,
        "agentool_name": agentool_name,
        "examples": examples
    }


async def export_catalog(ctx: RunContext[Any], format: str = 'json') -> Dict[str, Any]:
    """Export the full AgenTool catalog."""
    catalog = AgenToolRegistry.export_catalog()
    
    if format.lower() == 'yaml':
        try:
            import yaml
            catalog_formatted = yaml.dump(catalog, default_flow_style=False)
        except ImportError:
            return {
                "success": False,
                "error": "PyYAML not available for YAML format"
            }
    else:
        catalog_formatted = catalog
    
    return {
        "success": True,
        "format": format,
        "catalog": catalog_formatted
    }


async def generate_usage_guide(ctx: RunContext[Any], agentool_name: str) -> Dict[str, Any]:
    """Generate a usage guide for a specific AgenTool."""
    config = AgenToolRegistry.get(agentool_name)
    if not config:
        return {
            "success": False,
            "error": f"AgenTool '{agentool_name}' not found in registry"
        }
    
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
    
    return {
        "success": True,
        "usage_guide": guide
    }


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
                         include_tools: Optional[bool] = True) -> Dict[str, Any]:
    """Main routing function for AgenTool management operations."""
    
    try:
        # Registry introspection operations
        if operation == 'list_agentools':
            return await list_agentools(ctx, detailed=detailed)
        
        elif operation == 'get_agentool_info':
            if not agentool_name:
                return {"success": False, "error": "agentool_name is required for this operation"}
            return await get_agentool_info(ctx, agentool_name, detailed=detailed)
        
        elif operation == 'get_agentool_schema':
            if not agentool_name:
                return {"success": False, "error": "agentool_name is required for this operation"}
            return await get_agentool_schema(ctx, agentool_name)
        
        elif operation == 'search_agentools':
            return await search_agentools(ctx, tags=tags, name_pattern=name_pattern)
        
        elif operation == 'get_operations':
            if not agentool_name:
                return {"success": False, "error": "agentool_name is required for this operation"}
            return await get_operations(ctx, agentool_name)
        
        elif operation == 'get_tools_info':
            if not agentool_name:
                return {"success": False, "error": "agentool_name is required for this operation"}
            return await get_tools_info(ctx, agentool_name)
        
        elif operation == 'get_routing_config':
            if not agentool_name:
                return {"success": False, "error": "agentool_name is required for this operation"}
            return await get_routing_config(ctx, agentool_name)
        
        # Analysis and documentation operations
        elif operation == 'generate_dependency_graph':
            return await generate_dependency_graph(ctx, include_tools=include_tools)
        
        elif operation == 'analyze_agentool_usage':
            return await analyze_agentool_usage(ctx, agentool_name=agentool_name)
        
        elif operation == 'validate_dependencies':
            return await validate_dependencies(ctx, agentool_name=agentool_name)
        
        elif operation == 'get_api_specification':
            return await get_api_specification(ctx, format=format)
        
        elif operation == 'generate_docs':
            return await generate_docs(ctx, format=format, agentool_name=agentool_name)
        
        elif operation == 'get_examples':
            if not agentool_name:
                return {"success": False, "error": "agentool_name is required for this operation"}
            return await get_examples(ctx, agentool_name)
        
        elif operation == 'export_catalog':
            return await export_catalog(ctx, format=format)
        
        elif operation == 'generate_usage_guide':
            if not agentool_name:
                return {"success": False, "error": "agentool_name is required for this operation"}
            return await generate_usage_guide(ctx, agentool_name)
        
        # Placeholder for future operations
        elif operation in ['create_agentool_config', 'update_agentool_config', 
                          'register_agentool', 'unregister_agentool', 'validate_agentool_config',
                          'create_simple_agentool', 'create_tool_function',
                          'validate_tool_signature', 'infer_routing_config']:
            return {
                "success": False,
                "error": f"Operation '{operation}' is not yet implemented"
            }
        
        else:
            return {
                "success": False,
                "error": f"Unknown operation: {operation}"
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Error executing operation '{operation}': {str(e)}"
        }


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
        routing_config=routing_config,
        tools=[manage_agentool],
        system_prompt="Manage and introspect AgenTool registry operations with comprehensive capabilities.",
        description="Comprehensive management and introspection toolkit for AgenTool registry system",
        version="1.0.0",
        tags=["management", "introspection", "registry", "documentation"]
    )
    
    return agent