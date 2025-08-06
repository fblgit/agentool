"""
Templates AgenTool - Jinja2-based template rendering with storage integration.

This toolkit provides template management and rendering capabilities using Jinja2,
with automatic loading of templates from a templates directory and support for
variable references to storage_kv and storage_fs.

Features:
- Auto-loading of templates from templates/ directory at creation
- Variable resolution from storage_kv and storage_fs
- Template validation before saving
- Template rendering with variable substitution
- Ad-hoc template execution
- Integration with logging and metrics

Example Usage:
    >>> from agentoolkit.system.templates import create_templates_agent
    >>> from agentool.core.injector import get_injector
    >>> 
    >>> # Create and register the agent (loads all templates)
    >>> agent = create_templates_agent()
    >>> 
    >>> # Use through injector
    >>> injector = get_injector()
    >>> 
    >>> # Render a template with variables
    >>> result = await injector.run('templates', {
    ...     "operation": "render",
    ...     "template_name": "welcome_email",
    ...     "variables": {
    ...         "user_name": "!ref:storage_kv:user_123_name",
    ...         "company": "Acme Corp"
    ...     }
    ... })
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import RunContext

try:
    from jinja2 import Environment, FileSystemLoader, TemplateError, Template
    from jinja2.exceptions import TemplateSyntaxError, UndefinedError
except ImportError:
    raise ImportError("jinja2 is required for templates toolkit. Install with: pip install jinja2")

from agentool.base import BaseOperationInput
from agentool import create_agentool
from agentool.core.registry import RoutingConfig
from agentool.core.injector import get_injector


class TemplatesInput(BaseOperationInput):
    """Input schema for template operations."""
    operation: Literal['render', 'save', 'list', 'validate', 'exec'] = Field(
        description="The template operation to perform"
    )
    
    # Template identification
    template_name: Optional[str] = Field(None, description="Name of the template (without extension)")
    template_content: Optional[str] = Field(None, description="Template content for save/validate/exec")
    
    # Rendering variables
    variables: Optional[Dict[str, Any]] = Field(None, description="Variables for template rendering")
    
    # Options
    strict: bool = Field(default=False, description="Fail on undefined variables")
    
    @field_validator('template_name')
    def validate_template_name(cls, v, info):
        """Validate template_name is provided for operations that require it."""
        operation = info.data.get('operation')
        if operation in ['render'] and not v:
            raise ValueError(f"template_name is required for {operation} operation")
        return v
    
    @field_validator('template_content')
    def validate_template_content(cls, v, info):
        """Validate template_content is provided for operations that require it."""
        operation = info.data.get('operation')
        if operation in ['save', 'validate', 'exec'] and not v:
            raise ValueError(f"template_content is required for {operation} operation")
        return v


class TemplatesOutput(BaseModel):
    """Structured output for template operations."""
    success: bool = Field(description="Whether the operation succeeded")
    operation: str = Field(description="The operation that was performed")
    message: str = Field(description="Human-readable result message")
    data: Optional[Any] = Field(None, description="Operation-specific data")


# Global template storage
_templates: Dict[str, Template] = {}
_templates_dir: str = "templates"
_jinja_env: Optional[Environment] = None


def _initialize_templates() -> None:
    """Initialize the Jinja2 environment and load all templates."""
    global _jinja_env, _templates, _templates_dir
    
    # Create templates directory if it doesn't exist
    templates_path = Path(_templates_dir)
    if not templates_path.exists():
        templates_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize Jinja2 environment
    _jinja_env = Environment(
        loader=FileSystemLoader(str(templates_path)),
        autoescape=False,  # No auto-escaping by default
        trim_blocks=True,
        lstrip_blocks=True
    )
    
    # Load all templates recursively
    _templates.clear()
    template_files = list(templates_path.glob("**/*.jinja")) + list(templates_path.glob("**/*.j2"))
    
    for template_file in template_files:
        # Create template name with subdirectory path (e.g., "system/analyzer" for system/analyzer.jinja)
        relative_path = template_file.relative_to(templates_path)
        template_name = str(relative_path.with_suffix(''))  # Remove extension
        
        try:
            # Use relative path for Jinja2 loader
            template = _jinja_env.get_template(str(relative_path))
            _templates[template_name] = template
        except Exception as e:
            # Skip templates that fail to load
            print(f"Warning: Failed to load template {template_file}: {e}")


async def _resolve_variable_references(variables: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve variable references to storage_kv and storage_fs.
    
    Args:
        variables: Dictionary with potential reference values
        
    Returns:
        Dictionary with resolved values
    """
    if not variables:
        return {}
    
    resolved = {}
    injector = get_injector()
    
    for key, value in variables.items():
        if isinstance(value, str) and value.startswith("!ref:"):
            # Parse reference format: !ref:storage_type:path
            parts = value.split(":", 2)
            if len(parts) == 3:
                _, storage_type, path = parts
                
                if storage_type == "storage_kv":
                    # Resolve from storage_kv
                    try:
                        result = await injector.run('storage_kv', {
                            "operation": "get",
                            "key": path
                        })
                        
                        # storage_kv returns typed StorageKvOutput
                        # Check if key exists and has value
                        if result.success and result.data and result.data.get("exists"):
                            resolved[key] = result.data["value"]
                        else:
                            resolved[key] = f"<undefined:{value}>"
                    except Exception as e:
                        # Log the error for debugging
                        resolved[key] = f"<error:{value}>"
                
                elif storage_type == "storage_fs":
                    # Resolve from storage_fs
                    try:
                        result = await injector.run('storage_fs', {
                            "operation": "read",
                            "path": path
                        })
                        
                        # storage_fs returns typed StorageFsOutput
                        # Check if file was read successfully
                        if result.success and result.data:
                            resolved[key] = result.data["content"]
                        else:
                            resolved[key] = f"<undefined:{value}>"
                    except Exception as e:
                        # Log the error for debugging
                        resolved[key] = f"<error:{value}>"
                else:
                    resolved[key] = f"<invalid_ref:{value}>"
            else:
                resolved[key] = f"<invalid_ref:{value}>"
        else:
            # Keep non-reference values as-is
            resolved[key] = value
    
    return resolved


async def _log_operation(operation: str, success: bool, details: Dict[str, Any]) -> None:
    """Log template operations."""
    try:
        injector = get_injector()
        level = "INFO" if success else "ERROR"
        
        await injector.run('logging', {
            "operation": "log",
            "level": level,
            "logger_name": "templates",
            "message": f"Template operation: {operation}",
            "data": details
        })
    except Exception:
        # Logging failure shouldn't break the operation
        pass


async def _track_metric(metric_name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
    """Track metrics for template operations."""
    try:
        injector = get_injector()
        
        await injector.run('metrics', {
            "operation": "increment",
            "name": f"agentool.templates.{metric_name}",
            "value": value,
            "labels": labels
        })
    except Exception:
        # Metrics failure shouldn't break the operation
        pass


async def templates_render(ctx: RunContext[Any], template_name: str, variables: Optional[Dict[str, Any]], 
                          strict: bool) -> TemplatesOutput:
    """
    Render a pre-loaded template with variables.
    
    Args:
        ctx: Runtime context
        template_name: Name of the template to render
        variables: Variables for rendering (may contain storage references)
        strict: Whether to fail on undefined variables
        
    Returns:
        TemplatesOutput with rendered content
    """
    try:
        # Check if template exists
        if template_name not in _templates:
            error_msg = f"Template '{template_name}' not found in loaded templates"
            await _log_operation("render", False, {"template": template_name, "error": error_msg})
            await _track_metric("validation.errors", labels={"template": template_name, "error_type": "not_found"})
            # Return success=False for discovery operation (404-like)
            return TemplatesOutput(
                success=False,
                operation="render",
                message=error_msg,
                data=None
            )
        
        # Resolve variable references
        resolved_vars = await _resolve_variable_references(variables or {})
        
        # Get the template
        template = _templates[template_name]
        
        # Render the template
        try:
            # Get the template file path - try both .jinja and .j2 extensions
            template_path = None
            for ext in ['.jinja', '.j2']:
                potential_path = Path(_templates_dir) / f"{template_name}{ext}"
                if potential_path.exists():
                    template_path = potential_path
                    break
            
            if not template_path:
                raise FileNotFoundError(f"Template file not found for {template_name}")
            
            # Read template content
            template_content = template_path.read_text()
            
            if strict:
                # Strict mode - fail on undefined
                from jinja2 import StrictUndefined
                strict_env = Environment(
                    undefined=StrictUndefined,
                    autoescape=False,
                    trim_blocks=True,
                    lstrip_blocks=True
                )
                template_obj = strict_env.from_string(template_content)
                rendered = template_obj.render(**resolved_vars)
            else:
                # Lenient mode - undefined variables become empty strings
                from jinja2 import Undefined
                class SilentUndefined(Undefined):
                    def _fail_with_undefined_error(self, *args, **kwargs):
                        return ''
                    __str__ = lambda self: ''
                    __repr__ = lambda self: ''
                    __bool__ = lambda self: False
                    __nonzero__ = lambda self: False
                
                lenient_env = Environment(
                    undefined=SilentUndefined,
                    autoescape=False,
                    trim_blocks=True,
                    lstrip_blocks=True
                )
                template_obj = lenient_env.from_string(template_content)
                rendered = template_obj.render(**resolved_vars)
            
            # Track successful render with template as label
            await _track_metric("render.count", labels={"template": template_name})
            await _log_operation("render", True, {
                "template": template_name,
                "variables_count": len(resolved_vars),
                "output_length": len(rendered)
            })
            
            return TemplatesOutput(
                success=True,
                operation="render",
                message=f"Successfully rendered template '{template_name}'",
                data={
                    "rendered": rendered,
                    "template_name": template_name,
                    "variables_provided": len(variables or {}),
                    "variables_resolved": len(resolved_vars)
                }
            )
            
        except UndefinedError as e:
            await _track_metric("render.errors", labels={"template": template_name, "error_type": "undefined_variable"})
            await _log_operation("render", False, {"template": template_name, "error": str(e)})
            raise ValueError(f"Undefined variable in template '{template_name}': {str(e)}")
            
    except Exception as e:
        await _log_operation("render", False, {"template": template_name, "error": str(e)})
        raise


async def templates_save(ctx: RunContext[Any], template_name: str, template_content: str) -> TemplatesOutput:
    """
    Save a new template to the templates directory.
    
    Args:
        ctx: Runtime context
        template_name: Name for the template (without extension)
        template_content: Template content to save
        
    Returns:
        TemplatesOutput with save result
    """
    try:
        # Validate template syntax first
        try:
            validation_result = await templates_validate(ctx, template_content)
            # If validation succeeded, we'll get a result with data.valid = True
        except ValueError as e:
            # Validation failed - re-raise with better error message
            await _track_metric("validation.errors", labels={"template": template_name, "error_type": "validation_failed"})
            raise ValueError(f"Template validation failed for '{template_name}': {str(e)}")
        
        # Save to file system
        injector = get_injector()
        file_path = os.path.join(_templates_dir, f"{template_name}.jinja")
        
        save_result = await injector.run('storage_fs', {
            "operation": "write",
            "path": file_path,
            "content": template_content,
            "create_parents": True
        })
        
        # storage_fs returns typed StorageFsOutput
        # If we got here without exception, the save succeeded
        
        # Reload the template into memory
        try:
            template = _jinja_env.get_template(f"{template_name}.jinja")
            _templates[template_name] = template
            
            # Update template count metric (use set, not increment)
            try:
                injector = get_injector()
                await injector.run('metrics', {
                    "operation": "set",
                    "name": "agentool.templates.count",
                    "value": len(_templates)
                })
            except Exception:
                pass  # Metrics failure shouldn't break the operation
            await _log_operation("save", True, {
                "template": template_name,
                "size": len(template_content),
                "total_templates": len(_templates)
            })
            
            return TemplatesOutput(
                success=True,
                operation="save",
                message=f"Successfully saved template '{template_name}'",
                data={
                    "template_name": template_name,
                    "file_path": file_path,
                    "size": len(template_content),
                    "total_templates": len(_templates)
                }
            )
            
        except Exception as e:
            await _log_operation("save", False, {"template": template_name, "error": str(e)})
            raise RuntimeError(f"Template saved but failed to load '{template_name}': {str(e)}")
            
    except Exception as e:
        await _log_operation("save", False, {"template": template_name, "error": str(e)})
        raise


async def templates_list(ctx: RunContext[Any]) -> TemplatesOutput:
    """
    List all available templates.
    
    Args:
        ctx: Runtime context
        
    Returns:
        TemplatesOutput with list of templates
    """
    try:
        template_list = []
        
        for name, template in _templates.items():
            # Get template file info
            file_path = os.path.join(_templates_dir, template.filename)
            
            template_info = {
                "name": name,
                "filename": template.filename,
                "file_path": file_path
            }
            
            # Try to get file size
            try:
                template_info["size"] = os.path.getsize(file_path)
            except:
                template_info["size"] = None
            
            template_list.append(template_info)
        
        # Sort by name
        template_list.sort(key=lambda x: x["name"])
        
        return TemplatesOutput(
            success=True,
            operation="list",
            message=f"Found {len(template_list)} templates",
            data={
                "templates": template_list,
                "count": len(template_list),
                "directory": _templates_dir
            }
        )
        
    except Exception as e:
        # For list operations, we can return an error response since it's non-critical
        raise RuntimeError(f"Error listing templates: {str(e)}") from e


async def templates_validate(ctx: RunContext[Any], template_content: str) -> TemplatesOutput:
    """
    Validate template syntax.
    
    Args:
        ctx: Runtime context
        template_content: Template content to validate
        
    Returns:
        TemplatesOutput with validation result
    """
    try:
        # Try to compile the template
        try:
            _jinja_env.from_string(template_content)
            
            # Extract variables used in the template
            from jinja2 import meta
            ast = _jinja_env.parse(template_content)
            variables = meta.find_undeclared_variables(ast)
            
            return TemplatesOutput(
                success=True,
                operation="validate",
                message="Template syntax is valid",
                data={
                    "valid": True,
                    "variables": list(variables),
                    "size": len(template_content)
                }
            )
            
        except TemplateSyntaxError as e:
            await _log_operation("validate", False, {"error": str(e), "line": e.lineno})
            raise ValueError(f"Template syntax error at line {e.lineno}: {e.message}") from e
            
    except ValueError:
        # Re-raise ValueError as-is (from TemplateSyntaxError)
        raise
    except Exception as e:
        raise RuntimeError(f"Error validating template: {str(e)}") from e


async def templates_exec(ctx: RunContext[Any], template_content: str, variables: Optional[Dict[str, Any]], 
                        strict: bool) -> TemplatesOutput:
    """
    Execute a template string directly (ad-hoc rendering).
    
    Args:
        ctx: Runtime context
        template_content: Template content to render
        variables: Variables for rendering
        strict: Whether to fail on undefined variables
        
    Returns:
        TemplatesOutput with rendered content
    """
    try:
        # Validate the template first
        try:
            validation_result = await templates_validate(ctx, template_content)
            # If validation succeeded, we'll get a result with data.valid = True
        except ValueError as e:
            # Validation failed - re-raise with better error message
            raise ValueError(f"Template validation failed: {str(e)}")
        
        # Resolve variable references
        resolved_vars = await _resolve_variable_references(variables or {})
        
        # Compile and render the template
        try:
            if strict:
                # Strict mode - undefined variables cause errors
                from jinja2 import StrictUndefined
                strict_env = Environment(
                    undefined=StrictUndefined,
                    autoescape=False,
                    trim_blocks=True,
                    lstrip_blocks=True
                )
                template_obj = strict_env.from_string(template_content)
                rendered = template_obj.render(**resolved_vars)
            else:
                # Lenient mode - undefined variables become empty strings
                from jinja2 import Undefined
                class SilentUndefined(Undefined):
                    def _fail_with_undefined_error(self, *args, **kwargs):
                        return ''
                    __str__ = lambda self: ''
                    __repr__ = lambda self: ''
                    __bool__ = lambda self: False
                    __nonzero__ = lambda self: False
                
                lenient_env = Environment(
                    undefined=SilentUndefined,
                    autoescape=False,
                    trim_blocks=True,
                    lstrip_blocks=True
                )
                template_obj = lenient_env.from_string(template_content)
                rendered = template_obj.render(**resolved_vars)
            
            await _log_operation("exec", True, {
                "template_size": len(template_content),
                "variables_count": len(resolved_vars),
                "output_length": len(rendered)
            })
            
            return TemplatesOutput(
                success=True,
                operation="exec",
                message="Successfully executed template",
                data={
                    "rendered": rendered,
                    "template_size": len(template_content),
                    "variables_provided": len(variables or {}),
                    "variables_resolved": len(resolved_vars)
                }
            )
            
        except UndefinedError as e:
            await _log_operation("exec", False, {"error": str(e)})
            raise ValueError(f"Undefined variable in template: {str(e)}")
            
    except Exception as e:
        await _log_operation("exec", False, {"error": str(e)})
        raise


def create_templates_agent(templates_dir: str = "templates"):
    """
    Create and return the Templates AgenTool.
    
    Args:
        templates_dir: Directory to load templates from (default: "templates")
        
    Returns:
        Agent configured for template operations
    """
    global _templates_dir
    _templates_dir = templates_dir
    
    # Initialize templates on creation
    _initialize_templates()
    
    # Log creation with template count
    # Commenting out the logging and metrics initialization to avoid issues with test teardown
    # This code runs at module import time and can cause issues with async context
    # TODO: Move this to a proper initialization method that can be called when needed
    pass
    
    # Original code kept for reference:
    # try:
    #     from agentool.core.injector import get_injector
    #     injector = get_injector()
    #     ...logging and metrics code...
    
    templates_routing = RoutingConfig(
        operation_field='operation',
        operation_map={
            'render': ('templates_render', lambda x: {
                'template_name': x.template_name,
                'variables': x.variables,
                'strict': x.strict
            }),
            'save': ('templates_save', lambda x: {
                'template_name': x.template_name,
                'template_content': x.template_content
            }),
            'list': ('templates_list', lambda x: {}),
            'validate': ('templates_validate', lambda x: {
                'template_content': x.template_content
            }),
            'exec': ('templates_exec', lambda x: {
                'template_content': x.template_content,
                'variables': x.variables,
                'strict': x.strict
            }),
        }
    )
    
    return create_agentool(
        name='templates',
        input_schema=TemplatesInput,
        routing_config=templates_routing,
        tools=[templates_render, templates_save, templates_list, templates_validate, templates_exec],
        output_type=TemplatesOutput,
        use_typed_output=True,  # Enable typed output for templates
        system_prompt="Handle Jinja2 template operations with storage integration.",
        description="Template rendering with Jinja2, supporting storage_kv and storage_fs variable references",
        version="1.0.0",
        tags=["templates", "jinja2", "rendering", "storage-integration"],
        dependencies=["storage_fs", "storage_kv", "logging", "metrics"],
        examples=[
            {
                "description": "Render a template with variables",
                "input": {
                    "operation": "render",
                    "template_name": "welcome_email",
                    "variables": {
                        "user_name": "John Doe",
                        "company": "!ref:storage_kv:company_name"
                    }
                },
                "output": {
                    "operation": "render",
                    "message": "Successfully rendered template 'welcome_email'",
                    "data": {"rendered": "Welcome John Doe to Acme Corp!"}
                }
            },
            {
                "description": "Save a new template",
                "input": {
                    "operation": "save",
                    "template_name": "greeting",
                    "template_content": "Hello {{ name }}!"
                },
                "output": {
                    "operation": "save",
                    "message": "Successfully saved template 'greeting'"
                }
            },
            {
                "description": "Execute ad-hoc template",
                "input": {
                    "operation": "exec",
                    "template_content": "Result: {{ value * 2 }}",
                    "variables": {"value": 21}
                },
                "output": {
                    "operation": "exec",
                    "message": "Successfully executed template",
                    "data": {"rendered": "Result: 42"}
                }
            }
        ]
    )


# Create the agent instance
agent = create_templates_agent()