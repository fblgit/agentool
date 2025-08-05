# Templates AgenToolkit

> **Note**: For guidelines on creating new AgenToolkits, see [CRAFTING_AGENTOOLS.md](../CRAFTING_AGENTOOLS.md). For testing patterns and examples, refer to [tests/agentoolkit/test_templates.py](../../../tests/agentoolkit/test_templates.py).

## Overview

The Templates AgenToolkit provides Jinja2-based template rendering with storage integration capabilities. It automatically loads templates from a templates directory and supports variable references to storage_kv and storage_fs systems, making it ideal for dynamic content generation.

### Key Features
- Auto-loading of templates from configurable templates directory
- Jinja2 template engine with full syntax support  
- Variable resolution from storage_kv and storage_fs using reference syntax
- Template validation before saving or execution
- Ad-hoc template execution without saving to filesystem
- Strict and lenient rendering modes
- Integration with logging and metrics for observability
- Template management (save, list, validate)

## Creation Method

```python
from agentoolkit.system.templates import create_templates_agent

# Create with default templates directory
agent = create_templates_agent()

# Create with custom templates directory
agent = create_templates_agent(templates_dir="/path/to/templates")
```

The creation function returns a fully configured AgenTool with name `'templates'`. Templates are automatically loaded from the specified directory (default: "templates/") on creation.

## Input Schema

### TemplatesInput

The input schema is a Pydantic model with the following fields:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `operation` | `Literal['render', 'save', 'list', 'validate', 'exec']` | Yes | - | The template operation to perform |
| `template_name` | `Optional[str]` | No | None | Name of template (without extension) for render/save |
| `template_content` | `Optional[str]` | No | None | Template content for save/validate/exec operations |
| `variables` | `Optional[Dict[str, Any]]` | No | None | Variables for template rendering |
| `strict` | `bool` | No | False | Fail on undefined variables in templates |

## Operations Schema

The routing configuration maps operations to tool functions:

| Operation | Tool Function | Required Parameters | Description |
|-----------|--------------|-------------------|-------------|
| `render` | `templates_render` | `template_name`, `variables`, `strict` | Render a pre-loaded template with variables |
| `save` | `templates_save` | `template_name`, `template_content` | Save new template to filesystem and load into memory |
| `list` | `templates_list` | - | List all available templates with metadata |
| `validate` | `templates_validate` | `template_content` | Validate template syntax and extract variables |
| `exec` | `templates_exec` | `template_content`, `variables`, `strict` | Execute template content directly (ad-hoc) |

## Output Schema

### TemplatesOutput

All operations return a `TemplatesOutput` model with:

| Field | Type | Description |
|-------|------|-------------|
| `operation` | `str` | The operation that was performed |
| `message` | `str` | Human-readable result message |
| `data` | `Optional[Any]` | Operation-specific data |

### Operation-Specific Data Fields

- **render**: `rendered`, `template_name`, `variables_provided`, `variables_resolved`
- **save**: `template_name`, `file_path`, `size`, `total_templates`
- **list**: `templates`, `count`, `directory`
- **validate**: `valid`, `variables`, `size`
- **exec**: `rendered`, `template_size`, `variables_provided`, `variables_resolved`

## Variable Reference System

The toolkit supports special reference syntax for dynamic variable resolution:

### Reference Syntax
- `!ref:storage_kv:key_name` - Resolve from storage_kv
- `!ref:storage_fs:file_path` - Resolve from storage_fs (file content)

### Resolution Behavior
- Successful resolution: Returns actual value from storage
- Key/file not found: Returns `<undefined:reference>`
- Resolution error: Returns `<error:reference>`
- Invalid reference format: Returns `<invalid_ref:reference>`

## Dependencies

This AgenToolkit depends on:
- **storage_fs**: Used for template file operations
- **storage_kv**: Used for variable reference resolution
- **logging**: For operation logging and debugging
- **metrics**: For template operation metrics
- **jinja2**: External dependency for template engine

## Tools

### templates_render
```python
async def templates_render(ctx: RunContext[Any], template_name: str, variables: Optional[Dict[str, Any]], 
                          strict: bool) -> TemplatesOutput
```
Render a pre-loaded template with variable substitution and reference resolution.

**Key Features:**
- Automatic variable reference resolution from storage systems
- Strict mode for production (fails on undefined variables)
- Lenient mode for development (undefined variables become empty strings)
- Template existence validation

**Raises:**
- `ValueError`: If template not found or undefined variables in strict mode
- `RuntimeError`: For template rendering errors

### templates_save
```python
async def templates_save(ctx: RunContext[Any], template_name: str, template_content: str) -> TemplatesOutput
```
Save a new template to the filesystem and load it into memory for immediate use.

**Key Features:**
- Automatic template validation before saving
- Saves as .jinja file in templates directory
- Immediately loads template into memory cache
- Creates parent directories if needed

**Raises:**
- `ValueError`: If template content has syntax errors
- `RuntimeError`: If file operations fail or template loading fails

### templates_list
```python
async def templates_list(ctx: RunContext[Any]) -> TemplatesOutput
```
List all available templates with metadata including file paths and sizes.

**Key Features:**
- Returns sorted list of all loaded templates
- Includes file metadata (size, path)
- Shows both template name and filename

**Raises:**
- `RuntimeError`: If error accessing template information

### templates_validate
```python
async def templates_validate(ctx: RunContext[Any], template_content: str) -> TemplatesOutput
```
Validate template syntax and extract used variables without rendering.

**Key Features:**
- Jinja2 syntax validation
- Extracts undeclared variables used in template
- Returns validation result with variable list
- No side effects (doesn't save or render)

**Raises:**
- `ValueError`: If template has syntax errors (with line numbers)
- `RuntimeError`: For validation operation errors

### templates_exec
```python
async def templates_exec(ctx: RunContext[Any], template_content: str, variables: Optional[Dict[str, Any]], 
                        strict: bool) -> TemplatesOutput
```
Execute template content directly without saving to filesystem (ad-hoc rendering).

**Key Features:**
- Template validation before execution
- Variable reference resolution
- Strict and lenient rendering modes
- Useful for one-time template rendering

**Raises:**
- `ValueError`: If template syntax errors or undefined variables in strict mode
- `RuntimeError`: For template execution errors

## Template Loading

Templates are automatically loaded from the configured directory on agent creation:

### Supported File Extensions
- `.jinja` - Primary extension
- `.j2` - Alternative extension

### Loading Behavior
- All template files are loaded into memory on startup
- Failed template loads are logged as warnings but don't stop initialization
- Templates are accessible by filename without extension
- Template cache is updated when new templates are saved

## Exceptions

| Exception Type | Scenarios |
|---------------|-----------|
| `ValueError` | - Template not found<br>- Template syntax errors<br>- Undefined variables in strict mode<br>- Template validation failures |
| `RuntimeError` | - File system operation errors<br>- Template loading failures<br>- Variable resolution errors |
| `ImportError` | - Jinja2 library not available |
| `TemplateSyntaxError` | - Jinja2 template compilation errors |
| `UndefinedError` | - Undefined variables in strict mode |

## Usage Examples

### Basic Template Rendering
```python
from agentoolkit.system.templates import create_templates_agent
from agentool.core.injector import get_injector

# Create and register the agent
agent = create_templates_agent()
injector = get_injector()

# First, save a template
result = await injector.run('templates', {
    "operation": "save",
    "template_name": "greeting",
    "template_content": "Hello {{ name }}! Welcome to {{ company }}."
})

# Render the template
result = await injector.run('templates', {
    "operation": "render",
    "template_name": "greeting",
    "variables": {
        "name": "Alice",
        "company": "Acme Corp"
    }
})
# Returns: "Hello Alice! Welcome to Acme Corp."
```

### Storage Reference Integration
```python
# Store data in storage_kv first
await injector.run('storage_kv', {
    "operation": "set",
    "key": "user_profile",
    "value": {"name": "Bob Smith", "role": "Developer"}
})

await injector.run('storage_kv', {
    "operation": "set", 
    "key": "company_info",
    "value": "Tech Innovations Inc."
})

# Create template with storage references
template_content = """
Welcome {{ user.name }}!
Role: {{ user.role }}
Company: {{ company }}
"""

result = await injector.run('templates', {
    "operation": "save",
    "template_name": "user_welcome",
    "template_content": template_content
})

# Render with storage references
result = await injector.run('templates', {
    "operation": "render",
    "template_name": "user_welcome",
    "variables": {
        "user": "!ref:storage_kv:user_profile",
        "company": "!ref:storage_kv:company_info"
    }
})
```

### File System Integration
```python
# Store template data in files
await injector.run('storage_fs', {
    "operation": "write",
    "path": "/data/header.html",
    "content": "<h1>My Application</h1>"
})

# Template using file content
email_template = """
{{ header_content }}
<p>Dear {{ recipient }},</p>
<p>{{ message }}</p>
<p>Best regards,<br>{{ sender }}</p>
"""

result = await injector.run('templates', {
    "operation": "exec",
    "template_content": email_template,
    "variables": {
        "header_content": "!ref:storage_fs:/data/header.html",
        "recipient": "John Doe",
        "message": "Thank you for your order!",
        "sender": "Customer Service"
    }
})
```

### Template Management
```python
# List all available templates
result = await injector.run('templates', {
    "operation": "list"
})

# Validate template before using
result = await injector.run('templates', {
    "operation": "validate",
    "template_content": "Hello {{ name }}! Your balance is ${{ balance | round(2) }}."
})

# Check validation result
if result["data"]["valid"]:
    print(f"Template uses variables: {result['data']['variables']}")
```

### Advanced Template Features
```python
# Complex template with Jinja2 features
advanced_template = """
{% set total_items = items | length %}
<h2>Order Summary ({{ total_items }} items)</h2>
<ul>
{% for item in items %}
    <li>{{ item.name }} - ${{ item.price | round(2) }}
    {% if item.discount %}
        <span class="discount">{{ item.discount }}% off!</span>
    {% endif %}
    </li>
{% endfor %}
</ul>
<p><strong>Total: ${{ items | sum(attribute='price') | round(2) }}</strong></p>
"""

# Execute with complex data
result = await injector.run('templates', {
    "operation": "exec",
    "template_content": advanced_template,
    "variables": {
        "items": [
            {"name": "Widget A", "price": 19.99, "discount": 10},
            {"name": "Widget B", "price": 29.99, "discount": None},
            {"name": "Widget C", "price": 9.99, "discount": 5}
        ]
    }
})
```

### Strict vs Lenient Mode
```python
template_with_undefined = "Hello {{ name }}! Your score is {{ score }}."

# Lenient mode (default) - undefined variables become empty strings
result = await injector.run('templates', {
    "operation": "exec",
    "template_content": template_with_undefined,
    "variables": {"name": "Alice"},  # score is missing
    "strict": False
})
# Returns: "Hello Alice! Your score is ."

# Strict mode - fails on undefined variables
try:
    result = await injector.run('templates', {
        "operation": "exec",
        "template_content": template_with_undefined,
        "variables": {"name": "Alice"},  # score is missing
        "strict": True
    })
except ValueError as e:
    print(f"Template error: {e}")
```

### Reference Resolution Error Handling
```python
# Template with storage references
result = await injector.run('templates', {
    "operation": "exec",
    "template_content": "User: {{ user_name }}, Config: {{ app_config }}",
    "variables": {
        "user_name": "!ref:storage_kv:nonexistent_user",  # Key doesn't exist
        "app_config": "!ref:storage_fs:/missing/file.txt"  # File doesn't exist
    }
})
# Returns: "User: <undefined:!ref:storage_kv:nonexistent_user>, Config: <undefined:!ref:storage_fs:/missing/file.txt>"
```

## Testing

The test suite is located at `tests/agentoolkit/test_templates.py`. Tests cover:
- Template loading and caching
- All template operations (render, save, list, validate, exec)
- Variable reference resolution from storage systems
- Strict and lenient rendering modes
- Template syntax validation
- Error handling for missing templates and undefined variables
- File system operations and directory creation
- Integration with logging and metrics

To run tests:
```bash
pytest tests/agentoolkit/test_templates.py -v
```

## Notes

- Templates are loaded into memory on agent creation for fast rendering
- Template files should use `.jinja` or `.j2` extensions to be auto-loaded
- Variable references are resolved asynchronously before template rendering
- Undefined variable handling differs between strict and lenient modes
- Template validation uses Jinja2's AST parsing to extract variable names
- All template operations are logged and tracked with metrics for observability
- The templates directory is created automatically if it doesn't exist
- Template names should not include file extensions when referencing
- Large template files may impact memory usage due to in-memory caching
- Storage reference resolution failures are handled gracefully with placeholder values