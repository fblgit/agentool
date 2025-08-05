# AgenTool Management AgenToolkit

> **Note**: For guidelines on creating new AgenToolkits, see [CRAFTING_AGENTOOLS.md](../CRAFTING_AGENTOOLS.md). For testing patterns and examples, refer to [tests/agentoolkit/test_agentool.py](../../../tests/agentoolkit/test_agentool.py).

## Overview

The AgenTool Management AgenToolkit provides comprehensive management, introspection, and creation capabilities for the AgenTool registry system. It serves as the central hub for discovering, analyzing, and managing all registered AgenTools within the system.

### Key Features
- Complete registry introspection (list, search, get details)
- Schema and routing configuration inspection
- Dependency analysis and validation
- Usage pattern analysis
- Documentation generation (Markdown, JSON, YAML)
- API specification generation
- Dependency graph visualization
- Export catalog functionality
- Usage guide generation

## Creation Method

```python
from agentoolkit.management.agentool import create_agentool_management_agent

# Create the agent
agent = create_agentool_management_agent()
```

The creation function returns a fully configured AgenTool with name `'agentool_mgmt'`.

## Input Schema

### AgenToolManagementInput

The input schema is a Pydantic model with the following fields:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `operation` | `Literal[...]` | Yes | - | The management operation to perform (see operations table) |
| `agentool_name` | `Optional[str]` | No | None | Name of the AgenTool to operate on |
| `tags` | `Optional[List[str]]` | No | None | Tags to filter by for search operations |
| `name_pattern` | `Optional[str]` | No | None | Name pattern to search for |
| `description` | `Optional[str]` | No | None | Description for new AgenTool |
| `version` | `Optional[str]` | No | "1.0.0" | Version for new AgenTool |
| `tools` | `Optional[List[str]]` | No | None | Tool function names or definitions |
| `config_data` | `Optional[Dict[str, Any]]` | No | None | Configuration data |
| `format` | `Optional[Literal['json', 'markdown', 'yaml']]` | No | 'json' | Output format |
| `include_examples` | `Optional[bool]` | No | True | Include examples in output |
| `include_schemas` | `Optional[bool]` | No | True | Include schemas in output |
| `detailed` | `Optional[bool]` | No | False | Include detailed information |
| `include_tools` | `Optional[bool]` | No | True | Include tool information |

## Operations Schema

The routing configuration maps operations to tool functions:

### Registry Introspection Operations
| Operation | Description | Required Parameters |
|-----------|-------------|-------------------|
| `list_agentools` | List all registered AgenTools | `detailed` |
| `get_agentool_info` | Get detailed info about specific AgenTool | `agentool_name`, `detailed` |
| `get_agentool_schema` | Get JSON schema for AgenTool input | `agentool_name` |
| `search_agentools` | Search AgenTools by tags/pattern | `tags`, `name_pattern` |
| `get_operations` | Get available operations for AgenTool | `agentool_name` |
| `get_tools_info` | Get information about AgenTool's tools | `agentool_name` |
| `get_routing_config` | Get routing configuration details | `agentool_name` |

### Analysis and Documentation Operations
| Operation | Description | Required Parameters |
|-----------|-------------|-------------------|
| `generate_dependency_graph` | Generate dependency graph | `include_tools` |
| `analyze_agentool_usage` | Analyze usage patterns | `agentool_name` |
| `validate_dependencies` | Validate all dependencies | `agentool_name` |
| `get_api_specification` | Generate OpenAPI-like spec | `format` |
| `generate_docs` | Generate documentation | `format`, `agentool_name` |
| `get_examples` | Get usage examples | `agentool_name` |
| `export_catalog` | Export full catalog | `format` |
| `generate_usage_guide` | Generate usage guide | `agentool_name` |

### Management Operations (Planned)
| Operation | Description | Status |
|-----------|-------------|--------|
| `create_agentool_config` | Create AgenTool configuration | Not implemented |
| `update_agentool_config` | Update AgenTool configuration | Not implemented |
| `register_agentool` | Register new AgenTool | Not implemented |
| `unregister_agentool` | Unregister AgenTool | Not implemented |
| `validate_agentool_config` | Validate configuration | Not implemented |

## Output Schema

### Management Output

All operations return a dictionary with:

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Whether the operation succeeded |
| `error` | `Optional[str]` | Error message if operation failed |
| `...` | `Any` | Operation-specific data fields |

### Operation-Specific Data Fields

- **list_agentools**: `count`, `agentools` (list of AgenTool info)
- **get_agentool_info**: `agentool` (detailed AgenTool information)
- **get_agentool_schema**: `agentool_name`, `schema` (JSON schema)
- **search_agentools**: `search_criteria`, `count`, `results`
- **get_operations**: `agentool_name`, `operations` (list of operation names)
- **get_tools_info**: `agentool_name`, `tools` (tool metadata)
- **get_routing_config**: `agentool_name`, `routing_config`, `total_operations`
- **generate_dependency_graph**: `generated_at`, `include_tools`, `dependency_graph`
- **analyze_agentool_usage**: `analysis` (usage statistics and relationships)
- **validate_dependencies**: `all_valid`, `validation_results`
- **get_api_specification**: `format`, `specification`
- **generate_docs**: `format`, `documentation`
- **get_examples**: `agentool_name`, `examples`
- **export_catalog**: `format`, `catalog`
- **generate_usage_guide**: `usage_guide`

## Dependencies

This AgenToolkit has no external dependencies on other AgenToolkits. It directly interfaces with the AgenToolRegistry system.

## Tools

### manage_agentool
```python
async def manage_agentool(ctx: RunContext[Any], operation: str, **kwargs) -> Dict[str, Any]
```
Main routing function that dispatches to specific operation handlers based on the operation type.

**Registry Introspection Functions:**

### list_agentools
```python
async def list_agentools(ctx: RunContext[Any], detailed: bool = False) -> Dict[str, Any]
```
List all registered AgenTools with basic or detailed information.

### get_agentool_info
```python
async def get_agentool_info(ctx: RunContext[Any], agentool_name: str, detailed: bool = True) -> Dict[str, Any]
```
Get detailed information about a specific AgenTool including operations, tools, and metadata.

### get_agentool_schema
```python
async def get_agentool_schema(ctx: RunContext[Any], agentool_name: str) -> Dict[str, Any]
```
Get the JSON schema for an AgenTool's input validation.

### search_agentools
```python
async def search_agentools(ctx: RunContext[Any], tags: Optional[List[str]] = None, 
                          name_pattern: Optional[str] = None) -> Dict[str, Any]
```
Search AgenTools by tags or name pattern with detailed results.

### get_operations
```python
async def get_operations(ctx: RunContext[Any], agentool_name: str) -> Dict[str, Any]
```
Get available operations for a specific AgenTool.

### get_tools_info
```python
async def get_tools_info(ctx: RunContext[Any], agentool_name: str) -> Dict[str, Any]
```
Get information about tools used by an AgenTool.

### get_routing_config
```python
async def get_routing_config(ctx: RunContext[Any], agentool_name: str) -> Dict[str, Any]
```
Get the routing configuration details for an AgenTool.

**Analysis and Documentation Functions:**

### generate_dependency_graph
```python
async def generate_dependency_graph(ctx: RunContext[Any], include_tools: bool = True) -> Dict[str, Any]
```
Generate a dependency graph showing relationships between AgenTools.

### analyze_agentool_usage
```python
async def analyze_agentool_usage(ctx: RunContext[Any], agentool_name: Optional[str] = None) -> Dict[str, Any]
```
Analyze usage patterns and relationships for AgenTools.

### validate_dependencies
```python
async def validate_dependencies(ctx: RunContext[Any], agentool_name: Optional[str] = None) -> Dict[str, Any]
```
Validate that all dependencies for AgenTools are available.

### get_api_specification
```python
async def get_api_specification(ctx: RunContext[Any], format: str = 'json') -> Dict[str, Any]
```
Generate an OpenAPI-like specification for all AgenTools.

### generate_docs
```python
async def generate_docs(ctx: RunContext[Any], format: str = 'markdown', 
                       agentool_name: Optional[str] = None) -> Dict[str, Any]
```
Generate documentation for AgenTools in various formats.

### get_examples
```python
async def get_examples(ctx: RunContext[Any], agentool_name: str) -> Dict[str, Any]
```
Get usage examples for a specific AgenTool.

### export_catalog
```python
async def export_catalog(ctx: RunContext[Any], format: str = 'json') -> Dict[str, Any]
```
Export the full AgenTool catalog.

### generate_usage_guide
```python
async def generate_usage_guide(ctx: RunContext[Any], agentool_name: str) -> Dict[str, Any]
```
Generate a comprehensive usage guide for a specific AgenTool.

## Exceptions

| Exception Type | Scenarios |
|---------------|-----------|
| `KeyError` | - AgenTool not found in registry<br>- Invalid operation parameters |
| `ValueError` | - Invalid format specifications<br>- Missing required parameters |
| `RuntimeError` | - Registry operation failures<br>- Documentation generation errors |
| `ImportError` | - Missing optional dependencies (PyYAML for YAML format) |

## Usage Examples

### Basic Registry Introspection
```python
from agentoolkit.management.agentool import create_agentool_management_agent
from agentool.core.injector import get_injector

# Create and register the agent
agent = create_agentool_management_agent()
injector = get_injector()

# List all AgenTools
result = await injector.run('agentool_mgmt', {
    "operation": "list_agentools",
    "detailed": True
})

# Get detailed info about a specific AgenTool
result = await injector.run('agentool_mgmt', {
    "operation": "get_agentool_info",
    "agentool_name": "storage_kv",
    "detailed": True
})

# Get the input schema for an AgenTool
result = await injector.run('agentool_mgmt', {
    "operation": "get_agentool_schema",
    "agentool_name": "metrics"
})
```

### Search and Discovery
```python
# Search by tags
result = await injector.run('agentool_mgmt', {
    "operation": "search_agentools",
    "tags": ["storage", "filesystem"]
})

# Search by name pattern
result = await injector.run('agentool_mgmt', {
    "operation": "search_agentools",
    "name_pattern": "storage_*"
})

# Get operations for an AgenTool
result = await injector.run('agentool_mgmt', {
    "operation": "get_operations",
    "agentool_name": "config"
})
```

### Dependency Analysis
```python
# Generate dependency graph
result = await injector.run('agentool_mgmt', {
    "operation": "generate_dependency_graph",
    "include_tools": True
})

# Analyze usage patterns
result = await injector.run('agentool_mgmt', {
    "operation": "analyze_agentool_usage",
    "agentool_name": "storage_kv"
})

# Validate all dependencies
result = await injector.run('agentool_mgmt', {
    "operation": "validate_dependencies"
})
```

### Documentation Generation
```python
# Generate markdown docs for all AgenTools
result = await injector.run('agentool_mgmt', {
    "operation": "generate_docs",
    "format": "markdown"
})

# Generate docs for specific AgenTool
result = await injector.run('agentool_mgmt', {
    "operation": "generate_docs",
    "format": "markdown",
    "agentool_name": "logging"
})

# Get API specification
result = await injector.run('agentool_mgmt', {
    "operation": "get_api_specification",
    "format": "json"
})
```

### Export and Usage Guides
```python
# Export full catalog
result = await injector.run('agentool_mgmt', {
    "operation": "export_catalog",
    "format": "yaml"
})

# Generate usage guide
result = await injector.run('agentool_mgmt', {
    "operation": "generate_usage_guide",
    "agentool_name": "templates"
})

# Get examples for an AgenTool
result = await injector.run('agentool_mgmt', {
    "operation": "get_examples",
    "agentool_name": "crypto"
})
```

### Advanced Analysis
```python
# Get routing configuration details
result = await injector.run('agentool_mgmt', {
    "operation": "get_routing_config",
    "agentool_name": "scheduler"
})

# Get detailed tool information
result = await injector.run('agentool_mgmt', {
    "operation": "get_tools_info",
    "agentool_name": "queue"
})
```

## Testing

The test suite is located at `tests/agentoolkit/test_agentool.py`. Tests cover:
- All registry introspection operations
- Search functionality with tags and patterns
- Dependency analysis and validation
- Documentation generation in multiple formats
- Error handling for non-existent AgenTools
- Schema validation and export functionality

To run tests:
```bash
pytest tests/agentoolkit/test_agentool.py -v
```

## Notes

- This toolkit provides read-only access to the registry by design
- Future versions will include dynamic AgenTool creation and management
- Documentation generation supports multiple output formats (JSON, Markdown, YAML)
- Dependency validation helps ensure system integrity
- The toolkit serves as the foundation for AgenTool ecosystem management
- Search functionality supports both exact matches and pattern-based queries
- Generated documentation includes schema information, examples, and usage patterns
- API specification generation facilitates integration with external systems