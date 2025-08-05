"""
Tests for enhanced registry features.

This module tests the new discovery, introspection, and visualization
features added to the AgenTool registry.
"""

import pytest
import json
from datetime import datetime
from typing import Any, Dict, List
from pydantic import BaseModel, Field
from pydantic_ai import RunContext

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agentool import create_agentool, register_agentool_models
from src.agentool.core.registry import AgenToolRegistry, AgenToolConfig, RoutingConfig, ToolMetadata
from src.agentool.factory import extract_tool_metadata


class TestEnhancedRegistry:
    """Test suite for enhanced registry features."""
    
    def setup_method(self):
        """Clear registry before each test."""
        AgenToolRegistry.clear()
        register_agentool_models()
    
    def test_enhanced_config_fields(self):
        """Test that AgenToolConfig supports new metadata fields."""
        # Create a sample tool
        async def sample_tool(ctx: RunContext[Any], data: str) -> dict:
            """Process data and return result."""
            return {"processed": data}
        
        # Create tool metadata
        tool_meta = ToolMetadata(
            name="sample_tool",
            description="Process data and return result.",
            is_async=True,
            parameters=["data"],
            parameter_types={"data": "str"},
            return_type="dict"
        )
        
        # Create a test model
        class TestConfigModel(BaseModel):
            data: str
        
        # Create config with all new fields
        config = AgenToolConfig(
            input_schema=TestConfigModel,
            routing_config=RoutingConfig(operation_map={}),
            description="Test AgenTool",
            version="2.0.0",
            tags=["test", "sample"],
            tools_metadata=[tool_meta],
            dependencies=["pydantic", "asyncio"],
            examples=[{"input": {"test": "data"}, "output": {"result": "ok"}}]
        )
        
        # Verify all fields
        assert config.version == "2.0.0"
        assert config.tags == ["test", "sample"]
        assert len(config.tools_metadata) == 1
        assert config.tools_metadata[0].name == "sample_tool"
        assert config.dependencies == ["pydantic", "asyncio"]
        assert len(config.examples) == 1
        assert isinstance(config.created_at, datetime)
        assert isinstance(config.updated_at, datetime)
    
    def test_extract_tool_metadata(self):
        """Test tool metadata extraction."""
        # Test async tool with full annotations
        async def async_tool(ctx: RunContext[Any], name: str, count: int = 5) -> Dict[str, Any]:
            """Process name and count data.
            
            This is a multi-line docstring.
            """
            return {"name": name, "count": count}
        
        meta = extract_tool_metadata(async_tool)
        assert meta.name == "async_tool"
        assert meta.description == "Process name and count data."
        assert meta.is_async is True
        assert meta.parameters == ["name", "count"]
        assert "name" in meta.parameter_types
        assert "count" in meta.parameter_types
        assert meta.return_type is not None
        
        # Test sync tool without annotations
        def sync_tool(ctx, data):
            return data
        
        meta2 = extract_tool_metadata(sync_tool)
        assert meta2.name == "sync_tool"
        assert meta2.is_async is False
        assert meta2.parameters == ["data"]
        assert len(meta2.parameter_types) == 0  # No type annotations
    
    def test_list_detailed(self):
        """Test detailed listing of AgenTools."""
        # Create a sample AgenTool
        class TestInput(BaseModel):
            operation: str
            value: str
        
        async def test_tool(ctx: RunContext[Any], value: str) -> dict:
            """Test tool for processing values."""
            return {"result": value}
        
        routing = RoutingConfig(
            operation_map={
                'process': ('test_tool', lambda x: {'value': x.value})
            }
        )
        
        agent = create_agentool(
            name='test_detailed',
            input_schema=TestInput,
            routing_config=routing,
            tools=[test_tool],
            version="1.5.0",
            tags=["test", "detailed"],
            description="Test AgenTool for detailed listing"
        )
        
        # Get detailed list
        detailed = AgenToolRegistry.list_detailed()
        
        # Find our test agent
        test_agent = next((a for a in detailed if a['name'] == 'test_detailed'), None)
        assert test_agent is not None
        
        # Verify detailed information
        assert test_agent['version'] == "1.5.0"
        assert test_agent['tags'] == ["test", "detailed"]
        assert test_agent['description'] == "Test AgenTool for detailed listing"
        assert 'process' in test_agent['operations']
        assert len(test_agent['tools']) == 1
        assert test_agent['tools'][0]['name'] == 'test_tool'
        assert test_agent['tools'][0]['async'] is True
        assert 'input_schema' in test_agent
        assert 'created_at' in test_agent
        assert 'updated_at' in test_agent
    
    def test_search_functionality(self):
        """Test searching AgenTools by tags and name pattern."""
        # Create a simple model for testing
        class SearchTestModel(BaseModel):
            value: str
        
        # Create multiple AgenTools with different tags
        for i in range(3):
            config = AgenToolConfig(
                input_schema=SearchTestModel,
                routing_config=RoutingConfig(operation_map={}),
                tags=["storage"] if i == 0 else ["compute", "async"] if i == 1 else ["network"],
                description=f"Test tool {i}"
            )
            AgenToolRegistry.register(f"tool_{i}", config)
        
        # Also create one with 'storage' in the name
        config = AgenToolConfig(
            input_schema=SearchTestModel,
            routing_config=RoutingConfig(operation_map={}),
            tags=["database"],
            description="Storage manager"
        )
        AgenToolRegistry.register("storage_manager", config)
        
        # Search by tags
        storage_tools = AgenToolRegistry.search(tags=["storage"])
        assert "tool_0" in storage_tools
        assert "storage_manager" not in storage_tools  # Different tag
        
        compute_tools = AgenToolRegistry.search(tags=["compute"])
        assert "tool_1" in compute_tools
        assert len(compute_tools) == 1
        
        # Search by multiple tags (OR operation)
        multi_tag = AgenToolRegistry.search(tags=["storage", "network"])
        assert "tool_0" in multi_tag
        assert "tool_2" in multi_tag
        assert "tool_1" not in multi_tag
        
        # Search by name pattern
        storage_pattern = AgenToolRegistry.search(name_pattern="storage")
        assert "storage_manager" in storage_pattern
        assert "tool_0" not in storage_pattern
        
        # Search by both
        combined = AgenToolRegistry.search(tags=["database"], name_pattern="storage")
        assert "storage_manager" in combined
        assert len(combined) == 1
    
    def test_get_schema(self):
        """Test getting JSON schema for an AgenTool."""
        class SchemaTestInput(BaseModel):
            operation: str = Field(description="The operation to perform")
            data: Dict[str, Any] = Field(description="Input data")
            optional: str = Field(default="default", description="Optional field")
        
        config = AgenToolConfig(
            input_schema=SchemaTestInput,
            routing_config=RoutingConfig(operation_map={})
        )
        AgenToolRegistry.register("schema_test", config)
        
        # Get schema
        schema = AgenToolRegistry.get_schema("schema_test")
        assert schema is not None
        assert schema['type'] == 'object'
        assert 'properties' in schema
        assert 'operation' in schema['properties']
        assert 'data' in schema['properties']
        assert 'optional' in schema['properties']
        
        # Non-existent tool
        assert AgenToolRegistry.get_schema("non_existent") is None
    
    def test_get_tools_info(self):
        """Test getting tool information."""
        async def tool_a(ctx: RunContext[Any], x: int) -> int:
            """Double the input value."""
            return x * 2
        
        async def tool_b(ctx: RunContext[Any], name: str, active: bool = True) -> dict:
            """Process user information."""
            return {"name": name, "active": active}
        
        class TestInput(BaseModel):
            operation: str
            value: Any
        
        routing = RoutingConfig(
            operation_map={
                'double': ('tool_a', lambda x: {'x': x.value}),
                'process': ('tool_b', lambda x: {'name': x.value})
            }
        )
        
        agent = create_agentool(
            name='multi_tool',
            input_schema=TestInput,
            routing_config=routing,
            tools=[tool_a, tool_b]
        )
        
        # Get tools info
        tools_info = AgenToolRegistry.get_tools_info('multi_tool')
        assert tools_info is not None
        assert len(tools_info) == 2
        
        # Check tool_a info
        tool_a_info = next((t for t in tools_info if t['name'] == 'tool_a'), None)
        assert tool_a_info is not None
        assert tool_a_info['description'] == "Double the input value."
        assert tool_a_info['is_async'] is True
        assert tool_a_info['parameters'] == ['x']
        assert 'x' in tool_a_info['parameter_types']
        
        # Non-existent tool
        assert AgenToolRegistry.get_tools_info('non_existent') is None
    
    def test_get_operations(self):
        """Test getting operations information."""
        async def read_tool(ctx: RunContext[Any], key: str) -> str:
            """Read data by key."""
            return f"data_{key}"
        
        async def write_tool(ctx: RunContext[Any], key: str, value: str) -> bool:
            """Write data to key."""
            return True
        
        class OpInput(BaseModel):
            operation: str
            key: str
            value: str = None
        
        routing = RoutingConfig(
            operation_map={
                'read': ('read_tool', lambda x: {'key': x.key}),
                'write': ('write_tool', lambda x: {'key': x.key, 'value': x.value})
            }
        )
        
        agent = create_agentool(
            name='op_test',
            input_schema=OpInput,
            routing_config=routing,
            tools=[read_tool, write_tool]
        )
        
        # Get operations
        ops = AgenToolRegistry.get_operations('op_test')
        assert ops is not None
        assert 'read' in ops
        assert 'write' in ops
        
        assert ops['read']['tool'] == 'read_tool'
        assert ops['read']['description'] == 'Read data by key.'
        assert ops['read']['parameters'] == ['key']
        
        assert ops['write']['tool'] == 'write_tool'
        assert ops['write']['parameters'] == ['key', 'value']
    
    def test_export_catalog(self):
        """Test exporting the full catalog."""
        # Create a simple model for testing
        class CatalogTestInput(BaseModel):
            value: str
        
        # Create a few test AgenTools
        for i in range(2):
            config = AgenToolConfig(
                input_schema=CatalogTestInput,
                routing_config=RoutingConfig(operation_map={}),
                description=f"Test tool {i}",
                version=f"1.{i}.0"
            )
            AgenToolRegistry.register(f"catalog_test_{i}", config)
        
        # Export catalog
        catalog = AgenToolRegistry.export_catalog()
        
        assert catalog['version'] == "1.0.0"
        assert 'generated_at' in catalog
        assert catalog['total_agentools'] >= 2
        assert 'agentools' in catalog
        assert len(catalog['agentools']) >= 2
        
        # Check that our test tools are in the catalog
        names = [a['name'] for a in catalog['agentools']]
        assert 'catalog_test_0' in names
        assert 'catalog_test_1' in names
    
    def test_generate_markdown_docs(self):
        """Test markdown documentation generation."""
        # Create a comprehensive test AgenTool
        async def process_tool(ctx: RunContext[Any], data: str) -> dict:
            """Process input data."""
            return {"processed": data}
        
        class DocInput(BaseModel):
            operation: str
            data: str
        
        routing = RoutingConfig(
            operation_map={
                'process': ('process_tool', lambda x: {'data': x.data})
            }
        )
        
        agent = create_agentool(
            name='doc_test',
            input_schema=DocInput,
            routing_config=routing,
            tools=[process_tool],
            description="Documentation test AgenTool",
            version="1.0.0",
            tags=["test", "docs"],
            examples=[{
                "input": {"operation": "process", "data": "test"},
                "output": {"processed": "test"}
            }]
        )
        
        # Generate markdown
        docs = AgenToolRegistry.generate_markdown_docs()
        
        # Verify content
        assert "# AgenTool Registry Documentation" in docs
        assert "## doc_test" in docs
        assert "**Description:** Documentation test AgenTool" in docs
        assert "**Version:** 1.0.0" in docs
        assert "**Tags:** test, docs" in docs
        assert "### Operations" in docs
        assert "### Tools" in docs
        assert "process_tool (async)" in docs
        assert "### Input Schema" in docs
        assert "### Examples" in docs
        assert "```json" in docs
    
    def test_generate_dependency_graph(self):
        """Test dependency graph generation."""
        # Create a simple model for testing
        class DependencyTestModel(BaseModel):
            value: str
        
        # Create AgenTools with dependencies
        config1 = AgenToolConfig(
            input_schema=DependencyTestModel,
            routing_config=RoutingConfig(operation_map={}),
            dependencies=["pydantic", "asyncio"]
        )
        AgenToolRegistry.register("tool_a", config1)
        
        config2 = AgenToolConfig(
            input_schema=DependencyTestModel,
            routing_config=RoutingConfig(
                operation_map={
                    'use_a': ('tool_a', lambda x: {})  # References tool_a
                }
            ),
            dependencies=["pydantic"]
        )
        AgenToolRegistry.register("tool_b", config2)
        
        config3 = AgenToolConfig(
            input_schema=DependencyTestModel,
            routing_config=RoutingConfig(operation_map={}),
            dependencies=[]
        )
        AgenToolRegistry.register("tool_c", config3)
        
        # Generate graph
        graph = AgenToolRegistry.generate_dependency_graph()
        
        # Check structure
        assert 'agentools' in graph
        assert 'tools' in graph
        
        assert 'tool_a' in graph['agentools']
        assert 'tool_b' in graph['agentools']
        assert 'tool_c' in graph['agentools']
        
        # tool_a has only external dependencies
        assert graph['agentools']['tool_a'] == ["pydantic", "asyncio"]
        
        # tool_b depends on tool_a and pydantic
        assert "pydantic" in graph['agentools']['tool_b']
        assert "tool_a" in graph['agentools']['tool_b']
        
        # tool_c has no dependencies
        assert graph['agentools']['tool_c'] == []
    
    def test_generate_api_spec(self):
        """Test OpenAPI spec generation."""
        class APIInput(BaseModel):
            operation: str = Field(description="Operation to perform")
            value: int = Field(description="Input value")
        
        async def api_tool(ctx: RunContext[Any], value: int) -> dict:
            return {"result": value * 2}
        
        routing = RoutingConfig(
            operation_map={
                'double': ('api_tool', lambda x: {'value': x.value})
            }
        )
        
        agent = create_agentool(
            name='api_test',
            input_schema=APIInput,
            routing_config=routing,
            tools=[api_tool],
            description="API test AgenTool",
            tags=["api", "test"]
        )
        
        # Generate API spec
        spec = AgenToolRegistry.generate_api_spec()
        
        # Verify OpenAPI structure
        assert spec['openapi'] == '3.0.0'
        assert 'info' in spec
        assert spec['info']['title'] == 'AgenTool Registry API'
        assert 'paths' in spec
        
        # Check main endpoint
        assert '/agentools/api_test' in spec['paths']
        endpoint = spec['paths']['/agentools/api_test']['post']
        assert endpoint['summary'] == 'API test AgenTool'
        assert endpoint['operationId'] == 'execute_api_test'
        assert endpoint['tags'] == ['api', 'test']
        assert 'requestBody' in endpoint
        assert 'responses' in endpoint
        
        # Check operation-specific endpoint
        assert '/agentools/api_test/double' in spec['paths']
        op_endpoint = spec['paths']['/agentools/api_test/double']['post']
        assert op_endpoint['operationId'] == 'api_test_double'
        assert 'double' in op_endpoint['tags']
    
    def test_full_enhanced_agentool_creation(self):
        """Test creating an AgenTool with all enhanced features."""
        # Define comprehensive tools
        async def fetch_data(ctx: RunContext[Any], source: str, limit: int = 10) -> List[Dict[str, Any]]:
            """Fetch data from specified source with optional limit.
            
            Args:
                source: Data source identifier
                limit: Maximum number of records
                
            Returns:
                List of data records
            """
            return [{"id": i, "source": source} for i in range(limit)]
        
        async def process_data(ctx: RunContext[Any], data: List[Dict[str, Any]], operation: str) -> Dict[str, Any]:
            """Process data with specified operation."""
            return {
                "operation": operation,
                "record_count": len(data),
                "processed": True
            }
        
        # Create input schema
        class DataInput(BaseModel):
            """Input schema for data operations."""
            operation: str = Field(description="Operation type: fetch or process")
            source: str = Field(None, description="Data source for fetch")
            limit: int = Field(10, description="Fetch limit")
            data: List[Dict[str, Any]] = Field(None, description="Data for processing")
            process_op: str = Field("transform", description="Processing operation")
        
        # Create routing
        routing = RoutingConfig(
            operation_field='operation',
            operation_map={
                'fetch': ('fetch_data', lambda x: {'source': x.source, 'limit': x.limit}),
                'process': ('process_data', lambda x: {'data': x.data, 'operation': x.process_op})
            }
        )
        
        # Create AgenTool with all features
        agent = create_agentool(
            name='data_processor',
            input_schema=DataInput,
            routing_config=routing,
            tools=[fetch_data, process_data],
            system_prompt="Advanced data processing AgenTool with fetch and process capabilities.",
            description="Comprehensive data processing system with multiple operations",
            version="2.5.0",
            tags=["data", "processing", "async", "advanced"],
            dependencies=["pydantic", "asyncio", "typing"],
            examples=[
                {
                    "description": "Fetch data from API",
                    "input": {"operation": "fetch", "source": "api", "limit": 5},
                    "output": [{"id": 0, "source": "api"}, {"id": 1, "source": "api"}]
                },
                {
                    "description": "Process fetched data",
                    "input": {
                        "operation": "process",
                        "data": [{"id": 0}, {"id": 1}],
                        "process_op": "aggregate"
                    },
                    "output": {"operation": "aggregate", "record_count": 2, "processed": True}
                }
            ]
        )
        
        # Verify the AgenTool was created with all features
        config = AgenToolRegistry.get('data_processor')
        assert config is not None
        assert config.version == "2.5.0"
        assert len(config.tags) == 4
        assert len(config.tools_metadata) == 2
        assert len(config.examples) == 2
        
        # Verify tool metadata extraction worked
        fetch_meta = next((t for t in config.tools_metadata if t.name == 'fetch_data'), None)
        assert fetch_meta is not None
        assert fetch_meta.description.startswith("Fetch data from specified source")
        assert fetch_meta.is_async is True
        assert fetch_meta.parameters == ['source', 'limit']
        assert 'source' in fetch_meta.parameter_types
        assert 'limit' in fetch_meta.parameter_types
        
        # Test the full discovery flow
        detailed = AgenToolRegistry.list_detailed()
        processor = next((a for a in detailed if a['name'] == 'data_processor'), None)
        assert processor is not None
        
        # Search for it
        found = AgenToolRegistry.search(tags=['data', 'processing'])
        assert 'data_processor' in found
        
        found_pattern = AgenToolRegistry.search(name_pattern='processor')
        assert 'data_processor' in found_pattern
        
        # Get its operations
        ops = AgenToolRegistry.get_operations('data_processor')
        assert 'fetch' in ops
        assert 'process' in ops
        assert ops['fetch']['tool'] == 'fetch_data'


class TestRegistryCoverage:
    """Additional tests for registry coverage."""
    
    def setup_method(self):
        """Clear registry before each test."""
        AgenToolRegistry.clear()
        register_agentool_models()
    
    def test_routing_config_post_init(self):
        """Test RoutingConfig __post_init__ with None operation_map."""
        # Test the line 42-43 in registry.py
        config = RoutingConfig(operation_field='custom_op')
        assert config.operation_map == {}
        assert config.operation_field == 'custom_op'
    
    def test_list_names_empty_registry(self):
        """Test list_names with empty registry."""
        # Clear the registry
        AgenToolRegistry.clear()
        names = AgenToolRegistry.list_names()
        assert names == []
        assert len(names) == 0
    
    def test_get_none_result(self):
        """Test get() returning None for non-existent tool."""
        result = AgenToolRegistry.get('definitely_does_not_exist')
        assert result is None
    
    def test_search_no_matches(self):
        """Test search with no matching results."""
        # Create a test tool
        class TestModel(BaseModel):
            value: str
        
        config = AgenToolConfig(
            input_schema=TestModel,
            routing_config=RoutingConfig(operation_map={}),
            tags=["specific", "unique"],
            description="Test tool"
        )
        AgenToolRegistry.register("test_search", config)
        
        # Search with non-matching criteria
        results = AgenToolRegistry.search(tags=["nonexistent"])
        assert results == []
        
        results = AgenToolRegistry.search(name_pattern="xyz123")
        assert results == []
        
        # Search with both criteria non-matching
        results = AgenToolRegistry.search(tags=["nonexistent"], name_pattern="xyz123")
        assert results == []
    
    def test_get_schema_none(self):
        """Test get_schema returning None."""
        schema = AgenToolRegistry.get_schema('nonexistent_tool')
        assert schema is None
    
    def test_get_tools_info_none(self):
        """Test get_tools_info returning None."""
        info = AgenToolRegistry.get_tools_info('nonexistent_tool')
        assert info is None
    
    def test_get_operations_none(self):
        """Test get_operations returning None."""
        ops = AgenToolRegistry.get_operations('nonexistent_tool')
        assert ops is None
    
    def test_get_operations_without_tool_metadata(self):
        """Test get_operations when tool metadata is missing."""
        class TestModel(BaseModel):
            operation: str
            value: str
        
        # Create config without tool metadata
        config = AgenToolConfig(
            input_schema=TestModel,
            routing_config=RoutingConfig(
                operation_map={
                    'test': ('missing_tool', lambda x: {'value': x.value})
                }
            ),
            tools_metadata=[]  # No metadata for the tool
        )
        AgenToolRegistry.register('no_metadata', config)
        
        ops = AgenToolRegistry.get_operations('no_metadata')
        assert ops is not None
        assert 'test' in ops
        assert ops['test']['tool'] == 'missing_tool'
        assert ops['test']['description'] is None
        assert ops['test']['parameters'] == []
    
    def test_export_catalog_empty(self):
        """Test export_catalog with empty registry."""
        AgenToolRegistry.clear()
        catalog = AgenToolRegistry.export_catalog()
        
        assert catalog['version'] == '1.0.0'
        assert 'generated_at' in catalog
        assert catalog['total_agentools'] == 0
        assert catalog['agentools'] == []
    
    def test_generate_markdown_docs_comprehensive(self):
        """Test markdown generation with all possible fields."""
        # Create tools with different characteristics
        def sync_tool(ctx: RunContext[Any], data: str) -> str:
            """A synchronous tool."""
            return data
        
        async def async_tool_no_doc(ctx: RunContext[Any], value: int):
            return value * 2
        
        class ComplexModel(BaseModel):
            operation: str
            data: Dict[str, Any]
            optional: str = None
        
        # Create tool metadata manually to test all branches
        tool1 = ToolMetadata(
            name="sync_tool",
            description="A synchronous tool.",
            is_async=False,
            parameters=["data"],
            parameter_types={"data": "str"},
            return_type="str"
        )
        
        tool2 = ToolMetadata(
            name="async_tool_no_doc",
            description=None,  # No description
            is_async=True,
            parameters=[],  # No parameters
            parameter_types={},
            return_type=None  # No return type
        )
        
        # Config with examples
        config_with_examples = AgenToolConfig(
            input_schema=ComplexModel,
            routing_config=RoutingConfig(
                operation_map={
                    'sync': ('sync_tool', lambda x: {'data': str(x.data)}),
                    'async': ('async_tool_no_doc', lambda x: {'value': 1})
                }
            ),
            description="Complex tool with examples",
            version="3.0.0",
            tags=["complex", "example"],
            tools_metadata=[tool1, tool2],
            examples=[
                {
                    "description": "Example 1",
                    "input": {"operation": "sync", "data": {"key": "value"}},
                    "output": {"result": "processed"}
                },
                {
                    # Example without description
                    "input": {"operation": "async"},
                    # No output field
                }
            ]
        )
        
        # Config without description or tags
        config_minimal = AgenToolConfig(
            input_schema=ComplexModel,
            routing_config=RoutingConfig(
                operation_map={
                    'simple': ('unknown_tool', lambda x: {})
                }
            ),
            description=None,  # No description
            tags=[],  # No tags
            tools_metadata=[]  # No tool metadata
        )
        
        AgenToolRegistry.register('complex_tool', config_with_examples)
        AgenToolRegistry.register('minimal_tool', config_minimal)
        
        # Generate docs
        docs = AgenToolRegistry.generate_markdown_docs()
        
        # Verify all branches are covered
        assert "# AgenTool Registry Documentation" in docs
        assert "## complex_tool" in docs
        assert "## minimal_tool" in docs
        
        # Complex tool assertions
        assert "**Description:** Complex tool with examples" in docs
        assert "**Version:** 3.0.0" in docs
        assert "**Tags:** complex, example" in docs
        assert "### Operations" in docs
        assert "- **sync**: A synchronous tool." in docs
        assert "- **async**: Calls `async_tool_no_doc`" in docs  # No description case
        
        # Tool metadata assertions
        assert "### Tools" in docs
        assert "#### sync_tool (sync)" in docs
        assert "#### async_tool_no_doc (async)" in docs
        assert "- Parameters: data" in docs
        assert "- Returns: `str`" in docs
        
        # Examples assertions
        assert "### Examples" in docs
        assert "#### Example 1" in docs
        assert '"operation": "sync"' in docs
        
        # Minimal tool assertions (no description, no tags)
        assert "minimal_tool" in docs
        # Should not have description or tags sections for minimal_tool
    
    def test_generate_dependency_graph_no_dependencies(self):
        """Test dependency graph with tools having no dependencies."""
        class SimpleModel(BaseModel):
            value: str
        
        # Tool with empty dependencies
        config1 = AgenToolConfig(
            input_schema=SimpleModel,
            routing_config=RoutingConfig(operation_map={}),
            dependencies=[]
        )
        
        # Tool with self-reference (should be ignored)
        config2 = AgenToolConfig(
            input_schema=SimpleModel,
            routing_config=RoutingConfig(
                operation_map={
                    'self_ref': ('tool_b', lambda x: {})  # References itself
                }
            ),
            dependencies=[]
        )
        
        AgenToolRegistry.register('tool_a', config1)
        AgenToolRegistry.register('tool_b', config2)
        
        graph = AgenToolRegistry.generate_dependency_graph()
        
        assert 'agentools' in graph
        assert 'tools' in graph
        assert graph['agentools']['tool_a'] == []
        assert graph['agentools']['tool_b'] == []  # Self-reference is ignored
    
    def test_generate_api_spec_comprehensive(self):
        """Test API spec generation with various configurations."""
        class APIModel(BaseModel):
            operation: str
            data: str
        
        # Config without description
        config1 = AgenToolConfig(
            input_schema=APIModel,
            routing_config=RoutingConfig(
                operation_map={
                    'process': ('processor', lambda x: {'data': x.data})
                }
            ),
            description=None,  # No description
            tags=[]  # No tags
        )
        
        # Config with everything
        config2 = AgenToolConfig(
            input_schema=APIModel,
            routing_config=RoutingConfig(
                operation_map={
                    'analyze': ('analyzer', lambda x: {'data': x.data}),
                    'transform': ('transformer', lambda x: {'data': x.data})
                }
            ),
            description="Full featured API tool",
            tags=["api", "full"]
        )
        
        AgenToolRegistry.register('basic_api', config1)
        AgenToolRegistry.register('full_api', config2)
        
        spec = AgenToolRegistry.generate_api_spec()
        
        # Check basic API (no description case)
        assert '/agentools/basic_api' in spec['paths']
        basic_endpoint = spec['paths']['/agentools/basic_api']['post']
        assert basic_endpoint['summary'] == 'Execute basic_api AgenTool'  # Default summary
        assert basic_endpoint['tags'] == []  # Empty tags
        
        # Check operation endpoint
        assert '/agentools/basic_api/process' in spec['paths']
        process_endpoint = spec['paths']['/agentools/basic_api/process']['post']
        assert process_endpoint['tags'] == ['process']  # Only operation tag
        
        # Check full API
        assert '/agentools/full_api' in spec['paths']
        full_endpoint = spec['paths']['/agentools/full_api']['post']
        assert full_endpoint['summary'] == 'Full featured API tool'
        assert full_endpoint['tags'] == ['api', 'full']
        
        # Check multiple operations
        assert '/agentools/full_api/analyze' in spec['paths']
        assert '/agentools/full_api/transform' in spec['paths']
        
        analyze_endpoint = spec['paths']['/agentools/full_api/analyze']['post']
        assert 'analyze' in analyze_endpoint['tags']
        assert 'api' in analyze_endpoint['tags']
        assert 'full' in analyze_endpoint['tags']
    
    def test_list_detailed_with_empty_examples(self):
        """Test list_detailed when examples list is empty."""
        class TestModel(BaseModel):
            value: str
        
        config = AgenToolConfig(
            input_schema=TestModel,
            routing_config=RoutingConfig(operation_map={}),
            examples=[]  # Empty examples list
        )
        AgenToolRegistry.register('empty_examples', config)
        
        detailed = AgenToolRegistry.list_detailed()
        tool = next((t for t in detailed if t['name'] == 'empty_examples'), None)
        
        assert tool is not None
        # Should not have 'examples' key when list is empty
        assert 'examples' not in tool
    
    def test_search_case_insensitive(self):
        """Test that search is case insensitive for name patterns."""
        class TestModel(BaseModel):
            value: str
        
        config = AgenToolConfig(
            input_schema=TestModel,
            routing_config=RoutingConfig(operation_map={}),
            tags=["test"]
        )
        AgenToolRegistry.register('CamelCaseTool', config)
        
        # Search with different cases
        results = AgenToolRegistry.search(name_pattern='camel')
        assert 'CamelCaseTool' in results
        
        results = AgenToolRegistry.search(name_pattern='CAMEL')
        assert 'CamelCaseTool' in results
        
        results = AgenToolRegistry.search(name_pattern='case')
        assert 'CamelCaseTool' in results
    
    def test_dependency_graph_with_duplicate_deps(self):
        """Test dependency graph handles duplicates correctly."""
        class TestModel(BaseModel):
            value: str
        
        # Create tool_a first
        config_a = AgenToolConfig(
            input_schema=TestModel,
            routing_config=RoutingConfig(operation_map={}),
            dependencies=[]
        )
        AgenToolRegistry.register('dep_tool_a', config_a)
        
        # Tool with duplicate dependency entries
        config = AgenToolConfig(
            input_schema=TestModel,
            routing_config=RoutingConfig(
                operation_map={
                    'use_a': ('dep_tool_a', lambda x: {})  # References dep_tool_a
                }
            ),
            dependencies=['pydantic', 'dep_tool_a', 'pydantic']  # 'pydantic' appears twice
        )
        AgenToolRegistry.register('dup_deps', config)
        
        graph = AgenToolRegistry.generate_dependency_graph()
        
        assert 'agentools' in graph
        # Should not have duplicate 'dep_tool_a' even though it's in both dependencies and operation_map
        deps = graph['agentools']['dup_deps']
        assert deps.count('dep_tool_a') == 1
        assert deps.count('pydantic') == 2  # Original duplicates preserved from dependencies list
    
    def test_markdown_empty_registry(self):
        """Test markdown generation with empty registry."""
        AgenToolRegistry.clear()
        docs = AgenToolRegistry.generate_markdown_docs()
        
        assert "# AgenTool Registry Documentation" in docs
        assert "Total AgenTools: 0" in docs
        # Should not have any tool sections
        assert "##" not in docs.replace("# AgenTool", "")  # Ignore main header
    
    def test_markdown_with_json_in_examples(self):
        """Test markdown generation handles JSON in examples properly."""
        class TestModel(BaseModel):
            operation: str
            data: Dict[str, Any]
        
        config = AgenToolConfig(
            input_schema=TestModel,
            routing_config=RoutingConfig(operation_map={}),
            examples=[
                {
                    "input": {
                        "operation": "test",
                        "data": {"nested": {"deep": "value"}}
                    },
                    "output": {
                        "result": ["item1", "item2"],
                        "status": True
                    }
                }
            ]
        )
        AgenToolRegistry.register('json_examples', config)
        
        docs = AgenToolRegistry.generate_markdown_docs()
        
        # Verify JSON is properly formatted in markdown
        assert '```json' in docs
        assert '"nested": {' in docs
        assert '"deep": "value"' in docs
        # JSON arrays might be formatted with line breaks
        assert '"item1"' in docs and '"item2"' in docs
    
    def test_markdown_example_only_output(self):
        """Test markdown generation with example that only has output."""
        class TestModel(BaseModel):
            value: str
        
        config = AgenToolConfig(
            input_schema=TestModel,
            routing_config=RoutingConfig(operation_map={}),
            examples=[
                {
                    # Only output, no input
                    "output": {"result": "success", "code": 200}
                }
            ]
        )
        AgenToolRegistry.register('output_only_example', config)
        
        docs = AgenToolRegistry.generate_markdown_docs()
        
        # Should have output section but no input section
        assert "**Output:**" in docs
        assert "**Input:**" not in docs or docs.count("**Input:**") < docs.count("**Output:**")
        assert '"result": "success"' in docs
        assert '"code": 200' in docs


class TestRegistryVisualization:
    """Test visualization features separately."""
    
    def setup_method(self):
        """Setup test data."""
        AgenToolRegistry.clear()
        register_agentool_models()
        
        # Create a sample ecosystem
        from src.examples.agents.storage import create_storage_agent
        storage = create_storage_agent()
    
    def test_markdown_generation_with_real_data(self):
        """Test markdown generation with real AgenTool."""
        docs = AgenToolRegistry.generate_markdown_docs()
        
        # Should contain storage agent
        assert "## storage" in docs
        assert "A storage system supporting read, write, list, and delete operations" in docs
        assert "**Version:** 1.2.0" in docs
        assert "**Tags:** storage, database, async, json, crud" in docs
        
        # Should show operations
        assert "### Operations" in docs
        assert "- **read**:" in docs
        assert "- **write**:" in docs
        assert "- **list**:" in docs
        assert "- **delete**:" in docs
        
        # Should show examples
        assert "### Examples" in docs
        assert '"operation": "write"' in docs
        assert '"operation": "read"' in docs
    
    def test_api_spec_with_real_data(self):
        """Test API spec generation with real AgenTool."""
        spec = AgenToolRegistry.generate_api_spec()
        
        # Check storage endpoints
        assert '/agentools/storage' in spec['paths']
        assert '/agentools/storage/read' in spec['paths']
        assert '/agentools/storage/write' in spec['paths']
        
        # Verify tags
        storage_endpoint = spec['paths']['/agentools/storage']['post']
        assert set(storage_endpoint['tags']) == set(["storage", "database", "async", "json", "crud"])