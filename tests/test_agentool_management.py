"""
Tests for the AgenTool Management Toolkit.

This module tests the comprehensive management and introspection capabilities
of the AgenTool registry system.
"""

import pytest
import json
import asyncio
from typing import Any, Dict, List
from pydantic import BaseModel, Field
from pydantic_ai import RunContext

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agentool import create_agentool, register_agentool_models, RoutingConfig
from agentool.core.registry import AgenToolRegistry, AgenToolConfig
from agentool.core.injector import get_injector
from agentoolkit.management.agentool import create_agentool_management_agent


class TestAgenToolManagement:
    """Test suite for AgenTool Management functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        # Clear registry and register models
        AgenToolRegistry.clear()
        register_agentool_models()
        
        # Clear injector
        injector = get_injector()
        injector.clear()
        
        # Create some test AgenTools for testing
        self._create_test_agentools()
        
        # Create the management agent
        self.mgmt_agent = create_agentool_management_agent()
        injector.register('agentool_mgmt', self.mgmt_agent)
    
    def _create_test_agentools(self):
        """Create test AgenTools for testing management operations."""
        
        # Get injector
        injector = get_injector()
        
        # Test AgenTool 1: Simple calculator
        class CalcInput(BaseModel):
            operation: str
            a: float
            b: float
        
        async def add_tool(ctx: RunContext[Any], a: float, b: float) -> dict:
            """Add two numbers."""
            return {"result": a + b, "operation": "addition"}
        
        async def multiply_tool(ctx: RunContext[Any], a: float, b: float) -> dict:
            """Multiply two numbers."""
            return {"result": a * b, "operation": "multiplication"}
        
        calc_routing = RoutingConfig(
            operation_field='operation',
            operation_map={
                'add': ('add_tool', lambda x: {'a': x.a, 'b': x.b}),
                'multiply': ('multiply_tool', lambda x: {'a': x.a, 'b': x.b})
            }
        )
        
        calc_agent = create_agentool(
            name='calculator',
            input_schema=CalcInput,
            routing_config=calc_routing,
            tools=[add_tool, multiply_tool],
            description="Simple calculator for basic arithmetic operations",
            version="1.0.0",
            tags=["math", "calculator", "utility"],
            examples=[
                {
                    "title": "Addition example",
                    "input": {"operation": "add", "a": 5, "b": 3},
                    "output": {"result": 8, "operation": "addition"}
                }
            ]
        )
        
        # Test AgenTool 2: Text processor with dependencies
        class TextInput(BaseModel):
            operation: str
            text: str
        
        async def uppercase_tool(ctx: RunContext[Any], text: str) -> dict:
            """Convert text to uppercase."""
            return {"result": text.upper(), "original": text}
        
        async def lowercase_tool(ctx: RunContext[Any], text: str) -> dict:
            """Convert text to lowercase."""
            return {"result": text.lower(), "original": text}
        
        text_routing = RoutingConfig(
            operation_field='operation',
            operation_map={
                'upper': ('uppercase_tool', lambda x: {'text': x.text}),
                'lower': ('lowercase_tool', lambda x: {'text': x.text})
            }
        )
        
        text_agent = create_agentool(
            name='text_processor',
            input_schema=TextInput,
            routing_config=text_routing,
            tools=[uppercase_tool, lowercase_tool],
            description="Text processing utilities",
            version="2.1.0",
            tags=["text", "processing", "utility"],
            dependencies=["calculator"]  # Depends on calculator for some reason
        )
        
        # Store references
        self.calc_agent = calc_agent
        self.text_agent = text_agent
    
    def test_list_agentools_basic(self):
        """Test listing AgenTools with basic information."""
        async def run_test():
            injector = get_injector()
            
            # Test basic listing
            result = await injector.run('agentool_mgmt', {
                "operation": "list_agentools",
                "detailed": False
            })
            
            output = json.loads(result.output)
            
            # Print the basic list for visualization
            print("\n" + "="*80)
            print("BASIC AGENTOOL LIST:")
            print("="*80)
            print(json.dumps(output, indent=2))
            print("="*80 + "\n")
            
            assert output["success"] is True
            assert output["count"] >= 2  # At least our test agents
            
            # Check that calculator and text_processor are in the list
            agent_names = [agent["name"] for agent in output["agentools"]]
            assert "calculator" in agent_names
            assert "text_processor" in agent_names
            
            # Verify basic info structure
            calc_info = next(a for a in output["agentools"] if a["name"] == "calculator")
            assert calc_info["version"] == "1.0.0"
            assert calc_info["description"] == "Simple calculator for basic arithmetic operations"
            assert calc_info["operations_count"] == 2
            assert calc_info["tools_count"] == 2
            assert "math" in calc_info["tags"]
        
        asyncio.run(run_test())
    
    def test_list_agentools_detailed(self):
        """Test listing AgenTools with detailed information."""
        async def run_test():
            injector = get_injector()
            
            result = await injector.run('agentool_mgmt', {
                "operation": "list_agentools",
                "detailed": True
            })
            
            output = json.loads(result.output)
            
            # Print the detailed list for visualization
            print("\n" + "="*80)
            print("DETAILED AGENTOOL LIST:")
            print("="*80)
            print(json.dumps(output, indent=2))
            print("="*80 + "\n")
            
            assert output["success"] is True
            
            # Find calculator in detailed list
            calc_info = next(a for a in output["agentools"] if a["name"] == "calculator")
            
            # Check detailed fields
            assert "operations" in calc_info
            assert "tools" in calc_info
            assert "input_schema" in calc_info
            assert "created_at" in calc_info
            assert "updated_at" in calc_info
            
            # Verify operations
            assert "add" in calc_info["operations"]
            assert "multiply" in calc_info["operations"]
            
            # Verify tools structure
            assert len(calc_info["tools"]) == 2
            add_tool = next(t for t in calc_info["tools"] if t["name"] == "add_tool")
            assert add_tool["description"] == "Add two numbers."
            assert add_tool["async"] is True
            assert "a" in add_tool["params"]
            assert "b" in add_tool["params"]
        
        asyncio.run(run_test())
    
    def test_get_agentool_info(self):
        """Test getting detailed information about a specific AgenTool."""
        async def run_test():
            injector = get_injector()
            
            result = await injector.run('agentool_mgmt', {
                "operation": "get_agentool_info",
                "agentool_name": "calculator",
                "detailed": True
            })
            
            output = json.loads(result.output)
            
            # Print the AgenTool info for visualization
            print("\n" + "="*80)
            print("AGENTOOL INFO FOR CALCULATOR:")
            print("="*80)
            print(json.dumps(output, indent=2))
            print("="*80 + "\n")
            
            assert output["success"] is True
            
            agentool = output["agentool"]
            assert agentool["name"] == "calculator"
            assert agentool["version"] == "1.0.0"
            assert agentool["description"] == "Simple calculator for basic arithmetic operations"
            assert set(agentool["tags"]) == {"math", "calculator", "utility"}
            assert agentool["dependencies"] == []
            
            # Check operations
            assert set(agentool["operations"]) == {"add", "multiply"}
            assert agentool["operation_field"] == "operation"
            
            # Check tools
            assert len(agentool["tools"]) == 2
            tool_names = {t["name"] for t in agentool["tools"]}
            assert tool_names == {"add_tool", "multiply_tool"}
            
            # Check examples
            assert len(agentool["examples"]) == 1
            example = agentool["examples"][0]
            assert example["title"] == "Addition example"
            assert example["input"]["operation"] == "add"
        
        asyncio.run(run_test())
    
    def test_get_agentool_info_not_found(self):
        """Test getting info for non-existent AgenTool."""
        async def run_test():
            injector = get_injector()
            
            result = await injector.run('agentool_mgmt', {
                "operation": "get_agentool_info",
                "agentool_name": "nonexistent"
            })
            
            output = json.loads(result.output)
            assert output["success"] is False
            assert "not found" in output["error"]
        
        asyncio.run(run_test())
    
    def test_get_agentool_schema(self):
        """Test getting JSON schema for an AgenTool."""
        async def run_test():
            injector = get_injector()
            
            result = await injector.run('agentool_mgmt', {
                "operation": "get_agentool_schema",
                "agentool_name": "calculator"
            })
            
            output = json.loads(result.output)
            
            # Print the JSON schema for visualization
            print("\n" + "="*80)
            print("JSON SCHEMA FOR CALCULATOR:")
            print("="*80)
            print(json.dumps(output, indent=2))
            print("="*80 + "\n")
            
            assert output["success"] is True
            assert output["agentool_name"] == "calculator"
            
            schema = output["schema"]
            assert schema["type"] == "object"
            assert "properties" in schema
            
            # Check required properties
            properties = schema["properties"]
            assert "operation" in properties
            assert "a" in properties
            assert "b" in properties
            
            # Check property types
            assert properties["operation"]["type"] == "string"
            assert properties["a"]["type"] == "number"
            assert properties["b"]["type"] == "number"
        
        asyncio.run(run_test())
    
    def test_search_agentools(self):
        """Test searching AgenTools by tags and name patterns."""
        async def run_test():
            injector = get_injector()
            
            # Search by tags
            result = await injector.run('agentool_mgmt', {
                "operation": "search_agentools",
                "tags": ["math"]
            })
            
            output = json.loads(result.output)
            assert output["success"] is True
            assert output["count"] == 1
            assert output["results"][0]["name"] == "calculator"
            
            # Search by name pattern
            result = await injector.run('agentool_mgmt', {
                "operation": "search_agentools",
                "name_pattern": "text"
            })
            
            output = json.loads(result.output)
            assert output["success"] is True
            assert output["count"] == 1
            assert output["results"][0]["name"] == "text_processor"
            
            # Search by both (should find utility tag in both)
            result = await injector.run('agentool_mgmt', {
                "operation": "search_agentools",
                "tags": ["utility"]
            })
            
            output = json.loads(result.output)
            assert output["success"] is True
            assert output["count"] == 2
        
        asyncio.run(run_test())
    
    def test_get_operations(self):
        """Test getting operations for an AgenTool."""
        async def run_test():
            injector = get_injector()
            
            result = await injector.run('agentool_mgmt', {
                "operation": "get_operations",
                "agentool_name": "calculator"
            })
            
            output = json.loads(result.output)
            
            # Print the operations for visualization
            print("\n" + "="*80)
            print("OPERATIONS FOR CALCULATOR:")
            print("="*80)
            print(json.dumps(output, indent=2))
            print("="*80 + "\n")
            
            assert output["success"] is True
            assert output["agentool_name"] == "calculator"
            
            operations = output["operations"]
            assert "add" in operations
            assert "multiply" in operations
            
            # Check operation details
            add_op = operations["add"]
            assert add_op["tool"] == "add_tool"
            assert add_op["description"] == "Add two numbers."
            assert set(add_op["parameters"]) == {"a", "b"}
        
        asyncio.run(run_test())
    
    def test_get_tools_info(self):
        """Test getting tools information for an AgenTool."""
        async def run_test():
            injector = get_injector()
            
            result = await injector.run('agentool_mgmt', {
                "operation": "get_tools_info",
                "agentool_name": "text_processor"
            })
            
            output = json.loads(result.output)
            
            # Print the tools info for visualization
            print("\n" + "="*80)
            print("TOOLS INFO FOR TEXT_PROCESSOR:")
            print("="*80)
            print(json.dumps(output, indent=2))
            print("="*80 + "\n")
            
            assert output["success"] is True
            assert output["agentool_name"] == "text_processor"
            
            tools = output["tools"]
            assert len(tools) == 2
            
            tool_names = {t["name"] for t in tools}
            assert tool_names == {"uppercase_tool", "lowercase_tool"}
            
            # Check tool details
            upper_tool = next(t for t in tools if t["name"] == "uppercase_tool")
            assert upper_tool["description"] == "Convert text to uppercase."
            assert upper_tool["is_async"] is True
            assert "text" in upper_tool["parameters"]
        
        asyncio.run(run_test())
    
    def test_generate_dependency_graph(self):
        """Test generating dependency graph."""
        async def run_test():
            injector = get_injector()
            
            result = await injector.run('agentool_mgmt', {
                "operation": "generate_dependency_graph",
                "include_tools": True
            })
            
            output = json.loads(result.output)
            assert output["success"] is True
            assert output["include_tools"] is True
            
            graph = output["dependency_graph"]
            assert "agentools" in graph
            assert "tools" in graph
            
            # Check AgenTool dependencies
            agentools = graph["agentools"]
            assert "calculator" in agentools
            assert "text_processor" in agentools
            
            # text_processor depends on calculator
            assert "calculator" in agentools["text_processor"]
            # calculator has no dependencies
            assert agentools["calculator"] == []
            
            # Check tool dependencies
            tools = graph["tools"]
            assert "calculator" in tools
            assert "text_processor" in tools
            
            # Check tool lists
            calc_tools = tools["calculator"]
            assert "add_tool" in calc_tools
            assert "multiply_tool" in calc_tools
        
        asyncio.run(run_test())
    
    def test_analyze_agentool_usage(self):
        """Test analyzing AgenTool usage patterns."""
        async def run_test():
            injector = get_injector()
            
            # Analyze specific AgenTool
            result = await injector.run('agentool_mgmt', {
                "operation": "analyze_agentool_usage",
                "agentool_name": "calculator"
            })
            
            output = json.loads(result.output)
            assert output["success"] is True
            
            analysis = output["analysis"]
            assert analysis["agentool_name"] == "calculator"
            assert analysis["dependencies"] == []
            assert "text_processor" in analysis["dependents"]  # text_processor depends on calculator
            assert analysis["operations_count"] == 2
            assert analysis["tools_count"] == 2
            assert analysis["is_leaf"] is False  # Has dependents
            assert analysis["is_root"] is True   # No dependencies
            
            # Analyze all AgenTools
            result = await injector.run('agentool_mgmt', {
                "operation": "analyze_agentool_usage"
            })
            
            output = json.loads(result.output)
            assert output["success"] is True
            
            analysis = output["analysis"]
            assert analysis["total_agentools"] >= 2
            assert "calculator" in analysis["root_agentools"]
            assert "text_processor" in analysis["leaf_agentools"]
        
        asyncio.run(run_test())
    
    def test_get_routing_config(self):
        """Test fetching routing configuration for an AgenTool."""
        async def run_test():
            injector = get_injector()
            
            # Get routing config for calculator
            result = await injector.run('agentool_mgmt', {
                "operation": "get_routing_config",
                "agentool_name": "calculator"
            })
            
            output = json.loads(result.output)
            
            # Print the routing config for visualization
            print("\n" + "="*60)
            print("ROUTING CONFIG FOR CALCULATOR:")
            print("="*60)
            print(json.dumps(output, indent=2))
            print("="*60 + "\n")
            
            assert output["success"] is True
            assert output["agentool_name"] == "calculator"
            assert output["total_operations"] == 2
            
            routing_config = output["routing_config"]
            assert routing_config["operation_field"] == "operation"
            
            # Check operations
            operations = routing_config["operations"]
            assert "add" in operations
            assert "multiply" in operations
            
            # Check add operation details
            add_op = operations["add"]
            assert add_op["tool_name"] == "add_tool"
            assert add_op["tool_description"] == "Add two numbers."
            assert add_op["is_async"] is True
            assert add_op["has_mapper"] is True
            assert "a" in add_op["tool_parameters"]
            assert "b" in add_op["tool_parameters"]
            
            # Check multiply operation details  
            multiply_op = operations["multiply"]
            assert multiply_op["tool_name"] == "multiply_tool"
            assert multiply_op["tool_description"] == "Multiply two numbers."
            assert multiply_op["is_async"] is True
            assert multiply_op["has_mapper"] is True
            
            # Test with non-existent AgenTool
            result = await injector.run('agentool_mgmt', {
                "operation": "get_routing_config",
                "agentool_name": "non_existent"
            })
            
            output = json.loads(result.output)
            assert output["success"] is False
            assert "not found" in output["error"]
        
        asyncio.run(run_test())
    
    def test_validate_dependencies(self):
        """Test validating AgenTool dependencies."""
        async def run_test():
            injector = get_injector()
            
            # Validate all dependencies
            result = await injector.run('agentool_mgmt', {
                "operation": "validate_dependencies"
            })
            
            output = json.loads(result.output)
            assert output["success"] is True
            assert output["all_valid"] is True
            
            results = output["validation_results"]
            
            # Find calculator validation
            calc_result = next(r for r in results if r["agentool_name"] == "calculator")
            assert calc_result["valid"] is True
            assert calc_result["dependencies"] == []
            assert calc_result["missing_dependencies"] == []
            
            # Find text_processor validation
            text_result = next(r for r in results if r["agentool_name"] == "text_processor")
            assert text_result["valid"] is True
            assert "calculator" in text_result["dependencies"]
            assert "calculator" in text_result["available_dependencies"]
            assert text_result["missing_dependencies"] == []
        
        asyncio.run(run_test())
    
    def test_get_api_specification(self):
        """Test generating API specification."""
        async def run_test():
            injector = get_injector()
            
            result = await injector.run('agentool_mgmt', {
                "operation": "get_api_specification",
                "format": "json"
            })
            
            output = json.loads(result.output)
            
            # Print the API specification for visualization
            print("\n" + "="*80)
            print("API SPECIFICATION:")
            print("="*80)
            print(json.dumps(output, indent=2))
            print("="*80 + "\n")
            
            assert output["success"] is True
            assert output["format"] == "json"
            
            spec = output["specification"]
            assert spec["openapi"] == "3.0.0"
            assert "info" in spec
            assert "paths" in spec
            
            # Check paths
            paths = spec["paths"]
            assert "/agentools/calculator" in paths
            assert "/agentools/text_processor" in paths
            
            # Check calculator endpoint
            calc_endpoint = paths["/agentools/calculator"]
            assert "post" in calc_endpoint
            post_spec = calc_endpoint["post"]
            assert "requestBody" in post_spec
            assert "responses" in post_spec
        
        asyncio.run(run_test())
    
    def test_generate_docs_markdown(self):
        """Test generating markdown documentation."""
        async def run_test():
            injector = get_injector()
            
            # Generate docs for specific AgenTool
            result = await injector.run('agentool_mgmt', {
                "operation": "generate_docs",
                "format": "markdown",
                "agentool_name": "calculator"
            })
            
            output = json.loads(result.output)
            assert output["success"] is True
            assert output["format"] == "markdown"
            
            docs = output["documentation"]
            assert "# calculator" in docs
            assert "Simple calculator for basic arithmetic operations" in docs
            assert "## Operations" in docs
            assert "**add**:" in docs
            assert "**multiply**:" in docs
            assert "## Input Schema" in docs
            assert "```json" in docs
        
        asyncio.run(run_test())
    
    def test_generate_docs_json(self):
        """Test generating JSON documentation."""
        async def run_test():
            injector = get_injector()
            
            result = await injector.run('agentool_mgmt', {
                "operation": "generate_docs",
                "format": "json",
                "agentool_name": "calculator"
            })
            
            output = json.loads(result.output)
            assert output["success"] is True
            assert output["format"] == "json"
            
            docs = output["documentation"]
            assert docs["name"] == "calculator"
            assert docs["version"] == "1.0.0"
            assert docs["description"] == "Simple calculator for basic arithmetic operations"
            assert "operations" in docs
            assert "tools" in docs
        
        asyncio.run(run_test())
    
    def test_get_examples(self):
        """Test getting examples for an AgenTool."""
        async def run_test():
            injector = get_injector()
            
            # Test with AgenTool that has examples
            result = await injector.run('agentool_mgmt', {
                "operation": "get_examples",
                "agentool_name": "calculator"
            })
            
            output = json.loads(result.output)
            
            # Print the examples for calculator
            print("\n" + "="*80)
            print("EXAMPLES FOR CALCULATOR:")
            print("="*80)
            print(json.dumps(output, indent=2))
            print("="*80 + "\n")
            
            assert output["success"] is True
            assert output["agentool_name"] == "calculator"
            
            examples = output["examples"]
            assert len(examples) == 1
            assert examples[0]["title"] == "Addition example"
            assert examples[0]["input"]["operation"] == "add"
            
            # Test with AgenTool that has no examples (should generate basic ones)
            result = await injector.run('agentool_mgmt', {
                "operation": "get_examples",
                "agentool_name": "text_processor"
            })
            
            output = json.loads(result.output)
            
            # Print the generated examples for text_processor
            print("\n" + "="*80)
            print("GENERATED EXAMPLES FOR TEXT_PROCESSOR:")
            print("="*80)
            print(json.dumps(output, indent=2))
            print("="*80 + "\n")
            
            assert output["success"] is True
            
            examples = output["examples"]
            assert len(examples) == 2  # Generated for upper and lower operations
            
            operations = {e["input"]["operation"] for e in examples}
            assert operations == {"upper", "lower"}
        
        asyncio.run(run_test())
    
    def test_export_catalog(self):
        """Test exporting the full AgenTool catalog."""
        async def run_test():
            injector = get_injector()
            
            result = await injector.run('agentool_mgmt', {
                "operation": "export_catalog",
                "format": "json"
            })
            
            output = json.loads(result.output)
            assert output["success"] is True
            assert output["format"] == "json"
            
            catalog = output["catalog"]
            assert "version" in catalog
            assert "generated_at" in catalog
            assert "total_agentools" in catalog
            assert "agentools" in catalog
            
            # Check that our test agents are in the catalog
            agent_names = {a["name"] for a in catalog["agentools"]}
            assert "calculator" in agent_names
            assert "text_processor" in agent_names
        
        asyncio.run(run_test())
    
    def test_generate_usage_guide(self):
        """Test generating usage guide for an AgenTool."""
        async def run_test():
            injector = get_injector()
            
            result = await injector.run('agentool_mgmt', {
                "operation": "generate_usage_guide",
                "agentool_name": "calculator"
            })
            
            output = json.loads(result.output)
            assert output["success"] is True
            
            guide = output["usage_guide"]
            assert guide["agentool_name"] == "calculator"
            assert guide["description"] == "Simple calculator for basic arithmetic operations"
            assert guide["version"] == "1.0.0"
            
            # Check getting started section
            getting_started = guide["getting_started"]
            assert "import_statement" in getting_started
            assert "usage_pattern" in getting_started
            assert "calculator" in getting_started["usage_pattern"]
            
            # Check available operations
            operations = guide["available_operations"]
            assert len(operations) == 2
            
            op_names = {op["operation"] for op in operations}
            assert op_names == {"add", "multiply"}
            
            # Check operation details
            add_op = next(op for op in operations if op["operation"] == "add")
            assert add_op["tool"] == "add_tool"
            assert add_op["description"] == "Add two numbers."
            assert set(add_op["parameters"]) == {"a", "b"}
            assert add_op["example_input"]["operation"] == "add"
        
        asyncio.run(run_test())
    
    def test_error_handling(self):
        """Test error handling for various scenarios."""
        async def run_test():
            injector = get_injector()
            
            # Test missing required parameter
            result = await injector.run('agentool_mgmt', {
                "operation": "get_agentool_info"
                # Missing agentool_name
            })
            
            # This will return a validation error as plain text, not JSON
            assert "agentool_name is required" in result.output
            
            # Test unknown operation - this will also be a validation error
            try:
                result = await injector.run('agentool_mgmt', {
                    "operation": "unknown_operation"
                })
                # If we get here, it's an error response
                assert "Unknown operation" in result.output or "Input should be" in result.output
            except Exception as e:
                # This might raise a validation exception
                assert "unknown_operation" in str(e) or "Input should be" in str(e)
            
            # Test not yet implemented operation
            result = await injector.run('agentool_mgmt', {
                "operation": "create_agentool_config"
            })
            
            output = json.loads(result.output)
            assert output["success"] is False
            assert "not yet implemented" in output["error"]
        
        asyncio.run(run_test())
    
    def test_integration_with_existing_agentools(self):
        """Test that the management toolkit works with the rest of the system."""
        async def run_test():
            injector = get_injector()
            
            # First, use the management agent to get calculator info (detailed)
            mgmt_result = await injector.run('agentool_mgmt', {
                "operation": "get_agentool_info",
                "agentool_name": "calculator",
                "detailed": True
            })
            
            mgmt_output = json.loads(mgmt_result.output)
            assert mgmt_output["success"] is True
            
            # Now use the actual calculator
            calc_result = await injector.run('calculator', {
                "operation": "add",
                "a": 10,
                "b": 5
            })
            
            calc_output = json.loads(calc_result.output)
            assert calc_output["result"] == 15
            assert calc_output["operation"] == "addition"
            
            # Verify the management agent correctly reported the calculator's operations
            agentool_info = mgmt_output["agentool"]
            assert "add" in agentool_info["operations"]
        
        asyncio.run(run_test())