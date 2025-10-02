"""
Tests for the markdown_generator AgenToolkit.

This test suite validates all markdown generation operations including document creation,
section addition, table of contents generation, syntax validation, template application,
content formatting, quality assessment, and export functionality following AgenTool patterns.
"""

import asyncio
import json
import os
import tempfile
import pytest
from pathlib import Path
from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry
from pydantic_ai import models
models.ALLOW_MODEL_REQUESTS = True


class TestMarkdownGeneratorAgent:
    """Test suite for markdown_generator AgenTool."""
    
    def setup_method(self):
        """Clear registry and injector before each test."""
        AgenToolRegistry.clear()
        get_injector().clear()
        
        # Create required dependency agents
        # LLM agent for content processing and formatting
        from agentoolkit.llm import create_llm_agent
        llm_agent = create_llm_agent()
        
        # Templates agent for document templates
        from agentoolkit.system.templates import create_templates_agent
        templates_agent = create_templates_agent(templates_dir="src/templates")
        
        # Storage FS agent for file operations
        from agentoolkit.storage.fs import create_storage_fs_agent
        fs_agent = create_storage_fs_agent()
        
        # Storage KV agent for document caching
        from agentoolkit.storage.kv import create_storage_kv_agent, _kv_storage, _kv_expiry
        _kv_storage.clear()
        _kv_expiry.clear()
        kv_agent = create_storage_kv_agent()
        
        # Logging agent for operation logging
        from agentoolkit.system.logging import create_logging_agent
        logging_agent = create_logging_agent()
        
        # Metrics agent for performance tracking
        from agentoolkit.observability.metrics import create_metrics_agent
        metrics_agent = create_metrics_agent()
        
        # Import and create the markdown_generator agent
        from agentoolkit.llm.markdown_generator import create_markdown_generator_agent, _documents
        # Clear global document storage
        _documents.clear()
        agent = create_markdown_generator_agent()
    
    def test_create_document_operation(self):
        """Test document creation from templates."""
        
        async def run_test():
            injector = get_injector()
            
            # Test API documentation template
            result = await injector.run('markdown_generator', {
                "operation": "create_document",
                "template_name": "api_docs",
                "metadata": {
                    "title": "REST API Documentation", 
                    "author": "API Team",
                    "description": "Comprehensive API reference",
                    "version": "1.2.0"
                },
                "include_toc": True
            })
            
            assert result.success is True
            assert result.message == "Successfully created api_docs document template"
            
            data = result.data
            assert "document_id" in data
            assert "markdown" in data
            assert "sections" in data
            assert "word_count" in data
            
            document_id = data["document_id"]
            markdown_content = data["markdown"]
            sections = data["sections"]
            
            # Verify document ID format
            assert document_id.startswith("doc_")
            assert len(document_id) == 12  # doc_ + 8 hex chars
            
            # Verify template content structure
            assert "# REST API Documentation" in markdown_content
            assert "## Table of Contents" in markdown_content
            assert "## Overview" in markdown_content
            assert "## Authentication" in markdown_content
            assert "## Endpoints" in markdown_content
            assert "## Examples" in markdown_content
            assert "## Error Handling" in markdown_content
            
            # Verify metadata insertion
            assert "API Team" in markdown_content
            
            # Verify sections extraction
            assert "overview" in [s.lower() for s in sections]
            assert "authentication" in [s.lower() for s in sections]
            assert "endpoints" in [s.lower() for s in sections]
            
            # Verify word count
            assert data["word_count"] > 0
            
            print("\n=== test_create_document_operation Output ===")
            print(f"Document ID: {document_id}")
            print(f"Sections: {sections}")
            print(f"Word count: {data['word_count']}")
            print("=" * 40)
            
            # Test trend analysis template
            result2 = await injector.run('markdown_generator', {
                "operation": "create_document",
                "template_name": "trend_analysis",
                "metadata": {
                    "title": "AI Technology Trends 2024",
                    "author": "Research Team",
                    "sources": ["Industry Report", "Academic Papers", "Market Data"]
                },
                "include_toc": False
            })
            
            assert result2.success is True
            markdown2 = result2.data["markdown"]
            assert "# AI Technology Trends 2024" in markdown2
            assert "## Executive Summary" in markdown2
            assert "## Key Trends" in markdown2
            assert "Table of Contents" not in markdown2  # TOC disabled
            
            # Test comparison report template
            result3 = await injector.run('markdown_generator', {
                "operation": "create_document",
                "template_name": "comparison_report",
                "metadata": {"title": "Framework Comparison"}
            })
            
            assert result3.success is True
            markdown3 = result3.data["markdown"]
            assert "## Comparison Matrix" in markdown3
            assert "| Feature |" in markdown3  # Table structure
            
            # Test technical guide template
            result4 = await injector.run('markdown_generator', {
                "operation": "create_document",
                "template_name": "technical_guide",
                "metadata": {"title": "Installation Guide"}
            })
            
            assert result4.success is True
            markdown4 = result4.data["markdown"]
            assert "## Prerequisites" in markdown4
            assert "## Installation" in markdown4
            assert "```bash" in markdown4  # Code blocks present
        
        asyncio.run(run_test())
    
    def test_add_section_operation(self):
        """Test adding sections to existing documents."""
        
        async def run_test():
            injector = get_injector()
            
            # First create a document
            create_result = await injector.run('markdown_generator', {
                "operation": "create_document",
                "template_name": "technical_guide",
                "metadata": {"title": "Setup Guide", "author": "DevOps Team"}
            })
            
            assert create_result.success is True
            document_id = create_result.data["document_id"]
            
            # Add a configuration section
            result = await injector.run('markdown_generator', {
                "operation": "add_section",
                "document_id": document_id,
                "section_title": "Configuration",
                "content": """
                The system requires several configuration files to operate properly.
                
                Main configuration file: config.yaml
                Environment variables: .env file
                Database settings: database.conf
                
                Key configuration options:
                - server_port: 8080
                - debug_mode: false
                - database_url: postgresql://localhost/myapp
                """
            })
            
            assert result.success is True
            assert result.message == "Successfully added section 'Configuration' to document"
            
            data = result.data
            assert "section_added" in data
            assert "position" in data
            assert "updated_toc" in data
            assert "word_count" in data
            
            assert data["section_added"] == "Configuration"
            assert isinstance(data["position"], int)
            assert isinstance(data["updated_toc"], bool)
            assert data["word_count"] > 0
            
            print("\n=== test_add_section_operation Output ===")
            print(f"Section: {data['section_added']}")
            print(f"Position: {data['position']}")
            print(f"TOC updated: {data['updated_toc']}")
            print(f"New word count: {data['word_count']}")
            print("=" * 40)
            
            # Add another section with code
            result2 = await injector.run('markdown_generator', {
                "operation": "add_section",
                "document_id": document_id,
                "section_title": "API Examples",
                "content": """
                Here are some example API calls:
                
                ```python
                import requests
                
                # Get user data
                response = requests.get('/api/users/123')
                user = response.json()
                
                # Create new user
                new_user = {'name': 'John', 'email': 'john@example.com'}
                response = requests.post('/api/users', json=new_user)
                ```
                
                The API supports standard HTTP methods and returns JSON responses.
                """
            })
            
            assert result2.success is True
            assert result2.data["section_added"] == "API Examples"
            assert result2.data["word_count"] > data["word_count"]  # Should increase
        
        asyncio.run(run_test())
    
    def test_generate_toc_operation(self):
        """Test table of contents generation."""
        
        async def run_test():
            injector = get_injector()
            
            # Create a document with multiple headings
            create_result = await injector.run('markdown_generator', {
                "operation": "create_document", 
                "template_name": "technical_guide",
                "metadata": {"title": "Complete Guide"},
                "include_toc": False  # Start without TOC
            })
            
            document_id = create_result.data["document_id"]
            
            # Add several sections with different heading levels
            sections = [
                ("Overview", "This is the overview section."),
                ("Getting Started", "Initial setup instructions."),
                ("Advanced Configuration", "Detailed configuration options."),
                ("Troubleshooting", "Common issues and solutions.")
            ]
            
            for title, content in sections:
                await injector.run('markdown_generator', {
                    "operation": "add_section",
                    "document_id": document_id,
                    "section_title": title,
                    "content": content
                })
            
            # Generate TOC
            result = await injector.run('markdown_generator', {
                "operation": "generate_toc",
                "document_id": document_id,
                "max_heading_level": 3
            })
            
            assert result.success is True
            assert result.message == "Successfully generated table of contents"
            
            data = result.data
            assert "toc" in data
            assert "max_heading_level" in data
            assert "entries_count" in data
            
            toc = data["toc"]
            max_level = data["max_heading_level"]
            entries_count = data["entries_count"]
            
            # Verify TOC structure
            assert max_level == 3
            assert entries_count > 0
            
            # Check TOC content
            toc_lines = toc.split('\n')
            assert any("overview" in line.lower() for line in toc_lines)
            assert any("getting-started" in line.lower() for line in toc_lines)
            assert any("advanced-configuration" in line.lower() for line in toc_lines)
            assert any("troubleshooting" in line.lower() for line in toc_lines)
            
            # Check markdown link format
            assert any("](#" in line for line in toc_lines)
            assert any("- [" in line for line in toc_lines)
            
            print("\n=== test_generate_toc_operation Output ===")
            print(f"Max heading level: {max_level}")
            print(f"Entries count: {entries_count}")
            print(f"TOC preview: {toc.split(chr(10))[0]}")
            print("=" * 40)
            
            # Test with different max level
            result2 = await injector.run('markdown_generator', {
                "operation": "generate_toc",
                "document_id": document_id,
                "max_heading_level": 2
            })
            
            assert result2.success is True
            assert result2.data["max_heading_level"] == 2
        
        asyncio.run(run_test())
    
    def test_validate_syntax_operation(self):
        """Test markdown syntax validation."""
        
        async def run_test():
            injector = get_injector()
            
            # Test valid markdown
            valid_markdown = """
            # Main Title
            
            ## Section 1
            
            This is a paragraph with **bold** and *italic* text.
            
            ### Subsection
            
            Here's a list:
            - Item 1
            - Item 2
            - Item 3
            
            ```python
            def hello():
                print("Hello World")
            ```
            
            [Link to example](https://example.com)
            """
            
            result = await injector.run('markdown_generator', {
                "operation": "validate_syntax",
                "content": valid_markdown,
                "validation_level": "basic"
            })
            
            assert result.success is True
            assert result.message.startswith("Markdown validation passed")
            
            data = result.data
            assert "valid" in data
            assert "errors" in data
            assert "warnings" in data
            assert "score" in data
            
            assert data["valid"] is True
            assert len(data["errors"]) == 0
            assert isinstance(data["score"], (int, float))
            assert data["score"] > 80  # Valid markdown should score highly
            
            print("\n=== test_validate_syntax_operation (Valid) Output ===")
            print(f"Valid: {data['valid']}")
            print(f"Score: {data['score']}")
            print(f"Errors: {len(data['errors'])}")
            print(f"Warnings: {len(data['warnings'])}")
            print("=" * 40)
            
            # Test markdown with syntax errors
            invalid_markdown = """
            # Title
            
            [Broken link](missing-closing
            
            ```python
            def broken_code():
                print("missing closing
            
            ## Another Section
            
            ### 
            
            More content here.
            """
            
            result2 = await injector.run('markdown_generator', {
                "operation": "validate_syntax",
                "content": invalid_markdown,
                "validation_level": "comprehensive"
            })
            
            assert result2.success is False
            assert "failed with" in result2.message
            
            data2 = result2.data
            assert data2["valid"] is False
            assert len(data2["errors"]) > 0
            assert data2["score"] < data["score"]  # Should score lower
            
            # Check specific errors
            error_types = [error["type"] for error in data2["errors"]]
            assert "syntax" in error_types or "code_block" in error_types
            
            print("\n=== test_validate_syntax_operation (Invalid) Output ===")
            print(f"Valid: {data2['valid']}")
            print(f"Score: {data2['score']}")
            print(f"Errors: {len(data2['errors'])}")
            print(f"Error types: {error_types}")
            print("=" * 40)
            
            # Test strict validation
            try:
                result3 = await injector.run('markdown_generator', {
                    "operation": "validate_syntax",
                    "content": invalid_markdown,
                    "validation_level": "strict"
                })
                # Should raise SyntaxError for strict validation
                assert False, "Expected SyntaxError for strict validation"
            except SyntaxError as e:
                assert "validation failed" in str(e)
        
        asyncio.run(run_test())
    
    def test_apply_template_operation(self):
        """Test template application to existing documents."""
        
        async def run_test():
            injector = get_injector()
            
            # Create a basic document
            create_result = await injector.run('markdown_generator', {
                "operation": "create_document",
                "template_name": "technical_guide",
                "metadata": {"title": "Original Guide", "author": "Author"}
            })
            
            document_id = create_result.data["document_id"]
            
            # Add some content
            await injector.run('markdown_generator', {
                "operation": "add_section",
                "document_id": document_id,
                "section_title": "Custom Section",
                "content": "This is custom content added to the document."
            })
            
            # Apply a different template
            result = await injector.run('markdown_generator', {
                "operation": "apply_template",
                "document_id": document_id,
                "template_name": "api_docs",
                "formatting_options": {
                    "heading_style": "atx",
                    "include_timestamps": True
                }
            })
            
            assert result.success is True
            assert result.message == "Successfully applied template 'api_docs' to document"
            
            data = result.data
            assert "document_id" in data
            assert "template_applied" in data
            assert "word_count" in data
            assert "sections_count" in data
            
            assert data["document_id"] == document_id
            assert data["template_applied"] == "api_docs"
            assert data["word_count"] > 0
            assert data["sections_count"] > 0
            
            print("\n=== test_apply_template_operation Output ===")
            print(f"Template applied: {data['template_applied']}")
            print(f"Word count: {data['word_count']}")
            print(f"Sections: {data['sections_count']}")
            print("=" * 40)
            
            # Test applying comparison report template
            result2 = await injector.run('markdown_generator', {
                "operation": "apply_template",
                "document_id": document_id,
                "template_name": "comparison_report",
                "formatting_options": {"table_style": "github"}
            })
            
            assert result2.success is True
            assert result2.data["template_applied"] == "comparison_report"
        
        asyncio.run(run_test())
    
    def test_format_content_operation(self):
        """Test content formatting to markdown."""
        
        async def run_test():
            injector = get_injector()
            
            # Test plain text formatting
            plain_text = """
            INSTALLATION GUIDE
            
            Getting Started
            First, make sure you have Python installed on your system.
            Python version 3.8 or higher is required.
            
            Installation Steps:
            Step 1 - Download the package
            Step 2 - Extract the files
            Step 3 - Run the installer
            
            Configuration
            Edit the config file with your settings.
            Set the database URL in the configuration.
            Save and restart the application.
            """
            
            result = await injector.run('markdown_generator', {
                "operation": "format_content",
                "content": plain_text,
                "formatting_options": {
                    "preserve_structure": True,
                    "add_formatting": True,
                    "heading_style": "atx",
                    "code_highlighting": True
                }
            })
            
            assert result.success is True
            assert result.message == "Successfully formatted content"
            
            data = result.data
            assert "formatted_content" in data
            assert "word_count" in data
            assert "sections_count" in data
            assert "formatting_applied" in data
            
            formatted = data["formatted_content"]
            word_count = data["word_count"]
            sections_count = data["sections_count"]
            formatting_applied = data["formatting_applied"]
            
            # Verify formatting improvements
            assert "# " in formatted or "## " in formatted  # Headings formatted
            assert word_count > 0
            assert sections_count >= 0
            assert isinstance(formatting_applied, list)
            
            # Should contain structured content
            assert "installation" in formatted.lower()
            assert "configuration" in formatted.lower()
            
            print("\n=== test_format_content_operation Output ===")
            print(f"Word count: {word_count}")
            print(f"Sections: {sections_count}")
            print(f"Formatting applied: {formatting_applied}")
            print("=" * 40)
            
            # Test HTML content formatting
            html_content = """
            <h1>Web Development Guide</h1>
            <p>This guide covers <strong>modern web development</strong> practices.</p>
            <h2>Frontend Technologies</h2>
            <ul>
                <li>HTML5 for structure</li>
                <li>CSS3 for styling</li>
                <li>JavaScript for interactivity</li>
            </ul>
            <pre><code>console.log('Hello World');</code></pre>
            """
            
            result2 = await injector.run('markdown_generator', {
                "operation": "format_content",
                "content": html_content,
                "formatting_options": {"preserve_structure": False}
            })
            
            assert result2.success is True
            formatted2 = result2.data["formatted_content"]
            
            # Should convert HTML to markdown
            assert "<h1>" not in formatted2  # HTML tags removed
            assert "<p>" not in formatted2
            assert "<ul>" not in formatted2
            assert "web development" in formatted2.lower()
        
        asyncio.run(run_test())
    
    def test_assess_quality_operation(self):
        """Test document quality assessment."""
        
        async def run_test():
            injector = get_injector()
            
            # Create a comprehensive document
            create_result = await injector.run('markdown_generator', {
                "operation": "create_document",
                "template_name": "technical_guide",
                "metadata": {
                    "title": "Comprehensive Development Guide",
                    "author": "Technical Writing Team"
                }
            })
            
            document_id = create_result.data["document_id"]
            
            # Add substantial content with good structure
            sections = [
                ("Getting Started", """
                This comprehensive guide will walk you through setting up and using our development platform.
                Whether you're a beginner or an experienced developer, you'll find valuable information here.
                The platform supports multiple programming languages and provides extensive APIs for integration.
                """),
                ("Installation Instructions", """
                Follow these detailed steps to install the development environment:
                
                1. Download the installer from our official website
                2. Run the installer with administrator privileges  
                3. Choose your preferred installation directory
                4. Configure the environment variables as needed
                5. Verify the installation by running the test command
                
                The installation process typically takes 5-10 minutes depending on your system.
                """),
                ("Configuration Guide", """
                Proper configuration is essential for optimal performance:
                
                ### Database Configuration
                Set up your database connection in the config file.
                
                ### Security Settings
                Configure authentication and authorization properly.
                
                ### Performance Tuning
                Adjust memory settings and caching options.
                """),
                ("Best Practices", """
                Follow these industry-standard best practices:
                
                - Always use version control for your projects
                - Write comprehensive tests for your code
                - Document your APIs and functions clearly
                - Use consistent coding standards across your team
                - Implement proper error handling and logging
                
                These practices will improve code quality and maintainability.
                """)
            ]
            
            for title, content in sections:
                await injector.run('markdown_generator', {
                    "operation": "add_section",
                    "document_id": document_id,
                    "section_title": title,
                    "content": content
                })
            
            # Assess quality
            result = await injector.run('markdown_generator', {
                "operation": "assess_quality",
                "document_id": document_id
            })
            
            assert result.success is True
            assert result.message == "Document quality assessment completed"
            
            data = result.data
            assert "overall_score" in data
            assert "metrics" in data
            assert "recommendations" in data
            assert "word_count" in data
            assert "section_count" in data
            
            overall_score = data["overall_score"]
            metrics = data["metrics"]
            recommendations = data["recommendations"]
            
            # Verify score range
            assert 0 <= overall_score <= 100
            
            # Verify metrics structure
            expected_metrics = ["completeness", "structure", "readability", "formatting"]
            for metric in expected_metrics:
                assert metric in metrics
                assert 0 <= metrics[metric] <= 100
            
            # Well-structured document should score reasonably well
            assert overall_score >= 60
            assert metrics["completeness"] >= 70  # Good content length
            assert metrics["structure"] >= 70     # Good heading structure
            
            # Verify recommendations
            assert isinstance(recommendations, list)
            
            print("\n=== test_assess_quality_operation Output ===")
            print(f"Overall score: {overall_score}")
            print(f"Completeness: {metrics['completeness']}")
            print(f"Structure: {metrics['structure']}")
            print(f"Readability: {metrics['readability']}")
            print(f"Formatting: {metrics['formatting']}")
            print(f"Recommendations: {recommendations}")
            print(f"Word count: {data['word_count']}")
            print(f"Section count: {data['section_count']}")
            print("=" * 40)
            
            # Test with minimal document
            create_result2 = await injector.run('markdown_generator', {
                "operation": "create_document",
                "template_name": "technical_guide",
                "metadata": {"title": "Minimal Guide"}
            })
            
            document_id2 = create_result2.data["document_id"]
            
            result2 = await injector.run('markdown_generator', {
                "operation": "assess_quality",
                "document_id": document_id2
            })
            
            assert result2.success is True
            overall_score2 = result2.data["overall_score"]
            
            # Minimal document should score lower
            assert overall_score2 < overall_score
        
        asyncio.run(run_test())
    
    def test_export_operation(self):
        """Test document export to filesystem."""
        
        async def run_test():
            injector = get_injector()
            
            # Create a document to export
            create_result = await injector.run('markdown_generator', {
                "operation": "create_document",
                "template_name": "api_docs",
                "metadata": {
                    "title": "API Reference",
                    "author": "Documentation Team",
                    "version": "2.1.0"
                }
            })
            
            document_id = create_result.data["document_id"]
            
            # Add some content
            await injector.run('markdown_generator', {
                "operation": "add_section",
                "document_id": document_id,
                "section_title": "Getting Started",
                "content": "Quick start guide for using our API."
            })
            
            # Create temporary directory for export
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = os.path.join(temp_dir, "api_reference.md")
                
                # Export document
                result = await injector.run('markdown_generator', {
                    "operation": "export",
                    "document_id": document_id,
                    "output_path": output_path
                })
                
                assert result.success is True
                assert result.message == f"Successfully exported document to {output_path}"
                
                data = result.data
                assert "document_id" in data
                assert "output_path" in data
                assert "file_size" in data
                assert "word_count" in data
                
                assert data["document_id"] == document_id
                assert data["output_path"] == output_path
                assert data["file_size"] > 0
                assert data["word_count"] > 0
                
                # Verify file was created
                assert os.path.exists(output_path)
                
                # Verify file content
                with open(output_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check metadata header
                assert "Document ID:" in content
                assert document_id in content
                assert "Template: api_docs" in content
                assert "Word Count:" in content
                
                # Check document content
                assert "# API Reference" in content
                assert "Documentation Team" in content
                assert "Getting Started" in content
                assert "quick start guide" in content.lower()
                
                print("\n=== test_export_operation Output ===")
                print(f"Document ID: {data['document_id']}")
                print(f"Output path: {data['output_path']}")
                print(f"File size: {data['file_size']} bytes")
                print(f"Word count: {data['word_count']}")
                print("=" * 40)
                
                # Test export to different path
                output_path2 = os.path.join(temp_dir, "docs", "exported_api.md")
                
                result2 = await injector.run('markdown_generator', {
                    "operation": "export",
                    "document_id": document_id,
                    "output_path": output_path2
                })
                
                assert result2.success is True
                assert os.path.exists(output_path2)
                assert os.path.exists(os.path.dirname(output_path2))  # Directory created
        
        asyncio.run(run_test())
    
    def test_input_validation(self):
        """Test input validation for required fields."""
        
        async def run_test():
            injector = get_injector()
            
            # Test missing template_name for create_document
            with pytest.raises(ValueError) as exc_info:
                await injector.run('markdown_generator', {
                    "operation": "create_document",
                    # Missing required template_name
                })
            assert "template_name is required" in str(exc_info.value)
            
            # Test invalid template_name
            with pytest.raises(ValueError) as exc_info:
                await injector.run('markdown_generator', {
                    "operation": "create_document",
                    "template_name": "invalid_template"
                })
            assert "template_name must be one of" in str(exc_info.value)
            
            # Test missing document_id for add_section
            with pytest.raises(ValueError) as exc_info:
                await injector.run('markdown_generator', {
                    "operation": "add_section",
                    "section_title": "Test Section",
                    "content": "Test content"
                    # Missing required document_id
                })
            assert "document_id is required" in str(exc_info.value)
            
            # Test missing section_title for add_section
            with pytest.raises(ValueError) as exc_info:
                await injector.run('markdown_generator', {
                    "operation": "add_section",
                    "document_id": "doc_12345678",
                    "content": "Test content"
                    # Missing required section_title
                })
            assert "section_title is required" in str(exc_info.value)
            
            # Test missing content for add_section
            with pytest.raises(ValueError) as exc_info:
                await injector.run('markdown_generator', {
                    "operation": "add_section",
                    "document_id": "doc_12345678",
                    "section_title": "Test Section"
                    # Missing required content
                })
            assert "content is required" in str(exc_info.value)
            
            # Test missing output_path for export
            with pytest.raises(ValueError) as exc_info:
                await injector.run('markdown_generator', {
                    "operation": "export",
                    "document_id": "doc_12345678"
                    # Missing required output_path
                })
            assert "output_path is required" in str(exc_info.value)
            
            print("\n=== test_input_validation Output ===")
            print("Input validation tests completed")
            print("=" * 40)
        
        asyncio.run(run_test())
    
    def test_document_not_found_errors(self):
        """Test error handling for non-existent documents."""
        
        async def run_test():
            injector = get_injector()
            
            # Test add_section with non-existent document
            try:
                result = await injector.run('markdown_generator', {
                    "operation": "add_section",
                    "document_id": "doc_nonexist",
                    "section_title": "Test Section",
                    "content": "Test content"
                })
                assert False, "Expected FileNotFoundError for non-existent document"
            except FileNotFoundError as e:
                assert "not found" in str(e)
            
            # Test generate_toc with non-existent document
            try:
                result = await injector.run('markdown_generator', {
                    "operation": "generate_toc",
                    "document_id": "doc_missing"
                })
                assert False, "Expected FileNotFoundError for non-existent document"
            except FileNotFoundError as e:
                assert "not found" in str(e)
            
            # Test assess_quality with non-existent document
            try:
                result = await injector.run('markdown_generator', {
                    "operation": "assess_quality",
                    "document_id": "doc_absent"
                })
                assert False, "Expected FileNotFoundError for non-existent document"
            except FileNotFoundError as e:
                assert "not found" in str(e)
            
            # Test export with non-existent document
            try:
                result = await injector.run('markdown_generator', {
                    "operation": "export",
                    "document_id": "doc_void",
                    "output_path": "/tmp/test.md"
                })
                assert False, "Expected FileNotFoundError for non-existent document"
            except FileNotFoundError as e:
                assert "not found" in str(e)
            
            print("\n=== test_document_not_found_errors Output ===")
            print("Document not found error handling tested")
            print("=" * 40)
        
        asyncio.run(run_test())
    
    def test_invalid_operation(self):
        """Test handling of invalid operations."""
        
        async def run_test():
            injector = get_injector()
            
            # Test invalid operation - should get validation error from Pydantic
            result = await injector.run('markdown_generator', {
                "operation": "invalid_operation",
                "template_name": "api_docs"
            })
            
            # Check if the result has an error in the output
            if hasattr(result, 'output'):
                try:
                    output = json.loads(result.output) if isinstance(result.output, str) else result.output
                except json.JSONDecodeError:
                    output = result.output if hasattr(result, 'output') else str(result)
                
                output_str = str(output).lower()
                assert any(term in output_str for term in ['validation', 'invalid', 'error', 'literal'])
            
            print("\n=== test_invalid_operation Output ===")
            print("Invalid operation handled correctly")
            print("=" * 40)
        
        asyncio.run(run_test())