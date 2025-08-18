"""
markdown_generator AgenTool - Generates comprehensive, well-structured markdown documents from research data with templates and formatting.

This AgenTool provides complete markdown document generation capabilities with support for multiple templates,
quality assessment, syntax validation, and structured content organization. It integrates with LLM for intelligent
content processing, templates for document structure, and storage_fs for document persistence.

Key Features:
- Document Creation: Create structured documents from templates (API docs, trend analysis, comparison reports, technical guides)
- Content Management: Add sections, generate table of contents, and format content with proper markdown syntax
- Quality Assessment: Evaluate document completeness, structure, readability, and formatting quality
- Syntax Validation: Comprehensive markdown validation with error reporting and compatibility checking
- Template Support: Apply and manage document templates with variable substitution
- Export Capabilities: Save documents to filesystem with proper formatting and metadata

Usage Example:
    >>> from agentoolkit.markdown_generator import create_markdown_generator_agent
    >>> from agentool.core.injector import get_injector
    >>> 
    >>> # Create and register the agent
    >>> agent = create_markdown_generator_agent()
    >>> 
    >>> # Use through injector
    >>> injector = get_injector()
    >>> result = await injector.run('markdown_generator', {
    ...     "operation": "create_document",
    ...     "template_name": "api_docs",
    ...     "metadata": {"title": "My API", "author": "Developer"}
    ... })
"""

import json
import re
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import RunContext

import markdown
from markdown.extensions import codehilite, tables, toc

from agentool import create_agentool
from agentool.base import BaseOperationInput
from agentool.core.registry import RoutingConfig
from agentool.core.injector import get_injector


class MarkdownGeneratorInput(BaseOperationInput):
    """Input schema for markdown_generator operations.
    
    IMPORTANT: Every field here must map to function parameters via routing.
    Field names should be descriptive and match their usage in functions.
    """
    operation: Literal[
        'create_document', 'add_section', 'generate_toc', 'validate_syntax',
        'apply_template', 'format_content', 'assess_quality', 'export'
    ] = Field(
        description="The operation to perform"
    )
    
    # Fields for create_document
    template_name: Optional[str] = Field(
        None,
        description="Name of document template to use (api_docs, trend_analysis, comparison_report, technical_guide)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Document metadata including title, author, sources, and creation date"
    )
    include_toc: Optional[bool] = Field(
        None,
        description="Whether to include a table of contents"
    )
    
    # Fields for add_section
    document_id: Optional[str] = Field(
        None,
        description="Unique identifier for the document being worked on"
    )
    section_title: Optional[str] = Field(
        None,
        description="Title for a section being added"
    )
    content: Optional[str] = Field(
        None,
        description="Raw content or markdown text to process"
    )
    
    # Fields for generate_toc
    max_heading_level: Optional[int] = Field(
        None,
        description="Maximum heading level for TOC generation (1-6)"
    )
    
    # Fields for validate_syntax
    validation_level: Optional[Literal['basic', 'strict', 'comprehensive']] = Field(
        None,
        description="Level of markdown syntax validation to perform"
    )
    
    # Fields for format_content and apply_template
    formatting_options: Optional[Dict[str, Any]] = Field(
        None,
        description="Formatting preferences including heading styles, code block languages, and table formats"
    )
    
    # Fields for export
    output_path: Optional[str] = Field(
        None,
        description="File path where the final document should be saved"
    )
    
    @field_validator('template_name')
    @classmethod
    def validate_template_name(cls, v, info):
        """Validate template_name for create_document and apply_template operations."""
        operation = info.data.get('operation')
        if operation in ['create_document', 'apply_template'] and not v:
            raise ValueError(f"template_name is required for {operation}")
        if v and v not in ['api_docs', 'trend_analysis', 'comparison_report', 'technical_guide']:
            raise ValueError("template_name must be one of: api_docs, trend_analysis, comparison_report, technical_guide")
        return v
    
    @field_validator('document_id')
    @classmethod
    def validate_document_id(cls, v, info):
        """Validate document_id for operations that require it."""
        operation = info.data.get('operation')
        if operation in ['add_section', 'assess_quality'] and not v:
            raise ValueError(f"document_id is required for {operation}")
        return v
    
    @field_validator('section_title')
    @classmethod
    def validate_section_title(cls, v, info):
        """Validate section_title for add_section operation."""
        operation = info.data.get('operation')
        if operation == 'add_section' and not v:
            raise ValueError("section_title is required for add_section")
        return v
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v, info):
        """Validate content for operations that require it."""
        operation = info.data.get('operation')
        if operation in ['add_section', 'validate_syntax', 'format_content'] and not v:
            raise ValueError(f"content is required for {operation}")
        return v
    
    @field_validator('output_path')
    @classmethod
    def validate_output_path(cls, v, info):
        """Validate output_path for export operation."""
        operation = info.data.get('operation')
        if operation == 'export' and not v:
            raise ValueError("output_path is required for export")
        return v


class MarkdownGeneratorOutput(BaseModel):
    """Output schema for markdown_generator operations."""
    success: bool = Field(description="Whether the operation succeeded")
    message: str = Field(description="Human-readable result message")
    data: Optional[Dict[str, Any]] = Field(None, description="Operation-specific return data containing generated markdown, validation results, quality scores, or document metadata")


# Global document storage
_documents: Dict[str, Dict[str, Any]] = {}


async def markdown_generator_create_document(
    ctx: RunContext[Any],
    template_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    include_toc: Optional[bool] = None
) -> MarkdownGeneratorOutput:
    """
    Create a new markdown document from a template.
    
    This function generates a structured markdown document using predefined templates
    for different document types including API documentation, trend analysis, comparison
    reports, and technical guides.
    
    Args:
        ctx: Runtime context provided by the framework
        template_name: Template to use (api_docs, trend_analysis, comparison_report, technical_guide)
        metadata: Document metadata including title, author, sources, and creation date
        include_toc: Whether to include a table of contents (default: True)
        
    Returns:
        MarkdownGeneratorOutput with document creation results containing:
        - success: Whether the operation succeeded
        - message: Human-readable result message
        - data: Document ID, generated markdown, sections list, and word count
        
    Raises:
        ValueError: If template_name is invalid or not found
        KeyError: If required metadata fields are missing for template
        TemplateError: If template rendering fails due to missing variables
    """
    injector = get_injector()
    
    try:
        # Set defaults
        if metadata is None:
            metadata = {}
        if include_toc is None:
            include_toc = True
        
        # Generate document ID
        document_id = f"doc_{uuid.uuid4().hex[:8]}"
        
        # Set default metadata
        if 'creation_date' not in metadata:
            metadata['creation_date'] = datetime.now().isoformat()
        if 'author' not in metadata:
            metadata['author'] = 'System'
        
        # Create template variables
        template_vars = {
            **metadata,
            'include_toc': include_toc,
            'document_id': document_id
        }
        
        # Render template
        template_result = await injector.run('templates', {
            'operation': 'render',
            'template_name': f'markdown_{template_name}',
            'variables': template_vars
        })
        
        if not template_result.success:
            # Try creating template if it doesn't exist
            template_content = _get_default_template(template_name)
            
            await injector.run('templates', {
                'operation': 'save',
                'template_name': f'markdown_{template_name}',
                'template_content': template_content
            })
            
            # Retry rendering
            template_result = await injector.run('templates', {
                'operation': 'render',
                'template_name': f'markdown_{template_name}',
                'variables': template_vars
            })
            
            if not template_result.success:
                raise TemplateError(f"Failed to render template {template_name}: {template_result.message}")
        
        rendered_markdown = template_result.data['rendered']
        
        # Extract sections from markdown
        sections = _extract_sections(rendered_markdown)
        
        # Store document
        document_data = {
            'id': document_id,
            'template_name': template_name,
            'metadata': metadata,
            'markdown': rendered_markdown,
            'sections': sections,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'word_count': len(rendered_markdown.split())
        }
        
        _documents[document_id] = document_data
        
        # Cache in storage
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': f'document:{document_id}',
            'value': document_data,
            'namespace': 'markdown_generator',
            'ttl': 86400  # 24 hours
        })
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'markdown_generator',
            'message': f'Created document with template {template_name}',
            'data': {'document_id': document_id, 'template': template_name}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.markdown_generator.documents_created.count',
            'labels': {'template': template_name}
        })
        
        return MarkdownGeneratorOutput(
            success=True,
            message=f"Successfully created {template_name} document template",
            data={
                'document_id': document_id,
                'markdown': rendered_markdown,
                'sections': [s['title'] for s in sections],
                'word_count': document_data['word_count']
            }
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'markdown_generator',
            'message': 'Document creation failed',
            'data': {'error': str(e), 'template': template_name}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.markdown_generator.documents_created.errors'
        })
        
        raise


async def markdown_generator_add_section(
    ctx: RunContext[Any],
    document_id: str,
    section_title: str,
    content: str
) -> MarkdownGeneratorOutput:
    """
    Add a new section to an existing markdown document.
    
    This function adds a properly formatted section with content to an existing document,
    updates the table of contents if present, and maintains document structure.
    
    Args:
        ctx: Runtime context provided by the framework
        document_id: Unique identifier for the document being worked on
        section_title: Title for the new section being added
        content: Content to include in the new section
        
    Returns:
        MarkdownGeneratorOutput with section addition results containing:
        - success: Whether the operation succeeded
        - message: Human-readable result message
        - data: Section details, position, TOC update status, and word count
        
    Raises:
        FileNotFoundError: If document_id references non-existent document
        ContentError: If content is too large for single operation processing
    """
    injector = get_injector()
    
    try:
        # Retrieve document
        doc = await _get_document(document_id, injector)
        if not doc:
            raise FileNotFoundError(f"Document {document_id} not found")
        
        # Process content with LLM for formatting
        llm_result = await injector.run('llm', {
            'operation': 'markdownify',
            'content': content,
            'options': {'preserve_structure': True, 'add_formatting': True}
        })
        
        if llm_result.success and llm_result.data:
            formatted_content = llm_result.data.get('markdown', content)
        else:
            formatted_content = content
        
        # Create section markdown
        section_markdown = f"\n## {section_title}\n\n{formatted_content}\n"
        
        # Add section to document
        doc['markdown'] += section_markdown
        doc['updated_at'] = datetime.now().isoformat()
        
        # Update sections list
        section_info = {
            'title': section_title,
            'level': 2,
            'content': formatted_content,
            'position': len(doc['sections'])
        }
        doc['sections'].append(section_info)
        
        # Regenerate TOC if present
        toc_updated = False
        if '## Table of Contents' in doc['markdown'] or '[TOC]' in doc['markdown']:
            toc = _generate_table_of_contents(doc['markdown'])
            doc['markdown'] = _update_toc_in_markdown(doc['markdown'], toc)
            toc_updated = True
        
        # Update word count
        doc['word_count'] = len(doc['markdown'].split())
        
        # Store updated document
        _documents[document_id] = doc
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': f'document:{document_id}',
            'value': doc,
            'namespace': 'markdown_generator'
        })
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'markdown_generator',
            'message': f'Added section "{section_title}" to document {document_id}',
            'data': {'document_id': document_id, 'section_title': section_title}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.markdown_generator.sections_added.count'
        })
        
        return MarkdownGeneratorOutput(
            success=True,
            message=f"Successfully added section '{section_title}' to document",
            data={
                'section_added': section_title,
                'position': section_info['position'],
                'updated_toc': toc_updated,
                'word_count': doc['word_count']
            }
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'markdown_generator',
            'message': 'Failed to add section',
            'data': {'error': str(e), 'document_id': document_id}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.markdown_generator.sections_added.errors'
        })
        
        raise


async def markdown_generator_generate_toc(
    ctx: RunContext[Any],
    document_id: str,
    max_heading_level: Optional[int] = None
) -> MarkdownGeneratorOutput:
    """
    Generate or update table of contents for a document.
    
    This function creates a properly structured table of contents based on the document's
    heading hierarchy, with configurable maximum depth and anchor links.
    
    Args:
        ctx: Runtime context provided by the framework
        document_id: Unique identifier for the document
        max_heading_level: Maximum heading level to include in TOC (1-6, default: 3)
        
    Returns:
        MarkdownGeneratorOutput with TOC generation results
        
    Raises:
        FileNotFoundError: If document_id references non-existent document
    """
    injector = get_injector()
    
    try:
        if max_heading_level is None:
            max_heading_level = 3
        
        # Retrieve document
        doc = await _get_document(document_id, injector)
        if not doc:
            raise FileNotFoundError(f"Document {document_id} not found")
        
        # Generate TOC
        toc = _generate_table_of_contents(doc['markdown'], max_heading_level)
        
        # Update document with TOC
        doc['markdown'] = _update_toc_in_markdown(doc['markdown'], toc)
        doc['updated_at'] = datetime.now().isoformat()
        
        # Store updated document
        _documents[document_id] = doc
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': f'document:{document_id}',
            'value': doc,
            'namespace': 'markdown_generator'
        })
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'markdown_generator',
            'message': f'Generated TOC for document {document_id}',
            'data': {'document_id': document_id, 'max_level': max_heading_level}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.markdown_generator.toc_generated.count'
        })
        
        return MarkdownGeneratorOutput(
            success=True,
            message="Successfully generated table of contents",
            data={
                'toc': toc,
                'max_heading_level': max_heading_level,
                'entries_count': len(toc.split('\n')) - 1 if toc else 0
            }
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'markdown_generator',
            'message': 'Failed to generate TOC',
            'data': {'error': str(e), 'document_id': document_id}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.markdown_generator.toc_generated.errors'
        })
        
        raise


async def markdown_generator_validate_syntax(
    ctx: RunContext[Any],
    content: str,
    validation_level: Optional[str] = None
) -> MarkdownGeneratorOutput:
    """
    Validate markdown syntax and check for errors.
    
    This function performs comprehensive validation of markdown content including
    syntax checking, link validation, and structure analysis.
    
    Args:
        ctx: Runtime context provided by the framework
        content: Raw markdown content to validate
        validation_level: Level of validation (basic, strict, comprehensive, default: basic)
        
    Returns:
        MarkdownGeneratorOutput with validation results
        
    Raises:
        SyntaxError: If markdown contains invalid syntax and strict validation is enabled
    """
    injector = get_injector()
    
    try:
        if validation_level is None:
            validation_level = 'basic'
        
        errors = []
        warnings = []
        
        # Basic syntax validation
        try:
            md = markdown.Markdown(extensions=['codehilite', 'tables', 'toc'])
            html_output = md.convert(content)
        except Exception as e:
            errors.append({
                'line': 0,
                'type': 'syntax',
                'message': f'Markdown parsing error: {str(e)}'
            })
        
        # Check for unclosed code blocks
        code_block_pattern = r'```(\w+)?\n(.*?)```'
        content_lines = content.split('\n')
        
        in_code_block = False
        code_start_line = 0
        
        for i, line in enumerate(content_lines, 1):
            if line.startswith('```'):
                if not in_code_block:
                    in_code_block = True
                    code_start_line = i
                else:
                    in_code_block = False
            
        if in_code_block:
            errors.append({
                'line': code_start_line,
                'type': 'code_block',
                'message': 'Unclosed code block'
            })
        
        # Check for malformed links
        link_pattern = r'\[([^\]]*)\]\(([^)]*)\)'
        broken_link_pattern = r'\[([^\]]*)\]\([^)]*$'
        
        for i, line in enumerate(content_lines, 1):
            if re.search(broken_link_pattern, line):
                errors.append({
                    'line': i,
                    'type': 'syntax',
                    'message': 'Unclosed link bracket'
                })
        
        # Comprehensive validation
        if validation_level == 'comprehensive':
            # Check heading hierarchy
            headings = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
            prev_level = 0
            
            for i, (hashes, title) in enumerate(headings):
                level = len(hashes)
                if level > prev_level + 1 and prev_level > 0:
                    warnings.append({
                        'line': 0,
                        'type': 'structure',
                        'message': f'Heading level jumps from {prev_level} to {level}'
                    })
                prev_level = level
            
            # Check for empty headings
            for i, (hashes, title) in enumerate(headings):
                if not title.strip():
                    errors.append({
                        'line': 0,
                        'type': 'structure',
                        'message': 'Empty heading found'
                    })
        
        # Calculate quality score
        score = 100
        score -= len(errors) * 20
        score -= len(warnings) * 5
        score = max(0, min(100, score))
        
        is_valid = len(errors) == 0
        
        # Raise error if strict validation and errors found
        if validation_level == 'strict' and errors:
            raise SyntaxError(f"Markdown validation failed with {len(errors)} errors")
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'markdown_generator',
            'message': f'Validated markdown with {len(errors)} errors, {len(warnings)} warnings',
            'data': {'validation_level': validation_level, 'score': score}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.markdown_generator.validations.count',
            'labels': {'level': validation_level, 'valid': str(is_valid).lower()}
        })
        
        return MarkdownGeneratorOutput(
            success=is_valid,
            message=f"Markdown validation {'passed' if is_valid else f'failed with {len(errors)} errors'}",
            data={
                'valid': is_valid,
                'errors': errors,
                'warnings': warnings,
                'score': score
            }
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'markdown_generator',
            'message': 'Validation failed',
            'data': {'error': str(e)}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.markdown_generator.validations.errors'
        })
        
        raise


async def markdown_generator_apply_template(
    ctx: RunContext[Any],
    document_id: str,
    template_name: str,
    formatting_options: Optional[Dict[str, Any]] = None
) -> MarkdownGeneratorOutput:
    """
    Apply a template to an existing document.
    
    This function applies formatting and structure from a template to existing content
    while preserving the original content where possible.
    
    Args:
        ctx: Runtime context provided by the framework
        document_id: Document to apply template to
        template_name: Template to apply
        formatting_options: Custom formatting preferences
        
    Returns:
        MarkdownGeneratorOutput with template application results
        
    Raises:
        FileNotFoundError: If document_id references non-existent document
        ValueError: If template_name is invalid or not found
        TemplateError: If template rendering fails due to missing variables
    """
    injector = get_injector()
    
    try:
        # Retrieve document
        doc = await _get_document(document_id, injector)
        if not doc:
            raise FileNotFoundError(f"Document {document_id} not found")
        
        # Prepare template variables
        template_vars = {
            **doc['metadata'],
            'content': doc['markdown'],
            'sections': doc['sections'],
            'formatting_options': formatting_options or {}
        }
        
        # Apply template
        template_result = await injector.run('templates', {
            'operation': 'render',
            'template_name': f'markdown_{template_name}',
            'variables': template_vars
        })
        
        if not template_result.success:
            raise ValueError(f"Template {template_name} not found or failed to render")
        
        # Update document
        doc['markdown'] = template_result.data['rendered']
        doc['template_name'] = template_name
        doc['updated_at'] = datetime.now().isoformat()
        doc['sections'] = _extract_sections(doc['markdown'])
        doc['word_count'] = len(doc['markdown'].split())
        
        # Store updated document
        _documents[document_id] = doc
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': f'document:{document_id}',
            'value': doc,
            'namespace': 'markdown_generator'
        })
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'markdown_generator',
            'message': f'Applied template {template_name} to document {document_id}',
            'data': {'document_id': document_id, 'template': template_name}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.markdown_generator.templates_applied.count',
            'labels': {'template': template_name}
        })
        
        return MarkdownGeneratorOutput(
            success=True,
            message=f"Successfully applied template '{template_name}' to document",
            data={
                'document_id': document_id,
                'template_applied': template_name,
                'word_count': doc['word_count'],
                'sections_count': len(doc['sections'])
            }
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'markdown_generator',
            'message': 'Failed to apply template',
            'data': {'error': str(e), 'document_id': document_id, 'template': template_name}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.markdown_generator.templates_applied.errors'
        })
        
        raise


async def markdown_generator_format_content(
    ctx: RunContext[Any],
    content: str,
    formatting_options: Optional[Dict[str, Any]] = None
) -> MarkdownGeneratorOutput:
    """
    Format raw content into well-structured markdown.
    
    This function takes raw text or HTML content and converts it into properly
    formatted markdown with consistent styling and structure.
    
    Args:
        ctx: Runtime context provided by the framework
        content: Raw content to format
        formatting_options: Custom formatting preferences
        
    Returns:
        MarkdownGeneratorOutput with formatted content
    """
    injector = get_injector()
    
    try:
        if formatting_options is None:
            formatting_options = {}
        
        # Use LLM to format content
        llm_result = await injector.run('llm', {
            'operation': 'markdownify',
            'content': content,
            'options': {
                'preserve_structure': formatting_options.get('preserve_structure', True),
                'add_formatting': formatting_options.get('add_formatting', True),
                'heading_style': formatting_options.get('heading_style', 'atx'),
                'code_highlighting': formatting_options.get('code_highlighting', True)
            }
        })
        
        if llm_result.success and llm_result.data:
            formatted_content = llm_result.data.get('markdown', content)
        else:
            # Fallback basic formatting
            formatted_content = _basic_format_content(content, formatting_options)
        
        # Extract metrics
        word_count = len(formatted_content.split())
        sections = _extract_sections(formatted_content)
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'markdown_generator',
            'message': 'Content formatted successfully',
            'data': {'word_count': word_count, 'sections_count': len(sections)}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.markdown_generator.content_formatted.count'
        })
        
        return MarkdownGeneratorOutput(
            success=True,
            message="Successfully formatted content",
            data={
                'formatted_content': formatted_content,
                'word_count': word_count,
                'sections_count': len(sections),
                'formatting_applied': list(formatting_options.keys()) if formatting_options else []
            }
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'markdown_generator',
            'message': 'Content formatting failed',
            'data': {'error': str(e)}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.markdown_generator.content_formatted.errors'
        })
        
        raise


async def markdown_generator_assess_quality(
    ctx: RunContext[Any],
    document_id: str
) -> MarkdownGeneratorOutput:
    """
    Assess the quality and completeness of a document.
    
    This function evaluates various quality metrics including completeness,
    structure, readability, and formatting to provide an overall quality score.
    
    Args:
        ctx: Runtime context provided by the framework
        document_id: Document to assess
        
    Returns:
        MarkdownGeneratorOutput with quality assessment results
        
    Raises:
        FileNotFoundError: If document_id references non-existent document
    """
    injector = get_injector()
    
    try:
        # Retrieve document
        doc = await _get_document(document_id, injector)
        if not doc:
            raise FileNotFoundError(f"Document {document_id} not found")
        
        # Assess different quality aspects
        completeness_score = _assess_completeness(doc)
        structure_score = _assess_structure(doc['markdown'])
        readability_score = await _assess_readability(doc['markdown'], injector)
        formatting_score = _assess_formatting(doc['markdown'])
        
        # Calculate overall score
        overall_score = int((completeness_score + structure_score + readability_score + formatting_score) / 4)
        
        # Generate recommendations
        recommendations = []
        
        if completeness_score < 80:
            recommendations.append("Add more content to improve completeness")
        if structure_score < 80:
            recommendations.append("Improve document structure with better headings")
        if readability_score < 80:
            recommendations.append("Enhance readability with shorter paragraphs")
        if formatting_score < 80:
            recommendations.append("Fix formatting issues and add code examples")
        
        # Prepare metrics
        metrics = {
            'completeness': completeness_score,
            'structure': structure_score,
            'readability': readability_score,
            'formatting': formatting_score
        }
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'markdown_generator',
            'message': f'Assessed document {document_id} quality: {overall_score}%',
            'data': {'document_id': document_id, 'overall_score': overall_score}
        })
        
        await injector.run('metrics', {
            'operation': 'observe',
            'name': 'agentool.markdown_generator.quality_score.histogram',
            'value': overall_score
        })
        
        return MarkdownGeneratorOutput(
            success=True,
            message="Document quality assessment completed",
            data={
                'overall_score': overall_score,
                'metrics': metrics,
                'recommendations': recommendations,
                'word_count': doc['word_count'],
                'section_count': len(doc['sections'])
            }
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'markdown_generator',
            'message': 'Quality assessment failed',
            'data': {'error': str(e), 'document_id': document_id}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.markdown_generator.quality_assessments.errors'
        })
        
        raise


async def markdown_generator_export(
    ctx: RunContext[Any],
    document_id: str,
    output_path: str
) -> MarkdownGeneratorOutput:
    """
    Export a document to the filesystem.
    
    This function saves a markdown document to the specified file path with
    proper formatting and metadata preservation.
    
    Args:
        ctx: Runtime context provided by the framework
        document_id: Document to export
        output_path: File path where the document should be saved
        
    Returns:
        MarkdownGeneratorOutput with export results
        
    Raises:
        FileNotFoundError: If document_id references non-existent document
        IOError: If output_path is not writable or storage operations fail
    """
    injector = get_injector()
    
    try:
        # Retrieve document
        doc = await _get_document(document_id, injector)
        if not doc:
            raise FileNotFoundError(f"Document {document_id} not found")
        
        # Add metadata header
        metadata_header = f"""<!--
Document ID: {doc['id']}
Template: {doc.get('template_name', 'custom')}
Created: {doc['created_at']}
Updated: {doc['updated_at']}
Word Count: {doc['word_count']}
-->

"""
        
        final_content = metadata_header + doc['markdown']
        
        # Save to filesystem
        save_result = await injector.run('storage_fs', {
            'operation': 'write',
            'path': output_path,
            'content': final_content,
            'create_parents': True
        })
        
        if not save_result.success:
            raise IOError(f"Failed to save document: {save_result.message}")
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'markdown_generator',
            'message': f'Exported document {document_id} to {output_path}',
            'data': {'document_id': document_id, 'output_path': output_path}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.markdown_generator.documents_exported.count'
        })
        
        return MarkdownGeneratorOutput(
            success=True,
            message=f"Successfully exported document to {output_path}",
            data={
                'document_id': document_id,
                'output_path': output_path,
                'file_size': len(final_content),
                'word_count': doc['word_count']
            }
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'markdown_generator',
            'message': 'Document export failed',
            'data': {'error': str(e), 'document_id': document_id, 'output_path': output_path}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.markdown_generator.documents_exported.errors'
        })
        
        raise


# Helper functions
async def _get_document(document_id: str, injector) -> Optional[Dict[str, Any]]:
    """Retrieve document from cache or storage."""
    if document_id in _documents:
        return _documents[document_id]
    
    # Try to get from storage
    storage_result = await injector.run('storage_kv', {
        'operation': 'get',
        'key': f'document:{document_id}',
        'namespace': 'markdown_generator'
    })
    
    if storage_result.success and storage_result.data and storage_result.data.get('value'):
        doc = storage_result.data['value']
        _documents[document_id] = doc
        return doc
    
    return None


def _get_default_template(template_name: str) -> str:
    """Get default template content for a given template name."""
    templates = {
        'api_docs': '''# {{ title }}

{% if author %}*By {{ author }}*{% endif %}
{% if creation_date %}*Created: {{ creation_date }}*{% endif %}

{% if include_toc %}
## Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
- [Examples](#examples)
- [Error Handling](#error-handling)

{% endif %}
## Overview

{{ description | default("API documentation overview.") }}

## Authentication

Authentication details here.

## Endpoints

API endpoints documentation.

## Examples

Code examples and usage.

## Error Handling

Error response documentation.
''',
        'trend_analysis': '''# {{ title }}

{% if author %}*By {{ author }}*{% endif %}
{% if creation_date %}*Created: {{ creation_date }}*{% endif %}

{% if include_toc %}
## Table of Contents

- [Executive Summary](#executive-summary)
- [Key Trends](#key-trends)
- [Analysis](#analysis)
- [Recommendations](#recommendations)
- [Data Sources](#data-sources)

{% endif %}
## Executive Summary

Brief overview of key findings.

## Key Trends

1. **Trend 1**: Description
2. **Trend 2**: Description
3. **Trend 3**: Description

## Analysis

Detailed analysis of trends and data.

## Recommendations

Strategic recommendations based on analysis.

## Data Sources

{% if sources %}
{% for source in sources %}
- {{ source }}
{% endfor %}
{% else %}
Data sources will be listed here.
{% endif %}
''',
        'comparison_report': '''# {{ title }}

{% if author %}*By {{ author }}*{% endif %}
{% if creation_date %}*Created: {{ creation_date }}*{% endif %}

{% if include_toc %}
## Table of Contents

- [Summary](#summary)
- [Comparison Matrix](#comparison-matrix)
- [Detailed Analysis](#detailed-analysis)
- [Conclusion](#conclusion)

{% endif %}
## Summary

Comparison overview and key findings.

## Comparison Matrix

| Feature | Option A | Option B | Option C |
|---------|----------|----------|----------|
| Feature 1 | Value | Value | Value |
| Feature 2 | Value | Value | Value |

## Detailed Analysis

### Option A
Detailed analysis of option A.

### Option B
Detailed analysis of option B.

### Option C
Detailed analysis of option C.

## Conclusion

Final recommendations and decision criteria.
''',
        'technical_guide': '''# {{ title }}

{% if author %}*By {{ author }}*{% endif %}
{% if creation_date %}*Created: {{ creation_date }}*{% endif %}

{% if include_toc %}
## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

{% endif %}
## Prerequisites

Requirements before starting.

## Installation

Step-by-step installation instructions.

```bash
# Installation commands
```

## Configuration

Configuration options and settings.

## Usage

How to use the system or tool.

## Examples

Practical examples and code samples.

```python
# Example code
```

## Troubleshooting

Common issues and solutions.
'''
    }
    
    return templates.get(template_name, templates['technical_guide'])


def _extract_sections(markdown: str) -> List[Dict[str, Any]]:
    """Extract section information from markdown content."""
    sections = []
    lines = markdown.split('\n')
    
    for i, line in enumerate(lines):
        if line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            if level > 0 and level <= 6:
                title = line.lstrip('#').strip()
                sections.append({
                    'title': title,
                    'level': level,
                    'line': i + 1
                })
    
    return sections


def _generate_table_of_contents(markdown: str, max_level: int = 3) -> str:
    """Generate table of contents from markdown headings."""
    sections = _extract_sections(markdown)
    toc_lines = []
    
    for section in sections:
        if section['level'] <= max_level:
            indent = '  ' * (section['level'] - 1)
            anchor = section['title'].lower().replace(' ', '-').replace('[^a-z0-9-]', '')
            toc_lines.append(f"{indent}- [{section['title']}](#{anchor})")
    
    return '\n'.join(toc_lines)


def _update_toc_in_markdown(markdown: str, toc: str) -> str:
    """Update or insert table of contents in markdown."""
    lines = markdown.split('\n')
    
    # Find existing TOC
    toc_start = -1
    toc_end = -1
    
    for i, line in enumerate(lines):
        if 'Table of Contents' in line:
            toc_start = i
        elif toc_start >= 0 and line.startswith('#') and not line.startswith('##'):
            toc_end = i
            break
    
    if toc_start >= 0:
        # Replace existing TOC
        new_lines = lines[:toc_start+1] + [''] + toc.split('\n') + [''] + lines[toc_end:]
        return '\n'.join(new_lines)
    else:
        # Add new TOC after first heading
        for i, line in enumerate(lines):
            if line.startswith('#'):
                new_lines = lines[:i+1] + [''] + ['## Table of Contents'] + [''] + toc.split('\n') + [''] + lines[i+1:]
                return '\n'.join(new_lines)
    
    return markdown


def _basic_format_content(content: str, options: Dict[str, Any]) -> str:
    """Basic content formatting fallback."""
    # Simple text to markdown conversion
    lines = content.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            formatted_lines.append('')
        elif line.isupper() and len(line) > 3:
            # Likely a heading
            formatted_lines.append(f"## {line.title()}")
        elif line.endswith(':') and len(line.split()) <= 5:
            # Likely a subheading
            formatted_lines.append(f"### {line}")
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)


def _assess_completeness(doc: Dict[str, Any]) -> int:
    """Assess document completeness."""
    score = 50  # Base score
    
    if doc['word_count'] > 100:
        score += 20
    if doc['word_count'] > 500:
        score += 10
    if doc['word_count'] > 1000:
        score += 10
    
    if len(doc['sections']) >= 3:
        score += 10
    
    return min(100, score)


def _assess_structure(markdown: str) -> int:
    """Assess document structure quality."""
    sections = _extract_sections(markdown)
    score = 60  # Base score
    
    if len(sections) >= 3:
        score += 20
    if len(sections) >= 5:
        score += 10
    
    # Check heading hierarchy
    prev_level = 0
    hierarchy_good = True
    for section in sections:
        if section['level'] > prev_level + 1 and prev_level > 0:
            hierarchy_good = False
            break
        prev_level = section['level']
    
    if hierarchy_good:
        score += 10
    
    return min(100, score)


async def _assess_readability(markdown: str, injector) -> int:
    """Assess document readability using LLM."""
    try:
        llm_result = await injector.run('llm', {
            'operation': 'classification',
            'content': markdown,
            'classes': ['excellent', 'good', 'fair', 'poor'],
            'options': {'criteria': 'readability and clarity'}
        })
        
        if llm_result.success and llm_result.data:
            readability = llm_result.data.get('selected_class', 'fair')
            scores = {'excellent': 95, 'good': 80, 'fair': 65, 'poor': 40}
            return scores.get(readability, 65)
    except:
        pass
    
    # Fallback: simple readability assessment
    word_count = len(markdown.split())
    sentence_count = markdown.count('.') + markdown.count('!') + markdown.count('?')
    
    if sentence_count > 0:
        avg_words_per_sentence = word_count / sentence_count
        if avg_words_per_sentence <= 15:
            return 85
        elif avg_words_per_sentence <= 20:
            return 75
        else:
            return 60
    
    return 70


def _assess_formatting(markdown: str) -> int:
    """Assess document formatting quality."""
    score = 50  # Base score
    
    # Check for code blocks
    if '```' in markdown:
        score += 15
    
    # Check for lists
    if re.search(r'^[-*+]\s', markdown, re.MULTILINE) or re.search(r'^\d+\.\s', markdown, re.MULTILINE):
        score += 10
    
    # Check for tables
    if '|' in markdown and '---' in markdown:
        score += 10
    
    # Check for links
    if re.search(r'\[([^\]]*)\]\(([^)]*)\)', markdown):
        score += 10
    
    # Check for emphasis
    if '*' in markdown or '_' in markdown:
        score += 5
    
    return min(100, score)


# Custom exception classes
class TemplateError(Exception):
    """Raised when template operations fail."""
    pass

class ContentError(Exception):
    """Raised when content operations fail."""
    pass


# Routing configuration
markdown_generator_routing = RoutingConfig(
    operation_field='operation',
    operation_map={
        'create_document': ('markdown_generator_create_document', lambda x: {
            'template_name': x.template_name,
            'metadata': x.metadata,
            'include_toc': x.include_toc,
        }),
        'add_section': ('markdown_generator_add_section', lambda x: {
            'document_id': x.document_id,
            'section_title': x.section_title,
            'content': x.content,
        }),
        'generate_toc': ('markdown_generator_generate_toc', lambda x: {
            'document_id': x.document_id,
            'max_heading_level': x.max_heading_level,
        }),
        'validate_syntax': ('markdown_generator_validate_syntax', lambda x: {
            'content': x.content,
            'validation_level': x.validation_level,
        }),
        'apply_template': ('markdown_generator_apply_template', lambda x: {
            'document_id': x.document_id,
            'template_name': x.template_name,
            'formatting_options': x.formatting_options,
        }),
        'format_content': ('markdown_generator_format_content', lambda x: {
            'content': x.content,
            'formatting_options': x.formatting_options,
        }),
        'assess_quality': ('markdown_generator_assess_quality', lambda x: {
            'document_id': x.document_id,
        }),
        'export': ('markdown_generator_export', lambda x: {
            'document_id': x.document_id,
            'output_path': x.output_path,
        }),
    }
)


def create_markdown_generator_agent():
    """
    Create and return the markdown_generator AgenTool.
    
    Returns:
        Agent configured for markdown_generator operations
    """
    return create_agentool(
        name='markdown_generator',
        input_schema=MarkdownGeneratorInput,
        routing_config=markdown_generator_routing,
        tools=[
            markdown_generator_create_document,
            markdown_generator_add_section,
            markdown_generator_generate_toc,
            markdown_generator_validate_syntax,
            markdown_generator_apply_template,
            markdown_generator_format_content,
            markdown_generator_assess_quality,
            markdown_generator_export
        ],
        output_type=MarkdownGeneratorOutput,
        system_prompt="You are a markdown document generator that creates comprehensive, well-structured documents from research data with proper formatting, templates, and quality assessment.",
        description="Generates comprehensive markdown documents with create_document, add_section, generate_toc, validate_syntax, apply_template, format_content, assess_quality, and export operations",
        version="1.0.0",
        tags=["markdown", "documentation", "templates", "formatting", "quality-assessment"],
        dependencies=["llm", "templates", "storage_fs", "storage_kv", "logging", "metrics"],
        examples=[
            {
                "description": "Create a new document from template",
                "input": {
                    "operation": "create_document",
                    "template_name": "api_docs",
                    "metadata": {"title": "REST API Documentation", "author": "System", "sources": ["https://api.example.com"], "creation_date": "2025-01-20"},
                    "include_toc": True
                },
                "output": {
                    "success": True,
                    "message": "Successfully created API documentation template",
                    "data": {"document_id": "doc_123", "markdown": "# REST API Documentation\n\n## Table of Contents\n\n- [Overview](#overview)\n- [Authentication](#authentication)\n- [Endpoints](#endpoints)\n\n## Overview\n\n...", "sections": ["overview", "authentication", "endpoints"], "word_count": 0}
                }
            },
            {
                "description": "Add a section to existing document",
                "input": {
                    "operation": "add_section",
                    "document_id": "doc_123",
                    "section_title": "Error Handling",
                    "content": "The API returns standard HTTP status codes and structured error responses in JSON format."
                },
                "output": {
                    "success": True,
                    "message": "Successfully added section 'Error Handling' to document",
                    "data": {"section_added": "Error Handling", "position": 4, "updated_toc": True, "word_count": 156}
                }
            },
            {
                "description": "Validate markdown syntax",
                "input": {
                    "operation": "validate_syntax",
                    "content": "# Title\n\n[Broken link](missing-url\n\n```python\nprint('hello'\n```",
                    "validation_level": "comprehensive"
                },
                "output": {
                    "success": False,
                    "message": "Markdown validation failed with 2 errors",
                    "data": {"valid": False, "errors": [{"line": 3, "type": "syntax", "message": "Unclosed link bracket"}, {"line": 5, "type": "code_block", "message": "Unclosed code block"}], "warnings": [], "score": 65}
                }
            }
        ]
    )


# Create the agent instance
agent = create_markdown_generator_agent()
