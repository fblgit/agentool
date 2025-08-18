"""
content_extractor AgenTool - Extracts and structures content from raw HTML, text, and web pages with intelligent parsing.

This AgenTool provides comprehensive HTML parsing and content extraction capabilities using BeautifulSoup,
readability algorithms, and LLM-powered analysis. It removes boilerplate content, extracts structured data,
and provides quality assessment for various content types including documentation, blogs, news, and API references.

Key Features:
- HTML Parsing: Robust parsing using BeautifulSoup with lxml parser for malformed HTML handling
- Content Extraction: Intelligent main content extraction removing navigation, ads, and footers
- Structured Data: Extract headings, links, metadata, code blocks, and technical content
- Quality Assessment: Content scoring and relevance assessment with LLM integration
- Multi-format Support: Handle documentation sites, blogs, news articles, and API references

Usage Example:
    >>> from agentoolkit.content_extractor import create_content_extractor_agent
    >>> from agentool.core.injector import get_injector
    >>> 
    >>> # Create and register the agent
    >>> agent = create_content_extractor_agent()
    >>> 
    >>> # Use through injector
    >>> injector = get_injector()
    >>> result = await injector.run('content_extractor', {
    ...     "operation": "parse_html",
    ...     "content": "<html><body>...</body></html>",
    ...     "content_type": "blog"
    ... })
"""

import re
import json
import hashlib
from typing import Dict, Any, List, Optional, Literal, Union
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup, Comment
from readability import Document
import chardet
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import RunContext

from agentool import create_agentool
from agentool.base import BaseOperationInput
from agentool.core.registry import RoutingConfig
from agentool.core.injector import get_injector


class ContentExtractorInput(BaseOperationInput):
    """Input schema for content_extractor operations."""
    
    operation: Literal[
        'parse_html', 
        'extract_content', 
        'extract_metadata', 
        'extract_links', 
        'extract_code', 
        'score_quality', 
        'clean_content'
    ] = Field(description="Content extraction operation to perform")
    
    content: str = Field(description="Raw HTML or text content to process")
    
    url: Optional[str] = Field(
        None,
        description="Source URL for context and metadata extraction"
    )
    
    content_type: str = Field(
        default="html",
        description="Type of content for specialized extraction rules"
    )
    
    options: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional extraction options and configuration"
    )
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        """Validate content is not empty."""
        if not v or not v.strip():
            raise ValueError("Content cannot be empty")
        return v
    
    @field_validator('content_type')
    @classmethod
    def validate_content_type(cls, v):
        """Validate content_type is supported."""
        valid_types = ["html", "text", "documentation", "blog", "news", "api_reference"]
        if v not in valid_types:
            raise ValueError(f"content_type must be one of: {valid_types}")
        return v


class ContentExtractorOutput(BaseModel):
    """Output schema for content_extractor operations."""
    success: bool = Field(description="Whether the extraction operation succeeded")
    message: str = Field(description="Human-readable result message")
    data: Optional[Dict[str, Any]] = Field(None, description="Extracted content and metadata")


def _detect_encoding(content: bytes) -> str:
    """Detect content encoding using chardet."""
    if isinstance(content, str):
        return 'utf-8'
    
    detected = chardet.detect(content)
    return detected.get('encoding', 'utf-8') or 'utf-8'


def _clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    return text.strip()


def _extract_sections(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Extract structured sections from parsed HTML."""
    sections = []
    
    # Extract headings
    for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        level = int(heading.name[1])
        text = _clean_text(heading.get_text())
        if text:
            sections.append({
                "type": "heading",
                "level": level,
                "text": text,
                "id": heading.get('id'),
                "classes": heading.get('class', [])
            })
    
    # Extract paragraphs
    for para in soup.find_all('p'):
        text = _clean_text(para.get_text())
        if text and len(text) > 10:  # Filter out very short paragraphs
            sections.append({
                "type": "paragraph",
                "text": text,
                "classes": para.get('class', [])
            })
    
    # Extract lists
    for list_elem in soup.find_all(['ul', 'ol']):
        items = []
        for li in list_elem.find_all('li', recursive=False):
            item_text = _clean_text(li.get_text())
            if item_text:
                items.append(item_text)
        
        if items:
            sections.append({
                "type": "list",
                "list_type": list_elem.name,
                "items": items
            })
    
    return sections


def _extract_links(soup: BeautifulSoup, base_url: Optional[str] = None) -> List[Dict[str, Any]]:
    """Extract all links from the HTML."""
    links = []
    
    for link in soup.find_all('a', href=True):
        href = link['href'].strip()
        if not href or href.startswith('#'):
            continue
            
        # Resolve relative URLs
        if base_url and not urlparse(href).netloc:
            href = urljoin(base_url, href)
        
        text = _clean_text(link.get_text())
        title = link.get('title', '')
        
        links.append({
            "url": href,
            "text": text,
            "title": title,
            "rel": link.get('rel', []),
            "classes": link.get('class', [])
        })
    
    return links


def _extract_metadata(soup: BeautifulSoup, url: Optional[str] = None) -> Dict[str, Any]:
    """Extract metadata from HTML head and structured data."""
    metadata = {}
    
    # Basic metadata
    if url:
        metadata['url'] = url
        parsed_url = urlparse(url)
        metadata['domain'] = parsed_url.netloc
    
    # Title
    title_elem = soup.find('title')
    if title_elem:
        metadata['title'] = _clean_text(title_elem.get_text())
    
    # Meta tags
    for meta in soup.find_all('meta'):
        name = meta.get('name') or meta.get('property')
        content = meta.get('content')
        
        if name and content:
            # Handle common meta tags
            if name in ['description', 'author', 'generator']:
                metadata[name] = content
            elif name == 'keywords':
                metadata['keywords'] = [k.strip() for k in content.split(',')]
            elif name.startswith('og:'):
                if 'open_graph' not in metadata:
                    metadata['open_graph'] = {}
                metadata['open_graph'][name[3:]] = content
            elif name.startswith('twitter:'):
                if 'twitter' not in metadata:
                    metadata['twitter'] = {}
                metadata['twitter'][name[8:]] = content
    
    # Language
    html_elem = soup.find('html')
    if html_elem and html_elem.get('lang'):
        metadata['language'] = html_elem['lang']
    
    return metadata


def _extract_code_blocks(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Extract code blocks from HTML."""
    code_blocks = []
    
    # Pre/code blocks
    for pre in soup.find_all('pre'):
        code_elem = pre.find('code')
        if code_elem:
            code_text = code_elem.get_text()
        else:
            code_text = pre.get_text()
        
        if code_text.strip():
            # Try to detect language
            language = None
            classes = (code_elem or pre).get('class', [])
            for cls in classes:
                if cls.startswith('language-'):
                    language = cls[9:]
                    break
                elif cls.startswith('lang-'):
                    language = cls[5:]
                    break
                elif cls in ['python', 'javascript', 'java', 'cpp', 'c', 'html', 'css', 'sql', 'bash', 'json', 'xml', 'yaml']:
                    language = cls
                    break
            
            code_blocks.append({
                "language": language,
                "code": code_text,
                "line_count": len(code_text.split('\n'))
            })
    
    # Inline code
    for code in soup.find_all('code'):
        if code.parent.name != 'pre':  # Skip if already handled in pre blocks
            code_text = code.get_text().strip()
            if code_text and len(code_text) > 3:  # Filter very short inline code
                code_blocks.append({
                    "language": None,
                    "code": code_text,
                    "line_count": 1,
                    "inline": True
                })
    
    return code_blocks


def _calculate_reading_time(text: str, wpm: int = 200) -> float:
    """Calculate estimated reading time in minutes."""
    word_count = len(text.split())
    return round(word_count / wpm, 2)


def _remove_boilerplate(soup: BeautifulSoup) -> None:
    """Remove common boilerplate elements."""
    # Remove common boilerplate selectors
    boilerplate_selectors = [
        'nav', 'header', 'footer', 'aside',
        '.navigation', '.nav', '.menu',
        '.header', '.footer', '.sidebar',
        '.advertisement', '.ads', '.ad',
        '.social-media', '.share-buttons',
        '.comments', '.comment-section',
        '[role="navigation"]', '[role="banner"]', '[role="contentinfo"]'
    ]
    
    for selector in boilerplate_selectors:
        for elem in soup.select(selector):
            elem.decompose()
    
    # Remove script and style tags
    for elem in soup.find_all(['script', 'style', 'noscript']):
        elem.decompose()
    
    # Remove comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()


async def content_extractor_parse_html(
    ctx: RunContext[Any],
    content: str,
    url: Optional[str] = None,
    content_type: str = "html",
    options: Optional[Dict[str, Any]] = None
) -> ContentExtractorOutput:
    """
    Parse HTML content and extract main content with structure.
    
    Args:
        ctx: Runtime context provided by the framework
        content: Raw HTML content to parse
        url: Source URL for context and metadata extraction
        content_type: Type of content for specialized extraction rules
        options: Additional extraction options and configuration
        
    Returns:
        ContentExtractorOutput with parsed content including title, sections, links, and metadata
        
    Raises:
        ValueError: If content is invalid or cannot be parsed
        UnicodeDecodeError: If content encoding cannot be determined
    """
    injector = get_injector()
    
    try:
        # Parse options
        opts = options or {}
        preserve_formatting = opts.get('preserve_formatting', False)
        extract_images = opts.get('extract_images', True)
        include_tables = opts.get('include_tables', True)
        min_text_length = opts.get('min_text_length', 50)
        
        # Detect encoding if content is bytes
        if isinstance(content, bytes):
            encoding = _detect_encoding(content)
            content = content.decode(encoding, errors='ignore')
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(content, 'lxml')
        
        # Remove boilerplate content
        _remove_boilerplate(soup)
        
        # Extract title
        title = ""
        title_elem = soup.find('title')
        if title_elem:
            title = _clean_text(title_elem.get_text())
        elif soup.find('h1'):
            title = _clean_text(soup.find('h1').get_text())
        
        # Extract main content
        main_content = ""
        main_elem = soup.find(['main', 'article']) or soup.find('body')
        if main_elem:
            main_content = _clean_text(main_elem.get_text())
        
        # Convert to markdown if requested
        if not preserve_formatting and main_content:
            llm_result = await injector.run('llm', {
                'operation': 'markdownify',
                'content': main_content,
                'options': {'preserve_structure': True}
            })
            if llm_result.success:
                main_content = llm_result.data.get('result', main_content)
        
        # Extract structured sections
        sections = _extract_sections(soup)
        
        # Extract links
        links = _extract_links(soup, url)
        
        # Extract metadata
        metadata = _extract_metadata(soup, url)
        metadata['content_type'] = content_type
        
        # Calculate metrics
        word_count = len(main_content.split()) if main_content else 0
        reading_time = _calculate_reading_time(main_content) if main_content else 0
        
        # Log successful parsing
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'content_extractor',
            'message': f'Successfully parsed HTML content from {url or "unknown source"}',
            'data': {
                'content_type': content_type,
                'word_count': word_count,
                'sections_count': len(sections),
                'links_count': len(links)
            }
        })
        
        # Track metrics
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.content_extractor.parse_html.success',
            'labels': {'content_type': content_type}
        })
        
        await injector.run('metrics', {
            'operation': 'observe',
            'name': 'agentool.content_extractor.parse_html.word_count.histogram',
            'value': float(word_count)
        })
        
        return ContentExtractorOutput(
            success=True,
            message="Successfully parsed HTML content",
            data={
                "title": title,
                "content": main_content,
                "sections": sections,
                "links": links,
                "metadata": metadata,
                "word_count": word_count,
                "reading_time": reading_time
            }
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'content_extractor',
            'message': f'Failed to parse HTML content: {str(e)}',
            'data': {'url': url, 'content_type': content_type, 'error': str(e)}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.content_extractor.parse_html.errors',
            'labels': {'content_type': content_type}
        })
        
        raise ValueError(f"Failed to parse HTML content: {str(e)}") from e


async def content_extractor_extract_content(
    ctx: RunContext[Any],
    content: str,
    url: Optional[str] = None,
    content_type: str = "html",
    options: Optional[Dict[str, Any]] = None
) -> ContentExtractorOutput:
    """
    Extract main content using readability algorithms.
    
    Args:
        ctx: Runtime context provided by the framework
        content: Raw HTML content to extract from
        url: Source URL for context
        content_type: Type of content for specialized extraction
        options: Additional extraction options
        
    Returns:
        ContentExtractorOutput with extracted main content
    """
    injector = get_injector()
    
    try:
        # Use readability-lxml for main content extraction
        doc = Document(content, url=url)
        
        # Extract main content HTML
        main_content_html = doc.summary()
        title = _clean_text(doc.title()) if doc.title() else ""
        
        # Parse extracted content and convert to plain text
        soup = BeautifulSoup(main_content_html, 'lxml')
        main_content = _clean_text(soup.get_text())
        sections = _extract_sections(soup)
        
        word_count = len(main_content.split()) if main_content else 0
        reading_time = _calculate_reading_time(main_content) if main_content else 0
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'content_extractor',
            'message': 'Successfully extracted main content',
            'data': {'word_count': word_count, 'url': url}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.content_extractor.extract_content.success'
        })
        
        return ContentExtractorOutput(
            success=True,
            message="Successfully extracted main content",
            data={
                "title": title,
                "content": main_content,
                "sections": sections,
                "word_count": word_count,
                "reading_time": reading_time,
                "extraction_method": "readability"
            }
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'content_extractor',
            'message': f'Failed to extract content: {str(e)}',
            'data': {'url': url, 'error': str(e)}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.content_extractor.extract_content.errors'
        })
        
        raise ValueError(f"Failed to extract content: {str(e)}") from e


async def content_extractor_extract_metadata(
    ctx: RunContext[Any],
    content: str,
    url: Optional[str] = None,
    content_type: str = "html",
    options: Optional[Dict[str, Any]] = None
) -> ContentExtractorOutput:
    """
    Extract structured metadata from content.
    
    Args:
        ctx: Runtime context provided by the framework
        content: Raw HTML content to extract metadata from
        url: Source URL for context
        content_type: Type of content
        options: Additional extraction options
        
    Returns:
        ContentExtractorOutput with extracted metadata
    """
    injector = get_injector()
    
    try:
        soup = BeautifulSoup(content, 'lxml')
        metadata = _extract_metadata(soup, url)
        metadata['content_type'] = content_type
        
        # Extract structured data (JSON-LD, microdata)
        structured_data = []
        
        # JSON-LD
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                structured_data.append({
                    "type": "json-ld",
                    "data": data
                })
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Microdata
        for elem in soup.find_all(attrs={'itemscope': True}):
            item_type = elem.get('itemtype')
            if item_type:
                structured_data.append({
                    "type": "microdata",
                    "itemtype": item_type,
                    "properties": {}
                })
        
        if structured_data:
            metadata['structured_data'] = structured_data
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'content_extractor',
            'message': 'Successfully extracted metadata',
            'data': {'metadata_fields': len(metadata), 'url': url}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.content_extractor.extract_metadata.success'
        })
        
        return ContentExtractorOutput(
            success=True,
            message="Successfully extracted metadata",
            data={"metadata": metadata}
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'content_extractor',
            'message': f'Failed to extract metadata: {str(e)}',
            'data': {'url': url, 'error': str(e)}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.content_extractor.extract_metadata.errors'
        })
        
        raise ValueError(f"Failed to extract metadata: {str(e)}") from e


async def content_extractor_extract_links(
    ctx: RunContext[Any],
    content: str,
    url: Optional[str] = None,
    content_type: str = "html",
    options: Optional[Dict[str, Any]] = None
) -> ContentExtractorOutput:
    """
    Extract and categorize all links from content.
    
    Args:
        ctx: Runtime context provided by the framework
        content: Raw HTML content to extract links from
        url: Base URL for resolving relative links
        content_type: Type of content
        options: Additional extraction options
        
    Returns:
        ContentExtractorOutput with extracted and categorized links
    """
    injector = get_injector()
    
    try:
        soup = BeautifulSoup(content, 'lxml')
        links = _extract_links(soup, url)
        
        # Categorize links
        internal_links = []
        external_links = []
        
        if url:
            base_domain = urlparse(url).netloc
            for link in links:
                parsed_link = urlparse(link['url'])
                link_domain = parsed_link.netloc
                link_scheme = parsed_link.scheme
                
                # Internal links: same domain and http/https schemes, or relative URLs without scheme
                if (link_domain == base_domain and link_scheme in ['http', 'https']) or \
                   (not link_domain and not link_scheme):
                    internal_links.append(link)
                else:
                    external_links.append(link)
        else:
            external_links = links
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'content_extractor',
            'message': f'Successfully extracted {len(links)} links',
            'data': {
                'total_links': len(links),
                'internal_links': len(internal_links),
                'external_links': len(external_links)
            }
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.content_extractor.extract_links.success'
        })
        
        return ContentExtractorOutput(
            success=True,
            message=f"Successfully extracted {len(links)} links",
            data={
                "links": links,
                "internal_links": internal_links,
                "external_links": external_links,
                "link_count": len(links)
            }
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'content_extractor',
            'message': f'Failed to extract links: {str(e)}',
            'data': {'url': url, 'error': str(e)}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.content_extractor.extract_links.errors'
        })
        
        raise ValueError(f"Failed to extract links: {str(e)}") from e


async def content_extractor_extract_code(
    ctx: RunContext[Any],
    content: str,
    url: Optional[str] = None,
    content_type: str = "html",
    options: Optional[Dict[str, Any]] = None
) -> ContentExtractorOutput:
    """
    Extract and structure code blocks from content.
    
    Args:
        ctx: Runtime context provided by the framework
        content: Raw HTML content to extract code from
        url: Source URL for context
        content_type: Type of content
        options: Additional extraction options
        
    Returns:
        ContentExtractorOutput with extracted code blocks
    """
    injector = get_injector()
    
    try:
        soup = BeautifulSoup(content, 'lxml')
        code_blocks = _extract_code_blocks(soup)
        
        # Filter out inline code if requested
        opts = options or {}
        include_inline = opts.get('include_inline', True)
        
        if not include_inline:
            code_blocks = [block for block in code_blocks if not block.get('inline', False)]
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'content_extractor',
            'message': f'Successfully extracted {len(code_blocks)} code blocks',
            'data': {'code_blocks_count': len(code_blocks), 'url': url}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.content_extractor.extract_code.success'
        })
        
        return ContentExtractorOutput(
            success=True,
            message=f"Successfully extracted {len(code_blocks)} code blocks",
            data={"code_blocks": code_blocks}
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'content_extractor',
            'message': f'Failed to extract code blocks: {str(e)}',
            'data': {'url': url, 'error': str(e)}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.content_extractor.extract_code.errors'
        })
        
        raise ValueError(f"Failed to extract code blocks: {str(e)}") from e


async def content_extractor_score_quality(
    ctx: RunContext[Any],
    content: str,
    url: Optional[str] = None,
    content_type: str = "html",
    options: Optional[Dict[str, Any]] = None
) -> ContentExtractorOutput:
    """
    Score content quality and provide recommendations.
    
    Args:
        ctx: Runtime context provided by the framework
        content: Raw HTML content to score
        url: Source URL for context
        content_type: Type of content
        options: Additional scoring options
        
    Returns:
        ContentExtractorOutput with quality score and recommendations
    """
    injector = get_injector()
    
    try:
        soup = BeautifulSoup(content, 'lxml')
        _remove_boilerplate(soup)
        
        # Extract main content for analysis
        main_elem = soup.find(['main', 'article']) or soup.find('body')
        main_content = _clean_text(main_elem.get_text()) if main_elem else ""
        
        word_count = len(main_content.split()) if main_content else 0
        
        # Basic quality factors
        factors = {}
        
        # Content length score (0-1)
        if word_count >= 1000:
            factors['content_length'] = 1.0
        elif word_count >= 500:
            factors['content_length'] = 0.8
        elif word_count >= 100:
            factors['content_length'] = 0.6
        else:
            factors['content_length'] = 0.3
        
        # Structure score based on headings and sections
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        paragraphs = soup.find_all('p')
        
        structure_elements = len(headings) + len(paragraphs)
        if structure_elements >= 10:
            factors['structure_score'] = 1.0
        elif structure_elements >= 5:
            factors['structure_score'] = 0.8
        elif structure_elements >= 2:
            factors['structure_score'] = 0.6
        else:
            factors['structure_score'] = 0.3
        
        # Use LLM for advanced analysis
        llm_result = await injector.run('llm', {
            'operation': 'extraction',
            'content': main_content[:2000],  # Limit for analysis
            'extraction_schema': {
                'readability_score': 'float',
                'coherence_score': 'float',
                'informativeness_score': 'float'
            },
            'options': {'max_tokens': 500}
        })
        
        if llm_result.success and llm_result.data.get('result'):
            llm_scores = llm_result.data['result']
            factors['readability'] = llm_scores.get('readability_score', 0.5)
            factors['coherence'] = llm_scores.get('coherence_score', 0.5)
            factors['informativeness'] = llm_scores.get('informativeness_score', 0.5)
        else:
            factors['readability'] = 0.5
            factors['coherence'] = 0.5
            factors['informativeness'] = 0.5
        
        # Calculate overall score
        quality_score = sum(factors.values()) / len(factors)
        
        # Generate recommendations
        recommendations = []
        if factors['content_length'] < 0.7:
            recommendations.append("Add more detailed content")
        if factors['structure_score'] < 0.7:
            recommendations.append("Add more subheadings")
        if len(soup.find_all(['pre', 'code'])) == 0 and content_type in ['documentation', 'api_reference']:
            recommendations.append("Include code examples")
        if not soup.find_all('img') and content_type in ['blog', 'news']:
            recommendations.append("Add relevant images")
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'content_extractor',
            'message': f'Content quality scored: {quality_score:.2f}',
            'data': {
                'quality_score': quality_score,
                'word_count': word_count,
                'url': url
            }
        })
        
        await injector.run('metrics', {
            'operation': 'observe',
            'name': 'agentool.content_extractor.quality_score.histogram',
            'value': quality_score
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.content_extractor.score_quality.success'
        })
        
        return ContentExtractorOutput(
            success=True,
            message="Successfully scored content quality",
            data={
                "quality_score": round(quality_score, 2),
                "factors": factors,
                "recommendations": recommendations,
                "word_count": word_count
            }
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'content_extractor',
            'message': f'Failed to score content quality: {str(e)}',
            'data': {'url': url, 'error': str(e)}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.content_extractor.score_quality.errors'
        })
        
        raise ValueError(f"Failed to score content quality: {str(e)}") from e


async def content_extractor_clean_content(
    ctx: RunContext[Any],
    content: str,
    url: Optional[str] = None,
    content_type: str = "html",
    options: Optional[Dict[str, Any]] = None
) -> ContentExtractorOutput:
    """
    Clean and fix malformed HTML content.
    
    Args:
        ctx: Runtime context provided by the framework
        content: Raw HTML content to clean
        url: Source URL for context
        content_type: Type of content
        options: Additional cleaning options
        
    Returns:
        ContentExtractorOutput with cleaned content and issues fixed
    """
    injector = get_injector()
    
    try:
        opts = options or {}
        min_text_length = opts.get('min_text_length', 50)
        
        issues_fixed = []
        
        # Parse with error recovery
        soup = BeautifulSoup(content, 'lxml')
        
        # Fix common issues
        # Remove empty elements
        for elem in soup.find_all():
            if not elem.get_text(strip=True) and not elem.find_all(['img', 'br', 'hr', 'input']):
                elem.decompose()
                issues_fixed.append("empty_elements")
        
        # Fix nested elements of same type
        for tag in ['div', 'span', 'p']:
            while True:
                found_nested = False
                for elem in soup.find_all(tag):
                    parent = elem.parent
                    if parent and parent.name == tag and len(parent.contents) == 1:
                        # Unwrap nested element
                        parent.unwrap()
                        found_nested = True
                        break
                if not found_nested:
                    break
        
        if found_nested:
            issues_fixed.append("nested_elements")
        
        # Extract clean text
        clean_text = _clean_text(soup.get_text())
        
        # Filter by minimum length
        if len(clean_text) < min_text_length:
            issues_fixed.append("insufficient_content")
        
        word_count = len(clean_text.split())
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'content_extractor',
            'message': f'Successfully cleaned malformed HTML content',
            'data': {
                'issues_fixed': issues_fixed,
                'word_count': word_count,
                'url': url
            }
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.content_extractor.clean_content.success'
        })
        
        return ContentExtractorOutput(
            success=True,
            message="Successfully cleaned malformed HTML content",
            data={
                "content": clean_text,
                "issues_fixed": list(set(issues_fixed)),
                "word_count": word_count
            }
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'content_extractor',
            'message': f'Failed to clean content: {str(e)}',
            'data': {'url': url, 'error': str(e)}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.content_extractor.clean_content.errors'
        })
        
        raise ValueError(f"Failed to clean content: {str(e)}") from e


# Routing configuration
content_extractor_routing = RoutingConfig(
    operation_field='operation',
    operation_map={
        'parse_html': ('content_extractor_parse_html', lambda x: {
            'content': x.content,
            'url': x.url,
            'content_type': x.content_type,
            'options': x.options
        }),
        'extract_content': ('content_extractor_extract_content', lambda x: {
            'content': x.content,
            'url': x.url,
            'content_type': x.content_type,
            'options': x.options
        }),
        'extract_metadata': ('content_extractor_extract_metadata', lambda x: {
            'content': x.content,
            'url': x.url,
            'content_type': x.content_type,
            'options': x.options
        }),
        'extract_links': ('content_extractor_extract_links', lambda x: {
            'content': x.content,
            'url': x.url,
            'content_type': x.content_type,
            'options': x.options
        }),
        'extract_code': ('content_extractor_extract_code', lambda x: {
            'content': x.content,
            'url': x.url,
            'content_type': x.content_type,
            'options': x.options
        }),
        'score_quality': ('content_extractor_score_quality', lambda x: {
            'content': x.content,
            'url': x.url,
            'content_type': x.content_type,
            'options': x.options
        }),
        'clean_content': ('content_extractor_clean_content', lambda x: {
            'content': x.content,
            'url': x.url,
            'content_type': x.content_type,
            'options': x.options
        })
    }
)


def create_content_extractor_agent():
    """
    Create and return the content_extractor AgenTool.
    
    Returns:
        Agent configured for content extraction operations
    """
    return create_agentool(
        name='content_extractor',
        input_schema=ContentExtractorInput,
        routing_config=content_extractor_routing,
        tools=[
            content_extractor_parse_html,
            content_extractor_extract_content,
            content_extractor_extract_metadata,
            content_extractor_extract_links,
            content_extractor_extract_code,
            content_extractor_score_quality,
            content_extractor_clean_content
        ],
        output_type=ContentExtractorOutput,
        system_prompt="You are a specialized content extraction agent that parses HTML and text content to extract structured information. You intelligently remove boilerplate content, identify main content sections, extract metadata and links, parse code blocks, and assess content quality. You handle various content types including documentation, blogs, news articles, and API references with appropriate extraction strategies for each type.",
        description="Extracts and structures content from raw HTML, text, and web pages with intelligent parsing using operations: parse_html, extract_content, extract_metadata, extract_links, extract_code, score_quality, clean_content",
        version="1.0.0",
        tags=["content", "extraction", "html", "parsing", "beautifulsoup", "readability", "nlp"],
        dependencies=["llm", "logging", "metrics"],
        examples=[
            {
                "description": "Parse HTML content and extract main content",
                "input": {
                    "operation": "parse_html",
                    "content": "<html><head><title>Sample Page</title></head><body><nav>Navigation</nav><main><h1>Article Title</h1><p>Main content here.</p></main><footer>Footer</footer></body></html>",
                    "url": "https://example.com/article",
                    "content_type": "blog"
                },
                "output": {
                    "success": True,
                    "message": "Successfully parsed HTML content",
                    "data": {
                        "title": "Sample Page",
                        "content": "# Article Title\n\nMain content here.",
                        "sections": [
                            {"type": "heading", "level": 1, "text": "Article Title"},
                            {"type": "paragraph", "text": "Main content here."}
                        ],
                        "links": [],
                        "metadata": {"url": "https://example.com/article", "content_type": "blog"},
                        "word_count": 4,
                        "reading_time": 0.02
                    }
                }
            },
            {
                "description": "Extract and structure code blocks",
                "input": {
                    "operation": "extract_code",
                    "content": "<html><body><pre><code class=\"python\">def hello():\n    print('Hello World')</code></pre></body></html>",
                    "content_type": "documentation"
                },
                "output": {
                    "success": True,
                    "message": "Successfully extracted 1 code blocks",
                    "data": {
                        "code_blocks": [
                            {
                                "language": "python",
                                "code": "def hello():\n    print('Hello World')",
                                "line_count": 2
                            }
                        ]
                    }
                }
            },
            {
                "description": "Score content quality and relevance",
                "input": {
                    "operation": "score_quality",
                    "content": "<html><body><article><h1>Comprehensive Guide</h1><p>This is a detailed article with substantial content covering multiple aspects of the topic.</p></article></body></html>",
                    "content_type": "blog"
                },
                "output": {
                    "success": True,
                    "message": "Successfully scored content quality",
                    "data": {
                        "quality_score": 0.75,
                        "factors": {
                            "content_length": 0.8,
                            "structure_score": 0.7,
                            "readability": 0.75
                        },
                        "recommendations": ["Add more subheadings", "Include code examples"]
                    }
                }
            }
        ]
    )


# Create the agent instance
agent = create_content_extractor_agent()