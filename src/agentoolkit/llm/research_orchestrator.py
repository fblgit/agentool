"""
research_orchestrator AgenTool - Coordinates multi-source web research with intelligent query planning and content aggregation.

This AgenTool provides comprehensive web research capabilities that analyze research queries to determine optimal
search strategies, coordinate between browser automation and HTTP requests, and generate structured research reports.
It integrates with browser_manager, page_navigator, element_interactor, http, content_extractor, llm, storage_fs,
storage_kv, logging, and metrics to provide intelligent multi-source research orchestration.

Key Features:
- Research Planning: Analyze queries and determine optimal search strategies and content sources
- Multi-Source Execution: Coordinate browser automation for dynamic sites and HTTP requests for APIs
- Content Aggregation: Deduplicate and filter content with relevance scoring
- Session Management: Track research progress with intermediate result storage
- Report Generation: Generate comprehensive research reports with source attribution

Usage Example:
    >>> from agentoolkit.research_orchestrator import create_research_orchestrator_agent
    >>> from agentool.core.injector import get_injector
    >>> 
    >>> # Create and register the agent
    >>> agent = create_research_orchestrator_agent()
    >>> 
    >>> # Use through injector
    >>> injector = get_injector()
    >>> result = await injector.run('research_orchestrator', {
    ...     "operation": "plan_research",
    ...     "session_id": "research_001",
    ...     "query": "OpenAI GPT-4 API documentation",
    ...     "research_type": "documentation"
    ... })
"""

import json
import time
import uuid
import asyncio
from typing import Dict, Any, List, Optional, Literal, Union
from pydantic import BaseModel, Field, field_validator

from pydantic_ai import RunContext, Agent
from pydantic_ai.settings import ModelSettings

from agentool import create_agentool
from agentool.base import BaseOperationInput
from agentool.core.registry import RoutingConfig
from agentool.core.injector import get_injector

# Import research models
from .research_models import (
    ResearchPlan, ResearchFindings, ContentAggregation,
    QueryRefinement, ResearchProgress, ResearchReport, SessionData
)

# Model configuration for different operations
_MODEL_CONFIG = {
    '_default': {
        'model': 'openai:gpt-4o',
        'settings': ModelSettings(max_tokens=8192, temperature=0.7)
    },
    'plan_research': {
        'model': 'openai:gpt-4o',
        'settings': ModelSettings(max_tokens=2048, temperature=0.7)
    },
    'refine_query': {
        'model': 'openai:gpt-4o',
        'settings': ModelSettings(max_tokens=2048, temperature=0.7)
    },
    'aggregate_content': {
        'model': 'openai:gpt-4o',
        'settings': ModelSettings(max_tokens=3072, temperature=0.6)
    },
    'generate_report': {
        'model': 'openai:gpt-4o',
        'settings': ModelSettings(max_tokens=4096, temperature=0.6)
    },
    'extract_findings': {
        'model': 'openai:gpt-4o-mini',
        'settings': ModelSettings(max_tokens=1024, temperature=0.5)
    }
}

def get_model_config(operation: str) -> Dict[str, Any]:
    """Get model configuration for a specific operation."""
    return _MODEL_CONFIG.get(operation, _MODEL_CONFIG['_default'])


class DomainTracker:
    """Track domain access patterns and failures."""
    
    def __init__(self):
        self.domain_stats = {}  # domain -> {'attempts': n, 'failures': n, 'blocked': bool}
        self.blocked_domains = set()
        self.failure_threshold = 2  # Block domain after 2 failures
        
    def record_attempt(self, url: str, success: bool, content_size: int = 0, link_count: int = 0):
        """Record an access attempt for a domain."""
        domain = self.extract_domain(url)
        
        if domain not in self.domain_stats:
            self.domain_stats[domain] = {
                'attempts': 0,
                'failures': 0,
                'blocked': False,
                'avg_size': 0,
                'avg_links': 0
            }
        
        stats = self.domain_stats[domain]
        stats['attempts'] += 1
        
        if not success:
            stats['failures'] += 1
            
            # Block domain if too many failures
            if stats['failures'] >= self.failure_threshold:
                self.blocked_domains.add(domain)
                stats['blocked'] = True
        else:
            # Update averages for successful attempts
            if content_size > 0:
                stats['avg_size'] = (stats['avg_size'] * (stats['attempts'] - 1) + content_size) / stats['attempts']
            if link_count > 0:
                stats['avg_links'] = (stats['avg_links'] * (stats['attempts'] - 1) + link_count) / stats['attempts']
    
    def is_domain_blocked(self, url: str) -> bool:
        """Check if domain is blocked."""
        domain = self.extract_domain(url)
        return domain in self.blocked_domains
    
    def should_skip_domain(self, url: str) -> bool:
        """Check if we should skip this domain."""
        return self.is_domain_blocked(url)
    
    @staticmethod
    def extract_domain(url: str) -> str:
        """Extract domain from URL for tracking."""
        from urllib.parse import urlparse
        try:
            parsed = urlparse(url)
            # Get domain without subdomain for better blocking
            # e.g., "docs.openai.com" -> "openai.com"
            domain_parts = parsed.netloc.split('.')
            if len(domain_parts) > 2:
                # Keep last two parts (domain.tld)
                return '.'.join(domain_parts[-2:])
            return parsed.netloc
        except:
            return url


def analyze_page_structure(html: str, url: str) -> Dict[str, Any]:
    """
    Analyze page structure to detect if it's a blocked/captcha page.
    
    Returns dict with:
        - is_blocked: bool
        - reason: str
        - content_size: int
        - link_count: int
        - has_main_content: bool
    """
    if not html:
        return {
            'is_blocked': True,
            'reason': 'Empty content',
            'content_size': 0,
            'link_count': 0,
            'has_main_content': False
        }
    
    html_lower = html.lower()
    content_size = len(html.strip())
    
    # Count links (rough estimate)
    link_count = html_lower.count('<a ') + html_lower.count('<a>')
    
    # Check for main content indicators
    has_article = '<article' in html_lower
    has_main = '<main' in html_lower
    has_sections = html_lower.count('<section') > 1
    has_paragraphs = html_lower.count('<p>') > 3
    has_headers = html_lower.count('<h1') > 0 or html_lower.count('<h2') > 0
    
    has_main_content = any([has_article, has_main, has_sections, has_paragraphs, has_headers])
    
    # Structural patterns of blocked pages:
    # 1. Very small size (< 5KB) with few links
    # 2. Medium size (5-20KB) but no real content structure
    # 3. Specific blocking patterns in title/meta
    
    is_blocked = False
    reason = ''
    
    # Check page title for blocks
    title_match = html_lower.find('<title>')
    if title_match > -1:
        title_end = html_lower.find('</title>', title_match)
        if title_end > -1:
            title = html_lower[title_match+7:title_end]
            if any(x in title for x in ['just a moment', 'please wait', 'checking', 'attention required', '403', 'denied', 'blocked']):
                is_blocked = True
                reason = 'Blocking title detected'
    
    # Small page with very few links (typical of block pages)
    if not is_blocked and content_size < 5000 and link_count < 5:
        is_blocked = True
        reason = f'Small page ({content_size} bytes) with few links ({link_count})'
    
    # Medium page but no content structure
    if not is_blocked and content_size < 20000 and not has_main_content:
        is_blocked = True
        reason = f'No main content structure found ({content_size} bytes)'
    
    # Check for challenge/verification forms
    if not is_blocked and ('challenge-form' in html_lower or 'verification' in html_lower or 'cf-wrapper' in html_lower):
        # But only if it lacks main content
        if not has_main_content:
            is_blocked = True
            reason = 'Challenge/verification form without main content'
    
    return {
        'is_blocked': is_blocked,
        'reason': reason,
        'content_size': content_size,
        'link_count': link_count,
        'has_main_content': has_main_content
    }


class ResearchOrchestratorInput(BaseOperationInput):
    """Input schema for research_orchestrator operations.
    
    Supports comprehensive research orchestration with session management,
    intelligent query planning, and multi-source content aggregation.
    """
    operation: Literal['plan_research', 'execute_research', 'refine_query', 'aggregate_content', 'generate_report', 'track_progress', 'get_session'] = Field(
        description="Research operation to perform"
    )
    
    session_id: str = Field(
        description="Research session identifier for tracking progress and state"
    )
    
    # Fields for plan_research, refine_query
    query: Optional[str] = Field(
        None,
        description="Research query or question to investigate"
    )
    
    research_type: Optional[Literal['documentation', 'trend_analysis', 'comparative', 'fact_finding']] = Field(
        None,
        description="Type of research pattern to follow"
    )
    
    sources: Optional[List[str]] = Field(
        None,
        description="List of specific sources/URLs to research (optional)"
    )
    
    search_terms: Optional[List[str]] = Field(
        None,
        description="Additional search terms for query expansion"
    )
    
    max_sources: Optional[int] = Field(
        10,
        description="Maximum number of sources to research (default: 10)"
    )
    
    content_filter: Optional[Dict[str, Any]] = Field(
        None,
        description="Content filtering options for relevance and quality"
    )
    
    follow_up_query: Optional[str] = Field(
        None,
        description="Follow-up query based on initial findings (for refine_query operation)"
    )
    
    # Fields for generate_report
    report_format: Optional[Literal['markdown', 'structured_data', 'summary']] = Field(
        'markdown',
        description="Output format for research report"
    )
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v, info):
        """Validate query is provided for operations that require it."""
        operation = info.data.get('operation')
        if operation in ['plan_research', 'refine_query'] and not v:
            raise ValueError(f"query is required for {operation}")
        return v
    
    @field_validator('research_type')
    @classmethod
    def validate_research_type(cls, v, info):
        """Validate research_type for plan_research operation."""
        operation = info.data.get('operation')
        if operation == 'plan_research' and not v:
            raise ValueError("research_type is required for plan_research")
        return v
    
    @field_validator('follow_up_query')
    @classmethod
    def validate_follow_up_query(cls, v, info):
        """Validate follow_up_query for refine_query operation."""
        operation = info.data.get('operation')
        if operation == 'refine_query' and not v:
            raise ValueError("follow_up_query is required for refine_query")
        return v


class ResearchOrchestratorOutput(BaseModel):
    """Output schema for research_orchestrator operations."""
    success: bool = Field(description="Whether the research operation succeeded")
    message: str = Field(description="Human-readable result message")
    data: Optional[Dict[str, Any]] = Field(None, description="Operation-specific research data including plans, progress, content, and reports")


async def research_orchestrator_plan_research(
    ctx: RunContext[Any],
    session_id: str,
    query: str,
    research_type: str,
    sources: Optional[List[str]] = None,
    search_terms: Optional[List[str]] = None,
    max_sources: int = 10,
    content_filter: Optional[Dict[str, Any]] = None
) -> ResearchOrchestratorOutput:
    """
    Plan research strategy for a given query using structured output.
    
    Uses templates and Pydantic models for type-safe research planning.
    
    Args:
        ctx: Runtime context with model access
        session_id: Research session identifier for tracking
        query: Research query or question to investigate
        research_type: Type of research pattern (documentation, trend_analysis, comparative, fact_finding)
        sources: Optional list of specific sources/URLs to research
        search_terms: Optional additional search terms for query expansion
        max_sources: Maximum number of sources to research (default: 10)
        content_filter: Optional content filtering options
        
    Returns:
        ResearchOrchestratorOutput with research plan containing strategy, queries, and steps
    """
    injector = get_injector()
    
    try:
        # Validate required parameters
        if not query:
            raise ValueError("query is required for research planning")
        if not research_type:
            raise ValueError("research_type is required for research planning")
            
        # Load system template with schema
        template_result = await injector.run('templates', {
            'operation': 'render',
            'template_name': 'system/research_planner',
            'variables': {
                'schema_json': json.dumps(ResearchPlan.model_json_schema(), indent=2)
            }
        })
        
        if not template_result.success:
            raise RuntimeError(f"Failed to load system template: {template_result.message}")
            
        system_prompt = template_result.data.get('rendered', '')
        
        # Load user prompt template
        prompt_result = await injector.run('templates', {
            'operation': 'render',
            'template_name': 'prompts/research_planner',
            'variables': {
                'query': query,
                'research_type': research_type,
                'sources': sources or [],
                'search_terms': search_terms or [],
                'max_sources': max_sources,
                'content_filter': content_filter or {}
            }
        })
        
        if not prompt_result.success:
            raise RuntimeError(f"Failed to load prompt template: {prompt_result.message}")
            
        user_prompt = prompt_result.data.get('rendered', '')
        
        # Get model configuration for this operation
        config = get_model_config('plan_research')
        
        # Create Agent with structured output
        agent = Agent(
            config['model'],
            output_type=ResearchPlan,
            system_prompt=system_prompt,
            model_settings=config['settings']
        )
        
        # Generate research plan
        result = await agent.run(user_prompt)
        plan = result.output  # This is a ResearchPlan instance
        
        # Capture token usage
        usage = result.usage()
        
        # Track token metrics
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.research_orchestrator.tokens.total',
            'value': usage.total_tokens,
            'labels': {'operation': 'plan_research', 'model': config['model']}
        })
        
        # Create session data with the plan
        session_data = {
            'session_id': session_id,
            'query': query,
            'research_type': research_type,
            'max_sources': max_sources,
            'content_filter': content_filter or {},
            'plan': plan.model_dump(),  # Convert to dict for storage
            'sources': sources or [],  # Store the actual source URLs provided
            'status': 'planned',
            'created_at': time.time(),
            'sources_processed': 0,
            'content_collected': [],
            'progress_percentage': 0,
            'current_step': 'planning_complete'
        }
        
        # Store session data
        storage_result = await injector.run('storage_kv', {
            'operation': 'set',
            'key': f'research_session_{session_id}',
            'value': session_data,
            'namespace': 'research_orchestrator',
            'ttl': 7200  # 2 hours
        })
        
        if not storage_result.success:
            raise RuntimeError(f"Failed to store session: {storage_result.message}")
        
        # Log successful planning
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'research_orchestrator',
            'message': f'Research plan created for session {session_id}',
            'data': {
                'query': query, 
                'research_type': research_type, 
                'max_sources': max_sources,
                'queries_generated': len(plan.search_queries)
            }
        })
        
        # Track planning metric
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.research_orchestrator.plans_created.count',
            'labels': {'research_type': research_type}
        })
        
        return ResearchOrchestratorOutput(
            success=True,
            message=f"Successfully created research plan for session '{session_id}'",
            data={
                'plan_id': f'plan_{session_id}',
                'strategy': plan.strategy,
                'search_queries': plan.search_queries,
                'target_sources': plan.target_sources,
                'steps': plan.steps,
                'max_sources': plan.max_sources
            }
        )
        
    except Exception as e:
        # Log error
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'research_orchestrator',
            'message': f'Research planning failed for session {session_id}',
            'data': {'error': str(e), 'query': query}
        })
        
        # Track error metric
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.research_orchestrator.planning_errors.count'
        })
        
        raise


async def research_orchestrator_execute_research(
    ctx: RunContext[Any],
    session_id: str
) -> ResearchOrchestratorOutput:
    """
    Execute research plan with multi-source content collection.
    
    Coordinates browser automation for dynamic sites and HTTP requests for APIs,
    following the research plan to collect and process content from multiple sources.
    
    Args:
        ctx: Runtime context provided by the framework
        session_id: Research session identifier
        
    Returns:
        ResearchOrchestratorOutput with execution results and source processing status
        
    Raises:
        KeyError: If session not found
        ConnectionError: If browser or HTTP requests fail
        ContentExtractionError: If content extraction fails
    """
    injector = get_injector()
    
    try:
        # Get session data
        session_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': f'research_session_{session_id}',
            'namespace': 'research_orchestrator'
        })
        
        if not session_result.success:
            raise KeyError(f"Research session '{session_id}' not found")
        
        session_data = session_result.data.get('value', {})
        plan = session_data.get('plan', {})
        search_queries = plan.get('search_queries', [])
        max_sources = session_data.get('max_sources', 10)
        provided_sources = session_data.get('sources', [])  # Get the actual URLs if provided
        
        # Start browser for dynamic content
        browser_id = f"research_{session_id}"
        browser_result = await injector.run('browser_manager', {
            'operation': 'start_browser',
            'browser_id': browser_id,
            'options': {'headless': False}
        })
        
        if not browser_result.success:
            raise ConnectionError(f"Failed to start browser: {browser_result.message}")
        
        sources_processed = 0
        content_collected = []
        relevance_scores = []
        
        # Initialize domain tracker for this session
        domain_tracker = DomainTracker()
        
        # If we have provided sources, use them directly instead of searching
        if provided_sources:
            # Process the provided URLs directly
            for source_url in provided_sources[:max_sources]:
                try:
                    # Navigate directly to the provided URL
                    nav_result = await injector.run('page_navigator', {
                        'operation': 'navigate',
                        'browser_id': browser_id,
                        'url': source_url,
                        'timeout': 30000  # 30 seconds in milliseconds
                    })
                    
                    if nav_result.success:
                        # Get page content
                        content_result = await injector.run('page_navigator', {
                            'operation': 'get_content',
                            'browser_id': browser_id,
                            'extract_content': True
                        })
                        
                        if content_result.success:
                            raw_html = content_result.data.get('html', '')
                            
                            # Extract content from the page
                            # Use extract_documentation for documentation research type
                            research_type = session_data.get('research_type', 'html')
                            if research_type == 'documentation':
                                extract_result = await injector.run('content_extractor', {
                                    'operation': 'extract_documentation',
                                    'content': raw_html,
                                    'url': source_url,
                                    'content_type': 'documentation',
                                    'options': {
                                        'query': session_data.get('query', ''),
                                        'research_type': research_type
                                    }
                                })
                            else:
                                extract_result = await injector.run('content_extractor', {
                                    'operation': 'extract_content',
                                    'content': raw_html,
                                    'url': source_url,
                                    'content_type': research_type
                                })
                            
                            if extract_result.success:
                                # Score content quality
                                quality_result = await injector.run('content_extractor', {
                                    'operation': 'score_quality',
                                    'content': raw_html,
                                    'content_type': session_data.get('research_type', 'html')
                                })
                                
                                if quality_result.success:
                                    quality_score = quality_result.data.get('quality_score', 0)
                                    
                                    # Apply content filter
                                    content_filter = session_data.get('content_filter', {})
                                    relevance_threshold = content_filter.get('relevance_threshold', 0.3)
                                    
                                    if quality_score >= relevance_threshold:
                                        content_collected.append({
                                            'url': source_url,
                                            'content': extract_result.data,
                                            'quality_score': quality_score,
                                            'extracted_at': time.time()
                                        })
                                        relevance_scores.append(quality_score)
                                        sources_processed += 1
                    
                    # Rate limiting between page visits
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    await injector.run('logging', {
                        'operation': 'log',
                        'level': 'WARN',
                        'logger_name': 'research_orchestrator',
                        'message': f'Failed to process provided source {source_url}',
                        'data': {'error': str(e)}
                    })
        else:
            # No provided sources, use search queries
            for query in search_queries[:max_sources]:
                try:
                    # Use DuckDuckGo search engine (more automation-friendly than Google)
                    search_url = f"https://duckduckgo.com/?q={query.replace(' ', '+')}"
                    
                    # Navigate to search results (timeout in milliseconds)
                    nav_result = await injector.run('page_navigator', {
                        'operation': 'navigate',
                        'browser_id': browser_id,
                        'url': search_url,
                        'timeout': 30000  # 30 seconds in milliseconds
                    })
                    
                    if nav_result.success:
                        # Get page content
                        content_result = await injector.run('page_navigator', {
                            'operation': 'get_content',
                            'browser_id': browser_id,
                            'extract_content': True
                        })
                        
                        if content_result.success:
                            raw_html = content_result.data.get('html', '')
                            text_content = content_result.data.get('text_content', '')
                            
                            # For DuckDuckGo search results, extract links and visit them
                            # DuckDuckGo is more automation-friendly than Google
                            if 'duckduckgo.com' in search_url:
                                # Extract links from search results
                                links_result = await injector.run('content_extractor', {
                                    'operation': 'extract_links',
                                    'content': raw_html,
                                    'url': search_url
                                })
                                
                                if links_result.success:
                                    links = links_result.data.get('links', [])
                                    # Filter out DuckDuckGo's own links and ads
                                    # Links are dicts with 'url', 'text', 'title' etc.
                                    result_links = []
                                    for link in links[:20]:  # Check more links
                                        if isinstance(link, dict):
                                            url = link.get('url', '')
                                            # Skip DuckDuckGo internal links and non-http links
                                            if url and 'http' in url and not any(
                                                domain in url for domain in [
                                                    'duckduckgo.com', 'duck.com', 'duckduckgo.co',
                                                    'javascript:', 'mailto:', '#'
                                                ]
                                            ):
                                                result_links.append(url)
                                        elif isinstance(link, str) and 'http' in link:
                                            result_links.append(link)
                                    
                                    # Log found links
                                    await injector.run('logging', {
                                        'operation': 'log',
                                        'level': 'INFO',
                                        'logger_name': 'research_orchestrator',
                                        'message': f'Found {len(result_links)} external links from DuckDuckGo',
                                        'data': {'first_3_links': result_links[:3]}
                                    })
                                    
                                    # Visit actual result pages
                                    for result_url in result_links[:5]:  # Check up to 5 results
                                        if not result_url:  # Skip if URL is None or empty
                                            continue
                                        
                                        # Check if domain is already blocked
                                        if domain_tracker.should_skip_domain(result_url):
                                            await injector.run('logging', {
                                                'operation': 'log',
                                                'level': 'INFO',
                                                'logger_name': 'research_orchestrator',
                                                'message': f'Skipping blocked domain: {domain_tracker.extract_domain(result_url)}',
                                                'data': {'url': result_url}
                                            })
                                            continue
                                        
                                        try:
                                            # Navigate to actual content page
                                            actual_nav = await injector.run('page_navigator', {
                                                'operation': 'navigate',
                                                'browser_id': browser_id,
                                                'url': result_url,
                                                'timeout': 20000  # 20 seconds
                                            })
                                            
                                            if actual_nav.success:
                                                # Get actual page content
                                                actual_content = await injector.run('page_navigator', {
                                                    'operation': 'get_content',
                                                    'browser_id': browser_id,
                                                    'extract_content': True
                                                })
                                                
                                                if actual_content.success:
                                                    actual_html = actual_content.data.get('html', '')
                                                    
                                                    # Analyze page structure to detect blocks
                                                    page_analysis = analyze_page_structure(actual_html, result_url)
                                                    
                                                    if page_analysis['is_blocked']:
                                                        # Record failure for this domain
                                                        domain_tracker.record_attempt(
                                                            result_url, 
                                                            success=False,
                                                            content_size=page_analysis['content_size'],
                                                            link_count=page_analysis['link_count']
                                                        )
                                                        
                                                        await injector.run('logging', {
                                                            'operation': 'log',
                                                            'level': 'WARN',
                                                            'logger_name': 'research_orchestrator',
                                                            'message': f'Page blocked/captcha detected: {page_analysis["reason"]}',
                                                            'data': {
                                                                'url': result_url,
                                                                'domain': domain_tracker.extract_domain(result_url),
                                                                'content_size': page_analysis['content_size'],
                                                                'link_count': page_analysis['link_count']
                                                            }
                                                        })
                                                        continue
                                                    
                                                    # Extract content from actual page
                                                    # Use extract_documentation for documentation research type
                                                    research_type = session_data.get('research_type', 'html')
                                                    if research_type == 'documentation':
                                                        extract_result = await injector.run('content_extractor', {
                                                            'operation': 'extract_documentation',
                                                            'content': actual_html,
                                                            'url': result_url,
                                                            'content_type': 'documentation',
                                                            'options': {
                                                                'query': session_data.get('query', ''),
                                                                'research_type': research_type
                                                            }
                                                        })
                                                    else:
                                                        extract_result = await injector.run('content_extractor', {
                                                            'operation': 'extract_content',
                                                            'content': actual_html,
                                                            'url': result_url,
                                                            'content_type': research_type
                                                        })
                                                    
                                                    if extract_result.success:
                                                        # Record successful domain access
                                                        domain_tracker.record_attempt(
                                                            result_url,
                                                            success=True,
                                                            content_size=page_analysis['content_size'],
                                                            link_count=page_analysis['link_count']
                                                        )
                                                        
                                                        # Score actual content quality
                                                        quality_result = await injector.run('content_extractor', {
                                                            'operation': 'score_quality',
                                                            'content': actual_html,
                                                            'content_type': session_data.get('research_type', 'html')
                                                        })
                                                        
                                                        if not quality_result.success:
                                                            raise RuntimeError(f"Failed to score content quality: {quality_result.message}")
                                                        quality_score = quality_result.data.get('quality_score')
                                                        if quality_score is None:
                                                            raise ValueError("Quality score not returned")
                                                        
                                                        # Apply content filter
                                                        content_filter = session_data.get('content_filter', {})
                                                        relevance_threshold = content_filter.get('relevance_threshold', 0.3)
                                                        
                                                        if quality_score >= relevance_threshold:
                                                            content_collected.append({
                                                                'url': result_url,
                                                                'content': extract_result.data,
                                                                'quality_score': quality_score,
                                                                'extracted_at': time.time()
                                                            })
                                                            relevance_scores.append(quality_score)
                                                            sources_processed += 1
                                                            
                                                            # Stop if we have enough good sources
                                                            if sources_processed >= 2:
                                                                break
                                                        
                                                        # Rate limiting between actual page visits
                                                        await asyncio.sleep(1)
                                        except Exception as link_error:
                                            await injector.run('logging', {
                                                'operation': 'log',
                                                'level': 'WARN',
                                                'logger_name': 'research_orchestrator',
                                                'message': f'Failed to process result link in session {session_id}',
                                                'data': {'error': str(link_error), 'url': result_url}
                                            })
                            else:
                                # For direct URLs, extract content normally
                                # Use extract_documentation for documentation research type
                                research_type = session_data.get('research_type', 'html')
                                if research_type == 'documentation':
                                    extract_result = await injector.run('content_extractor', {
                                        'operation': 'extract_documentation',
                                        'content': raw_html,
                                        'url': search_url,
                                        'content_type': 'documentation',
                                        'options': {
                                            'query': session_data.get('query', ''),
                                            'research_type': research_type
                                        }
                                    })
                                else:
                                    extract_result = await injector.run('content_extractor', {
                                        'operation': 'extract_content',
                                        'content': raw_html,
                                        'url': search_url,
                                        'content_type': research_type
                                    })
                                
                                if is_success:
                                    extracted_content = extract_data.get('data') if isinstance(extract_data, dict) else extract_data.data
                                    
                                    # Score content quality
                                    quality_result = await injector.run('content_extractor', {
                                        'operation': 'score_quality',
                                        'content': raw_html,
                                        'content_type': session_data.get('research_type', 'html')
                                    })
                                    
                                    if not quality_result.success:
                                        raise RuntimeError(f"Failed to score content quality: {quality_result.message}")
                                    quality_score = quality_result.data.get('quality_score')
                                    if quality_score is None:
                                        raise ValueError("Quality score not returned")
                                    
                                    # Apply content filter
                                    content_filter = session_data.get('content_filter', {})
                                    relevance_threshold = content_filter.get('relevance_threshold', 0.3)
                                    
                                    if quality_score >= relevance_threshold:
                                        content_collected.append({
                                            'url': search_url,
                                            'content': extracted_content,
                                            'quality_score': quality_score,
                                            'extracted_at': time.time()
                                        })
                                        relevance_scores.append(quality_score)
                                        sources_processed += 1
                    
                    # Rate limiting
                    await asyncio.sleep(1)
                    
                except Exception as source_error:
                    await injector.run('logging', {
                        'operation': 'log',
                        'level': 'WARN',
                        'logger_name': 'research_orchestrator',
                        'message': f'Failed to process source in session {session_id}',
                        'data': {'error': str(source_error), 'query': query}
                    })
        
        # Clean up browser
        await injector.run('browser_manager', {
            'operation': 'stop_browser',
            'browser_id': browser_id
        })
        
        # Update session with results
        session_data.update({
            'status': 'executed',
            'sources_processed': sources_processed,
            'content_collected': content_collected,
            'relevance_scores': relevance_scores,
            'progress_percentage': 60,  # Execution complete, aggregation pending
            'current_step': 'content_aggregation',
            'executed_at': time.time()
        })
        
        # Store updated session
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': f'research_session_{session_id}',
            'value': session_data,
            'namespace': 'research_orchestrator',
            'ttl': 7200
        })
        
        # Cache content to filesystem
        if content_collected:
            content_cache = {
                'session_id': session_id,
                'content': content_collected,
                'cached_at': time.time()
            }
            
            await injector.run('storage_fs', {
                'operation': 'write',
                'path': f'research/cache/{session_id}_content.json',
                'content': json.dumps(content_cache, indent=2),
                'create_parents': True
            })
        
        # Log domain statistics
        if domain_tracker.domain_stats:
            await injector.run('logging', {
                'operation': 'log',
                'level': 'INFO',
                'logger_name': 'research_orchestrator',
                'message': f'Domain access statistics for session {session_id}',
                'data': {
                    'blocked_domains': list(domain_tracker.blocked_domains),
                    'domain_stats': domain_tracker.domain_stats
                }
            })
        
        # Log successful execution
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'research_orchestrator',
            'message': f'Research executed for session {session_id}',
            'data': {'sources_processed': sources_processed, 'content_items': len(content_collected)}
        })
        
        # Track execution metrics
        await injector.run('metrics', {
            'operation': 'observe',
            'name': 'agentool.research_orchestrator.sources_processed.histogram',
            'value': sources_processed
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.research_orchestrator.research_executed.count'
        })
        
        return ResearchOrchestratorOutput(
            success=True,
            message=f"Successfully executed research for session '{session_id}', found {sources_processed} relevant sources",
            data={
                'sources_processed': sources_processed,
                'content_extracted': len(content_collected) > 0,
                'deduplication_performed': True,  # Would implement deduplication logic
                'relevance_scores': relevance_scores,
                'next_step': 'aggregate_content'
            }
        )
        
    except Exception as e:
        # Cleanup browser on error
        try:
            await injector.run('browser_manager', {
                'operation': 'stop_browser',
                'browser_id': f"research_{session_id}"
            })
        except:
            pass
        
        # Log error
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'research_orchestrator',
            'message': f'Research execution failed for session {session_id}',
            'data': {'error': str(e)}
        })
        
        # Track error metric
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.research_orchestrator.execution_errors.count'
        })
        
        raise


async def research_orchestrator_refine_query(
    ctx: RunContext[Any],
    session_id: str,
    follow_up_query: str
) -> ResearchOrchestratorOutput:
    """
    Refine research query based on initial findings using structured output.
    
    Uses templates and Pydantic models for type-safe query refinement.
    
    Args:
        ctx: Runtime context with model access
        session_id: Research session identifier
        follow_up_query: Follow-up query based on initial findings
        
    Returns:
        ResearchOrchestratorOutput with refined research strategy
    """
    injector = get_injector()
    
    try:
        # Get session data
        session_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': f'research_session_{session_id}',
            'namespace': 'research_orchestrator'
        })
        
        if not session_result.success:
            raise KeyError(f"Research session '{session_id}' not found")
        
        session_data = session_result.data.get('value', {})
        original_query = session_data.get('query', '')
        content_collected = session_data.get('content_collected', [])
        
        # Ensure follow_up_query is provided
        if not follow_up_query:
            raise ValueError("follow_up_query is required for refine_query")
        
        # Prepare findings summary
        findings_summary = ""
        knowledge_gaps = []
        if content_collected:
            content_texts = [item.get('content', {}).get('content', '') for item in content_collected[:3]]
            findings_summary = '\n'.join(content_texts)[:1000]  # Limit size
        else:
            findings_summary = "No content collected yet."
            knowledge_gaps = ["Initial research not yet executed"]
        
        # Load system template with schema
        template_result = await injector.run('templates', {
            'operation': 'render',
            'template_name': 'system/query_refiner',
            'variables': {
                'schema_json': json.dumps(QueryRefinement.model_json_schema(), indent=2)
            }
        })
        
        if not template_result.success:
            raise RuntimeError(f"Failed to load system template: {template_result.message}")
            
        system_prompt = template_result.data.get('rendered', '')
        
        # Load user prompt template
        prompt_result = await injector.run('templates', {
            'operation': 'render',
            'template_name': 'prompts/query_refiner',
            'variables': {
                'session_id': session_id,
                'original_query': original_query,
                'follow_up_query': follow_up_query,
                'findings_summary': findings_summary,
                'knowledge_gaps': knowledge_gaps
            }
        })
        
        if not prompt_result.success:
            raise RuntimeError(f"Failed to load prompt template: {prompt_result.message}")
            
        user_prompt = prompt_result.data.get('rendered', '')
        
        # Get model configuration for this operation
        config = get_model_config('refine_query')
        
        # Create Agent with structured output
        agent = Agent(
            config['model'],
            output_type=QueryRefinement,
            system_prompt=system_prompt,
            model_settings=config['settings']
        )
        
        # Generate query refinement
        result = await agent.run(user_prompt)
        refinement = result.output  # This is a QueryRefinement instance
        
        # Capture token usage
        usage = result.usage()
        
        # Track token metrics
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.research_orchestrator.tokens.total',
            'value': usage.total_tokens,
            'labels': {'operation': 'refine_query', 'model': config['model']}
        })
        
        # Update session with refinement
        session_data.update({
            'follow_up_query': follow_up_query,
            'refinement': refinement.model_dump(),  # Convert to dict for storage
            'refined_at': time.time(),
            'status': 'refined'
        })
        
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': f'research_session_{session_id}',
            'value': session_data,
            'namespace': 'research_orchestrator',
            'ttl': 7200
        })
        
        # Log refinement
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'research_orchestrator',
            'message': f'Query refined for session {session_id}',
            'data': {
                'follow_up_query': follow_up_query,
                'refined_queries_count': len(refinement.refined_queries),
                'gaps_identified': len(refinement.knowledge_gaps)
            }
        })
        
        # Track refinement metric
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.research_orchestrator.queries_refined.count'
        })
        
        return ResearchOrchestratorOutput(
            success=True,
            message=f"Successfully refined research query for session '{session_id}'",
            data={
                'refined_queries': refinement.refined_queries,
                'knowledge_gaps': refinement.knowledge_gaps,
                'suggested_sources': refinement.suggested_sources,
                'refinement_rationale': refinement.refinement_rationale,
                'priority_topics': refinement.priority_topics
            }
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'research_orchestrator',
            'message': f'Query refinement failed for session {session_id}',
            'data': {'error': str(e)}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.research_orchestrator.refinement_errors.count'
        })
        
        raise


async def research_orchestrator_aggregate_content(
    ctx: RunContext[Any],
    session_id: str
) -> ResearchOrchestratorOutput:
    """
    Aggregate and deduplicate collected content.
    
    Processes all collected content to remove duplicates, assess relevance,
    and create structured aggregated results for report generation.
    
    Args:
        ctx: Runtime context provided by the framework
        session_id: Research session identifier
        
    Returns:
        ResearchOrchestratorOutput with aggregated content results
        
    Raises:
        KeyError: If session not found
        DuplicationError: If content deduplication fails
    """
    injector = get_injector()
    
    try:
        # Get session data
        session_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': f'research_session_{session_id}',
            'namespace': 'research_orchestrator'
        })
        
        if not session_result.success:
            raise KeyError(f"Research session '{session_id}' not found")
        
        session_data = session_result.data.get('value', {})
        content_collected = session_data.get('content_collected', [])
        
        if not content_collected:
            return ResearchOrchestratorOutput(
                success=True,
                message=f"No content to aggregate for session '{session_id}'",
                data={'aggregated_count': 0, 'duplicates_removed': 0}
            )
        
        # Deduplicate content based on similarity
        unique_content = []
        duplicates_removed = 0
        
        for item in content_collected:
            content_text = item.get('content', {}).get('content', '')
            is_duplicate = False
            
            # Simple deduplication by content similarity
            for existing in unique_content:
                existing_text = existing.get('content', {}).get('content', '')
                if content_text and existing_text:
                    # Simple similarity check (would use more sophisticated algorithm in production)
                    similarity = len(set(content_text.split()) & set(existing_text.split())) / max(len(set(content_text.split())), 1)
                    if similarity > 0.8:  # 80% similarity threshold
                        is_duplicate = True
                        duplicates_removed += 1
                        break
            
            if not is_duplicate:
                unique_content.append(item)
        
        # Sort by quality score
        unique_content.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        
        # Generate content summary using LLM
        content_texts = [item.get('content', {}).get('content', '')[:500] for item in unique_content[:5]]
        combined_content = '\n\n'.join(content_texts)
        
        summary_result = await injector.run('llm', {
            'operation': 'summary',
            'content': combined_content,
            'options': {'max_length': 200},
            'model': 'gpt-4o'
        })
        
        content_summary = "Research findings aggregated from multiple sources."
        if summary_result.success:
            content_summary = summary_result.data.get('summary', content_summary)
        
        # Create aggregated data structure
        aggregated_data = {
            'unique_sources': len(unique_content),
            'duplicates_removed': duplicates_removed,
            'content_items': unique_content,
            'summary': content_summary,
            'aggregated_at': time.time(),
            'quality_scores': [item.get('quality_score', 0) for item in unique_content]
        }
        
        # Update session
        session_data.update({
            'aggregated_content': aggregated_data,
            'status': 'aggregated',
            'progress_percentage': 85,
            'current_step': 'report_generation'
        })
        
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': f'research_session_{session_id}',
            'value': session_data,
            'namespace': 'research_orchestrator',
            'ttl': 7200
        })
        
        # Log aggregation
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'research_orchestrator',
            'message': f'Content aggregated for session {session_id}',
            'data': {
                'unique_sources': len(unique_content),
                'duplicates_removed': duplicates_removed
            }
        })
        
        # Track aggregation metrics
        await injector.run('metrics', {
            'operation': 'observe',
            'name': 'agentool.research_orchestrator.content_aggregated.histogram',
            'value': len(unique_content)
        })
        
        return ResearchOrchestratorOutput(
            success=True,
            message=f"Successfully aggregated content for session '{session_id}'",
            data={
                'unique_sources': len(unique_content),
                'duplicates_removed': duplicates_removed,
                'summary': content_summary,
                'quality_scores': aggregated_data['quality_scores']
            }
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'research_orchestrator',
            'message': f'Content aggregation failed for session {session_id}',
            'data': {'error': str(e)}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.research_orchestrator.aggregation_errors.count'
        })
        
        raise


async def research_orchestrator_generate_report(
    ctx: RunContext[Any],
    session_id: str,
    report_format: str = 'markdown'
) -> ResearchOrchestratorOutput:
    """
    Generate comprehensive research report.
    
    Creates structured research reports with source attribution, confidence scoring,
    and key findings in the specified format.
    
    Args:
        ctx: Runtime context provided by the framework
        session_id: Research session identifier
        report_format: Output format ('markdown', 'structured_data', 'summary')
        
    Returns:
        ResearchOrchestratorOutput with generated report details
        
    Raises:
        KeyError: If session not found
        LLMError: If report generation fails
        StorageError: If report saving fails
    """
    injector = get_injector()
    
    try:
        # Get session data
        session_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': f'research_session_{session_id}',
            'namespace': 'research_orchestrator'
        })
        
        if not session_result.success:
            raise KeyError(f"Research session '{session_id}' not found")
        
        session_data = session_result.data.get('value', {})
        aggregated_content = session_data.get('aggregated_content', {})
        query = session_data.get('query', 'Research Query')
        research_type = session_data.get('research_type', 'general')
        
        if not aggregated_content:
            raise RuntimeError(f"No aggregated content available for session '{session_id}'. Run aggregate_content first.")
        
        # Prepare report data
        content_items = aggregated_content.get('content_items', [])
        summary = aggregated_content.get('summary', '')
        quality_scores = aggregated_content.get('quality_scores', [])
        
        # For documentation type, check if we have extracted documentation
        if research_type == 'documentation':
            # Get the raw content collected during execution
            content_collected = session_data.get('content_collected', [])
            # Check if we have documentation extracts
            has_documentation = any(
                'sections' in item.get('content', {}).get('data', {})
                for item in content_collected
            )
            if has_documentation:
                # Use documentation extracts directly
                content_items = content_collected
        
        # Calculate confidence score
        confidence_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
        
        # Load system template with schema for report generation
        template_result = await injector.run('templates', {
            'operation': 'render',
            'template_name': 'system/report_generator',
            'variables': {
                'schema_json': json.dumps(ResearchReport.model_json_schema(), indent=2)
            }
        })
        
        if not template_result.success:
            raise RuntimeError(f"Failed to load system template: {template_result.message}")
            
        system_prompt = template_result.data.get('rendered', '')
        
        # Prepare findings for template
        plan_summary = session_data.get('plan', {})
        findings_summary = {
            'content_items': content_items[:10],  # Limit to top 10 items
            'summary': summary,
            'quality_scores': quality_scores,
            'aggregated_data': aggregated_content
        }
        
        # Load user prompt template
        prompt_result = await injector.run('templates', {
            'operation': 'render',
            'template_name': 'prompts/report_generator',
            'variables': {
                'session_id': session_id,
                'query': query,
                'research_type': research_type,
                'plan': json.dumps(plan_summary, indent=2),
                'findings': json.dumps(findings_summary, indent=2),
                'report_format': report_format
            }
        })
        
        if not prompt_result.success:
            raise RuntimeError(f"Failed to load prompt template: {prompt_result.message}")
            
        user_prompt = prompt_result.data.get('rendered', '')
        
        # Get model configuration for this operation
        config = get_model_config('generate_report')
        
        # Create Agent with structured output
        agent = Agent(
            config['model'],
            output_type=ResearchReport,
            system_prompt=system_prompt,
            model_settings=config['settings']
        )
        
        # Generate research report
        result = await agent.run(user_prompt)
        report = result.output  # This is a ResearchReport instance
        
        # Capture token usage
        usage = result.usage()
        
        # Track token metrics
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.research_orchestrator.tokens.total',
            'value': usage.total_tokens,
            'labels': {'operation': 'generate_report', 'model': config['model']}
        })
        
        # Calculate confidence if not provided
        if report.confidence_score == 0:
            report.confidence_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
        
        key_findings = report.key_findings
        
        # Generate report based on format
        if report_format == 'markdown':
            # Build markdown from structured report
            markdown_content = f"""# {report.title}

## Executive Summary
{report.executive_summary}

## Key Findings
{chr(10).join(['- ' + finding for finding in report.key_findings])}

## Detailed Sections
"""
            # For documentation type, also include extracted documentation sections
            if research_type == 'documentation' and content_items:
                markdown_content += "\n## Extracted Documentation\n"
                for item_idx, item in enumerate(content_items):
                    if isinstance(item, dict) and 'content' in item:
                        content_data = item['content']
                        if isinstance(content_data, dict) and 'data' in content_data:
                            extract_data = content_data['data']
                            if isinstance(extract_data, dict) and 'sections' in extract_data:
                                # We have documentation sections
                                url = item.get('url', 'Unknown Source')
                                markdown_content += f"\n### Source: {url}\n"
                                
                                for section in extract_data.get('sections', []):
                                    if isinstance(section, dict):
                                        heading = section.get('heading', 'Section')
                                        content = section.get('content', '')
                                        markdown_content += f"\n#### {heading}\n"
                                        markdown_content += f"{content}\n"
                                        
                                        # Add code blocks if present
                                        code_blocks = section.get('code_blocks', [])
                                        if code_blocks:
                                            for code in code_blocks:
                                                markdown_content += f"\n{code}\n"
                markdown_content += "\n## Analysis\n"
            
            # Add detailed sections from LLM analysis
            for section in report.detailed_sections:
                markdown_content += f"\n### {section.get('header', 'Section')}\n"
                markdown_content += f"{section.get('content', '')}\n"
            
            markdown_content += f"""
## Methodology
{report.methodology}

## Sources
{chr(10).join([f"- [{source.get('title', 'Source')}]({source.get('url', '#')})" for source in report.sources])}

## Limitations
{chr(10).join(['- ' + limitation for limitation in report.limitations])}

## Recommendations
{chr(10).join(['- ' + rec for rec in report.recommendations])}

## Metadata
- **Research Type**: {research_type}
- **Sources Analyzed**: {report.source_count}
- **Confidence Score**: {report.confidence_score:.2f}
- **Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            final_content = markdown_content
            
            # Save report to filesystem (use relative path for testing compatibility)
            report_path = f'research/reports/{session_id}_report.md'
            save_result = await injector.run('storage_fs', {
                'operation': 'write',
                'path': report_path,
                'content': final_content,
                'create_parents': True
            })
            
            if not save_result.success:
                raise RuntimeError(f"Failed to save report: {save_result.message}")
            
            # Update report with saved path
            report.report_path = report_path
            
            report_data = {
                'report': final_content,  # Include the actual report content
                'report_path': report_path,
                'title': report.title,
                'summary': report.executive_summary,
                'source_count': report.source_count,
                'confidence_score': report.confidence_score,
                'key_findings': report.key_findings
            }
            
        elif report_format == 'structured_data':
            # Return full structured report data
            report_data = report.model_dump()
            report_data['session_id'] = session_id
            report_data['generated_at'] = time.time()
            report_data['report'] = report_data.get('executive_summary', '')  # Include summary as report
            
        else:  # summary format
            report_data = {
                'report': report.executive_summary,  # Include summary as report
                'title': report.title,
                'summary': report.executive_summary,
                'key_findings': report.key_findings[:3],  # Top 3 findings
                'confidence_score': report.confidence_score,
                'source_count': report.source_count,
                'recommendations': report.recommendations[:2] if report.recommendations else []
            }
        
        # Update session status
        session_data.update({
            'status': 'completed',
            'progress_percentage': 100,
            'current_step': 'report_generated',
            'report': report_data,
            'completed_at': time.time()
        })
        
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': f'research_session_{session_id}',
            'value': session_data,
            'namespace': 'research_orchestrator',
            'ttl': 7200
        })
        
        # Log report generation
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'research_orchestrator',
            'message': f'Research report generated for session {session_id}',
            'data': {
                'format': report_format,
                'source_count': len(content_items),
                'confidence_score': confidence_score
            }
        })
        
        # Track report generation
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.research_orchestrator.reports_generated.count',
            'labels': {'format': report_format}
        })
        
        return ResearchOrchestratorOutput(
            success=True,
            message=f"Successfully generated research report for session '{session_id}'",
            data=report_data
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'research_orchestrator',
            'message': f'Report generation failed for session {session_id}',
            'data': {'error': str(e)}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.research_orchestrator.report_errors.count'
        })
        
        raise


async def research_orchestrator_track_progress(
    ctx: RunContext[Any],
    session_id: str
) -> ResearchOrchestratorOutput:
    """
    Track research session progress.
    
    Provides detailed progress information including completion percentage,
    current step, and processing statistics.
    
    Args:
        ctx: Runtime context provided by the framework
        session_id: Research session identifier
        
    Returns:
        ResearchOrchestratorOutput with progress information
        
    Raises:
        KeyError: If session not found
    """
    injector = get_injector()
    
    try:
        # Get session data
        session_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': f'research_session_{session_id}',
            'namespace': 'research_orchestrator'
        })
        
        if not session_result.success:
            raise KeyError(f"Research session '{session_id}' not found")
        
        session_data = session_result.data.get('value', {})
        
        # Determine completed and remaining steps
        status = session_data.get('status', 'unknown')
        current_step = session_data.get('current_step', 'initialization')
        progress_percentage = session_data.get('progress_percentage', 0)
        
        # Map status to completed steps
        status_steps = {
            'planned': ['plan_research'],
            'executed': ['plan_research', 'execute_research'],
            'refined': ['plan_research', 'execute_research', 'refine_query'],
            'aggregated': ['plan_research', 'execute_research', 'aggregate_content'],
            'completed': ['plan_research', 'execute_research', 'aggregate_content', 'generate_report']
        }
        
        completed_steps = status_steps.get(status, [])
        all_steps = ['plan_research', 'execute_research', 'aggregate_content', 'generate_report']
        remaining_steps = [step for step in all_steps if step not in completed_steps]
        
        sources_processed = session_data.get('sources_processed', 0)
        max_sources = session_data.get('max_sources', 10)
        
        progress_data = {
            'progress_percentage': progress_percentage,
            'current_step': current_step,
            'completed_steps': completed_steps,
            'remaining_steps': remaining_steps,
            'sources_processed': sources_processed,
            'total_sources': max_sources,
            'status': status,
            'created_at': session_data.get('created_at'),
            'last_updated': session_data.get('executed_at') or session_data.get('created_at')
        }
        
        # Log progress check
        await injector.run('logging', {
            'operation': 'log',
            'level': 'DEBUG',
            'logger_name': 'research_orchestrator',
            'message': f'Progress tracked for session {session_id}',
            'data': {'progress': progress_percentage, 'status': status}
        })
        
        return ResearchOrchestratorOutput(
            success=True,
            message=f"Research session '{session_id}' is {progress_percentage}% complete",
            data=progress_data
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'research_orchestrator',
            'message': f'Progress tracking failed for session {session_id}',
            'data': {'error': str(e)}
        })
        
        raise


async def research_orchestrator_get_session(
    ctx: RunContext[Any],
    session_id: str
) -> ResearchOrchestratorOutput:
    """
    Get complete research session data.
    
    Retrieves all session information including plan, progress, content, and results.
    
    Args:
        ctx: Runtime context provided by the framework
        session_id: Research session identifier
        
    Returns:
        ResearchOrchestratorOutput with complete session data
        
    Raises:
        KeyError: If session not found
    """
    injector = get_injector()
    
    try:
        # Get session data
        session_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': f'research_session_{session_id}',
            'namespace': 'research_orchestrator'
        })
        
        if not session_result.success:
            raise KeyError(f"Research session '{session_id}' not found")
        
        session_data = session_result.data.get('value', {})
        
        # Remove large content arrays for overview (optional)
        overview_data = dict(session_data)
        if 'content_collected' in overview_data and len(overview_data['content_collected']) > 5:
            overview_data['content_collected'] = overview_data['content_collected'][:5]  # First 5 items only
            overview_data['content_truncated'] = True
        
        # Log session access
        await injector.run('logging', {
            'operation': 'log',
            'level': 'DEBUG',
            'logger_name': 'research_orchestrator',
            'message': f'Session data retrieved for {session_id}',
            'data': {'status': session_data.get('status')}
        })
        
        return ResearchOrchestratorOutput(
            success=True,
            message=f"Retrieved research session '{session_id}'",
            data=overview_data
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'research_orchestrator',
            'message': f'Session retrieval failed for {session_id}',
            'data': {'error': str(e)}
        })
        
        raise


# Routing configuration
research_orchestrator_routing = RoutingConfig(
    operation_field='operation',
    operation_map={
        'plan_research': ('research_orchestrator_plan_research', lambda x: {
            'session_id': x.session_id,
            'query': x.query,
            'research_type': x.research_type,
            'sources': x.sources,
            'search_terms': x.search_terms,
            'max_sources': x.max_sources or 10,
            'content_filter': x.content_filter
        }),
        'execute_research': ('research_orchestrator_execute_research', lambda x: {
            'session_id': x.session_id
        }),
        'refine_query': ('research_orchestrator_refine_query', lambda x: {
            'session_id': x.session_id,
            'follow_up_query': x.follow_up_query
        }),
        'aggregate_content': ('research_orchestrator_aggregate_content', lambda x: {
            'session_id': x.session_id
        }),
        'generate_report': ('research_orchestrator_generate_report', lambda x: {
            'session_id': x.session_id,
            'report_format': x.report_format or 'markdown'
        }),
        'track_progress': ('research_orchestrator_track_progress', lambda x: {
            'session_id': x.session_id
        }),
        'get_session': ('research_orchestrator_get_session', lambda x: {
            'session_id': x.session_id
        }),
    }
)


def create_research_orchestrator_agent():
    """
    Create and return the research_orchestrator AgenTool.
    
    Returns:
        Agent configured for research orchestration operations
    """
    return create_agentool(
        name='research_orchestrator',
        input_schema=ResearchOrchestratorInput,
        routing_config=research_orchestrator_routing,
        tools=[
            research_orchestrator_plan_research,
            research_orchestrator_execute_research,
            research_orchestrator_refine_query,
            research_orchestrator_aggregate_content,
            research_orchestrator_generate_report,
            research_orchestrator_track_progress,
            research_orchestrator_get_session
        ],
        output_type=ResearchOrchestratorOutput,
        system_prompt="You are a research orchestrator that coordinates multi-source web research with intelligent query planning and content aggregation. You analyze research queries, execute comprehensive searches using browser automation and HTTP requests, aggregate content with deduplication and relevance filtering, and generate structured research reports with source attribution.",
        description="Coordinates multi-source web research with operations: plan_research, execute_research, refine_query, aggregate_content, generate_report, track_progress, get_session",
        version="1.0.0",
        tags=["research", "orchestration", "web-scraping", "content-analysis", "automation"],
        dependencies=["browser_manager", "page_navigator", "element_interactor", "http", "content_extractor", "llm", "storage_fs", "storage_kv", "logging", "metrics"],
        examples=[
            {
                "description": "Plan research for API documentation lookup",
                "input": {
                    "operation": "plan_research",
                    "session_id": "research_001",
                    "query": "How to use the OpenAI GPT-4 API for text generation",
                    "research_type": "documentation",
                    "max_sources": 5
                },
                "output": {
                    "success": True,
                    "message": "Successfully created research plan for session 'research_001'",
                    "data": {
                        "plan_id": "plan_001",
                        "strategy": "documentation_focused",
                        "search_queries": ["OpenAI GPT-4 API documentation", "GPT-4 text generation examples"],
                        "target_sources": ["https://platform.openai.com/docs", "official documentation sites"],
                        "estimated_duration": 300,
                        "steps": ["search_official_docs", "extract_api_examples", "verify_current_version"]
                    }
                }
            },
            {
                "description": "Execute research plan with browser automation",
                "input": {
                    "operation": "execute_research",
                    "session_id": "research_001"
                },
                "output": {
                    "success": True,
                    "message": "Successfully executed research for session 'research_001', found 4 relevant sources",
                    "data": {
                        "sources_processed": 4,
                        "content_extracted": True,
                        "deduplication_performed": True,
                        "relevance_scores": [0.95, 0.87, 0.82, 0.76],
                        "next_step": "aggregate_content"
                    }
                }
            },
            {
                "description": "Generate comprehensive research report",
                "input": {
                    "operation": "generate_report",
                    "session_id": "research_001",
                    "report_format": "markdown"
                },
                "output": {
                    "success": True,
                    "message": "Successfully generated research report for session 'research_001'",
                    "data": {
                        "report_path": "/research/reports/research_001_report.md",
                        "summary": "OpenAI GPT-4 API provides text generation via REST endpoints...",
                        "source_count": 4,
                        "confidence_score": 0.89,
                        "key_findings": ["API requires authentication token", "Rate limits apply", "Multiple models available"]
                    }
                }
            }
        ]
    )


# Create the agent instance
agent = create_research_orchestrator_agent()
