"""
Research Conductor AgenTool - High-level interface for comprehensive web research.

This AgenTool provides a simplified interface for conducting research, requiring only
a topic and research type. It orchestrates search term generation, search execution,
URL filtering, and research planning to produce comprehensive research reports.

Key Features:
- Search Term Generation: Automatically generate optimized search queries
- URL Filtering: Index-based prioritization for efficient processing  
- Research Planning: Coordinate research execution with research_orchestrator
- Single Entry Point: Simple interface requiring only topic and research type

Usage Example:
    >>> from agentoolkit.llm.research_conductor import create_research_conductor_agent
    >>> from agentool.core.injector import get_injector
    >>> 
    >>> # Create and register the agent
    >>> agent = create_research_conductor_agent()
    >>> 
    >>> # Use through injector
    >>> injector = get_injector()
    >>> result = await injector.run('research_conductor', {
    ...     "operation": "conduct_research",
    ...     "topic": "OpenAI o1 model capabilities",
    ...     "research_type": "documentation"
    ... })
"""

import json
import time
import uuid
import asyncio
from typing import Dict, Any, List, Optional, Literal, Union
from pydantic import BaseModel, Field, field_validator
from urllib.parse import urlparse

from pydantic_ai import RunContext, Agent
from pydantic_ai.settings import ModelSettings

from agentool import create_agentool
from agentool.base import BaseOperationInput
from agentool.core.registry import RoutingConfig
from agentool.core.injector import get_injector

# Import conductor models
from .conductor_models import (
    SearchTermGeneration, URLFilterResult, URLClassification,
    SearchResult, ResearchRequest, ConductorOutput, ConductorConfig
)

# Import templates
from jinja2 import Template


# Model configuration for each operation
_MODEL_CONFIG = {
    'default': {
        'model': 'openai:gpt-4o',
        'temperature': 0.2,
        'max_tokens': 16384,
    },
    'generate_search_terms': {
        'model': 'openai:gpt-4o-mini',
        'temperature': 0.7,
        'max_tokens': 4096,
    },
    'filter_urls': {
        'model': 'openai:gpt-4o-mini',
        'temperature': 0.1,
        'max_tokens': 4096,
    }
}


class ResearchConductorInput(BaseOperationInput):
    """Input schema for research conductor operations."""
    
    operation: Literal[
        'generate_search_terms',
        'execute_searches', 
        'filter_urls',
        'conduct_research'
    ] = Field(
        description="The research conductor operation to perform"
    )
    
    # For generate_search_terms
    topic: Optional[str] = Field(
        None,
        description="Research topic or question"
    )
    research_type: Optional[Literal[
        'documentation', 'trend_analysis', 'comparative', 'fact_finding'
    ]] = Field(
        'documentation',
        description="Type of research to conduct"
    )
    preferred_domains: Optional[List[str]] = Field(
        None,
        description="Domains to prioritize if found"
    )
    
    # For execute_searches
    search_terms: Optional[SearchTermGeneration] = Field(
        None,
        description="Generated search terms to execute"
    )
    max_searches: Optional[int] = Field(
        3,
        description="Maximum number of searches to execute"
    )
    
    # For filter_urls
    search_results: Optional[List[SearchResult]] = Field(
        None,
        description="Search results to filter and prioritize"
    )
    
    # For conduct_research (simplified entry point)
    max_sources: Optional[int] = Field(
        5,
        ge=1,
        le=20,
        description="Maximum number of sources to process"
    )
    relevance_threshold: Optional[float] = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score for content"
    )
    quality_threshold: Optional[float] = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Minimum quality score for content"
    )
    exclude_domains: Optional[List[str]] = Field(
        None,
        description="Domains to exclude from research"
    )
    
    @field_validator('topic')
    def validate_topic(cls, v: Optional[str], info) -> Optional[str]:
        """Validate topic is provided for operations that need it."""
        if info.data.get('operation') in ['generate_search_terms', 'conduct_research']:
            if not v or not v.strip():
                raise ValueError(f"topic is required for {info.data.get('operation')} operation")
        return v
    
    @field_validator('search_terms')
    def validate_search_terms(cls, v: Optional[SearchTermGeneration], info) -> Optional[SearchTermGeneration]:
        """Validate search_terms for execute_searches."""
        if info.data.get('operation') == 'execute_searches' and not v:
            raise ValueError("search_terms is required for execute_searches operation")
        return v
    
    @field_validator('search_results')
    def validate_search_results(cls, v: Optional[List[SearchResult]], info) -> Optional[List[SearchResult]]:
        """Validate search_results for filter_urls."""
        if info.data.get('operation') == 'filter_urls' and not v:
            raise ValueError("search_results is required for filter_urls operation")
        return v


async def generate_search_terms(
    ctx: RunContext[Any],
    topic: str,
    research_type: str,
    preferred_domains: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate optimized search terms for a research topic.
    
    Uses LLM to create various types of search queries including
    primary terms, refinements, boolean queries, and domain hints.
    """
    # Load templates
    with open('src/templates/system/search_term_generator.jinja', 'r') as f:
        system_template = Template(f.read())
    with open('src/templates/prompts/search_term_generator.jinja', 'r') as f:
        user_template = Template(f.read())
    
    # Prepare template variables
    template_vars = {
        'topic': topic,
        'research_type': research_type,
        'preferred_domains': preferred_domains or [],
        'schema_json': json.dumps(SearchTermGeneration.model_json_schema(), indent=2)
    }
    
    # Render templates
    system_prompt = system_template.render(**template_vars)
    user_prompt = user_template.render(**template_vars)
    
    # Get model config
    config = _MODEL_CONFIG.get('generate_search_terms', _MODEL_CONFIG['default'])
    
    # Create agent with structured output
    agent = Agent(
        model=config['model'],
        result_type=SearchTermGeneration,
        system_prompt=system_prompt
    )
    
    # Run agent
    result = await agent.run(
        user_prompt,
        model_settings=ModelSettings(
            temperature=config['temperature'],
            max_tokens=config['max_tokens']
        )
    )
    
    return {
        'success': True,
        'search_terms': result.data.model_dump(),
        'primary_count': len(result.data.primary_terms),
        'refinement_count': len(result.data.refinement_terms),
        'boolean_count': len(result.data.boolean_queries),
        'message': f"Generated {len(result.data.primary_terms)} primary and {len(result.data.refinement_terms)} refinement search terms"
    }


async def execute_searches(
    ctx: RunContext[Any],
    search_terms: SearchTermGeneration,
    max_searches: int = 3
) -> Dict[str, Any]:
    """
    Execute searches using generated search terms.
    
    Uses browser automation to perform DuckDuckGo searches and
    collect URLs from results.
    """
    injector = get_injector()
    search_results = []
    urls_discovered = set()
    browser_id = f"search_{uuid.uuid4().hex[:8]}"
    
    try:
        # Start browser for searches
        browser_result = await injector.run('browser_manager', {
            'operation': 'start_browser',
            'browser_id': browser_id,
            'options': {'headless': False}
        })
        
        if not browser_result.success:
            raise RuntimeError("Failed to start browser for searches")
        
        # Limit searches to max_searches
        all_terms = search_terms.primary_terms[:max_searches]
        
        for search_query in all_terms:
            try:
                # Construct DuckDuckGo search URL
                search_url = f"https://duckduckgo.com/?q={search_query.replace(' ', '+')}"
                
                # Navigate to search results
                nav_result = await injector.run('page_navigator', {
                    'operation': 'navigate',
                    'browser_id': browser_id,
                    'url': search_url,
                    'timeout': 30000
                })
                
                if not nav_result.success:
                    continue
                
                # Get page content
                content_result = await injector.run('page_navigator', {
                    'operation': 'get_content',
                    'browser_id': browser_id,
                    'extract_content': True
                })
                
                if content_result.success:
                    raw_html = content_result.data.get('html', '')
                    
                    # Extract links from search results
                    links_result = await injector.run('content_extractor', {
                        'operation': 'extract_links',
                        'content': raw_html,
                        'url': search_url
                    })
                    
                    if links_result.success:
                        links = links_result.data.get('links', [])
                        
                        # Filter out DuckDuckGo's own links
                        urls = []
                        for link in links[:20]:
                            if isinstance(link, dict):
                                url = link.get('url', '')
                                title = link.get('title', '') or link.get('text', '')
                                # Skip DuckDuckGo internal links
                                if url and 'http' in url and not any(
                                    domain in url for domain in [
                                        'duckduckgo.com', 'duck.com', 'duckduckgo.co',
                                        'javascript:', 'mailto:', '#'
                                    ]
                                ):
                                    url_info = {
                                        'url': url,
                                        'title': title[:200] if title else 'No title',
                                        'snippet': '',  # We'll get snippets later if needed
                                        'domain': urlparse(url).netloc
                                    }
                                    urls.append(url_info)
                                    urls_discovered.add(url)
                        
                        if urls:
                            search_result = SearchResult(
                                query=search_query,
                                urls=urls[:10],  # Limit to 10 URLs per search
                                total_results=len(urls),
                                search_engine='duckduckgo'
                            )
                            search_results.append(search_result)
                
                # Small delay between searches
                await asyncio.sleep(2.0)
                
            except Exception as e:
                print(f"Search failed for '{search_query}': {e}")
                continue
        
    finally:
        # Always stop the browser
        try:
            await injector.run('browser_manager', {
                'operation': 'stop_browser',
                'browser_id': browser_id
            })
        except:
            pass
    
    return {
        'success': True if search_results else False,
        'search_results': [sr.model_dump() for sr in search_results],
        'searches_executed': len(search_results),
        'urls_discovered': len(urls_discovered),
        'message': f"Executed {len(search_results)} searches, discovered {len(urls_discovered)} unique URLs"
    }


async def filter_urls(
    ctx: RunContext[Any],
    topic: str,
    research_type: str,
    search_results: List[SearchResult],
    preferred_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Filter and prioritize URLs from search results using indices.
    
    Uses LLM to categorize URLs by priority based on relevance,
    authority, and quality. Returns indices instead of full URLs
    for efficiency.
    """
    # Flatten all URLs with indices
    all_urls = []
    url_index = 0
    
    for search_result in search_results:
        for url_info in search_result.urls:
            # Skip excluded domains
            if exclude_domains and url_info['domain'] in exclude_domains:
                continue
            
            all_urls.append({
                'index': url_index,
                'url': url_info['url'],
                'title': url_info['title'],
                'snippet': url_info.get('snippet', ''),
                'domain': url_info['domain'],
                'query': search_result.query
            })
            url_index += 1
    
    if not all_urls:
        return {
            'success': False,
            'message': "No URLs to filter after applying exclusions"
        }
    
    # Load templates
    with open('src/templates/system/url_filter.jinja', 'r') as f:
        system_template = Template(f.read())
    with open('src/templates/prompts/url_filter.jinja', 'r') as f:
        user_template = Template(f.read())
    
    # Prepare template variables
    template_vars = {
        'topic': topic,
        'research_type': research_type,
        'search_results': all_urls,
        'preferred_domains': preferred_domains or [],
        'exclude_domains': exclude_domains or [],
        'schema_json': json.dumps(URLFilterResult.model_json_schema(), indent=2)
    }
    
    # Render templates
    system_prompt = system_template.render(**template_vars)
    user_prompt = user_template.render(**template_vars)
    
    # Get model config
    config = _MODEL_CONFIG.get('filter_urls', _MODEL_CONFIG['default'])
    
    # Create agent with structured output
    agent = Agent(
        model=config['model'],
        result_type=URLFilterResult,
        system_prompt=system_prompt
    )
    
    # Run agent
    result = await agent.run(
        user_prompt,
        model_settings=ModelSettings(
            temperature=config['temperature'],
            max_tokens=config['max_tokens']
        )
    )
    
    # Get prioritized indices
    filter_result = result.data
    prioritized_indices = filter_result.get_prioritized_indices(max_urls=10)
    
    # Map indices back to URLs for the response
    prioritized_urls = [all_urls[idx] for idx in prioritized_indices if idx < len(all_urls)]
    
    return {
        'success': True,
        'filter_result': filter_result.model_dump(),
        'prioritized_urls': prioritized_urls,
        'high_priority_count': len(filter_result.high_priority),
        'medium_priority_count': len(filter_result.medium_priority),
        'low_priority_count': len(filter_result.low_priority),
        'excluded_count': len(filter_result.excluded),
        'message': f"Filtered {len(all_urls)} URLs: {len(filter_result.high_priority)} high, {len(filter_result.medium_priority)} medium, {len(filter_result.low_priority)} low priority"
    }


async def conduct_research(
    ctx: RunContext[Any],
    topic: str,
    research_type: str = 'documentation',
    max_sources: int = 5,
    relevance_threshold: float = 0.5,
    quality_threshold: float = 0.5,
    preferred_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Conduct comprehensive research on a topic (single entry point).
    
    This orchestrates the entire research process:
    1. Generate search terms
    2. Execute searches
    3. Filter and prioritize URLs
    4. Create research plan
    5. Execute research and generate report
    """
    injector = get_injector()
    session_id = f"conductor_{uuid.uuid4().hex[:8]}"
    
    try:
        # Step 1: Generate search terms
        search_terms_result = await generate_search_terms(
            ctx, topic, research_type, preferred_domains
        )
        
        if not search_terms_result['success']:
            raise RuntimeError("Failed to generate search terms")
        
        search_terms = SearchTermGeneration(**search_terms_result['search_terms'])
        
        # Step 2: Execute searches
        searches_result = await execute_searches(
            ctx, search_terms, max_searches=3
        )
        
        if not searches_result['success']:
            raise RuntimeError("Failed to execute searches")
        
        search_results = [
            SearchResult(**sr) for sr in searches_result['search_results']
        ]
        
        # Step 3: Filter and prioritize URLs
        filter_result = await filter_urls(
            ctx, topic, research_type, search_results,
            preferred_domains, exclude_domains
        )
        
        if not filter_result['success']:
            raise RuntimeError("Failed to filter URLs")
        
        prioritized_urls = filter_result['prioritized_urls'][:max_sources]
        
        # Step 4: Create research plan using research_orchestrator
        # Extract just the URLs from prioritized_urls
        source_urls = [url['url'] for url in prioritized_urls]
        
        # Extract search terms that were used
        all_search_terms = search_terms.primary_terms + search_terms.refinement_terms
        
        plan_result = await injector.run('research_orchestrator', {
            'operation': 'plan_research',
            'session_id': session_id,
            'query': topic,
            'research_type': research_type,
            'sources': source_urls,  # List of URLs to research
            'search_terms': all_search_terms[:10],  # Additional search terms
            'max_sources': max_sources,
            'content_filter': {
                'relevance_threshold': relevance_threshold if relevance_threshold is not None else 0.3,
                'quality_threshold': quality_threshold if quality_threshold is not None else 0.3
            }
        })
        
        # Handle the AgentRunResult object - the injector returns ResearchOrchestratorOutput
        if hasattr(plan_result, 'success'):
            # It's already a ResearchOrchestratorOutput object
            if not plan_result.success:
                raise RuntimeError(f"Failed to create research plan: {plan_result.message}")
            plan_data = plan_result.data
        elif hasattr(plan_result, 'data'):
            plan_data = plan_result.data
        elif hasattr(plan_result, 'output'):
            plan_data = json.loads(plan_result.output) if isinstance(plan_result.output, str) else plan_result.output
        else:
            raise RuntimeError(f"Unexpected plan_result type: {type(plan_result)}")
        
        # Step 5: Execute research (it will use the plan from session)
        # The execute_research operation only needs session_id
        sources_processed = 0
        try:
            execute_result = await injector.run('research_orchestrator', {
                'operation': 'execute_research',
                'session_id': session_id
            })
            
            # Handle the AgentRunResult object - the injector returns ResearchOrchestratorOutput
            if hasattr(execute_result, 'success'):
                # It's already a ResearchOrchestratorOutput object
                if execute_result.success:
                    # The execute_research stores content in session, not in response
                    # We need to check sources_processed to know if content was collected
                    sources_processed = execute_result.data.get('sources_processed', 0)
            elif hasattr(execute_result, 'data'):
                execute_data = execute_result.data
                if execute_data and isinstance(execute_data, dict):
                    sources_processed = execute_data.get('sources_processed', 0)
                    
        except Exception as e:
            print(f"Failed to execute research: {e}")
            # Continue anyway, we might still have some content in session
        
        # Step 6: Aggregate content before generating report
        # The research_orchestrator requires aggregation before report generation
        try:
            aggregate_result = await injector.run('research_orchestrator', {
                'operation': 'aggregate_content',
                'session_id': session_id
            })
            
            # Check if aggregation succeeded
            if hasattr(aggregate_result, 'success') and not aggregate_result.success:
                print(f"Warning: Content aggregation failed: {aggregate_result.message}")
        except Exception as e:
            print(f"Warning: Failed to aggregate content: {e}")
        
        # Step 7: Generate final report
        # The generate_report operation uses aggregated session data
        if sources_processed > 0 or True:  # Always try to generate report from session
            report_result = await injector.run('research_orchestrator', {
                'operation': 'generate_report',
                'session_id': session_id,
                'report_format': 'markdown'
            })
            
            # Handle the AgentRunResult object - the injector returns ResearchOrchestratorOutput
            if hasattr(report_result, 'success'):
                # It's already a ResearchOrchestratorOutput object
                if not report_result.success:
                    raise RuntimeError(f"Failed to generate report: {report_result.message}")
                report_data = report_result.data
            elif hasattr(report_result, 'data'):
                report_data = report_result.data
            elif hasattr(report_result, 'output'):
                report_data = json.loads(report_result.output) if isinstance(report_result.output, str) else report_result.output
            else:
                report_data = report_result
            
            # Create ConductorOutput
            output = ConductorOutput(
                success=True,
                session_id=session_id,
                search_terms_generated=len(search_terms.primary_terms) + len(search_terms.refinement_terms),
                sources_discovered=searches_result['urls_discovered'],
                sources_prioritized=filter_result['high_priority_count'],
                research_plan_created=True,
                message="Research completed successfully",
                report_path=report_data.get('report_path') if isinstance(report_data, dict) else None
            )
            
            return {
                'success': True,
                'conductor_output': output.model_dump(),
                'report': report_data.get('report', 'No report content generated') if isinstance(report_data, dict) else str(report_data),
                'session_id': session_id,
                'message': f"Successfully completed research on '{topic}' with {sources_processed} sources"
            }
        
        else:
            return {
                'success': False,
                'session_id': session_id,
                'message': "No valid research findings collected"
            }
            
    except Exception as e:
        return {
            'success': False,
            'session_id': session_id,
            'error': str(e),
            'message': f"Research failed: {e}"
        }


# Routing configuration
routing = RoutingConfig(
    operation_field='operation',
    operation_map={
        'generate_search_terms': (
            'generate_search_terms',
            lambda x: {
                'topic': x.topic,
                'research_type': x.research_type,
                'preferred_domains': x.preferred_domains
            }
        ),
        'execute_searches': (
            'execute_searches',
            lambda x: {
                'search_terms': x.search_terms,
                'max_searches': x.max_searches
            }
        ),
        'filter_urls': (
            'filter_urls',
            lambda x: {
                'topic': x.topic,
                'research_type': x.research_type,
                'search_results': x.search_results,
                'preferred_domains': x.preferred_domains,
                'exclude_domains': x.exclude_domains
            }
        ),
        'conduct_research': (
            'conduct_research',
            lambda x: {
                'topic': x.topic,
                'research_type': x.research_type,
                'max_sources': x.max_sources,
                'relevance_threshold': x.relevance_threshold,
                'quality_threshold': x.quality_threshold,
                'preferred_domains': x.preferred_domains,
                'exclude_domains': x.exclude_domains
            }
        )
    }
)


def create_research_conductor_agent():
    """Create and return the research conductor AgenTool."""
    return create_agentool(
        name='research_conductor',
        input_schema=ResearchConductorInput,
        routing_config=routing,
        tools=[
            generate_search_terms,
            execute_searches,
            filter_urls,
            conduct_research
        ],
        system_prompt="Orchestrate comprehensive web research with optimized search strategies.",
        description="High-level research interface requiring only topic and research type",
        version="1.0.0",
        tags=["research", "conductor", "orchestration", "llm"]
    )


# Export the agent instance
agent = create_research_conductor_agent()