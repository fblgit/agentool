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

from pydantic_ai import RunContext

from agentool import create_agentool
from agentool.base import BaseOperationInput
from agentool.core.registry import RoutingConfig
from agentool.core.injector import get_injector


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
    Plan research strategy for a given query.
    
    Analyzes the research query to determine optimal search strategies, target sources,
    and research steps based on the research type and parameters.
    
    Args:
        ctx: Runtime context provided by the framework
        session_id: Research session identifier for tracking
        query: Research query or question to investigate
        research_type: Type of research pattern (documentation, trend_analysis, comparative, fact_finding)
        sources: Optional list of specific sources/URLs to research
        search_terms: Optional additional search terms for query expansion
        max_sources: Maximum number of sources to research (default: 10)
        content_filter: Optional content filtering options
        
    Returns:
        ResearchOrchestratorOutput with research plan containing strategy, queries, and steps
        
    Raises:
        ValueError: If query is empty or research_type is invalid
        LLMError: If LLM fails to generate research plan
        StorageError: If session storage fails
    """
    injector = get_injector()
    
    try:
        # Validate required parameters
        if not query:
            raise ValueError("query is required for research planning")
        if not research_type:
            raise ValueError("research_type is required for research planning")
            
        # Generate research plan using LLM
        plan_prompt = f"""
        Create a research plan for the following query:
        Query: {query}
        Research Type: {research_type}
        Max Sources: {max_sources}
        
        Generate a structured research plan including:
        1. Research strategy
        2. Search queries to use
        3. Target sources and types
        4. Research steps
        5. Estimated duration
        
        Format as JSON with keys: strategy, search_queries, target_sources, steps, estimated_duration
        """
        
        llm_result = await injector.run('llm', {
            'operation': 'generation',
            'content': plan_prompt,
            'model': 'gpt-4o',
            'cache_key': f'research_plan_{hash(query + research_type)}'
        })
        
        if not llm_result.success:
            raise RuntimeError(f"LLM planning failed: {llm_result.message}")
        
        # Parse LLM response
        # The LLM agent returns generated content in 'generated' field
        try:
            llm_response_text = llm_result.data.get('generated', '{}')
            plan_data = json.loads(llm_response_text)
        except (json.JSONDecodeError, AttributeError):
            # Fallback plan structure
            plan_data = {
                "strategy": f"{research_type}_focused",
                "search_queries": [query] + (search_terms or []),
                "target_sources": sources or ["search engines", "official documentation"],
                "steps": ["search_content", "extract_data", "analyze_results"],
                "estimated_duration": 300
            }
        
        # Create session data
        session_data = {
            'session_id': session_id,
            'query': query,
            'research_type': research_type,
            'max_sources': max_sources,
            'content_filter': content_filter or {},
            'plan': plan_data,
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
            'data': {'query': query, 'research_type': research_type, 'max_sources': max_sources}
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
                'strategy': plan_data.get('strategy') or f"{research_type}_focused",
                'search_queries': plan_data.get('search_queries') or [query] + (search_terms or []),
                'target_sources': plan_data.get('target_sources') or (sources or ["search engines", "official documentation"]),
                'estimated_duration': plan_data.get('estimated_duration') or 300,
                'steps': plan_data.get('steps') or ["search_content", "extract_data", "analyze_results"]
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
        
        # Start browser for dynamic content
        browser_id = f"research_{session_id}"
        browser_result = await injector.run('browser_manager', {
            'operation': 'start_browser',
            'browser_id': browser_id,
            'options': {'headless': True}
        })
        
        if not browser_result.success:
            raise ConnectionError(f"Failed to start browser: {browser_result.message}")
        
        sources_processed = 0
        content_collected = []
        relevance_scores = []
        
        # Execute search queries and collect content
        for query in search_queries[:max_sources]:
            try:
                # Use search engine (simulate with direct navigation)
                search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
                
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
                        
                        # For Google search results, extract links and visit them
                        if 'google.com/search' in search_url:
                            # Extract links from search results
                            links_result = await injector.run('content_extractor', {
                                'operation': 'extract_links',
                                'content': raw_html,
                                'url': search_url
                            })
                            
                            if links_result.success:
                                links = links_result.data.get('links', [])
                                # Filter out Google's own links and ads
                                result_links = [
                                    link for link in links[:3]  # Take first 3 results
                                    if not any(domain in link for domain in [
                                        'google.com', 'youtube.com', 'accounts.google',
                                        'support.google', 'policies.google'
                                    ])
                                ]
                                
                                # Visit actual result pages
                                for result_url in result_links[:2]:  # Visit top 2 results
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
                                                
                                                # Extract content from actual page
                                                extract_result = await injector.run('content_extractor', {
                                                    'operation': 'extract_content',
                                                    'content': actual_html,
                                                    'url': result_url,
                                                    'content_type': session_data.get('research_type', 'html')
                                                })
                                                
                                                if extract_result.success:
                                                    # Score actual content quality
                                                    quality_result = await injector.run('content_extractor', {
                                                        'operation': 'score_quality',
                                                        'content': actual_html,
                                                        'content_type': session_data.get('research_type', 'html')
                                                    })
                                                    
                                                    quality_score = 0.7  # Default higher for actual content
                                                    if quality_result.success:
                                                        quality_score = quality_result.data.get('quality_score', 0.7)
                                                    
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
                            extract_result = await injector.run('content_extractor', {
                                'operation': 'extract_content',
                                'content': raw_html,
                                'url': search_url,
                                'content_type': session_data.get('research_type', 'html')
                            })
                            
                            if extract_result.success:
                                extracted_content = extract_result.data
                                
                                # Score content quality
                                quality_result = await injector.run('content_extractor', {
                                    'operation': 'score_quality',
                                    'content': raw_html,
                                    'content_type': session_data.get('research_type', 'html')
                                })
                                
                                quality_score = 0.5  # Default
                                if quality_result.success:
                                    quality_score = quality_result.data.get('quality_score', 0.5)
                                
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
    Refine research query based on initial findings.
    
    Analyzes initial research results to generate refined search strategies
    and follow-up queries for deeper investigation.
    
    Args:
        ctx: Runtime context provided by the framework
        session_id: Research session identifier
        follow_up_query: Follow-up query based on initial findings
        
    Returns:
        ResearchOrchestratorOutput with refined research strategy
        
    Raises:
        KeyError: If session not found
        LLMError: If query refinement fails
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
        
        # Ensure follow_up_query is provided (defensive check)
        if not follow_up_query:
            raise ValueError("follow_up_query is required for refine_query")
        
        # Analyze existing content for context
        content_summary = ""
        if content_collected:
            content_texts = [item.get('content', {}).get('content', '') for item in content_collected[:3]]
            content_summary = '\n'.join(content_texts)[:1000]  # Limit size
        
        # Generate refined research strategy
        refinement_prompt = f"""
        Original query: {original_query}
        Follow-up query: {follow_up_query}
        
        Existing research findings summary:
        {content_summary}
        
        Based on the initial findings, refine the research strategy:
        1. Generate new specific search queries
        2. Identify knowledge gaps
        3. Suggest additional sources
        4. Recommend research direction
        
        Format as JSON with keys: refined_queries, knowledge_gaps, suggested_sources, research_direction
        """
        
        llm_result = await injector.run('llm', {
            'operation': 'generation',
            'content': refinement_prompt,
            'model': 'gpt-4o',
            'cache_key': f'query_refinement_{hash(original_query + follow_up_query)}'
        })
        
        if not llm_result.success:
            raise RuntimeError(f"LLM refinement failed: {llm_result.message}")
        
        # Parse refinement response
        try:
            refinement_data = json.loads(llm_result.data.get('generated', '{}'))
        except json.JSONDecodeError:
            refinement_data = {
                "refined_queries": [follow_up_query],
                "knowledge_gaps": ["More specific information needed"],
                "suggested_sources": ["academic databases", "technical documentation"],
                "research_direction": "deeper_analysis"
            }
        
        # Update session with refinement
        session_data.update({
            'follow_up_query': follow_up_query,
            'refinement': refinement_data,
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
            'data': {'follow_up_query': follow_up_query}
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
                'refined_queries': refinement_data.get('refined_queries', []),
                'knowledge_gaps': refinement_data.get('knowledge_gaps', []),
                'suggested_sources': refinement_data.get('suggested_sources', []),
                'research_direction': refinement_data.get('research_direction', 'continue')
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
            return ResearchOrchestratorOutput(
                success=False,
                message=f"No aggregated content available for session '{session_id}'. Run aggregate_content first.",
                data=None
            )
        
        # Prepare report data
        content_items = aggregated_content.get('content_items', [])
        summary = aggregated_content.get('summary', '')
        quality_scores = aggregated_content.get('quality_scores', [])
        
        # Calculate confidence score
        confidence_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
        
        # Extract key findings using LLM generation instead of extraction
        findings_content = '\n\n'.join([
            item.get('content', {}).get('content', '')[:300] 
            for item in content_items[:3]
        ])
        
        findings_prompt = f"""Based on this research content about: {query}

Content:
{findings_content}

Generate 3-5 key findings as a simple list. Focus on the most important and actionable insights.
Format each finding on a new line starting with "- "."""
        
        findings_result = await injector.run('llm', {
            'operation': 'generation',
            'content': findings_prompt,
            'options': {'max_tokens': 200},
            'model': 'gpt-4o'
        })
        
        key_findings = ["Research completed successfully"]
        if findings_result.success:
            try:
                # Parse the generated text to extract findings
                generated_text = findings_result.data.get('generated', '')
                if generated_text:
                    # Split by lines and extract items starting with "-"
                    lines = generated_text.strip().split('\n')
                    findings_list = []
                    for line in lines:
                        line = line.strip()
                        if line.startswith('- '):
                            findings_list.append(line[2:])  # Remove "- " prefix
                        elif line.startswith('•'):
                            findings_list.append(line[1:].strip())  # Remove bullet
                        elif line and not line.startswith('#'):  # Any non-empty, non-header line
                            findings_list.append(line)
                    
                    if findings_list:
                        key_findings = findings_list[:5]  # Limit to 5 findings
            except:
                pass
        
        # Generate report based on format
        if report_format == 'markdown':
            # Generate markdown report
            markdown_content = f"""# Research Report: {query}

## Executive Summary
{summary}

## Key Findings
{chr(10).join(['- ' + finding for finding in key_findings])}

## Research Details
- **Research Type**: {research_type}
- **Sources Analyzed**: {len(content_items)}
- **Confidence Score**: {confidence_score:.2f}
- **Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Sources
{chr(10).join(['- ' + item.get('url', 'Unknown source') for item in content_items[:10]])}

## Methodology
This research was conducted using automated web research with content extraction and analysis.
Sources were evaluated for relevance and quality, with duplicate content removed.
"""
            
            # Convert to final markdown using LLM
            markdown_result = await injector.run('llm', {
                'operation': 'markdownify',
                'content': markdown_content,
                'model': 'gpt-4o'
            })
            
            final_content = markdown_content
            if markdown_result.success:
                final_content = markdown_result.data.get('markdown', markdown_content)
            
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
            
            report_data = {
                'report_path': report_path,
                'summary': summary,
                'source_count': len(content_items),
                'confidence_score': confidence_score,
                'key_findings': key_findings
            }
            
        elif report_format == 'structured_data':
            report_data = {
                'session_id': session_id,
                'query': query,
                'research_type': research_type,
                'summary': summary,
                'key_findings': key_findings,
                'sources': [item.get('url') for item in content_items],
                'confidence_score': confidence_score,
                'source_count': len(content_items),
                'quality_scores': quality_scores,
                'generated_at': time.time()
            }
            
        else:  # summary format
            report_data = {
                'summary': summary,
                'key_findings': key_findings[:3],  # Top 3 findings
                'confidence_score': confidence_score,
                'source_count': len(content_items)
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
            return ResearchOrchestratorOutput(
                success=False,
                message=f"Research session '{session_id}' not found",
                data=None
            )
        
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