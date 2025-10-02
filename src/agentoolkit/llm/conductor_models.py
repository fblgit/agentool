"""Data models for the Research Conductor.

These models define structures for search term generation,
URL filtering, and research orchestration.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class SearchTermGeneration(BaseModel):
    """Generated search terms optimized for the research topic.
    
    Provides different types of search queries to maximize
    relevant results from search engines.
    """
    primary_terms: List[str] = Field(
        description="Main search queries directly related to the topic (e.g., ['OpenAI o1 model', 'o1-preview capabilities'])"
    )
    refinement_terms: List[str] = Field(
        description="Follow-up queries for deeper investigation (e.g., ['o1 model vs GPT-4', 'o1 model pricing'])"
    )
    boolean_queries: List[str] = Field(
        default_factory=list,
        description="Advanced search queries with operators (e.g., ['OpenAI o1 OR o1-preview', 'o1 model -GPT-3'])"
    )
    domain_hints: List[str] = Field(
        default_factory=list,
        description="Suggested domains to focus on (e.g., ['openai.com', 'github.com', 'arxiv.org'])"
    )
    
    @field_validator('primary_terms')
    def validate_primary_terms(cls, v: List[str]) -> List[str]:
        """Ensure at least one primary search term."""
        if not v:
            raise ValueError("At least one primary search term must be provided")
        return v


class URLClassification(BaseModel):
    """Classification of a URL by index."""
    index: int = Field(
        description="Index of the URL in the provided list (0-based)"
    )
    reason: str = Field(
        description="Brief reason for this classification"
    )


class URLFilterResult(BaseModel):
    """Filtered and prioritized URLs from search results using indices.
    
    Categorizes discovered URLs by relevance using their indices
    for efficient processing.
    """
    high_priority: List[URLClassification] = Field(
        default_factory=list,
        description="Indices of most relevant URLs that should be processed first"
    )
    medium_priority: List[URLClassification] = Field(
        default_factory=list,
        description="Indices of moderately relevant URLs to process if capacity allows"
    )
    low_priority: List[URLClassification] = Field(
        default_factory=list,
        description="Indices of potentially useful but less relevant URLs"
    )
    excluded: List[URLClassification] = Field(
        default_factory=list,
        description="Indices of URLs to skip with reasons (spam, irrelevant, duplicate)"
    )
    
    def get_prioritized_indices(self, max_urls: int = 10) -> List[int]:
        """Get prioritized list of URL indices up to max_urls."""
        indices = []
        for item in self.high_priority:
            if len(indices) >= max_urls:
                break
            indices.append(item.index)
        
        for item in self.medium_priority:
            if len(indices) >= max_urls:
                break
            indices.append(item.index)
            
        for item in self.low_priority:
            if len(indices) >= max_urls:
                break
            indices.append(item.index)
            
        return indices


class SearchResult(BaseModel):
    """Raw search result from search engine."""
    query: str = Field(
        description="The search query used"
    )
    urls: List[Dict[str, str]] = Field(
        description="List of URLs with titles and snippets"
    )
    total_results: int = Field(
        description="Total number of results found"
    )
    search_engine: str = Field(
        default="duckduckgo",
        description="Search engine used"
    )


class ConductorConfig(BaseModel):
    """Configuration for research conductor operations."""
    max_search_terms: int = Field(
        default=10,
        description="Maximum number of search terms to generate"
    )
    max_preliminary_searches: int = Field(
        default=3,
        description="Maximum number of preliminary searches to execute"
    )
    max_urls_to_evaluate: int = Field(
        default=30,
        description="Maximum URLs to evaluate for filtering"
    )
    search_delay: float = Field(
        default=2.0,
        description="Delay between searches in seconds"
    )
    include_boolean_queries: bool = Field(
        default=True,
        description="Whether to generate boolean search queries"
    )
    prioritize_recent: bool = Field(
        default=True,
        description="Whether to prioritize recent content"
    )


class ResearchRequest(BaseModel):
    """Simplified research request from user."""
    topic: str = Field(
        description="Research topic or question"
    )
    research_type: Literal['documentation', 'trend_analysis', 'comparative', 'fact_finding'] = Field(
        default='documentation',
        description="Type of research to conduct"
    )
    max_sources: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of sources to process"
    )
    relevance_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score for content"
    )
    quality_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum quality score for content"
    )
    preferred_domains: Optional[List[str]] = Field(
        None,
        description="Domains to prioritize if found"
    )
    exclude_domains: Optional[List[str]] = Field(
        None,
        description="Domains to exclude from research"
    )


class ConductorOutput(BaseModel):
    """Output from research conductor."""
    success: bool = Field(
        description="Whether research was conducted successfully"
    )
    session_id: str = Field(
        description="Research session identifier"
    )
    search_terms_generated: int = Field(
        description="Number of search terms generated"
    )
    sources_discovered: int = Field(
        description="Number of sources discovered"
    )
    sources_prioritized: int = Field(
        description="Number of high-priority sources identified"
    )
    research_plan_created: bool = Field(
        description="Whether research plan was successfully created"
    )
    message: str = Field(
        description="Status message"
    )
    report_path: Optional[str] = Field(
        None,
        description="Path to generated report if completed"
    )