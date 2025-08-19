"""Data models for the Research Orchestrator.

These models define the structure for research planning, execution,
findings aggregation, and report generation following the workflow pattern.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class ResearchPlan(BaseModel):
    """Research plan output from planning phase.
    
    Defines the strategy and queries for conducting research on a topic.
    """
    strategy: str = Field(
        description="High-level research strategy describing the approach (e.g., 'Comprehensive documentation search with official sources prioritized')"
    )
    search_queries: List[str] = Field(
        description="List of search queries to execute (e.g., ['OpenAI o1 model capabilities', 'o1-preview API documentation', 'o1 vs GPT-4 comparison'])"
    )
    target_sources: List[str] = Field(
        description="Prioritized list of source types to focus on (e.g., ['Official documentation', 'API references', 'Technical blogs', 'Community forums'])"
    )
    steps: List[str] = Field(
        description="Ordered list of research steps to execute (e.g., ['Search for official documentation', 'Extract API details', 'Analyze capabilities', 'Compare with existing models'])"
    )
    content_filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Filtering criteria for content relevance (e.g., {'relevance_threshold': 0.7, 'quality_threshold': 0.6, 'max_age_days': 90})"
    )
    max_sources: int = Field(
        default=10,
        description="Maximum number of sources to process"
    )
    
    @field_validator('search_queries')
    def validate_queries_not_empty(cls, v: List[str]) -> List[str]:
        """Ensure at least one search query is provided."""
        if not v:
            raise ValueError("At least one search query must be provided")
        return v


class ResearchFindings(BaseModel):
    """Research findings from execution phase.
    
    Contains extracted content and metadata from research execution.
    """
    sources_processed: int = Field(
        description="Number of sources successfully processed"
    )
    content_extracted: bool = Field(
        description="Whether content was successfully extracted from sources"
    )
    deduplication_performed: bool = Field(
        description="Whether duplicate content was identified and removed"
    )
    extracted_links: List[str] = Field(
        default_factory=list,
        description="List of URLs extracted from search results"
    )
    relevance_scores: List[float] = Field(
        default_factory=list,
        description="Relevance scores for each piece of extracted content (0.0 to 1.0)"
    )
    content_summaries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Summaries of extracted content with source attribution"
    )
    failed_sources: List[str] = Field(
        default_factory=list,
        description="URLs that failed to load or extract content from"
    )
    
    @field_validator('relevance_scores')
    def validate_scores_range(cls, v: List[float]) -> List[float]:
        """Ensure all scores are between 0 and 1."""
        for score in v:
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"Relevance score {score} must be between 0.0 and 1.0")
        return v


class ContentAggregation(BaseModel):
    """Aggregated content from multiple research sources.
    
    Combines and deduplicates findings from research execution.
    """
    unique_sources: int = Field(
        description="Number of unique sources after deduplication"
    )
    duplicates_removed: int = Field(
        description="Number of duplicate entries removed"
    )
    summary: str = Field(
        description="High-level summary of aggregated content"
    )
    key_topics: List[str] = Field(
        default_factory=list,
        description="Main topics identified across all content"
    )
    content_clusters: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Grouped content by topic or theme"
    )
    quality_score: float = Field(
        ge=0.0, le=1.0,
        description="Overall quality score of aggregated content"
    )


class QueryRefinement(BaseModel):
    """Refined search queries based on initial findings.
    
    Provides improved queries to fill knowledge gaps or explore deeper.
    """
    refined_queries: List[str] = Field(
        description="New search queries based on findings (e.g., ['o1 model temperature settings', 'o1 context window limitations'])"
    )
    knowledge_gaps: List[str] = Field(
        default_factory=list,
        description="Identified gaps in current research (e.g., ['Pricing information not found', 'Rate limits unclear'])"
    )
    suggested_sources: List[str] = Field(
        default_factory=list,
        description="New sources to explore based on references found"
    )
    refinement_rationale: str = Field(
        description="Explanation of why these refinements are suggested"
    )
    priority_topics: List[str] = Field(
        default_factory=list,
        description="Topics that need more investigation"
    )


class ResearchProgress(BaseModel):
    """Progress tracking for ongoing research session.
    
    Provides status updates on research execution.
    """
    progress_percentage: float = Field(
        ge=0.0, le=100.0,
        description="Overall completion percentage"
    )
    current_step: str = Field(
        description="Current step being executed"
    )
    sources_processed: int = Field(
        description="Number of sources processed so far"
    )
    total_sources: int = Field(
        description="Total number of sources to process"
    )
    status: Literal['planning', 'executing', 'aggregating', 'reporting', 'completed', 'failed'] = Field(
        description="Current status of research session"
    )
    elapsed_time: float = Field(
        description="Time elapsed in seconds"
    )
    estimated_remaining: Optional[float] = Field(
        default=None,
        description="Estimated time remaining in seconds"
    )


class ResearchReport(BaseModel):
    """Final research report with structured findings.
    
    Comprehensive report with all research results and metadata.
    """
    title: str = Field(
        description="Report title based on research topic"
    )
    executive_summary: str = Field(
        description="High-level summary of findings (2-3 paragraphs)"
    )
    key_findings: List[str] = Field(
        description="Bullet points of most important discoveries"
    )
    detailed_sections: List[Dict[str, Any]] = Field(
        description="Detailed report sections with headers and content"
    )
    sources: List[Dict[str, str]] = Field(
        description="All sources cited in the report with URLs and titles"
    )
    methodology: str = Field(
        description="Description of research methodology used"
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Known limitations or gaps in the research"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommended next steps or actions based on findings"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Overall confidence in research completeness and accuracy"
    )
    source_count: int = Field(
        description="Total number of sources referenced"
    )
    report_path: Optional[str] = Field(
        default=None,
        description="Path where report was saved"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the research session"
    )
    
    @field_validator('key_findings')
    def validate_key_findings_not_empty(cls, v: List[str]) -> List[str]:
        """Ensure at least one key finding is provided."""
        if not v:
            raise ValueError("At least one key finding must be provided")
        return v


class SessionData(BaseModel):
    """Complete research session data.
    
    Contains all information about a research session.
    """
    session_id: str = Field(
        description="Unique identifier for the research session"
    )
    query: str = Field(
        description="Original research query"
    )
    research_type: Literal['documentation', 'trend_analysis', 'comparative', 'fact_finding'] = Field(
        description="Type of research conducted"
    )
    status: Literal['active', 'completed', 'failed', 'cancelled'] = Field(
        description="Current session status"
    )
    created_at: float = Field(
        description="Timestamp when session was created"
    )
    completed_at: Optional[float] = Field(
        default=None,
        description="Timestamp when session completed"
    )
    plan: Optional[ResearchPlan] = Field(
        default=None,
        description="Research plan if created"
    )
    findings: Optional[ResearchFindings] = Field(
        default=None,
        description="Research findings if executed"
    )
    aggregation: Optional[ContentAggregation] = Field(
        default=None,
        description="Aggregated content if processed"
    )
    refinements: List[QueryRefinement] = Field(
        default_factory=list,
        description="Query refinements made during research"
    )
    report: Optional[ResearchReport] = Field(
        default=None,
        description="Final report if generated"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if session failed"
    )