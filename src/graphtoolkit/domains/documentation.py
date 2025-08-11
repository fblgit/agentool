"""GraphToolkit Documentation Domain Definition.

Domain for creating comprehensive documentation.
Phases: Content Analyzer → Structure Designer → Writer → Reviewer
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..core.registry import register_phase
from ..core.types import ModelParameters, PhaseDefinition, StorageType, TemplateConfig

# Input/Output Schemas for Documentation Domain

class ContentAnalyzerInput(BaseModel):
    """Input for content analysis phase."""
    source_code: Dict[str, str] = Field(description='Source code files to document')
    existing_docs: Optional[List[str]] = Field(None, description='Existing documentation')
    documentation_type: str = Field(description='Type of documentation (API, user, technical)')
    target_audience: str = Field(description='Target audience for documentation')
    project_context: Dict[str, Any] = Field(description='Project context and metadata')


class ContentAnalyzerOutput(BaseModel):
    """Output from content analysis phase."""
    content_outline: Dict[str, Any] = Field(description='Documentation outline')
    key_concepts: List[Dict[str, str]] = Field(description='Key concepts to document')
    api_endpoints: Optional[List[Dict[str, Any]]] = Field(None, description='API endpoints found')
    code_examples_needed: List[str] = Field(description='Areas needing code examples')
    documentation_gaps: List[str] = Field(description='Identified documentation gaps')
    complexity_assessment: Dict[str, str] = Field(description='Complexity levels by section')


class StructureDesignerInput(BaseModel):
    """Input for structure design phase."""
    content_outline: Dict[str, Any] = Field(description='Content outline from analysis')
    key_concepts: List[Dict[str, str]] = Field(description='Key concepts to organize')
    documentation_type: str = Field(description='Type of documentation')
    style_guide: Optional[Dict[str, Any]] = Field(None, description='Documentation style guide')


class StructureDesignerOutput(BaseModel):
    """Output from structure design phase."""
    document_structure: Dict[str, Any] = Field(description='Hierarchical document structure')
    section_templates: Dict[str, str] = Field(description='Templates for each section type')
    navigation_schema: Dict[str, Any] = Field(description='Navigation structure')
    cross_references: List[Dict[str, str]] = Field(description='Cross-reference map')
    metadata_schema: Dict[str, Any] = Field(description='Document metadata structure')


class WriterInput(BaseModel):
    """Input for writing phase."""
    document_structure: Dict[str, Any] = Field(description='Document structure to follow')
    section_templates: Dict[str, str] = Field(description='Templates for sections')
    key_concepts: List[Dict[str, str]] = Field(description='Concepts to explain')
    code_examples: Optional[Dict[str, str]] = Field(None, description='Code examples to include')
    style_guide: Optional[Dict[str, Any]] = Field(None, description='Writing style guide')


class WriterOutput(BaseModel):
    """Output from writing phase."""
    documentation: Dict[str, str] = Field(description='Written documentation by section')
    code_snippets: List[Dict[str, Any]] = Field(description='Code snippets with explanations')
    diagrams_suggested: List[Dict[str, str]] = Field(description='Suggested diagrams')
    glossary: Dict[str, str] = Field(description='Glossary of terms')
    examples: List[Dict[str, Any]] = Field(description='Usage examples')


class ReviewerInput(BaseModel):
    """Input for review phase."""
    documentation: Dict[str, str] = Field(description='Documentation to review')
    document_structure: Dict[str, Any] = Field(description='Expected structure')
    target_audience: str = Field(description='Target audience')
    review_criteria: List[str] = Field(description='Review criteria')
    style_guide: Optional[Dict[str, Any]] = Field(None, description='Style guide for review')


class ReviewerOutput(BaseModel):
    """Output from review phase."""
    review_feedback: List[Dict[str, Any]] = Field(description='Review feedback by section')
    accuracy_score: float = Field(description='Technical accuracy score')
    completeness_score: float = Field(description='Completeness score')
    clarity_score: float = Field(description='Clarity and readability score')
    improvements: List[str] = Field(description='Suggested improvements')
    final_documentation: Dict[str, str] = Field(description='Final reviewed documentation')
    quality_score: float = Field(description='Overall quality score')


# Phase Definitions for Documentation Domain

content_analyzer_phase = PhaseDefinition(
    phase_name='content_analyzer',
    atomic_nodes=[
        'dependency_check',
        'load_dependencies',
        'template_render',
        'llm_call',
        'schema_validation',
        'save_phase_output',
        'state_update',
        'quality_gate'
    ],
    input_schema=ContentAnalyzerInput,
    output_schema=ContentAnalyzerOutput,
    dependencies=[],
    templates=TemplateConfig(
        system_template='templates/system/doc_analyzer.jinja',
        user_template='templates/prompts/analyze_content.jinja',
        variables={
            'analysis_mode': 'comprehensive',
            'include_examples': 'true'
        }
    ),
    storage_pattern='workflow/{workflow_id}/doc_analysis',
    storage_type=StorageType.KV,
    quality_threshold=0.8,
    allow_refinement=True,
    max_refinements=2,
    model_config=ModelParameters(
        temperature=0.7,
        max_tokens=2000
    ),
    domain='documentation'
)

structure_designer_phase = PhaseDefinition(
    phase_name='structure_designer',
    atomic_nodes=[
        'dependency_check',
        'load_dependencies',
        'template_render',
        'llm_call',
        'schema_validation',
        'save_phase_output',
        'state_update',
        'quality_gate'
    ],
    input_schema=StructureDesignerInput,
    output_schema=StructureDesignerOutput,
    dependencies=['content_analyzer'],
    templates=TemplateConfig(
        system_template='templates/system/doc_structure_designer.jinja',
        user_template='templates/prompts/design_doc_structure.jinja',
        variables={
            'structure_style': 'hierarchical',
            'include_navigation': 'true'
        }
    ),
    storage_pattern='workflow/{workflow_id}/doc_structure',
    storage_type=StorageType.KV,
    quality_threshold=0.85,
    allow_refinement=True,
    max_refinements=2,
    model_config=ModelParameters(
        temperature=0.6,
        max_tokens=1500
    ),
    domain='documentation'
)

writer_phase = PhaseDefinition(
    phase_name='writer',
    atomic_nodes=[
        'dependency_check',
        'load_dependencies',
        'template_render',
        'llm_call',
        'schema_validation',
        'save_phase_output',
        'state_update',
        'quality_gate'
    ],
    input_schema=WriterInput,
    output_schema=WriterOutput,
    dependencies=['content_analyzer', 'structure_designer'],
    templates=TemplateConfig(
        system_template='templates/system/doc_writer.jinja',
        user_template='templates/prompts/write_documentation.jinja',
        variables={
            'writing_style': 'clear_technical',
            'include_examples': 'true',
            'example_style': 'practical'
        }
    ),
    storage_pattern='workflow/{workflow_id}/documentation',
    storage_type=StorageType.FS,  # Use file storage for documentation
    quality_threshold=0.85,
    allow_refinement=True,
    max_refinements=3,
    model_config=ModelParameters(
        temperature=0.7,
        max_tokens=4000
    ),
    domain='documentation'
)

reviewer_phase = PhaseDefinition(
    phase_name='reviewer',
    atomic_nodes=[
        'dependency_check',
        'load_dependencies',
        'template_render',
        'llm_call',
        'schema_validation',
        'save_phase_output',
        'state_update',
        'quality_gate'
    ],
    input_schema=ReviewerInput,
    output_schema=ReviewerOutput,
    dependencies=['content_analyzer', 'structure_designer', 'writer'],
    templates=TemplateConfig(
        system_template='templates/system/doc_reviewer.jinja',
        user_template='templates/prompts/review_documentation.jinja',
        variables={
            'review_depth': 'thorough',
            'check_accuracy': 'true',
            'check_completeness': 'true',
            'check_clarity': 'true'
        }
    ),
    storage_pattern='workflow/{workflow_id}/review',
    storage_type=StorageType.KV,
    quality_threshold=0.9,
    allow_refinement=True,
    max_refinements=2,
    model_config=ModelParameters(
        temperature=0.5,
        max_tokens=2500
    ),
    domain='documentation'
)


def register_documentation_phases():
    """Register all documentation domain phases."""
    register_phase('documentation.content_analyzer', content_analyzer_phase)
    register_phase('documentation.structure_designer', structure_designer_phase)
    register_phase('documentation.writer', writer_phase)
    register_phase('documentation.reviewer', reviewer_phase)


# Auto-register on import
register_documentation_phases()