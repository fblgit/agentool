"""
GraphToolkit API Domain.

Phase definitions for API design and generation workflows.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from ..core.types import (
    PhaseDefinition,
    TemplateConfig,
    ModelParameters,
    StorageType
)
from ..core.registry import get_registry


# Input/Output Schemas for API phases

class APIAnalyzerInput(BaseModel):
    """Input schema for API analyzer phase."""
    requirements: str = Field(description="API requirements and use cases")
    domain: str = Field(default="api", description="Workflow domain")
    existing_apis: Optional[List[Dict[str, Any]]] = Field(default=None, description="Existing API definitions")


class APIAnalyzerOutput(BaseModel):
    """Output schema for API analyzer phase."""
    endpoints: List[Dict[str, Any]] = Field(description="Identified API endpoints")
    data_models: List[Dict[str, Any]] = Field(description="Required data models")
    authentication: Dict[str, Any] = Field(description="Authentication requirements")
    success: bool = Field(default=True)


class APIDesignerInput(BaseModel):
    """Input schema for API designer phase."""
    endpoints: List[Dict[str, Any]] = Field(description="API endpoints to design")
    data_models: List[Dict[str, Any]] = Field(description="Data models to use")


class APIDesignerOutput(BaseModel):
    """Output schema for API designer phase."""
    openapi_spec: Dict[str, Any] = Field(description="OpenAPI specification")
    schemas: Dict[str, Any] = Field(description="JSON schemas for data models")
    success: bool = Field(default=True)


class APIGeneratorInput(BaseModel):
    """Input schema for API generator phase."""
    openapi_spec: Dict[str, Any] = Field(description="OpenAPI specification")
    framework: str = Field(default="fastapi", description="Target framework")


class APIGeneratorOutput(BaseModel):
    """Output schema for API generator phase."""
    implementation: Dict[str, str] = Field(description="Generated API code files")
    tests: Dict[str, str] = Field(description="Generated test files")
    documentation: str = Field(description="API documentation")
    success: bool = Field(default=True)


# Phase Definitions

API_ANALYZER_PHASE = PhaseDefinition(
    phase_name="analyzer",
    atomic_nodes=[
        "dependency_check",
        "load_dependencies",
        "template_render",
        "llm_call",
        "schema_validation",
        "save_output",
        "state_update",
        "quality_gate"
    ],
    input_schema=APIAnalyzerInput,
    output_schema=APIAnalyzerOutput,
    dependencies=[],
    templates=TemplateConfig(
        system_template="system/api/analyzer",
        user_template="prompts/api/analyze_requirements",
        variables={
            "framework": "API",
            "purpose": "Analyze API requirements"
        }
    ),
    storage_pattern="workflow/{workflow_id}/api_analyzer",
    storage_type=StorageType.KV,
    quality_threshold=0.8,
    allow_refinement=True,
    max_refinements=3,
    model_config=ModelParameters(
        temperature=0.7,
        max_tokens=2000
    ),
    domain="api"
)

API_DESIGNER_PHASE = PhaseDefinition(
    phase_name="designer",
    atomic_nodes=[
        "dependency_check",
        "load_dependencies",
        "template_render",
        "llm_call",
        "schema_validation",
        "save_output",
        "state_update",
        "quality_gate"
    ],
    input_schema=APIDesignerInput,
    output_schema=APIDesignerOutput,
    dependencies=["analyzer"],
    templates=TemplateConfig(
        system_template="system/api/designer",
        user_template="prompts/api/design_schema",
        variables={
            "framework": "OpenAPI",
            "purpose": "Design API schema"
        }
    ),
    storage_pattern="workflow/{workflow_id}/api_designer",
    storage_type=StorageType.KV,
    quality_threshold=0.85,
    allow_refinement=True,
    max_refinements=3,
    model_config=ModelParameters(
        temperature=0.5,
        max_tokens=3000
    ),
    domain="api"
)

API_GENERATOR_PHASE = PhaseDefinition(
    phase_name="generator",
    atomic_nodes=[
        "dependency_check",
        "load_dependencies",
        "template_render",
        "llm_call",
        "schema_validation",
        "save_output",
        "state_update",
        "quality_gate"
    ],
    input_schema=APIGeneratorInput,
    output_schema=APIGeneratorOutput,
    dependencies=["designer"],
    templates=TemplateConfig(
        system_template="system/api/generator",
        user_template="prompts/api/generate_implementation",
        variables={
            "framework": "FastAPI",
            "purpose": "Generate API implementation"
        }
    ),
    storage_pattern="workflow/{workflow_id}/api_generator",
    storage_type=StorageType.KV,
    quality_threshold=0.9,
    allow_refinement=True,
    max_refinements=3,
    model_config=ModelParameters(
        temperature=0.3,
        max_tokens=4000
    ),
    domain="api"
)


def register_api_domain():
    """Register all API domain phases with the registry."""
    registry = get_registry()
    
    registry.register_phase("api.analyzer", API_ANALYZER_PHASE)
    registry.register_phase("api.designer", API_DESIGNER_PHASE)
    registry.register_phase("api.generator", API_GENERATOR_PHASE)


# Auto-register on import
register_api_domain()