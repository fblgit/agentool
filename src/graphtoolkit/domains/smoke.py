"""
Smoke domain for lightweight E2E testing.

This domain implements a simple recipe generation workflow that exercises
all GraphToolkit capabilities with minimal complexity. Perfect for testing
with real LLM calls without high costs or complexity.
"""

from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from ..core.types import PhaseDefinition, StorageType
from ..core.registry import register_phase


# Output schemas for each phase

class IngredientAnalysis(BaseModel):
    """Output from ingredient analyzer phase."""
    ingredients: List[str] = Field(description="List of analyzed ingredients")
    compatibility_score: float = Field(ge=0, le=1, description="How well ingredients work together")
    suggested_cuisine: str = Field(description="Best cuisine type for these ingredients")
    cooking_methods: List[str] = Field(description="Recommended cooking methods")
    nutritional_highlights: List[str] = Field(description="Key nutritional benefits")
    warnings: List[str] = Field(default_factory=list, description="Allergy or compatibility warnings")


class RecipeDesign(BaseModel):
    """Output from recipe designer phase."""
    recipe_name: str = Field(description="Name of the recipe")
    cuisine_type: str = Field(description="Type of cuisine")
    difficulty: Literal["easy", "medium", "hard"] = Field(description="Recipe difficulty level")
    prep_time_minutes: int = Field(gt=0, description="Preparation time in minutes")
    cook_time_minutes: int = Field(ge=0, description="Cooking time in minutes")
    servings: int = Field(gt=0, description="Number of servings")
    techniques: List[str] = Field(description="Cooking techniques required")
    equipment_needed: List[str] = Field(description="Kitchen equipment required")


class IngredientAmount(BaseModel):
    """An ingredient with its amount."""
    ingredient: str = Field(description="Name of the ingredient")
    amount: str = Field(description="Amount of the ingredient (e.g., '2 cups', '1 tbsp')")


class DetailedRecipe(BaseModel):
    """Output from recipe crafter phase."""
    name: str = Field(description="Recipe name")
    ingredients_with_amounts: List[IngredientAmount] = Field(
        description="List of ingredients with their amounts"
    )
    instructions: List[str] = Field(
        min_length=3, description="Step-by-step cooking instructions"
    )
    tips: List[str] = Field(description="Helpful cooking tips")
    variations: List[str] = Field(description="Recipe variations")


class RecipeEvaluation(BaseModel):
    """Output from recipe evaluator phase."""
    completeness_score: float = Field(ge=0, le=1, description="How complete the recipe is")
    clarity_score: float = Field(ge=0, le=1, description="How clear the instructions are")
    practicality_score: float = Field(ge=0, le=1, description="How practical to make")
    overall_score: float = Field(ge=0, le=1, description="Overall recipe quality")
    improvements: List[str] = Field(description="Suggested improvements")
    ready_to_cook: bool = Field(description="Whether recipe is ready to use")


# Input schemas

class SmokeWorkflowInput(BaseModel):
    """Input for smoke workflow."""
    ingredients: List[str] = Field(min_length=1, description="List of available ingredients")
    dietary_restrictions: List[str] = Field(
        default_factory=list, description="Dietary restrictions or preferences"
    )
    cuisine_preference: Optional[str] = Field(None, description="Preferred cuisine type")
    max_cook_time: Optional[int] = Field(None, description="Maximum cooking time in minutes")


# Register phases for smoke domain

from ..core.types import TemplateConfig, ModelParameters

# Phase 1: Ingredient Analyzer
analyzer_phase = PhaseDefinition(
    phase_name='ingredient_analyzer',
    domain='smoke',
    atomic_nodes=[
        'dependency_check',
        'template_render',
        'llm_call',
        'schema_validation',
        'save_phase_output',
        'state_update',
        'quality_gate'
    ],
    input_schema=SmokeWorkflowInput,
    output_schema=IngredientAnalysis,
    dependencies=[],
    templates=TemplateConfig(
        system_template='smoke/system/analyzer.jinja',
        user_template='smoke/prompts/analyze_ingredients.jinja'
    ),
    storage_pattern='workflow/{workflow_id}/ingredient_analysis',
    storage_type=StorageType.KV,
    quality_threshold=0.6,
    allow_refinement=True,
    max_refinements=2,
    model_config=ModelParameters(
        temperature=0.7,
        max_tokens=500
    )
)

register_phase('smoke.ingredient_analyzer', analyzer_phase)

# Phase 2: Recipe Designer
designer_phase = PhaseDefinition(
    phase_name='recipe_designer',
    domain='smoke',
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
    input_schema=None,  # Uses output from analyzer
    output_schema=RecipeDesign,
    dependencies=['ingredient_analyzer'],
    templates=TemplateConfig(
        system_template='smoke/system/designer.jinja',
        user_template='smoke/prompts/design_recipe.jinja'
    ),
    storage_pattern='workflow/{workflow_id}/recipe_design',
    storage_type=StorageType.KV,
    quality_threshold=0.7,
    allow_refinement=True,
    max_refinements=2,
    model_config=ModelParameters(
        temperature=0.8,
        max_tokens=400
    )
)

register_phase('smoke.recipe_designer', designer_phase)

# Phase 3: Recipe Crafter
crafter_phase = PhaseDefinition(
    phase_name='recipe_crafter',
    domain='smoke',
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
    input_schema=None,  # Uses output from designer
    output_schema=DetailedRecipe,
    dependencies=['recipe_designer'],
    templates=TemplateConfig(
        system_template='smoke/system/crafter.jinja',
        user_template='smoke/prompts/craft_instructions.jinja'
    ),
    storage_pattern='workflow/{workflow_id}/detailed_recipe',
    storage_type=StorageType.KV,
    quality_threshold=0.75,
    allow_refinement=True,
    max_refinements=2,
    model_config=ModelParameters(
        temperature=0.6,
        max_tokens=800
    )
)

register_phase('smoke.recipe_crafter', crafter_phase)

# Phase 4: Recipe Evaluator
evaluator_phase = PhaseDefinition(
    phase_name='recipe_evaluator',
    domain='smoke',
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
    input_schema=None,  # Uses output from crafter
    output_schema=RecipeEvaluation,
    dependencies=['recipe_designer', 'recipe_crafter'],
    templates=TemplateConfig(
        system_template='smoke/system/evaluator.jinja',
        user_template='smoke/prompts/evaluate_recipe.jinja'
    ),
    storage_pattern='workflow/{workflow_id}/evaluation',
    storage_type=StorageType.KV,
    quality_threshold=0.8,
    allow_refinement=True,
    max_refinements=1,
    model_config=ModelParameters(
        temperature=0.5,
        max_tokens=400
    )
)

register_phase('smoke.recipe_evaluator', evaluator_phase)


# Helper function to create smoke workflow
def create_smoke_workflow(
    ingredients: List[str],
    dietary_restrictions: Optional[List[str]] = None,
    cuisine_preference: Optional[str] = None,
    max_cook_time: Optional[int] = None,
    workflow_id: Optional[str] = None,
    enable_refinement: bool = True
) -> tuple:
    """Create a smoke workflow for recipe generation.
    
    Args:
        ingredients: List of available ingredients
        dietary_restrictions: Optional dietary restrictions
        cuisine_preference: Optional cuisine preference
        max_cook_time: Optional maximum cooking time
        workflow_id: Optional workflow ID
        enable_refinement: Whether to enable refinement loops
        
    Returns:
        Tuple of (WorkflowDefinition, WorkflowState)
    """
    from ..core.types import WorkflowDefinition, WorkflowState
    from ..domains import build_workflow_definition
    import uuid
    
    # Generate workflow ID if not provided
    if not workflow_id:
        workflow_id = f"smoke-{uuid.uuid4().hex[:8]}"
    
    # Build workflow definition
    workflow_def = build_workflow_definition('smoke')
    # WorkflowDefinition is frozen, so we need to create a new one with refinement setting
    from dataclasses import replace
    workflow_def = replace(workflow_def, enable_refinement=enable_refinement)
    
    # Validate input and create initial state
    try:
        validated_input = SmokeWorkflowInput(
            ingredients=ingredients,
            dietary_restrictions=dietary_restrictions or [],
            cuisine_preference=cuisine_preference,
            max_cook_time=max_cook_time
        )
        input_dict = validated_input.model_dump()
    except Exception as e:
        # If validation fails, create state with error
        initial_state = WorkflowState(
            workflow_def=workflow_def,
            workflow_id=workflow_id,
            domain='smoke',
            current_phase='ingredient_analyzer',
            current_node='error',
            completed_phases=set(),
            phase_outputs={},
            domain_data={
                'ingredients': ingredients,
                'dietary_restrictions': dietary_restrictions or [],
                'cuisine_preference': cuisine_preference,
                'max_cook_time': max_cook_time,
                'error': f'Input validation failed: {str(e)}',
                'error_node': 'input_validation',
                'error_time': datetime.now().isoformat()
            },
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        return workflow_def, initial_state
    
    # Create initial state with validated input
    initial_state = WorkflowState(
        workflow_def=workflow_def,
        workflow_id=workflow_id,
        domain='smoke',
        current_phase='ingredient_analyzer',
        current_node='dependency_check',
        completed_phases=set(),
        phase_outputs={},
        domain_data={
            'ingredients': ingredients,
            'dietary_restrictions': dietary_restrictions or [],
            'cuisine_preference': cuisine_preference,
            'max_cook_time': max_cook_time,
            'input': input_dict
        },
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    return workflow_def, initial_state