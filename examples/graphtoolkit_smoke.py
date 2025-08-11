#!/usr/bin/env python
"""Clean test of smoke workflow to verify template fixes."""

import asyncio
import logging
from graphtoolkit.core.executor import execute_smoke_workflow

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """Run smoke workflow with all 4 phases."""
    print("\n" + "="*80)
    print("TESTING SMOKE WORKFLOW - ALL 4 PHASES")
    print("="*80 + "\n")
    
    result = await execute_smoke_workflow(
        ingredients=["chicken breast", "rice", "garlic", "broccoli"],
        dietary_restrictions=["low-sodium"],
        cuisine_preference="Asian",
        max_cook_time=30,
        model="openai:gpt-4o-mini",
        enable_refinement=False
    )
    
    print(f"\nWorkflow completed: {result.success}")
    print(f"Completed phases: {result.state.completed_phases}")
    
    if result.success:
        # Check all 4 phases completed
        expected_phases = {'ingredient_analyzer', 'recipe_designer', 'recipe_crafter', 'recipe_evaluator'}
        if result.state.completed_phases == expected_phases:
            print("\n✅ SUCCESS: All 4 phases completed!")
            
            # Show final evaluation
            if 'recipe_evaluator_output' in result.state.domain_data:
                evaluation = result.state.domain_data['recipe_evaluator_output']
                print(f"\nFinal Evaluation:")
                # Handle both dict and Pydantic model
                if hasattr(evaluation, 'overall_score'):
                    print(f"- Overall Score: {evaluation.overall_score}")
                    print(f"- Ready to Cook: {evaluation.ready_to_cook}")
                elif isinstance(evaluation, dict):
                    print(f"- Overall Score: {evaluation.get('overall_score', 'N/A')}")
                    print(f"- Ready to Cook: {evaluation.get('ready_to_cook', 'N/A')}")
        else:
            missing = expected_phases - result.state.completed_phases
            print(f"\n❌ FAILED: Missing phases: {missing}")
            
            # Show error if any
            if 'error' in result.state.domain_data:
                print(f"\nError: {result.state.domain_data['error']}")
                print(f"Error node: {result.state.domain_data.get('error_node', 'unknown')}")
    else:
        print(f"\n❌ Workflow failed")
        if hasattr(result, 'error'):
            print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())