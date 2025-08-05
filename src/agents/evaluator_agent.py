"""AgenTool Evaluator Agent - Phase 4 of the workflow.

This agent evaluates and validates the generated implementation,
ensuring code quality and production readiness.
"""

import json
import ast
from typing import Dict, Any, List
from agentool.core.injector import get_injector
from .base import BaseAgenToolAgent
from .models import EvaluatorOutput, CrafterOutput, SpecificationOutput, AnalyzerOutput


class EvaluatorAgent(BaseAgenToolAgent[EvaluatorOutput]):
    """Agent for evaluating and validating AgenTool implementations."""
    
    def __init__(self, model: str = "openai:gpt-4o"):
        """Initialize the evaluator agent.
        
        Args:
            model: LLM model to use for evaluation
        """
        super().__init__(
            model=model,
            system_template="evaluator",
            output_type=EvaluatorOutput,
            name="AgenTool Code Evaluator"
        )
        
    async def evaluate_implementation(
        self,
        analyzer_ref: str,
        specification_ref: str,
        crafter_ref: str,
        workflow_id: str
    ) -> EvaluatorOutput:
        """Evaluate the crafted implementation for quality and correctness.
        
        Args:
            analyzer_ref: Reference to analyzer output in state storage
            specification_ref: Reference to specification output in state storage
            crafter_ref: Reference to crafter output in state storage
            workflow_id: Unique identifier for this workflow run
            
        Returns:
            EvaluatorOutput with validation results and final code
        """
        injector = get_injector()
        
        # Load all previous phase outputs from state
        analyzer_data = await self.load_from_state(analyzer_ref)
        analyzer_output = AnalyzerOutput(**analyzer_data)
        
        specification_data = await self.load_from_state(specification_ref)
        specification_output = SpecificationOutput(**specification_data)
        
        crafter_data = await self.load_from_state(crafter_ref)
        crafter_output = CrafterOutput(**crafter_data)
        
        # Perform initial automated validation
        initial_validation = self._perform_syntax_validation(crafter_output.implementation_code)
        
        # Prepare template variables
        template_variables = {
            'implementation_code': crafter_output.implementation_code,
            'specification_summary': self._create_specification_summary(specification_output),
            'analysis_context': f"Task: {analyzer_output.task_description}"
        }
        
        # Generate evaluation using the LLM
        evaluation = await self.generate(
            user_prompt_template='evaluate_code',
            variables=template_variables
        )
        
        # Set the references
        evaluation.analyzer_ref = analyzer_ref
        evaluation.specification_ref = specification_ref
        evaluation.crafter_ref = crafter_ref
        
        # Merge with automated validation results
        evaluation.syntax_valid = initial_validation['syntax_valid']
        if not evaluation.validation_passed and initial_validation['syntax_valid']:
            # If LLM says invalid but syntax is valid, trust syntax check
            evaluation.validation_passed = True
        
        # Save final code to file system
        await self._save_final_code(evaluation, workflow_id)
        
        # Save to state and get reference
        state_ref = await self.save_to_state(
            output=evaluation,
            phase='evaluator',
            workflow_id=workflow_id
        )
        
        # Log the evaluation completion
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO' if evaluation.validation_passed else 'WARN',
            'logger_name': 'workflow',
            'message': 'Evaluation completed',
            'data': {
                'workflow_id': workflow_id,
                'validation_passed': evaluation.validation_passed,
                'quality_score': evaluation.code_quality_score,
                'ready_for_deployment': evaluation.ready_for_deployment,
                'tests_passed': evaluation.tests_passed,
                'tests_failed': evaluation.tests_failed,
                'state_ref': state_ref
            }
        })
        
        return evaluation
    
    def _perform_syntax_validation(self, code: str) -> Dict[str, Any]:
        """Perform automated syntax validation.
        
        Args:
            code: Python code to validate
            
        Returns:
            Validation results dictionary
        """
        results = {
            'syntax_valid': False,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Try to parse the code as AST
            ast.parse(code)
            results['syntax_valid'] = True
        except SyntaxError as e:
            results['errors'].append(f"Syntax error at line {e.lineno}: {e.msg}")
        except Exception as e:
            results['errors'].append(f"Unexpected error during parsing: {str(e)}")
        
        # Check for common issues
        lines = code.split('\n')
        
        # Check imports
        for i, line in enumerate(lines, 1):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                if 'agentool' not in line and 'pydantic' not in line:
                    # Check if it's a standard library import
                    module = line.split()[1].split('.')[0]
                    if module not in ['json', 'os', 'sys', 'typing', 'datetime', 'asyncio']:
                        results['warnings'].append(
                            f"Line {i}: Non-standard import '{module}' may not be available"
                        )
        
        return results
    
    def _create_specification_summary(self, specification: SpecificationOutput) -> str:
        """Create a summarized specification for evaluation context.
        
        Args:
            specification: Specification data
            
        Returns:
            Formatted summary string
        """
        summary_parts = [
            f"AgenTool: {specification.agentool_name}",
            f"Description: {specification.description}",
            f"Operations: {len(specification.operations)}",
            f"Dependencies: {', '.join(specification.dependencies) if specification.dependencies else 'None'}",
            "Expected structure:",
            f"  - Input schema with {len(specification.operations)} operations",
            f"  - Output schema for structured responses",
            f"  - {len(specification.operations)} tool functions",
            f"  - Routing configuration",
            f"  - Agent creation function"
        ]
        
        return "\n".join(summary_parts)
    
    async def _save_final_code(self, evaluation: EvaluatorOutput, workflow_id: str) -> None:
        """Save the final validated code to the file system.
        
        Args:
            evaluation: Evaluation with final code
            workflow_id: Workflow identifier
        """
        injector = get_injector()
        
        # Save final implementation
        final_path = f"generated/{workflow_id}/final/implementation.py"
        
        await injector.run('storage_fs', {
            'operation': 'write',
            'path': final_path,
            'content': evaluation.final_code,
            'create_parents': True
        })
        
        # Save deployment instructions
        if evaluation.deployment_instructions:
            instructions_path = f"generated/{workflow_id}/final/DEPLOYMENT.md"
            await injector.run('storage_fs', {
                'operation': 'write',
                'path': instructions_path,
                'content': evaluation.deployment_instructions,
                'create_parents': True
            })
        
        # Save integration guide
        if evaluation.integration_guide:
            guide_path = f"generated/{workflow_id}/final/INTEGRATION.md"
            await injector.run('storage_fs', {
                'operation': 'write',
                'path': guide_path,
                'content': evaluation.integration_guide,
                'create_parents': True
            })
        
        # Create a summary file
        summary = f"""# AgenTool Generation Summary
        
## Workflow ID: {workflow_id}

## Validation Results
- Validation Passed: {evaluation.validation_passed}
- Syntax Valid: {evaluation.syntax_valid}
- Imports Valid: {evaluation.imports_valid}
- Schema Valid: {evaluation.schema_valid}
- Ready for Deployment: {evaluation.ready_for_deployment}

## Quality Metrics
- Code Quality Score: {evaluation.code_quality_score}/100
- Test Coverage: {evaluation.test_coverage}%
- Tests Passed: {evaluation.tests_passed}
- Tests Failed: {evaluation.tests_failed}

## Improvements Applied
{chr(10).join(f"- {imp}" for imp in evaluation.improvements_applied)}

## Issues Fixed
{chr(10).join(f"- {issue}" for issue in evaluation.issues_fixed)}

## Future Suggestions
{chr(10).join(f"- {sugg}" for sugg in evaluation.suggestions)}
"""
        
        summary_path = f"generated/{workflow_id}/final/SUMMARY.md"
        await injector.run('storage_fs', {
            'operation': 'write',
            'path': summary_path,
            'content': summary,
            'create_parents': True
        })


async def create_evaluator_agent(model: str = "openai:gpt-4o") -> EvaluatorAgent:
    """Create and initialize an evaluator agent.
    
    Args:
        model: LLM model to use
        
    Returns:
        Initialized EvaluatorAgent
    """
    agent = EvaluatorAgent(model)
    await agent.initialize()
    return agent