"""GraphToolkit Validation Atomic Nodes.

Validation nodes for schema checking and quality gates.
"""

import importlib
import importlib.util
import json
import logging
import sys
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional

from ...core.factory import register_node_class
from ...core.types import ValidationResult, WorkflowState
from ..base import AtomicNode, BaseNode, GraphRunContext, NonRetryableError

logger = logging.getLogger(__name__)


@dataclass
class SchemaValidationNode(AtomicNode[WorkflowState, Any, ValidationResult]):
    """Validate LLM output against output schema.
    Triggers refinement on failure, not retry.
    """
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> ValidationResult:
        """Validate phase output against schema."""
        phase_name = ctx.state.current_phase
        phase_def = ctx.state.workflow_def.phases.get(phase_name)
        if not phase_def:
            raise NonRetryableError(f'Phase {phase_name} not found')
        
        # Get output data from state
        output_key = f'{phase_name}_llm_response'
        output_data = ctx.state.domain_data.get(output_key)
        
        if output_data is None:
            return ValidationResult(
                valid=False,
                errors=['No output data to validate'],
                warnings=[],
                metadata={'phase': phase_name}
            )
        
        # Validate against output schema
        if phase_def.output_schema:
            try:
                # If output is a string, try to parse as JSON
                if isinstance(output_data, str):
                    try:
                        output_data = json.loads(output_data)
                    except json.JSONDecodeError:
                        pass  # Keep as string
                
                # Validate with Pydantic schema
                if isinstance(output_data, dict):
                    validated = phase_def.output_schema(**output_data)
                elif isinstance(output_data, phase_def.output_schema):
                    validated = output_data
                else:
                    # Try to coerce
                    validated = phase_def.output_schema(output_data)
                
                # Store validated output
                ctx.state.domain_data[f'{phase_name}_validated'] = validated
                
                return ValidationResult(
                    valid=True,
                    errors=[],
                    warnings=[],
                    metadata={
                        'phase': phase_name,
                        'schema': phase_def.output_schema.__name__
                    }
                )
                
            except Exception as e:
                logger.error(f'Schema validation failed: {e}')
                from ...exceptions import SchemaValidationError
                raise SchemaValidationError(f'Schema validation failed for {phase_name}: {e}') from e
        
        # No schema to validate against
        return ValidationResult(
            valid=True,
            errors=[],
            warnings=['No schema defined for validation'],
            metadata={'phase': phase_name}
        )
    
    async def update_state(self, state: WorkflowState, result: ValidationResult) -> WorkflowState:
        """Update state with validation result."""
        phase_name = state.current_phase
        
        return replace(
            state,
            validation_results={
                **state.validation_results,
                phase_name: result
            }
        )


@dataclass
class QualityGateNode(BaseNode[WorkflowState, Any, WorkflowState]):
    """Check quality score and determine next action.
    Can trigger refinement or continue to next phase.
    """
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Check quality and route accordingly."""
        phase_name = ctx.state.current_phase
        logger.debug(f"[QualityGateNode] === ENTRY === Phase: {phase_name}")
        logger.debug(f"[QualityGateNode] Workflow: {ctx.state.workflow_id}")
        logger.debug(f"[QualityGateNode] Quality scores: {ctx.state.quality_scores}")
        logger.debug(f"[QualityGateNode] Validation results: {list(ctx.state.validation_results.keys())}")
        
        phase_def = ctx.state.workflow_def.phases.get(phase_name)
        if not phase_def:
            logger.error(f"[QualityGateNode] Phase {phase_name} not found")
            raise NonRetryableError(f'Phase {phase_name} not found')
        
        logger.debug(f"[QualityGateNode] Phase quality threshold: {phase_def.quality_threshold}")
        
        # Get validation result
        validation = ctx.state.validation_results.get(phase_name)
        logger.debug(f"[QualityGateNode] Validation result for {phase_name}: {validation}")
        
        # Calculate quality score
        logger.debug(f"[QualityGateNode] Calculating quality score for {phase_name}")
        quality_score = self._calculate_quality_score(ctx.state, validation)
        logger.debug(f"[QualityGateNode] Calculated quality score: {quality_score}")
        
        # Update state with quality score
        new_state = replace(
            ctx.state,
            quality_scores={
                **ctx.state.quality_scores,
                phase_name: quality_score
            }
        )
        
        # Check if quality meets threshold
        meets_threshold = quality_score >= phase_def.quality_threshold
        logger.debug(f"[QualityGateNode] Quality check: {quality_score} >= {phase_def.quality_threshold} = {meets_threshold}")
        
        if meets_threshold:
            logger.info(f'[QualityGateNode] Quality gate PASSED for {phase_name}: {quality_score:.2f}')
            logger.debug(f"[QualityGateNode] === EXIT === Returning NextPhaseNode")
            # Continue to next phase
            from .control import NextPhaseNode
            return NextPhaseNode()
        
        # Check if we can refine
        refinement_count = ctx.state.refinement_count.get(phase_name, 0)
        logger.debug(f"[QualityGateNode] Current refinement count for {phase_name}: {refinement_count}")
        logger.debug(f"[QualityGateNode] Allow refinement: {phase_def.allow_refinement}, Max refinements: {phase_def.max_refinements}")
        
        can_refine = (
            phase_def.allow_refinement and
            refinement_count < phase_def.max_refinements
        )
        logger.debug(f"[QualityGateNode] Can refine: {can_refine}")
        
        if can_refine:
            logger.info(f'[QualityGateNode] Quality gate FAILED, triggering refinement for {phase_name}')
            logger.debug(f'[QualityGateNode] Score {quality_score:.2f} < threshold {phase_def.quality_threshold}')
            logger.debug(f'[QualityGateNode] Refinement {refinement_count + 1}/{phase_def.max_refinements}')
            # Trigger refinement
            from .control import RefinementNode
            feedback = self._generate_feedback(validation, quality_score)
            logger.debug(f"[QualityGateNode] Feedback: {feedback[:100] if feedback else 'None'}...")
            logger.debug(f"[QualityGateNode] === EXIT === Returning RefinementNode")
            return RefinementNode(feedback=feedback)
        
        # Cannot refine further, fail the workflow
        logger.error(f'[QualityGateNode] Quality below threshold and max refinements reached for {phase_name}')
        logger.error(f"[QualityGateNode] Quality {quality_score} < {phase_def.quality_threshold} is unacceptable")
        from ...exceptions import WorkflowError
        raise WorkflowError(f"Quality threshold not met for {phase_name}: {quality_score} < {phase_def.quality_threshold} after max refinements")
        
        # Mark that we accepted below threshold
        new_state = replace(
            new_state,
            domain_data={
                **new_state.domain_data,
                f'{phase_name}_accepted_below_threshold': True
            }
        )
        
        # Continue to next phase
        logger.debug(f"[QualityGateNode] === EXIT === Returning NextPhaseNode (below threshold but accepted)")
        from .control import NextPhaseNode
        return NextPhaseNode()
    
    def _calculate_quality_score(self, state: WorkflowState, validation: Optional[ValidationResult]) -> float:
        """Calculate quality score based on validation and other factors."""
        if not validation:
            return 0.5  # Default score if no validation
        
        if not validation.valid:
            return 0.0  # Failed validation
        
        # Check for quality score in metadata
        if validation.metadata and 'quality_score' in validation.metadata:
            return validation.metadata['quality_score']
        
        # Calculate based on errors and warnings
        base_score = 1.0
        
        # Deduct for errors (shouldn't have any if valid)
        base_score -= len(validation.errors) * 0.2
        
        # Deduct for warnings
        base_score -= len(validation.warnings) * 0.05
        
        return max(0.0, min(1.0, base_score))
    
    def _generate_feedback(self, validation: Optional[ValidationResult], quality_score: float) -> str:
        """Generate refinement feedback."""
        feedback_parts = [
            f'Quality score: {quality_score:.2f}',
            'Please refine the output to improve quality.'
        ]
        
        if validation:
            if validation.errors:
                feedback_parts.append(f"Errors: {', '.join(validation.errors)}")
            if validation.warnings:
                feedback_parts.append(f"Warnings: {', '.join(validation.warnings)}")
        
        return '\n'.join(feedback_parts)


@dataclass
class DependencyValidationNode(AtomicNode[WorkflowState, Any, bool]):
    """Validate that required dependencies are available.
    """
    required_tools: List[str]
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> bool:
        """Check if all required tools are available."""
        # Check in domain_data for tool availability
        available_tools = ctx.state.domain_data.get('available_tools', [])
        
        missing = []
        for tool in self.required_tools:
            if tool not in available_tools:
                missing.append(tool)
        
        if missing:
            logger.error(f'Missing required tools: {missing}')
            from ...exceptions import DependencyError
            raise DependencyError(f'Missing required tools: {missing}')
        
        return True


@dataclass
class DataValidationNode(AtomicNode[WorkflowState, Any, ValidationResult]):
    """Validate data structure and content.
    """
    data_field: str
    validation_rules: Optional[Dict[str, Any]] = None
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> ValidationResult:
        """Validate data from state."""
        data = ctx.state.domain_data.get(self.data_field)
        
        if data is None:
            return ValidationResult(
                valid=False,
                errors=[f"Data field '{self.data_field}' not found"],
                warnings=[],
                metadata={'field': self.data_field}
            )
        
        errors = []
        warnings = []
        
        # Apply validation rules
        if self.validation_rules:
            for rule_name, rule_config in self.validation_rules.items():
                if rule_name == 'required_fields':
                    # Check required fields
                    for field in rule_config:
                        if isinstance(data, dict) and field not in data:
                            errors.append(f"Required field '{field}' missing")
                
                elif rule_name == 'min_length':
                    # Check minimum length
                    if len(data) < rule_config:
                        warnings.append(f'Data length {len(data)} below minimum {rule_config}')
                
                elif rule_name == 'max_length':
                    # Check maximum length
                    if len(data) > rule_config:
                        errors.append(f'Data length {len(data)} exceeds maximum {rule_config}')
                
                elif rule_name == 'type':
                    # Check data type
                    expected_type = rule_config
                    if not isinstance(data, expected_type):
                        errors.append(f'Expected type {expected_type.__name__}, got {type(data).__name__}')
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata={
                'field': self.data_field,
                'data_type': type(data).__name__
            }
        )


@dataclass
class SyntaxValidationNode(AtomicNode[WorkflowState, Any, ValidationResult]):
    """Validate Python syntax for generated code.
    """
    code_field: str = 'generated_code'
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> ValidationResult:
        """Validate Python syntax."""
        code = ctx.state.domain_data.get(self.code_field)
        
        if not code:
            return ValidationResult(
                valid=False,
                errors=['No code to validate'],
                warnings=[],
                metadata={'field': self.code_field}
            )
        
        import ast
        
        try:
            ast.parse(code)
            return ValidationResult(
                valid=True,
                errors=[],
                warnings=[],
                metadata={
                    'field': self.code_field,
                    'lines': len(code.splitlines())
                }
            )
        except SyntaxError as e:
            return ValidationResult(
                valid=False,
                errors=[f'Syntax error at line {e.lineno}: {e.msg}'],
                warnings=[],
                metadata={
                    'field': self.code_field,
                    'line': e.lineno,
                    'offset': e.offset
                }
            )


@dataclass
class ImportValidationNode(AtomicNode[WorkflowState, Any, ValidationResult]):
    """Validate that imports in generated code are available.
    Integrates with agentoolkit's existing capabilities.
    """
    code_field: str = 'generated_code'
    check_stdlib: bool = True
    check_installed: bool = True
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> ValidationResult:
        """Verify all imports in the code are available."""
        code = ctx.state.domain_data.get(self.code_field)
        
        if not code:
            return ValidationResult(
                valid=False,
                errors=['No code to validate'],
                warnings=[],
                metadata={'field': self.code_field}
            )
        
        import ast
        
        errors = []
        warnings = []
        imports_found = []
        
        try:
            # Parse the code to extract imports
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name
                        imports_found.append(module_name)
                        if not self._check_import(module_name):
                            errors.append(f"Module '{module_name}' not found")
                            
                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module
                    if module_name:
                        imports_found.append(module_name)
                        if not self._check_import(module_name):
                            errors.append(f"Module '{module_name}' not found")
                        else:
                            # Check specific imports from the module
                            for alias in node.names:
                                import_name = alias.name
                                if import_name != '*' and not self._check_from_import(module_name, import_name):
                                    warnings.append(f"Cannot verify '{import_name}' from '{module_name}'")
            
            # Check for common modules that might be needed
            if 'pydantic' in code and 'pydantic' not in imports_found:
                warnings.append("Code uses 'pydantic' but no import found")
            
            if 'typing' in code and 'typing' not in imports_found:
                warnings.append("Code uses type hints but 'typing' not imported")
                
            return ValidationResult(
                valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metadata={
                    'field': self.code_field,
                    'imports_found': imports_found,
                    'total_imports': len(imports_found)
                }
            )
            
        except SyntaxError as e:
            return ValidationResult(
                valid=False,
                errors=[f'Cannot parse code for import validation: {e}'],
                warnings=[],
                metadata={'field': self.code_field}
            )
    
    def _check_import(self, module_name: str) -> bool:
        """Check if a module can be imported."""
        import importlib.util
        
        # Check if it's a built-in module
        if module_name in sys.builtin_module_names:
            return True
        
        # Try to find the module spec
        try:
            spec = importlib.util.find_spec(module_name)
            return spec is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False
    
    def _check_from_import(self, module_name: str, import_name: str) -> bool:
        """Check if a specific item can be imported from a module."""
        try:
            module = importlib.import_module(module_name)
            return hasattr(module, import_name)
        except (ImportError, ModuleNotFoundError):
            return False


# Register validation nodes
register_node_class('schema_validation', SchemaValidationNode)
register_node_class('quality_gate', QualityGateNode)
register_node_class('dependency_validation', DependencyValidationNode)
register_node_class('data_validation', DataValidationNode)
register_node_class('syntax_validation', SyntaxValidationNode)
register_node_class('import_validation', ImportValidationNode)