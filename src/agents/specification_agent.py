"""AgenTool Specification Agent - Phase 2 of the workflow.

This agent creates detailed specifications for new AgenTools based on
the analysis results from Phase 1.
"""

import json
from typing import Dict, Any
from agentool.core.injector import get_injector
from .base import BaseAgenToolAgent
from .models import SpecificationOutput, AnalyzerOutput


class SpecificationAgent(BaseAgenToolAgent[SpecificationOutput]):
    """Agent for creating AgenTool specifications from analysis."""
    
    def __init__(self, model: str = "openai:gpt-4o"):
        """Initialize the specification agent.
        
        Args:
            model: LLM model to use for specification creation
        """
        super().__init__(
            model=model,
            system_template="specification",
            output_type=SpecificationOutput,
            name="AgenTool Specification Designer"
        )
        
    async def create_specification(
        self,
        analyzer_ref: str,
        workflow_id: str
    ) -> SpecificationOutput:
        """Create an AgenTool specification based on analysis.
        
        Args:
            analyzer_ref: Reference to analyzer output in state storage
            workflow_id: Unique identifier for this workflow run
            
        Returns:
            SpecificationOutput with complete AgenTool specification
        """
        injector = get_injector()
        
        # Load analyzer output from state
        analyzer_data = await self.load_from_state(analyzer_ref)
        analyzer_output = AnalyzerOutput(**analyzer_data)
        
        # Prepare template variables with summarized analysis
        template_variables = {
            'task_description': analyzer_output.task_description,
            'analyzer_summary': analyzer_output.analysis_summary,
            'relevant_tools': analyzer_output.relevant_tools,
            'patterns': list(analyzer_output.patterns_identified.values()),
            'recommended_dependencies': analyzer_output.relevant_tools[:5]  # Top 5 as dependencies
        }
        
        # Generate specification using the LLM
        specification = await self.generate(
            user_prompt_template='create_specification',
            variables=template_variables
        )
        
        # Set the analyzer reference
        specification.analyzer_ref = analyzer_ref
        
        # Validate and enhance the specification
        specification = await self._enhance_specification(specification, analyzer_output)
        
        # Save to state and get reference
        state_ref = await self.save_to_state(
            output=specification,
            phase='specification',
            workflow_id=workflow_id
        )
        
        # Log the specification creation
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'workflow',
            'message': 'Specification created',
            'data': {
                'workflow_id': workflow_id,
                'agentool_name': specification.agentool_name,
                'operations': len(specification.operations),
                'dependencies': len(specification.dependencies),
                'state_ref': state_ref
            }
        })
        
        return specification
    
    async def _enhance_specification(
        self,
        specification: SpecificationOutput,
        analyzer_output: AnalyzerOutput
    ) -> SpecificationOutput:
        """Enhance the specification with additional validation and defaults.
        
        Args:
            specification: Initial specification from LLM
            analyzer_output: Analysis data for context
            
        Returns:
            Enhanced specification
        """
        # Ensure all required fields have sensible defaults
        if not specification.version:
            specification.version = "1.0.0"
        
        # Validate dependencies exist in the catalog
        valid_deps = []
        for dep in specification.dependencies:
            if dep in analyzer_output.available_tools:
                valid_deps.append(dep)
            else:
                print(f"Warning: Dependency {dep} not found in catalog, removing")
        specification.dependencies = valid_deps
        
        # Add default tags if none provided
        if not specification.tags:
            specification.tags = ["generated", "workflow", "agentool"]
        
        # Ensure examples have both input and output
        for example in specification.examples:
            if 'input' not in example:
                example['input'] = {'operation': 'default'}
            if 'output' not in example and 'error' not in example:
                example['output'] = {'success': True, 'message': 'Operation completed'}
        
        # Validate routing config has all operations
        if specification.routing_config:
            operation_map = specification.routing_config.get('operation_map', {})
            for op in specification.operations:
                op_name = op.get('name', '') if isinstance(op, dict) else str(op)
                if op_name and op_name not in operation_map:
                    print(f"Warning: Operation {op_name} missing from routing config")
        
        return specification


async def create_specification_agent(model: str = "openai:gpt-4o") -> SpecificationAgent:
    """Create and initialize a specification agent.
    
    Args:
        model: LLM model to use
        
    Returns:
        Initialized SpecificationAgent
    """
    agent = SpecificationAgent(model)
    await agent.initialize()
    return agent