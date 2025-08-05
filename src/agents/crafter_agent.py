"""AgenTool Crafter Agent - Phase 3 of the workflow.

This agent generates the actual Python implementation code based on
the specification from Phase 2.
"""

import json
from typing import Dict, Any
from agentool.core.injector import get_injector
from .base import BaseAgenToolAgent
from .models import CrafterOutput, SpecificationOutput, AnalyzerOutput


class CrafterAgent(BaseAgenToolAgent[CrafterOutput]):
    """Agent for crafting AgenTool implementations from specifications."""
    
    def __init__(self, model: str = "openai:gpt-4o"):
        """Initialize the crafter agent.
        
        Args:
            model: LLM model to use for code generation
        """
        super().__init__(
            model=model,
            system_template="crafter",
            output_type=CrafterOutput,
            name="AgenTool Implementation Crafter"
        )
        
    async def craft_implementation(
        self,
        analyzer_ref: str,
        specification_ref: str,
        workflow_id: str
    ) -> CrafterOutput:
        """Craft an AgenTool implementation from specification.
        
        Args:
            analyzer_ref: Reference to analyzer output in state storage
            specification_ref: Reference to specification output in state storage
            workflow_id: Unique identifier for this workflow run
            
        Returns:
            CrafterOutput with complete implementation code
        """
        injector = get_injector()
        
        # Load previous phase outputs from state
        analyzer_data = await self.load_from_state(analyzer_ref)
        analyzer_output = AnalyzerOutput(**analyzer_data)
        
        specification_data = await self.load_from_state(specification_ref)
        specification_output = SpecificationOutput(**specification_data)
        
        # Prepare template variables
        template_variables = {
            'specification': specification_data,  # Use raw dict for template
            'analysis_context': self._create_analysis_context(analyzer_output)
        }
        
        # Generate implementation using the LLM
        implementation = await self.generate(
            user_prompt_template='craft_implementation',
            variables=template_variables
        )
        
        # Set the references
        implementation.analyzer_ref = analyzer_ref
        implementation.specification_ref = specification_ref
        
        # Post-process the implementation
        implementation = await self._post_process_implementation(
            implementation,
            specification_output
        )
        
        # Save generated code to file system
        await self._save_code_to_file(implementation, workflow_id)
        
        # Save to state and get reference
        state_ref = await self.save_to_state(
            output=implementation,
            phase='crafter',
            workflow_id=workflow_id
        )
        
        # Log the implementation creation
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'workflow',
            'message': 'Implementation crafted',
            'data': {
                'workflow_id': workflow_id,
                'file_path': implementation.file_path,
                'lines_of_code': implementation.estimated_lines,
                'functions': len(implementation.tool_functions),
                'state_ref': state_ref
            }
        })
        
        return implementation
    
    def _create_analysis_context(self, analyzer_output: AnalyzerOutput) -> str:
        """Create a summarized analysis context for the crafter.
        
        Args:
            analyzer_output: Analysis data
            
        Returns:
            Formatted context string
        """
        context_parts = [
            f"Task: {analyzer_output.task_description}",
            f"Complexity: {analyzer_output.complexity_score}/10",
            f"Relevant tools: {', '.join(analyzer_output.relevant_tools[:5])}",
            "Patterns to follow:"
        ]
        
        for pattern_name, pattern_desc in analyzer_output.patterns_identified.items():
            context_parts.append(f"  - {pattern_name}: {pattern_desc}")
        
        return "\n".join(context_parts)
    
    async def _post_process_implementation(
        self,
        implementation: CrafterOutput,
        specification: SpecificationOutput
    ) -> CrafterOutput:
        """Post-process the generated implementation.
        
        Args:
            implementation: Generated implementation
            specification: Specification for validation
            
        Returns:
            Post-processed implementation
        """
        # Set default file path if not provided
        if not implementation.file_path:
            implementation.file_path = f"src/agentoolkit/{specification.agentool_name}.py"
        
        # Estimate lines of code if not provided
        if not implementation.estimated_lines:
            implementation.estimated_lines = len(implementation.implementation_code.split('\n'))
        
        # Ensure all tool functions are documented
        if not implementation.tool_functions:
            # Try to extract from implementation code
            implementation.tool_functions = self._extract_functions(implementation.implementation_code)
        
        return implementation
    
    def _extract_functions(self, code: str) -> list:
        """Extract function definitions from code.
        
        Args:
            code: Python code string
            
        Returns:
            List of function information dictionaries
        """
        functions = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip().startswith('async def ') or line.strip().startswith('def '):
                # Extract function name
                func_line = line.strip()
                func_name = func_line.split('(')[0].replace('async def ', '').replace('def ', '')
                
                # Find the end of the function (next def or class)
                func_lines = [line]
                for j in range(i + 1, len(lines)):
                    if lines[j].strip().startswith('def ') or lines[j].strip().startswith('class '):
                        break
                    func_lines.append(lines[j])
                
                functions.append({
                    'name': func_name,
                    'code': '\n'.join(func_lines)
                })
        
        return functions
    
    async def _save_code_to_file(self, implementation: CrafterOutput, workflow_id: str) -> None:
        """Save the generated code to the file system.
        
        Args:
            implementation: Implementation with code
            workflow_id: Workflow identifier
        """
        injector = get_injector()
        
        # Save main implementation file
        file_path = f"generated/{workflow_id}/{implementation.file_path}"
        
        await injector.run('storage_fs', {
            'operation': 'write',
            'path': file_path,
            'content': implementation.implementation_code,
            'create_parents': True
        })
        
        # Save test file if present
        if implementation.test_code:
            test_path = file_path.replace('.py', '_test.py')
            await injector.run('storage_fs', {
                'operation': 'write',
                'path': test_path,
                'content': implementation.test_code,
                'create_parents': True
            })


async def create_crafter_agent(model: str = "openai:gpt-4o") -> CrafterAgent:
    """Create and initialize a crafter agent.
    
    Args:
        model: LLM model to use
        
    Returns:
        Initialized CrafterAgent
    """
    agent = CrafterAgent(model)
    await agent.initialize()
    return agent