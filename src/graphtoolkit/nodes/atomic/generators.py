"""GraphToolkit Generator Nodes.

Advanced and simple generator nodes for different complexity levels.
Per workflow-graph-system.md lines 826-827, 876-887.
"""

import logging
from dataclasses import dataclass, replace
from typing import Any, Dict

from ...core.factory import register_node_class
from ...core.types import WorkflowState
from ..base import AtomicNode, BaseNode, GraphRunContext

logger = logging.getLogger(__name__)


@dataclass
class SimpleGeneratorNode(AtomicNode[WorkflowState, Any, str]):
    """Simple generator for normal complexity tasks.
    
    Per workflow-graph-system.md line 826:
    - Used for normal complexity generation
    - Basic template-based generation
    """
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> str:
        """Generate simple output based on templates."""
        phase_name = ctx.state.current_phase
        
        # Get input data
        input_data = ctx.state.domain_data.get(f'{phase_name}_input', {})
        
        # For simple generation, use basic templates
        template = ctx.state.domain_data.get('simple_template', '')
        
        if not template:
            # Generate based on domain
            if ctx.state.domain == 'agentool':
                return self._generate_simple_agentool(input_data)
            elif ctx.state.domain == 'api':
                return self._generate_simple_api(input_data)
            else:
                return self._generate_generic(input_data)
        
        # Render template with data
        try:
            # Simple string formatting
            return template.format(**input_data)
        except Exception as e:
            logger.error(f'Simple generation failed: {e}')
            return str(input_data)
    
    def _generate_simple_agentool(self, data: Dict[str, Any]) -> str:
        """Generate simple AgenTool code."""
        tool_name = data.get('name', 'SimpleTool')
        return f'''
from agentool import BaseTool

class {tool_name}(BaseTool):
    """Auto-generated simple tool."""
    
    def execute(self, input_data):
        # Simple implementation
        return {{"success": True, "data": input_data}}
'''
    
    def _generate_simple_api(self, data: Dict[str, Any]) -> str:
        """Generate simple API endpoint."""
        endpoint = data.get('endpoint', '/api/simple')
        return f'''
@app.route('{endpoint}', methods=['GET'])
def handle_{endpoint.replace('/', '_')}():
    """Auto-generated simple endpoint."""
    return jsonify({{"success": True, "endpoint": "{endpoint}"}})
'''
    
    def _generate_generic(self, data: Dict[str, Any]) -> str:
        """Generate generic output."""
        return f'Generated output for: {data}'
    
    async def update_state(self, state: WorkflowState, result: str) -> WorkflowState:
        """Store generated output."""
        return replace(
            state,
            domain_data={
                **state.domain_data,
                f'{state.current_phase}_output': result,
                'generated_output': result,
                'generator_type': 'simple'
            }
        )


@dataclass
class AdvancedGeneratorNode(AtomicNode[WorkflowState, Any, Any]):
    """Advanced generator for high complexity tasks.
    
    Per workflow-graph-system.md line 827:
    - Used for high complexity generation
    - Multi-step generation with validation
    - Supports iterative refinement
    """
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> Any:
        """Generate advanced output with validation and optimization."""
        phase_name = ctx.state.current_phase
        
        # Get complexity analysis from state
        complexity_data = ctx.state.domain_data.get('complexity_analysis', {})
        requirements = ctx.state.domain_data.get('requirements', {})
        
        # Multi-step generation process
        steps = []
        
        # Step 1: Analyze requirements
        analysis = self._analyze_requirements(requirements, complexity_data)
        steps.append(('analysis', analysis))
        
        # Step 2: Generate structure
        structure = self._generate_structure(analysis)
        steps.append(('structure', structure))
        
        # Step 3: Generate implementation
        implementation = await self._generate_implementation(structure, ctx)
        steps.append(('implementation', implementation))
        
        # Step 4: Optimize output
        optimized = self._optimize_output(implementation)
        steps.append(('optimization', optimized))
        
        # Return complete result
        return {
            'final_output': optimized,
            'generation_steps': steps,
            'complexity': complexity_data,
            'generator_type': 'advanced'
        }
    
    def _analyze_requirements(self, requirements: Dict, complexity: Dict) -> Dict:
        """Analyze requirements for generation."""
        return {
            'requirements': requirements,
            'complexity_level': complexity.get('level', 'high'),
            'components_needed': complexity.get('components', []),
            'patterns_identified': complexity.get('patterns', [])
        }
    
    def _generate_structure(self, analysis: Dict) -> Dict:
        """Generate structural blueprint."""
        components = analysis.get('components_needed', [])
        
        return {
            'modules': [f'Module_{comp}' for comp in components],
            'interfaces': [f'Interface_{comp}' for comp in components],
            'dependencies': analysis.get('patterns_identified', []),
            'architecture': 'modular' if len(components) > 3 else 'monolithic'
        }
    
    async def _generate_implementation(self, structure: Dict, ctx: GraphRunContext) -> str:
        """Generate actual implementation."""
        # This would use LLM for complex generation
        # For now, create a sophisticated template
        
        modules = structure.get('modules', [])
        interfaces = structure.get('interfaces', [])
        
        if ctx.state.domain == 'agentool':
            return self._generate_advanced_agentool(structure)
        elif ctx.state.domain == 'api':
            return self._generate_advanced_api(structure)
        else:
            return self._generate_advanced_generic(structure)
    
    def _generate_advanced_agentool(self, structure: Dict) -> str:
        """Generate advanced AgenTool with multiple components."""
        modules = structure.get('modules', [])
        
        code = """
from agentool import BaseAgent, Registry, Injector
from typing import Any, Dict, Optional
import asyncio

"""
        
        # Generate interfaces
        for interface in structure.get('interfaces', []):
            code += f'''
class {interface}:
    """Interface for {interface}."""
    async def execute(self, data: Any) -> Any:
        raise NotImplementedError
'''
        
        # Generate modules
        for module in modules:
            code += f'''
class {module}({structure.get('interfaces', ['BaseInterface'])[0]}):
    """Advanced implementation of {module}."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {{}}
        self.registry = Registry()
        
    async def execute(self, data: Any) -> Any:
        # Advanced processing logic
        result = await self._process(data)
        return self._validate(result)
    
    async def _process(self, data: Any) -> Any:
        # Complex processing implementation
        return data
    
    def _validate(self, result: Any) -> Any:
        # Validation logic
        return result
'''
        
        # Generate orchestrator
        code += f'''
class AdvancedOrchestrator:
    """Orchestrates multiple modules."""
    
    def __init__(self):
        self.modules = {{
            {', '.join([f'"{m}": {m}()' for m in modules])}
        }}
    
    async def run(self, input_data: Dict) -> Dict:
        results = {{}}
        for name, module in self.modules.items():
            results[name] = await module.execute(input_data)
        return results
'''
        
        return code
    
    def _generate_advanced_api(self, structure: Dict) -> str:
        """Generate advanced API with multiple endpoints."""
        modules = structure.get('modules', [])
        
        code = """
from flask import Flask, jsonify, request
from typing import Any, Dict
import asyncio

app = Flask(__name__)

"""
        
        for module in modules:
            endpoint = f'/api/v2/{module.lower()}'
            code += f'''
@app.route('{endpoint}', methods=['GET', 'POST', 'PUT', 'DELETE'])
async def handle_{module.lower()}():
    """Advanced {module} endpoint with full CRUD."""
    method = request.method
    
    if method == 'GET':
        # Complex query logic
        filters = request.args.to_dict()
        result = await query_{module.lower()}(filters)
        return jsonify(result)
    
    elif method == 'POST':
        # Complex creation logic
        data = request.json
        result = await create_{module.lower()}(data)
        return jsonify(result), 201
    
    elif method == 'PUT':
        # Complex update logic
        data = request.json
        result = await update_{module.lower()}(data)
        return jsonify(result)
    
    elif method == 'DELETE':
        # Complex deletion logic
        id = request.args.get('id')
        result = await delete_{module.lower()}(id)
        return jsonify(result), 204

async def query_{module.lower()}(filters: Dict) -> Dict:
    # Implementation
    return {{"items": [], "total": 0}}

async def create_{module.lower()}(data: Dict) -> Dict:
    # Implementation
    return {{"id": "new", "data": data}}

async def update_{module.lower()}(data: Dict) -> Dict:
    # Implementation
    return {{"updated": True, "data": data}}

async def delete_{module.lower()}(id: str) -> Dict:
    # Implementation
    return {{"deleted": True, "id": id}}
'''
        
        return code
    
    def _generate_advanced_generic(self, structure: Dict) -> str:
        """Generate advanced generic output."""
        return f'Advanced implementation with structure: {structure}'
    
    def _optimize_output(self, implementation: str) -> str:
        """Optimize the generated output."""
        # Apply optimizations
        # - Remove redundant code
        # - Improve performance patterns
        # - Add caching where beneficial
        
        # For now, just clean up formatting
        lines = implementation.split('\n')
        optimized = []
        
        for line in lines:
            # Remove multiple blank lines
            if line.strip() or (optimized and optimized[-1].strip()):
                optimized.append(line)
        
        return '\n'.join(optimized)
    
    async def update_state(self, state: WorkflowState, result: Any) -> WorkflowState:
        """Store advanced generation result."""
        return replace(
            state,
            domain_data={
                **state.domain_data,
                f'{state.current_phase}_output': result,
                'advanced_generation': result,
                'generator_type': 'advanced'
            }
        )


@dataclass
class GeneratorRoutingNode(BaseNode[WorkflowState, Any, WorkflowState]):
    """Routes to appropriate generator based on complexity.
    
    Per workflow-graph-system.md lines 877-887:
    - Routes based on complexity analysis stored in state
    - Uses state-driven conditions
    """
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Route to appropriate generator based on complexity."""
        # Check if we have complexity routing condition
        if 'complexity_routing' in ctx.state.workflow_def.conditions:
            condition = ctx.state.workflow_def.conditions['complexity_routing']
            
            if condition.evaluate(ctx.state):
                logger.info('High complexity detected, using AdvancedGeneratorNode')
                return AdvancedGeneratorNode()
            else:
                logger.info('Normal complexity, using SimpleGeneratorNode')
                return SimpleGeneratorNode()
        
        # Fallback: check domain_data directly
        complexity = ctx.state.domain_data.get('complexity', 'normal')
        
        if complexity == 'high':
            logger.info(f'Routing to AdvancedGeneratorNode for complexity: {complexity}')
            return AdvancedGeneratorNode()
        else:
            logger.info(f'Routing to SimpleGeneratorNode for complexity: {complexity}')
            return SimpleGeneratorNode()


# Register generator nodes
register_node_class('simple_generator', SimpleGeneratorNode)
register_node_class('advanced_generator', AdvancedGeneratorNode)
register_node_class('generator_routing', GeneratorRoutingNode)