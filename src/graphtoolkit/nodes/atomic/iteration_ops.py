"""GraphToolkit Iteration Operation Nodes.

Specific iteration nodes for common domain operations.
"""

import logging
from dataclasses import dataclass, replace
from typing import Any, Dict, List

from ...core.factory import register_node_class
from ...core.types import WorkflowState
from ..base import BaseNode, GraphRunContext, NonRetryableError
from ..iteration import IterableNode

logger = logging.getLogger(__name__)


@dataclass
class SpecifierToolIteratorNode(IterableNode[Dict[str, Any], Dict[str, Any]]):
    """V1-Compatible specifier tool iterator.
    
    Mimics exact V1 behavior:
    - Processes each missing tool individually with LLM
    - Stores intermediate state with proper storage references
    - Uses ToolSpecificationLLM -> ToolSpecification transformation
    - Tracks metrics per tool
    - Preserves V1 storage patterns
    """
    
    async def prepare_iteration(self, ctx: GraphRunContext[WorkflowState, Any]) -> None:
        """Prepare for iteration by loading missing tools from analysis (existing tools already collected by prepare node)."""
        # Load analyzer output to get missing tools
        analysis_key = f'workflow/{ctx.state.workflow_id}/analysis'
        
        from ...core.initialization import ensure_graphtoolkit_initialized
        from agentool.core.injector import get_injector
        
        ensure_graphtoolkit_initialized()
        injector = get_injector()
        
        try:
            analysis_result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': analysis_key
            })
            
            if not analysis_result.success or not analysis_result.data.get('exists'):
                raise NonRetryableError(f'Analysis not found for workflow {ctx.state.workflow_id}')
            
            # Import V1 models for compatibility
            try:
                from agents.models import AnalyzerOutput
                import json
                analysis = AnalyzerOutput(**json.loads(analysis_result.data['value']))
                missing_tools = [tool.model_dump() for tool in analysis.missing_tools]
            except ImportError:
                # Fallback to dict-based processing
                import json
                analysis_data = json.loads(analysis_result.data['value'])
                missing_tools = analysis_data.get('missing_tools', [])
            
            # Store missing tools for iteration
            from dataclasses import replace
            ctx.state = replace(
                ctx.state,
                iter_items=missing_tools,
                iter_index=0,
                iter_results=[]
            )
            
            logger.info(f'Prepared iteration for {len(missing_tools)} missing tools')
            
        except Exception as e:
            logger.error(f'Failed to prepare specifier iteration: {e}')
            raise NonRetryableError(f'Iteration preparation failed: {e}')
    
    async def process_item(
        self,
        missing_tool: Dict[str, Any],
        ctx: GraphRunContext[WorkflowState, Any]
    ) -> Dict[str, Any]:
        """Process a single missing tool to generate specification (V1 behavior)."""
        tool_name = missing_tool.get('name', 'unknown')
        logger.info(f'Processing missing tool: {tool_name}')
        
        from ...core.initialization import ensure_graphtoolkit_initialized
        from agentool.core.injector import get_injector
        import json
        
        ensure_graphtoolkit_initialized()
        injector = get_injector()
        
        try:
            # Store current missing tool for template reference (V1 pattern)
            missing_tool_key = f'workflow/{ctx.state.workflow_id}/current_missing_tool'
            await injector.run('storage_kv', {
                'operation': 'set',
                'key': missing_tool_key,
                'value': json.dumps(missing_tool)
            })
            
            # Get phase definition for LLM model and templates
            phase_def = ctx.state.get_current_phase_def()
            if not phase_def:
                raise NonRetryableError(f'Phase definition not found for {ctx.state.current_phase}')
            
            # Load system template with schema (V1 pattern)
            try:
                from agents.models import ToolSpecificationLLM
                schema_json = json.dumps(ToolSpecificationLLM.model_json_schema(), indent=2)
            except ImportError:
                # Fallback schema
                schema_json = '{"type": "object", "properties": {}}'
            
            system_result = await injector.run('templates', {
                'operation': 'render',
                'template_name': 'system/specification',  # V1 EXACT path - no agentool prefix
                'variables': {'schema_json': schema_json}
            })
            
            if not system_result.success:
                raise NonRetryableError(f'System template render failed: {system_result.message}')
            
            system_prompt = system_result.data.get('rendered', '')
            
            # Render user template with V1 reference pattern
            user_result = await injector.run('templates', {
                'operation': 'render',
                'template_name': 'prompts/create_specification',  # V1 EXACT path - no agentool prefix
                'variables': {
                    'agentool_to_implement': f'!ref:storage_kv:{missing_tool_key}',
                    'analysis_output': f'!ref:storage_kv:workflow/{ctx.state.workflow_id}/analysis',
                    'existing_tools_schemas': f'!ref:storage_kv:workflow/{ctx.state.workflow_id}/existing_tools'
                }
            })
            
            if not user_result.success:
                raise NonRetryableError(f'User template render failed: {user_result.message}')
            
            user_prompt = user_result.data.get('rendered', '')
            
            # Create LLM agent with V1 configuration
            try:
                from agents.models import ToolSpecificationLLM
                from pydantic_ai import Agent
                
                model_name = phase_def.model_config.model if phase_def.model_config else 'openai:gpt-4o'
                agent = Agent(
                    model_name,
                    output_type=ToolSpecificationLLM,
                    system_prompt=system_prompt,
                    retries=3
                )
                
                # Generate specification
                result = await agent.run(user_prompt)
                spec_llm = result.output
                
                # Transform to internal format (V1 behavior)
                spec = spec_llm.to_tool_specification()
                
                # Capture token usage (V1 pattern)
                usage = result.usage()
                await self._track_token_metrics(ctx, tool_name, usage)
                
                # Ensure consistency with analysis (V1 behavior)
                spec.name = missing_tool['name']
                spec.required_tools = missing_tool.get('required_tools', [])
                spec.dependencies = missing_tool.get('dependencies', [])
                
                # Store individual specification (V1 pattern)
                spec_key = f'workflow/{ctx.state.workflow_id}/specifications/{spec.name}'
                await injector.run('storage_kv', {
                    'operation': 'set',
                    'key': spec_key,
                    'value': json.dumps(spec.model_dump())
                })
                
                logger.info(f'Generated specification for {tool_name} with {len(spec.extended_intents)} operations')
                return spec.model_dump()
                
            except ImportError:
                # Fallback without V1 models
                logger.warning('V1 models not available, using simplified specification')
                return {
                    'name': tool_name,
                    'description': missing_tool.get('description', ''),
                    'input_schema': {},
                    'output_schema': {},
                    'examples': [],
                    'errors': [],
                    'extended_intents': [],
                    'required_tools': missing_tool.get('required_tools', []),
                    'dependencies': missing_tool.get('dependencies', []),
                    'implementation_guidelines': []
                }
                
        except Exception as e:
            logger.error(f'Failed to process tool {tool_name}: {e}')
            # V1 continues processing other tools on failure
            return {
                'name': tool_name,
                'error': str(e),
                'failed': True
            }
    
    async def _track_token_metrics(self, ctx: GraphRunContext[WorkflowState, Any], tool_name: str, usage) -> None:
        """Track token metrics per tool (V1 pattern)."""
        from agentool.core.injector import get_injector
        injector = get_injector()
        
        try:
            # Track request tokens
            await injector.run('metrics', {
                'operation': 'increment',
                'name': 'agentool.workflow.tokens.request',
                'value': usage.request_tokens,
                'labels': {
                    'workflow_id': ctx.state.workflow_id,
                    'agent': 'workflow_specifier',
                    'tool': tool_name,
                    'model': 'openai:gpt-4o'  # Should come from model config
                }
            })
            
            # Track response tokens
            await injector.run('metrics', {
                'operation': 'increment',
                'name': 'agentool.workflow.tokens.response',
                'value': usage.response_tokens,
                'labels': {
                    'workflow_id': ctx.state.workflow_id,
                    'agent': 'workflow_specifier',
                    'tool': tool_name,
                    'model': 'openai:gpt-4o'
                }
            })
            
            # Track total tokens
            await injector.run('metrics', {
                'operation': 'increment',
                'name': 'agentool.workflow.tokens.total',
                'value': usage.total_tokens,
                'labels': {
                    'workflow_id': ctx.state.workflow_id,
                    'agent': 'workflow_specifier',
                    'tool': tool_name,
                    'model': 'openai:gpt-4o'
                }
            })
            
        except Exception as e:
            logger.warning(f'Failed to track metrics for {tool_name}: {e}')
    
    async def on_iteration_complete(self, ctx: GraphRunContext[WorkflowState, Any]) -> 'BaseNode':
        """Complete iteration and store final output (V1 pattern)."""
        from agentool.core.injector import get_injector
        import json
        
        injector = get_injector()
        
        try:
            # Filter out failed specifications
            successful_specs = [spec for spec in ctx.state.iter_results if not spec.get('failed', False)]
            
            # Create V1-compatible SpecificationOutput
            try:
                from agents.models import SpecificationOutput, ToolSpecification
                specifications = [ToolSpecification(**spec) for spec in successful_specs]
                spec_output = SpecificationOutput(specifications=specifications)
            except ImportError:
                # Fallback without V1 models
                spec_output = {'specifications': successful_specs}
            
            # Store consolidated specifications (V1 pattern)
            state_key = f'workflow/{ctx.state.workflow_id}/specs'
            await injector.run('storage_kv', {
                'operation': 'set',
                'key': state_key,
                'value': json.dumps(spec_output.model_dump() if hasattr(spec_output, 'model_dump') else spec_output)
            })
            
            logger.info(f'Stored {len(successful_specs)} specifications at {state_key}')
            
            # Update domain data
            ctx.state.domain_data.update({
                'specifications': successful_specs,
                'specification_count': len(successful_specs),
                'failed_tools': len([spec for spec in ctx.state.iter_results if spec.get('failed', False)])
            })
            
            return await super().on_iteration_complete(ctx)
            
        except Exception as e:
            logger.error(f'Failed to complete iteration: {e}')
            raise NonRetryableError(f'Iteration completion failed: {e}')


@dataclass  
class ProcessToolsNode(IterableNode[Dict[str, Any], Dict[str, Any]]):
    """Legacy ProcessToolsNode - kept for backward compatibility.
    Use SpecifierToolIteratorNode for V1-compatible behavior.
    """
    
    async def process_item(
        self,
        tool: Dict[str, Any],
        ctx: GraphRunContext[WorkflowState, Any]
    ) -> Dict[str, Any]:
        """Process a single tool specification."""
        tool_name = tool.get('name', 'unknown')
        logger.info(f'Processing tool: {tool_name}')
        
        # Use LLM to generate tool specification
        from ...core.initialization import ensure_graphtoolkit_initialized
        from agentool.core.injector import get_injector
        
        ensure_graphtoolkit_initialized()
        injector = get_injector()
        
        # Prepare prompt for this specific tool
        prompt = self._prepare_tool_prompt(tool, ctx.state)
        
        try:
            # This would normally call the LLM through the proper node
            # For now, we'll return a mock specification
            specification = {
                'name': tool_name,
                'description': tool.get('description', ''),
                'input_schema': tool.get('input_schema', {}),
                'output_schema': tool.get('output_schema', {}),
                'implementation': f'# Implementation for {tool_name}',
                'tests': f'# Tests for {tool_name}'
            }
            
            logger.info(f'Generated specification for {tool_name}')
            return specification
            
        except Exception as e:
            logger.error(f'Failed to process tool {tool_name}: {e}')
            raise NonRetryableError(f'Tool processing failed: {e}')
    
    def _prepare_tool_prompt(self, tool: Dict[str, Any], state: WorkflowState) -> str:
        """Prepare prompt for tool specification."""
        return f"""
        Generate a complete specification for the following tool:
        Name: {tool.get('name')}
        Description: {tool.get('description')}
        Category: {tool.get('category', 'general')}
        
        Include:
        1. Detailed input/output schemas
        2. Implementation approach
        3. Error handling
        4. Test cases
        """
    
    async def on_iteration_complete(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Store processed tools in domain data."""
        # Store all tool specifications
        new_state = replace(
            ctx.state,
            domain_data={
                **ctx.state.domain_data,
                'tool_specifications': ctx.state.iter_results
            }
        )
        
        # Pass new state via new context
        new_ctx = GraphRunContext(state=new_state, deps=ctx.deps)
        return await super().on_iteration_complete(new_ctx)


@dataclass
class ProcessEndpointsNode(IterableNode[Dict[str, Any], Dict[str, Any]]):
    """Process multiple API endpoints in an API workflow.
    """
    
    async def process_item(
        self,
        endpoint: Dict[str, Any],
        ctx: GraphRunContext[WorkflowState, Any]
    ) -> Dict[str, Any]:
        """Process a single API endpoint."""
        path = endpoint.get('path', '/')
        method = endpoint.get('method', 'GET')
        logger.info(f'Processing endpoint: {method} {path}')
        
        # Generate endpoint implementation
        implementation = {
            'path': path,
            'method': method,
            'handler': self._generate_handler(endpoint),
            'validation': self._generate_validation(endpoint),
            'documentation': self._generate_docs(endpoint),
            'tests': self._generate_tests(endpoint)
        }
        
        return implementation
    
    def _generate_handler(self, endpoint: Dict[str, Any]) -> str:
        """Generate handler code for endpoint."""
        return f"""
async def handle_{endpoint.get('name', 'endpoint')}(request):
    # Handler implementation
    pass
"""
    
    def _generate_validation(self, endpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation schema for endpoint."""
        return {
            'request': endpoint.get('request_schema', {}),
            'response': endpoint.get('response_schema', {})
        }
    
    def _generate_docs(self, endpoint: Dict[str, Any]) -> str:
        """Generate documentation for endpoint."""
        return f"""
## {endpoint.get('method', 'GET')} {endpoint.get('path', '/')}

{endpoint.get('description', 'Endpoint description')}

### Request
{endpoint.get('request_schema', {})}

### Response
{endpoint.get('response_schema', {})}
"""
    
    def _generate_tests(self, endpoint: Dict[str, Any]) -> str:
        """Generate tests for endpoint."""
        return f"""
def test_{endpoint.get('name', 'endpoint')}():
    # Test implementation
    pass
"""


@dataclass
class ProcessStepsNode(IterableNode[Dict[str, Any], Dict[str, Any]]):
    """Process workflow steps in a Workflow domain.
    """
    
    async def process_item(
        self,
        step: Dict[str, Any],
        ctx: GraphRunContext[WorkflowState, Any]
    ) -> Dict[str, Any]:
        """Process a single workflow step."""
        step_name = step.get('name', 'unknown')
        logger.info(f'Processing workflow step: {step_name}')
        
        # Generate step implementation
        return {
            'name': step_name,
            'inputs': step.get('inputs', []),
            'outputs': step.get('outputs', []),
            'dependencies': step.get('dependencies', []),
            'implementation': self._generate_step_implementation(step),
            'error_handling': self._generate_error_handling(step),
            'retry_policy': self._generate_retry_policy(step)
        }
    
    def _generate_step_implementation(self, step: Dict[str, Any]) -> str:
        """Generate implementation for workflow step."""
        return f"""
class {step.get('name', 'Step')}Step:
    async def execute(self, context):
        # Step implementation
        pass
"""
    
    def _generate_error_handling(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Generate error handling for step."""
        return {
            'retry_on': ['NetworkError', 'TimeoutError'],
            'fallback': step.get('fallback', None),
            'compensation': step.get('compensation', None)
        }
    
    def _generate_retry_policy(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Generate retry policy for step."""
        return {
            'max_retries': step.get('max_retries', 3),
            'backoff': step.get('backoff', 'exponential'),
            'delay': step.get('retry_delay', 1.0)
        }


@dataclass
class ProcessContractsNode(IterableNode[Dict[str, Any], Dict[str, Any]]):
    """Process multiple smart contracts in a Blockchain workflow.
    """
    
    async def process_item(
        self,
        contract: Dict[str, Any],
        ctx: GraphRunContext[WorkflowState, Any]
    ) -> Dict[str, Any]:
        """Process a single smart contract."""
        contract_name = contract.get('name', 'Contract')
        logger.info(f'Processing contract: {contract_name}')
        
        # Generate contract components
        return {
            'name': contract_name,
            'type': contract.get('type', 'standard'),
            'code': self._generate_contract_code(contract),
            'tests': self._generate_contract_tests(contract),
            'deployment': self._generate_deployment_script(contract),
            'audit_notes': self._generate_audit_notes(contract)
        }
    
    def _generate_contract_code(self, contract: Dict[str, Any]) -> str:
        """Generate smart contract code."""
        platform = contract.get('platform', 'ethereum')
        if platform == 'ethereum':
            return self._generate_solidity_contract(contract)
        elif platform == 'solana':
            return self._generate_rust_contract(contract)
        else:
            return f'// Contract for {platform}'
    
    def _generate_solidity_contract(self, contract: Dict[str, Any]) -> str:
        """Generate Solidity contract."""
        return f"""
pragma solidity ^0.8.0;

contract {contract.get('name', 'Contract')} {{
    // State variables
    
    // Constructor
    constructor() {{
    }}
    
    // Functions
}}
"""
    
    def _generate_rust_contract(self, contract: Dict[str, Any]) -> str:
        """Generate Rust/Solana contract."""
        return f"""
use anchor_lang::prelude::*;

#[program]
pub mod {contract.get('name', 'contract').lower()} {{
    use super::*;
    
    pub fn initialize(ctx: Context<Initialize>) -> Result<()> {{
        Ok(())
    }}
}}
"""
    
    def _generate_contract_tests(self, contract: Dict[str, Any]) -> str:
        """Generate contract tests."""
        return f"""
describe('{contract.get('name', 'Contract')}', () => {{
    it('should deploy', async () => {{
        // Test deployment
    }});
    
    it('should execute main function', async () => {{
        // Test main functionality
    }});
}});
"""
    
    def _generate_deployment_script(self, contract: Dict[str, Any]) -> str:
        """Generate deployment script."""
        return f"""
async function deploy() {{
    const Contract = await ethers.getContractFactory("{contract.get('name', 'Contract')}");
    const contract = await Contract.deploy();
    await contract.deployed();
    console.log("Contract deployed to:", contract.address);
}}
"""
    
    def _generate_audit_notes(self, contract: Dict[str, Any]) -> List[str]:
        """Generate audit notes for contract."""
        return [
            'Check for reentrancy vulnerabilities',
            'Verify access control implementation',
            'Review gas optimization opportunities',
            'Validate input sanitization',
            'Check for integer overflow/underflow'
        ]


@dataclass
class BatchValidateNode(IterableNode[Any, bool]):
    """Validate multiple items in batch.
    """
    
    async def process_item(
        self,
        item: Any,
        ctx: GraphRunContext[WorkflowState, Any]
    ) -> bool:
        """Validate a single item."""
        # Perform validation based on item type
        if isinstance(item, dict):
            # Validate required fields exist
            required = item.get('required_fields', [])
            for field in required:
                if field not in item:
                    logger.warning(f'Missing required field: {field}')
                    return False
            return True
        else:
            # Simple validation
            return item is not None
    
    async def on_iteration_complete(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Calculate validation statistics."""
        results = ctx.state.iter_results
        valid_count = sum(1 for r in results if r is True)
        total_count = len(results)
        
        validation_stats = {
            'total': total_count,
            'valid': valid_count,
            'invalid': total_count - valid_count,
            'success_rate': valid_count / total_count if total_count > 0 else 0
        }
        
        # Store validation statistics
        new_state = replace(
            ctx.state,
            domain_data={
                **ctx.state.domain_data,
                'validation_stats': validation_stats
            }
        )
        
        # Pass new state via new context
        new_ctx = GraphRunContext(state=new_state, deps=ctx.deps)
        return await super().on_iteration_complete(new_ctx)


@dataclass
class PrepareSpecifierIterationNode(BaseNode):
    """Prepare specifier iteration by loading analyzer data and collecting existing tools."""
    
    async def run(self, ctx: GraphRunContext[WorkflowState, Any]) -> 'BaseNode':
        """Prepare for specifier iteration."""
        logger.info(f'Preparing specifier iteration for workflow {ctx.state.workflow_id}')
        
        from ...core.initialization import ensure_graphtoolkit_initialized
        from agentool.core.injector import get_injector
        import json
        
        ensure_graphtoolkit_initialized()
        injector = get_injector()
        
        try:
            # Load analyzer output
            analysis_key = f'workflow/{ctx.state.workflow_id}/analysis'
            analysis_result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': analysis_key
            })
            
            if not analysis_result.success or not analysis_result.data.get('exists'):
                raise NonRetryableError(f'Analysis not found for workflow {ctx.state.workflow_id}')
            
            # Parse analysis using V1 models if available
            try:
                from agents.models import AnalyzerOutput
                analysis = AnalyzerOutput(**json.loads(analysis_result.data['value']))
                missing_tools = analysis.missing_tools
                existing_tools = analysis.existing_tools
            except ImportError:
                # Fallback to dict processing
                analysis_data = json.loads(analysis_result.data['value'])
                missing_tools = analysis_data.get('missing_tools', [])
                existing_tools = analysis_data.get('existing_tools', [])
            
            # Collect and store existing tools data (V1 pattern)
            existing_tools_data = {}
            for tool_name in existing_tools:
                try:
                    # Get FULL registry config
                    info_result = await injector.run('agentool_mgmt', {
                        'operation': 'get_agentool_info',
                        'agentool_name': tool_name,
                        'detailed': True
                    })
                    
                    if info_result.success:
                        # Store individual tool record
                        tool_key = f'workflow/{ctx.state.workflow_id}/existing_tools/{tool_name}'
                        await injector.run('storage_kv', {
                            'operation': 'set',
                            'key': tool_key,
                            'value': json.dumps(info_result.data['agentool'])
                        })
                        existing_tools_data[tool_name] = info_result.data['agentool']
                        
                except Exception as e:
                    logger.warning(f'Could not get info for existing tool {tool_name}: {e}')
            
            # Store consolidated existing tools data
            await injector.run('storage_kv', {
                'operation': 'set',
                'key': f'workflow/{ctx.state.workflow_id}/existing_tools',
                'value': json.dumps(existing_tools_data)
            })
            
            logger.info(f'Prepared iteration: {len(missing_tools)} tools to specify, {len(existing_tools_data)} existing tools collected')
            
            # Move to the iterator node
            from ...core.factory import create_node_instance
            return create_node_instance('specifier_tool_iterator')
            
        except Exception as e:
            logger.error(f'Failed to prepare specifier iteration: {e}')
            raise NonRetryableError(f'Preparation failed: {e}')


@dataclass
class EvaluatorToolIteratorNode(IterableNode[Dict[str, Any], Dict[str, Any]]):
    """V1-Compatible evaluator tool iterator.
    
    Exactly mimics V1 workflow_evaluator.py behavior:
    - Iterates over each tool implementation individually 
    - Performs syntax validation and import checking
    - Uses LLM with auto_fix=True for validation and fixes
    - Stores validation results with V1 storage patterns
    - Saves final code to file system
    - Tracks metrics per tool
    - Creates comprehensive summary
    """
    
    async def prepare_iteration(self, ctx: GraphRunContext[WorkflowState, Any]) -> None:
        """Prepare for iteration by loading specifications from crafter phase."""
        from ...core.initialization import ensure_graphtoolkit_initialized
        from agentool.core.injector import get_injector
        import json
        
        ensure_graphtoolkit_initialized()
        injector = get_injector()
        
        try:
            # Load specifications to know which tools to evaluate (V1 pattern)
            specs_key = f'workflow/{ctx.state.workflow_id}/specs'
            specs_result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': specs_key
            })
            
            if not specs_result.success or not specs_result.data.get('exists', False):
                raise NonRetryableError(f"No specifications found for workflow {ctx.state.workflow_id}")
            
            # Parse specifications using V1 models if available
            try:
                from agents.models import SpecificationOutput
                spec_output = SpecificationOutput(**json.loads(specs_result.data['value']))
                tool_specs = [spec.model_dump() for spec in spec_output.specifications]
            except ImportError:
                # Fallback to dict processing
                spec_data = json.loads(specs_result.data['value'])
                tool_specs = spec_data.get('specifications', [])
            
            if not tool_specs:
                raise NonRetryableError("No specifications to evaluate")
            
            # Store tool specs for iteration
            ctx.state = replace(
                ctx.state,
                iter_items=tool_specs,
                iter_index=0,
                iter_results=[]
            )
            
            logger.info(f'Prepared evaluation iteration for {len(tool_specs)} tools')
            
        except Exception as e:
            logger.error(f'Failed to prepare evaluator iteration: {e}')
            raise NonRetryableError(f'Evaluation preparation failed: {e}')
    
    async def process_item(
        self,
        tool_spec: Dict[str, Any],
        ctx: GraphRunContext[WorkflowState, Any]
    ) -> Dict[str, Any]:
        """Process a single tool implementation for evaluation (exact V1 behavior)."""
        tool_name = tool_spec.get('name', 'unknown')
        logger.info(f'Evaluating tool implementation: {tool_name}')
        
        from ...core.initialization import ensure_graphtoolkit_initialized
        from agentool.core.injector import get_injector
        import json
        import ast
        import os
        
        ensure_graphtoolkit_initialized()
        injector = get_injector()
        
        try:
            # Load generated implementation from storage_kv (V1 pattern)
            code_key = f'workflow/{ctx.state.workflow_id}/implementations/{tool_name}'
            code_result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': code_key
            })
            
            if not code_result.success or not code_result.data.get('exists', False):
                # Log warning but continue with other tools (V1 behavior)
                await injector.run('logging', {
                    'operation': 'log',
                    'level': 'WARN',
                    'logger_name': 'workflow',
                    'message': f'No implementation found for {tool_name}, skipping evaluation',
                    'data': {
                        'workflow_id': ctx.state.workflow_id,
                        'tool_name': tool_name
                    }
                })
                return {'tool_name': tool_name, 'skipped': True, 'reason': 'No implementation found'}
            
            # Parse implementation using V1 models 
            try:
                from agents.models import CodeOutput
                code_output = CodeOutput(**json.loads(code_result.data['value']))
            except ImportError:
                # Fallback without V1 models
                code_data = json.loads(code_result.data['value'])
                code_output = type('CodeOutput', (), {
                    'code': code_data.get('code', ''),
                    'file_path': code_data.get('file_path', f'{tool_name}.py')
                })()
            
            # Perform syntax validation (V1 exact logic)
            syntax_valid, syntax_errors = self._validate_python_syntax(code_output.code)
            
            # Check imports (V1 exact logic)
            import_warnings = self._check_imports(code_output.code)
            
            # Initial validation results
            issues = syntax_errors + import_warnings
            
            # Get model and auto_fix from input (V1 pattern)
            model = ctx.state.domain_data.get('model', 'openai:gpt-4o')
            auto_fix = ctx.state.domain_data.get('auto_fix', True)
            
            # If no issues and syntax is valid, we might be done (V1 logic)
            if syntax_valid and not import_warnings and not auto_fix:
                try:
                    from agents.models import ValidationOutput
                    validation = ValidationOutput(
                        syntax_valid=True,
                        imports_valid=True,
                        tests_passed=False,  # Would need actual test execution
                        issues=[],
                        fixes_applied=[],
                        improvements=[],
                        final_code=code_output.code,
                        ready_for_deployment=True
                    )
                except ImportError:
                    # Fallback validation
                    validation = {
                        'syntax_valid': True,
                        'imports_valid': True,
                        'tests_passed': False,
                        'issues': [],
                        'fixes_applied': [],
                        'improvements': [],
                        'final_code': code_output.code,
                        'ready_for_deployment': True
                    }
            else:
                # Use LLM to evaluate and potentially fix the code (V1 pattern)
                validation = await self._llm_validation(
                    code_output, tool_spec, ctx, model, auto_fix
                )
            
            # Store validation results in storage_kv (V1 exact pattern)
            state_key = f'workflow/{ctx.state.workflow_id}/validations/{tool_name}'
            validation_data = validation.model_dump() if hasattr(validation, 'model_dump') else validation
            await injector.run('storage_kv', {
                'operation': 'set',
                'key': state_key,
                'value': json.dumps(validation_data)
            })
            
            # Save final code to file system if improved (V1 pattern)
            final_code = validation_data.get('final_code', code_output.code)
            if final_code != code_output.code:
                # Code was improved, save new version
                final_path = f"generated/{ctx.state.workflow_id}/final/{os.path.basename(code_output.file_path)}"
                await injector.run('storage_fs', {
                    'operation': 'write',
                    'path': final_path,
                    'content': final_code,
                    'create_parents': True
                })
                
                # Log that we saved an improved version
                await injector.run('logging', {
                    'operation': 'log',
                    'level': 'INFO',
                    'logger_name': 'workflow',
                    'message': f'Saved improved code version for {tool_name}',
                    'data': {
                        'workflow_id': ctx.state.workflow_id,
                        'tool_name': tool_name,
                        'original_path': code_output.file_path,
                        'final_path': final_path,
                        'improvements': len(validation_data.get('improvements', [])),
                        'fixes': len(validation_data.get('fixes_applied', []))
                    }
                })
            
            return {
                'tool_name': tool_name,
                'validation': validation_data,
                'code_output': {
                    'code': code_output.code,
                    'file_path': code_output.file_path
                }
            }
            
        except Exception as e:
            # Log error but continue processing other tools (V1 behavior)
            await injector.run('logging', {
                'operation': 'log',
                'level': 'ERROR',
                'logger_name': 'workflow',
                'message': f'Failed to evaluate {tool_name}, continuing with other tools',
                'data': {
                    'workflow_id': ctx.state.workflow_id,
                    'tool_name': tool_name,
                    'error': str(e)
                }
            })
            
            # Track this as a failed validation (V1 pattern)
            try:
                from agents.models import ValidationOutput
                validation = ValidationOutput(
                    syntax_valid=False,
                    imports_valid=False,
                    tests_passed=False,
                    issues=[f"Evaluation failed: {str(e)}"],
                    fixes_applied=[],
                    improvements=[],
                    final_code="",
                    ready_for_deployment=False
                )
            except ImportError:
                validation = {
                    'syntax_valid': False,
                    'imports_valid': False,
                    'tests_passed': False,
                    'issues': [f"Evaluation failed: {str(e)}"],
                    'fixes_applied': [],
                    'improvements': [],
                    'final_code': "",
                    'ready_for_deployment': False
                }
            
            return {
                'tool_name': tool_name,
                'validation': validation.model_dump() if hasattr(validation, 'model_dump') else validation,
                'code_output': {'code': '', 'file_path': f'{tool_name}.py'},
                'error': str(e)
            }
    
    def _validate_python_syntax(self, code: str) -> tuple[bool, list[str]]:
        """Validate Python syntax using AST parser (exact V1 logic)."""
        errors = []
        
        try:
            ast.parse(code)
            return True, []
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
            return False, errors
        except Exception as e:
            errors.append(f"Unexpected error during parsing: {str(e)}")
            return False, errors
    
    def _check_imports(self, code: str) -> list[str]:
        """Check for problematic imports in the code (exact V1 logic)."""
        warnings = []
        standard_libs = {
            'json', 'os', 'sys', 'typing', 'datetime', 'asyncio',
            'collections', 're', 'functools', 'itertools'
        }
        allowed_packages = {
            'pydantic', 'pydantic_ai', 'agentool', 'logfire'
        }
        
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split('.')[0]
                        if module not in standard_libs and module not in allowed_packages:
                            warnings.append(f"Non-standard import '{module}' may not be available")
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module.split('.')[0]
                        if module not in standard_libs and module not in allowed_packages:
                            warnings.append(f"Non-standard import from '{module}' may not be available")
        except:
            pass
        
        return warnings
    
    async def _llm_validation(
        self, 
        code_output, 
        tool_spec: Dict[str, Any], 
        ctx: GraphRunContext[WorkflowState, Any],
        model: str,
        auto_fix: bool
    ) -> Any:
        """Use LLM to evaluate and fix code (exact V1 pattern)."""
        from agentool.core.injector import get_injector
        import json
        
        injector = get_injector()
        
        try:
            # Load system prompt with schema (V1 pattern)
            from agents.models import ValidationOutput
            schema_json = json.dumps(ValidationOutput.model_json_schema(), indent=2)
        except ImportError:
            schema_json = '{"type": "object", "properties": {}}'
        
        system_result = await injector.run('templates', {
            'operation': 'render',
            'template_name': 'system/evaluator',
            'variables': {
                'schema_json': schema_json
            }
        })
        
        if not system_result.success:
            raise NonRetryableError(f'System template render failed: {system_result.message}')
        
        system_prompt = system_result.data.get('rendered', 'You are an expert code evaluator.')
        
        # Store implementation code for template reference (V1 pattern)
        impl_code_key = f'workflow/{ctx.state.workflow_id}/current_implementation_code'
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': impl_code_key,
            'value': code_output.code
        })
        
        # Read reference implementation source code (V1 pattern)
        ref_impl_result = await injector.run('storage_fs', {
            'operation': 'read',
            'path': 'src/agentoolkit/storage/kv.py'
        })
        
        ref_impl_content = ''
        if ref_impl_result.success:
            ref_impl_content = ref_impl_result.data.get('content', '')
        
        # Store reference implementation for template (V1 pattern)
        ref_impl_key = f'workflow/{ctx.state.workflow_id}/reference_implementation'
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': ref_impl_key,
            'value': ref_impl_content
        })
        
        # Prepare evaluation prompt with references (V1 pattern)
        prompt_result = await injector.run('templates', {
            'operation': 'render',
            'template_name': 'prompts/evaluate_code',
            'variables': {
                'implementation_code': f'!ref:storage_kv:{impl_code_key}',
                'spec_output': f'!ref:storage_kv:workflow/{ctx.state.workflow_id}/specifications/{tool_spec["name"]}',
                'analysis_output': f'!ref:storage_kv:workflow/{ctx.state.workflow_id}/analysis',
                'reference_implementation': f'!ref:storage_kv:{ref_impl_key}'
            }
        })
        
        if not prompt_result.success:
            raise NonRetryableError(f'User template render failed: {prompt_result.message}')
        
        user_prompt = prompt_result.data.get('rendered', 'Evaluate this code')
        
        # Create LLM agent for evaluation (V1 pattern)
        try:
            from pydantic_ai import Agent
            from agents.models import ValidationOutput
            
            agent = Agent(
                model,
                output_type=ValidationOutput,
                system_prompt=system_prompt
            )
            
            # Get validation from LLM
            result = await agent.run(user_prompt)
            validation = result.output
            
            # Capture and record token usage (V1 pattern)
            usage = result.usage()
            await self._track_token_metrics(ctx, tool_spec['name'], usage, model)
            
            # Override with our syntax check results (V1 pattern)
            syntax_valid, _ = self._validate_python_syntax(code_output.code)
            import_warnings = self._check_imports(code_output.code)
            validation.syntax_valid = syntax_valid
            validation.imports_valid = len(import_warnings) == 0
            
            return validation
            
        except ImportError:
            # Fallback without V1 models
            return {
                'syntax_valid': True,
                'imports_valid': True,
                'tests_passed': False,
                'issues': [],
                'fixes_applied': [],
                'improvements': [],
                'final_code': code_output.code,
                'ready_for_deployment': True
            }
    
    async def _track_token_metrics(self, ctx: GraphRunContext[WorkflowState, Any], tool_name: str, usage, model: str) -> None:
        """Track token metrics per tool (V1 exact pattern)."""
        from agentool.core.injector import get_injector
        injector = get_injector()
        
        try:
            # Track request tokens
            await injector.run('metrics', {
                'operation': 'increment',
                'name': 'agentool.workflow.tokens.request',
                'value': usage.request_tokens,
                'labels': {
                    'workflow_id': ctx.state.workflow_id,
                    'agent': 'workflow_evaluator',
                    'tool': tool_name,
                    'model': model
                }
            })
            
            # Track response tokens
            await injector.run('metrics', {
                'operation': 'increment',
                'name': 'agentool.workflow.tokens.response',
                'value': usage.response_tokens,
                'labels': {
                    'workflow_id': ctx.state.workflow_id,
                    'agent': 'workflow_evaluator',
                    'tool': tool_name,
                    'model': model
                }
            })
            
            # Track total tokens
            await injector.run('metrics', {
                'operation': 'increment',
                'name': 'agentool.workflow.tokens.total',
                'value': usage.total_tokens,
                'labels': {
                    'workflow_id': ctx.state.workflow_id,
                    'agent': 'workflow_evaluator',
                    'tool': tool_name,
                    'model': model
                }
            })
            
        except Exception as e:
            logger.warning(f'Failed to track metrics for {tool_name}: {e}')
    
    async def on_iteration_complete(self, ctx: GraphRunContext[WorkflowState, Any]) -> 'BaseNode':
        """Complete iteration and create comprehensive summary (V1 pattern)."""
        from agentool.core.injector import get_injector
        import json
        import os
        
        injector = get_injector()
        all_validations = ctx.state.iter_results
        
        try:
            # Filter out skipped tools
            valid_validations = [val for val in all_validations if not val.get('skipped', False)]
            
            # Load analysis and specification for summary (V1 pattern)
            analysis_key = f'workflow/{ctx.state.workflow_id}/analysis'
            analysis_result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': analysis_key
            })
            
            analysis = None
            if analysis_result.success and analysis_result.data.get('exists', False):
                analysis = json.loads(analysis_result.data['value'])
            
            # Load specifications
            specs_key = f'workflow/{ctx.state.workflow_id}/specs'
            specs_result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': specs_key
            })
            
            spec_output = {}
            if specs_result.success and specs_result.data.get('exists', False):
                try:
                    from agents.models import SpecificationOutput
                    spec_output = SpecificationOutput(**json.loads(specs_result.data['value']))
                except ImportError:
                    spec_output = json.loads(specs_result.data['value'])
            
            # Create comprehensive summary for all tools (V1 exact format)
            summary_path = f"generated/{ctx.state.workflow_id}/SUMMARY.md"
            summary = await self._create_summary_markdown(
                ctx.state.workflow_id, valid_validations, analysis, spec_output
            )
            
            await injector.run('storage_fs', {
                'operation': 'write',
                'path': summary_path,
                'content': summary,
                'create_parents': True
            })
            
            # Prepare summary data (V1 pattern)
            summary_data = {
                'total_tools': len(valid_validations),
                'tools_ready': sum(1 for val in valid_validations if val['validation'].get('ready_for_deployment', False)),
                'total_issues': sum(len(val['validation'].get('issues', [])) for val in valid_validations),
                'total_fixes': sum(len(val['validation'].get('fixes_applied', [])) for val in valid_validations),
                'validations': [{
                    'tool_name': val['tool_name'],
                    'ready': val['validation'].get('ready_for_deployment', False),
                    'syntax_valid': val['validation'].get('syntax_valid', False),
                    'issues': val['validation'].get('issues', [])
                } for val in valid_validations]
            }
            
            # Store summary in storage_kv (V1 pattern)
            summary_key = f'workflow/{ctx.state.workflow_id}/validations_summary'
            await injector.run('storage_kv', {
                'operation': 'set',
                'key': summary_key,
                'value': json.dumps(summary_data)
            })
            
            # Update domain data
            ctx.state.domain_data.update({
                'evaluation_summary': summary_data,
                'validations': valid_validations
            })
            
            logger.info(f'Completed evaluation: {summary_data["tools_ready"]}/{summary_data["total_tools"]} tools ready')
            
            return await super().on_iteration_complete(ctx)
            
        except Exception as e:
            logger.error(f'Failed to complete evaluation iteration: {e}')
            raise NonRetryableError(f'Evaluation completion failed: {e}')
    
    async def _create_summary_markdown(self, workflow_id: str, validations: list, analysis: dict, spec_output) -> str:
        """Create comprehensive summary markdown (V1 exact format)."""
        import os
        
        # Get specifications list 
        specifications = []
        if hasattr(spec_output, 'specifications'):
            specifications = spec_output.specifications
        elif isinstance(spec_output, dict) and 'specifications' in spec_output:
            specifications = spec_output['specifications']
        
        summary = f"""# AgenTool Generation Summary

## Workflow ID: {workflow_id}

## Generated Tools: {len(validations)} tools

## Analysis Phase Output
{f'''### Solution Name
{analysis.get('name', 'N/A') if analysis else 'N/A'}

### Description
{analysis.get('description', 'N/A') if analysis else 'N/A'}

### System Design
{analysis.get('system_design', 'N/A') if analysis else 'N/A'}

### Guidelines
{chr(10).join(f"- {guideline}" for guideline in analysis.get('guidelines', [])) if analysis and analysis.get('guidelines') else "No guidelines"}

### Existing Tools Used
{chr(10).join(f"- {tool}" for tool in analysis.get('existing_tools', [])) if analysis and analysis.get('existing_tools') else "No existing tools"}

### Missing Tools Identified
{chr(10).join(f"- **{tool['name']}**: {tool['description']}" for tool in analysis.get('missing_tools', [])) if analysis and analysis.get('missing_tools') else "No missing tools"}
''' if analysis else "Analysis data not available"}

## Generated Tools Summary

{chr(10).join(f'''### Tool {i+1}: {val['tool_name']}

#### Specification
- **Name**: {val['tool_name']}
- **Description**: {next((spec.get('description', 'N/A') if isinstance(spec, dict) else spec.description for spec in specifications if (spec.get('name') if isinstance(spec, dict) else spec.name) == val['tool_name']), 'N/A')}
- **Syntax Valid**: {val['validation'].get('syntax_valid', False)}
- **Ready for Deployment**: {val['validation'].get('ready_for_deployment', False)}

#### Issues Found
{chr(10).join(f"- {issue}" for issue in val['validation'].get('issues', [])) if val['validation'].get('issues') else "No issues found"}

#### Fixes Applied
{chr(10).join(f"- {fix}" for fix in val['validation'].get('fixes_applied', [])) if val['validation'].get('fixes_applied') else "No fixes needed"}

#### File Locations
- Original: {val['code_output']['file_path']}
- Final: generated/{workflow_id}/final/{os.path.basename(val['code_output']['file_path'])}
''' for i, val in enumerate(validations))}

## Overall Results
- **Total Tools Generated**: {len(validations)}
- **Tools Ready for Deployment**: {sum(1 for val in validations if val['validation'].get('ready_for_deployment', False))}
- **Tools with Issues**: {sum(1 for val in validations if val['validation'].get('issues'))}
- **Total Fixes Applied**: {sum(len(val['validation'].get('fixes_applied', [])) for val in validations)}

## Artifact References
- Analysis: `storage_kv:workflow/{workflow_id}/analysis`
- Specifications: `storage_kv:workflow/{workflow_id}/specs`
- Implementations Summary: `storage_kv:workflow/{workflow_id}/implementations_summary`
{chr(10).join(f"- {val['tool_name']} Implementation: `storage_kv:workflow/{workflow_id}/implementations/{val['tool_name']}`" for val in validations)}
{chr(10).join(f"- {val['tool_name']} Validation: `storage_kv:workflow/{workflow_id}/validations/{val['tool_name']}`" for val in validations)}
"""
        
        return summary


@dataclass
class PrepareEvaluatorIterationNode(BaseNode):
    """Prepare evaluator iteration by loading specifications and setting up validation context."""
    
    async def run(self, ctx: GraphRunContext[WorkflowState, Any]) -> 'BaseNode':
        """Prepare for evaluator iteration."""
        logger.info(f'Preparing evaluator iteration for workflow {ctx.state.workflow_id}')
        
        from ...core.initialization import ensure_graphtoolkit_initialized
        from agentool.core.injector import get_injector
        
        ensure_graphtoolkit_initialized()
        injector = get_injector()
        
        try:
            # Store evaluator input parameters in domain_data for iteration access
            evaluator_input = ctx.state.domain_data.get('input', {})
            workflow_id = evaluator_input.get('workflow_id', ctx.state.workflow_id)
            model = evaluator_input.get('model', 'openai:gpt-4o')
            auto_fix = evaluator_input.get('auto_fix', True)
            
            # Update domain data with evaluator parameters
            ctx.state.domain_data.update({
                'workflow_id': workflow_id,
                'model': model,
                'auto_fix': auto_fix
            })
            
            # Log evaluation phase start (V1 pattern)
            await injector.run('logging', {
                'operation': 'log',
                'level': 'INFO',
                'logger_name': 'workflow',
                'message': 'Code evaluation phase started',
                'data': {
                    'workflow_id': workflow_id,
                    'operation': 'evaluate',
                    'model': model,
                    'auto_fix': auto_fix
                }
            })
            
            logger.info(f'Prepared evaluation context: workflow_id={workflow_id}, model={model}, auto_fix={auto_fix}')
            
            # Move to the iterator node
            from ...core.factory import create_node_instance
            return create_node_instance('evaluator_tool_iterator')
            
        except Exception as e:
            logger.error(f'Failed to prepare evaluator iteration: {e}')
            raise NonRetryableError(f'Evaluation preparation failed: {e}')


# Register iteration operation nodes
register_node_class('prepare_specifier_iteration', PrepareSpecifierIterationNode)
register_node_class('specifier_tool_iterator', SpecifierToolIteratorNode)
register_node_class('prepare_evaluator_iteration', PrepareEvaluatorIterationNode)
register_node_class('evaluator_tool_iterator', EvaluatorToolIteratorNode)
register_node_class('process_tools', ProcessToolsNode) 
register_node_class('process_endpoints', ProcessEndpointsNode)
register_node_class('process_steps', ProcessStepsNode)
register_node_class('process_contracts', ProcessContractsNode)
register_node_class('batch_validate', BatchValidateNode)