"""
GraphToolkit Iteration Operation Nodes.

Specific iteration nodes for common domain operations.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, replace
import logging

from ..iteration import IterableNode
from ..base import GraphRunContext, NonRetryableError
from ...core.types import WorkflowState
from ...core.factory import register_node_class

logger = logging.getLogger(__name__)


@dataclass
class ProcessToolsNode(IterableNode[Dict[str, Any], Dict[str, Any]]):
    """
    Process multiple tools in an AgenTool workflow.
    Iterates over missing tools to generate specifications.
    """
    
    async def process_item(
        self,
        tool: Dict[str, Any],
        ctx: GraphRunContext[WorkflowState, Any]
    ) -> Dict[str, Any]:
        """Process a single tool specification."""
        tool_name = tool.get('name', 'unknown')
        logger.info(f"Processing tool: {tool_name}")
        
        # Use LLM to generate tool specification
        from agentool.core.injector import get_injector
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
                'implementation': f"# Implementation for {tool_name}",
                'tests': f"# Tests for {tool_name}"
            }
            
            logger.info(f"Generated specification for {tool_name}")
            return specification
            
        except Exception as e:
            logger.error(f"Failed to process tool {tool_name}: {e}")
            raise NonRetryableError(f"Tool processing failed: {e}")
    
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
        ctx.state = new_state
        
        return await super().on_iteration_complete(ctx)


@dataclass
class ProcessEndpointsNode(IterableNode[Dict[str, Any], Dict[str, Any]]):
    """
    Process multiple API endpoints in an API workflow.
    """
    
    async def process_item(
        self,
        endpoint: Dict[str, Any],
        ctx: GraphRunContext[WorkflowState, Any]
    ) -> Dict[str, Any]:
        """Process a single API endpoint."""
        path = endpoint.get('path', '/')
        method = endpoint.get('method', 'GET')
        logger.info(f"Processing endpoint: {method} {path}")
        
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
    """
    Process workflow steps in a Workflow domain.
    """
    
    async def process_item(
        self,
        step: Dict[str, Any],
        ctx: GraphRunContext[WorkflowState, Any]
    ) -> Dict[str, Any]:
        """Process a single workflow step."""
        step_name = step.get('name', 'unknown')
        logger.info(f"Processing workflow step: {step_name}")
        
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
    """
    Process multiple smart contracts in a Blockchain workflow.
    """
    
    async def process_item(
        self,
        contract: Dict[str, Any],
        ctx: GraphRunContext[WorkflowState, Any]
    ) -> Dict[str, Any]:
        """Process a single smart contract."""
        contract_name = contract.get('name', 'Contract')
        logger.info(f"Processing contract: {contract_name}")
        
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
            return f"// Contract for {platform}"
    
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
            f"Check for reentrancy vulnerabilities",
            f"Verify access control implementation",
            f"Review gas optimization opportunities",
            f"Validate input sanitization",
            f"Check for integer overflow/underflow"
        ]


@dataclass
class BatchValidateNode(IterableNode[Any, bool]):
    """
    Validate multiple items in batch.
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
                    logger.warning(f"Missing required field: {field}")
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
        ctx.state = new_state
        
        return await super().on_iteration_complete(ctx)


# Register iteration operation nodes
register_node_class('process_tools', ProcessToolsNode)
register_node_class('process_endpoints', ProcessEndpointsNode)
register_node_class('process_steps', ProcessStepsNode)
register_node_class('process_contracts', ProcessContractsNode)
register_node_class('batch_validate', BatchValidateNode)