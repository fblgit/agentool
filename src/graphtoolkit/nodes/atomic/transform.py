"""GraphToolkit Transform Nodes.

Nodes for data transformation operations like JSON parsing, code formatting, etc.
"""

import ast
import json
import logging
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional

from ...core.factory import register_node_class
from ...core.types import WorkflowState
from ..base import AtomicNode, GraphRunContext, NonRetryableError

logger = logging.getLogger(__name__)


@dataclass
class JSONParseNode(AtomicNode[WorkflowState, Any, Any]):
    """Parse JSON strings to Python objects.
    """
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> Any:
        """Parse JSON from domain data."""
        phase_name = ctx.state.current_phase
        
        # Look for JSON string in various places
        json_str = None
        
        # Check for LLM response
        llm_response_key = f'{phase_name}_llm_response'
        if llm_response_key in ctx.state.domain_data:
            json_str = ctx.state.domain_data[llm_response_key]
        
        # Check for raw JSON data
        elif 'json_data' in ctx.state.domain_data:
            json_str = ctx.state.domain_data['json_data']
        
        # Check for any string that looks like JSON
        else:
            for key, value in ctx.state.domain_data.items():
                if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                    json_str = value
                    break
        
        if json_str is None:
            raise NonRetryableError('No JSON data found to parse')
        
        try:
            # Clean up common JSON issues
            if isinstance(json_str, str):
                # Remove markdown code blocks if present
                if '```json' in json_str:
                    json_str = json_str.split('```json')[1].split('```')[0]
                elif '```' in json_str:
                    json_str = json_str.split('```')[1].split('```')[0]
                
                # Parse JSON
                return json.loads(json_str.strip())
            
            # Already parsed
            return json_str
            
        except json.JSONDecodeError as e:
            raise NonRetryableError(f'Invalid JSON: {e}')
    
    async def update_state(self, state: WorkflowState, result: Any) -> WorkflowState:
        """Store parsed JSON in domain data."""
        return replace(
            state,
            domain_data={
                **state.domain_data,
                f'{state.current_phase}_parsed': result,
                'parsed_json': result
            }
        )


@dataclass
class JSONSerializeNode(AtomicNode[WorkflowState, Any, str]):
    """Serialize Python objects to JSON strings.
    """
    indent: int = 2
    sort_keys: bool = True
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> str:
        """Serialize data to JSON."""
        phase_name = ctx.state.current_phase
        
        # Find data to serialize
        data = None
        
        # Check for phase output
        phase_output_key = f'{phase_name}_output'
        if phase_output_key in ctx.state.domain_data:
            data = ctx.state.domain_data[phase_output_key]
        
        # Check for any structured data
        elif 'data_to_serialize' in ctx.state.domain_data:
            data = ctx.state.domain_data['data_to_serialize']
        
        # Use iteration results if available
        elif ctx.state.iter_results:
            data = ctx.state.iter_results
        
        if data is None:
            raise NonRetryableError('No data found to serialize')
        
        try:
            # Convert Pydantic models to dict if needed
            if hasattr(data, 'model_dump'):
                data = data.model_dump()
            elif hasattr(data, 'dict'):
                data = data.dict()
            
            # Serialize to JSON
            return json.dumps(
                data,
                indent=self.indent,
                sort_keys=self.sort_keys,
                default=str  # Convert non-serializable objects to strings
            )
            
        except Exception as e:
            raise NonRetryableError(f'Failed to serialize to JSON: {e}')
    
    async def update_state(self, state: WorkflowState, result: str) -> WorkflowState:
        """Store JSON string in domain data."""
        return replace(
            state,
            domain_data={
                **state.domain_data,
                f'{state.current_phase}_json': result,
                'serialized_json': result
            }
        )


@dataclass
class CodeFormatNode(AtomicNode[WorkflowState, Any, str]):
    """Format Python code using black or autopep8 style.
    """
    line_length: int = 88
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> str:
        """Format Python code."""
        phase_name = ctx.state.current_phase
        
        # Find code to format
        code = None
        
        # Check for generated code
        if 'generated_code' in ctx.state.domain_data:
            code = ctx.state.domain_data['generated_code']
        elif f'{phase_name}_code' in ctx.state.domain_data:
            code = ctx.state.domain_data[f'{phase_name}_code']
        elif 'code' in ctx.state.domain_data:
            code = ctx.state.domain_data['code']
        
        if code is None:
            raise NonRetryableError('No code found to format')
        
        try:
            # First, validate it's valid Python
            ast.parse(code)
            
            # Try to use black if available
            try:
                import black
                mode = black.Mode(line_length=self.line_length)
                return black.format_str(code, mode=mode)
            except ImportError:
                pass
            
            # Try autopep8
            try:
                import autopep8
                return autopep8.fix_code(code, options={'max_line_length': self.line_length})
            except ImportError:
                pass
            
            # If no formatters available, at least ensure consistent indentation
            lines = code.split('\n')
            formatted_lines = []
            for line in lines:
                # Basic formatting - ensure 4 spaces for indentation
                if line.startswith('\t'):
                    line = line.replace('\t', '    ')
                formatted_lines.append(line)
            
            return '\n'.join(formatted_lines)
            
        except SyntaxError as e:
            raise NonRetryableError(f'Invalid Python code: {e}')
        except Exception as e:
            raise NonRetryableError(f'Failed to format code: {e}')
    
    async def update_state(self, state: WorkflowState, result: str) -> WorkflowState:
        """Store formatted code in domain data."""
        return replace(
            state,
            domain_data={
                **state.domain_data,
                f'{state.current_phase}_formatted': result,
                'formatted_code': result
            }
        )


@dataclass
class DataMergeNode(AtomicNode[WorkflowState, Any, Any]):
    """Merge multiple data sources into one.
    """
    merge_keys: List[str]  # Keys to merge from domain_data
    merge_strategy: str = 'update'  # update, append, deep_merge
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> Any:
        """Merge data from multiple sources."""
        merged = {}
        
        for key in self.merge_keys:
            if key in ctx.state.domain_data:
                data = ctx.state.domain_data[key]
                
                if self.merge_strategy == 'update':
                    # Dictionary update (shallow merge)
                    if isinstance(data, dict):
                        merged.update(data)
                    elif isinstance(data, list):
                        if 'items' not in merged:
                            merged['items'] = []
                        merged['items'].extend(data)
                    else:
                        merged[key] = data
                        
                elif self.merge_strategy == 'append':
                    # Append to lists
                    if key not in merged:
                        merged[key] = []
                    if isinstance(data, list):
                        merged[key].extend(data)
                    else:
                        merged[key].append(data)
                        
                elif self.merge_strategy == 'deep_merge':
                    # Deep merge for nested dicts
                    merged = self._deep_merge(merged, {key: data})
        
        return merged
    
    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    async def update_state(self, state: WorkflowState, result: Any) -> WorkflowState:
        """Store merged data in domain data."""
        return replace(
            state,
            domain_data={
                **state.domain_data,
                f'{state.current_phase}_merged': result,
                'merged_data': result
            }
        )


@dataclass
class DataFilterNode(AtomicNode[WorkflowState, Any, List[Any]]):
    """Filter collections based on criteria.
    """
    filter_func: Optional[Callable] = None
    filter_field: Optional[str] = None
    filter_value: Optional[Any] = None
    filter_op: str = 'equals'  # equals, contains, greater, less
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> List[Any]:
        """Filter data based on criteria."""
        # Get collection to filter
        collection = None
        
        if ctx.state.iter_results:
            collection = ctx.state.iter_results
        elif 'collection' in ctx.state.domain_data:
            collection = ctx.state.domain_data['collection']
        else:
            # Look for any list in domain data
            for value in ctx.state.domain_data.values():
                if isinstance(value, list):
                    collection = value
                    break
        
        if collection is None:
            raise NonRetryableError('No collection found to filter')
        
        filtered = []
        
        for item in collection:
            if self.filter_func:
                # Use custom filter function
                if self.filter_func(item):
                    filtered.append(item)
                    
            elif self.filter_field:
                # Filter by field value
                if isinstance(item, dict):
                    field_value = item.get(self.filter_field)
                else:
                    field_value = getattr(item, self.filter_field, None)
                
                if self._check_condition(field_value, self.filter_value, self.filter_op):
                    filtered.append(item)
            else:
                # No filter criteria, pass all
                filtered.append(item)
        
        logger.info(f'Filtered {len(collection)} items to {len(filtered)}')
        return filtered
    
    def _check_condition(self, value: Any, target: Any, op: str) -> bool:
        """Check if value meets condition."""
        if op == 'equals':
            return value == target
        elif op == 'contains':
            return target in value if value else False
        elif op == 'greater':
            return value > target if value else False
        elif op == 'less':
            return value < target if value else False
        else:
            return True
    
    async def update_state(self, state: WorkflowState, result: List[Any]) -> WorkflowState:
        """Store filtered data in domain data."""
        return replace(
            state,
            domain_data={
                **state.domain_data,
                f'{state.current_phase}_filtered': result,
                'filtered_data': result
            }
        )


# Register transform nodes
register_node_class('json_parse', JSONParseNode)
register_node_class('json_serialize', JSONSerializeNode)
register_node_class('code_format', CodeFormatNode)
register_node_class('data_merge', DataMergeNode)
register_node_class('data_filter', DataFilterNode)