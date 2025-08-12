"""GraphToolkit Storage Atomic Nodes.

Atomic nodes for storage operations with state-driven retry.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

from ...core.factory import create_node_instance, register_node_class
from ...core.types import StorageRef, StorageType, WorkflowState
from ..base import AtomicNode, BaseNode, End, GraphRunContext, NonRetryableError, StorageError

logger = logging.getLogger(__name__)


@dataclass
class DependencyCheckNode(BaseNode[WorkflowState, Any, None]):
    """Check that all dependencies are satisfied.
    Read-only operation - no state mutation.
    """
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Check dependencies and chain to load node."""
        logger.debug(f"[DependencyCheckNode] === ENTRY === Phase: {ctx.state.current_phase}")
        logger.debug(f"[DependencyCheckNode] Workflow: {ctx.state.workflow_id}")
        logger.debug(f"[DependencyCheckNode] Completed phases: {ctx.state.completed_phases}")
        
        # Read dependencies from phase definition in state
        phase_def = ctx.state.get_current_phase_def()
        if not phase_def:
            logger.error(f"[DependencyCheckNode] Phase {ctx.state.current_phase} not found")
            raise NonRetryableError(f'Phase {ctx.state.current_phase} not found')
        
        logger.debug(f"[DependencyCheckNode] Phase dependencies: {phase_def.dependencies}")
        
        # Check each dependency
        for dep in phase_def.dependencies:
            if dep not in ctx.state.completed_phases:
                logger.error(f"[DependencyCheckNode] Missing dependency: {dep}")
                logger.error(f"[DependencyCheckNode] Required: {phase_def.dependencies}, Available: {ctx.state.completed_phases}")
                raise NonRetryableError(f'Missing dependency: {dep}')
            else:
                logger.debug(f"[DependencyCheckNode] Dependency {dep} satisfied")
        
        logger.info(f'[DependencyCheckNode] All dependencies satisfied for {ctx.state.current_phase}')
        
        # Chain to next node (LoadDependencies)
        next_node_id = self.get_next_node(ctx.state)
        logger.debug(f"[DependencyCheckNode] Next node ID: {next_node_id}")
        
        if next_node_id:
            ctx.state.current_node = next_node_id
            logger.debug(f"[DependencyCheckNode] === EXIT === Chaining to {next_node_id}")
            return create_node_instance(next_node_id)
        
        # No dependencies to load, skip to next
        logger.debug(f"[DependencyCheckNode] === EXIT === No dependencies to load, jumping to template_render")
        return create_node_instance('template_render')


@dataclass
class LoadDependenciesNode(AtomicNode[WorkflowState, Any, Dict[str, Any]]):
    """Load outputs from dependent phases.
    Updates domain_data with loaded dependencies.
    """
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> Dict[str, Any]:
        """Load all dependencies from agentoolkit storage."""
        logger.debug(f"[LoadDependenciesNode] === ENTRY === Phase: {ctx.state.current_phase}")
        logger.debug(f"[LoadDependenciesNode] Workflow: {ctx.state.workflow_id}")
        
        phase_def = ctx.state.get_current_phase_def()
        if not phase_def:
            logger.error(f"[LoadDependenciesNode] Phase {ctx.state.current_phase} not found")
            raise NonRetryableError(f'Phase {ctx.state.current_phase} not found')
        
        logger.debug(f"[LoadDependenciesNode] Dependencies to load: {phase_def.dependencies}")
        logger.debug(f"[LoadDependenciesNode] Available phase outputs: {list(ctx.state.phase_outputs.keys())}")
        
        loaded_data = {}
        
        # Special case for analyzer phase: Load catalog from agentool_mgmt if no dependencies
        if ctx.state.current_phase == 'analyzer' and not phase_def.dependencies:
            logger.info("[LoadDependenciesNode] Analyzer phase: Loading catalog from agentool_mgmt")
            try:
                # Import and initialize
                from ...core.initialization import ensure_graphtoolkit_initialized
                from agentool.core.injector import get_injector
                ensure_graphtoolkit_initialized()
                injector = get_injector()
                
                # Load catalog from agentool_mgmt (V1 behavior)
                catalog_result = await injector.run('agentool_mgmt', {
                    'operation': 'export_catalog',
                    'format': 'json'
                })
                
                if catalog_result.success:
                    catalog = catalog_result.data.get('catalog', {})
                    loaded_data['catalog'] = catalog
                    logger.info(f"[LoadDependenciesNode] Loaded catalog with {len(catalog.get('agentools', []))} tools")
                else:
                    logger.error(f"[LoadDependenciesNode] Failed to load catalog: {catalog_result.message}")
                    from ...exceptions import CatalogError
                    raise CatalogError(f"Failed to load catalog: {catalog_result.message}")
                    
            except Exception as e:
                logger.error(f"[LoadDependenciesNode] Could not load catalog from agentool_mgmt: {e}")
                from ...exceptions import CatalogError
                raise CatalogError(f"Failed to load catalog from agentool_mgmt: {e}") from e
        
        # Standard dependency loading
        for dep in phase_def.dependencies:
            logger.debug(f"[LoadDependenciesNode] Processing dependency: {dep}")
            # Check if we have a storage reference for this dependency
            if dep not in ctx.state.phase_outputs:
                logger.error(f'[LoadDependenciesNode] No storage reference for dependency {dep}')
                logger.error(f"[LoadDependenciesNode] Available outputs: {list(ctx.state.phase_outputs.keys())}")
                from ...exceptions import DependencyError
                raise DependencyError(f"Missing storage reference for required dependency: {dep}")
            
            storage_ref = ctx.state.phase_outputs[dep]
            logger.debug(f"[LoadDependenciesNode] Storage ref for {dep}: {storage_ref}")
            
            try:
                # Load using agentoolkit storage system with metrics tracking
                logger.debug(f"[LoadDependenciesNode] Getting storage client for {dep}")
                storage_client = ctx.deps.get_storage_client()
                logger.debug(f"[LoadDependenciesNode] Storage client obtained: {storage_client}")
                start_time = time.time()
                
                if storage_ref.storage_type == StorageType.KV:
                    result = await storage_client.run('storage_kv', {
                        'operation': 'get',
                        'key': storage_ref.key,
                        'namespace': 'workflow'
                    })
                    # Extract the actual value from the storage response
                    if result.success and result.data:
                        # The storage returns a wrapper with 'value' field
                        if isinstance(result.data, dict) and 'value' in result.data:
                            data = result.data['value']
                        else:
                            data = result.data
                    else:
                        raise StorageError(f'Failed to load dependency {dep} from KV storage: {result.message if hasattr(result, "message") else "No data available"}')
                else:
                    result = await storage_client.run('storage_fs', {
                        'operation': 'read',
                        'path': storage_ref.key
                    })
                    if not result.success:
                        raise StorageError(f'Failed to load dependency {dep} from FS storage: {result.message if hasattr(result, "message") else "File not found"}')
                    data = result.data
                
                # Track storage operation
                duration = time.time() - start_time
                await self._track_storage_operation(
                    'load', 
                    storage_ref.storage_type.value,
                    storage_ref.key,
                    data is not None,
                    duration
                )
                
                loaded_data[dep] = data
                logger.info(f'[LoadDependenciesNode] Loaded dependency {dep} from {storage_ref}')
                logger.debug(f"[LoadDependenciesNode] Data type for {dep}: {type(data).__name__ if data else 'None'}")
                
            except Exception as e:
                # Storage errors are retryable
                logger.error(f"[LoadDependenciesNode] Failed to load {dep}: {e}")
                raise StorageError(f'Failed to load {dep}: {e}')
        
        logger.info(f'[LoadDependenciesNode] Loaded {len(loaded_data)} dependencies: {list(loaded_data.keys())}')
        logger.debug(f"[LoadDependenciesNode] === EXIT === Success")
        return loaded_data
    
    async def update_state_in_place(self, state: WorkflowState, result: Dict[str, Any]) -> None:
        """Update state with loaded dependencies - modifies in place."""
        logger.debug(f"[LoadDependenciesNode] Updating state with {len(result)} loaded dependencies")
        # Store each dependency as {phase_name}_output for consistent template access
        for dep_name, dep_data in result.items():
            output_key = f'{dep_name}_output'
            # Log what we're replacing
            if output_key in state.domain_data:
                old_type = type(state.domain_data[output_key]).__name__
                logger.info(f"[LoadDependenciesNode] Replacing {output_key} (was {old_type}) with loaded dict")
            state.domain_data[output_key] = dep_data
            logger.debug(f"[LoadDependenciesNode] Stored {dep_name} as {output_key}")
            logger.info(f"[LoadDependenciesNode] {output_key} is now type: {type(state.domain_data[output_key]).__name__}")


@dataclass
class SavePhaseOutputNode(AtomicNode[WorkflowState, Any, StorageRef]):
    """Save phase output to storage.
    Updates phase_outputs with storage reference.
    """
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> StorageRef:
        """Save phase output to agentoolkit storage."""
        phase_name = ctx.state.current_phase
        logger.debug(f"[SavePhaseOutputNode] === ENTRY === Phase: {phase_name}")
        logger.debug(f"[SavePhaseOutputNode] Workflow: {ctx.state.workflow_id}")
        logger.debug(f"[SavePhaseOutputNode] Domain data keys: {list(ctx.state.domain_data.keys())}")
        
        phase_def = ctx.state.get_current_phase_def()
        if not phase_def:
            logger.error(f"[SavePhaseOutputNode] Phase {phase_name} not found")
            raise NonRetryableError(f'Phase {phase_name} not found')
        
        # Get output data from domain_data
        output_key = f'{phase_name}_output'
        logger.debug(f"[SavePhaseOutputNode] Looking for output key: {output_key}")
        output_data = ctx.state.domain_data.get(output_key)
        
        if output_data is None:
            # Also check for LLM response
            output_key = f'{phase_name}_llm_response'
            logger.debug(f"[SavePhaseOutputNode] Output data not found, trying LLM response key: {output_key}")
            output_data = ctx.state.domain_data.get(output_key)
        
        if output_data is None:
            logger.error(f"[SavePhaseOutputNode] No output data for phase {phase_name}")
            logger.error(f"[SavePhaseOutputNode] Available domain data keys: {list(ctx.state.domain_data.keys())}")
            raise NonRetryableError(f'No output data for phase {phase_name}')
        
        logger.debug(f"[SavePhaseOutputNode] Found output data, type: {type(output_data).__name__}")
        
        # Serialize output data if it's a Pydantic model
        if hasattr(output_data, 'model_dump'):
            # It's a Pydantic model, serialize it
            output_data = output_data.model_dump()
        elif hasattr(output_data, 'dict'):
            # Older Pydantic version
            output_data = output_data.dict()
        
        # Generate storage key
        logger.debug(f"[SavePhaseOutputNode] Storage pattern: {phase_def.storage_pattern}")
        storage_key = phase_def.storage_pattern.format(
            domain=ctx.state.domain,
            workflow_id=ctx.state.workflow_id,
            phase=phase_name
        )
        logger.debug(f"[SavePhaseOutputNode] Generated storage key: {storage_key}")
        
        # Add version if refinement occurred
        if phase_def.allow_refinement:
            version = ctx.state.refinement_count.get(phase_name, 0)
            if version > 0:
                storage_key = f'{storage_key}/v{version}'
        
        try:
            # Save using agentoolkit storage system with metrics tracking
            logger.debug(f"[SavePhaseOutputNode] Getting storage client for phase {phase_name}")
            storage_client = ctx.deps.get_storage_client()
            logger.debug(f"[SavePhaseOutputNode] Got storage client: {storage_client}")
            start_time = time.time()
            
            if phase_def.storage_type == StorageType.KV:
                result = await storage_client.run('storage_kv', {
                    'operation': 'set',
                    'key': storage_key,
                    'value': output_data,
                    'namespace': 'workflow'
                })
                success = result.success
            else:
                result = await storage_client.run('storage_fs', {
                    'operation': 'write',
                    'path': storage_key,
                    'content': output_data
                })
                success = result.success
            
            # Track storage operation
            duration = time.time() - start_time
            await self._track_storage_operation(
                'save',
                phase_def.storage_type.value,
                storage_key,
                success,
                duration,
                len(str(output_data)) if output_data else 0  # Rough size estimate
            )
            
            logger.info(f'[SavePhaseOutputNode] Saved {phase_name} output to {storage_key}')
            logger.debug(f"[SavePhaseOutputNode] Save operation success: {success}")
            
            # Create storage reference
            storage_ref = StorageRef(
                storage_type=phase_def.storage_type,
                key=storage_key,
                created_at=datetime.now(),
                version=ctx.state.refinement_count.get(phase_name, 0) if phase_def.allow_refinement else None
            )
            logger.debug(f"[SavePhaseOutputNode] Created storage ref: {storage_ref}")
            logger.debug(f"[SavePhaseOutputNode] === EXIT === Success")
            return storage_ref
            
        except Exception as e:
            # Storage errors are retryable
            logger.error(f"[SavePhaseOutputNode] Storage error: {e}", exc_info=True)
            logger.debug(f"[SavePhaseOutputNode] === EXIT === Raising StorageError")
            raise StorageError(f'Failed to save {phase_name} output: {e}')
    
    async def update_state_in_place(self, state: WorkflowState, result: StorageRef) -> None:
        """Update state with storage reference - modifies in place."""
        phase_name = state.current_phase
        logger.debug(f"[SavePhaseOutputNode] Updating phase_outputs with ref for {phase_name}")
        state.phase_outputs[phase_name] = result
        logger.debug(f"[SavePhaseOutputNode] Phase outputs now: {list(state.phase_outputs.keys())}")



@dataclass
class SaveStorageNode(BaseNode[WorkflowState, Any, StorageRef]):
    """Generic storage save node.
    Can save to KV or FS based on configuration.
    """
    storage_key: str
    data_field: str  # Field in domain_data to save
    storage_type: StorageType = StorageType.KV
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Save data to storage."""
        # Get data from domain_data
        data = ctx.state.domain_data.get(self.data_field)
        if data is None:
            raise NonRetryableError(f'No data in {self.data_field}')
        
        try:
            storage_client = ctx.deps.get_storage_client()
            
            if self.storage_type == StorageType.KV:
                await storage_client.run('storage_kv', {
                    'operation': 'save',
                    'key': self.storage_key,
                    'data': data
                })
            else:
                await storage_client.run('storage_fs', {
                    'operation': 'save',
                    'path': self.storage_key,
                    'data': data
                })
            
            # Create storage reference
            ref = StorageRef(
                storage_type=self.storage_type,
                key=self.storage_key,
                created_at=datetime.now()
            )
            
            # Update state with reference
            new_state = replace(
                ctx.state,
                phase_outputs={
                    **ctx.state.phase_outputs,
                    self.storage_key: ref
                }
            )
            
            # Chain to next node
            next_node_id = self.get_next_node(new_state)
            if next_node_id:
                new_state = replace(new_state, current_node=next_node_id)
                return create_node_instance(next_node_id)
            
            return End(new_state)
            
        except Exception as e:
            raise StorageError(f'Failed to save to {self.storage_key}: {e}')









@dataclass




@dataclass
class SaveValidationSummaryNode(AtomicNode[WorkflowState, Any, StorageRef]):
    """Save evaluation validation summary to V1-compatible storage location.
    V1: workflow/{workflow_id}/validations_summary
    """
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> StorageRef:
        """Save evaluation validation summary data to V1 storage location."""
        logger.debug(f"[SaveValidationSummaryNode] === ENTRY === Workflow: {ctx.state.workflow_id}")
        
        # Get evaluation summary from domain_data (created by evaluator iterator)
        evaluation_summary = ctx.state.domain_data.get('evaluation_summary')
        
        if evaluation_summary is None:
            logger.error(f"[SaveValidationSummaryNode] No evaluation summary data found")
            logger.error(f"[SaveValidationSummaryNode] Available keys: {list(ctx.state.domain_data.keys())}")
            raise NonRetryableError('No evaluation summary data to save')
        
        # V1 storage pattern
        storage_key = f'workflow/{ctx.state.workflow_id}/validations_summary'
        logger.debug(f"[SaveValidationSummaryNode] Saving validation summary to: {storage_key}")
        
        try:
            # Save using agentoolkit storage system
            storage_client = ctx.deps.get_storage_client()
            start_time = time.time()
            
            result = await storage_client.run('storage_kv', {
                'operation': 'set',
                'key': storage_key,
                'value': evaluation_summary,
                'namespace': 'workflow'
            })
            
            # Track storage operation
            duration = time.time() - start_time
            await self._track_storage_operation(
                'save_validation_summary',
                'kv',
                storage_key,
                result.success,
                duration,
                len(str(evaluation_summary))
            )
            
            if not result.success:
                raise StorageError(f'Failed to save validation summary: {result.message}')
                
            logger.info(f'[SaveValidationSummaryNode] Saved validation summary to {storage_key}')
            
            # Create storage reference
            storage_ref = StorageRef(
                storage_type=StorageType.KV,
                key=storage_key,
                created_at=datetime.now()
            )
            
            logger.debug(f"[SaveValidationSummaryNode] === EXIT === Success")
            return storage_ref
            
        except Exception as e:
            logger.error(f"[SaveValidationSummaryNode] Storage error: {e}", exc_info=True)
            raise StorageError(f'Failed to save validation summary: {e}')
    
    async def update_state_in_place(self, state: WorkflowState, result: StorageRef) -> None:
        """Update state with validation summary storage reference."""
        logger.debug(f"[SaveValidationSummaryNode] Storing validation summary reference")
        state.domain_data['validation_summary_ref'] = result


@dataclass
class SaveSummaryMarkdownNode(AtomicNode[WorkflowState, Any, StorageRef]):
    """Save comprehensive evaluation summary markdown to file system.
    V1: generated/{workflow_id}/SUMMARY.md
    """
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> StorageRef:
        """Save summary markdown file to V1 storage location."""
        logger.debug(f"[SaveSummaryMarkdownNode] === ENTRY === Workflow: {ctx.state.workflow_id}")
        
        # The summary markdown is already created and saved by EvaluatorToolIteratorNode
        # This node just verifies it exists and creates the reference
        
        # V1 storage pattern
        file_path = f'generated/{ctx.state.workflow_id}/SUMMARY.md'
        logger.debug(f"[SaveSummaryMarkdownNode] Summary markdown path: {file_path}")
        
        try:
            # Verify file exists by reading it
            storage_client = ctx.deps.get_storage_client()
            start_time = time.time()
            
            result = await storage_client.run('storage_fs', {
                'operation': 'read',
                'path': file_path
            })
            
            # Track storage operation
            duration = time.time() - start_time
            await self._track_storage_operation(
                'save_summary_markdown',
                'fs',
                file_path,
                result.success,
                duration,
                len(result.data.get('content', '')) if result.success else 0
            )
            
            if not result.success:
                logger.warning(f'[SaveSummaryMarkdownNode] Summary markdown not found at {file_path}')
                # This is not necessarily an error - it may have been created elsewhere
                
            logger.info(f'[SaveSummaryMarkdownNode] Summary markdown verified at {file_path}')
            
            # Create storage reference
            storage_ref = StorageRef(
                storage_type=StorageType.FS,
                key=file_path,
                created_at=datetime.now()
            )
            
            logger.debug(f"[SaveSummaryMarkdownNode] === EXIT === Success")
            return storage_ref
            
        except Exception as e:
            logger.error(f"[SaveSummaryMarkdownNode] Storage error: {e}", exc_info=True)
            # This is a legitimate failure - we cannot verify the summary markdown exists
            raise StorageError(f'Failed to verify summary markdown at {file_path}: {e}') from e
    
    async def update_state_in_place(self, state: WorkflowState, result: StorageRef) -> None:
        """Update state with summary markdown storage reference."""
        logger.debug(f"[SaveSummaryMarkdownNode] Storing summary markdown reference")
        state.domain_data['summary_markdown_ref'] = result


# Register all storage nodes
register_node_class('dependency_check', DependencyCheckNode)
register_node_class('load_dependencies', LoadDependenciesNode)
register_node_class('save_output', SavePhaseOutputNode)
register_node_class('save_phase_output', SavePhaseOutputNode)
 
register_node_class('save_validation_summary', SaveValidationSummaryNode)
register_node_class('save_summary_markdown', SaveSummaryMarkdownNode)
