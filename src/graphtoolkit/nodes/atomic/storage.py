"""GraphToolkit Storage Atomic Nodes.

Atomic nodes for storage operations with state-driven retry.
"""

import logging
import time
from dataclasses import dataclass, replace
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
        # Read dependencies from phase definition in state
        phase_def = ctx.state.get_current_phase_def()
        if not phase_def:
            raise NonRetryableError(f'Phase {ctx.state.current_phase} not found')
        
        # Check each dependency
        for dep in phase_def.dependencies:
            if dep not in ctx.state.completed_phases:
                raise NonRetryableError(f'Missing dependency: {dep}')
        
        logger.info(f'All dependencies satisfied for {ctx.state.current_phase}')
        
        # Chain to next node (LoadDependencies)
        next_node_id = self.get_next_node(ctx.state)
        if next_node_id:
            new_state = replace(ctx.state, current_node=next_node_id)
            return create_node_instance(next_node_id)
        
        # No dependencies to load, skip to next
        return create_node_instance('template_render')


@dataclass
class LoadDependenciesNode(AtomicNode[WorkflowState, Any, Dict[str, Any]]):
    """Load outputs from dependent phases.
    Updates domain_data with loaded dependencies.
    """
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> Dict[str, Any]:
        """Load all dependencies from agentoolkit storage."""
        phase_def = ctx.state.get_current_phase_def()
        if not phase_def:
            raise NonRetryableError(f'Phase {ctx.state.current_phase} not found')
        
        loaded_data = {}
        
        for dep in phase_def.dependencies:
            # Check if we have a storage reference for this dependency
            if dep not in ctx.state.phase_outputs:
                logger.warning(f'No storage reference for dependency {dep}')
                continue
            
            storage_ref = ctx.state.phase_outputs[dep]
            
            try:
                # Load using agentoolkit storage system with metrics tracking
                storage_client = ctx.deps.get_storage_client()
                start_time = time.time()
                
                if storage_ref.storage_type == StorageType.KV:
                    result = await storage_client.run('storage_kv', {
                        'operation': 'get',
                        'key': storage_ref.key,
                        'namespace': 'workflow'
                    })
                    # The result is already the output object (StorageKvOutput)
                    data = result.data if result.success else None
                else:
                    result = await storage_client.run('storage_fs', {
                        'operation': 'read',
                        'path': storage_ref.key
                    })
                    data = result.data if result.success else None
                
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
                logger.info(f'Loaded dependency {dep} from {storage_ref}')
                
            except Exception as e:
                # Storage errors are retryable
                raise StorageError(f'Failed to load {dep}: {e}')
        
        return loaded_data
    
    async def update_state(self, state: WorkflowState, result: Dict[str, Any]) -> WorkflowState:
        """Update state with loaded dependencies."""
        return replace(
            state,
            domain_data={
                **state.domain_data,
                'loaded_dependencies': result
            }
        )


@dataclass
class SavePhaseOutputNode(AtomicNode[WorkflowState, Any, StorageRef]):
    """Save phase output to storage.
    Updates phase_outputs with storage reference.
    """
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> StorageRef:
        """Save phase output to agentoolkit storage."""
        phase_name = ctx.state.current_phase
        phase_def = ctx.state.get_current_phase_def()
        if not phase_def:
            raise NonRetryableError(f'Phase {phase_name} not found')
        
        # Get output data from domain_data
        output_key = f'{phase_name}_output'
        output_data = ctx.state.domain_data.get(output_key)
        
        if output_data is None:
            # Also check for LLM response
            output_key = f'{phase_name}_llm_response'
            output_data = ctx.state.domain_data.get(output_key)
        
        if output_data is None:
            raise NonRetryableError(f'No output data for phase {phase_name}')
        
        # Generate storage key
        storage_key = phase_def.storage_pattern.format(
            domain=ctx.state.domain,
            workflow_id=ctx.state.workflow_id,
            phase=phase_name
        )
        
        # Add version if refinement occurred
        if phase_def.allow_refinement:
            version = ctx.state.refinement_count.get(phase_name, 0)
            if version > 0:
                storage_key = f'{storage_key}/v{version}'
        
        try:
            # Save using agentoolkit storage system with metrics tracking
            storage_client = ctx.deps.get_storage_client()
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
            
            logger.info(f'Saved {phase_name} output to {storage_key}')
            
            # Create storage reference
            return StorageRef(
                storage_type=phase_def.storage_type,
                key=storage_key,
                created_at=datetime.now(),
                version=ctx.state.refinement_count.get(phase_name, 0) if phase_def.allow_refinement else None
            )
            
        except Exception as e:
            # Storage errors are retryable
            raise StorageError(f'Failed to save {phase_name} output: {e}')
    
    async def update_state(self, state: WorkflowState, result: StorageRef) -> WorkflowState:
        """Update state with storage reference."""
        phase_name = state.current_phase
        return replace(
            state,
            phase_outputs={
                **state.phase_outputs,
                phase_name: result
            }
        )


@dataclass
class LoadStorageNode(BaseNode[WorkflowState, Any, Any]):
    """Generic storage load node.
    Can load from KV or FS based on configuration.
    """
    storage_key: str
    storage_type: StorageType = StorageType.KV
    required: bool = True
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Load data from agentoolkit storage."""
        try:
            storage_client = ctx.deps.get_storage_client()
            
            if self.storage_type == StorageType.KV:
                result = await storage_client.run('storage_kv', {
                    'operation': 'load',
                    'key': self.storage_key
                })
                data = result.data if result.success else None
            else:
                result = await storage_client.run('storage_fs', {
                    'operation': 'load',
                    'path': self.storage_key
                })
                data = result.data if result.success else None
            
            if data is None and self.required:
                raise NonRetryableError(f'Required key not found: {self.storage_key}')
            
            # Store in domain_data
            new_state = replace(
                ctx.state,
                domain_data={
                    **ctx.state.domain_data,
                    f'loaded_{self.storage_key}': data
                }
            )
            
            # Chain to next node
            next_node_id = self.get_next_node(new_state)
            if next_node_id:
                new_state = replace(new_state, current_node=next_node_id)
                return create_node_instance(next_node_id)
            
            return End(new_state)
            
        except Exception as e:
            raise StorageError(f'Failed to load {self.storage_key}: {e}')


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
class BatchLoadNode(BaseNode[WorkflowState, Any, Dict[str, Any]]):
    """Load multiple items in parallel.
    """
    storage_keys: List[str]
    storage_type: StorageType = StorageType.KV
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Load multiple items from storage."""
        import asyncio
        
        async def load_item(key: str) -> tuple[str, Any]:
            try:
                storage_client = ctx.deps.get_storage_client()
                
                if self.storage_type == StorageType.KV:
                    result = await storage_client.run('storage_kv', {
                        'operation': 'load',
                        'key': key
                    })
                    data = result.data if result.success else None
                else:
                    result = await storage_client.run('storage_fs', {
                        'operation': 'load',
                        'path': key
                    })
                    data = result.data if result.success else None
                    
                return key, data
            except Exception as e:
                logger.error(f'Failed to load {key}: {e}')
                return key, None
        
        # Load all items in parallel
        tasks = [load_item(key) for key in self.storage_keys]
        results = await asyncio.gather(*tasks)
        
        # Convert to dict
        loaded_data = dict(results)
        
        # Update state
        new_state = replace(
            ctx.state,
            domain_data={
                **ctx.state.domain_data,
                'batch_loaded': loaded_data
            }
        )
        
        # Chain to next node
        next_node_id = self.get_next_node(new_state)
        if next_node_id:
            new_state = replace(new_state, current_node=next_node_id)
            return create_node_instance(next_node_id)
        
        return End(new_state)


@dataclass
class BatchSaveNode(BaseNode[WorkflowState, Any, List[StorageRef]]):
    """Save multiple items in batch.
    """
    storage_prefix: str
    storage_type: StorageType = StorageType.KV
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Save multiple items to storage."""
        import asyncio
        
        # Get items from iteration results or domain data
        items = ctx.state.iter_results if ctx.state.iter_results else []
        
        if not items:
            logger.warning('No items to save in batch')
            return self._continue_chain(ctx.state, [])
        
        async def save_item(idx: int, item: Any) -> StorageRef:
            key = f'{self.storage_prefix}/{idx}'
            try:
                storage_client = ctx.deps.get_storage_client()
                
                if self.storage_type == StorageType.KV:
                    await storage_client.run('storage_kv', {
                        'operation': 'save',
                        'key': key,
                        'data': item
                    })
                else:
                    await storage_client.run('storage_fs', {
                        'operation': 'save',
                        'path': key,
                        'data': item
                    })
                
                return StorageRef(
                    storage_type=self.storage_type,
                    key=key,
                    created_at=datetime.now()
                )
            except Exception as e:
                logger.error(f'Failed to save item {idx}: {e}')
                return None
        
        # Save all items in parallel
        tasks = [save_item(i, item) for i, item in enumerate(items)]
        refs = await asyncio.gather(*tasks)
        
        # Filter out failed saves
        valid_refs = [ref for ref in refs if ref is not None]
        
        logger.info(f'Batch saved {len(valid_refs)}/{len(items)} items')
        
        return self._continue_chain(ctx.state, valid_refs)
    
    def _continue_chain(self, state: WorkflowState, refs: List[StorageRef]) -> BaseNode:
        """Continue to next node with updated state."""
        new_state = replace(
            state,
            domain_data={
                **state.domain_data,
                f'{state.current_phase}_batch_refs': refs,
                f'{state.current_phase}_batch_count': len(refs)
            }
        )
        
        next_node_id = self.get_next_node(new_state)
        if next_node_id:
            new_state = replace(new_state, current_node=next_node_id)
            return create_node_instance(next_node_id)
        
        return End(new_state)


# Register all storage nodes
register_node_class('dependency_check', DependencyCheckNode)
register_node_class('load_dependencies', LoadDependenciesNode)
register_node_class('save_output', SavePhaseOutputNode)
register_node_class('save_phase_output', SavePhaseOutputNode)
register_node_class('load_storage', LoadStorageNode)
register_node_class('save_storage', SaveStorageNode)
register_node_class('batch_load', BatchLoadNode)
register_node_class('batch_save', BatchSaveNode)