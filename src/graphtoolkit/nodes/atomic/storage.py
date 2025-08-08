"""
GraphToolkit Storage Atomic Nodes.

Atomic nodes for storage operations with state-driven retry.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, replace
from datetime import datetime
import logging

from ..base import (
    AtomicNode,
    BaseNode,
    StorageError,
    NonRetryableError,
    GraphRunContext,
    End
)
from ...core.types import (
    WorkflowState,
    StorageRef,
    StorageType
)
from ...core.factory import register_node_class, create_node_instance


logger = logging.getLogger(__name__)


@dataclass
class DependencyCheckNode(BaseNode[WorkflowState, Any, None]):
    """
    Check that all dependencies are satisfied.
    Read-only operation - no state mutation.
    """
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Check dependencies and chain to load node."""
        # Read dependencies from phase definition in state
        phase_def = ctx.state.workflow_def.phases.get(ctx.state.current_phase)
        if not phase_def:
            raise NonRetryableError(f"Phase {ctx.state.current_phase} not found")
        
        # Check each dependency
        for dep in phase_def.dependencies:
            if dep not in ctx.state.completed_phases:
                raise NonRetryableError(f"Missing dependency: {dep}")
        
        logger.info(f"All dependencies satisfied for {ctx.state.current_phase}")
        
        # Chain to next node (LoadDependencies)
        next_node_id = self.get_next_node(ctx.state)
        if next_node_id:
            new_state = replace(ctx.state, current_node=next_node_id)
            return create_node_instance(next_node_id)
        
        # No dependencies to load, skip to next
        return create_node_instance('template_render')


@dataclass
class LoadDependenciesNode(AtomicNode[WorkflowState, Any, Dict[str, Any]]):
    """
    Load outputs from dependent phases.
    Updates domain_data with loaded dependencies.
    """
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> Dict[str, Any]:
        """Load all dependencies from storage."""
        phase_def = ctx.state.workflow_def.phases.get(ctx.state.current_phase)
        if not phase_def:
            raise NonRetryableError(f"Phase {ctx.state.current_phase} not found")
        
        loaded_data = {}
        
        for dep in phase_def.dependencies:
            # Check if we have a storage reference for this dependency
            if dep not in ctx.state.phase_outputs:
                logger.warning(f"No storage reference for dependency {dep}")
                continue
            
            storage_ref = ctx.state.phase_outputs[dep]
            
            try:
                # Load from appropriate storage backend
                if storage_ref.storage_type == StorageType.KV:
                    data = await ctx.deps.storage_client.load_kv(storage_ref.key)
                else:
                    data = await ctx.deps.storage_client.load_fs(storage_ref.key)
                
                loaded_data[dep] = data
                logger.info(f"Loaded dependency {dep} from {storage_ref}")
                
            except Exception as e:
                # Storage errors are retryable
                raise StorageError(f"Failed to load {dep}: {e}")
        
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
    """
    Save phase output to storage.
    Updates phase_outputs with storage reference.
    """
    
    async def perform_operation(self, ctx: GraphRunContext[WorkflowState, Any]) -> StorageRef:
        """Save phase output to storage."""
        phase_name = ctx.state.current_phase
        phase_def = ctx.state.workflow_def.phases.get(phase_name)
        if not phase_def:
            raise NonRetryableError(f"Phase {phase_name} not found")
        
        # Get output data from domain_data
        output_key = f'{phase_name}_output'
        output_data = ctx.state.domain_data.get(output_key)
        
        if output_data is None:
            # Also check for LLM response
            output_key = f'{phase_name}_llm_response'
            output_data = ctx.state.domain_data.get(output_key)
        
        if output_data is None:
            raise NonRetryableError(f"No output data for phase {phase_name}")
        
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
                storage_key = f"{storage_key}/v{version}"
        
        try:
            # Save to appropriate storage backend
            if phase_def.storage_type == StorageType.KV:
                await ctx.deps.storage_client.save_kv(storage_key, output_data)
            else:
                await ctx.deps.storage_client.save_fs(storage_key, output_data)
            
            logger.info(f"Saved {phase_name} output to {storage_key}")
            
            # Create storage reference
            return StorageRef(
                storage_type=phase_def.storage_type,
                key=storage_key,
                created_at=datetime.now(),
                version=ctx.state.refinement_count.get(phase_name, 0) if phase_def.allow_refinement else None
            )
            
        except Exception as e:
            # Storage errors are retryable
            raise StorageError(f"Failed to save {phase_name} output: {e}")
    
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
    """
    Generic storage load node.
    Can load from KV or FS based on configuration.
    """
    storage_key: str
    storage_type: StorageType = StorageType.KV
    required: bool = True
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Load data from storage."""
        try:
            if self.storage_type == StorageType.KV:
                data = await ctx.deps.storage_client.load_kv(self.storage_key)
            else:
                data = await ctx.deps.storage_client.load_fs(self.storage_key)
            
            if data is None and self.required:
                raise NonRetryableError(f"Required key not found: {self.storage_key}")
            
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
            raise StorageError(f"Failed to load {self.storage_key}: {e}")


@dataclass
class SaveStorageNode(BaseNode[WorkflowState, Any, StorageRef]):
    """
    Generic storage save node.
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
            raise NonRetryableError(f"No data in {self.data_field}")
        
        try:
            if self.storage_type == StorageType.KV:
                await ctx.deps.storage_client.save_kv(self.storage_key, data)
            else:
                await ctx.deps.storage_client.save_fs(self.storage_key, data)
            
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
            raise StorageError(f"Failed to save to {self.storage_key}: {e}")


@dataclass
class BatchLoadNode(BaseNode[WorkflowState, Any, Dict[str, Any]]):
    """
    Load multiple items in parallel.
    """
    storage_keys: List[str]
    storage_type: StorageType = StorageType.KV
    
    async def execute(self, ctx: GraphRunContext[WorkflowState, Any]) -> BaseNode:
        """Load multiple items from storage."""
        import asyncio
        
        async def load_item(key: str) -> tuple[str, Any]:
            try:
                if self.storage_type == StorageType.KV:
                    data = await ctx.deps.storage_client.load_kv(key)
                else:
                    data = await ctx.deps.storage_client.load_fs(key)
                return key, data
            except Exception as e:
                logger.error(f"Failed to load {key}: {e}")
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


# Register all storage nodes
register_node_class('dependency_check', DependencyCheckNode)
register_node_class('load_dependencies', LoadDependenciesNode)
register_node_class('save_output', SavePhaseOutputNode)
register_node_class('save_phase_output', SavePhaseOutputNode)
register_node_class('load_storage', LoadStorageNode)
register_node_class('save_storage', SaveStorageNode)
register_node_class('batch_load', BatchLoadNode)