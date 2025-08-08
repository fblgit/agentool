"""
GraphToolkit State Persistence.

State persistence and recovery mechanisms.
"""

from typing import Optional, Any, Dict
from dataclasses import asdict
from pathlib import Path
import json
import logging
from datetime import datetime

from ..core.types import (
    WorkflowState,
    StorageRef,
    StorageType
)


logger = logging.getLogger(__name__)


class StateSerializer:
    """Serialize and deserialize workflow state."""
    
    @staticmethod
    def serialize(state: WorkflowState) -> str:
        """
        Serialize workflow state to JSON string.
        
        Args:
            state: Workflow state to serialize
            
        Returns:
            JSON string representation
        """
        # Convert to dict
        state_dict = asdict(state)
        
        # Custom serialization for complex types
        state_dict = StateSerializer._prepare_for_json(state_dict)
        
        return json.dumps(state_dict, indent=2, default=str)
    
    @staticmethod
    def deserialize(json_str: str) -> WorkflowState:
        """
        Deserialize workflow state from JSON string.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            Workflow state object
        """
        state_dict = json.loads(json_str)
        
        # Reconstruct complex types
        state_dict = StateSerializer._restore_from_json(state_dict)
        
        # Reconstruct WorkflowState
        return WorkflowState(**state_dict)
    
    @staticmethod
    def _prepare_for_json(obj: Any) -> Any:
        """Prepare object for JSON serialization."""
        if isinstance(obj, dict):
            return {k: StateSerializer._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [StateSerializer._prepare_for_json(item) for item in obj]
        elif isinstance(obj, set):
            return {'__type__': 'set', 'values': list(obj)}
        elif isinstance(obj, datetime):
            return {'__type__': 'datetime', 'value': obj.isoformat()}
        elif isinstance(obj, StorageRef):
            return {
                '__type__': 'StorageRef',
                'storage_type': obj.storage_type.value if hasattr(obj.storage_type, 'value') else obj.storage_type,
                'key': obj.key,
                'created_at': obj.created_at.isoformat() if obj.created_at else None,
                'version': obj.version,
                'size_bytes': obj.size_bytes
            }
        elif hasattr(obj, '__dict__'):
            # Handle Pydantic models and dataclasses
            return {
                '__type__': obj.__class__.__name__,
                '__module__': obj.__class__.__module__,
                'data': asdict(obj) if hasattr(obj, '__dataclass_fields__') else obj.dict() if hasattr(obj, 'dict') else obj.__dict__
            }
        else:
            return obj
    
    @staticmethod
    def _restore_from_json(obj: Any) -> Any:
        """Restore object from JSON representation."""
        if isinstance(obj, dict):
            if '__type__' in obj:
                type_name = obj['__type__']
                
                if type_name == 'set':
                    return set(obj['values'])
                elif type_name == 'datetime':
                    return datetime.fromisoformat(obj['value'])
                elif type_name == 'StorageRef':
                    return StorageRef(
                        storage_type=StorageType(obj['storage_type']) if obj['storage_type'] else StorageType.KV,
                        key=obj['key'],
                        created_at=datetime.fromisoformat(obj['created_at']) if obj['created_at'] else None,
                        version=obj.get('version'),
                        size_bytes=obj.get('size_bytes')
                    )
                else:
                    # Try to reconstruct other types
                    return obj.get('data', obj)
            else:
                return {k: StateSerializer._restore_from_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [StateSerializer._restore_from_json(item) for item in obj]
        else:
            return obj


class StatePersistence:
    """Base class for state persistence backends."""
    
    async def save(self, state: WorkflowState) -> StorageRef:
        """
        Save workflow state.
        
        Args:
            state: Workflow state to save
            
        Returns:
            Storage reference for the saved state
        """
        raise NotImplementedError
    
    async def load(self, ref: StorageRef) -> WorkflowState:
        """
        Load workflow state.
        
        Args:
            ref: Storage reference
            
        Returns:
            Loaded workflow state
        """
        raise NotImplementedError
    
    async def list_checkpoints(self, workflow_id: str) -> list[StorageRef]:
        """
        List all checkpoints for a workflow.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            List of storage references
        """
        raise NotImplementedError


class FileStatePersistence(StatePersistence):
    """File-based state persistence."""
    
    def __init__(self, base_dir: str = "workflow_states"):
        """
        Initialize file-based persistence.
        
        Args:
            base_dir: Base directory for state files
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    async def save(self, state: WorkflowState) -> StorageRef:
        """Save state to file."""
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{state.workflow_id}_{state.current_phase}_{timestamp}.json"
        filepath = self.base_dir / state.workflow_id / filename
        
        # Create directory
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Serialize and save
        json_str = StateSerializer.serialize(state)
        filepath.write_text(json_str)
        
        logger.info(f"Saved state to {filepath}")
        
        return StorageRef(
            storage_type=StorageType.FS,
            key=str(filepath),
            created_at=datetime.now(),
            size_bytes=len(json_str)
        )
    
    async def load(self, ref: StorageRef) -> WorkflowState:
        """Load state from file."""
        filepath = Path(ref.key)
        
        if not filepath.exists():
            raise FileNotFoundError(f"State file not found: {filepath}")
        
        json_str = filepath.read_text()
        state = StateSerializer.deserialize(json_str)
        
        logger.info(f"Loaded state from {filepath}")
        return state
    
    async def list_checkpoints(self, workflow_id: str) -> list[StorageRef]:
        """List all checkpoint files for a workflow."""
        workflow_dir = self.base_dir / workflow_id
        
        if not workflow_dir.exists():
            return []
        
        checkpoints = []
        for filepath in sorted(workflow_dir.glob("*.json")):
            stat = filepath.stat()
            checkpoints.append(
                StorageRef(
                    storage_type=StorageType.FS,
                    key=str(filepath),
                    created_at=datetime.fromtimestamp(stat.st_mtime),
                    size_bytes=stat.st_size
                )
            )
        
        return checkpoints


class KVStatePersistence(StatePersistence):
    """Key-value store based state persistence."""
    
    def __init__(self, storage_client: Any, prefix: str = "workflow_states"):
        """
        Initialize KV-based persistence.
        
        Args:
            storage_client: Storage client with KV operations
            prefix: Key prefix for state storage
        """
        self.storage_client = storage_client
        self.prefix = prefix
    
    async def save(self, state: WorkflowState) -> StorageRef:
        """Save state to KV store."""
        # Generate key
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        key = f"{self.prefix}/{state.workflow_id}/{state.current_phase}/{timestamp}"
        
        # Serialize state
        json_str = StateSerializer.serialize(state)
        
        # Save to KV store
        await self.storage_client.save_kv(key, json_str)
        
        logger.info(f"Saved state to KV: {key}")
        
        return StorageRef(
            storage_type=StorageType.KV,
            key=key,
            created_at=datetime.now(),
            size_bytes=len(json_str)
        )
    
    async def load(self, ref: StorageRef) -> WorkflowState:
        """Load state from KV store."""
        # Load from KV store
        json_str = await self.storage_client.load_kv(ref.key)
        
        if json_str is None:
            raise KeyError(f"State not found in KV: {ref.key}")
        
        state = StateSerializer.deserialize(json_str)
        
        logger.info(f"Loaded state from KV: {ref.key}")
        return state
    
    async def list_checkpoints(self, workflow_id: str) -> list[StorageRef]:
        """List all checkpoints for a workflow in KV store."""
        # This would need to be implemented based on the specific KV store
        # For now, return empty list
        logger.warning("KV checkpoint listing not implemented")
        return []


def create_checkpoint(
    state: WorkflowState,
    persistence: StatePersistence
) -> StorageRef:
    """
    Create a checkpoint of the current state.
    
    Args:
        state: Current workflow state
        persistence: Persistence backend
        
    Returns:
        Storage reference for the checkpoint
    """
    import asyncio
    
    # Run async save in sync context if needed
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        # Already in async context
        task = persistence.save(state)
        return task
    else:
        # Sync context
        return loop.run_until_complete(persistence.save(state))


def restore_checkpoint(
    ref: StorageRef,
    persistence: StatePersistence
) -> WorkflowState:
    """
    Restore state from a checkpoint.
    
    Args:
        ref: Storage reference
        persistence: Persistence backend
        
    Returns:
        Restored workflow state
    """
    import asyncio
    
    # Run async load in sync context if needed
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        # Already in async context
        task = persistence.load(ref)
        return task
    else:
        # Sync context
        return loop.run_until_complete(persistence.load(ref))