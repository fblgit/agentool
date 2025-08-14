"""
GraphToolkit State Persistence.

Integration with pydantic_graph's persistence patterns for workflow state saving and resumption.
"""

from typing import Optional, Any, Dict, Type
from pathlib import Path
from dataclasses import dataclass, replace
import json
import logging
from datetime import datetime

from pydantic_graph import FileStatePersistence, Graph, BaseNode
from pydantic_graph.persistence import StatePersistence, Snapshot

from ..core.types import WorkflowState, WorkflowDefinition
from ..nodes.generic import GenericPhaseNode

logger = logging.getLogger(__name__)


@dataclass
class WorkflowPersistence:
    """
    Manages workflow state persistence and recovery.
    
    Features:
    - Save workflow state at any point
    - Resume from saved state
    - Checkpoint management
    - Crash recovery
    """
    
    persistence_dir: Path
    workflow_id: str
    
    def __post_init__(self):
        """Initialize persistence directory."""
        self.persistence_dir = Path(self.persistence_dir)
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        
        # Create workflow-specific directory
        self.workflow_dir = self.persistence_dir / self.workflow_id
        self.workflow_dir.mkdir(exist_ok=True)
        
        # File paths
        self.state_file = self.workflow_dir / "state.json"
        self.checkpoint_dir = self.workflow_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def get_file_persistence(self) -> FileStatePersistence:
        """Get pydantic_graph FileStatePersistence instance."""
        return FileStatePersistence(self.state_file)
    
    async def save_state(self, state: WorkflowState, node: Optional[BaseNode] = None) -> bool:
        """
        Save current workflow state.
        
        Args:
            state: Current workflow state
            node: Current node (optional)
            
        Returns:
            Success status
        """
        try:
            # Create state snapshot
            snapshot = {
                'workflow_id': state.workflow_id,
                'domain': state.domain,
                'current_phase': state.current_phase,
                'current_node': state.current_node,
                'completed_phases': list(state.completed_phases),
                'phase_outputs': {k: v.dict() if hasattr(v, 'dict') else str(v) 
                                 for k, v in state.phase_outputs.items()},
                'domain_data': state.domain_data,
                'quality_scores': state.quality_scores,
                'validation_results': {k: v.dict() if hasattr(v, 'dict') else v 
                                      for k, v in state.validation_results.items()},
                'refinement_count': state.refinement_count,
                'iter_items': state.iter_items,
                'iter_results': state.iter_results,
                'iter_index': state.iter_index,
                'retry_counts': state.retry_counts,
                'total_token_usage': {k: v.dict() if hasattr(v, 'dict') else v 
                                     for k, v in state.total_token_usage.items()},
                'created_at': state.created_at.isoformat(),
                'updated_at': datetime.now().isoformat(),
                'node_type': type(node).__name__ if node else None
            }
            
            # Save to file
            with open(self.state_file, 'w') as f:
                json.dump(snapshot, f, indent=2, default=str)
            
            logger.info(f"Saved workflow state: {self.workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            from ..exceptions import StorageError
            raise StorageError(f"Failed to save workflow state: {e}") from e
    
    async def load_state(self) -> Optional[WorkflowState]:
        """
        Load saved workflow state.
        
        Returns:
            Restored workflow state or None
        """
        if not self.state_file.exists():
            logger.info(f"No saved state found for workflow: {self.workflow_id}")
            return None
        
        try:
            with open(self.state_file, 'r') as f:
                snapshot = json.load(f)
            
            # Reconstruct WorkflowState
            # Note: WorkflowDefinition needs to be reconstructed separately
            state = WorkflowState(
                workflow_id=snapshot['workflow_id'],
                domain=snapshot['domain'],
                workflow_def=WorkflowDefinition(
                    domain=snapshot['domain'],
                    phases={},  # Would need to be loaded from definition
                    phase_sequence=[],
                    node_configs={}
                ),
                current_phase=snapshot['current_phase'],
                current_node=snapshot['current_node'],
                completed_phases=set(snapshot['completed_phases']),
                phase_outputs=snapshot.get('phase_outputs', {}),
                domain_data=snapshot.get('domain_data', {}),
                quality_scores=snapshot.get('quality_scores', {}),
                validation_results=snapshot.get('validation_results', {}),
                refinement_count=snapshot.get('refinement_count', {}),
                refinement_history=snapshot.get('refinement_history', {}),
                iter_items=snapshot.get('iter_items', []),
                iter_results=snapshot.get('iter_results', []),
                iter_index=snapshot.get('iter_index', 0),
                retry_counts=snapshot.get('retry_counts', {}),
                total_token_usage=snapshot.get('total_token_usage', {}),
                created_at=datetime.fromisoformat(snapshot['created_at']),
                updated_at=datetime.fromisoformat(snapshot['updated_at'])
            )
            
            logger.info(f"Loaded workflow state: {self.workflow_id}")
            logger.info(f"  Current phase: {state.current_phase}")
            logger.info(f"  Current node: {state.current_node}")
            logger.info(f"  Completed phases: {state.completed_phases}")
            
            return state
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None
    
    async def create_checkpoint(self, state: WorkflowState, checkpoint_name: Optional[str] = None) -> str:
        """
        Create a checkpoint of current state.
        
        Args:
            state: Current workflow state
            checkpoint_name: Optional checkpoint name
            
        Returns:
            Checkpoint ID
        """
        if not checkpoint_name:
            checkpoint_name = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}.json"
        
        try:
            # Save state to checkpoint
            snapshot = {
                'checkpoint_name': checkpoint_name,
                'created_at': datetime.now().isoformat(),
                'workflow_id': state.workflow_id,
                'current_phase': state.current_phase,
                'current_node': state.current_node,
                'completed_phases': list(state.completed_phases),
                'domain_data': state.domain_data,
                'quality_scores': state.quality_scores,
                'iter_index': state.iter_index
            }
            
            with open(checkpoint_file, 'w') as f:
                json.dump(snapshot, f, indent=2, default=str)
            
            logger.info(f"Created checkpoint: {checkpoint_name}")
            return checkpoint_name
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            raise
    
    async def restore_checkpoint(self, checkpoint_name: str, workflow_def: WorkflowDefinition) -> Optional[WorkflowState]:
        """
        Restore state from checkpoint.
        
        Args:
            checkpoint_name: Checkpoint to restore
            workflow_def: Workflow definition
            
        Returns:
            Restored state or None
        """
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}.json"
        
        if not checkpoint_file.exists():
            logger.error(f"Checkpoint not found: {checkpoint_name}")
            return None
        
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            
            # Restore state from checkpoint
            state = WorkflowState(
                workflow_id=checkpoint['workflow_id'],
                domain=workflow_def.domain,
                workflow_def=workflow_def,
                current_phase=checkpoint['current_phase'],
                current_node=checkpoint['current_node'],
                completed_phases=set(checkpoint['completed_phases']),
                domain_data=checkpoint.get('domain_data', {}),
                quality_scores=checkpoint.get('quality_scores', {}),
                iter_index=checkpoint.get('iter_index', 0),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            logger.info(f"Restored from checkpoint: {checkpoint_name}")
            return state
            
        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}")
            return None
    
    def list_checkpoints(self) -> list[Dict[str, Any]]:
        """List all available checkpoints."""
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                    checkpoints.append({
                        'name': checkpoint_file.stem,
                        'created_at': data.get('created_at'),
                        'current_phase': data.get('current_phase'),
                        'completed_phases': data.get('completed_phases', [])
                    })
            except:
                pass
        
        return sorted(checkpoints, key=lambda x: x['created_at'], reverse=True)
    
    async def cleanup_old_checkpoints(self, keep_latest: int = 5):
        """Remove old checkpoints, keeping only the latest N."""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) > keep_latest:
            for checkpoint in checkpoints[keep_latest:]:
                checkpoint_file = self.checkpoint_dir / f"{checkpoint['name']}.json"
                checkpoint_file.unlink(missing_ok=True)
                logger.info(f"Removed old checkpoint: {checkpoint['name']}")


class PersistentWorkflowExecutor:
    """
    Executor that supports workflow persistence and recovery.
    """
    
    def __init__(self, graph: Graph, persistence_dir: Path = Path("workflows")):
        """
        Initialize persistent executor.
        
        Args:
            graph: Workflow graph
            persistence_dir: Directory for persistence files
        """
        self.graph = graph
        self.persistence_dir = Path(persistence_dir)
        self.persistence_dir.mkdir(exist_ok=True)
    
    async def run_with_persistence(
        self,
        workflow_def: WorkflowDefinition,
        workflow_id: str,
        initial_data: Optional[Dict[str, Any]] = None,
        resume: bool = True
    ) -> WorkflowState:
        """
        Run workflow with persistence support.
        
        Args:
            workflow_def: Workflow definition
            workflow_id: Unique workflow ID
            initial_data: Initial domain data
            resume: Whether to resume from saved state
            
        Returns:
            Final workflow state
        """
        persistence = WorkflowPersistence(self.persistence_dir, workflow_id)
        
        # Try to load existing state if resuming
        state = None
        if resume:
            state = await persistence.load_state()
            if state:
                logger.info(f"Resuming workflow from phase: {state.current_phase}")
                # Update workflow definition
                state = replace(state, workflow_def=workflow_def)
        
        # Create new state if not resuming or no saved state
        if not state:
            state = WorkflowState(
                workflow_id=workflow_id,
                domain=workflow_def.domain,
                workflow_def=workflow_def,
                current_phase=workflow_def.phase_sequence[0] if workflow_def.phase_sequence else "",
                current_node="",
                domain_data=initial_data or {}
            )
            logger.info(f"Starting new workflow: {workflow_id}")
        
        # Get pydantic_graph persistence
        file_persistence = persistence.get_file_persistence()
        
        # Run workflow with persistence
        try:
            # Start from appropriate node
            if state.current_node:
                from ..core.factory import create_node_instance
                start_node = create_node_instance(state.current_node)
            else:
                start_node = GenericPhaseNode()
            
            # Execute with persistence
            result = await self.graph.run(
                start_node,
                state=state,
                persistence=file_persistence
            )
            
            # Save final state
            if hasattr(result, 'data'):
                await persistence.save_state(result.data)
                return result.data
            else:
                await persistence.save_state(state)
                return state
                
        except KeyboardInterrupt:
            # Save state on interruption
            logger.info("Workflow interrupted, saving state...")
            await persistence.save_state(state)
            raise
        
        except Exception as e:
            # Save state on error
            logger.error(f"Workflow error: {e}, saving state...")
            await persistence.save_state(state)
            raise
    
    async def run_with_checkpoints(
        self,
        workflow_def: WorkflowDefinition,
        workflow_id: str,
        checkpoint_phases: list[str],
        initial_data: Optional[Dict[str, Any]] = None
    ) -> WorkflowState:
        """
        Run workflow with automatic checkpointing at specified phases.
        
        Args:
            workflow_def: Workflow definition
            workflow_id: Unique workflow ID
            checkpoint_phases: Phases where checkpoints should be created
            initial_data: Initial domain data
            
        Returns:
            Final workflow state
        """
        persistence = WorkflowPersistence(self.persistence_dir, workflow_id)
        
        state = WorkflowState(
            workflow_id=workflow_id,
            domain=workflow_def.domain,
            workflow_def=workflow_def,
            current_phase=workflow_def.phase_sequence[0] if workflow_def.phase_sequence else "",
            current_node="",
            domain_data=initial_data or {}
        )
        
        # Use Graph.iter() for phase-by-phase execution
        file_persistence = persistence.get_file_persistence()
        
        async with self.graph.iter(
            GenericPhaseNode(),
            state=state,
            persistence=file_persistence
        ) as run:
            async for node in run:
                # Check if we should checkpoint
                if state.current_phase in checkpoint_phases:
                    checkpoint_name = f"{state.current_phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    await persistence.create_checkpoint(state, checkpoint_name)
                    logger.info(f"Created checkpoint at phase: {state.current_phase}")
                
                # Update state from node execution
                if hasattr(node, 'state'):
                    state = node.state
        
        return state