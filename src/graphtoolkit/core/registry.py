"""GraphToolkit Phase Registry.

Central registry for phase definitions across all domains.
Enables dynamic workflow construction from registered phases.
"""

import logging
from typing import Dict, List, Optional, Set

from .types import ConditionConfig, NodeConfig, PhaseDefinition, RetryBackoff, WorkflowDefinition

logger = logging.getLogger(__name__)


class PhaseRegistry:
    """Singleton registry for all phase definitions."""
    
    _instance: Optional['PhaseRegistry'] = None
    _phases: Dict[str, PhaseDefinition] = {}
    _domains: Set[str] = set()
    
    def __new__(cls) -> 'PhaseRegistry':
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._phases = {}
            cls._instance._domains = set()
        return cls._instance
    
    def register_phase(self, phase_key: str, phase_def: PhaseDefinition) -> None:
        """Register a phase definition.
        
        Args:
            phase_key: Unique key in format "domain.phase_name"
            phase_def: Phase definition to register
        """
        if '.' not in phase_key:
            raise ValueError(f"Phase key must be in format 'domain.phase_name', got: {phase_key}")
        
        domain = phase_key.split('.')[0]
        self._domains.add(domain)
        self._phases[phase_key] = phase_def
        logger.info(f'Registered phase: {phase_key}')
    
    def register_domain(self, domain: str, phases: Dict[str, PhaseDefinition]) -> None:
        """Register all phases for a domain.
        
        Args:
            domain: Domain name
            phases: Dictionary of phase_name -> PhaseDefinition
        """
        for phase_name, phase_def in phases.items():
            phase_key = f'{domain}.{phase_name}'
            self.register_phase(phase_key, phase_def)
        logger.info(f'Registered {len(phases)} phases for domain: {domain}')
    
    def get_phase(self, phase_key: str) -> Optional[PhaseDefinition]:
        """Get a phase definition by key.
        
        Args:
            phase_key: Phase key in format "domain.phase_name"
            
        Returns:
            Phase definition or None if not found
        """
        return self._phases.get(phase_key)
    
    def get_domain_phases(self, domain: str) -> Dict[str, PhaseDefinition]:
        """Get all phases for a domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Dictionary of phase definitions for the domain
        """
        result = {}
        prefix = f'{domain}.'
        for key, phase_def in self._phases.items():
            if key.startswith(prefix):
                phase_name = key[len(prefix):]
                result[phase_name] = phase_def
        return result
    
    def list_domains(self) -> List[str]:
        """List all registered domains."""
        return sorted(self._domains)
    
    def list_phases(self, domain: Optional[str] = None) -> List[str]:
        """List all registered phase keys.
        
        Args:
            domain: Optional domain filter
            
        Returns:
            List of phase keys
        """
        if domain:
            prefix = f'{domain}.'
            return sorted([k for k in self._phases.keys() if k.startswith(prefix)])
        return sorted(self._phases.keys())
    
    def validate_phase_sequence(self, domain: str, phases: List[str]) -> bool:
        """Validate that a phase sequence is valid.
        
        Args:
            domain: Domain name
            phases: List of phase names
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        for phase_name in phases:
            phase_key = f'{domain}.{phase_name}'
            phase_def = self.get_phase(phase_key)
            
            if not phase_def:
                raise ValueError(f'Unknown phase: {phase_key}')
            
            # Check dependencies are in sequence before this phase
            for dep in phase_def.dependencies:
                if dep not in phases:
                    raise ValueError(f'Phase {phase_name} depends on {dep} which is not in sequence')
                if phases.index(dep) >= phases.index(phase_name):
                    raise ValueError(f'Phase {phase_name} depends on {dep} which comes after it')
        
        return True
    
    def create_workflow_definition(
        self,
        domain: str,
        phases: List[str],
        node_configs: Optional[Dict[str, NodeConfig]] = None,
        conditions: Optional[Dict[str, ConditionConfig]] = None,
        enable_refinement: bool = True,
        enable_parallel: bool = False
    ) -> WorkflowDefinition:
        """Create a workflow definition from registered phases.
        
        Args:
            domain: Domain name
            phases: List of phase names to include
            node_configs: Optional node configurations
            conditions: Optional condition configurations
            enable_refinement: Enable refinement loops
            enable_parallel: Enable parallel execution
            
        Returns:
            Complete workflow definition
        """
        # Validate phase sequence
        self.validate_phase_sequence(domain, phases)
        
        # Collect phase definitions
        phase_defs = {}
        for phase_name in phases:
            phase_key = f'{domain}.{phase_name}'
            phase_def = self.get_phase(phase_key)
            if phase_def:
                phase_defs[phase_name] = phase_def
        
        # Generate default node configs if not provided
        if node_configs is None:
            node_configs = self._generate_default_node_configs(phase_defs)
        
        # Create workflow definition
        return WorkflowDefinition(
            domain=domain,
            phases=phase_defs,
            phase_sequence=phases,
            node_configs=node_configs,
            conditions=conditions or {},
            enable_refinement=enable_refinement,
            enable_parallel=enable_parallel
        )
    
    def _generate_default_node_configs(self, phase_defs: Dict[str, PhaseDefinition]) -> Dict[str, NodeConfig]:
        """Generate default node configurations based on node IDs."""
        configs = {}
        
        # Collect all unique node IDs
        node_ids = set()
        for phase_def in phase_defs.values():
            node_ids.update(phase_def.atomic_nodes)
        
        # Generate config for each node
        for node_id in node_ids:
            if 'llm' in node_id.lower():
                # LLM nodes are expensive and retryable
                configs[node_id] = NodeConfig(
                    node_type='llm_call',
                    retryable=True,
                    max_retries=3,
                    retry_backoff=RetryBackoff.EXPONENTIAL,
                    retry_delay=2.0,
                    timeout=60.0
                )
            elif 'load' in node_id.lower() or 'save' in node_id.lower():
                # Storage nodes - usually not retryable for local storage
                configs[node_id] = NodeConfig(
                    node_type='storage',
                    retryable=False,  # Local storage rarely fails
                    max_retries=0,
                    timeout=10.0
                )
            elif 'template' in node_id.lower():
                # Template nodes are deterministic, cache them
                configs[node_id] = NodeConfig(
                    node_type='template',
                    retryable=False,
                    cacheable=True,
                    cache_ttl=3600
                )
            elif 'validation' in node_id.lower() or 'quality' in node_id.lower():
                # Validation nodes trigger refinement, not retry
                configs[node_id] = NodeConfig(
                    node_type='validation',
                    retryable=False,
                    max_retries=0
                )
            elif 'iterate' in node_id.lower() or 'process' in node_id.lower():
                # Iteration nodes
                configs[node_id] = NodeConfig(
                    node_type='iteration',
                    retryable=False,
                    iter_enabled=True
                )
            else:
                # Default config
                configs[node_id] = NodeConfig(
                    node_type='default',
                    retryable=False,
                    max_retries=0
                )
        
        return configs
    
    def clear(self) -> None:
        """Clear all registered phases (mainly for testing)."""
        self._phases.clear()
        self._domains.clear()
        logger.info('Cleared phase registry')


# Global registry instance
PHASE_REGISTRY = PhaseRegistry()


def get_registry() -> PhaseRegistry:
    """Get the global phase registry instance."""
    return PHASE_REGISTRY

def register_phase(phase_key: str, phase_def: PhaseDefinition) -> None:
    """Register a phase definition in the global registry."""
    PHASE_REGISTRY.register_phase(phase_key, phase_def)

def get_phase(phase_key: str) -> Optional[PhaseDefinition]:
    """Get a phase definition from the global registry."""
    return PHASE_REGISTRY.get_phase(phase_key)