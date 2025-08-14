"""GraphToolkit Centralized Initialization System.

This module provides centralized initialization of all agentoolkit components
needed by GraphToolkit. It ensures all create_*_agent() functions are called
exactly once and handles proper lifecycle management.

This solves the critical issue where GraphToolkit nodes call get_injector()
expecting agents to be available, but no centralized place ensures they are
created first.
"""

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Callable
from contextlib import asynccontextmanager

from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry

logger = logging.getLogger(__name__)

# GraphToolkit metrics to be initialized
_GRAPHTOOLKIT_METRICS = {
    # Node execution metrics
    'graphtoolkit.dependencycheck.executions.total': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.dependencycheck.executions.success': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.dependencycheck.executions.failure': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.dependencycheck.duration.seconds': {'type': 'timer', 'unit': 'seconds'},
    
    'graphtoolkit.loaddependencies.executions.total': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.loaddependencies.executions.success': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.loaddependencies.executions.failure': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.loaddependencies.duration.seconds': {'type': 'timer', 'unit': 'seconds'},
    
    'graphtoolkit.templaterender.executions.total': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.templaterender.executions.success': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.templaterender.executions.failure': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.templaterender.duration.seconds': {'type': 'timer', 'unit': 'seconds'},
    
    'graphtoolkit.llmcall.executions.total': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.llmcall.executions.success': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.llmcall.executions.failure': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.llmcall.duration.seconds': {'type': 'timer', 'unit': 'seconds'},
    
    'graphtoolkit.schemavalidation.executions.total': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.schemavalidation.executions.success': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.schemavalidation.executions.failure': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.schemavalidation.duration.seconds': {'type': 'timer', 'unit': 'seconds'},
    
    'graphtoolkit.savephaseoutput.executions.total': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.savephaseoutput.executions.success': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.savephaseoutput.executions.failure': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.savephaseoutput.duration.seconds': {'type': 'timer', 'unit': 'seconds'},
    
    'graphtoolkit.error.executions.total': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.error.executions.success': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.error.executions.failure': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.error.duration.seconds': {'type': 'timer', 'unit': 'seconds'},
    
    # Missing node metrics from warnings
    'graphtoolkit.stateupdate.executions.total': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.stateupdate.executions.success': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.stateupdate.executions.failure': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.stateupdate.duration.seconds': {'type': 'timer', 'unit': 'seconds'},
    
    'graphtoolkit.qualitygate.executions.total': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.qualitygate.executions.success': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.qualitygate.executions.failure': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.qualitygate.duration.seconds': {'type': 'timer', 'unit': 'seconds'},
    
    'graphtoolkit.refinement.executions.total': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.refinement.executions.success': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.refinement.executions.failure': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.refinement.duration.seconds': {'type': 'timer', 'unit': 'seconds'},
    
    'graphtoolkit.nextphase.executions.total': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.nextphase.executions.success': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.nextphase.executions.failure': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.nextphase.duration.seconds': {'type': 'timer', 'unit': 'seconds'},
    
    # Storage operation metrics
    'graphtoolkit.storage.operations.total': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.storage.operations.success': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.storage.operations.failure': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.storage.duration.seconds': {'type': 'timer', 'unit': 'seconds'},
    
    # Workflow-level metrics
    'graphtoolkit.workflow.executions.total': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.workflow.executions.success': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.workflow.executions.failure': {'type': 'counter', 'unit': 'count'},
    'graphtoolkit.workflow.duration.seconds': {'type': 'timer', 'unit': 'seconds'},
    'graphtoolkit.workflow.phases.completed': {'type': 'gauge', 'unit': 'count'},
}


@dataclass
class InitializationConfig:
    """Configuration for GraphToolkit initialization."""
    
    # Core storage components (always needed)
    enable_storage_kv: bool = True
    enable_storage_fs: bool = True
    enable_vector: bool = False
    
    # System components (templates required for GraphToolkit)
    enable_templates: bool = True
    enable_logging: bool = True
    enable_config: bool = False
    enable_scheduler: bool = False
    enable_queue: bool = False
    
    # Observability (metrics required for tracking)
    enable_metrics: bool = True
    
    # Security and network (optional)
    enable_crypto: bool = False
    enable_auth: bool = False
    enable_session: bool = False
    enable_http: bool = False
    
    # Management (required for agentool domain)
    enable_agentool_management: bool = True
    
    # Workflow agents (only for AgenTool domain)
    enable_workflow_agents: bool = False
    
    # Testing configuration
    test_mode: bool = False
    clear_on_init: bool = False


class GraphToolkitInitializer:
    """Centralized initializer for all agentoolkit components.
    
    This class ensures that all required agentoolkit components are properly
    initialized before any GraphToolkit node tries to use get_injector().
    
    Key features:
    - Thread-safe singleton pattern
    - Lazy initialization (components created only when first needed)
    - Configuration-driven component selection
    - Proper cleanup for testing
    - Comprehensive error handling and logging
    """
    
    _instance: Optional['GraphToolkitInitializer'] = None
    _lock = threading.Lock()
    
    def __init__(self):
        """Private constructor. Use get_instance() instead."""
        self.config: Optional[InitializationConfig] = None
        self.initialized_components: Set[str] = set()
        self.component_agents: Dict[str, Any] = {}
        self._initialization_lock = threading.Lock()
        self._is_initialized = False
        
    @classmethod
    def get_instance(cls) -> 'GraphToolkitInitializer':
        """Get the singleton instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.cleanup()
                cls._instance = None
    
    def initialize(self, config: Optional[InitializationConfig] = None) -> None:
        """Initialize all configured agentoolkit components.
        
        Args:
            config: Configuration specifying which components to initialize.
                   If None, uses default configuration.
        """
        with self._initialization_lock:
            if self._is_initialized:
                logger.debug("GraphToolkit already initialized")
                return
                
            self.config = config or InitializationConfig()
            logger.info("Starting GraphToolkit initialization")
            
            try:
                # Clear registries if in test mode or requested
                if self.config.test_mode or self.config.clear_on_init:
                    self._clear_registries()
                
                # Initialize components in dependency order
                logger.info("Initializing storage components...")
                self._initialize_storage_components()
                logger.info("Initializing system components...")
                self._initialize_system_components()  
                logger.info("Initializing observability components...")
                self._initialize_observability_components()
                logger.info("Initializing security components...")
                self._initialize_security_components()
                logger.info("Initializing network components...")
                self._initialize_network_components()
                logger.info("Initializing management components...")
                self._initialize_management_components()
                logger.info("Initializing workflow components...")
                self._initialize_workflow_components()
                
                self._is_initialized = True
                logger.info(f"GraphToolkit initialization complete. "
                          f"Initialized components: {sorted(self.initialized_components)}")
                
            except Exception as e:
                logger.error(f"GraphToolkit initialization failed: {e}")
                # Clean up partial initialization
                self.cleanup()
                raise
    
    def ensure_initialized(self, config: Optional[InitializationConfig] = None) -> None:
        """Ensure initialization has occurred, initializing if necessary."""
        if not self._is_initialized:
            self.initialize(config)
    
    def is_initialized(self) -> bool:
        """Check if initialization is complete."""
        return self._is_initialized
    
    def get_initialized_components(self) -> Set[str]:
        """Get the set of initialized component names."""
        return self.initialized_components.copy()
    
    def cleanup(self) -> None:
        """Clean up all initialized components (mainly for testing)."""
        logger.info("Cleaning up GraphToolkit components")
        
        # Clear storage state
        self._clear_storage_state()
        
        # Clear registries
        self._clear_registries()
        
        # Reset state
        self.initialized_components.clear()
        self.component_agents.clear()
        self._is_initialized = False
        
        logger.info("GraphToolkit cleanup complete")
    
    def _clear_registries(self) -> None:
        """Clear AgenTool registries."""
        try:
            AgenToolRegistry.clear()
            get_injector().clear()
            logger.debug("Cleared AgenTool registries")
        except Exception as e:
            logger.warning(f"Error clearing registries: {e}")
    
    def _clear_storage_state(self) -> None:
        """Clear storage-specific global state."""
        try:
            # Clear KV storage state
            from agentoolkit.storage.kv import _kv_storage, _kv_expiry
            _kv_storage.clear()
            _kv_expiry.clear()
            logger.debug("Cleared KV storage state")
        except ImportError:
            # KV storage not available
            pass
        except Exception as e:
            logger.warning(f"Error clearing storage state: {e}")
    
    def _initialize_storage_components(self) -> None:
        """Initialize storage components."""
        if self.config.enable_storage_kv:
            self._initialize_component('storage_kv', self._create_storage_kv_agent)
            
        if self.config.enable_storage_fs:
            self._initialize_component('storage_fs', self._create_storage_fs_agent)
            
        if self.config.enable_vector:
            self._initialize_component('vector', self._create_vector_agent)
    
    def _initialize_system_components(self) -> None:
        """Initialize system components."""
        if self.config.enable_templates:
            self._initialize_component('templates', self._create_templates_agent)
            
        if self.config.enable_logging:
            self._initialize_component('logging', self._create_logging_agent)
            
        if self.config.enable_config:
            self._initialize_component('config', self._create_config_agent)
            
        if self.config.enable_scheduler:
            self._initialize_component('scheduler', self._create_scheduler_agent)
            
        if self.config.enable_queue:
            self._initialize_component('queue', self._create_queue_agent)
    
    def _initialize_observability_components(self) -> None:
        """Initialize observability components."""
        if self.config.enable_metrics:
            self._initialize_component('metrics', self._create_metrics_agent)
            # Initialize GraphToolkit-specific metrics
            self._initialize_graphtoolkit_metrics()
    
    def _initialize_security_components(self) -> None:
        """Initialize security components."""
        if self.config.enable_crypto:
            self._initialize_component('crypto', self._create_crypto_agent)
            
        if self.config.enable_auth:
            self._initialize_component('auth', self._create_auth_agent)
            
        if self.config.enable_session:
            self._initialize_component('session', self._create_session_agent)
    
    def _initialize_network_components(self) -> None:
        """Initialize network components."""
        if self.config.enable_http:
            self._initialize_component('http', self._create_http_agent)
    
    def _initialize_management_components(self) -> None:
        """Initialize management components."""
        logger.info(f"Initializing management components, enable_agentool_management={self.config.enable_agentool_management}")
        if self.config.enable_agentool_management:
            logger.info("Initializing agentool_mgmt component...")
            self._initialize_component('agentool_mgmt', self._create_agentool_management_agent)
        else:
            logger.info("Skipping agentool_mgmt - not enabled in config")
    
    def _initialize_workflow_components(self) -> None:
        """Initialize workflow components."""
        if self.config.enable_workflow_agents:
            # These are used by the AgenTool domain workflow phases
            workflow_components = [
                ('workflow_analyzer', self._create_workflow_analyzer_agent),
                ('workflow_specifier', self._create_workflow_specifier_agent), 
                ('workflow_crafter', self._create_workflow_crafter_agent),
                ('workflow_evaluator', self._create_workflow_evaluator_agent),
            ]
            
            for name, creator in workflow_components:
                self._initialize_component(name, creator)
    
    def _initialize_component(self, name: str, creator_func: Callable) -> None:
        """Initialize a single component with error handling."""
        try:
            agent = creator_func()
            self.component_agents[name] = agent
            self.initialized_components.add(name)
            logger.debug(f"Initialized component: {name}")
        except ImportError as e:
            logger.warning(f"Component {name} not available: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize component {name}: {e}")
            # Re-raise for essential components
            if name in ['storage_kv', 'storage_fs', 'templates', 'metrics']:
                raise
    
    # Component creator methods
    # These import and call the create_*_agent functions
    
    def _create_storage_kv_agent(self):
        from agentoolkit.storage.kv import create_storage_kv_agent
        return create_storage_kv_agent()
    
    def _create_storage_fs_agent(self):
        from agentoolkit.storage.fs import create_storage_fs_agent
        return create_storage_fs_agent()
    
    def _create_vector_agent(self):
        from agentoolkit.storage.vector import create_vector_agent
        return create_vector_agent()
    
    def _create_templates_agent(self):
        from agentoolkit.system.templates import create_templates_agent
        # Use the templates directory at the project root where our smoke templates are stored
        import os
        from pathlib import Path
        
        # Go from src/graphtoolkit/core to the project root, then to templates
        templates_dir = Path(__file__).parent.parent.parent.parent / "templates"
        
        if not templates_dir.exists():
            # Fallback to current directory templates
            templates_dir = Path("templates")
            
        logger.info(f"Creating templates agent with directory: {templates_dir.absolute()}")
        logger.info(f"Templates directory exists: {templates_dir.exists()}")
        if templates_dir.exists():
            logger.info(f"Templates directory contents: {list(templates_dir.iterdir())}")
            
        return create_templates_agent(templates_dir=str(templates_dir.absolute()))
    
    def _create_logging_agent(self):
        from agentoolkit.system.logging import create_logging_agent
        return create_logging_agent()
    
    def _create_config_agent(self):
        from agentoolkit.system.config import create_config_agent
        return create_config_agent()
    
    def _create_scheduler_agent(self):
        from agentoolkit.system.scheduler import create_scheduler_agent
        return create_scheduler_agent()
    
    def _create_queue_agent(self):
        from agentoolkit.system.queue import create_queue_agent
        return create_queue_agent()
    
    def _create_metrics_agent(self):
        from agentoolkit.observability.metrics import create_metrics_agent
        return create_metrics_agent()
    
    def _create_crypto_agent(self):
        from agentoolkit.security.crypto import create_crypto_agent
        return create_crypto_agent()
    
    def _create_auth_agent(self):
        from agentoolkit.auth.auth import create_auth_agent
        return create_auth_agent()
    
    def _create_session_agent(self):
        from agentoolkit.auth.session import create_session_agent
        return create_session_agent()
    
    def _create_http_agent(self):
        from agentoolkit.network.http import create_http_agent
        return create_http_agent()
    
    def _create_agentool_management_agent(self):
        from agentoolkit.management.agentool import create_agentool_management_agent
        return create_agentool_management_agent()
    
    def _create_workflow_analyzer_agent(self):
        from agentoolkit.workflows.workflow_analyzer import create_workflow_analyzer_agent
        return create_workflow_analyzer_agent()
    
    def _create_workflow_specifier_agent(self):
        from agentoolkit.workflows.workflow_specifier import create_workflow_specifier_agent
        return create_workflow_specifier_agent()
    
    def _create_workflow_crafter_agent(self):
        from agentoolkit.workflows.workflow_crafter import create_workflow_crafter_agent
        return create_workflow_crafter_agent()
    
    def _create_workflow_evaluator_agent(self):
        from agentoolkit.workflows.workflow_evaluator import create_workflow_evaluator_agent
        return create_workflow_evaluator_agent()
    
    def _initialize_graphtoolkit_metrics(self) -> None:
        """Initialize all GraphToolkit metrics using the metrics agent."""
        try:
            injector = get_injector()
            
            # Create all metrics
            import asyncio
            
            for metric_name, config in _GRAPHTOOLKIT_METRICS.items():
                try:
                    # Check if we're already in an event loop
                    try:
                        loop = asyncio.get_running_loop()
                        # Already in a loop, create task instead
                        task = loop.create_task(
                            injector.run('metrics', {
                                'operation': 'create',
                                'name': metric_name,
                                'metric_type': config['type'],
                                'unit': config['unit'],
                                'description': f'GraphToolkit metric: {metric_name}'
                            })
                        )
                        # Don't wait for it to avoid blocking
                        logger.debug(f"Scheduled metric creation: {metric_name}")
                    except RuntimeError:
                        # No loop running, use run_until_complete
                        loop = asyncio.get_event_loop()
                        loop.run_until_complete(
                            injector.run('metrics', {
                                'operation': 'create',
                                'name': metric_name,
                                'metric_type': config['type'],
                                'unit': config['unit'],
                                'description': f'GraphToolkit metric: {metric_name}'
                            })
                        )
                        logger.debug(f"Created metric: {metric_name}")
                except Exception as e:
                    # Metric might already exist, that's ok
                    logger.debug(f"Metric {metric_name} initialization: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to initialize GraphToolkit metrics: {e}")


# Module-level convenience functions

def initialize_graphtoolkit(config: Optional[InitializationConfig] = None) -> None:
    """Initialize GraphToolkit with all required agentoolkit components.
    
    This is the main entry point for GraphToolkit initialization.
    Call this before using any GraphToolkit functionality.
    
    Args:
        config: Optional configuration specifying which components to initialize.
               If None, uses sensible defaults.
    """
    initializer = GraphToolkitInitializer.get_instance()
    initializer.initialize(config)


def ensure_graphtoolkit_initialized(config: Optional[InitializationConfig] = None) -> None:
    """Ensure GraphToolkit is initialized, initializing if necessary.
    
    This is safe to call multiple times - initialization only happens once.
    
    Args:
        config: Optional configuration for initialization.
    """
    initializer = GraphToolkitInitializer.get_instance()
    initializer.ensure_initialized(config)


def is_graphtoolkit_initialized() -> bool:
    """Check if GraphToolkit has been initialized."""
    try:
        initializer = GraphToolkitInitializer.get_instance()
        return initializer.is_initialized()
    except:
        return False


def cleanup_graphtoolkit() -> None:
    """Clean up GraphToolkit components (mainly for testing)."""
    try:
        initializer = GraphToolkitInitializer.get_instance()
        initializer.cleanup()
    except:
        pass  # Already cleaned up


def reset_graphtoolkit() -> None:
    """Reset GraphToolkit completely (for testing)."""
    GraphToolkitInitializer.reset_instance()


@asynccontextmanager
async def graphtoolkit_context(config: Optional[InitializationConfig] = None):
    """Async context manager for GraphToolkit initialization and cleanup.
    
    Usage:
        async with graphtoolkit_context() as gt:
            # Use GraphToolkit functionality
            pass
        # Cleanup happens automatically
    """
    try:
        initialize_graphtoolkit(config)
        yield GraphToolkitInitializer.get_instance()
    finally:
        cleanup_graphtoolkit()


# Default configurations for common scenarios

def default_config() -> InitializationConfig:
    """Default configuration with essential components."""
    return InitializationConfig(
        # Storage components
        enable_storage_kv=True,
        enable_storage_fs=True,
        enable_vector=True,
        
        # System components
        enable_templates=True,
        enable_logging=True,
        enable_config=True,
        enable_scheduler=True,
        enable_queue=True,
        
        # Observability
        enable_metrics=True,
        
        # Security and network
        enable_crypto=True,
        enable_auth=True,
        enable_session=True,
        enable_http=True,
        
        # Management
        enable_agentool_management=True,
        
        # Workflow agents
        enable_workflow_agents=True,
    )


def test_config() -> InitializationConfig:
    """Configuration optimized for testing."""
    return InitializationConfig(
        enable_storage_kv=True,
        enable_storage_fs=True,
        enable_templates=True,
        enable_logging=False,  # Reduce noise in tests
        enable_metrics=False,  # Reduce noise in tests
        test_mode=True,
        clear_on_init=True,
    )


def full_config() -> InitializationConfig:
    """Configuration with all components enabled."""
    return InitializationConfig(
        enable_storage_kv=True,
        enable_storage_fs=True,
        enable_vector=True,
        enable_templates=True,
        enable_logging=True,
        enable_config=True,
        enable_scheduler=True,
        enable_queue=True,
        enable_metrics=True,
        enable_crypto=True,
        enable_auth=True,
        enable_session=True,
        enable_http=True,
        enable_agentool_management=True,
        enable_workflow_agents=True,
    )


