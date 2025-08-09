"""GraphToolkit Dependency Injection System.

Provides services and configuration for workflow execution.
Following pydantic_graph patterns - configuration lives in deps.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Import pydantic_graph GraphDeps base class
try:
    from pydantic_graph import GraphDeps
except ImportError:
    # Fallback for development
    class GraphDeps:
        pass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import PhaseDefinition

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """LLM model configuration."""
    provider: str = 'openai'
    model: str = 'gpt-4'
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3


@dataclass
class StorageConfig:
    """Storage configuration."""
    kv_backend: str = 'memory'  # memory, redis, dynamodb
    fs_backend: str = 'local'   # local, s3, gcs
    base_path: str = '/tmp/graphtoolkit'
    connection_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemplateEngine:
    """Template engine configuration."""
    template_dir: str = 'templates'
    cache_templates: bool = True
    auto_reload: bool = False
    strict_undefined: bool = False


@dataclass
class DomainValidator:
    """Domain-specific validator."""
    domain: str
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self, data: Any) -> bool:
        """Validate domain data."""
        # Implementation would be domain-specific
        return True


@dataclass(frozen=True)
class WorkflowDeps(GraphDeps):
    """Services and configuration for workflow execution.
    
    Per workflow-graph-system.md design:
    - WorkflowDefinition is in WorkflowState (enables state-driven conditions)
    - Dependencies contain only services and utilities
    - Configuration access via ctx.state.workflow_def
    """
    
    # Core services
    models: ModelConfig = field(default_factory=ModelConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    template_engine: TemplateEngine = field(default_factory=TemplateEngine)
    
    # Phase registry (populated from registry)
    phase_registry: Dict[str, 'PhaseDefinition'] = field(default_factory=dict)
    
    # Executors for parallel processing
    process_executor: Optional[ProcessPoolExecutor] = None
    thread_executor: Optional[ThreadPoolExecutor] = None
    
    # Domain validators
    domain_validators: Dict[str, DomainValidator] = field(default_factory=dict)
    
    # Additional services
    metrics_enabled: bool = False
    logging_level: str = 'INFO'
    cache_enabled: bool = True
    
    @classmethod
    def create_default(cls) -> 'WorkflowDeps':
        """Create default dependencies."""
        return cls(
            models=ModelConfig(),
            storage=StorageConfig(),
            template_engine=TemplateEngine(),
            phase_registry={},
            process_executor=ProcessPoolExecutor(max_workers=4),
            thread_executor=ThreadPoolExecutor(max_workers=10),
            domain_validators={},
            metrics_enabled=False,
            logging_level='INFO',
            cache_enabled=True
        )
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'WorkflowDeps':
        """Create dependencies from configuration dict."""
        return cls(
            models=ModelConfig(**config.get('models', {})),
            storage=StorageConfig(**config.get('storage', {})),
            template_engine=TemplateEngine(**config.get('templates', {})),
            phase_registry=config.get('phase_registry', {}),
            process_executor=ProcessPoolExecutor(
                max_workers=config.get('process_workers', 4)
            ),
            thread_executor=ThreadPoolExecutor(
                max_workers=config.get('thread_workers', 10)
            ),
            domain_validators=config.get('domain_validators', {}),
            metrics_enabled=config.get('metrics_enabled', False),
            logging_level=config.get('logging_level', 'INFO'),
            cache_enabled=config.get('cache_enabled', True)
        )
    
    def validate_domain(self, domain: str, data: Any) -> bool:
        """Validate data for a specific domain."""
        if domain in self.domain_validators:
            return self.domain_validators[domain].validate(data)
        return True
    
    def get_storage_client(self):
        """Get storage client for operations."""
        from agentool.core.injector import get_injector
        return get_injector()
    
    def get_llm_client(self):
        """Get LLM client for operations."""
        # This would return the appropriate LLM client based on models config
        return None
    
    def cleanup(self):
        """Clean up resources."""
        if self.process_executor:
            self.process_executor.shutdown(wait=True)
        if self.thread_executor:
            self.thread_executor.shutdown(wait=True)