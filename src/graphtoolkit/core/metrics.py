"""GraphToolkit Metrics Integration.

Provides metrics tracking integration with agentoolkit metrics system.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from agentool.core.injector import get_injector

logger = logging.getLogger(__name__)


@dataclass 
class MetricsConfig:
    """Configuration for metrics tracking."""
    enabled: bool = True
    prefix: str = 'graphtoolkit'
    track_node_duration: bool = True
    track_node_execution: bool = True
    track_errors: bool = True
    track_retries: bool = True
    auto_create_metrics: bool = True


class MetricsMixin:
    """Mixin for atomic nodes to provide metrics tracking.
    Integrates with agentoolkit metrics system.
    """
    
    async def _track_execution_start(self, node_name: str, phase: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Track the start of node execution."""
        try:
            start_time = time.time()
            
            injector = get_injector()
            
            # Track execution start
            base_labels = {
                'node': node_name,
                'phase': phase
            }
            if labels:
                base_labels.update(labels)
            
            # Increment execution count
            await injector.run('metrics', {
                'operation': 'increment',
                'name': f'graphtoolkit.{node_name}.executions.total',
                'value': 1,
                'labels': base_labels
            })
            
            # Log start
            await injector.run('logging', {
                'operation': 'log',
                'level': 'DEBUG',
                'logger_name': f'graphtoolkit.{node_name}',
                'message': f'Starting execution of {node_name} in phase {phase}',
                'data': base_labels
            })
            
            return start_time
            
        except Exception as e:
            logger.warning(f'Failed to track execution start for {node_name}: {e}')
            return time.time()
    
    async def _track_execution_success(self, node_name: str, phase: str, start_time: float, 
                                     result: Any = None, labels: Optional[Dict[str, str]] = None):
        """Track successful node execution."""
        try:
            duration = time.time() - start_time
            
            injector = get_injector()
            
            base_labels = {
                'node': node_name,
                'phase': phase,
                'status': 'success'
            }
            if labels:
                base_labels.update(labels)
            
            # Track success
            await injector.run('metrics', {
                'operation': 'increment',
                'name': f'graphtoolkit.{node_name}.executions.success',
                'value': 1,
                'labels': base_labels
            })
            
            # Track duration
            await injector.run('metrics', {
                'operation': 'observe',
                'name': f'graphtoolkit.{node_name}.duration.seconds',
                'value': duration,
                'labels': base_labels
            })
            
            # Track phase-level metrics
            await injector.run('metrics', {
                'operation': 'increment',
                'name': f'graphtoolkit.phase.{phase}.nodes.success',
                'value': 1,
                'labels': {'node': node_name}
            })
            
            # Log success
            await injector.run('logging', {
                'operation': 'log',
                'level': 'INFO',
                'logger_name': f'graphtoolkit.{node_name}',
                'message': f'Successfully executed {node_name} in {duration:.3f}s',
                'data': {
                    **base_labels,
                    'duration_seconds': duration,
                    'result_type': type(result).__name__ if result is not None else None
                }
            })
            
        except Exception as e:
            logger.warning(f'Failed to track execution success for {node_name}: {e}')
    
    async def _track_execution_failure(self, node_name: str, phase: str, start_time: float, 
                                     error: Exception, labels: Optional[Dict[str, str]] = None):
        """Track failed node execution."""
        try:
            duration = time.time() - start_time
            
            injector = get_injector()
            
            base_labels = {
                'node': node_name,
                'phase': phase,
                'status': 'failure',
                'error_type': type(error).__name__
            }
            if labels:
                base_labels.update(labels)
            
            # Track failure
            await injector.run('metrics', {
                'operation': 'increment',
                'name': f'graphtoolkit.{node_name}.executions.failure',
                'value': 1,
                'labels': base_labels
            })
            
            # Track error-specific metric
            await injector.run('metrics', {
                'operation': 'increment',
                'name': f'graphtoolkit.errors.{type(error).__name__}',
                'value': 1,
                'labels': {
                    'node': node_name,
                    'phase': phase
                }
            })
            
            # Track phase-level failure
            await injector.run('metrics', {
                'operation': 'increment',
                'name': f'graphtoolkit.phase.{phase}.nodes.failure',
                'value': 1,
                'labels': {'node': node_name}
            })
            
            # Log failure
            await injector.run('logging', {
                'operation': 'log',
                'level': 'ERROR',
                'logger_name': f'graphtoolkit.{node_name}',
                'message': f'Failed to execute {node_name} after {duration:.3f}s: {str(error)}',
                'data': {
                    **base_labels,
                    'duration_seconds': duration,
                    'error_message': str(error)
                }
            })
            
        except Exception as e:
            logger.warning(f'Failed to track execution failure for {node_name}: {e}')
    
    async def _track_retry_attempt(self, node_name: str, phase: str, retry_count: int, 
                                 error: Exception, labels: Optional[Dict[str, str]] = None):
        """Track retry attempts."""
        try:
            injector = get_injector()
            
            base_labels = {
                'node': node_name,
                'phase': phase,
                'retry_count': str(retry_count),
                'error_type': type(error).__name__
            }
            if labels:
                base_labels.update(labels)
            
            # Track retry attempt
            await injector.run('metrics', {
                'operation': 'increment',
                'name': f'graphtoolkit.{node_name}.retries.total',
                'value': 1,
                'labels': base_labels
            })
            
            # Track retry by error type
            await injector.run('metrics', {
                'operation': 'increment',
                'name': f'graphtoolkit.retries.{type(error).__name__}',
                'value': 1,
                'labels': {
                    'node': node_name,
                    'phase': phase,
                    'retry_count': str(retry_count)
                }
            })
            
            # Log retry
            await injector.run('logging', {
                'operation': 'log',
                'level': 'WARN',
                'logger_name': f'graphtoolkit.{node_name}',
                'message': f'Retrying {node_name} (attempt {retry_count}) after error: {str(error)}',
                'data': {
                    **base_labels,
                    'error_message': str(error)
                }
            })
            
        except Exception as e:
            logger.warning(f'Failed to track retry attempt for {node_name}: {e}')
    
    async def _track_storage_operation(self, operation: str, storage_type: str, key: str, 
                                     success: bool, duration: float, size_bytes: Optional[int] = None):
        """Track storage operations."""
        try:
            injector = get_injector()
            
            labels = {
                'operation': operation,
                'storage_type': storage_type,
                'status': 'success' if success else 'failure'
            }
            
            # Track storage operation count
            await injector.run('metrics', {
                'operation': 'increment',
                'name': 'graphtoolkit.storage.operations.total',
                'value': 1,
                'labels': labels
            })
            
            # Track storage operation duration
            await injector.run('metrics', {
                'operation': 'observe',
                'name': 'graphtoolkit.storage.duration.seconds',
                'value': duration,
                'labels': labels
            })
            
            # Track bytes if provided
            if size_bytes is not None:
                await injector.run('metrics', {
                    'operation': 'observe',
                    'name': 'graphtoolkit.storage.bytes',
                    'value': size_bytes,
                    'labels': {
                        'operation': operation,
                        'storage_type': storage_type
                    }
                })
            
        except Exception as e:
            logger.warning(f'Failed to track storage operation: {e}')
    
    async def _track_llm_operation(self, model: str, prompt_tokens: int, completion_tokens: int, 
                                 duration: float, success: bool, temperature: Optional[float] = None):
        """Track LLM operations."""
        try:
            injector = get_injector()
            
            labels = {
                'model': model,
                'status': 'success' if success else 'failure'
            }
            
            if temperature is not None:
                labels['temperature'] = str(temperature)
            
            # Track LLM calls
            await injector.run('metrics', {
                'operation': 'increment',
                'name': 'graphtoolkit.llm.calls.total',
                'value': 1,
                'labels': labels
            })
            
            # Track tokens
            total_tokens = prompt_tokens + completion_tokens
            await injector.run('metrics', {
                'operation': 'increment',
                'name': 'graphtoolkit.llm.tokens.total',
                'value': total_tokens,
                'labels': labels
            })
            
            await injector.run('metrics', {
                'operation': 'increment',
                'name': 'graphtoolkit.llm.tokens.prompt',
                'value': prompt_tokens,
                'labels': labels
            })
            
            await injector.run('metrics', {
                'operation': 'increment',
                'name': 'graphtoolkit.llm.tokens.completion',
                'value': completion_tokens,
                'labels': labels
            })
            
            # Track duration
            await injector.run('metrics', {
                'operation': 'observe',
                'name': 'graphtoolkit.llm.duration.seconds',
                'value': duration,
                'labels': labels
            })
            
        except Exception as e:
            logger.warning(f'Failed to track LLM operation: {e}')
    
    def _get_node_name(self) -> str:
        """Get the node name for metrics."""
        return self.__class__.__name__.replace('Node', '').lower()