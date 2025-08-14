"""GraphToolKit exception hierarchy for strict error handling."""

from typing import Optional, List, Any, Tuple


class GraphToolKitError(Exception):
    """Base exception for all GraphToolKit errors."""
    pass


class DependencyError(GraphToolKitError):
    """Raised when required dependencies are missing or invalid."""
    pass


class SchemaValidationError(GraphToolKitError):
    """Raised when schema validation fails."""
    pass


class StorageError(GraphToolKitError):
    """Raised when storage operations fail."""
    pass


class TemplateError(GraphToolKitError):
    """Raised when template operations fail."""
    pass


class BatchProcessingError(GraphToolKitError):
    """Raised when batch processing fails with partial results."""
    
    def __init__(self, message: str, errors: Optional[List[Tuple[int, Any, Exception]]] = None):
        """Initialize with message and optional error details.
        
        Args:
            message: Error message
            errors: List of (index, item, exception) tuples for failed items
        """
        super().__init__(message)
        self.errors = errors or []


class CatalogError(GraphToolKitError):
    """Raised when catalog operations fail."""
    pass


class NodeExecutionError(GraphToolKitError):
    """Raised when node execution fails."""
    pass


class WorkflowError(GraphToolKitError):
    """Raised when workflow-level operations fail."""
    pass


class ConfigurationError(GraphToolKitError):
    """Raised when configuration is invalid or missing."""
    pass