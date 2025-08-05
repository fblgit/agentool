"""
Base schema for AgenTools.

This module provides the base schema that all AgenTool input schemas
should extend. It's part of the core framework, not an example.
"""

from pydantic import BaseModel, Field, ConfigDict


class BaseOperationInput(BaseModel):
    """
    Base schema for operation-based AgenTools.
    
    This provides a common base for all AgenTool input schemas that
    use an operation field to route to different tools.
    
    Attributes:
        operation: The operation to perform. Subclasses typically
                  override this with a Literal type for validation.
    
    Example:
        >>> from typing import Literal
        >>> from agentool import BaseOperationInput
        >>> 
        >>> class MyToolInput(BaseOperationInput):
        ...     operation: Literal['read', 'write', 'delete']
        ...     key: str
        ...     value: Optional[str] = None
    """
    model_config = ConfigDict(extra='forbid')  # Strict validation by default
    
    operation: str = Field(description="The operation to perform")