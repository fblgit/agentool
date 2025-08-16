"""
element_interactor AgenTool - Fine-grained DOM element interaction and manipulation capabilities.

This AgenTool provides comprehensive DOM element interaction and manipulation capabilities for Playwright browser automation.
It integrates with browser_manager for browser instance management, page_navigator for page state validation, 
storage_fs for saving screenshots and artifacts, logging for operation tracking, and metrics for performance monitoring.

Key Features:
- Element Finding: Support for CSS selectors, XPath, and text content selectors
- Interaction Operations: Click, type, select, hover, drag operations with advanced options
- Form Handling: File uploads, form filling, and submission capabilities
- Element State Queries: Text content, attributes, visibility, and element properties
- Wait Conditions: Element visibility, stability, and state waiting mechanisms
- Advanced Interactions: Keyboard shortcuts, mouse actions, and element screenshots

Usage Example:
    >>> from agentoolkit.element_interactor import create_element_interactor_agent
    >>> from agentool.core.injector import get_injector
    >>> 
    >>> # Create and register the agent
    >>> agent = create_element_interactor_agent()
    >>> 
    >>> # Use through injector
    >>> injector = get_injector()
    >>> result = await injector.run('element_interactor', {
    ...     "operation": "find_element",
    ...     "browser_id": "browser_123",
    ...     "selector": "#login-button"
    ... })
"""

import time
from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import RunContext

from agentool import create_agentool
from agentool.base import BaseOperationInput
from agentool.core.registry import RoutingConfig
from agentool.core.injector import get_injector


class ElementInteractorInput(BaseOperationInput):
    """Input schema for element_interactor operations."""
    
    operation: Literal[
        'find_element', 'click_element', 'type_text', 'select_option', 
        'get_element_text', 'get_element_attribute', 'wait_for_element', 
        'upload_file', 'hover_element', 'drag_element', 'keyboard_shortcut', 
        'element_screenshot'
    ] = Field(description="The element interaction operation to perform")
    
    browser_id: str = Field(description="Browser instance identifier from browser_manager")
    selector: str = Field(description="CSS selector, XPath (prefix with xpath=), or text content selector (prefix with text=) for element targeting")
    
    # Operation-specific fields
    text: Optional[str] = Field(None, description="Text to type into an element (for type_text operation)")
    value: Optional[str] = Field(None, description="Value to select in dropdown/select element (for select_option operation)")
    attribute: Optional[str] = Field(None, description="HTML attribute name to retrieve (for get_element_attribute operation)")
    file_path: Optional[str] = Field(None, description="Path to file for upload (for upload_file operation)")
    timeout: int = Field(30000, description="Timeout in milliseconds for element operations")
    wait_condition: str = Field("visible", description="Condition to wait for when using wait_for_element operation")
    target_selector: Optional[str] = Field(None, description="Target element selector for drag operations (for drag_element operation)")
    keys: Optional[str] = Field(None, description="Keyboard shortcut keys (e.g., 'Control+A', 'Enter') for keyboard_shortcut operation")
    clear_first: bool = Field(True, description="Whether to clear existing text before typing (for type_text operation)")
    click_count: int = Field(1, description="Number of clicks to perform (for click_element operation)")
    force: bool = Field(False, description="Force the action even if element is not actionable (for interaction operations)")
    
    @field_validator('text')
    @classmethod
    def validate_text_for_type_text(cls, v, info):
        """Validate text is provided for type_text operation."""
        operation = info.data.get('operation')
        if operation == 'type_text' and not v:
            raise ValueError("text is required for type_text operation")
        return v
    
    @field_validator('value')
    @classmethod
    def validate_value_for_select_option(cls, v, info):
        """Validate value is provided for select_option operation."""
        operation = info.data.get('operation')
        if operation == 'select_option' and not v:
            raise ValueError("value is required for select_option operation")
        return v
    
    @field_validator('attribute')
    @classmethod
    def validate_attribute_for_get_element_attribute(cls, v, info):
        """Validate attribute is provided for get_element_attribute operation."""
        operation = info.data.get('operation')
        if operation == 'get_element_attribute' and not v:
            raise ValueError("attribute is required for get_element_attribute operation")
        return v
    
    @field_validator('file_path')
    @classmethod
    def validate_file_path_for_upload_file(cls, v, info):
        """Validate file_path is provided for upload_file operation."""
        operation = info.data.get('operation')
        if operation == 'upload_file' and not v:
            raise ValueError("file_path is required for upload_file operation")
        return v
    
    @field_validator('target_selector')
    @classmethod
    def validate_target_selector_for_drag_element(cls, v, info):
        """Validate target_selector is provided for drag_element operation."""
        operation = info.data.get('operation')
        if operation == 'drag_element' and not v:
            raise ValueError("target_selector is required for drag_element operation")
        return v
    
    @field_validator('keys')
    @classmethod
    def validate_keys_for_keyboard_shortcut(cls, v, info):
        """Validate keys is provided for keyboard_shortcut operation."""
        operation = info.data.get('operation')
        if operation == 'keyboard_shortcut' and not v:
            raise ValueError("keys is required for keyboard_shortcut operation")
        return v
    
    @field_validator('wait_condition')
    @classmethod
    def validate_wait_condition(cls, v):
        """Validate wait_condition is one of the allowed values."""
        allowed_conditions = ["visible", "hidden", "attached", "detached", "stable"]
        if v not in allowed_conditions:
            raise ValueError(f"wait_condition must be one of {allowed_conditions}")
        return v


class ElementInteractorOutput(BaseModel):
    """Output schema for element_interactor operations."""
    success: bool = Field(description="Whether the element operation succeeded")
    message: str = Field(description="Human-readable result message describing the operation outcome")
    data: Optional[Dict[str, Any]] = Field(None, description="Operation-specific return data including element properties, text content, or interaction results")


async def element_interactor_find_element(
    ctx: RunContext[Any],
    browser_id: str,
    selector: str,
    timeout: int = 30000
) -> ElementInteractorOutput:
    """
    Find an element using CSS selector, XPath, or text content.
    
    Locates an element on the current page using various selector types.
    Supports CSS selectors, XPath (prefix with xpath=), and text content 
    selectors (prefix with text=).
    
    Args:
        ctx: Runtime context provided by the framework
        browser_id: Browser instance identifier from browser_manager
        selector: CSS selector, XPath, or text content selector for element targeting
        timeout: Timeout in milliseconds for element operations (default: 30000)
        
    Returns:
        ElementInteractorOutput with operation results containing:
        - success: Whether the element was found
        - message: Human-readable result message
        - data: Element properties including tag name, visibility, and enabled status
        
    Raises:
        BrowserNotFoundError: If browser_id does not exist or is closed
        ElementNotFoundError: If selector does not match any element within timeout
        InvalidSelectorError: If provided selector syntax is malformed
        TimeoutError: If element operation exceeds specified timeout
    """
    injector = get_injector()
    
    try:
        # Get browser instance from browser_manager
        browser_result = await injector.run('browser_manager', {
            'operation': 'get_browser',
            'browser_id': browser_id
        })
        
        if not browser_result.success:
            await injector.run('logging', {
                'operation': 'log',
                'level': 'ERROR',
                'logger_name': 'element_interactor',
                'message': f'Browser {browser_id} not found for find_element operation',
                'data': {'browser_id': browser_id, 'selector': selector}
            })
            
            await injector.run('metrics', {
                'operation': 'increment',
                'name': 'agentool.element_interactor.browser_not_found.errors'
            })
            
            return ElementInteractorOutput(
                success=False,
                message=f"Browser instance '{browser_id}' not found",
                data={"element_found": False, "selector": selector, "timeout_ms": timeout}
            )
        
        # Get page instance from browser data
        browser_data = browser_result.data
        page = browser_data.get('page')
        
        if not page:
            raise RuntimeError(f"No active page found for browser {browser_id}")
        
        # Normalize selector based on prefix
        playwright_selector = selector
        if selector.startswith('xpath='):
            playwright_selector = selector
        elif selector.startswith('text='):
            playwright_selector = selector
        else:
            # CSS selector - use as-is
            playwright_selector = selector
        
        # Find the element with timeout
        try:
            element = await page.locator(playwright_selector).first
            
            # Wait for element to be attached within timeout
            await element.wait_for(state="attached", timeout=timeout)
            
            # Get element properties
            tag_name = await element.evaluate("element => element.tagName.toLowerCase()")
            is_visible = await element.is_visible()
            is_enabled = await element.is_enabled()
            
            await injector.run('logging', {
                'operation': 'log',
                'level': 'INFO',
                'logger_name': 'element_interactor',
                'message': f'Successfully found element with selector {selector}',
                'data': {
                    'browser_id': browser_id,
                    'selector': selector,
                    'tag_name': tag_name,
                    'visible': is_visible,
                    'enabled': is_enabled
                }
            })
            
            await injector.run('metrics', {
                'operation': 'increment',
                'name': 'agentool.element_interactor.find_element.success'
            })
            
            return ElementInteractorOutput(
                success=True,
                message=f"Successfully found element with selector '{selector}'",
                data={
                    "element_found": True,
                    "selector": selector,
                    "tag_name": tag_name,
                    "visible": is_visible,
                    "enabled": is_enabled
                }
            )
            
        except Exception as element_error:
            await injector.run('logging', {
                'operation': 'log',
                'level': 'ERROR',
                'logger_name': 'element_interactor',
                'message': f'Element not found with selector {selector}',
                'data': {
                    'browser_id': browser_id,
                    'selector': selector,
                    'timeout_ms': timeout,
                    'error': str(element_error)
                }
            })
            
            await injector.run('metrics', {
                'operation': 'increment',
                'name': 'agentool.element_interactor.find_element.element_not_found'
            })
            
            return ElementInteractorOutput(
                success=False,
                message=f"Element not found with selector '{selector}'",
                data={
                    "element_found": False,
                    "selector": selector,
                    "timeout_ms": timeout
                }
            )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'element_interactor',
            'message': 'Find element operation failed',
            'data': {
                'browser_id': browser_id,
                'selector': selector,
                'error': str(e)
            }
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.element_interactor.find_element.errors'
        })
        
        raise RuntimeError(f"Find element operation failed: {str(e)}") from e


async def element_interactor_click_element(
    ctx: RunContext[Any],
    browser_id: str,
    selector: str,
    timeout: int = 30000,
    click_count: int = 1,
    force: bool = False
) -> ElementInteractorOutput:
    """
    Click an element on the page.
    
    Performs click interactions on elements with support for multiple clicks
    and forced interactions when elements are not immediately actionable.
    
    Args:
        ctx: Runtime context provided by the framework
        browser_id: Browser instance identifier from browser_manager
        selector: CSS selector, XPath, or text content selector for element targeting
        timeout: Timeout in milliseconds for element operations (default: 30000)
        click_count: Number of clicks to perform (default: 1)
        force: Force the action even if element is not actionable (default: False)
        
    Returns:
        ElementInteractorOutput with operation results containing:
        - success: Whether the click was successful
        - message: Human-readable result message
        - data: Click operation details
        
    Raises:
        BrowserNotFoundError: If browser_id does not exist or is closed
        ElementNotFoundError: If selector does not match any element within timeout
        ElementNotInteractableError: If element exists but cannot be clicked
        TimeoutError: If element operation exceeds specified timeout
    """
    injector = get_injector()
    
    try:
        # Get browser instance
        browser_result = await injector.run('browser_manager', {
            'operation': 'get_browser',
            'browser_id': browser_id
        })
        
        if not browser_result.success:
            return ElementInteractorOutput(
                success=False,
                message=f"Browser instance '{browser_id}' not found",
                data={"clicked": False, "selector": selector, "click_count": click_count}
            )
        
        page = browser_result.data.get('page')
        if not page:
            raise RuntimeError(f"No active page found for browser {browser_id}")
        
        # Normalize selector
        playwright_selector = selector
        element = page.locator(playwright_selector).first
        
        # Wait for element to be actionable or force click
        if not force:
            await element.wait_for(state="visible", timeout=timeout)
        
        # Perform click(s)
        for i in range(click_count):
            await element.click(force=force, timeout=timeout)
            if click_count > 1 and i < click_count - 1:
                await page.wait_for_timeout(100)  # Small delay between clicks
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'element_interactor',
            'message': f'Successfully clicked element {click_count} time(s)',
            'data': {
                'browser_id': browser_id,
                'selector': selector,
                'click_count': click_count,
                'force': force
            }
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.element_interactor.click_element.success'
        })
        
        return ElementInteractorOutput(
            success=True,
            message=f"Successfully clicked element with selector '{selector}'",
            data={
                "clicked": True,
                "selector": selector,
                "click_count": click_count
            }
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'element_interactor',
            'message': 'Click element operation failed',
            'data': {
                'browser_id': browser_id,
                'selector': selector,
                'error': str(e)
            }
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.element_interactor.click_element.errors'
        })
        
        raise RuntimeError(f"Click element operation failed: {str(e)}") from e


async def element_interactor_type_text(
    ctx: RunContext[Any],
    browser_id: str,
    selector: str,
    text: str,
    timeout: int = 30000,
    clear_first: bool = True
) -> ElementInteractorOutput:
    """
    Type text into an input element.
    
    Types text into input fields, textareas, and other editable elements
    with optional clearing of existing content.
    
    Args:
        ctx: Runtime context provided by the framework
        browser_id: Browser instance identifier from browser_manager
        selector: CSS selector, XPath, or text content selector for element targeting
        text: Text to type into the element
        timeout: Timeout in milliseconds for element operations (default: 30000)
        clear_first: Whether to clear existing text before typing (default: True)
        
    Returns:
        ElementInteractorOutput with operation results containing:
        - success: Whether the text was typed successfully
        - message: Human-readable result message
        - data: Text input operation details
        
    Raises:
        BrowserNotFoundError: If browser_id does not exist or is closed
        ElementNotFoundError: If selector does not match any element within timeout
        ElementNotInteractableError: If element exists but cannot receive text input
        TimeoutError: If element operation exceeds specified timeout
    """
    injector = get_injector()
    
    try:
        # Get browser instance
        browser_result = await injector.run('browser_manager', {
            'operation': 'get_browser',
            'browser_id': browser_id
        })
        
        if not browser_result.success:
            return ElementInteractorOutput(
                success=False,
                message=f"Browser instance '{browser_id}' not found",
                data={"text_entered": text, "selector": selector, "cleared_first": clear_first}
            )
        
        page = browser_result.data.get('page')
        if not page:
            raise RuntimeError(f"No active page found for browser {browser_id}")
        
        element = page.locator(selector).first
        
        # Wait for element to be visible and enabled
        await element.wait_for(state="visible", timeout=timeout)
        
        # Clear existing text if requested
        if clear_first:
            await element.clear(timeout=timeout)
        
        # Type the text
        await element.type(text, timeout=timeout)
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'element_interactor',
            'message': f'Successfully typed text into element',
            'data': {
                'browser_id': browser_id,
                'selector': selector,
                'text_length': len(text),
                'cleared_first': clear_first
            }
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.element_interactor.type_text.success'
        })
        
        return ElementInteractorOutput(
            success=True,
            message=f"Successfully typed text into element with selector '{selector}'",
            data={
                "text_entered": text,
                "selector": selector,
                "cleared_first": clear_first
            }
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'element_interactor',
            'message': 'Type text operation failed',
            'data': {
                'browser_id': browser_id,
                'selector': selector,
                'error': str(e)
            }
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.element_interactor.type_text.errors'
        })
        
        raise RuntimeError(f"Type text operation failed: {str(e)}") from e


async def element_interactor_select_option(
    ctx: RunContext[Any],
    browser_id: str,
    selector: str,
    value: str,
    timeout: int = 30000
) -> ElementInteractorOutput:
    """
    Select an option from a dropdown/select element.
    
    Selects options from HTML select elements by value, text, or index.
    
    Args:
        ctx: Runtime context provided by the framework
        browser_id: Browser instance identifier from browser_manager
        selector: CSS selector, XPath, or text content selector for select element
        value: Value to select in dropdown (can be option value, visible text, or index)
        timeout: Timeout in milliseconds for element operations (default: 30000)
        
    Returns:
        ElementInteractorOutput with operation results containing:
        - success: Whether the option was selected successfully
        - message: Human-readable result message
        - data: Select operation details
        
    Raises:
        BrowserNotFoundError: If browser_id does not exist or is closed
        ElementNotFoundError: If selector does not match any element within timeout
        ElementNotInteractableError: If element exists but is not a select element
        ValueError: If the specified value/option is not available
        TimeoutError: If element operation exceeds specified timeout
    """
    injector = get_injector()
    
    try:
        # Get browser instance
        browser_result = await injector.run('browser_manager', {
            'operation': 'get_browser',
            'browser_id': browser_id
        })
        
        if not browser_result.success:
            return ElementInteractorOutput(
                success=False,
                message=f"Browser instance '{browser_id}' not found",
                data={"selected": False, "selector": selector, "value": value}
            )
        
        page = browser_result.data.get('page')
        if not page:
            raise RuntimeError(f"No active page found for browser {browser_id}")
        
        element = page.locator(selector).first
        
        # Wait for element to be visible
        await element.wait_for(state="visible", timeout=timeout)
        
        # Select option by value, text, or index
        try:
            # Try selecting by value first
            await element.select_option(value, timeout=timeout)
        except Exception:
            try:
                # Try selecting by visible text
                await element.select_option(label=value, timeout=timeout)
            except Exception:
                # Try selecting by index if value is numeric
                if value.isdigit():
                    await element.select_option(index=int(value), timeout=timeout)
                else:
                    raise ValueError(f"Option '{value}' not found in select element")
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'element_interactor',
            'message': f'Successfully selected option in element',
            'data': {
                'browser_id': browser_id,
                'selector': selector,
                'value': value
            }
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.element_interactor.select_option.success'
        })
        
        return ElementInteractorOutput(
            success=True,
            message=f"Successfully selected option '{value}' in element with selector '{selector}'",
            data={
                "selected": True,
                "selector": selector,
                "value": value
            }
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'element_interactor',
            'message': 'Select option operation failed',
            'data': {
                'browser_id': browser_id,
                'selector': selector,
                'value': value,
                'error': str(e)
            }
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.element_interactor.select_option.errors'
        })
        
        raise RuntimeError(f"Select option operation failed: {str(e)}") from e


async def element_interactor_get_element_text(
    ctx: RunContext[Any],
    browser_id: str,
    selector: str,
    timeout: int = 30000
) -> ElementInteractorOutput:
    """
    Get text content from an element.
    
    Retrieves the visible text content from an element, excluding HTML tags.
    
    Args:
        ctx: Runtime context provided by the framework
        browser_id: Browser instance identifier from browser_manager
        selector: CSS selector, XPath, or text content selector for element targeting
        timeout: Timeout in milliseconds for element operations (default: 30000)
        
    Returns:
        ElementInteractorOutput with operation results containing:
        - success: Whether the text was retrieved successfully
        - message: Human-readable result message
        - data: Element text content
        
    Raises:
        BrowserNotFoundError: If browser_id does not exist or is closed
        ElementNotFoundError: If selector does not match any element within timeout
        TimeoutError: If element operation exceeds specified timeout
    """
    injector = get_injector()
    
    try:
        # Get browser instance
        browser_result = await injector.run('browser_manager', {
            'operation': 'get_browser',
            'browser_id': browser_id
        })
        
        if not browser_result.success:
            return ElementInteractorOutput(
                success=False,
                message=f"Browser instance '{browser_id}' not found",
                data={"text": None, "selector": selector}
            )
        
        page = browser_result.data.get('page')
        if not page:
            raise RuntimeError(f"No active page found for browser {browser_id}")
        
        element = page.locator(selector).first
        
        # Wait for element to be attached
        await element.wait_for(state="attached", timeout=timeout)
        
        # Get text content
        text_content = await element.text_content()
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'element_interactor',
            'message': f'Successfully retrieved text from element',
            'data': {
                'browser_id': browser_id,
                'selector': selector,
                'text_length': len(text_content) if text_content else 0
            }
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.element_interactor.get_element_text.success'
        })
        
        return ElementInteractorOutput(
            success=True,
            message=f"Successfully retrieved text from element with selector '{selector}'",
            data={
                "text": text_content,
                "selector": selector
            }
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'element_interactor',
            'message': 'Get element text operation failed',
            'data': {
                'browser_id': browser_id,
                'selector': selector,
                'error': str(e)
            }
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.element_interactor.get_element_text.errors'
        })
        
        raise RuntimeError(f"Get element text operation failed: {str(e)}") from e


async def element_interactor_get_element_attribute(
    ctx: RunContext[Any],
    browser_id: str,
    selector: str,
    attribute: str,
    timeout: int = 30000
) -> ElementInteractorOutput:
    """
    Get an attribute value from an element.
    
    Retrieves the specified HTML attribute value from an element.
    
    Args:
        ctx: Runtime context provided by the framework
        browser_id: Browser instance identifier from browser_manager
        selector: CSS selector, XPath, or text content selector for element targeting
        attribute: HTML attribute name to retrieve
        timeout: Timeout in milliseconds for element operations (default: 30000)
        
    Returns:
        ElementInteractorOutput with operation results containing:
        - success: Whether the attribute was retrieved successfully
        - message: Human-readable result message
        - data: Element attribute value
        
    Raises:
        BrowserNotFoundError: If browser_id does not exist or is closed
        ElementNotFoundError: If selector does not match any element within timeout
        TimeoutError: If element operation exceeds specified timeout
    """
    injector = get_injector()
    
    try:
        # Get browser instance
        browser_result = await injector.run('browser_manager', {
            'operation': 'get_browser',
            'browser_id': browser_id
        })
        
        if not browser_result.success:
            return ElementInteractorOutput(
                success=False,
                message=f"Browser instance '{browser_id}' not found",
                data={"attribute": attribute, "value": None, "selector": selector}
            )
        
        page = browser_result.data.get('page')
        if not page:
            raise RuntimeError(f"No active page found for browser {browser_id}")
        
        element = page.locator(selector).first
        
        # Wait for element to be attached
        await element.wait_for(state="attached", timeout=timeout)
        
        # Get attribute value
        attribute_value = await element.get_attribute(attribute)
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'element_interactor',
            'message': f'Successfully retrieved attribute {attribute} from element',
            'data': {
                'browser_id': browser_id,
                'selector': selector,
                'attribute': attribute,
                'has_value': attribute_value is not None
            }
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.element_interactor.get_element_attribute.success'
        })
        
        return ElementInteractorOutput(
            success=True,
            message=f"Successfully retrieved attribute '{attribute}' from element with selector '{selector}'",
            data={
                "attribute": attribute,
                "value": attribute_value,
                "selector": selector
            }
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'element_interactor',
            'message': 'Get element attribute operation failed',
            'data': {
                'browser_id': browser_id,
                'selector': selector,
                'attribute': attribute,
                'error': str(e)
            }
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.element_interactor.get_element_attribute.errors'
        })
        
        raise RuntimeError(f"Get element attribute operation failed: {str(e)}") from e


async def element_interactor_wait_for_element(
    ctx: RunContext[Any],
    browser_id: str,
    selector: str,
    timeout: int = 30000,
    wait_condition: str = "visible"
) -> ElementInteractorOutput:
    """
    Wait for an element to meet a specific condition.
    
    Waits for an element to become visible, hidden, attached, detached, or stable
    within the specified timeout period.
    
    Args:
        ctx: Runtime context provided by the framework
        browser_id: Browser instance identifier from browser_manager
        selector: CSS selector, XPath, or text content selector for element targeting
        timeout: Timeout in milliseconds for element operations (default: 30000)
        wait_condition: Condition to wait for - visible, hidden, attached, detached, stable (default: visible)
        
    Returns:
        ElementInteractorOutput with operation results containing:
        - success: Whether the condition was met within timeout
        - message: Human-readable result message
        - data: Wait operation details including actual wait time
        
    Raises:
        BrowserNotFoundError: If browser_id does not exist or is closed
        TimeoutError: If element condition is not met within specified timeout
        ValueError: If wait_condition is not supported
    """
    injector = get_injector()
    start_time = time.time()
    
    try:
        # Get browser instance
        browser_result = await injector.run('browser_manager', {
            'operation': 'get_browser',
            'browser_id': browser_id
        })
        
        if not browser_result.success:
            return ElementInteractorOutput(
                success=False,
                message=f"Browser instance '{browser_id}' not found",
                data={"condition_met": False, "selector": selector, "wait_condition": wait_condition}
            )
        
        page = browser_result.data.get('page')
        if not page:
            raise RuntimeError(f"No active page found for browser {browser_id}")
        
        element = page.locator(selector).first
        
        # Wait for the specified condition
        await element.wait_for(state=wait_condition, timeout=timeout)
        
        wait_time_ms = int((time.time() - start_time) * 1000)
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'element_interactor',
            'message': f'Successfully waited for element condition {wait_condition}',
            'data': {
                'browser_id': browser_id,
                'selector': selector,
                'wait_condition': wait_condition,
                'wait_time_ms': wait_time_ms
            }
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.element_interactor.wait_for_element.success'
        })
        
        return ElementInteractorOutput(
            success=True,
            message=f"Successfully waited for element with selector '{selector}' to become {wait_condition}",
            data={
                "condition_met": True,
                "selector": selector,
                "wait_condition": wait_condition,
                "wait_time_ms": wait_time_ms
            }
        )
        
    except Exception as e:
        wait_time_ms = int((time.time() - start_time) * 1000)
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'element_interactor',
            'message': 'Wait for element operation failed',
            'data': {
                'browser_id': browser_id,
                'selector': selector,
                'wait_condition': wait_condition,
                'wait_time_ms': wait_time_ms,
                'error': str(e)
            }
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.element_interactor.wait_for_element.errors'
        })
        
        raise RuntimeError(f"Wait for element operation failed: {str(e)}") from e


async def element_interactor_upload_file(
    ctx: RunContext[Any],
    browser_id: str,
    selector: str,
    file_path: str,
    timeout: int = 30000
) -> ElementInteractorOutput:
    """
    Upload a file to a file input element.
    
    Sets files on file input elements for form upload operations.
    
    Args:
        ctx: Runtime context provided by the framework
        browser_id: Browser instance identifier from browser_manager
        selector: CSS selector, XPath, or text content selector for file input element
        file_path: Path to file for upload
        timeout: Timeout in milliseconds for element operations (default: 30000)
        
    Returns:
        ElementInteractorOutput with operation results containing:
        - success: Whether the file was uploaded successfully
        - message: Human-readable result message
        - data: Upload operation details
        
    Raises:
        BrowserNotFoundError: If browser_id does not exist or is closed
        ElementNotFoundError: If selector does not match any element within timeout
        FileNotFoundError: If specified file path does not exist
        ElementNotInteractableError: If element is not a file input
        TimeoutError: If element operation exceeds specified timeout
    """
    injector = get_injector()
    
    try:
        # Check if file exists using storage_fs
        file_check = await injector.run('storage_fs', {
            'operation': 'exists',
            'path': file_path
        })
        
        if not file_check.success or not file_check.data.get('exists'):
            await injector.run('logging', {
                'operation': 'log',
                'level': 'ERROR',
                'logger_name': 'element_interactor',
                'message': f'File not found for upload: {file_path}',
                'data': {
                    'browser_id': browser_id,
                    'selector': selector,
                    'file_path': file_path
                }
            })
            
            await injector.run('metrics', {
                'operation': 'increment',
                'name': 'agentool.element_interactor.upload_file.file_not_found'
            })
            
            return ElementInteractorOutput(
                success=False,
                message=f"File not found: {file_path}",
                data={"file_uploaded": False, "file_path": file_path, "selector": selector}
            )
        
        # Get browser instance
        browser_result = await injector.run('browser_manager', {
            'operation': 'get_browser',
            'browser_id': browser_id
        })
        
        if not browser_result.success:
            return ElementInteractorOutput(
                success=False,
                message=f"Browser instance '{browser_id}' not found",
                data={"file_uploaded": False, "file_path": file_path, "selector": selector}
            )
        
        page = browser_result.data.get('page')
        if not page:
            raise RuntimeError(f"No active page found for browser {browser_id}")
        
        element = page.locator(selector).first
        
        # Wait for element to be visible
        await element.wait_for(state="visible", timeout=timeout)
        
        # Set the file on the input element
        await element.set_input_files(file_path)
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'element_interactor',
            'message': f'Successfully uploaded file to element',
            'data': {
                'browser_id': browser_id,
                'selector': selector,
                'file_path': file_path
            }
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.element_interactor.upload_file.success'
        })
        
        return ElementInteractorOutput(
            success=True,
            message=f"Successfully uploaded file to element with selector '{selector}'",
            data={
                "file_uploaded": True,
                "file_path": file_path,
                "selector": selector
            }
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'element_interactor',
            'message': 'Upload file operation failed',
            'data': {
                'browser_id': browser_id,
                'selector': selector,
                'file_path': file_path,
                'error': str(e)
            }
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.element_interactor.upload_file.errors'
        })
        
        raise RuntimeError(f"Upload file operation failed: {str(e)}") from e


async def element_interactor_hover_element(
    ctx: RunContext[Any],
    browser_id: str,
    selector: str,
    timeout: int = 30000
) -> ElementInteractorOutput:
    """
    Hover over an element.
    
    Performs a mouse hover action over the specified element.
    
    Args:
        ctx: Runtime context provided by the framework
        browser_id: Browser instance identifier from browser_manager
        selector: CSS selector, XPath, or text content selector for element targeting
        timeout: Timeout in milliseconds for element operations (default: 30000)
        
    Returns:
        ElementInteractorOutput with operation results containing:
        - success: Whether the hover was successful
        - message: Human-readable result message
        - data: Hover operation details
        
    Raises:
        BrowserNotFoundError: If browser_id does not exist or is closed
        ElementNotFoundError: If selector does not match any element within timeout
        ElementNotInteractableError: If element exists but cannot be hovered
        TimeoutError: If element operation exceeds specified timeout
    """
    injector = get_injector()
    
    try:
        # Get browser instance
        browser_result = await injector.run('browser_manager', {
            'operation': 'get_browser',
            'browser_id': browser_id
        })
        
        if not browser_result.success:
            return ElementInteractorOutput(
                success=False,
                message=f"Browser instance '{browser_id}' not found",
                data={"hovered": False, "selector": selector}
            )
        
        page = browser_result.data.get('page')
        if not page:
            raise RuntimeError(f"No active page found for browser {browser_id}")
        
        element = page.locator(selector).first
        
        # Wait for element to be visible
        await element.wait_for(state="visible", timeout=timeout)
        
        # Hover over the element
        await element.hover(timeout=timeout)
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'element_interactor',
            'message': f'Successfully hovered over element',
            'data': {
                'browser_id': browser_id,
                'selector': selector
            }
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.element_interactor.hover_element.success'
        })
        
        return ElementInteractorOutput(
            success=True,
            message=f"Successfully hovered over element with selector '{selector}'",
            data={
                "hovered": True,
                "selector": selector
            }
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'element_interactor',
            'message': 'Hover element operation failed',
            'data': {
                'browser_id': browser_id,
                'selector': selector,
                'error': str(e)
            }
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.element_interactor.hover_element.errors'
        })
        
        raise RuntimeError(f"Hover element operation failed: {str(e)}") from e


async def element_interactor_drag_element(
    ctx: RunContext[Any],
    browser_id: str,
    selector: str,
    target_selector: str,
    timeout: int = 30000
) -> ElementInteractorOutput:
    """
    Drag an element to another element.
    
    Performs drag and drop operation from source element to target element.
    
    Args:
        ctx: Runtime context provided by the framework
        browser_id: Browser instance identifier from browser_manager
        selector: CSS selector for source element to drag
        target_selector: CSS selector for target element to drop on
        timeout: Timeout in milliseconds for element operations (default: 30000)
        
    Returns:
        ElementInteractorOutput with operation results containing:
        - success: Whether the drag operation was successful
        - message: Human-readable result message
        - data: Drag operation details
        
    Raises:
        BrowserNotFoundError: If browser_id does not exist or is closed
        ElementNotFoundError: If source or target selector does not match any element within timeout
        ElementNotInteractableError: If elements exist but cannot be dragged/dropped
        TimeoutError: If element operation exceeds specified timeout
    """
    injector = get_injector()
    
    try:
        # Get browser instance
        browser_result = await injector.run('browser_manager', {
            'operation': 'get_browser',
            'browser_id': browser_id
        })
        
        if not browser_result.success:
            return ElementInteractorOutput(
                success=False,
                message=f"Browser instance '{browser_id}' not found",
                data={"dragged": False, "selector": selector, "target_selector": target_selector}
            )
        
        page = browser_result.data.get('page')
        if not page:
            raise RuntimeError(f"No active page found for browser {browser_id}")
        
        source_element = page.locator(selector).first
        target_element = page.locator(target_selector).first
        
        # Wait for both elements to be visible
        await source_element.wait_for(state="visible", timeout=timeout)
        await target_element.wait_for(state="visible", timeout=timeout)
        
        # Perform drag and drop
        await source_element.drag_to(target_element, timeout=timeout)
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'element_interactor',
            'message': f'Successfully dragged element to target',
            'data': {
                'browser_id': browser_id,
                'selector': selector,
                'target_selector': target_selector
            }
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.element_interactor.drag_element.success'
        })
        
        return ElementInteractorOutput(
            success=True,
            message=f"Successfully dragged element '{selector}' to '{target_selector}'",
            data={
                "dragged": True,
                "selector": selector,
                "target_selector": target_selector
            }
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'element_interactor',
            'message': 'Drag element operation failed',
            'data': {
                'browser_id': browser_id,
                'selector': selector,
                'target_selector': target_selector,
                'error': str(e)
            }
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.element_interactor.drag_element.errors'
        })
        
        raise RuntimeError(f"Drag element operation failed: {str(e)}") from e


async def element_interactor_keyboard_shortcut(
    ctx: RunContext[Any],
    browser_id: str,
    selector: str,
    keys: str,
    timeout: int = 30000
) -> ElementInteractorOutput:
    """
    Send keyboard shortcut keys to an element.
    
    Performs keyboard shortcut actions on elements, such as Ctrl+A, Ctrl+C, Enter, etc.
    
    Args:
        ctx: Runtime context provided by the framework
        browser_id: Browser instance identifier from browser_manager
        selector: CSS selector, XPath, or text content selector for element targeting
        keys: Keyboard shortcut keys (e.g., 'Control+A', 'Enter', 'Escape')
        timeout: Timeout in milliseconds for element operations (default: 30000)
        
    Returns:
        ElementInteractorOutput with operation results containing:
        - success: Whether the keyboard shortcut was successful
        - message: Human-readable result message
        - data: Keyboard shortcut operation details
        
    Raises:
        BrowserNotFoundError: If browser_id does not exist or is closed
        ElementNotFoundError: If selector does not match any element within timeout
        ElementNotInteractableError: If element exists but cannot receive keyboard input
        TimeoutError: If element operation exceeds specified timeout
        ValueError: If keys format is invalid
    """
    injector = get_injector()
    
    try:
        # Get browser instance
        browser_result = await injector.run('browser_manager', {
            'operation': 'get_browser',
            'browser_id': browser_id
        })
        
        if not browser_result.success:
            return ElementInteractorOutput(
                success=False,
                message=f"Browser instance '{browser_id}' not found",
                data={"keys_sent": False, "selector": selector, "keys": keys}
            )
        
        page = browser_result.data.get('page')
        if not page:
            raise RuntimeError(f"No active page found for browser {browser_id}")
        
        element = page.locator(selector).first
        
        # Wait for element to be visible and focus on it
        await element.wait_for(state="visible", timeout=timeout)
        await element.focus()
        
        # Send the keyboard shortcut
        await element.press(keys, timeout=timeout)
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'element_interactor',
            'message': f'Successfully sent keyboard shortcut to element',
            'data': {
                'browser_id': browser_id,
                'selector': selector,
                'keys': keys
            }
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.element_interactor.keyboard_shortcut.success'
        })
        
        return ElementInteractorOutput(
            success=True,
            message=f"Successfully sent keyboard shortcut '{keys}' to element with selector '{selector}'",
            data={
                "keys_sent": True,
                "selector": selector,
                "keys": keys
            }
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'element_interactor',
            'message': 'Keyboard shortcut operation failed',
            'data': {
                'browser_id': browser_id,
                'selector': selector,
                'keys': keys,
                'error': str(e)
            }
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.element_interactor.keyboard_shortcut.errors'
        })
        
        raise RuntimeError(f"Keyboard shortcut operation failed: {str(e)}") from e


async def element_interactor_element_screenshot(
    ctx: RunContext[Any],
    browser_id: str,
    selector: str,
    timeout: int = 30000
) -> ElementInteractorOutput:
    """
    Capture a screenshot of a specific element.
    
    Takes a screenshot of the specified element and saves it using storage_fs.
    
    Args:
        ctx: Runtime context provided by the framework
        browser_id: Browser instance identifier from browser_manager
        selector: CSS selector, XPath, or text content selector for element targeting
        timeout: Timeout in milliseconds for element operations (default: 30000)
        
    Returns:
        ElementInteractorOutput with operation results containing:
        - success: Whether the screenshot was captured successfully
        - message: Human-readable result message
        - data: Screenshot details including path and dimensions
        
    Raises:
        BrowserNotFoundError: If browser_id does not exist or is closed
        ElementNotFoundError: If selector does not match any element within timeout
        TimeoutError: If element operation exceeds specified timeout
        FileSystemError: If screenshot cannot be saved to storage_fs
    """
    injector = get_injector()
    
    try:
        # Get browser instance
        browser_result = await injector.run('browser_manager', {
            'operation': 'get_browser',
            'browser_id': browser_id
        })
        
        if not browser_result.success:
            return ElementInteractorOutput(
                success=False,
                message=f"Browser instance '{browser_id}' not found",
                data={"screenshot_taken": False, "selector": selector}
            )
        
        page = browser_result.data.get('page')
        if not page:
            raise RuntimeError(f"No active page found for browser {browser_id}")
        
        element = page.locator(selector).first
        
        # Wait for element to be visible
        await element.wait_for(state="visible", timeout=timeout)
        
        # Generate screenshot filename
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_filename = f"element_screenshot_{browser_id}_{timestamp}.png"
        screenshot_path = f"screenshots/{screenshot_filename}"
        
        # Ensure screenshots directory exists
        await injector.run('storage_fs', {
            'operation': 'mkdir',
            'path': 'screenshots',
            'create_parents': True
        })
        
        # Take screenshot of the element
        screenshot_bytes = await element.screenshot()
        
        # Save screenshot using storage_fs
        import base64
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode()
        
        # Write as binary file (base64 decoded)
        with open(screenshot_path, 'wb') as f:
            f.write(screenshot_bytes)
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'element_interactor',
            'message': f'Successfully captured element screenshot',
            'data': {
                'browser_id': browser_id,
                'selector': selector,
                'screenshot_path': screenshot_path,
                'size_bytes': len(screenshot_bytes)
            }
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.element_interactor.element_screenshot.success'
        })
        
        return ElementInteractorOutput(
            success=True,
            message=f"Successfully captured screenshot of element with selector '{selector}'",
            data={
                "screenshot_taken": True,
                "selector": selector,
                "screenshot_path": screenshot_path,
                "size_bytes": len(screenshot_bytes)
            }
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'element_interactor',
            'message': 'Element screenshot operation failed',
            'data': {
                'browser_id': browser_id,
                'selector': selector,
                'error': str(e)
            }
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.element_interactor.element_screenshot.errors'
        })
        
        raise RuntimeError(f"Element screenshot operation failed: {str(e)}") from e


# Routing configuration
element_interactor_routing = RoutingConfig(
    operation_field='operation',
    operation_map={
        'find_element': ('element_interactor_find_element', lambda x: {
            'browser_id': x.browser_id,
            'selector': x.selector,
            'timeout': x.timeout,
        }),
        'click_element': ('element_interactor_click_element', lambda x: {
            'browser_id': x.browser_id,
            'selector': x.selector,
            'timeout': x.timeout,
            'click_count': x.click_count,
            'force': x.force,
        }),
        'type_text': ('element_interactor_type_text', lambda x: {
            'browser_id': x.browser_id,
            'selector': x.selector,
            'text': x.text,
            'timeout': x.timeout,
            'clear_first': x.clear_first,
        }),
        'select_option': ('element_interactor_select_option', lambda x: {
            'browser_id': x.browser_id,
            'selector': x.selector,
            'value': x.value,
            'timeout': x.timeout,
        }),
        'get_element_text': ('element_interactor_get_element_text', lambda x: {
            'browser_id': x.browser_id,
            'selector': x.selector,
            'timeout': x.timeout,
        }),
        'get_element_attribute': ('element_interactor_get_element_attribute', lambda x: {
            'browser_id': x.browser_id,
            'selector': x.selector,
            'attribute': x.attribute,
            'timeout': x.timeout,
        }),
        'wait_for_element': ('element_interactor_wait_for_element', lambda x: {
            'browser_id': x.browser_id,
            'selector': x.selector,
            'timeout': x.timeout,
            'wait_condition': x.wait_condition,
        }),
        'upload_file': ('element_interactor_upload_file', lambda x: {
            'browser_id': x.browser_id,
            'selector': x.selector,
            'file_path': x.file_path,
            'timeout': x.timeout,
        }),
        'hover_element': ('element_interactor_hover_element', lambda x: {
            'browser_id': x.browser_id,
            'selector': x.selector,
            'timeout': x.timeout,
        }),
        'drag_element': ('element_interactor_drag_element', lambda x: {
            'browser_id': x.browser_id,
            'selector': x.selector,
            'target_selector': x.target_selector,
            'timeout': x.timeout,
        }),
        'keyboard_shortcut': ('element_interactor_keyboard_shortcut', lambda x: {
            'browser_id': x.browser_id,
            'selector': x.selector,
            'keys': x.keys,
            'timeout': x.timeout,
        }),
        'element_screenshot': ('element_interactor_element_screenshot', lambda x: {
            'browser_id': x.browser_id,
            'selector': x.selector,
            'timeout': x.timeout,
        }),
    }
)


def create_element_interactor_agent():
    """
    Create and return the element_interactor AgenTool.
    
    Returns:
        Agent configured for element_interactor operations
    """
    return create_agentool(
        name='element_interactor',
        input_schema=ElementInteractorInput,
        routing_config=element_interactor_routing,
        tools=[
            element_interactor_find_element, element_interactor_click_element,
            element_interactor_type_text, element_interactor_select_option,
            element_interactor_get_element_text, element_interactor_get_element_attribute,
            element_interactor_wait_for_element, element_interactor_upload_file,
            element_interactor_hover_element, element_interactor_drag_element,
            element_interactor_keyboard_shortcut, element_interactor_element_screenshot
        ],
        output_type=ElementInteractorOutput,
        system_prompt="You are an expert DOM element interaction agent that provides fine-grained browser automation capabilities. You can find elements using various selectors, perform interactions like clicking and typing, handle form operations, query element states, and execute advanced interactions like drag-and-drop and keyboard shortcuts. You work with browser instances managed by browser_manager and integrate with page_navigator for page state validation.",
        description="Provides fine-grained DOM element interaction and manipulation capabilities including find_element, click_element, type_text, select_option, get_element_text, get_element_attribute, wait_for_element, upload_file, hover_element, drag_element, keyboard_shortcut, and element_screenshot operations",
        version="1.0.0",
        tags=["browser", "automation", "dom", "elements", "interaction"],
        dependencies=["browser_manager", "page_navigator", "storage_fs", "logging", "metrics"],
        examples=[
            {
                "description": "Find an element using CSS selector",
                "input": {
                    "operation": "find_element",
                    "browser_id": "browser_123",
                    "selector": "#login-button"
                },
                "output": {
                    "success": True,
                    "message": "Successfully found element with selector '#login-button'",
                    "data": {
                        "element_found": True,
                        "selector": "#login-button",
                        "tag_name": "button",
                        "visible": True,
                        "enabled": True
                    }
                }
            },
            {
                "description": "Click an element",
                "input": {
                    "operation": "click_element",
                    "browser_id": "browser_123",
                    "selector": "button[type='submit']"
                },
                "output": {
                    "success": True,
                    "message": "Successfully clicked element with selector 'button[type='submit']'",
                    "data": {
                        "clicked": True,
                        "selector": "button[type='submit']",
                        "click_count": 1
                    }
                }
            },
            {
                "description": "Type text into an input field",
                "input": {
                    "operation": "type_text",
                    "browser_id": "browser_123",
                    "selector": "input[name='username']",
                    "text": "john.doe@example.com"
                },
                "output": {
                    "success": True,
                    "message": "Successfully typed text into element with selector 'input[name='username']'",
                    "data": {
                        "text_entered": "john.doe@example.com",
                        "selector": "input[name='username']",
                        "cleared_first": True
                    }
                }
            }
        ]
    )


# Create the agent instance
agent = create_element_interactor_agent()