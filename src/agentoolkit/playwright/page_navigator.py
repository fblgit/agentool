"""
page_navigator AgenTool - Handles page navigation, URL management, and high-level browser interactions.

This AgenTool provides comprehensive page navigation capabilities including URL navigation,
waiting for page states, content extraction with BeautifulSoup, screenshot capture,
cookie management, storage operations, and JavaScript evaluation.

It integrates with browser_manager for browser instance management, storage_fs for file operations,
storage_kv for state persistence, logging for operation tracking, and metrics for performance monitoring.

Key Features:
- Page Navigation: Navigate to URLs, back, forward, refresh with configurable wait conditions
- Content Extraction: HTML content extraction with BeautifulSoup parsing for structured data
- Screenshot Capture: Full page and viewport screenshots saved to storage_fs
- Cookie Management: Get, set, delete, and clear cookies with domain/security options
- Storage Operations: Local and session storage manipulation (get, set, remove, clear)
- JavaScript Evaluation: Execute custom JavaScript code and return results
- State Management: Page state tracking and navigation metadata persistence

Usage Example:
    >>> from agentoolkit.page_navigator import create_page_navigator_agent
    >>> from agentool.core.injector import get_injector
    >>> 
    >>> # Create and register the agent
    >>> agent = create_page_navigator_agent()
    >>> 
    >>> # Use through injector
    >>> injector = get_injector()
    >>> result = await injector.run('page_navigator', {
    ...     "operation": "navigate",
    ...     "browser_id": "browser_123",
    ...     "url": "https://example.com",
    ...     "wait_condition": "load"
    ... })
"""

import time
from typing import Dict, Any, List, Optional, Literal, Union
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import RunContext

from agentool import create_agentool
from agentool.base import BaseOperationInput
from agentool.core.registry import RoutingConfig
from agentool.core.injector import get_injector

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


class PageNavigatorInput(BaseOperationInput):
    """Input schema for page navigator operations."""
    
    operation: Literal[
        'navigate', 'back', 'forward', 'refresh', 'wait_for_load',
        'capture_screenshot', 'get_content', 'manage_cookies',
        'manage_storage', 'evaluate'
    ] = Field(description="Navigation operation to perform")
    
    browser_id: str = Field(description="Browser session identifier")
    
    # Navigation parameters
    url: Optional[str] = Field(None, description="URL for navigate operation")
    wait_condition: Optional[Literal['load', 'domcontentloaded', 'networkidle']] = Field(
        None, description="Wait condition for navigation or wait_for_load operation"
    )
    timeout: Optional[float] = Field(None, description="Timeout in milliseconds for operation")
    selector: Optional[str] = Field(None, description="CSS selector for element-based wait conditions")
    
    # Screenshot parameters
    screenshot_path: Optional[str] = Field(None, description="File path for screenshot capture")
    full_page: Optional[bool] = Field(None, description="Whether to capture full page screenshot")
    
    # Cookie management parameters
    cookie_action: Optional[Literal['get', 'set', 'delete', 'clear']] = Field(
        None, description="Cookie management action"
    )
    cookie_name: Optional[str] = Field(None, description="Cookie name for get/set/delete operations")
    cookie_value: Optional[str] = Field(None, description="Cookie value for set operation")
    cookie_options: Optional[Dict[str, Any]] = Field(None, description="Cookie options like domain, path, secure, httpOnly")
    
    # Storage management parameters
    storage_type: Optional[Literal['local', 'session']] = Field(
        None, description="Storage type for manage_storage operation"
    )
    storage_action: Optional[Literal['get', 'set', 'remove', 'clear']] = Field(
        None, description="Storage management action"
    )
    storage_key: Optional[str] = Field(None, description="Storage key for get/set/remove operations")
    storage_value: Optional[str] = Field(None, description="Storage value for set operation")
    
    # Evaluation parameters
    script: Optional[str] = Field(None, description="JavaScript code to evaluate")
    extract_content: Optional[bool] = Field(None, description="Whether to parse content with BeautifulSoup for get_content operation")
    
    @field_validator('url')
    @classmethod
    def validate_url_for_navigate(cls, v, info):
        """Validate URL is provided for navigate operation."""
        operation = info.data.get('operation')
        if operation == 'navigate' and not v:
            raise ValueError("url is required for navigate operation")
        return v
    
    @field_validator('screenshot_path')
    @classmethod
    def validate_screenshot_path(cls, v, info):
        """Validate screenshot_path for capture_screenshot operation."""
        operation = info.data.get('operation')
        if operation == 'capture_screenshot' and not v:
            raise ValueError("screenshot_path is required for capture_screenshot operation")
        return v
    
    @field_validator('cookie_action')
    @classmethod
    def validate_cookie_action(cls, v, info):
        """Validate cookie_action for manage_cookies operation."""
        operation = info.data.get('operation')
        if operation == 'manage_cookies' and not v:
            raise ValueError("cookie_action is required for manage_cookies operation")
        return v
    
    @field_validator('storage_type', 'storage_action')
    @classmethod
    def validate_storage_fields(cls, v, info):
        """Validate storage fields for manage_storage operation."""
        operation = info.data.get('operation')
        field_name = info.field_name
        if operation == 'manage_storage' and v is None:
            raise ValueError(f"{field_name} is required for manage_storage operation")
        return v
    
    @field_validator('script')
    @classmethod
    def validate_script(cls, v, info):
        """Validate script for evaluate operation."""
        operation = info.data.get('operation')
        if operation == 'evaluate' and not v:
            raise ValueError("script is required for evaluate operation")
        return v


class PageNavigatorOutput(BaseModel):
    """Output schema for page navigator operations."""
    success: bool = Field(description="Whether the operation succeeded")
    message: str = Field(description="Human-readable result message")
    data: Optional[Dict[str, Any]] = Field(None, description="Operation-specific return data")


async def _get_browser_page(browser_id: str):
    """Get browser instance and page from browser_manager."""
    # Import here to avoid circular dependency
    from agentoolkit.playwright.browser_manager import _browsers
    
    # Get browser instance directly from browser_manager's storage
    browser_instance = _browsers.get(browser_id)
    if not browser_instance:
        raise ValueError(f"Browser not found: {browser_id}")
    
    browser = browser_instance['browser']
    page = browser_instance.get('page')
    
    if not page:
        # Create a new page if none exists
        context = browser_instance.get('context')
        if not context:
            context = await browser.new_context()
            browser_instance['context'] = context
        page = await context.new_page()
        browser_instance['page'] = page
    
    return browser, page


async def page_navigator_navigate(
    ctx: RunContext[Any],
    browser_id: str,
    url: str,
    wait_condition: Optional[str] = None,
    timeout: Optional[float] = None
) -> PageNavigatorOutput:
    """
    Navigate to a URL with optional wait conditions.
    
    Args:
        ctx: Runtime context provided by the framework
        browser_id: Browser session identifier
        url: URL to navigate to
        wait_condition: Wait condition after navigation ('load', 'domcontentloaded', 'networkidle')
        timeout: Navigation timeout in milliseconds
        
    Returns:
        PageNavigatorOutput with navigation results including URL, title, and load time
        
    Raises:
        ValueError: If browser_id is invalid
        RuntimeError: If navigation fails
    """
    injector = get_injector()
    start_time = time.time()
    
    try:
        browser, page = await _get_browser_page(browser_id)
        
        # Set timeout if provided
        if timeout:
            page.set_default_timeout(timeout)
        
        # Navigate to URL
        await page.goto(url, wait_until=wait_condition or 'load')
        
        # Get page information
        title = await page.title()
        current_url = page.url
        load_time = int((time.time() - start_time) * 1000)
        
        # Store page state in storage_kv
        page_state = {
            'url': current_url,
            'title': title,
            'last_navigation': datetime.now(timezone.utc).isoformat(),
            'load_time': load_time
        }
        
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': f'page_state',
            'value': page_state,
            'namespace': f'browser:{browser_id}',
            'ttl': 3600
        })
        
        # Log navigation
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'page_navigator',
            'message': f'Successfully navigated to {url}',
            'data': {'browser_id': browser_id, 'url': url, 'load_time': load_time}
        })
        
        # Track metrics
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.page_navigator.navigations.total',
            'labels': {'browser_id': browser_id}
        })
        
        await injector.run('metrics', {
            'operation': 'observe',
            'name': 'agentool.page_navigator.navigation.duration_ms',
            'value': load_time,
            'labels': {'browser_id': browser_id}
        })
        
        return PageNavigatorOutput(
            success=True,
            message=f"Successfully navigated to {url}",
            data={
                'url': current_url,
                'title': title,
                'load_time': load_time
            }
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'page_navigator',
            'message': f'Navigation failed: {str(e)}',
            'data': {'browser_id': browser_id, 'url': url, 'error': str(e)}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.page_navigator.navigations.errors',
            'labels': {'browser_id': browser_id}
        })
        
        return PageNavigatorOutput(
            success=False,
            message=f"Navigation timeout after {timeout}ms" if "timeout" in str(e).lower() else f"Navigation failed: {str(e)}",
            data={
                'error': 'TimeoutError' if "timeout" in str(e).lower() else 'NavigationError',
                'url': url,
                'timeout': timeout
            }
        )


async def page_navigator_back(
    ctx: RunContext[Any],
    browser_id: str,
    wait_condition: Optional[str] = None,
    timeout: Optional[float] = None
) -> PageNavigatorOutput:
    """Navigate back in browser history."""
    injector = get_injector()
    
    try:
        browser, page = await _get_browser_page(browser_id)
        
        if timeout:
            page.set_default_timeout(timeout)
        
        await page.go_back(wait_until=wait_condition or 'load')
        
        current_url = page.url
        title = await page.title()
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'page_navigator',
            'message': 'Successfully navigated back',
            'data': {'browser_id': browser_id, 'url': current_url}
        })
        
        return PageNavigatorOutput(
            success=True,
            message="Successfully navigated back",
            data={'url': current_url, 'title': title}
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'page_navigator',
            'message': f'Back navigation failed: {str(e)}',
            'data': {'browser_id': browser_id, 'error': str(e)}
        })
        
        return PageNavigatorOutput(
            success=False,
            message=f"Back navigation failed: {str(e)}",
            data={'error': 'NavigationError'}
        )


async def page_navigator_forward(
    ctx: RunContext[Any],
    browser_id: str,
    wait_condition: Optional[str] = None,
    timeout: Optional[float] = None
) -> PageNavigatorOutput:
    """Navigate forward in browser history."""
    injector = get_injector()
    
    try:
        browser, page = await _get_browser_page(browser_id)
        
        if timeout:
            page.set_default_timeout(timeout)
        
        await page.go_forward(wait_until=wait_condition or 'load')
        
        current_url = page.url
        title = await page.title()
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'page_navigator',
            'message': 'Successfully navigated forward',
            'data': {'browser_id': browser_id, 'url': current_url}
        })
        
        return PageNavigatorOutput(
            success=True,
            message="Successfully navigated forward",
            data={'url': current_url, 'title': title}
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'page_navigator',
            'message': f'Forward navigation failed: {str(e)}',
            'data': {'browser_id': browser_id, 'error': str(e)}
        })
        
        return PageNavigatorOutput(
            success=False,
            message=f"Forward navigation failed: {str(e)}",
            data={'error': 'NavigationError'}
        )


async def page_navigator_refresh(
    ctx: RunContext[Any],
    browser_id: str,
    wait_condition: Optional[str] = None,
    timeout: Optional[float] = None
) -> PageNavigatorOutput:
    """Refresh the current page."""
    injector = get_injector()
    
    try:
        browser, page = await _get_browser_page(browser_id)
        
        if timeout:
            page.set_default_timeout(timeout)
        
        await page.reload(wait_until=wait_condition or 'load')
        
        current_url = page.url
        title = await page.title()
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'page_navigator',
            'message': 'Successfully refreshed page',
            'data': {'browser_id': browser_id, 'url': current_url}
        })
        
        return PageNavigatorOutput(
            success=True,
            message="Successfully refreshed page",
            data={'url': current_url, 'title': title}
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'page_navigator',
            'message': f'Page refresh failed: {str(e)}',
            'data': {'browser_id': browser_id, 'error': str(e)}
        })
        
        return PageNavigatorOutput(
            success=False,
            message=f"Page refresh failed: {str(e)}",
            data={'error': 'NavigationError'}
        )


async def page_navigator_wait_for_load(
    ctx: RunContext[Any],
    browser_id: str,
    wait_condition: Optional[str] = None,
    timeout: Optional[float] = None,
    selector: Optional[str] = None
) -> PageNavigatorOutput:
    """Wait for page load or specific elements."""
    injector = get_injector()
    start_time = time.time()
    
    try:
        browser, page = await _get_browser_page(browser_id)
        
        if timeout:
            page.set_default_timeout(timeout)
        
        if selector:
            # Wait for specific element
            await page.wait_for_selector(selector, timeout=timeout)
            wait_type = f"element '{selector}'"
        else:
            # Wait for load state
            condition = wait_condition or 'load'
            await page.wait_for_load_state(condition, timeout=timeout)
            wait_type = f"load state '{condition}'"
        
        wait_time = int((time.time() - start_time) * 1000)
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'page_navigator',
            'message': f'Successfully waited for {wait_type}',
            'data': {'browser_id': browser_id, 'wait_time': wait_time}
        })
        
        return PageNavigatorOutput(
            success=True,
            message=f"Successfully waited for {wait_type}",
            data={
                'wait_condition': wait_condition,
                'selector': selector,
                'wait_time_ms': wait_time
            }
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'page_navigator',
            'message': f'Wait operation failed: {str(e)}',
            'data': {'browser_id': browser_id, 'error': str(e)}
        })
        
        return PageNavigatorOutput(
            success=False,
            message=f"Wait operation timeout: {str(e)}",
            data={
                'error': 'TimeoutError',
                'wait_condition': wait_condition,
                'timeout': timeout
            }
        )


async def page_navigator_capture_screenshot(
    ctx: RunContext[Any],
    browser_id: str,
    screenshot_path: str,
    full_page: Optional[bool] = None
) -> PageNavigatorOutput:
    """Capture page screenshot and save to storage."""
    injector = get_injector()
    
    try:
        browser, page = await _get_browser_page(browser_id)
        
        # Capture screenshot
        screenshot_options = {'path': screenshot_path}
        if full_page is not None:
            screenshot_options['full_page'] = full_page
            
        screenshot_bytes = await page.screenshot(**screenshot_options)
        
        # Save to storage_fs (Playwright already saves to path, but we verify)
        fs_result = await injector.run('storage_fs', {
            'operation': 'exists',
            'path': screenshot_path
        })
        
        if not fs_result.success:
            raise RuntimeError(f"Screenshot was not saved to {screenshot_path}")
        
        # Get file size and dimensions (approximate)
        file_size = len(screenshot_bytes) if screenshot_bytes else 0
        viewport = await page.viewport_size()
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'page_navigator',
            'message': f'Screenshot captured successfully',
            'data': {'browser_id': browser_id, 'path': screenshot_path, 'size': file_size}
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.page_navigator.screenshots.total',
            'labels': {'browser_id': browser_id}
        })
        
        return PageNavigatorOutput(
            success=True,
            message="Screenshot captured successfully",
            data={
                'path': screenshot_path,
                'size_bytes': file_size,
                'dimensions': {
                    'width': viewport['width'] if viewport else 0,
                    'height': viewport['height'] if viewport else 0
                },
                'full_page': full_page or False
            }
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'page_navigator',
            'message': f'Screenshot capture failed: {str(e)}',
            'data': {'browser_id': browser_id, 'error': str(e)}
        })
        
        return PageNavigatorOutput(
            success=False,
            message=f"Screenshot capture failed: {str(e)}",
            data={'error': 'FileSystemError', 'path': screenshot_path}
        )


async def page_navigator_get_content(
    ctx: RunContext[Any],
    browser_id: str,
    extract_content: Optional[bool] = None
) -> PageNavigatorOutput:
    """Extract page content with optional BeautifulSoup parsing."""
    injector = get_injector()
    
    try:
        browser, page = await _get_browser_page(browser_id)
        
        # Get HTML content
        html = await page.content()
        title = await page.title()
        url = page.url
        
        data = {
            'html': html,
            'title': title,
            'url': url
        }
        
        # Parse with BeautifulSoup if requested and available
        if extract_content and BeautifulSoup:
            try:
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract text content
                data['text_content'] = soup.get_text(separator=' ', strip=True)
                
                # Extract links
                links = [a.get('href') for a in soup.find_all('a', href=True)]
                data['links'] = [link for link in links if link.startswith(('http', 'https', '/'))]
                
                # Extract forms
                forms = []
                for form in soup.find_all('form'):
                    forms.append({
                        'action': form.get('action', ''),
                        'method': form.get('method', 'GET').upper()
                    })
                data['forms'] = forms
                
                # Extract headings
                headings = []
                for level in range(1, 7):
                    for heading in soup.find_all(f'h{level}'):
                        headings.append({
                            'level': level,
                            'text': heading.get_text(strip=True)
                        })
                data['headings'] = headings
                
            except Exception as parse_error:
                await injector.run('logging', {
                    'operation': 'log',
                    'level': 'WARN',
                    'logger_name': 'page_navigator',
                    'message': f'BeautifulSoup parsing failed: {str(parse_error)}',
                    'data': {'browser_id': browser_id}
                })
        
        elif extract_content and not BeautifulSoup:
            await injector.run('logging', {
                'operation': 'log',
                'level': 'WARN',
                'logger_name': 'page_navigator',
                'message': 'BeautifulSoup not available for content extraction',
                'data': {'browser_id': browser_id}
            })
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'page_navigator',
            'message': 'Page content extracted successfully',
            'data': {'browser_id': browser_id, 'content_length': len(html)}
        })
        
        return PageNavigatorOutput(
            success=True,
            message="Page content extracted successfully",
            data=data
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'page_navigator',
            'message': f'Content extraction failed: {str(e)}',
            'data': {'browser_id': browser_id, 'error': str(e)}
        })
        
        return PageNavigatorOutput(
            success=False,
            message=f"Content extraction failed: {str(e)}",
            data={'error': 'NavigationError'}
        )


async def page_navigator_manage_cookies(
    ctx: RunContext[Any],
    browser_id: str,
    cookie_action: str,
    cookie_name: Optional[str] = None,
    cookie_value: Optional[str] = None,
    cookie_options: Optional[Dict[str, Any]] = None
) -> PageNavigatorOutput:
    """Manage browser cookies."""
    injector = get_injector()
    
    try:
        browser, page = await _get_browser_page(browser_id)
        context = page.context
        
        if cookie_action == 'get':
            if cookie_name:
                cookies = await context.cookies()
                target_cookie = next((c for c in cookies if c['name'] == cookie_name), None)
                if target_cookie:
                    data = target_cookie
                    message = f"Retrieved cookie '{cookie_name}'"
                else:
                    data = None
                    message = f"Cookie '{cookie_name}' not found"
            else:
                cookies = await context.cookies()
                data = {'cookies': cookies, 'count': len(cookies)}
                message = f"Retrieved {len(cookies)} cookies"
                
        elif cookie_action == 'set':
            if not cookie_name or cookie_value is None:
                raise ValueError("cookie_name and cookie_value are required for set action")
                
            cookie_data = {
                'name': cookie_name,
                'value': cookie_value,
                'url': page.url
            }
            
            if cookie_options:
                cookie_data.update(cookie_options)
                
            await context.add_cookies([cookie_data])
            data = {'name': cookie_name, 'value': cookie_value}
            if cookie_options:
                data.update(cookie_options)
            message = f"Cookie '{cookie_name}' set successfully"
            
        elif cookie_action == 'delete':
            if not cookie_name:
                raise ValueError("cookie_name is required for delete action")
                
            # Playwright doesn't have direct delete, so we clear by setting expired cookie
            expired_cookie = {
                'name': cookie_name,
                'value': '',
                'url': page.url,
                'expires': 0
            }
            await context.add_cookies([expired_cookie])
            data = {'deleted': cookie_name}
            message = f"Cookie '{cookie_name}' deleted successfully"
            
        elif cookie_action == 'clear':
            await context.clear_cookies()
            data = {'cleared': True}
            message = "All cookies cleared successfully"
            
        else:
            raise ValueError(f"Invalid cookie action: {cookie_action}")
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'page_navigator',
            'message': f'Cookie {cookie_action} operation successful',
            'data': {'browser_id': browser_id, 'action': cookie_action}
        })
        
        return PageNavigatorOutput(
            success=True,
            message=message,
            data=data
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'page_navigator',
            'message': f'Cookie {cookie_action} operation failed: {str(e)}',
            'data': {'browser_id': browser_id, 'error': str(e)}
        })
        
        return PageNavigatorOutput(
            success=False,
            message=f"Cookie {cookie_action} operation failed: {str(e)}",
            data={'error': 'CookieError', 'action': cookie_action}
        )


async def page_navigator_manage_storage(
    ctx: RunContext[Any],
    browser_id: str,
    storage_type: str,
    storage_action: str,
    storage_key: Optional[str] = None,
    storage_value: Optional[str] = None
) -> PageNavigatorOutput:
    """Manage browser local and session storage."""
    injector = get_injector()
    
    try:
        browser, page = await _get_browser_page(browser_id)
        
        # Build JavaScript for storage operations
        storage_obj = 'localStorage' if storage_type == 'local' else 'sessionStorage'
        
        if storage_action == 'get':
            if storage_key:
                script = f"{storage_obj}.getItem('{storage_key}')"
                result = await page.evaluate(script)
                data = {'key': storage_key, 'value': result}
                message = f"Retrieved {storage_type} storage item '{storage_key}'"
            else:
                # Get all keys and values
                script = f"""
                (() => {{
                    const items = {{}};
                    for (let i = 0; i < {storage_obj}.length; i++) {{
                        const key = {storage_obj}.key(i);
                        items[key] = {storage_obj}.getItem(key);
                    }}
                    return items;
                }})()
                """
                items = await page.evaluate(script)
                data = {'items': items, 'count': len(items)}
                message = f"Retrieved {len(items)} {storage_type} storage items"
                
        elif storage_action == 'set':
            if not storage_key or storage_value is None:
                raise ValueError("storage_key and storage_value are required for set action")
                
            script = f"{storage_obj}.setItem('{storage_key}', '{storage_value}')"
            await page.evaluate(script)
            data = {'key': storage_key, 'value': storage_value}
            message = f"Set {storage_type} storage item '{storage_key}'"
            
        elif storage_action == 'remove':
            if not storage_key:
                raise ValueError("storage_key is required for remove action")
                
            script = f"{storage_obj}.removeItem('{storage_key}')"
            await page.evaluate(script)
            data = {'removed': storage_key}
            message = f"Removed {storage_type} storage item '{storage_key}'"
            
        elif storage_action == 'clear':
            script = f"{storage_obj}.clear()"
            await page.evaluate(script)
            data = {'cleared': True}
            message = f"Cleared all {storage_type} storage items"
            
        else:
            raise ValueError(f"Invalid storage action: {storage_action}")
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'page_navigator',
            'message': f'Storage {storage_action} operation successful',
            'data': {'browser_id': browser_id, 'type': storage_type, 'action': storage_action}
        })
        
        return PageNavigatorOutput(
            success=True,
            message=message,
            data=data
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'page_navigator',
            'message': f'Storage {storage_action} operation failed: {str(e)}',
            'data': {'browser_id': browser_id, 'error': str(e)}
        })
        
        return PageNavigatorOutput(
            success=False,
            message=f"Storage {storage_action} operation failed: {str(e)}",
            data={'error': 'StorageError', 'type': storage_type, 'action': storage_action}
        )


async def page_navigator_evaluate(
    ctx: RunContext[Any],
    browser_id: str,
    script: str
) -> PageNavigatorOutput:
    """Evaluate JavaScript code in the page context."""
    injector = get_injector()
    
    try:
        browser, page = await _get_browser_page(browser_id)
        
        # Execute JavaScript
        result = await page.evaluate(script)
        
        # Handle different result types
        if result is None:
            data = {'result': None, 'type': 'null'}
        elif isinstance(result, (bool, int, float, str)):
            data = {'result': result, 'type': type(result).__name__}
        else:
            # Convert complex objects to string representation
            data = {'result': str(result), 'type': 'object'}
        
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'page_navigator',
            'message': 'JavaScript evaluation successful',
            'data': {'browser_id': browser_id, 'script_length': len(script)}
        })
        
        return PageNavigatorOutput(
            success=True,
            message="JavaScript evaluation completed successfully",
            data=data
        )
        
    except Exception as e:
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'page_navigator',
            'message': f'JavaScript evaluation failed: {str(e)}',
            'data': {'browser_id': browser_id, 'error': str(e)}
        })
        
        return PageNavigatorOutput(
            success=False,
            message=f"JavaScript evaluation failed: {str(e)}",
            data={'error': 'JavaScriptError', 'script': script[:100]}  # Truncate script in error
        )


# Routing configuration
page_navigator_routing = RoutingConfig(
    operation_field='operation',
    operation_map={
        'navigate': ('page_navigator_navigate', lambda x: {
            'browser_id': x.browser_id,
            'url': x.url,
            'wait_condition': x.wait_condition,
            'timeout': x.timeout,
        }),
        'back': ('page_navigator_back', lambda x: {
            'browser_id': x.browser_id,
            'wait_condition': x.wait_condition,
            'timeout': x.timeout,
        }),
        'forward': ('page_navigator_forward', lambda x: {
            'browser_id': x.browser_id,
            'wait_condition': x.wait_condition,
            'timeout': x.timeout,
        }),
        'refresh': ('page_navigator_refresh', lambda x: {
            'browser_id': x.browser_id,
            'wait_condition': x.wait_condition,
            'timeout': x.timeout,
        }),
        'wait_for_load': ('page_navigator_wait_for_load', lambda x: {
            'browser_id': x.browser_id,
            'wait_condition': x.wait_condition,
            'timeout': x.timeout,
            'selector': x.selector,
        }),
        'capture_screenshot': ('page_navigator_capture_screenshot', lambda x: {
            'browser_id': x.browser_id,
            'screenshot_path': x.screenshot_path,
            'full_page': x.full_page,
        }),
        'get_content': ('page_navigator_get_content', lambda x: {
            'browser_id': x.browser_id,
            'extract_content': x.extract_content,
        }),
        'manage_cookies': ('page_navigator_manage_cookies', lambda x: {
            'browser_id': x.browser_id,
            'cookie_action': x.cookie_action,
            'cookie_name': x.cookie_name,
            'cookie_value': x.cookie_value,
            'cookie_options': x.cookie_options,
        }),
        'manage_storage': ('page_navigator_manage_storage', lambda x: {
            'browser_id': x.browser_id,
            'storage_type': x.storage_type,
            'storage_action': x.storage_action,
            'storage_key': x.storage_key,
            'storage_value': x.storage_value,
        }),
        'evaluate': ('page_navigator_evaluate', lambda x: {
            'browser_id': x.browser_id,
            'script': x.script,
        }),
    }
)


def create_page_navigator_agent():
    """
    Create and return the page_navigator AgenTool.
    
    Returns:
        Agent configured for page navigation operations
    """
    return create_agentool(
        name='page_navigator',
        input_schema=PageNavigatorInput,
        routing_config=page_navigator_routing,
        tools=[
            page_navigator_navigate, page_navigator_back, page_navigator_forward,
            page_navigator_refresh, page_navigator_wait_for_load,
            page_navigator_capture_screenshot, page_navigator_get_content,
            page_navigator_manage_cookies, page_navigator_manage_storage,
            page_navigator_evaluate
        ],
        output_type=PageNavigatorOutput,
        system_prompt="You are a browser page navigation assistant. You handle URL navigation, content extraction, screenshot capture, cookie management, storage operations, and JavaScript evaluation with comprehensive error handling and state tracking.",
        description="Handles page navigation, URL management, and high-level browser interactions with operations: navigate, back, forward, refresh, wait_for_load, capture_screenshot, get_content, manage_cookies, manage_storage, evaluate",
        version="1.0.0",
        tags=["browser", "navigation", "playwright", "automation", "content"],
        dependencies=["browser_manager", "storage_fs", "storage_kv", "logging", "metrics"],
        examples=[
            {
                "description": "Navigate to a URL and wait for page load",
                "input": {
                    "operation": "navigate",
                    "browser_id": "browser_123",
                    "url": "https://example.com",
                    "wait_condition": "load",
                    "timeout": 30000
                },
                "output": {
                    "success": True,
                    "message": "Successfully navigated to https://example.com",
                    "data": {
                        "url": "https://example.com",
                        "title": "Example Domain",
                        "load_time": 1250
                    }
                }
            },
            {
                "description": "Capture full page screenshot",
                "input": {
                    "operation": "capture_screenshot",
                    "browser_id": "browser_123",
                    "screenshot_path": "screenshots/page.png",
                    "full_page": True
                },
                "output": {
                    "success": True,
                    "message": "Screenshot captured successfully",
                    "data": {
                        "path": "screenshots/page.png",
                        "size_bytes": 45678,
                        "dimensions": {"width": 1920, "height": 1080}
                    }
                }
            },
            {
                "description": "Get page content with BeautifulSoup parsing",
                "input": {
                    "operation": "get_content",
                    "browser_id": "browser_123",
                    "extract_content": True
                },
                "output": {
                    "success": True,
                    "message": "Page content extracted successfully",
                    "data": {
                        "html": "<html><body>...</body></html>",
                        "title": "Example Page",
                        "text_content": "Page text content...",
                        "links": ["https://example.com/link1"],
                        "forms": [{"action": "/submit", "method": "POST"}]
                    }
                }
            }
        ]
    )


# Create the agent instance
agent = create_page_navigator_agent()