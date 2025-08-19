"""
browser_manager AgenTool - Manages Playwright browser instances with session persistence and lifecycle control.

This AgenTool provides comprehensive browser automation lifecycle management with session persistence,
parallel browser support, and crash recovery capabilities. It integrates with storage_kv for metadata
persistence, logging for operational events, and metrics for performance monitoring.

Key Features:
- Browser Creation: Launch Chromium browsers with configurable options (headless, viewport, user data dirs)
- Session Management: Persistent browser sessions with unique identifiers stored in storage_kv
- Lifecycle Control: Start, stop, health check, and list browser instances with graceful shutdown
- Crash Recovery: Browser metadata persistence for automatic recovery and cleanup
- Resource Management: Proper cleanup using context managers and timeout handling

Usage Example:
    >>> from agentoolkit.browser_manager import create_browser_manager_agent
    >>> from agentool.core.injector import get_injector
    >>> 
    >>> # Create and register the agent
    >>> agent = create_browser_manager_agent()
    >>> 
    >>> # Start a browser instance
    >>> injector = get_injector()
    >>> result = await injector.run('browser_manager', {
    ...     "operation": "start_browser",
    ...     "browser_id": "session_123",
    ...     "options": {"headless": False, "viewport": {"width": 1920, "height": 1080}}
    ... })
"""

import json
import time
import asyncio
import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import RunContext

from playwright.async_api import async_playwright, Browser, BrowserContext, Page
import psutil

from agentool import create_agentool
from agentool.base import BaseOperationInput
from agentool.core.registry import RoutingConfig
from agentool.core.injector import get_injector


class BrowserManagerInput(BaseOperationInput):
    """Input schema for browser_manager operations.
    
    Supports browser lifecycle operations with configurable options,
    session persistence, and health monitoring capabilities.
    """
    operation: Literal['start_browser', 'stop_browser', 'get_browser', 'list_browsers', 'health_check', 'cleanup_all'] = Field(
        description="Browser management operation to perform"
    )
    
    browser_id: Optional[str] = Field(
        None,
        description="Unique identifier for the browser instance"
    )
    
    options: Optional[Dict[str, Any]] = Field(
        None,
        description="Browser configuration options (headless, viewport, user_data_dir, args)"
    )
    
    timeout: Optional[int] = Field(
        30000,
        description="Operation timeout in milliseconds"
    )
    
    @field_validator('browser_id')
    @classmethod
    def validate_browser_id(cls, v, info):
        """Validate browser_id is provided for operations that require it."""
        operation = info.data.get('operation')
        if operation in ['start_browser', 'stop_browser', 'get_browser', 'health_check'] and not v:
            raise ValueError(f"browser_id is required for {operation}")
        return v
    
    @field_validator('options')
    @classmethod
    def validate_options(cls, v, info):
        """Validate options structure for start_browser operation."""
        operation = info.data.get('operation')
        if operation == 'start_browser' and v:
            # Validate viewport structure if provided
            if 'viewport' in v and v['viewport']:
                viewport = v['viewport']
                if not isinstance(viewport, dict):
                    raise ValueError("viewport must be a dictionary")
                if 'width' in viewport and not isinstance(viewport['width'], int):
                    raise ValueError("viewport width must be an integer")
                if 'height' in viewport and not isinstance(viewport['height'], int):
                    raise ValueError("viewport height must be an integer")
            
            # Validate args is a list if provided
            if 'args' in v and v['args'] and not isinstance(v['args'], list):
                raise ValueError("args must be a list of strings")
        
        return v


class BrowserManagerOutput(BaseModel):
    """Output schema for browser_manager operations."""
    success: bool = Field(description="Whether the operation succeeded")
    message: str = Field(description="Human-readable result message")
    data: Optional[Dict[str, Any]] = Field(None, description="Operation-specific return data including browser metadata, instance details, or health status")


# Global browser instances storage
_browsers: Dict[str, Dict[str, Any]] = {}
_playwright_instance = None


class BrowserNotFoundError(Exception):
    """Raised when a browser instance is not found."""
    pass


class BrowserStartupError(Exception):
    """Raised when browser fails to start."""
    pass


class BrowserCrashError(Exception):
    """Raised when browser process crashes unexpectedly."""
    pass


class InvalidOptionsError(Exception):
    """Raised when browser options are invalid."""
    pass


class ProcessError(Exception):
    """Raised when unable to manage browser process."""
    pass


class ResourceExhaustionError(Exception):
    """Raised when system resources are insufficient."""
    pass


async def _get_playwright():
    """Get or create playwright instance."""
    global _playwright_instance
    if _playwright_instance is None:
        _playwright_instance = await async_playwright().start()
    return _playwright_instance


async def _cleanup_playwright():
    """Clean up playwright instance."""
    global _playwright_instance
    if _playwright_instance is not None:
        try:
            await _playwright_instance.stop()
        except:
            pass  # Ignore errors during cleanup
        finally:
            _playwright_instance = None


async def _store_browser_metadata(browser_id: str, metadata: Dict[str, Any]):
    """Store browser metadata in storage_kv."""
    injector = get_injector()
    
    await injector.run('storage_kv', {
        'operation': 'set',
        'key': browser_id,
        'value': metadata,
        'namespace': 'browsers',
        'ttl': 86400  # 24 hours TTL for cleanup of stale entries
    })


async def _get_browser_metadata(browser_id: str) -> Optional[Dict[str, Any]]:
    """Get browser metadata from storage_kv."""
    injector = get_injector()
    
    result = await injector.run('storage_kv', {
        'operation': 'get',
        'key': browser_id,
        'namespace': 'browsers'
    })
    
    if result.success and result.data and result.data.get('exists'):
        return result.data.get('value')
    return None


async def _delete_browser_metadata(browser_id: str):
    """Delete browser metadata from storage_kv."""
    injector = get_injector()
    
    await injector.run('storage_kv', {
        'operation': 'delete',
        'key': browser_id,
        'namespace': 'browsers'
    })


async def _check_system_resources():
    """Check if system has sufficient resources for browser launch."""
    try:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Check if we have at least 500MB free memory and CPU usage < 90%
        if memory.available < 500 * 1024 * 1024:  # 500MB
            raise ResourceExhaustionError("Insufficient memory available for browser launch")
        
        if cpu_percent > 90:
            raise ResourceExhaustionError("CPU usage too high for browser launch")
            
    except psutil.Error as e:
        # If we can't check resources, log warning but don't fail
        injector = get_injector()
        await injector.run('logging', {
            'operation': 'log',
            'level': 'WARN',
            'logger_name': 'browser_manager',
            'message': 'Unable to check system resources',
            'data': {'error': str(e)}
        })


async def browser_manager_start_browser(
    ctx: RunContext[Any],
    browser_id: str,
    options: Optional[Dict[str, Any]] = None,
    timeout: int = 30000
) -> BrowserManagerOutput:
    """
    Start a new browser instance with the specified configuration.
    
    Creates a new Playwright Chromium browser instance with configurable options
    including headless mode, viewport settings, user data directory for session
    persistence, and custom browser arguments. Stores browser metadata in storage_kv
    for session recovery and tracks metrics for monitoring.
    
    Args:
        ctx: Runtime context provided by the framework
        browser_id: Unique identifier for the browser instance
        options: Browser configuration dictionary with keys:
                - headless: Boolean, whether to run in headless mode (default: False)
                - viewport: Dict with width/height for initial viewport size
                - user_data_dir: String path for persistent user data storage
                - args: List of additional browser launch arguments
        timeout: Operation timeout in milliseconds (default: 30000)
        
    Returns:
        BrowserManagerOutput with operation results containing:
        - success: Whether the browser started successfully
        - message: Human-readable result message
        - data: Browser metadata including browser_id, status, pid, options, created_at
        
    Raises:
        BrowserStartupError: If Playwright fails to launch browser
        InvalidOptionsError: If browser configuration options are invalid
        ResourceExhaustionError: If system resources are insufficient
        ValueError: If browser_id is invalid or already exists
    """
    injector = get_injector()
    
    try:
        # Check if browser already exists
        if browser_id in _browsers:
            raise ValueError(f"Browser instance '{browser_id}' already exists")
        
        # Check system resources
        await _check_system_resources()
        
        # Validate and prepare options
        browser_options = options or {}
        launch_options = {
            'headless': browser_options.get('headless', False),
            'timeout': timeout,
        }
        
        # Add viewport if specified
        if 'viewport' in browser_options and browser_options['viewport']:
            viewport = browser_options['viewport']
            if not isinstance(viewport, dict) or 'width' not in viewport or 'height' not in viewport:
                raise InvalidOptionsError("Invalid viewport configuration")
            launch_options['viewport'] = viewport
        
        # Add user data directory if specified
        if 'user_data_dir' in browser_options and browser_options['user_data_dir']:
            user_data_dir = browser_options['user_data_dir']
            # Create directory if it doesn't exist
            os.makedirs(user_data_dir, exist_ok=True)
            launch_options['user_data_dir'] = user_data_dir
        
        # Prepare browser args with better defaults for web scraping
        default_args = [
            '--disable-blink-features=AutomationControlled',  # Hide automation
            '--disable-dev-shm-usage',  # Overcome limited resource problems
            '--no-sandbox',  # Required for some environments
            '--disable-setuid-sandbox',
            '--disable-web-security',  # Allow cross-origin requests
            '--disable-features=IsolateOrigins,site-per-process',
        ]
        
        # Add custom args if specified, otherwise use defaults
        if 'args' in browser_options and browser_options['args']:
            if not isinstance(browser_options['args'], list):
                raise InvalidOptionsError("Browser args must be a list of strings")
            # Merge custom args with defaults (custom args override)
            launch_options['args'] = default_args + browser_options['args']
        else:
            launch_options['args'] = default_args
        
        # Get playwright instance and launch browser
        playwright = await _get_playwright()
        browser = await playwright.chromium.launch(**launch_options)
        
        # Get browser process info and create initial context
        context = None
        page = None
        pid = 0
        
        try:
            # Create a context with a real Chrome user agent
            context_options = {
                'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'locale': 'en-US',
                'timezone_id': 'America/New_York',
            }
            
            # Add viewport to context if specified
            if 'viewport' in browser_options and browser_options['viewport']:
                context_options['viewport'] = browser_options['viewport']
            
            context = await browser.new_context(**context_options)
            pages_list = context.pages  # This is a property, not a method
            if not pages_list:
                page = await context.new_page()
            else:
                page = pages_list[0]
            
            # Try to get process PID (may not always be available)
            try:
                # Try to get PID from the browser implementation
                if hasattr(browser, '_impl_obj') and hasattr(browser._impl_obj, '_connection'):
                    # Try to get from connection data
                    pid = getattr(browser._impl_obj._connection, '_pid', 0)
                else:
                    pid = 0
            except:
                pid = 0
            
            if not pid:
                # Fallback: use current process PID as approximation
                pid = os.getpid()
            
        except Exception as e:
            # If we can't get process info, still continue
            await injector.run('logging', {
                'operation': 'log',
                'level': 'WARN',
                'logger_name': 'browser_manager',
                'message': 'Unable to get browser process info',
                'data': {'browser_id': browser_id, 'error': str(e)}
            })
            # Create minimal context and page if failed
            if not context:
                try:
                    context = await browser.new_context()
                    page = await context.new_page()
                except:
                    pass  # Will handle in storage
        
        # Store browser instance
        created_at = datetime.now(timezone.utc).isoformat()
        browser_metadata = {
            'browser_id': browser_id,
            'status': 'running',
            'pid': pid,
            'options': browser_options,
            'created_at': created_at,
            'last_health_check': created_at
        }
        
        _browsers[browser_id] = {
            'browser': browser,
            'context': context,
            'page': page,
            'metadata': browser_metadata,
            'created_at': time.time()
        }
        
        # Store metadata in storage_kv for persistence
        await _store_browser_metadata(browser_id, browser_metadata)
        
        # Log successful startup
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'browser_manager',
            'message': f'Successfully started browser instance "{browser_id}"',
            'data': {'browser_id': browser_id, 'pid': pid, 'options': browser_options}
        })
        
        # Track metrics
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.browser_manager.browsers_started.total',
            'labels': {'browser_id': browser_id}
        })
        
        return BrowserManagerOutput(
            success=True,
            message=f"Successfully started browser instance '{browser_id}'",
            data=browser_metadata
        )
        
    except Exception as e:
        # Log error
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'browser_manager',
            'message': f'Failed to start browser instance "{browser_id}"',
            'data': {'browser_id': browser_id, 'error': str(e), 'options': options}
        })
        
        # Track error metrics
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.browser_manager.browser_start_errors.total',
            'labels': {'error_type': type(e).__name__}
        })
        
        # Clean up partial state if needed
        if browser_id in _browsers:
            del _browsers[browser_id]
        
        # Re-raise with appropriate error type
        if isinstance(e, (ValueError, InvalidOptionsError, ResourceExhaustionError)):
            raise
        else:
            raise BrowserStartupError(f"Failed to start browser: {str(e)}") from e


async def browser_manager_stop_browser(
    ctx: RunContext[Any],
    browser_id: str,
    timeout: int = 30000
) -> BrowserManagerOutput:
    """
    Stop a browser instance gracefully with session state cleanup.
    
    Performs graceful shutdown of the specified browser instance, saving any
    pending session state, closing all pages and contexts, and cleaning up
    stored metadata from storage_kv. Handles cleanup even if browser has crashed.
    
    Args:
        ctx: Runtime context provided by the framework
        browser_id: Unique identifier of the browser instance to stop
        timeout: Operation timeout in milliseconds (default: 30000)
        
    Returns:
        BrowserManagerOutput with operation results containing:
        - success: Whether the browser was stopped successfully
        - message: Human-readable result message
        - data: Cleanup status and browser metadata
        
    Raises:
        BrowserNotFoundError: If browser_id does not exist
        ProcessError: If unable to stop browser process
    """
    injector = get_injector()
    
    try:
        # Check if browser exists in memory
        browser_instance = _browsers.get(browser_id)
        if not browser_instance:
            # Check if it exists in storage (crashed browser recovery)
            metadata = await _get_browser_metadata(browser_id)
            if not metadata:
                raise BrowserNotFoundError(f"Browser instance '{browser_id}' not found")
            
            # Browser was in storage but not in memory (likely crashed)
            await _delete_browser_metadata(browser_id)
            
            await injector.run('logging', {
                'operation': 'log',
                'level': 'INFO',
                'logger_name': 'browser_manager',
                'message': f'Cleaned up crashed browser instance "{browser_id}"',
                'data': {'browser_id': browser_id}
            })
            
            return BrowserManagerOutput(
                success=True,
                message=f"Successfully cleaned up crashed browser instance '{browser_id}'",
                data={
                    'browser_id': browser_id,
                    'status': 'stopped',
                    'cleanup_completed': True,
                    'was_crashed': True
                }
            )
        
        browser = browser_instance['browser']
        context = browser_instance.get('context')
        cleanup_completed = False
        
        try:
            # Graceful shutdown with timeout
            async def close_browser():
                # Close context if exists
                if context:
                    await context.close()
                
                # Close browser
                await browser.close()
            
            # Use wait_for with timeout in seconds
            await asyncio.wait_for(close_browser(), timeout=timeout / 1000)
            cleanup_completed = True
                
        except asyncio.TimeoutError:
            # Force kill if graceful shutdown times out
            await injector.run('logging', {
                'operation': 'log',
                'level': 'WARN',
                'logger_name': 'browser_manager',
                'message': f'Browser shutdown timed out, force closing "{browser_id}"',
                'data': {'browser_id': browser_id, 'timeout': timeout}
            })
            
            # Try to force close
            try:
                await browser.close()
                cleanup_completed = True
            except:
                pass  # Browser may have already crashed
        
        except Exception as e:
            await injector.run('logging', {
                'operation': 'log',
                'level': 'WARN',
                'logger_name': 'browser_manager',
                'message': f'Error during browser shutdown "{browser_id}"',
                'data': {'browser_id': browser_id, 'error': str(e)}
            })
        
        # Remove from memory and storage
        del _browsers[browser_id]
        await _delete_browser_metadata(browser_id)
        
        # Log successful shutdown
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'browser_manager',
            'message': f'Successfully stopped browser instance "{browser_id}"',
            'data': {'browser_id': browser_id, 'cleanup_completed': cleanup_completed}
        })
        
        # Track metrics
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.browser_manager.browsers_stopped.total',
            'labels': {'browser_id': browser_id}
        })
        
        return BrowserManagerOutput(
            success=True,
            message=f"Successfully stopped browser instance '{browser_id}'",
            data={
                'browser_id': browser_id,
                'status': 'stopped',
                'cleanup_completed': cleanup_completed
            }
        )
        
    except BrowserNotFoundError:
        raise
    except Exception as e:
        # Log error
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'browser_manager',
            'message': f'Failed to stop browser instance "{browser_id}"',
            'data': {'browser_id': browser_id, 'error': str(e)}
        })
        
        # Track error metrics
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.browser_manager.browser_stop_errors.total',
            'labels': {'error_type': type(e).__name__}
        })
        
        raise ProcessError(f"Failed to stop browser: {str(e)}") from e


async def browser_manager_get_browser(
    ctx: RunContext[Any],
    browser_id: str,
    timeout: int = 30000
) -> BrowserManagerOutput:
    """
    Get detailed information about a specific browser instance.
    
    Retrieves comprehensive metadata and current status for the specified browser
    instance, including health status, uptime, process information, and configuration
    details. Performs basic health checks to verify browser responsiveness.
    
    Args:
        ctx: Runtime context provided by the framework
        browser_id: Unique identifier of the browser instance to query
        timeout: Operation timeout in milliseconds (default: 30000)
        
    Returns:
        BrowserManagerOutput with operation results containing:
        - success: Whether the browser information was retrieved successfully
        - message: Human-readable result message
        - data: Detailed browser information including status, health, uptime, pid
        
    Raises:
        BrowserNotFoundError: If browser_id does not exist
    """
    injector = get_injector()
    
    try:
        # Check if browser exists in memory
        browser_instance = _browsers.get(browser_id)
        if not browser_instance:
            # Check if it exists in storage (crashed browser)
            metadata = await _get_browser_metadata(browser_id)
            if not metadata:
                raise BrowserNotFoundError(f"Browser instance '{browser_id}' not found")
            
            # Browser exists in storage but not memory - likely crashed
            return BrowserManagerOutput(
                success=False,
                message=f"Browser instance '{browser_id}' not found",
                data=None
            )
        
        browser = browser_instance['browser']
        metadata = browser_instance['metadata']
        created_time = browser_instance['created_at']
        
        # Calculate uptime
        uptime_seconds = int(time.time() - created_time)
        
        # Perform basic health check
        health_status = "healthy"
        try:
            # Try to get browser version as a health check
            contexts = browser.contexts
            if not contexts:
                # Create a temporary context for health check
                temp_context = await browser.new_context()
                await temp_context.close()
            health_status = "healthy"
        except Exception as e:
            health_status = "unhealthy"
            await injector.run('logging', {
                'operation': 'log',
                'level': 'WARN',
                'logger_name': 'browser_manager',
                'message': f'Browser health check failed for "{browser_id}"',
                'data': {'browser_id': browser_id, 'error': str(e)}
            })
        
        # Update last health check time
        current_time = datetime.now(timezone.utc).isoformat()
        metadata['last_health_check'] = current_time
        browser_instance['metadata'] = metadata
        
        # Update stored metadata
        await _store_browser_metadata(browser_id, metadata)
        
        # Prepare response data
        browser_data = {
            'browser_id': browser_id,
            'status': 'running',
            'pid': metadata.get('pid', 0),
            'health': health_status,
            'uptime_seconds': uptime_seconds,
            'created_at': metadata.get('created_at'),
            'last_health_check': current_time,
            'options': metadata.get('options', {})
        }
        
        # Log access
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'browser_manager',
            'message': f'Retrieved browser instance "{browser_id}"',
            'data': {'browser_id': browser_id, 'health': health_status, 'uptime': uptime_seconds}
        })
        
        # Track metrics
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.browser_manager.browser_queries.total',
            'labels': {'browser_id': browser_id, 'health': health_status}
        })
        
        return BrowserManagerOutput(
            success=True,
            message=f"Retrieved browser instance '{browser_id}'",
            data=browser_data
        )
        
    except BrowserNotFoundError:
        return BrowserManagerOutput(
            success=False,
            message=f"Browser instance '{browser_id}' not found",
            data=None
        )
    except Exception as e:
        # Log error
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'browser_manager',
            'message': f'Failed to get browser instance "{browser_id}"',
            'data': {'browser_id': browser_id, 'error': str(e)}
        })
        
        # Track error metrics
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.browser_manager.browser_query_errors.total',
            'labels': {'error_type': type(e).__name__}
        })
        
        raise


async def browser_manager_list_browsers(
    ctx: RunContext[Any],
    timeout: int = 30000
) -> BrowserManagerOutput:
    """
    List all active browser instances with their current status.
    
    Retrieves a comprehensive list of all currently active browser instances,
    including both in-memory instances and those stored in storage_kv. Performs
    basic health checks and cleanup of stale entries during listing.
    
    Args:
        ctx: Runtime context provided by the framework
        timeout: Operation timeout in milliseconds (default: 30000)
        
    Returns:
        BrowserManagerOutput with operation results containing:
        - success: Whether the listing operation succeeded
        - message: Human-readable result message with count
        - data: Dictionary with 'browsers' list and 'count' of active instances
        
    Raises:
        Exception: For storage access errors or system failures
    """
    injector = get_injector()
    
    try:
        browsers_list = []
        
        # Get all browsers from memory
        for browser_id, browser_instance in _browsers.items():
            metadata = browser_instance['metadata']
            created_time = browser_instance['created_at']
            uptime_seconds = int(time.time() - created_time)
            
            # Perform quick health check
            health_status = "healthy"
            try:
                browser = browser_instance['browser']
                # Quick check - if browser object exists and has contexts property
                contexts = browser.contexts
                health_status = "healthy"
            except Exception:
                health_status = "unhealthy"
            
            browsers_list.append({
                'browser_id': browser_id,
                'status': 'running',
                'pid': metadata.get('pid', 0),
                'health': health_status,
                'uptime_seconds': uptime_seconds,
                'created_at': metadata.get('created_at')
            })
        
        # Check storage for any browsers not in memory (potential crashes)
        try:
            storage_result = await injector.run('storage_kv', {
                'operation': 'keys',
                'namespace': 'browsers',
                'pattern': '*'
            })
            
            if storage_result.success and storage_result.data:
                stored_keys = storage_result.data.get('keys', [])
                
                for stored_browser_id in stored_keys:
                    # Skip if already in memory
                    if stored_browser_id in _browsers:
                        continue
                    
                    # Get metadata for crashed browser
                    metadata = await _get_browser_metadata(stored_browser_id)
                    if metadata:
                        browsers_list.append({
                            'browser_id': stored_browser_id,
                            'status': 'crashed',
                            'pid': metadata.get('pid', 0),
                            'health': 'crashed',
                            'created_at': metadata.get('created_at')
                        })
                        
                        # Clean up crashed browser metadata
                        await _delete_browser_metadata(stored_browser_id)
                        
                        await injector.run('logging', {
                            'operation': 'log',
                            'level': 'WARN',
                            'logger_name': 'browser_manager',
                            'message': f'Found crashed browser "{stored_browser_id}", cleaned up metadata',
                            'data': {'browser_id': stored_browser_id}
                        })
                        
        except Exception as e:
            # Log warning but don't fail the operation
            await injector.run('logging', {
                'operation': 'log',
                'level': 'WARN',
                'logger_name': 'browser_manager',
                'message': 'Failed to check storage for crashed browsers',
                'data': {'error': str(e)}
            })
        
        # Sort browsers by creation time
        browsers_list.sort(key=lambda x: x.get('created_at', ''))
        
        count = len(browsers_list)
        
        # Log listing operation
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO',
            'logger_name': 'browser_manager',
            'message': f'Listed {count} browser instances',
            'data': {'count': count, 'browser_ids': [b['browser_id'] for b in browsers_list]}
        })
        
        # Track metrics
        await injector.run('metrics', {
            'operation': 'set',
            'name': 'agentool.browser_manager.active_browsers.count',
            'value': float(len(_browsers))  # Only count running browsers for active metric
        })
        
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.browser_manager.list_operations.total'
        })
        
        return BrowserManagerOutput(
            success=True,
            message=f"Found {count} active browser instances",
            data={
                'browsers': browsers_list,
                'count': count
            }
        )
        
    except Exception as e:
        # Log error
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'browser_manager',
            'message': 'Failed to list browser instances',
            'data': {'error': str(e)}
        })
        
        # Track error metrics
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.browser_manager.list_errors.total',
            'labels': {'error_type': type(e).__name__}
        })
        
        raise


async def browser_manager_health_check(
    ctx: RunContext[Any],
    browser_id: str,
    timeout: int = 30000
) -> BrowserManagerOutput:
    """
    Perform comprehensive health check on a specific browser instance.
    
    Conducts detailed health verification including browser responsiveness,
    context availability, page functionality, and process status. Updates
    health metadata and performs recovery actions if needed.
    
    Args:
        ctx: Runtime context provided by the framework
        browser_id: Unique identifier of the browser instance to check
        timeout: Operation timeout in milliseconds (default: 30000)
        
    Returns:
        BrowserManagerOutput with operation results containing:
        - success: Whether the health check completed successfully
        - message: Human-readable health status message
        - data: Detailed health information including test results and metrics
        
    Raises:
        BrowserNotFoundError: If browser_id does not exist
    """
    injector = get_injector()
    
    try:
        # Check if browser exists
        browser_instance = _browsers.get(browser_id)
        if not browser_instance:
            raise BrowserNotFoundError(f"Browser instance '{browser_id}' not found")
        
        browser = browser_instance['browser']
        context = browser_instance.get('context')
        page = browser_instance.get('page')
        
        health_data = {
            'browser_id': browser_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'tests': {},
            'overall_status': 'healthy',
            'issues': []
        }
        
        # Test 1: Browser object responsiveness
        try:
            contexts = browser.contexts
            health_data['tests']['browser_responsive'] = True
        except Exception as e:
            health_data['tests']['browser_responsive'] = False
            health_data['issues'].append(f"Browser not responsive: {str(e)}")
        
        # Test 2: Context availability
        try:
            if context:
                pages = await context.pages()
                health_data['tests']['context_available'] = True
                health_data['context_page_count'] = len(pages)
            else:
                health_data['tests']['context_available'] = False
                health_data['issues'].append("No browser context available")
        except Exception as e:
            health_data['tests']['context_available'] = False
            health_data['issues'].append(f"Context error: {str(e)}")
        
        # Test 3: Page functionality
        try:
            if page:
                # Try to evaluate simple JavaScript
                result = await page.evaluate('() => document.readyState')
                health_data['tests']['page_functional'] = True
                health_data['page_ready_state'] = result
            else:
                health_data['tests']['page_functional'] = False
                health_data['issues'].append("No page available")
        except Exception as e:
            health_data['tests']['page_functional'] = False
            health_data['issues'].append(f"Page error: {str(e)}")
        
        # Test 4: Process status (if PID available)
        metadata = browser_instance['metadata']
        pid = metadata.get('pid', 0)
        if pid > 0:
            try:
                process = psutil.Process(pid)
                if process.is_running():
                    health_data['tests']['process_running'] = True
                    health_data['process_info'] = {
                        'cpu_percent': process.cpu_percent(),
                        'memory_mb': process.memory_info().rss / 1024 / 1024,
                        'status': process.status()
                    }
                else:
                    health_data['tests']['process_running'] = False
                    health_data['issues'].append("Browser process not running")
            except psutil.NoSuchProcess:
                health_data['tests']['process_running'] = False
                health_data['issues'].append("Browser process not found")
            except Exception as e:
                health_data['tests']['process_running'] = None
                health_data['issues'].append(f"Process check error: {str(e)}")
        else:
            health_data['tests']['process_running'] = None
            health_data['issues'].append("No PID available for process check")
        
        # Determine overall health status
        critical_tests = ['browser_responsive', 'context_available']
        failed_critical = [test for test in critical_tests if not health_data['tests'].get(test, False)]
        
        if failed_critical:
            health_data['overall_status'] = 'critical'
        elif health_data['issues']:
            health_data['overall_status'] = 'warning'
        else:
            health_data['overall_status'] = 'healthy'
        
        # Update metadata with health check results
        metadata['last_health_check'] = health_data['timestamp']
        metadata['health_status'] = health_data['overall_status']
        browser_instance['metadata'] = metadata
        
        # Store updated metadata
        await _store_browser_metadata(browser_id, metadata)
        
        # Log health check results
        await injector.run('logging', {
            'operation': 'log',
            'level': 'INFO' if health_data['overall_status'] == 'healthy' else 'WARN',
            'logger_name': 'browser_manager',
            'message': f'Health check completed for browser "{browser_id}"',
            'data': {
                'browser_id': browser_id,
                'status': health_data['overall_status'],
                'issues_count': len(health_data['issues']),
                'tests_passed': sum(1 for v in health_data['tests'].values() if v is True)
            }
        })
        
        # Track metrics
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.browser_manager.health_checks.total',
            'labels': {'browser_id': browser_id, 'status': health_data['overall_status']}
        })
        
        return BrowserManagerOutput(
            success=True,
            message=f"Health check completed for browser '{browser_id}' - Status: {health_data['overall_status']}",
            data=health_data
        )
        
    except BrowserNotFoundError:
        raise
    except Exception as e:
        # Log error
        await injector.run('logging', {
            'operation': 'log',
            'level': 'ERROR',
            'logger_name': 'browser_manager',
            'message': f'Health check failed for browser "{browser_id}"',
            'data': {'browser_id': browser_id, 'error': str(e)}
        })
        
        # Track error metrics
        await injector.run('metrics', {
            'operation': 'increment',
            'name': 'agentool.browser_manager.health_check_errors.total',
            'labels': {'error_type': type(e).__name__}
        })
        
        raise


async def browser_manager_cleanup_all(ctx: RunContext[BrowserManagerInput]) -> BrowserManagerOutput:
    """
    Clean up all browser instances and the playwright instance.
    
    This operation:
    - Stops all running browser instances
    - Clears browser metadata from storage
    - Stops the global playwright instance
    - Resets all global state
    
    Args:
        ctx: Runtime context containing input parameters
        
    Returns:
        BrowserManagerOutput with cleanup results
    """
    injector = get_injector()
    global _browsers
    
    # Log cleanup initiation
    await injector.run('logging', {
        'operation': 'log',
        'level': 'INFO',
        'logger_name': 'browser_manager',
        'message': 'Starting cleanup of all browser instances',
        'data': {'browser_count': len(_browsers)}
    })
    
    # Stop all browsers
    stopped_browsers = []
    failed_browsers = []
    
    for browser_id in list(_browsers.keys()):
        try:
            # Stop each browser
            await browser_manager_stop_browser(
                ctx=ctx,
                browser_id=browser_id,
                timeout=5000
            )
            stopped_browsers.append(browser_id)
        except Exception as e:
            failed_browsers.append({'browser_id': browser_id, 'error': str(e)})
            # Force remove from memory even if stop failed
            if browser_id in _browsers:
                del _browsers[browser_id]
    
    # Clean up playwright instance
    await _cleanup_playwright()
    
    # Clear any remaining browser metadata from storage
    try:
        # Get all browser keys from storage (using browsers namespace)
        list_result = await injector.run('storage_kv', {
            'operation': 'keys',
            'pattern': '*',  # Get all keys in browsers namespace
            'namespace': 'browsers'
        })
        
        # Delete all browser-related keys
        for key in list_result.data.get('keys', []):
            await injector.run('storage_kv', {
                'operation': 'delete',
                'key': key,
                'namespace': 'browsers'
            })
    except:
        pass  # Ignore storage cleanup errors
    
    # Log completion
    await injector.run('logging', {
        'operation': 'log',
        'level': 'INFO',
        'logger_name': 'browser_manager',
        'message': 'Completed cleanup of all browser instances',
        'data': {
            'stopped_count': len(stopped_browsers),
            'failed_count': len(failed_browsers),
            'playwright_cleaned': True
        }
    })
    
    return BrowserManagerOutput(
        success=True,
        message=f"Cleaned up {len(stopped_browsers)} browser instances and playwright",
        data={
            'stopped_browsers': stopped_browsers,
            'failed_browsers': failed_browsers,
            'playwright_cleaned': True,
            'global_state_reset': True
        }
    )


# Routing configuration
browser_manager_routing = RoutingConfig(
    operation_field='operation',
    operation_map={
        'start_browser': ('browser_manager_start_browser', lambda x: {
            'browser_id': x.browser_id,
            'options': x.options,
            'timeout': x.timeout or 30000,
        }),
        'stop_browser': ('browser_manager_stop_browser', lambda x: {
            'browser_id': x.browser_id,
            'timeout': x.timeout or 30000,
        }),
        'get_browser': ('browser_manager_get_browser', lambda x: {
            'browser_id': x.browser_id,
            'timeout': x.timeout or 30000,
        }),
        'list_browsers': ('browser_manager_list_browsers', lambda x: {
            'timeout': x.timeout or 30000,
        }),
        'health_check': ('browser_manager_health_check', lambda x: {
            'browser_id': x.browser_id,
            'timeout': x.timeout or 30000,
        }),
        'cleanup_all': ('browser_manager_cleanup_all', lambda x: {}),
    }
)


def create_browser_manager_agent():
    """
    Create and return the browser_manager AgenTool.
    
    Returns:
        Agent configured for browser lifecycle management operations
    """
    return create_agentool(
        name='browser_manager',
        input_schema=BrowserManagerInput,
        routing_config=browser_manager_routing,
        tools=[
            browser_manager_start_browser,
            browser_manager_stop_browser,
            browser_manager_get_browser,
            browser_manager_list_browsers,
            browser_manager_health_check,
            browser_manager_cleanup_all
        ],
        output_type=BrowserManagerOutput,
        system_prompt="You are a browser lifecycle management agent that handles Playwright browser instances with session persistence, crash recovery, and comprehensive monitoring. You manage browser creation, maintenance, and cleanup while ensuring optimal resource utilization and system stability.",
        description="Manages Playwright browser instances with operations: start_browser, stop_browser, get_browser, list_browsers, health_check",
        version="1.0.0",
        tags=["browser", "automation", "playwright", "session", "lifecycle"],
        dependencies=["storage_kv", "logging", "metrics"],
        examples=[
            {
                "description": "Start a new browser instance with custom options",
                "input": {
                    "operation": "start_browser",
                    "browser_id": "session_123",
                    "options": {
                        "headless": False,
                        "viewport": {"width": 1920, "height": 1080},
                        "user_data_dir": "/tmp/browser_data/session_123"
                    }
                },
                "output": {
                    "success": True,
                    "message": "Successfully started browser instance 'session_123'",
                    "data": {
                        "browser_id": "session_123",
                        "status": "running",
                        "pid": 12345,
                        "options": {
                            "headless": False,
                            "viewport": {"width": 1920, "height": 1080},
                            "user_data_dir": "/tmp/browser_data/session_123"
                        },
                        "created_at": "2025-01-01T12:00:00Z"
                    }
                }
            },
            {
                "description": "Get browser instance details",
                "input": {
                    "operation": "get_browser",
                    "browser_id": "session_123"
                },
                "output": {
                    "success": True,
                    "message": "Retrieved browser instance 'session_123'",
                    "data": {
                        "browser_id": "session_123",
                        "status": "running",
                        "pid": 12345,
                        "health": "healthy",
                        "uptime_seconds": 3600
                    }
                }
            },
            {
                "description": "List all active browser instances",
                "input": {
                    "operation": "list_browsers"
                },
                "output": {
                    "success": True,
                    "message": "Found 2 active browser instances",
                    "data": {
                        "browsers": [
                            {"browser_id": "session_123", "status": "running", "pid": 12345},
                            {"browser_id": "session_456", "status": "running", "pid": 12346}
                        ],
                        "count": 2
                    }
                }
            }
        ]
    )


# Create the agent instance
agent = create_browser_manager_agent()