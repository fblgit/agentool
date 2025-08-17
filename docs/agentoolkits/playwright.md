# Playwright AgenToolkit

> **Note**: For guidelines on creating new AgenToolkits, see [CRAFTING_AGENTOOLS.md](../CRAFTING_AGENTOOLS.md). For testing patterns and examples, refer to [tests/agentoolkit/test_playwright.py](../../tests/agentoolkit/test_playwright.py).

## Overview

The Playwright AgenToolkit provides comprehensive browser automation capabilities through three integrated components that work together to deliver end-to-end web automation solutions. Built on Microsoft's Playwright library, it offers reliable browser automation with modern web support, cross-browser compatibility, and enterprise-grade stability.

### Key Features
- **Full Browser Lifecycle Management**: Launch, monitor, and terminate browser instances with session persistence
- **Advanced Navigation Capabilities**: URL navigation, page state management, content extraction, and screenshot capture
- **Fine-Grained Element Interaction**: DOM element finding, clicking, typing, form handling, and advanced interactions
- **Multi-Browser Support**: Chromium browser support with configurable options
- **Session Persistence**: Browser sessions with unique identifiers and crash recovery
- **Content Processing**: HTML parsing with BeautifulSoup integration for structured data extraction
- **State Management**: Cookie and storage manipulation (localStorage, sessionStorage)
- **JavaScript Execution**: Custom script evaluation in page context
- **File Operations**: Screenshot capture and file upload handling
- **Comprehensive Monitoring**: Health checks, metrics tracking, and detailed logging

## Architecture

The Playwright AgenToolkit consists of three specialized agents that work together:

```
Browser Manager → Page Navigator → Element Interactor
      ↓                ↓                 ↓
  Lifecycle         Navigation       Interactions
  Management        & Content        & Automation
```

## Components

### 1. Browser Manager Agent

Manages browser instances with session persistence and lifecycle control.

```python
from agentoolkit.playwright.browser_manager import create_browser_manager_agent

agent = create_browser_manager_agent()
```

**Operations**: `start_browser`, `stop_browser`, `get_browser`, `list_browsers`, `health_check`, `cleanup_all`

### 2. Page Navigator Agent

Handles page navigation, content extraction, and high-level browser interactions.

```python
from agentoolkit.playwright.page_navigator import create_page_navigator_agent

agent = create_page_navigator_agent()
```

**Operations**: `navigate`, `back`, `forward`, `refresh`, `wait_for_load`, `capture_screenshot`, `get_content`, `manage_cookies`, `manage_storage`, `evaluate`

### 3. Element Interactor Agent

Provides fine-grained DOM element interaction and manipulation capabilities.

```python
from agentoolkit.playwright.element_interactor import create_element_interactor_agent

agent = create_element_interactor_agent()
```

**Operations**: `find_element`, `click_element`, `type_text`, `select_option`, `get_element_text`, `get_element_attribute`, `wait_for_element`, `upload_file`, `hover_element`, `drag_element`, `keyboard_shortcut`, `element_screenshot`

## Input/Output Schemas

### Browser Manager Input Schema

#### BrowserManagerInput

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `operation` | `Literal['start_browser', 'stop_browser', 'get_browser', 'list_browsers', 'health_check', 'cleanup_all']` | Yes | - | Browser management operation to perform |
| `browser_id` | `Optional[str]` | No* | None | Unique identifier for the browser instance |
| `options` | `Optional[Dict[str, Any]]` | No | None | Browser configuration options |
| `timeout` | `Optional[int]` | No | 30000 | Operation timeout in milliseconds |

*Required for: `start_browser`, `stop_browser`, `get_browser`, `health_check`

#### Browser Options Structure

| Field | Type | Description |
|-------|------|-------------|
| `headless` | `bool` | Whether to run in headless mode (default: False) |
| `viewport` | `Dict[str, int]` | Viewport size with `width` and `height` |
| `user_data_dir` | `str` | Path for persistent user data storage |
| `args` | `List[str]` | Additional browser launch arguments |

### Page Navigator Input Schema

#### PageNavigatorInput

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `operation` | `Literal['navigate', 'back', 'forward', 'refresh', 'wait_for_load', 'capture_screenshot', 'get_content', 'manage_cookies', 'manage_storage', 'evaluate']` | Yes | - | Navigation operation to perform |
| `browser_id` | `str` | Yes | - | Browser session identifier |
| `url` | `Optional[str]` | No* | None | URL for navigate operation |
| `wait_condition` | `Optional[Literal['load', 'domcontentloaded', 'networkidle']]` | No | None | Wait condition for navigation |
| `timeout` | `Optional[float]` | No | None | Timeout in milliseconds |
| `selector` | `Optional[str]` | No | None | CSS selector for element-based wait conditions |
| `screenshot_path` | `Optional[str]` | No* | None | File path for screenshot capture |
| `full_page` | `Optional[bool]` | No | None | Whether to capture full page screenshot |
| `cookie_action` | `Optional[Literal['get', 'set', 'delete', 'clear']]` | No* | None | Cookie management action |
| `cookie_name` | `Optional[str]` | No | None | Cookie name for operations |
| `cookie_value` | `Optional[str]` | No | None | Cookie value for set operation |
| `cookie_options` | `Optional[Dict[str, Any]]` | No | None | Cookie options (domain, path, secure, httpOnly) |
| `storage_type` | `Optional[Literal['local', 'session']]` | No* | None | Storage type for manage_storage |
| `storage_action` | `Optional[Literal['get', 'set', 'remove', 'clear']]` | No* | None | Storage management action |
| `storage_key` | `Optional[str]` | No | None | Storage key for operations |
| `storage_value` | `Optional[str]` | No | None | Storage value for set operation |
| `script` | `Optional[str]` | No* | None | JavaScript code to evaluate |
| `extract_content` | `Optional[bool]` | No | None | Whether to parse content with BeautifulSoup |

*Required depending on operation

### Element Interactor Input Schema

#### ElementInteractorInput

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `operation` | `Literal['find_element', 'click_element', 'type_text', 'select_option', 'get_element_text', 'get_element_attribute', 'wait_for_element', 'upload_file', 'hover_element', 'drag_element', 'keyboard_shortcut', 'element_screenshot']` | Yes | - | Element interaction operation |
| `browser_id` | `str` | Yes | - | Browser instance identifier |
| `selector` | `str` | Yes | - | CSS selector, XPath (prefix: xpath=), or text selector (prefix: text=) |
| `text` | `Optional[str]` | No* | None | Text to type into element |
| `value` | `Optional[str]` | No* | None | Value to select in dropdown |
| `attribute` | `Optional[str]` | No* | None | HTML attribute name to retrieve |
| `file_path` | `Optional[str]` | No* | None | Path to file for upload |
| `timeout` | `int` | No | 30000 | Timeout in milliseconds |
| `wait_condition` | `str` | No | "visible" | Condition to wait for (visible, hidden, attached, detached, stable) |
| `target_selector` | `Optional[str]` | No* | None | Target element for drag operations |
| `keys` | `Optional[str]` | No* | None | Keyboard shortcut keys |
| `clear_first` | `bool` | No | True | Whether to clear existing text before typing |
| `click_count` | `int` | No | 1 | Number of clicks to perform |
| `force` | `bool` | No | False | Force action even if element not actionable |

*Required depending on operation

### Output Schemas

All three agents return similar output structures:

#### BrowserManagerOutput / PageNavigatorOutput / ElementInteractorOutput

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Whether the operation succeeded |
| `message` | `str` | Human-readable result message |
| `data` | `Optional[Dict[str, Any]]` | Operation-specific return data |

## Operations

### Browser Manager Operations

| Operation | Function | Description | Key Data Fields |
|-----------|----------|-------------|-----------------|
| `start_browser` | `browser_manager_start_browser` | Launch new browser instance | `browser_id`, `status`, `pid`, `options`, `created_at` |
| `stop_browser` | `browser_manager_stop_browser` | Stop browser gracefully | `browser_id`, `status`, `cleanup_completed` |
| `get_browser` | `browser_manager_get_browser` | Get browser information | `browser_id`, `status`, `health`, `uptime_seconds`, `pid` |
| `list_browsers` | `browser_manager_list_browsers` | List all active browsers | `browsers` (array), `count` |
| `health_check` | `browser_manager_health_check` | Comprehensive health check | `tests`, `overall_status`, `issues`, `process_info` |
| `cleanup_all` | `browser_manager_cleanup_all` | Clean up all browsers | `stopped_browsers`, `failed_browsers`, `playwright_cleaned` |

### Page Navigator Operations

| Operation | Function | Description | Key Data Fields |
|-----------|----------|-------------|-----------------|
| `navigate` | `page_navigator_navigate` | Navigate to URL | `url`, `title`, `load_time` |
| `back` | `page_navigator_back` | Navigate back in history | `url`, `title` |
| `forward` | `page_navigator_forward` | Navigate forward in history | `url`, `title` |
| `refresh` | `page_navigator_refresh` | Refresh current page | `url`, `title` |
| `wait_for_load` | `page_navigator_wait_for_load` | Wait for page/element load | `wait_condition`, `selector`, `wait_time_ms` |
| `capture_screenshot` | `page_navigator_capture_screenshot` | Capture page screenshot | `path`, `size_bytes`, `dimensions`, `full_page` |
| `get_content` | `page_navigator_get_content` | Extract page content | `html`, `title`, `text_content`, `links`, `forms`, `headings`* |
| `manage_cookies` | `page_navigator_manage_cookies` | Cookie operations | Varies by action: cookie data, count, etc. |
| `manage_storage` | `page_navigator_manage_storage` | Browser storage operations | Varies by action: storage items, values, etc. |
| `evaluate` | `page_navigator_evaluate` | Execute JavaScript | `result`, `type` |

*When `extract_content=True`

### Element Interactor Operations

| Operation | Function | Description | Key Data Fields |
|-----------|----------|-------------|-----------------|
| `find_element` | `element_interactor_find_element` | Find element by selector | `element_found`, `tag_name`, `visible`, `enabled` |
| `click_element` | `element_interactor_click_element` | Click element | `clicked`, `click_count` |
| `type_text` | `element_interactor_type_text` | Type text into element | `text_entered`, `cleared_first` |
| `select_option` | `element_interactor_select_option` | Select dropdown option | `selected`, `value` |
| `get_element_text` | `element_interactor_get_element_text` | Get element text content | `text` |
| `get_element_attribute` | `element_interactor_get_element_attribute` | Get element attribute | `attribute`, `value` |
| `wait_for_element` | `element_interactor_wait_for_element` | Wait for element condition | `condition_met`, `wait_condition`, `wait_time_ms` |
| `upload_file` | `element_interactor_upload_file` | Upload file to input | `file_uploaded`, `file_path` |
| `hover_element` | `element_interactor_hover_element` | Hover over element | `hovered` |
| `drag_element` | `element_interactor_drag_element` | Drag element to target | `dragged`, `target_selector` |
| `keyboard_shortcut` | `element_interactor_keyboard_shortcut` | Send keyboard shortcut | `keys_sent`, `keys` |
| `element_screenshot` | `element_interactor_element_screenshot` | Screenshot specific element | `screenshot_taken`, `screenshot_path`, `size_bytes` |

## Usage Examples

### Complete Browser Automation Workflow

```python
from agentoolkit.playwright import (
    create_browser_manager_agent,
    create_page_navigator_agent, 
    create_element_interactor_agent
)
from agentool.core.injector import get_injector

# Create and register all agents
browser_agent = create_browser_manager_agent()
navigator_agent = create_page_navigator_agent()
interactor_agent = create_element_interactor_agent()

injector = get_injector()

# 1. Start a browser instance
browser_result = await injector.run('browser_manager', {
    "operation": "start_browser",
    "browser_id": "automation_session",
    "options": {
        "headless": False,
        "viewport": {"width": 1920, "height": 1080},
        "user_data_dir": "/tmp/browser_data"
    }
})

# 2. Navigate to a page
nav_result = await injector.run('page_navigator', {
    "operation": "navigate",
    "browser_id": "automation_session",
    "url": "https://example.com/login",
    "wait_condition": "load"
})

# 3. Find and interact with login elements
find_result = await injector.run('element_interactor', {
    "operation": "find_element",
    "browser_id": "automation_session",
    "selector": "#username"
})

# 4. Type credentials
await injector.run('element_interactor', {
    "operation": "type_text",
    "browser_id": "automation_session",
    "selector": "#username",
    "text": "user@example.com"
})

await injector.run('element_interactor', {
    "operation": "type_text",
    "browser_id": "automation_session",
    "selector": "#password",
    "text": "secretpassword"
})

# 5. Submit form
await injector.run('element_interactor', {
    "operation": "click_element",
    "browser_id": "automation_session",
    "selector": "button[type='submit']"
})

# 6. Wait for page load and capture screenshot
await injector.run('page_navigator', {
    "operation": "wait_for_load",
    "browser_id": "automation_session",
    "wait_condition": "networkidle"
})

await injector.run('page_navigator', {
    "operation": "capture_screenshot",
    "browser_id": "automation_session",
    "screenshot_path": "screenshots/after_login.png",
    "full_page": True
})

# 7. Clean up
await injector.run('browser_manager', {
    "operation": "stop_browser",
    "browser_id": "automation_session"
})
```

### Content Extraction and Processing

```python
# Navigate and extract structured content
await injector.run('page_navigator', {
    "operation": "navigate",
    "browser_id": "session_123",
    "url": "https://news.example.com"
})

# Extract page content with BeautifulSoup parsing
content_result = await injector.run('page_navigator', {
    "operation": "get_content",
    "browser_id": "session_123",
    "extract_content": True
})

# Content result includes:
# - html: Raw HTML content
# - text_content: Clean text content
# - links: Array of found links
# - forms: Array of form elements
# - headings: Array of heading elements with levels
```

### Form Automation with File Upload

```python
# Navigate to form page
await injector.run('page_navigator', {
    "operation": "navigate",
    "browser_id": "form_session",
    "url": "https://example.com/upload-form"
})

# Fill text fields
await injector.run('element_interactor', {
    "operation": "type_text",
    "browser_id": "form_session",
    "selector": "input[name='title']",
    "text": "Document Title"
})

# Select dropdown option
await injector.run('element_interactor', {
    "operation": "select_option",
    "browser_id": "form_session",
    "selector": "select[name='category']",
    "value": "documents"
})

# Upload file
await injector.run('element_interactor', {
    "operation": "upload_file",
    "browser_id": "form_session",
    "selector": "input[type='file']",
    "file_path": "/path/to/document.pdf"
})

# Submit form
await injector.run('element_interactor', {
    "operation": "click_element",
    "browser_id": "form_session",
    "selector": "button[type='submit']"
})
```

### Cookie and Storage Management

```python
# Set authentication cookies
await injector.run('page_navigator', {
    "operation": "manage_cookies",
    "browser_id": "session_123",
    "cookie_action": "set",
    "cookie_name": "auth_token",
    "cookie_value": "abc123def456",
    "cookie_options": {
        "domain": ".example.com",
        "path": "/",
        "secure": True,
        "httpOnly": True
    }
})

# Manage localStorage
await injector.run('page_navigator', {
    "operation": "manage_storage",
    "browser_id": "session_123",
    "storage_type": "local",
    "storage_action": "set",
    "storage_key": "user_preferences",
    "storage_value": '{"theme": "dark", "lang": "en"}'
})

# Get all cookies
cookies_result = await injector.run('page_navigator', {
    "operation": "manage_cookies",
    "browser_id": "session_123",
    "cookie_action": "get"
})
```

### Advanced Element Interactions

```python
# Wait for element to be visible
await injector.run('element_interactor', {
    "operation": "wait_for_element",
    "browser_id": "session_123",
    "selector": ".dynamic-content",
    "wait_condition": "visible",
    "timeout": 10000
})

# Hover to reveal dropdown
await injector.run('element_interactor', {
    "operation": "hover_element",
    "browser_id": "session_123",
    "selector": ".menu-item"
})

# Use keyboard shortcuts
await injector.run('element_interactor', {
    "operation": "keyboard_shortcut",
    "browser_id": "session_123",
    "selector": "textarea",
    "keys": "Control+A"
})

# Drag and drop
await injector.run('element_interactor', {
    "operation": "drag_element",
    "browser_id": "session_123",
    "selector": ".draggable-item",
    "target_selector": ".drop-zone"
})

# Get element attribute
attr_result = await injector.run('element_interactor', {
    "operation": "get_element_attribute",
    "browser_id": "session_123",
    "selector": "img.profile-pic",
    "attribute": "src"
})
```

### Browser Health Monitoring

```python
# Comprehensive health check
health_result = await injector.run('browser_manager', {
    "operation": "health_check",
    "browser_id": "session_123"
})

# Health result includes:
# - tests: Dict of test results (browser_responsive, context_available, etc.)
# - overall_status: "healthy", "warning", or "critical"
# - issues: Array of detected issues
# - process_info: CPU, memory usage if available

# List all browser instances
list_result = await injector.run('browser_manager', {
    "operation": "list_browsers"
})

# Returns browsers with status, health, uptime information
```

### JavaScript Execution

```python
# Execute custom JavaScript
js_result = await injector.run('page_navigator', {
    "operation": "evaluate",
    "browser_id": "session_123",
    "script": """
    return {
        title: document.title,
        url: window.location.href,
        viewport: {
            width: window.innerWidth,
            height: window.innerHeight
        },
        readyState: document.readyState
    };
    """
})

# Result includes the returned JavaScript value
# with type information
```

## Integration Patterns

### With Storage Agentoolkits

```python
# Save browser session data
session_data = {
    "browser_id": "session_123",
    "last_url": nav_result.data["url"],
    "cookies": cookies_result.data["cookies"]
}

await injector.run('storage_kv', {
    "operation": "set",
    "key": "browser_session",
    "value": session_data,
    "namespace": "automation"
})

# Save screenshots to file system
await injector.run('storage_fs', {
    "operation": "exists",
    "path": "screenshots"
})
```

### With Logging and Metrics

The toolkit automatically integrates with logging and metrics agentoolkits:

```python
# Logs are automatically generated for all operations
# Metrics are tracked for:
# - Browser starts/stops/errors
# - Navigation operations and timing
# - Element interaction success/failure rates
# - Health check results
```

### Parallel Browser Sessions

```python
# Start multiple browser sessions
sessions = ["session_1", "session_2", "session_3"]

for session_id in sessions:
    await injector.run('browser_manager', {
        "operation": "start_browser",
        "browser_id": session_id,
        "options": {"headless": True}
    })

# Perform parallel operations
import asyncio

async def automate_session(session_id, url):
    await injector.run('page_navigator', {
        "operation": "navigate",
        "browser_id": session_id,
        "url": url
    })
    # ... perform automation tasks
    
tasks = [
    automate_session("session_1", "https://site1.com"),
    automate_session("session_2", "https://site2.com"),
    automate_session("session_3", "https://site3.com")
]

await asyncio.gather(*tasks)
```

## Error Handling

### Browser Manager Errors

| Exception | Scenarios | Resolution |
|-----------|-----------|------------|
| `BrowserNotFoundError` | Browser ID not found | Check browser exists with `list_browsers` |
| `BrowserStartupError` | Playwright launch failure | Check system resources, permissions |
| `InvalidOptionsError` | Invalid browser options | Validate options structure |
| `ResourceExhaustionError` | Insufficient system resources | Free memory/CPU, reduce browser count |
| `ProcessError` | Browser process management | Force cleanup, restart system |

### Page Navigator Errors

| Exception | Scenarios | Resolution |
|-----------|-----------|------------|
| `RuntimeError` | Navigation timeout, page errors | Increase timeout, check network |
| `NavigationError` | Invalid URLs, network failures | Validate URL, check connectivity |
| `TimeoutError` | Page load timeout | Increase timeout, check page performance |
| `FileSystemError` | Screenshot save failure | Check storage permissions and space |

### Element Interactor Errors

| Exception | Scenarios | Resolution |
|-----------|-----------|------------|
| `BrowserNotFoundError` | Browser session closed | Restart browser session |
| `ElementNotFoundError` | Selector doesn't match | Verify selector, wait for element |
| `ElementNotInteractableError` | Element not clickable/visible | Wait for element, check page state |
| `TimeoutError` | Element operation timeout | Increase timeout, check element state |
| `InvalidSelectorError` | Malformed selector syntax | Fix selector syntax |

### Error Response Format

```python
# Failed operation response
{
    "success": False,
    "message": "Element not found with selector '#missing-element'",
    "data": {
        "error": "ElementNotFoundError",
        "selector": "#missing-element", 
        "timeout_ms": 30000
    }
}
```

## Best Practices

### Browser Session Management

1. **Use Unique Session IDs**: Avoid conflicts with descriptive identifiers
2. **Clean Up Sessions**: Always stop browsers when done
3. **Monitor Health**: Regular health checks for long-running sessions
4. **Resource Limits**: Monitor system resources with multiple browsers

### Element Selection

1. **Stable Selectors**: Use IDs or data attributes over position-based selectors
2. **Wait Strategies**: Always wait for elements before interaction
3. **Fallback Selectors**: Have backup selectors for dynamic content
4. **Selector Validation**: Test selectors in browser dev tools first

### Performance Optimization

1. **Headless Mode**: Use for better performance when visual feedback not needed
2. **Selective Screenshots**: Only capture when necessary
3. **Content Extraction**: Use `extract_content=True` only when needed
4. **Timeout Tuning**: Adjust timeouts based on page performance

### Error Recovery

1. **Retry Logic**: Implement retry for transient failures
2. **Graceful Degradation**: Handle missing elements gracefully
3. **Session Recovery**: Save session state for crash recovery
4. **Health Monitoring**: Regular health checks for early issue detection

## Dependencies

### Required Dependencies

- **browser_manager**: For browser instance management (used by page_navigator and element_interactor)
- **storage_fs**: For screenshot and file operations
- **storage_kv**: For session state persistence
- **logging**: For operation logging and debugging
- **metrics**: For performance monitoring and analytics

### Optional Dependencies

- **BeautifulSoup4**: For advanced HTML content parsing (`extract_content=True`)

### System Dependencies

- **Playwright**: Microsoft Playwright browser automation library
- **psutil**: For system resource monitoring and process management
- **Chromium**: Browser engine (automatically installed by Playwright)

## Testing

The test suite is located at `tests/agentoolkit/test_playwright.py`. Tests cover:

### Browser Manager Tests
- Browser lifecycle (start, stop, cleanup)
- Session persistence and recovery
- Health monitoring and metrics
- Resource management and limits
- Error conditions and timeouts

### Page Navigator Tests  
- Navigation operations (forward, back, refresh)
- Content extraction with BeautifulSoup
- Screenshot capture and storage
- Cookie and storage management
- JavaScript evaluation
- Wait conditions and timeouts

### Element Interactor Tests
- Element finding with various selectors
- Interaction operations (click, type, select)
- Advanced interactions (hover, drag, keyboard)
- File upload handling
- Element screenshots
- Error handling and recovery

### Integration Tests
- Multi-agent workflows
- Session sharing between agents
- Error propagation and recovery
- Performance under load

To run tests:
```bash
# Run all Playwright tests
pytest tests/agentoolkit/test_playwright.py -v

# Run specific test categories
pytest tests/agentoolkit/test_playwright.py::test_browser_manager -v
pytest tests/agentoolkit/test_playwright.py::test_page_navigator -v
pytest tests/agentoolkit/test_playwright.py::test_element_interactor -v

# Run with coverage
pytest tests/agentoolkit/test_playwright.py --cov=src/agentoolkit/playwright
```

## Notes

### Browser Configuration
- Default browser is Chromium (Chrome/Edge engine)
- Viewport defaults to system default if not specified
- User data directories enable session persistence across restarts
- Custom browser arguments support advanced configuration

### Selector Support
- **CSS Selectors**: Standard CSS selector syntax (`#id`, `.class`, `tag[attr]`)
- **XPath**: Prefix with `xpath=` (`xpath=//div[@class='content']`)
- **Text Content**: Prefix with `text=` (`text=Click me`)
- **Playwright Selectors**: All Playwright selector types supported

### Performance Considerations
- Headless mode provides ~30% better performance
- Screenshot operations are I/O intensive
- BeautifulSoup parsing adds processing overhead
- Multiple browser instances consume significant memory

### Security Considerations
- User data directories may contain sensitive session data
- Screenshot files may contain confidential information
- JavaScript evaluation can access page data and cookies
- File upload operations require careful path validation

### Troubleshooting
- Check browser logs in user data directory for issues
- Use health checks to diagnose browser problems  
- Monitor system resources during automation
- Enable verbose logging for debugging element selectors

The Playwright AgenToolkit provides enterprise-grade browser automation capabilities suitable for testing, web scraping, process automation, and any scenario requiring programmatic browser control.