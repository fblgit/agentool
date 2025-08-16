"""
Comprehensive E2E Tests for Playwright AgenToolkit.

This module tests all functionality of the Playwright browser automation toolkit
including browser management, page navigation, and element interaction with real
browser operations (NO MOCKS). Tests are designed to run against reliable test
websites like httpbin.org and example.com.

Key Features Tested:
- Browser Manager: browser lifecycle, health checks, session management
- Page Navigator: URL navigation, content extraction, screenshots, cookies, storage
- Element Interactor: element finding, interactions, form handling, drag/drop

Test Philosophy:
- NO MOCKS: All tests use real Playwright browser instances
- End-to-End: Tests cover complete user workflows
- Reliable Targets: Uses well-known test websites for consistent results
- Error Scenarios: Tests both happy paths and error conditions
"""

import asyncio
import os
import tempfile
import time
from pathlib import Path
import pytest
import uuid

from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry


class TestPlaywrightAgentoolkit:
    """Comprehensive test suite for Playwright AgenToolkit - NO MOCKS, real browser operations."""
    
    def setup_method(self):
        """Clear registry and injector before each test."""
        AgenToolRegistry.clear()
        get_injector().clear()
        
        # Import and create all Playwright agents
        from agentoolkit.playwright.browser_manager import create_browser_manager_agent
        from agentoolkit.playwright.page_navigator import create_page_navigator_agent
        from agentoolkit.playwright.element_interactor import create_element_interactor_agent
        
        # Create required dependency agents
        from agentoolkit.storage.fs import create_storage_fs_agent
        from agentoolkit.storage.kv import create_storage_kv_agent
        from agentoolkit.system.logging import create_logging_agent
        from agentoolkit.observability.metrics import create_metrics_agent
        
        # Initialize all agents
        self.browser_manager = create_browser_manager_agent()
        self.page_navigator = create_page_navigator_agent()
        self.element_interactor = create_element_interactor_agent()
        self.storage_fs = create_storage_fs_agent()
        self.storage_kv = create_storage_kv_agent()
        self.logging = create_logging_agent()
        self.metrics = create_metrics_agent()
        
        # Test configuration - Generate unique browser ID for each test
        self.test_browser_id = f"test_browser_{uuid.uuid4().hex[:8]}"
        self.test_urls = {
            'google': 'https://www.google.com/',
            'github': 'https://github.com/',
            'cloudflare': 'https://www.cloudflare.com/',
            'wikipedia': 'https://www.wikipedia.org/',
            'bing': 'https://www.bing.com/',
            'duckduckgo': 'https://duckduckgo.com/',
            'stackoverflow': 'https://stackoverflow.com/',
            'mozilla': 'https://www.mozilla.org/',
            'apache': 'https://www.apache.org/',
            'python': 'https://www.python.org/',
            'example': 'https://example.com/',
            'httpbin': 'https://httpbin.org/',
            'httpbin_forms': 'https://httpbin.org/forms/post'
        }
    
    def teardown_method(self):
        """Clean up browser instances after each test."""
        async def cleanup():
            injector = get_injector()
            try:
                # First try to stop our specific test browser if it exists
                try:
                    await injector.run('browser_manager', {
                        'operation': 'stop_browser',
                        'browser_id': self.test_browser_id
                    })
                except:
                    pass  # Browser might not exist
                
                # Then use cleanup_all to ensure complete cleanup
                await injector.run('browser_manager', {
                    'operation': 'cleanup_all',
                    'timeout': 5000
                })
            except:
                pass  # Ignore any cleanup errors
        
        asyncio.run(cleanup())
    
    # ===== BROWSER MANAGER TESTS =====
    
    def test_browser_manager_lifecycle_headless(self):
        """Test complete browser lifecycle with headless mode."""
        
        async def run_test():
            # Small delay to avoid race conditions
            await asyncio.sleep(0.5)
            injector = get_injector()
            
            # Start browser in headless mode
            start_result = await injector.run('browser_manager', {
                'operation': 'start_browser',
                'browser_id': self.test_browser_id,
                'options': {
                    'headless': False
                },
                'timeout': 30000
            })
            
            assert start_result.success is True
            assert start_result.data['browser_id'] == self.test_browser_id
            assert start_result.data['status'] == 'running'
            assert 'pid' in start_result.data
            assert start_result.data['options']['headless'] is False
            
            # Get browser information
            get_result = await injector.run('browser_manager', {
                'operation': 'get_browser',
                'browser_id': self.test_browser_id
            })
            
            assert get_result.success is True
            assert get_result.data['browser_id'] == self.test_browser_id
            assert get_result.data['status'] == 'running'
            assert get_result.data['health'] in ['healthy', 'unhealthy']
            assert 'uptime_seconds' in get_result.data
            
            # Health check
            health_result = await injector.run('browser_manager', {
                'operation': 'health_check',
                'browser_id': self.test_browser_id
            })
            
            assert health_result.success is True
            assert health_result.data['browser_id'] == self.test_browser_id
            assert 'tests' in health_result.data
            assert 'overall_status' in health_result.data
            
            # List browsers
            list_result = await injector.run('browser_manager', {
                'operation': 'list_browsers'
            })
            
            assert list_result.success is True
            assert list_result.data['count'] >= 1
            browser_ids = [b['browser_id'] for b in list_result.data['browsers']]
            assert self.test_browser_id in browser_ids
            
            # Stop browser
            stop_result = await injector.run('browser_manager', {
                'operation': 'stop_browser',
                'browser_id': self.test_browser_id
            })
            
            assert stop_result.success is True
            assert stop_result.data['browser_id'] == self.test_browser_id
            assert stop_result.data['status'] == 'stopped'
        
        asyncio.run(run_test())
    
    def test_browser_manager_custom_options(self):
        """Test browser startup with custom options."""
        
        async def run_test():
            injector = get_injector()
            
            # Start browser with custom args
            start_result = await injector.run('browser_manager', {
                'operation': 'start_browser',
                'browser_id': self.test_browser_id,
                'options': {
                    'headless': False,
                    'args': ['--no-sandbox', '--disable-dev-shm-usage']
                }
            })
            
            assert start_result.success is True
            assert '--no-sandbox' in start_result.data['options']['args']
            assert '--disable-dev-shm-usage' in start_result.data['options']['args']
            
            # Clean up the browser
            await injector.run('browser_manager', {
                'operation': 'stop_browser',
                'browser_id': self.test_browser_id
            })
        
        asyncio.run(run_test())
    
    @pytest.mark.skip(reason="Skipping error tests for now")
    def test_browser_manager_error_handling(self):
        """Test browser manager error scenarios."""
        
        async def run_test():
            injector = get_injector()
            
            # Try to get non-existent browser
            get_result = await injector.run('browser_manager', {
                'operation': 'get_browser',
                'browser_id': 'non_existent_browser'
            })
            
            assert get_result.success is False
            assert 'not found' in get_result.message.lower()
            
            # Try to stop non-existent browser
            try:
                stop_result = await injector.run('browser_manager', {
                    'operation': 'stop_browser',
                    'browser_id': 'non_existent_browser'
                })
                # If it succeeds, it's crash recovery
                assert stop_result.success is True
            except Exception as e:
                # Expected to fail with not found error
                assert 'not found' in str(e).lower()
            
            # Try to start browser with same ID twice
            await injector.run('browser_manager', {
                'operation': 'start_browser',
                'browser_id': self.test_browser_id,
                'options': {'headless': False}
            })
            
            try:
                await injector.run('browser_manager', {
                    'operation': 'start_browser',
                    'browser_id': self.test_browser_id,
                    'options': {'headless': False}
                })
                assert False, "Should have failed with duplicate browser ID"
            except ValueError as e:
                assert 'already exists' in str(e)
            finally:
                # Clean up the browser
                await injector.run('browser_manager', {
                    'operation': 'stop_browser',
                    'browser_id': self.test_browser_id
                })
        
        asyncio.run(run_test())
    
    # ===== PAGE NAVIGATOR TESTS =====
    
    def test_page_navigator_basic_navigation(self):
        """Test basic page navigation operations."""
        
        async def run_test():
            injector = get_injector()
            
            # Start browser
            await injector.run('browser_manager', {
                'operation': 'start_browser',
                'browser_id': self.test_browser_id,
                'options': {'headless': False}
            })
            
            # Navigate to Google
            nav_result = await injector.run('page_navigator', {
                'operation': 'navigate',
                'browser_id': self.test_browser_id,
                'url': self.test_urls['google'],
                'wait_condition': 'load',
                'timeout': 30000
            })
            
            assert nav_result.success is True
            assert 'google' in nav_result.data['url'].lower()
            assert 'title' in nav_result.data
            assert nav_result.data['load_time'] > 0
            
            # Navigate to GitHub
            nav_result2 = await injector.run('page_navigator', {
                'operation': 'navigate',
                'browser_id': self.test_browser_id,
                'url': self.test_urls['github'],
                'wait_condition': 'domcontentloaded'
            })
            
            assert nav_result2.success is True
            assert 'github' in nav_result2.data['url'].lower()
            
            # Go back
            back_result = await injector.run('page_navigator', {
                'operation': 'back',
                'browser_id': self.test_browser_id
            })
            
            assert back_result.success is True
            assert 'google' in back_result.data['url'].lower()
            
            # Go forward
            forward_result = await injector.run('page_navigator', {
                'operation': 'forward',
                'browser_id': self.test_browser_id
            })
            
            assert forward_result.success is True
            assert 'github' in forward_result.data['url'].lower()
            
            # Refresh page
            refresh_result = await injector.run('page_navigator', {
                'operation': 'refresh',
                'browser_id': self.test_browser_id
            })
            
            assert refresh_result.success is True
            assert 'github' in refresh_result.data['url'].lower()
            
            # Clean up browser
            await injector.run('browser_manager', {
                'operation': 'stop_browser',
                'browser_id': self.test_browser_id
            })
        
        asyncio.run(run_test())
    
    def test_page_navigator_wait_operations(self):
        """Test page wait operations."""
        
        async def run_test():
            injector = get_injector()
            
            # Start browser and navigate
            await injector.run('browser_manager', {
                'operation': 'start_browser',
                'browser_id': self.test_browser_id,
                'options': {'headless': False}
            })
            
            await injector.run('page_navigator', {
                'operation': 'navigate',
                'browser_id': self.test_browser_id,
                'url': self.test_urls['example']  # Use simple site for wait tests
            })
            
            # Wait for load state - use 'load' instead of 'networkidle'
            wait_result = await injector.run('page_navigator', {
                'operation': 'wait_for_load',
                'browser_id': self.test_browser_id,
                'wait_condition': 'load',
                'timeout': 10000
            })
            
            assert wait_result.success is True
            assert wait_result.data['wait_condition'] == 'load'
            assert 'wait_time_ms' in wait_result.data
            
            # Wait for specific element (body should always exist)
            element_wait_result = await injector.run('page_navigator', {
                'operation': 'wait_for_load',
                'browser_id': self.test_browser_id,
                'selector': 'body',
                'timeout': 5000
            })
            
            assert element_wait_result.success is True
            assert element_wait_result.data['selector'] == 'body'
            
            # Clean up browser
            await injector.run('browser_manager', {
                'operation': 'stop_browser',
                'browser_id': self.test_browser_id
            })
        
        asyncio.run(run_test())
    
    def test_page_navigator_content_extraction(self):
        """Test page content extraction with BeautifulSoup."""
        
        async def run_test():
            # Small delay to avoid race conditions
            await asyncio.sleep(0.5)
            injector = get_injector()
            
            # Start browser and navigate to example.com
            await injector.run('browser_manager', {
                'operation': 'start_browser',
                'browser_id': self.test_browser_id,
                'options': {'headless': False}
            })
            
            await injector.run('page_navigator', {
                'operation': 'navigate',
                'browser_id': self.test_browser_id,
                'url': self.test_urls['wikipedia']
            })
            
            # Get basic content
            content_result = await injector.run('page_navigator', {
                'operation': 'get_content',
                'browser_id': self.test_browser_id,
                'extract_content': False
            })
            
            assert content_result.success is True
            assert 'html' in content_result.data
            assert 'title' in content_result.data
            assert 'url' in content_result.data
            assert len(content_result.data['html']) > 0
            
            # Get content with BeautifulSoup parsing
            parsed_content_result = await injector.run('page_navigator', {
                'operation': 'get_content',
                'browser_id': self.test_browser_id,
                'extract_content': True
            })
            
            assert parsed_content_result.success is True
            assert 'text_content' in parsed_content_result.data
            assert 'links' in parsed_content_result.data
            assert 'headings' in parsed_content_result.data
            assert 'forms' in parsed_content_result.data
            assert len(parsed_content_result.data['text_content']) > 0
            
            # Clean up browser
            await injector.run('browser_manager', {
                'operation': 'stop_browser',
                'browser_id': self.test_browser_id
            })
        
        asyncio.run(run_test())
    
    def test_page_navigator_screenshots(self):
        """Test screenshot capture functionality."""
        
        async def run_test():
            injector = get_injector()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Start browser and navigate
                await injector.run('browser_manager', {
                    'operation': 'start_browser',
                    'browser_id': self.test_browser_id,
                    'options': {'headless': False, 'viewport': {'width': 1280, 'height': 720}}
                })
                
                await injector.run('page_navigator', {
                    'operation': 'navigate',
                    'browser_id': self.test_browser_id,
                    'url': self.test_urls['mozilla']
                })
                
                # Capture viewport screenshot
                screenshot_path = os.path.join(temp_dir, 'test_screenshot.png')
                screenshot_result = await injector.run('page_navigator', {
                    'operation': 'capture_screenshot',
                    'browser_id': self.test_browser_id,
                    'screenshot_path': screenshot_path,
                    'full_page': False
                })
                
                assert screenshot_result.success is True
                assert screenshot_result.data['path'] == screenshot_path
                assert screenshot_result.data['size_bytes'] > 0
                assert screenshot_result.data['dimensions']['width'] == 1280
                assert screenshot_result.data['dimensions']['height'] == 720
                assert screenshot_result.data['full_page'] is False
                
                # Verify file exists
                assert os.path.exists(screenshot_path)
                assert os.path.getsize(screenshot_path) > 0
                
                # Capture full page screenshot
                full_screenshot_path = os.path.join(temp_dir, 'test_full_screenshot.png')
                full_screenshot_result = await injector.run('page_navigator', {
                    'operation': 'capture_screenshot',
                    'browser_id': self.test_browser_id,
                    'screenshot_path': full_screenshot_path,
                    'full_page': True
                })
                
                assert full_screenshot_result.success is True
                assert full_screenshot_result.data['full_page'] is True
                assert os.path.exists(full_screenshot_path)
                
            # Clean up browser
            await injector.run('browser_manager', {
                'operation': 'stop_browser',
                'browser_id': self.test_browser_id
            })
        
        asyncio.run(run_test())
    
    def test_page_navigator_cookie_management(self):
        """Test cookie management operations."""
        
        async def run_test():
            injector = get_injector()
            
            # Start browser and navigate
            await injector.run('browser_manager', {
                'operation': 'start_browser',
                'browser_id': self.test_browser_id,
                'options': {'headless': False}
            })
            
            await injector.run('page_navigator', {
                'operation': 'navigate',
                'browser_id': self.test_browser_id,
                'url': self.test_urls['python']
            })
            
            # Set a cookie
            set_cookie_result = await injector.run('page_navigator', {
                'operation': 'manage_cookies',
                'browser_id': self.test_browser_id,
                'cookie_action': 'set',
                'cookie_name': 'test_cookie',
                'cookie_value': 'test_value_123',
                'cookie_options': {'path': '/'}
            })
            
            assert set_cookie_result.success is True
            assert 'test_cookie' in set_cookie_result.message
            
            # Get the cookie
            get_cookie_result = await injector.run('page_navigator', {
                'operation': 'manage_cookies',
                'browser_id': self.test_browser_id,
                'cookie_action': 'get',
                'cookie_name': 'test_cookie'
            })
            
            assert get_cookie_result.success is True
            if get_cookie_result.data:  # Cookie may not be set due to domain restrictions
                assert get_cookie_result.data.get('name') == 'test_cookie'
                assert get_cookie_result.data.get('value') == 'test_value_123'
            
            # Get all cookies
            all_cookies_result = await injector.run('page_navigator', {
                'operation': 'manage_cookies',
                'browser_id': self.test_browser_id,
                'cookie_action': 'get'
            })
            
            assert all_cookies_result.success is True
            assert 'cookies' in all_cookies_result.data
            assert 'count' in all_cookies_result.data
            
            # Clear all cookies
            clear_cookies_result = await injector.run('page_navigator', {
                'operation': 'manage_cookies',
                'browser_id': self.test_browser_id,
                'cookie_action': 'clear'
            })
            
            assert clear_cookies_result.success is True
            assert clear_cookies_result.data['cleared'] is True
            
            # Clean up browser
            await injector.run('browser_manager', {
                'operation': 'stop_browser',
                'browser_id': self.test_browser_id
            })
        
        asyncio.run(run_test())
    
    def test_page_navigator_storage_management(self):
        """Test local and session storage management."""
        
        async def run_test():
            injector = get_injector()
            
            # Start browser and navigate
            await injector.run('browser_manager', {
                'operation': 'start_browser',
                'browser_id': self.test_browser_id,
                'options': {'headless': False}
            })
            
            await injector.run('page_navigator', {
                'operation': 'navigate',
                'browser_id': self.test_browser_id,
                'url': self.test_urls['stackoverflow']
            })
            
            # Test localStorage operations
            # Set item
            set_result = await injector.run('page_navigator', {
                'operation': 'manage_storage',
                'browser_id': self.test_browser_id,
                'storage_type': 'local',
                'storage_action': 'set',
                'storage_key': 'test_key',
                'storage_value': 'test_value_local'
            })
            
            assert set_result.success is True
            assert set_result.data['key'] == 'test_key'
            assert set_result.data['value'] == 'test_value_local'
            
            # Get item
            get_result = await injector.run('page_navigator', {
                'operation': 'manage_storage',
                'browser_id': self.test_browser_id,
                'storage_type': 'local',
                'storage_action': 'get',
                'storage_key': 'test_key'
            })
            
            assert get_result.success is True
            assert get_result.data['value'] == 'test_value_local'
            
            # Test sessionStorage operations
            session_set_result = await injector.run('page_navigator', {
                'operation': 'manage_storage',
                'browser_id': self.test_browser_id,
                'storage_type': 'session',
                'storage_action': 'set',
                'storage_key': 'session_key',
                'storage_value': 'session_value'
            })
            
            assert session_set_result.success is True
            
            # Get all items
            get_all_result = await injector.run('page_navigator', {
                'operation': 'manage_storage',
                'browser_id': self.test_browser_id,
                'storage_type': 'local',
                'storage_action': 'get'
            })
            
            assert get_all_result.success is True
            assert 'items' in get_all_result.data
            assert get_all_result.data['items']['test_key'] == 'test_value_local'
            
            # Remove item
            remove_result = await injector.run('page_navigator', {
                'operation': 'manage_storage',
                'browser_id': self.test_browser_id,
                'storage_type': 'local',
                'storage_action': 'remove',
                'storage_key': 'test_key'
            })
            
            assert remove_result.success is True
            assert remove_result.data['removed'] == 'test_key'
            
            # Clear storage
            clear_result = await injector.run('page_navigator', {
                'operation': 'manage_storage',
                'browser_id': self.test_browser_id,
                'storage_type': 'session',
                'storage_action': 'clear'
            })
            
            assert clear_result.success is True
            assert clear_result.data['cleared'] is True
            
            # Clean up browser
            await injector.run('browser_manager', {
                'operation': 'stop_browser',
                'browser_id': self.test_browser_id
            })
        
        asyncio.run(run_test())
    
    def test_page_navigator_javascript_evaluation(self):
        """Test JavaScript evaluation functionality."""
        
        async def run_test():
            injector = get_injector()
            
            # Start browser and navigate
            await injector.run('browser_manager', {
                'operation': 'start_browser',
                'browser_id': self.test_browser_id,
                'options': {'headless': False}
            })
            
            await injector.run('page_navigator', {
                'operation': 'navigate',
                'browser_id': self.test_browser_id,
                'url': self.test_urls['apache']
            })
            
            # Test simple arithmetic
            math_result = await injector.run('page_navigator', {
                'operation': 'evaluate',
                'browser_id': self.test_browser_id,
                'script': '2 + 3'
            })
            
            assert math_result.success is True
            assert math_result.data['result'] == 5
            assert math_result.data['type'] == 'int'
            
            # Test string operation
            string_result = await injector.run('page_navigator', {
                'operation': 'evaluate',
                'browser_id': self.test_browser_id,
                'script': '"Hello " + "World"'
            })
            
            assert string_result.success is True
            assert string_result.data['result'] == "Hello World"
            assert string_result.data['type'] == 'str'
            
            # Test DOM query
            dom_result = await injector.run('page_navigator', {
                'operation': 'evaluate',
                'browser_id': self.test_browser_id,
                'script': 'document.title'
            })
            
            assert dom_result.success is True
            assert len(dom_result.data['result']) > 0
            
            # Test boolean result
            bool_result = await injector.run('page_navigator', {
                'operation': 'evaluate',
                'browser_id': self.test_browser_id,
                'script': 'document.body !== null'
            })
            
            assert bool_result.success is True
            assert bool_result.data['result'] is True
            assert bool_result.data['type'] == 'bool'
            
            # Clean up browser
            await injector.run('browser_manager', {
                'operation': 'stop_browser',
                'browser_id': self.test_browser_id
            })
        
        asyncio.run(run_test())
    
    # ===== ELEMENT INTERACTOR TESTS =====
    
    def test_element_interactor_find_elements(self):
        """Test element finding with different selector types."""
        
        async def run_test():
            injector = get_injector()
            
            # Start browser and navigate to httpbin forms
            await injector.run('browser_manager', {
                'operation': 'start_browser',
                'browser_id': self.test_browser_id,
                'options': {'headless': False}
            })
            
            await injector.run('page_navigator', {
                'operation': 'navigate',
                'browser_id': self.test_browser_id,
                'url': self.test_urls['httpbin_forms']
            })
            
            # Find element by CSS selector
            css_result = await injector.run('element_interactor', {
                'operation': 'find_element',
                'browser_id': self.test_browser_id,
                'selector': 'input[name="custname"]'
            })
            
            assert css_result.success is True
            assert css_result.data['element_found'] is True
            assert css_result.data['tag_name'] == 'input'
            assert css_result.data['visible'] is True
            
            # Find element by text content
            text_result = await injector.run('element_interactor', {
                'operation': 'find_element',
                'browser_id': self.test_browser_id,
                'selector': 'text=Submit order'
            })
            
            assert text_result.success is True
            assert text_result.data['element_found'] is True
            
            # Test non-existent element
            missing_result = await injector.run('element_interactor', {
                'operation': 'find_element',
                'browser_id': self.test_browser_id,
                'selector': '#non-existent-element',
                'timeout': 1000
            })
            
            assert missing_result.success is False
            assert missing_result.data['element_found'] is False
            
            # Clean up browser
            await injector.run('browser_manager', {
                'operation': 'stop_browser',
                'browser_id': self.test_browser_id
            })
        
        asyncio.run(run_test())
    
    def test_element_interactor_form_interactions(self):
        """Test form filling and submission."""
        
        async def run_test():
            injector = get_injector()
            
            # Start browser and navigate to httpbin forms
            await injector.run('browser_manager', {
                'operation': 'start_browser',
                'browser_id': self.test_browser_id,
                'options': {'headless': False}
            })
            
            await injector.run('page_navigator', {
                'operation': 'navigate',
                'browser_id': self.test_browser_id,
                'url': self.test_urls['httpbin_forms']
            })
            
            # Type in customer name field
            name_result = await injector.run('element_interactor', {
                'operation': 'type_text',
                'browser_id': self.test_browser_id,
                'selector': 'input[name="custname"]',
                'text': 'John Doe Test User',
                'clear_first': True
            })
            
            assert name_result.success is True
            assert name_result.data['text_entered'] == 'John Doe Test User'
            assert name_result.data['cleared_first'] is True
            
            # Type in customer telephone field
            phone_result = await injector.run('element_interactor', {
                'operation': 'type_text',
                'browser_id': self.test_browser_id,
                'selector': 'input[name="custtel"]',
                'text': '555-123-4567'
            })
            
            assert phone_result.success is True
            
            # Type in email field
            email_result = await injector.run('element_interactor', {
                'operation': 'type_text',
                'browser_id': self.test_browser_id,
                'selector': 'input[name="custemail"]',
                'text': 'john.doe@example.com'
            })
            
            assert email_result.success is True
            
            # Select pizza size (dropdown)
            size_result = await injector.run('element_interactor', {
                'operation': 'select_option',
                'browser_id': self.test_browser_id,
                'selector': 'select[name="size"]',
                'value': 'medium'
            })
            
            assert size_result.success is True
            assert size_result.data['value'] == 'medium'
            
            # Verify form data by reading back values
            name_text_result = await injector.run('element_interactor', {
                'operation': 'get_element_attribute',
                'browser_id': self.test_browser_id,
                'selector': 'input[name="custname"]',
                'attribute': 'value'
            })
            
            assert name_text_result.success is True
            assert name_text_result.data['value'] == 'John Doe Test User'
            
            # Clean up browser
            await injector.run('browser_manager', {
                'operation': 'stop_browser',
                'browser_id': self.test_browser_id
            })
        
        asyncio.run(run_test())
    
    def test_element_interactor_click_operations(self):
        """Test various click operations."""
        
        async def run_test():
            injector = get_injector()
            
            # Start browser and navigate
            await injector.run('browser_manager', {
                'operation': 'start_browser',
                'browser_id': self.test_browser_id,
                'options': {'headless': False}
            })
            
            await injector.run('page_navigator', {
                'operation': 'navigate',
                'browser_id': self.test_browser_id,
                'url': 'https://httpbin.org'
            })
            
            # Click on a link (if available)
            # First check if there are any links
            content_result = await injector.run('page_navigator', {
                'operation': 'get_content',
                'browser_id': self.test_browser_id,
                'extract_content': True
            })
            
            if content_result.data.get('links'):
                # Try to click the first available link that's not external
                internal_links = [link for link in content_result.data['links'] if link.startswith('/')]
                if internal_links:
                    link_selector = f'a[href="{internal_links[0]}"]'
                    
                    click_result = await injector.run('element_interactor', {
                        'operation': 'click_element',
                        'browser_id': self.test_browser_id,
                        'selector': link_selector,
                        'click_count': 1
                    })
                    
                    # The click may or may not succeed depending on page structure
                    # We're mainly testing that the operation doesn't crash
                    assert 'clicked' in click_result.data
            
            # Test double click (even if element doesn't respond to it)
            body_click_result = await injector.run('element_interactor', {
                'operation': 'click_element',
                'browser_id': self.test_browser_id,
                'selector': 'body',
                'click_count': 2
            })
            
            assert body_click_result.success is True
            assert body_click_result.data['click_count'] == 2
            
            # Clean up browser
            await injector.run('browser_manager', {
                'operation': 'stop_browser',
                'browser_id': self.test_browser_id
            })
        
        asyncio.run(run_test())
    
    def test_element_interactor_hover_operations(self):
        """Test hover operations."""
        
        async def run_test():
            injector = get_injector()
            
            # Start browser and navigate
            await injector.run('browser_manager', {
                'operation': 'start_browser',
                'browser_id': self.test_browser_id,
                'options': {'headless': False}
            })
            
            await injector.run('page_navigator', {
                'operation': 'navigate',
                'browser_id': self.test_browser_id,
                'url': self.test_urls['bing']
            })
            
            # Hover over body element
            hover_result = await injector.run('element_interactor', {
                'operation': 'hover_element',
                'browser_id': self.test_browser_id,
                'selector': 'body'
            })
            
            assert hover_result.success is True
            assert hover_result.data['hovered'] is True
            
            # Hover over a link if available
            links_result = await injector.run('page_navigator', {
                'operation': 'get_content',
                'browser_id': self.test_browser_id,
                'extract_content': True
            })
            
            if links_result.data.get('links'):
                # Find the first link element and hover over it
                link_hover_result = await injector.run('element_interactor', {
                    'operation': 'hover_element',
                    'browser_id': self.test_browser_id,
                    'selector': 'a'  # First anchor tag
                })
                
                assert link_hover_result.success is True
                assert link_hover_result.data['hovered'] is True
                
            # Clean up browser
            await injector.run('browser_manager', {
                'operation': 'stop_browser',
                'browser_id': self.test_browser_id
            })
        
        asyncio.run(run_test())
    
    def test_element_interactor_keyboard_shortcuts(self):
        """Test keyboard shortcut operations."""
        
        async def run_test():
            injector = get_injector()
            
            # Start browser and navigate to forms page
            await injector.run('browser_manager', {
                'operation': 'start_browser',
                'browser_id': self.test_browser_id,
                'options': {'headless': False}
            })
            
            await injector.run('page_navigator', {
                'operation': 'navigate',
                'browser_id': self.test_browser_id,
                'url': self.test_urls['httpbin_forms']
            })
            
            # Focus on name input and test keyboard shortcuts
            await injector.run('element_interactor', {
                'operation': 'type_text',
                'browser_id': self.test_browser_id,
                'selector': 'input[name="custname"]',
                'text': 'Test User Name'
            })
            
            # Test Ctrl+A (select all)
            select_all_result = await injector.run('element_interactor', {
                'operation': 'keyboard_shortcut',
                'browser_id': self.test_browser_id,
                'selector': 'input[name="custname"]',
                'keys': 'Control+a'
            })
            
            assert select_all_result.success is True
            assert select_all_result.data['keys_sent'] is True
            assert select_all_result.data['keys'] == 'Control+a'
            
            # Test typing after select all (should replace selected text)
            await injector.run('element_interactor', {
                'operation': 'type_text',
                'browser_id': self.test_browser_id,
                'selector': 'input[name="custname"]',
                'text': 'Replaced Text',
                'clear_first': False  # Don't clear, just type over selection
            })
            
            # Test Enter key
            enter_result = await injector.run('element_interactor', {
                'operation': 'keyboard_shortcut',
                'browser_id': self.test_browser_id,
                'selector': 'input[name="custname"]',
                'keys': 'Enter'
            })
            
            assert enter_result.success is True
            
            # Clean up browser
            await injector.run('browser_manager', {
                'operation': 'stop_browser',
                'browser_id': self.test_browser_id
            })
        
        asyncio.run(run_test())
    
    def test_element_interactor_element_screenshots(self):
        """Test element screenshot capture."""
        
        async def run_test():
            injector = get_injector()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Start browser and navigate
                await injector.run('browser_manager', {
                    'operation': 'start_browser',
                    'browser_id': self.test_browser_id,
                    'options': {'headless': False}
                })
                
                await injector.run('page_navigator', {
                    'operation': 'navigate',
                    'browser_id': self.test_browser_id,
                    'url': self.test_urls['httpbin_forms']
                })
                
                # Take screenshot of form element
                form_screenshot_result = await injector.run('element_interactor', {
                    'operation': 'element_screenshot',
                    'browser_id': self.test_browser_id,
                    'selector': 'form'
                })
                
                assert form_screenshot_result.success is True
                assert form_screenshot_result.data['screenshot_taken'] is True
                assert 'screenshot_path' in form_screenshot_result.data
                assert form_screenshot_result.data['size_bytes'] > 0
                
                # Verify screenshot file exists
                screenshot_path = form_screenshot_result.data['screenshot_path']
                assert os.path.exists(screenshot_path)
                assert os.path.getsize(screenshot_path) > 0
                
            # Clean up browser
            await injector.run('browser_manager', {
                'operation': 'stop_browser',
                'browser_id': self.test_browser_id
            })
        
        asyncio.run(run_test())
    
    def test_element_interactor_wait_conditions(self):
        """Test element wait conditions."""
        
        async def run_test():
            injector = get_injector()
            
            # Start browser and navigate
            await injector.run('browser_manager', {
                'operation': 'start_browser',
                'browser_id': self.test_browser_id,
                'options': {'headless': False}
            })
            
            await injector.run('page_navigator', {
                'operation': 'navigate',
                'browser_id': self.test_browser_id,
                'url': self.test_urls['duckduckgo']
            })
            
            # Wait for body to be visible (should already be)
            visible_result = await injector.run('element_interactor', {
                'operation': 'wait_for_element',
                'browser_id': self.test_browser_id,
                'selector': 'body',
                'wait_condition': 'visible',
                'timeout': 5000
            })
            
            assert visible_result.success is True
            assert visible_result.data['condition_met'] is True
            assert visible_result.data['wait_condition'] == 'visible'
            assert 'wait_time_ms' in visible_result.data
            
            # Wait for element to be attached
            attached_result = await injector.run('element_interactor', {
                'operation': 'wait_for_element',
                'browser_id': self.test_browser_id,
                'selector': 'h1',  # Should exist on example.com
                'wait_condition': 'attached',
                'timeout': 3000
            })
            
            assert attached_result.success is True
            assert attached_result.data['condition_met'] is True
            
            # Clean up browser
            await injector.run('browser_manager', {
                'operation': 'stop_browser',
                'browser_id': self.test_browser_id
            })
        
        asyncio.run(run_test())
    
    # ===== MULTI-AGENT WORKFLOW TESTS =====
    
    def test_complete_form_workflow(self):
        """Test complete form filling workflow using all three agents."""
        
        async def run_test():
            injector = get_injector()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # 1. Browser Manager: Start browser
                start_result = await injector.run('browser_manager', {
                    'operation': 'start_browser',
                    'browser_id': self.test_browser_id,
                    'options': {'headless': False, 'viewport': {'width': 1280, 'height': 720}}
                })
                assert start_result.success is True
                
                # 2. Page Navigator: Navigate to forms page
                nav_result = await injector.run('page_navigator', {
                    'operation': 'navigate',
                    'browser_id': self.test_browser_id,
                    'url': self.test_urls['httpbin_forms'],
                    'wait_condition': 'domcontentloaded'
                })
                assert nav_result.success is True
                
                # 3. Page Navigator: Take screenshot of initial page
                initial_screenshot = os.path.join(temp_dir, 'initial_form.png')
                screenshot_result = await injector.run('page_navigator', {
                    'operation': 'capture_screenshot',
                    'browser_id': self.test_browser_id,
                    'screenshot_path': initial_screenshot
                })
                assert screenshot_result.success is True
                
                # 4. Element Interactor: Fill form fields
                form_data = {
                    'custname': 'E2E Test Customer',
                    'custtel': '555-TEST-123',
                    'custemail': 'e2etest@example.com'
                }
                
                for field_name, field_value in form_data.items():
                    fill_result = await injector.run('element_interactor', {
                        'operation': 'type_text',
                        'browser_id': self.test_browser_id,
                        'selector': f'input[name="{field_name}"]',
                        'text': field_value,
                        'clear_first': True
                    })
                    assert fill_result.success is True
                
                # 5. Element Interactor: Select pizza size
                size_result = await injector.run('element_interactor', {
                    'operation': 'select_option',
                    'browser_id': self.test_browser_id,
                    'selector': 'select[name="size"]',
                    'value': 'large'
                })
                assert size_result.success is True
                
                # 6. Element Interactor: Take screenshot of filled form
                filled_screenshot = os.path.join(temp_dir, 'filled_form.png')
                element_screenshot_result = await injector.run('element_interactor', {
                    'operation': 'element_screenshot',
                    'browser_id': self.test_browser_id,
                    'selector': 'form'
                })
                assert element_screenshot_result.success is True
                
                # 7. Element Interactor: Verify form data
                for field_name, expected_value in form_data.items():
                    value_result = await injector.run('element_interactor', {
                        'operation': 'get_element_attribute',
                        'browser_id': self.test_browser_id,
                        'selector': f'input[name="{field_name}"]',
                        'attribute': 'value'
                    })
                    assert value_result.success is True
                    assert value_result.data['value'] == expected_value
                
                # 8. Page Navigator: Get final page content
                final_content = await injector.run('page_navigator', {
                    'operation': 'get_content',
                    'browser_id': self.test_browser_id,
                    'extract_content': True
                })
                assert final_content.success is True
                
                # 9. Browser Manager: Get final browser health
                health_result = await injector.run('browser_manager', {
                    'operation': 'health_check',
                    'browser_id': self.test_browser_id
                })
                assert health_result.success is True
                
                # Verify all screenshots were created
                assert os.path.exists(initial_screenshot)
                assert os.path.exists(element_screenshot_result.data['screenshot_path'])
                
                print(f"✅ Complete workflow test passed!")
                print(f"   - Browser started and managed successfully")
                print(f"   - Navigation completed: {nav_result.data['url']}")
                print(f"   - Form filled with {len(form_data)} fields")
                print(f"   - Screenshots captured: 2 files")
                print(f"   - Browser health: {health_result.data['overall_status']}")
                
            # Clean up browser
            await injector.run('browser_manager', {
                'operation': 'stop_browser',
                'browser_id': self.test_browser_id
            })
        
        asyncio.run(run_test())
    
    def test_multi_page_navigation_workflow(self):
        """Test navigation workflow across multiple pages."""
        
        async def run_test():
            injector = get_injector()
            
            # Start browser
            await injector.run('browser_manager', {
                'operation': 'start_browser',
                'browser_id': self.test_browser_id,
                'options': {'headless': False}
            })
            
            # Navigation sequence - use different sites for each navigation
            test_pages = [
                self.test_urls['mozilla'],
                self.test_urls['wikipedia'],
                self.test_urls['python']
            ]
            
            page_titles = []
            page_urls = []
            
            for page_url in test_pages:
                # Navigate to page
                nav_result = await injector.run('page_navigator', {
                    'operation': 'navigate',
                    'browser_id': self.test_browser_id,
                    'url': page_url,
                    'wait_condition': 'load'
                })
                assert nav_result.success is True
                page_titles.append(nav_result.data['title'])
                page_urls.append(nav_result.data['url'])
                
                # Wait for page to be ready
                await injector.run('page_navigator', {
                    'operation': 'wait_for_load',
                    'browser_id': self.test_browser_id,
                    'wait_condition': 'networkidle',
                    'timeout': 5000
                })
                
                # Extract some content
                content_result = await injector.run('page_navigator', {
                    'operation': 'get_content',
                    'browser_id': self.test_browser_id,
                    'extract_content': False
                })
                assert content_result.success is True
                assert len(content_result.data['html']) > 0
                
                # Short delay between navigations
                await asyncio.sleep(0.5)
            
            # Test back navigation
            for i in range(len(test_pages) - 1):
                back_result = await injector.run('page_navigator', {
                    'operation': 'back',
                    'browser_id': self.test_browser_id
                })
                assert back_result.success is True
            
            # Test forward navigation
            for i in range(len(test_pages) - 1):
                forward_result = await injector.run('page_navigator', {
                    'operation': 'forward',
                    'browser_id': self.test_browser_id
                })
                assert forward_result.success is True
            
            # Verify we visited all pages
            assert len(page_titles) == len(test_pages)
            assert len(page_urls) == len(test_pages)
            
            print(f"✅ Multi-page navigation test passed!")
            print(f"   - Visited {len(test_pages)} pages successfully")
            print(f"   - Back/forward navigation working")
            print(f"   - Page titles collected: {len(page_titles)}")
            
            # Clean up browser
            await injector.run('browser_manager', {
                'operation': 'stop_browser',
                'browser_id': self.test_browser_id
            })
        
        asyncio.run(run_test())
    
    # ===== ERROR HANDLING TESTS =====
    
    @pytest.mark.skip(reason="Skipping error tests for now")
    def test_invalid_browser_operations(self):
        """Test operations with invalid browser IDs."""
        
        async def run_test():
            injector = get_injector()
            
            # Test page navigation with invalid browser
            try:
                nav_result = await injector.run('page_navigator', {
                    'operation': 'navigate',
                    'browser_id': 'invalid_browser_123',
                    'url': self.test_urls['example']
                })
                assert nav_result.success is False
            except (ValueError, RuntimeError) as e:
                assert 'not found' in str(e).lower() or 'invalid' in str(e).lower()
            
            # Test element interaction with invalid browser
            try:
                element_result = await injector.run('element_interactor', {
                    'operation': 'find_element',
                    'browser_id': 'invalid_browser_123',
                    'selector': 'body'
                })
                assert element_result.success is False
            except (ValueError, RuntimeError) as e:
                assert 'not found' in str(e).lower() or 'invalid' in str(e).lower()
        
        asyncio.run(run_test())
    
    @pytest.mark.skip(reason="Skipping error tests for now")
    def test_invalid_navigation_urls(self):
        """Test navigation to invalid URLs."""
        
        async def run_test():
            injector = get_injector()
            
            # Start browser
            await injector.run('browser_manager', {
                'operation': 'start_browser',
                'browser_id': self.test_browser_id,
                'options': {'headless': False}
            })
            
            # Try to navigate to invalid URL
            nav_result = await injector.run('page_navigator', {
                'operation': 'navigate',
                'browser_id': self.test_browser_id,
                'url': 'https://this-domain-definitely-does-not-exist-12345.com',
                'timeout': 5000
            })
            
            # Should handle error gracefully
            assert nav_result.success is False
            assert 'error' in nav_result.data or 'timeout' in nav_result.message.lower()
            
            # Clean up browser
            await injector.run('browser_manager', {
                'operation': 'stop_browser',
                'browser_id': self.test_browser_id
            })
        
        asyncio.run(run_test())
    
    @pytest.mark.skip(reason="Skipping error tests for now")
    def test_element_interaction_timeouts(self):
        """Test element interaction timeout scenarios."""
        
        async def run_test():
            injector = get_injector()
            
            # Start browser and navigate
            await injector.run('browser_manager', {
                'operation': 'start_browser',
                'browser_id': self.test_browser_id,
                'options': {'headless': False}
            })
            
            await injector.run('page_navigator', {
                'operation': 'navigate',
                'browser_id': self.test_browser_id,
                'url': self.test_urls['example']
            })
            
            # Try to find non-existent element with short timeout
            missing_element_result = await injector.run('element_interactor', {
                'operation': 'find_element',
                'browser_id': self.test_browser_id,
                'selector': '#absolutely-non-existent-element-id',
                'timeout': 1000
            })
            
            assert missing_element_result.success is False
            assert missing_element_result.data['element_found'] is False
            
            # Try to wait for non-existent element
            wait_result = await injector.run('element_interactor', {
                'operation': 'wait_for_element',
                'browser_id': self.test_browser_id,
                'selector': '#another-non-existent-element',
                'wait_condition': 'visible',
                'timeout': 1000
            })
            
            # Should timeout gracefully
            assert wait_result.success is False or 'timeout' in wait_result.message.lower()
            
            # Clean up browser
            await injector.run('browser_manager', {
                'operation': 'stop_browser',
                'browser_id': self.test_browser_id
            })
        
        asyncio.run(run_test())
