"""
Tests for HTTP toolkit.

This module tests all functionality of the HTTP client toolkit
including various HTTP methods, authentication, and error handling.
"""

import json
import asyncio
from unittest.mock import patch, MagicMock, Mock
import urllib.request
import urllib.error
import pytest

from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry


class TestHttp:
    """Test suite for HTTP toolkit."""
    
    def setup_method(self):
        """Clear registry and injector before each test."""
        AgenToolRegistry.clear()
        get_injector().clear()
        
        # Import and create the agents
        from agentoolkit.network.http import create_http_agent
        from agentoolkit.auth.session import create_session_agent
        
        # Create agents
        http_agent = create_http_agent()
        session_agent = create_session_agent()
    
    def test_http_get_request(self):
        """Test basic GET request."""
        
        async def run_test():
            injector = get_injector()
            
            # Mock urllib.request.urlopen
            mock_response = MagicMock()
            mock_response.read.return_value = b'{"result": "success", "data": [1, 2, 3]}'
            mock_response.getcode.return_value = 200
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.geturl.return_value = "https://api.example.com/data"
            
            with patch('urllib.request.urlopen', return_value=mock_response):
                result = await injector.run('http', {
                    "operation": "get",
                    "url": "https://api.example.com/data",
                    "timeout": 30,
                    "verify_ssl": True,
                    "follow_redirects": True
                })
                
                if hasattr(result, 'output'):
                    data = json.loads(result.output)
                else:
                    data = result
                
                # No longer checking success field - function now throws exceptions on failure
                assert data["operation"] == "get"
                assert data["data"]["status_code"] == 200
                assert data["data"]["body"]["result"] == "success"
                assert data["data"]["body"]["data"] == [1, 2, 3]
        
        asyncio.run(run_test())
    
    def test_http_get_with_params(self):
        """Test GET request with query parameters."""
        
        async def run_test():
            injector = get_injector()
            
            # Mock urllib.request.urlopen
            mock_response = MagicMock()
            mock_response.read.return_value = b'{"filtered": true}'
            mock_response.getcode.return_value = 200
            mock_response.headers = {}
            mock_response.geturl.return_value = "https://api.example.com/data?page=1&limit=10"
            
            with patch('urllib.request.urlopen') as mock_urlopen:
                mock_urlopen.return_value = mock_response
                
                result = await injector.run('http', {
                    "operation": "get",
                    "url": "https://api.example.com/data",
                    "params": {"page": 1, "limit": 10},
                    "timeout": 30,
                    "verify_ssl": True,
                    "follow_redirects": True
                })
                
                # Verify the URL was constructed correctly
                call_args = mock_urlopen.call_args
                request = call_args[0][0]
                assert "page=1" in request.full_url
                assert "limit=10" in request.full_url
                
                if hasattr(result, 'output'):
                    data = json.loads(result.output)
                else:
                    data = result
                
                # No longer checking success field - function now throws exceptions on failure
        
        asyncio.run(run_test())
    
    def test_http_post_json(self):
        """Test POST request with JSON data."""
        
        async def run_test():
            injector = get_injector()
            
            # Mock urllib.request.urlopen
            mock_response = MagicMock()
            mock_response.read.return_value = b'{"id": 123, "created": true}'
            mock_response.getcode.return_value = 201
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.geturl.return_value = "https://api.example.com/users"
            
            with patch('urllib.request.urlopen') as mock_urlopen:
                mock_urlopen.return_value = mock_response
                
                result = await injector.run('http', {
                    "operation": "post",
                    "url": "https://api.example.com/users",
                    "json_data": {"name": "John Doe", "email": "john@example.com"},
                    "timeout": 30,
                    "verify_ssl": True,
                    "follow_redirects": True
                })
                
                # Verify the request body
                call_args = mock_urlopen.call_args
                request = call_args[0][0]
                assert request.data is not None
                body = json.loads(request.data.decode())
                assert body["name"] == "John Doe"
                assert body["email"] == "john@example.com"
                assert request.headers["Content-type"] == "application/json"
                
                if hasattr(result, 'output'):
                    data = json.loads(result.output)
                else:
                    data = result
                
                # No longer checking success field - function now throws exceptions on failure
                assert data["data"]["status_code"] == 201
                assert data["data"]["body"]["id"] == 123
        
        asyncio.run(run_test())
    
    def test_http_bearer_auth(self):
        """Test request with bearer token authentication."""
        
        async def run_test():
            injector = get_injector()
            
            # Mock urllib.request.urlopen
            mock_response = MagicMock()
            mock_response.read.return_value = b'{"authenticated": true}'
            mock_response.getcode.return_value = 200
            mock_response.headers = {}
            mock_response.geturl.return_value = "https://api.example.com/protected"
            
            with patch('urllib.request.urlopen') as mock_urlopen:
                mock_urlopen.return_value = mock_response
                
                result = await injector.run('http', {
                    "operation": "get",
                    "url": "https://api.example.com/protected",
                    "auth_type": "bearer",
                    "auth_token": "secret_token_123",
                    "timeout": 30,
                    "verify_ssl": True,
                    "follow_redirects": True
                })
                
                # Verify authorization header
                call_args = mock_urlopen.call_args
                request = call_args[0][0]
                assert request.headers["Authorization"] == "Bearer secret_token_123"
                
                if hasattr(result, 'output'):
                    data = json.loads(result.output)
                else:
                    data = result
                
                # No longer checking success field - function now throws exceptions on failure
        
        asyncio.run(run_test())
    
    def test_http_basic_auth(self):
        """Test request with basic authentication."""
        
        async def run_test():
            injector = get_injector()
            
            # Mock urllib.request.urlopen
            mock_response = MagicMock()
            mock_response.read.return_value = b'{"authenticated": true}'
            mock_response.getcode.return_value = 200
            mock_response.headers = {}
            mock_response.geturl.return_value = "https://api.example.com/protected"
            
            with patch('urllib.request.urlopen') as mock_urlopen:
                mock_urlopen.return_value = mock_response
                
                result = await injector.run('http', {
                    "operation": "get",
                    "url": "https://api.example.com/protected",
                    "auth_type": "basic",
                    "username": "admin",
                    "password": "secret123",
                    "timeout": 30,
                    "verify_ssl": True,
                    "follow_redirects": True
                })
                
                # Verify authorization header
                call_args = mock_urlopen.call_args
                request = call_args[0][0]
                # Basic auth should be base64 encoded "admin:secret123"
                import base64
                expected = "Basic " + base64.b64encode(b"admin:secret123").decode()
                assert request.headers["Authorization"] == expected
                
                if hasattr(result, 'output'):
                    data = json.loads(result.output)
                else:
                    data = result
                
                # No longer checking success field - function now throws exceptions on failure
        
        asyncio.run(run_test())
    
    def test_http_put_request(self):
        """Test PUT request."""
        
        async def run_test():
            injector = get_injector()
            
            # Mock urllib.request.urlopen
            mock_response = MagicMock()
            mock_response.read.return_value = b'{"updated": true}'
            mock_response.getcode.return_value = 200
            mock_response.headers = {}
            mock_response.geturl.return_value = "https://api.example.com/users/123"
            
            with patch('urllib.request.urlopen') as mock_urlopen:
                mock_urlopen.return_value = mock_response
                
                result = await injector.run('http', {
                    "operation": "put",
                    "url": "https://api.example.com/users/123",
                    "json_data": {"name": "Jane Doe"},
                    "timeout": 30,
                    "verify_ssl": True,
                    "follow_redirects": True
                })
                
                # Verify request method
                call_args = mock_urlopen.call_args
                request = call_args[0][0]
                assert request.get_method() == "PUT"
                
                if hasattr(result, 'output'):
                    data = json.loads(result.output)
                else:
                    data = result
                
                # No longer checking success field - function now throws exceptions on failure
                assert data["operation"] == "put"
        
        asyncio.run(run_test())
    
    def test_http_delete_request(self):
        """Test DELETE request."""
        
        async def run_test():
            injector = get_injector()
            
            # Mock urllib.request.urlopen
            mock_response = MagicMock()
            mock_response.read.return_value = b''
            mock_response.getcode.return_value = 204
            mock_response.headers = {}
            mock_response.geturl.return_value = "https://api.example.com/users/123"
            
            with patch('urllib.request.urlopen') as mock_urlopen:
                mock_urlopen.return_value = mock_response
                
                result = await injector.run('http', {
                    "operation": "delete",
                    "url": "https://api.example.com/users/123",
                    "timeout": 30,
                    "verify_ssl": True,
                    "follow_redirects": True
                })
                
                # Verify request method
                call_args = mock_urlopen.call_args
                request = call_args[0][0]
                assert request.get_method() == "DELETE"
                
                if hasattr(result, 'output'):
                    data = json.loads(result.output)
                else:
                    data = result
                
                # No longer checking success field - function now throws exceptions on failure
                assert data["operation"] == "delete"
                assert data["data"]["status_code"] == 204
        
        asyncio.run(run_test())
    
    def test_http_patch_request(self):
        """Test PATCH request."""
        
        async def run_test():
            injector = get_injector()
            
            # Mock urllib.request.urlopen
            mock_response = MagicMock()
            mock_response.read.return_value = b'{"patched": true}'
            mock_response.getcode.return_value = 200
            mock_response.headers = {}
            mock_response.geturl.return_value = "https://api.example.com/users/123"
            
            with patch('urllib.request.urlopen') as mock_urlopen:
                mock_urlopen.return_value = mock_response
                
                result = await injector.run('http', {
                    "operation": "patch",
                    "url": "https://api.example.com/users/123",
                    "json_data": {"email": "new@example.com"},
                    "timeout": 30,
                    "verify_ssl": True,
                    "follow_redirects": True
                })
                
                # Verify request method
                call_args = mock_urlopen.call_args
                request = call_args[0][0]
                assert request.get_method() == "PATCH"
                
                if hasattr(result, 'output'):
                    data = json.loads(result.output)
                else:
                    data = result
                
                # No longer checking success field - function now throws exceptions on failure
                assert data["operation"] == "patch"
        
        asyncio.run(run_test())
    
    def test_http_head_request(self):
        """Test HEAD request."""
        
        async def run_test():
            injector = get_injector()
            
            # Mock urllib.request.urlopen
            mock_response = MagicMock()
            mock_response.getcode.return_value = 200
            mock_response.headers = {
                "Content-Type": "text/html",
                "Content-Length": "1234"
            }
            mock_response.geturl.return_value = "https://example.com"
            
            with patch('urllib.request.urlopen') as mock_urlopen:
                mock_urlopen.return_value = mock_response
                
                result = await injector.run('http', {
                    "operation": "head",
                    "url": "https://example.com",
                    "timeout": 30,
                    "verify_ssl": True,
                    "follow_redirects": True
                })
                
                # Verify request method
                call_args = mock_urlopen.call_args
                request = call_args[0][0]
                assert request.get_method() == "HEAD"
                
                if hasattr(result, 'output'):
                    data = json.loads(result.output)
                else:
                    data = result
                
                # No longer checking success field - function now throws exceptions on failure
                assert data["operation"] == "head"
                assert "Content-Type" in data["data"]["headers"]
        
        asyncio.run(run_test())
    
    def test_http_options_request(self):
        """Test OPTIONS request."""
        
        async def run_test():
            injector = get_injector()
            
            # Mock urllib.request.urlopen
            mock_response = MagicMock()
            mock_response.getcode.return_value = 200
            mock_response.headers = {
                "Allow": "GET, POST, PUT, DELETE"
            }
            mock_response.geturl.return_value = "https://api.example.com/users"
            
            with patch('urllib.request.urlopen') as mock_urlopen:
                mock_urlopen.return_value = mock_response
                
                result = await injector.run('http', {
                    "operation": "options",
                    "url": "https://api.example.com/users",
                    "timeout": 30,
                    "verify_ssl": True,
                    "follow_redirects": True
                })
                
                # Verify request method
                call_args = mock_urlopen.call_args
                request = call_args[0][0]
                assert request.get_method() == "OPTIONS"
                
                if hasattr(result, 'output'):
                    data = json.loads(result.output)
                else:
                    data = result
                
                # No longer checking success field - function now throws exceptions on failure
                assert data["operation"] == "options"
                assert "GET" in data["data"]["allowed_methods"]
                assert "POST" in data["data"]["allowed_methods"]
        
        asyncio.run(run_test())
    
    def test_http_error_handling(self):
        """Test HTTP error handling."""
        
        async def run_test():
            injector = get_injector()
            
            # Mock urllib.error.HTTPError
            mock_error = urllib.error.HTTPError(
                url="https://api.example.com/notfound",
                code=404,
                msg="Not Found",
                hdrs={},
                fp=None
            )
            mock_error.read = lambda: b'{"error": "Resource not found"}'
            
            with patch('urllib.request.urlopen', side_effect=mock_error):
                try:
                    result = await injector.run('http', {
                        "operation": "get",
                        "url": "https://api.example.com/notfound",
                        "timeout": 30,
                        "verify_ssl": True,
                        "follow_redirects": True
                    })
                    assert False, "Expected RuntimeError to be raised for HTTP 404"
                except RuntimeError as e:
                    assert "404" in str(e)
                    assert "Resource not found" in str(e)
                    assert "https://api.example.com/notfound" in str(e)
        
        asyncio.run(run_test())
    
    def test_http_url_error_handling(self):
        """Test URL error handling."""
        
        async def run_test():
            injector = get_injector()
            
            # Mock urllib.error.URLError
            mock_error = urllib.error.URLError("Name or service not known")
            
            with patch('urllib.request.urlopen', side_effect=mock_error):
                try:
                    result = await injector.run('http', {
                        "operation": "get",
                        "url": "https://nonexistent-domain.example",
                        "timeout": 30,
                        "verify_ssl": True,
                        "follow_redirects": True
                    })
                    assert False, "Expected RuntimeError to be raised for URL error"
                except RuntimeError as e:
                    assert "URL error" in str(e)
                    assert "Name or service not known" in str(e)
                    assert "https://nonexistent-domain.example" in str(e)
        
        asyncio.run(run_test())
    
    def test_http_general_exception_handling(self):
        """Test general exception handling."""
        
        async def run_test():
            injector = get_injector()
            
            # Mock a general exception
            with patch('urllib.request.urlopen', side_effect=ConnectionError("Connection failed")):
                try:
                    result = await injector.run('http', {
                        "operation": "post",
                        "url": "https://api.example.com/data",
                        "json_data": {"test": "data"},
                        "timeout": 30,
                        "verify_ssl": True,
                        "follow_redirects": True
                    })
                    assert False, "Expected RuntimeError to be raised for connection error"
                except RuntimeError as e:
                    assert "Failed to perform POST request" in str(e)
                    assert "Connection failed" in str(e)
                    assert "https://api.example.com/data" in str(e)
        
        asyncio.run(run_test())
    
    def test_http_form_data(self):
        """Test POST request with form data."""
        
        async def run_test():
            injector = get_injector()
            
            # Mock urllib.request.urlopen
            mock_response = MagicMock()
            mock_response.read.return_value = b'{"form_received": true}'
            mock_response.getcode.return_value = 200
            mock_response.headers = {}
            mock_response.geturl.return_value = "https://api.example.com/form"
            
            with patch('urllib.request.urlopen') as mock_urlopen:
                mock_urlopen.return_value = mock_response
                
                result = await injector.run('http', {
                    "operation": "post",
                    "url": "https://api.example.com/form",
                    "form_data": {"username": "john", "age": "25"},
                    "timeout": 30,
                    "verify_ssl": True,
                    "follow_redirects": True
                })
                
                # Verify form encoding
                call_args = mock_urlopen.call_args
                request = call_args[0][0]
                assert request.headers["Content-type"] == "application/x-www-form-urlencoded"
                body = request.data.decode()
                assert "username=john" in body
                assert "age=25" in body
                
                if hasattr(result, 'output'):
                    data = json.loads(result.output)
                else:
                    data = result
                
                # No longer checking success field - function now throws exceptions on failure
        
        asyncio.run(run_test())