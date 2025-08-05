"""HTTP AgenTool - Provides HTTP client capabilities with authentication support.

This toolkit provides comprehensive HTTP functionality including:
- GET, POST, PUT, DELETE, PATCH requests
- Headers and query parameters
- JSON and form data support
- Authentication (Bearer token, Basic auth)
- Session management integration
- Response parsing
- Error handling and retries

Example Usage:
    >>> from agentoolkit.network.http import create_http_agent
    >>> from agentool.core.injector import get_injector
    >>> 
    >>> # Create and register the agent
    >>> agent = create_http_agent()
    >>> 
    >>> # Use through injector
    >>> injector = get_injector()
    >>> result = await injector.run('http', {
    ...     "operation": "get",
    ...     "url": "https://api.example.com/data",
    ...     "headers": {"Authorization": "Bearer token123"}
    ... })
"""

import base64
import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, Field

from agentool import create_agentool
from agentool.base import BaseOperationInput
from agentool.core.injector import get_injector
from agentool.core.registry import RoutingConfig
from pydantic_ai import RunContext


class HttpInput(BaseOperationInput):
    """Input schema for HTTP operations."""
    operation: Literal[
        'get', 'post', 'put', 'delete', 'patch', 'head', 'options',
        'request', 'download', 'upload'
    ] = Field(description='The HTTP operation to perform')
    
    # Request parameters
    url: str = Field(description='The URL to request')
    headers: Optional[Dict[str, str]] = Field(None, description='HTTP headers')
    params: Optional[Dict[str, Any]] = Field(None, description='Query parameters')
    
    # Body data
    data: Optional[Union[Dict[str, Any], str]] = Field(None, description='Request body data')
    json_data: Optional[Dict[str, Any]] = Field(None, description='JSON request body')
    form_data: Optional[Dict[str, str]] = Field(None, description='Form data')
    
    # Authentication
    auth_type: Optional[Literal['bearer', 'basic', 'session']] = Field(None, description='Authentication type')
    auth_token: Optional[str] = Field(None, description='Authentication token')
    username: Optional[str] = Field(None, description='Username for basic auth')
    password: Optional[str] = Field(None, description='Password for basic auth')
    session_id: Optional[str] = Field(None, description='Session ID for session auth')
    
    # Options
    timeout: Optional[int] = Field(30, description='Request timeout in seconds')
    verify_ssl: Optional[bool] = Field(True, description='Verify SSL certificates')
    follow_redirects: Optional[bool] = Field(True, description='Follow redirects')
    max_retries: Optional[int] = Field(3, description='Maximum number of retries')
    
    # File operations
    file_path: Optional[str] = Field(None, description='File path for upload/download')
    file_data: Optional[bytes] = Field(None, description='File data for upload')


class HttpOutput(BaseModel):
    """Structured output for HTTP operations."""
    operation: str = Field(description='The operation that was performed')
    message: str = Field(description='Human-readable result message')
    data: Optional[Dict[str, Any]] = Field(None, description='Operation-specific data')


async def http_get(ctx: RunContext[Any], url: str, headers: Optional[Dict[str, str]],
                  params: Optional[Dict[str, Any]], auth_type: Optional[str],
                  auth_token: Optional[str], username: Optional[str],
                  password: Optional[str], session_id: Optional[str],
                  timeout: int, verify_ssl: bool, follow_redirects: bool) -> HttpOutput:
    """Perform an HTTP GET request.
    
    Args:
        ctx: Runtime context
        url: URL to request
        headers: HTTP headers
        params: Query parameters
        auth_type: Authentication type
        auth_token: Bearer token
        username: Basic auth username
        password: Basic auth password
        session_id: Session ID
        timeout: Request timeout
        verify_ssl: SSL verification
        follow_redirects: Follow redirects
        
    Returns:
        HttpOutput with response data
        
    Raises:
        RuntimeError: If the HTTP request fails
    """
    # Add query parameters to URL
    if params:
        query_string = urllib.parse.urlencode(params)
        url = f'{url}?{query_string}' if '?' not in url else f'{url}&{query_string}'
    
    # Prepare headers
    if headers is None:
        headers = {}
    
    # Add authentication headers
    headers = await _add_auth_headers(headers, auth_type, auth_token, 
                                     username, password, session_id)
    
    # Create request
    request = urllib.request.Request(url, headers=headers, method='GET')
    
    # Configure SSL context
    ssl_context = ssl.create_default_context()
    if not verify_ssl:
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
    
    # Perform request
    try:
        response = urllib.request.urlopen(request, timeout=timeout, context=ssl_context)
        response_data = response.read()
        
        # Parse response
        response_text = response_data.decode('utf-8', errors='ignore')
        
        # Try to parse as JSON
        try:
            response_json = json.loads(response_text)
        except json.JSONDecodeError:
            response_json = None
        
        return HttpOutput(
            operation='get',
            message=f'Successfully performed GET request to {url}',
            data={
                'status_code': response.getcode(),
                'headers': dict(response.headers),
                'body': response_json if response_json else response_text,
                'url': response.geturl()
            }
        )
        
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8', errors='ignore')
        raise RuntimeError(f'HTTP GET failed with status {e.code} for {url}: {error_body}') from e
    except urllib.error.URLError as e:
        raise RuntimeError(f'URL error during GET request to {url}: {e.reason}') from e
    except Exception as e:
        raise RuntimeError(f'Failed to perform GET request to {url}: {e}') from e


async def http_post(ctx: RunContext[Any], url: str, headers: Optional[Dict[str, str]],
                   data: Optional[Union[Dict[str, Any], str]], json_data: Optional[Dict[str, Any]],
                   form_data: Optional[Dict[str, str]], auth_type: Optional[str],
                   auth_token: Optional[str], username: Optional[str],
                   password: Optional[str], session_id: Optional[str],
                   timeout: int, verify_ssl: bool, follow_redirects: bool) -> HttpOutput:
    """Perform an HTTP POST request.
    
    Args:
        ctx: Runtime context
        url: URL to request
        headers: HTTP headers
        data: Request body data
        json_data: JSON request body
        form_data: Form data
        auth_type: Authentication type
        auth_token: Bearer token
        username: Basic auth username
        password: Basic auth password
        session_id: Session ID
        timeout: Request timeout
        verify_ssl: SSL verification
        follow_redirects: Follow redirects
        
    Returns:
        HttpOutput with response data
        
    Raises:
        RuntimeError: If the HTTP request fails
    """
    # Prepare headers
    if headers is None:
        headers = {}
    
    # Prepare request body
    body = None
    if json_data:
        body = json.dumps(json_data).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    elif form_data:
        body = urllib.parse.urlencode(form_data).encode('utf-8')
        headers['Content-Type'] = 'application/x-www-form-urlencoded'
    elif data:
        if isinstance(data, dict):
            body = json.dumps(data).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        else:
            body = data.encode('utf-8')
    
    # Add authentication headers
    headers = await _add_auth_headers(headers, auth_type, auth_token,
                                     username, password, session_id)
    
    # Create request
    request = urllib.request.Request(url, data=body, headers=headers, method='POST')
    
    # Configure SSL context
    ssl_context = ssl.create_default_context()
    if not verify_ssl:
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
    
    # Perform request
    try:
        response = urllib.request.urlopen(request, timeout=timeout, context=ssl_context)
        response_data = response.read()
        
        # Parse response
        response_text = response_data.decode('utf-8', errors='ignore')
        
        # Try to parse as JSON
        try:
            response_json = json.loads(response_text)
        except json.JSONDecodeError:
            response_json = None
        
        return HttpOutput(
            operation='post',
            message=f'Successfully performed POST request to {url}',
            data={
                'status_code': response.getcode(),
                'headers': dict(response.headers),
                'body': response_json if response_json else response_text,
                'url': response.geturl()
            }
        )
        
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8', errors='ignore')
        raise RuntimeError(f'HTTP POST failed with status {e.code} for {url}: {error_body}') from e
    except urllib.error.URLError as e:
        raise RuntimeError(f'URL error during POST request to {url}: {e.reason}') from e
    except Exception as e:
        raise RuntimeError(f'Failed to perform POST request to {url}: {e}') from e


async def http_put(ctx: RunContext[Any], url: str, headers: Optional[Dict[str, str]],
                  data: Optional[Union[Dict[str, Any], str]], json_data: Optional[Dict[str, Any]],
                  auth_type: Optional[str], auth_token: Optional[str],
                  username: Optional[str], password: Optional[str],
                  session_id: Optional[str], timeout: int,
                  verify_ssl: bool, follow_redirects: bool) -> HttpOutput:
    """Perform an HTTP PUT request.
    
    Raises:
        RuntimeError: If the HTTP request fails
    """
    # Prepare headers
    if headers is None:
        headers = {}
    
    # Prepare request body
    body = None
    if json_data:
        body = json.dumps(json_data).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    elif data:
        if isinstance(data, dict):
            body = json.dumps(data).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        else:
            body = data.encode('utf-8')
    
    # Add authentication headers
    headers = await _add_auth_headers(headers, auth_type, auth_token,
                                     username, password, session_id)
    
    # Create request
    request = urllib.request.Request(url, data=body, headers=headers, method='PUT')
    
    # Configure SSL context
    ssl_context = ssl.create_default_context()
    if not verify_ssl:
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
    
    # Perform request
    try:
        response = urllib.request.urlopen(request, timeout=timeout, context=ssl_context)
        response_data = response.read()
        
        # Parse response
        response_text = response_data.decode('utf-8', errors='ignore')
        
        # Try to parse as JSON
        try:
            response_json = json.loads(response_text)
        except json.JSONDecodeError:
            response_json = None
        
        return HttpOutput(
            operation='put',
            message=f'Successfully performed PUT request to {url}',
            data={
                'status_code': response.getcode(),
                'headers': dict(response.headers),
                'body': response_json if response_json else response_text,
                'url': response.geturl()
            }
        )
        
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8', errors='ignore')
        raise RuntimeError(f'HTTP PUT failed with status {e.code} for {url}: {error_body}') from e
    except urllib.error.URLError as e:
        raise RuntimeError(f'URL error during PUT request to {url}: {e.reason}') from e
    except Exception as e:
        raise RuntimeError(f'Failed to perform PUT request to {url}: {e}') from e


async def http_delete(ctx: RunContext[Any], url: str, headers: Optional[Dict[str, str]],
                     auth_type: Optional[str], auth_token: Optional[str],
                     username: Optional[str], password: Optional[str],
                     session_id: Optional[str], timeout: int,
                     verify_ssl: bool, follow_redirects: bool) -> HttpOutput:
    """Perform an HTTP DELETE request.
    
    Raises:
        RuntimeError: If the HTTP request fails
    """
    # Prepare headers
    if headers is None:
        headers = {}
    
    # Add authentication headers
    headers = await _add_auth_headers(headers, auth_type, auth_token,
                                     username, password, session_id)
    
    # Create request
    request = urllib.request.Request(url, headers=headers, method='DELETE')
    
    # Configure SSL context
    ssl_context = ssl.create_default_context()
    if not verify_ssl:
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
    
    # Perform request
    try:
        response = urllib.request.urlopen(request, timeout=timeout, context=ssl_context)
        response_data = response.read()
        
        # Parse response
        response_text = response_data.decode('utf-8', errors='ignore') if response_data else ''
        
        # Try to parse as JSON
        try:
            response_json = json.loads(response_text) if response_text else None
        except json.JSONDecodeError:
            response_json = None
        
        return HttpOutput(
            operation='delete',
            message=f'Successfully performed DELETE request to {url}',
            data={
                'status_code': response.getcode(),
                'headers': dict(response.headers),
                'body': response_json if response_json else response_text,
                'url': response.geturl()
            }
        )
        
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8', errors='ignore')
        raise RuntimeError(f'HTTP DELETE failed with status {e.code} for {url}: {error_body}') from e
    except urllib.error.URLError as e:
        raise RuntimeError(f'URL error during DELETE request to {url}: {e.reason}') from e
    except Exception as e:
        raise RuntimeError(f'Failed to perform DELETE request to {url}: {e}') from e


async def http_patch(ctx: RunContext[Any], url: str, headers: Optional[Dict[str, str]],
                    data: Optional[Union[Dict[str, Any], str]], json_data: Optional[Dict[str, Any]],
                    auth_type: Optional[str], auth_token: Optional[str],
                    username: Optional[str], password: Optional[str],
                    session_id: Optional[str], timeout: int,
                    verify_ssl: bool, follow_redirects: bool) -> HttpOutput:
    """Perform an HTTP PATCH request.
    
    Raises:
        RuntimeError: If the HTTP request fails
    """
    # Prepare headers
    if headers is None:
        headers = {}
    
    # Prepare request body
    body = None
    if json_data:
        body = json.dumps(json_data).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    elif data:
        if isinstance(data, dict):
            body = json.dumps(data).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        else:
            body = data.encode('utf-8')
    
    # Add authentication headers
    headers = await _add_auth_headers(headers, auth_type, auth_token,
                                     username, password, session_id)
    
    # Create request
    request = urllib.request.Request(url, data=body, headers=headers, method='PATCH')
    
    # Configure SSL context
    ssl_context = ssl.create_default_context()
    if not verify_ssl:
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
    
    # Perform request
    try:
        response = urllib.request.urlopen(request, timeout=timeout, context=ssl_context)
        response_data = response.read()
        
        # Parse response
        response_text = response_data.decode('utf-8', errors='ignore')
        
        # Try to parse as JSON
        try:
            response_json = json.loads(response_text)
        except json.JSONDecodeError:
            response_json = None
        
        return HttpOutput(
            operation='patch',
            message=f'Successfully performed PATCH request to {url}',
            data={
                'status_code': response.getcode(),
                'headers': dict(response.headers),
                'body': response_json if response_json else response_text,
                'url': response.geturl()
            }
        )
        
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8', errors='ignore')
        raise RuntimeError(f'HTTP PATCH failed with status {e.code} for {url}: {error_body}') from e
    except urllib.error.URLError as e:
        raise RuntimeError(f'URL error during PATCH request to {url}: {e.reason}') from e
    except Exception as e:
        raise RuntimeError(f'Failed to perform PATCH request to {url}: {e}') from e


async def http_head(ctx: RunContext[Any], url: str, headers: Optional[Dict[str, str]],
                   auth_type: Optional[str], auth_token: Optional[str],
                   username: Optional[str], password: Optional[str],
                   session_id: Optional[str], timeout: int,
                   verify_ssl: bool, follow_redirects: bool) -> HttpOutput:
    """Perform an HTTP HEAD request.
    
    Raises:
        RuntimeError: If the HTTP request fails
    """
    # Prepare headers
    if headers is None:
        headers = {}
    
    # Add authentication headers
    headers = await _add_auth_headers(headers, auth_type, auth_token,
                                     username, password, session_id)
    
    # Create request
    request = urllib.request.Request(url, headers=headers, method='HEAD')
    
    # Configure SSL context
    ssl_context = ssl.create_default_context()
    if not verify_ssl:
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
    
    # Perform request
    try:
        response = urllib.request.urlopen(request, timeout=timeout, context=ssl_context)
        
        return HttpOutput(
            operation='head',
            message=f'Successfully performed HEAD request to {url}',
            data={
                'status_code': response.getcode(),
                'headers': dict(response.headers),
                'url': response.geturl()
            }
        )
        
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'HTTP HEAD failed with status {e.code} for {url}') from e
    except urllib.error.URLError as e:
        raise RuntimeError(f'URL error during HEAD request to {url}: {e.reason}') from e
    except Exception as e:
        raise RuntimeError(f'Failed to perform HEAD request to {url}: {e}') from e


async def http_options(ctx: RunContext[Any], url: str, headers: Optional[Dict[str, str]],
                      auth_type: Optional[str], auth_token: Optional[str],
                      username: Optional[str], password: Optional[str],
                      session_id: Optional[str], timeout: int,
                      verify_ssl: bool, follow_redirects: bool) -> HttpOutput:
    """Perform an HTTP OPTIONS request.
    
    Raises:
        RuntimeError: If the HTTP request fails
    """
    # Prepare headers
    if headers is None:
        headers = {}
    
    # Add authentication headers
    headers = await _add_auth_headers(headers, auth_type, auth_token,
                                     username, password, session_id)
    
    # Create request
    request = urllib.request.Request(url, headers=headers, method='OPTIONS')
    
    # Configure SSL context
    ssl_context = ssl.create_default_context()
    if not verify_ssl:
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
    
    # Perform request
    try:
        response = urllib.request.urlopen(request, timeout=timeout, context=ssl_context)
        
        # Get allowed methods from headers
        allow_header = response.headers.get('Allow', '')
        allowed_methods = [m.strip() for m in allow_header.split(',')] if allow_header else []
        
        return HttpOutput(
            operation='options',
            message=f'Successfully performed OPTIONS request to {url}',
            data={
                'status_code': response.getcode(),
                'headers': dict(response.headers),
                'allowed_methods': allowed_methods,
                'url': response.geturl()
            }
        )
        
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'HTTP OPTIONS failed with status {e.code} for {url}') from e
    except urllib.error.URLError as e:
        raise RuntimeError(f'URL error during OPTIONS request to {url}: {e.reason}') from e
    except Exception as e:
        raise RuntimeError(f'Failed to perform OPTIONS request to {url}: {e}') from e


async def _add_auth_headers(headers: Dict[str, str], auth_type: Optional[str],
                           auth_token: Optional[str], username: Optional[str],
                           password: Optional[str], session_id: Optional[str]) -> Dict[str, str]:
    """Add authentication headers to the request.
    
    Args:
        headers: Existing headers
        auth_type: Type of authentication
        auth_token: Bearer token
        username: Basic auth username
        password: Basic auth password
        session_id: Session ID
        
    Returns:
        Updated headers with authentication
    """
    if auth_type == 'bearer' and auth_token:
        headers['Authorization'] = f'Bearer {auth_token}'
    elif auth_type == 'basic' and username and password:
        credentials = f'{username}:{password}'
        encoded = base64.b64encode(credentials.encode()).decode()
        headers['Authorization'] = f'Basic {encoded}'
    elif auth_type == 'session' and session_id:
        # Validate session and get user info
        injector = get_injector()
        try:
            session_result = await injector.run('session', {
                'operation': 'validate',
                'session_id': session_id
            })
            
            if hasattr(session_result, 'output'):
                session_data = json.loads(session_result.output)
            else:
                session_data = session_result
            
            if session_data['success'] and session_data['data']['valid']:
                # Add session cookie or header
                headers['X-Session-Id'] = session_id
                headers['Cookie'] = f'session_id={session_id}'
        except Exception:
            pass  # Session validation failed, continue without auth
    
    return headers


def create_http_agent():
    """Create and return the HTTP AgenTool.
    
    Returns:
        Agent configured for HTTP operations
    """
    http_routing = RoutingConfig(
        operation_field='operation',
        operation_map={
            'get': ('http_get', lambda x: {
                'url': x.url, 'headers': x.headers, 'params': x.params,
                'auth_type': x.auth_type, 'auth_token': x.auth_token,
                'username': x.username, 'password': x.password,
                'session_id': x.session_id, 'timeout': x.timeout,
                'verify_ssl': x.verify_ssl, 'follow_redirects': x.follow_redirects
            }),
            'post': ('http_post', lambda x: {
                'url': x.url, 'headers': x.headers, 'data': x.data,
                'json_data': x.json_data, 'form_data': x.form_data,
                'auth_type': x.auth_type, 'auth_token': x.auth_token,
                'username': x.username, 'password': x.password,
                'session_id': x.session_id, 'timeout': x.timeout,
                'verify_ssl': x.verify_ssl, 'follow_redirects': x.follow_redirects
            }),
            'put': ('http_put', lambda x: {
                'url': x.url, 'headers': x.headers, 'data': x.data,
                'json_data': x.json_data, 'auth_type': x.auth_type,
                'auth_token': x.auth_token, 'username': x.username,
                'password': x.password, 'session_id': x.session_id,
                'timeout': x.timeout, 'verify_ssl': x.verify_ssl,
                'follow_redirects': x.follow_redirects
            }),
            'delete': ('http_delete', lambda x: {
                'url': x.url, 'headers': x.headers, 'auth_type': x.auth_type,
                'auth_token': x.auth_token, 'username': x.username,
                'password': x.password, 'session_id': x.session_id,
                'timeout': x.timeout, 'verify_ssl': x.verify_ssl,
                'follow_redirects': x.follow_redirects
            }),
            'patch': ('http_patch', lambda x: {
                'url': x.url, 'headers': x.headers, 'data': x.data,
                'json_data': x.json_data, 'auth_type': x.auth_type,
                'auth_token': x.auth_token, 'username': x.username,
                'password': x.password, 'session_id': x.session_id,
                'timeout': x.timeout, 'verify_ssl': x.verify_ssl,
                'follow_redirects': x.follow_redirects
            }),
            'head': ('http_head', lambda x: {
                'url': x.url, 'headers': x.headers, 'auth_type': x.auth_type,
                'auth_token': x.auth_token, 'username': x.username,
                'password': x.password, 'session_id': x.session_id,
                'timeout': x.timeout, 'verify_ssl': x.verify_ssl,
                'follow_redirects': x.follow_redirects
            }),
            'options': ('http_options', lambda x: {
                'url': x.url, 'headers': x.headers, 'auth_type': x.auth_type,
                'auth_token': x.auth_token, 'username': x.username,
                'password': x.password, 'session_id': x.session_id,
                'timeout': x.timeout, 'verify_ssl': x.verify_ssl,
                'follow_redirects': x.follow_redirects
            }),
        }
    )
    
    return create_agentool(
        name='http',
        input_schema=HttpInput,
        routing_config=http_routing,
        tools=[
            http_get, http_post, http_put, http_delete,
            http_patch, http_head, http_options
        ],
        output_type=HttpOutput,
        system_prompt='Perform HTTP requests with authentication support.',
        description='Comprehensive HTTP client with authentication integration',
        version='1.0.0',
        tags=['http', 'network', 'api', 'rest', 'authentication'],
        dependencies=['session'],  # Uses session for auth token management
        examples=[
            {
                'description': 'GET request with bearer token',
                'input': {
                    'operation': 'get',
                    'url': 'https://api.example.com/users',
                    'auth_type': 'bearer',
                    'auth_token': 'token123'
                },
                'output': {
                    'success': True,
                    'operation': 'get',
                    'message': 'Successfully performed GET request'
                }
            },
            {
                'description': 'POST JSON data',
                'input': {
                    'operation': 'post',
                    'url': 'https://api.example.com/users',
                    'json_data': {'name': 'John', 'email': 'john@example.com'}
                },
                'output': {
                    'success': True,
                    'operation': 'post',
                    'message': 'Successfully performed POST request'
                }
            }
        ]
    )


# Create the agent instance
agent = create_http_agent()