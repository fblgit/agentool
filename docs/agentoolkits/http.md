# HTTP AgenToolkit

> **Note**: For guidelines on creating new AgenToolkits, see [CRAFTING_AGENTOOLS.md](../CRAFTING_AGENTOOLS.md). For testing patterns and examples, refer to [tests/agentoolkit/test_http.py](../../../tests/agentoolkit/test_http.py).

## Overview

The HTTP AgenToolkit provides comprehensive HTTP client capabilities with authentication support. It handles all standard HTTP methods, multiple authentication types, SSL verification, request/response handling, and integrates with the session toolkit for authenticated requests.

### Key Features
- All HTTP methods (GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS)
- Multiple authentication types (Bearer, Basic, Session)
- JSON and form data support
- Query parameters and headers management
- SSL certificate verification control
- Timeout and retry configuration
- Redirect following
- Response parsing (JSON auto-detection)
- Session integration for authenticated requests

## Creation Method

```python
from agentoolkit.network.http import create_http_agent

# Create the agent
agent = create_http_agent()
```

The creation function returns a fully configured AgenTool with name `'http'`.

## Input Schema

### HttpInput

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `operation` | `Literal['get', 'post', 'put', 'delete', 'patch', 'head', 'options', 'request', 'download', 'upload']` | Yes | - | The HTTP operation to perform |
| `url` | `str` | Yes | - | The URL to request |
| `headers` | `Optional[Dict[str, str]]` | No | None | HTTP headers |
| `params` | `Optional[Dict[str, Any]]` | No | None | Query parameters |
| `data` | `Optional[Union[Dict[str, Any], str]]` | No | None | Request body data |
| `json_data` | `Optional[Dict[str, Any]]` | No | None | JSON request body |
| `form_data` | `Optional[Dict[str, str]]` | No | None | Form data |
| `auth_type` | `Optional[Literal['bearer', 'basic', 'session']]` | No | None | Authentication type |
| `auth_token` | `Optional[str]` | No | None | Bearer authentication token |
| `username` | `Optional[str]` | No | None | Username for basic auth |
| `password` | `Optional[str]` | No | None | Password for basic auth |
| `session_id` | `Optional[str]` | No | None | Session ID for session auth |
| `timeout` | `Optional[int]` | No | 30 | Request timeout in seconds |
| `verify_ssl` | `Optional[bool]` | No | True | Verify SSL certificates |
| `follow_redirects` | `Optional[bool]` | No | True | Follow redirects |
| `max_retries` | `Optional[int]` | No | 3 | Maximum number of retries |
| `file_path` | `Optional[str]` | No | None | File path for upload/download |
| `file_data` | `Optional[bytes]` | No | None | File data for upload |

## Operations Schema

| Operation | Tool Function | Key Parameters | Description |
|-----------|--------------|----------------|-------------|
| `get` | `http_get` | `url`, `headers`, `params`, auth fields | HTTP GET request |
| `post` | `http_post` | `url`, `headers`, `data`/`json_data`/`form_data`, auth fields | HTTP POST request |
| `put` | `http_put` | `url`, `headers`, `data`/`json_data`, auth fields | HTTP PUT request |
| `delete` | `http_delete` | `url`, `headers`, auth fields | HTTP DELETE request |
| `patch` | `http_patch` | `url`, `headers`, `data`/`json_data`, auth fields | HTTP PATCH request |
| `head` | `http_head` | `url`, `headers`, auth fields | HTTP HEAD request |
| `options` | `http_options` | `url`, `headers`, auth fields | HTTP OPTIONS request |

## Output Schema

### HttpOutput

All operations return an `HttpOutput` model with:

| Field | Type | Description |
|-------|------|-------------|
| `operation` | `str` | The operation that was performed |
| `message` | `str` | Human-readable result message |
| `data` | `Optional[Dict[str, Any]]` | Operation-specific data |

### Operation-Specific Data Fields

- **All operations**: `status_code`, `headers`, `url`
- **GET, POST, PUT, PATCH, DELETE**: `body` (JSON object or text string)
- **OPTIONS**: `allowed_methods` (list of allowed HTTP methods)
- **HEAD**: Headers only (no body)

## Dependencies

This AgenToolkit optionally depends on:
- **session**: For session-based authentication (when `auth_type='session'`)

## Authentication Types

### Bearer Token
```python
{
    "auth_type": "bearer",
    "auth_token": "your-api-token"
}
# Adds: Authorization: Bearer your-api-token
```

### Basic Authentication
```python
{
    "auth_type": "basic",
    "username": "user",
    "password": "pass"
}
# Adds: Authorization: Basic base64(user:pass)
```

### Session Authentication
```python
{
    "auth_type": "session",
    "session_id": "session-token"
}
# Adds: X-Session-Id and Cookie headers
```

## Usage Examples

### Simple GET Request
```python
from agentoolkit.network.http import create_http_agent
from agentool.core.injector import get_injector

agent = create_http_agent()
injector = get_injector()

# Simple GET
result = await injector.run('http', {
    "operation": "get",
    "url": "https://api.example.com/users",
    "params": {"page": 1, "limit": 10}
})
```

### POST with JSON Data
```python
# POST JSON data
result = await injector.run('http', {
    "operation": "post",
    "url": "https://api.example.com/users",
    "json_data": {
        "name": "John Doe",
        "email": "john@example.com"
    },
    "headers": {"X-API-Version": "v1"}
})
```

### Authenticated Requests
```python
# Bearer token auth
result = await injector.run('http', {
    "operation": "get",
    "url": "https://api.example.com/profile",
    "auth_type": "bearer",
    "auth_token": "eyJhbGciOiJIUzI1NiIs..."
})

# Basic auth
result = await injector.run('http', {
    "operation": "delete",
    "url": "https://api.example.com/resource/123",
    "auth_type": "basic",
    "username": "admin",
    "password": "secret"
})
```

### Form Data Submission
```python
# POST form data
result = await injector.run('http', {
    "operation": "post",
    "url": "https://example.com/login",
    "form_data": {
        "username": "user",
        "password": "pass",
        "remember": "true"
    }
})
```

### Advanced Options
```python
# Custom timeout and SSL handling
result = await injector.run('http', {
    "operation": "get",
    "url": "https://internal-api.local/data",
    "timeout": 60,  # 60 seconds
    "verify_ssl": False,  # Skip SSL verification
    "follow_redirects": False
})
```

## Error Handling

The toolkit raises `RuntimeError` for various HTTP failures:
- HTTP error responses (4xx, 5xx status codes)
- URL errors (invalid URLs, connection failures)
- SSL verification failures
- Timeout errors
- General request failures

Error messages include:
- HTTP status code
- Error body content (when available)
- URL that failed
- Specific error reason

## Testing

The test suite is located at `tests/agentoolkit/test_http.py`. Tests cover:
- All HTTP methods
- Authentication types
- Request body formats (JSON, form data, raw)
- Query parameters
- Header management
- SSL verification
- Error handling
- Session integration

To run tests:
```bash
pytest tests/agentoolkit/test_http.py -v
```

## Notes

- Uses Python's built-in `urllib` for HTTP requests
- Automatically detects and parses JSON responses
- Form data is URL-encoded automatically
- SSL context is configured based on `verify_ssl` parameter
- Session authentication validates session before adding headers
- Default timeout is 30 seconds
- Response body encoding defaults to UTF-8 with error handling