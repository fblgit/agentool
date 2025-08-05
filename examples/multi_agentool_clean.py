#!/usr/bin/env python3
"""
Multi-AgenTool Example with Clean Dependency Injection

This example demonstrates the cleanest approach to dependency injection:
1. Using the registry as a dependency container with automatic registration
2. Tools access other agents through the injector
3. No manual dependency wiring needed
4. Clean, type-safe approach

Architecture:
- KV Storage: Base primitive for key-value storage
- Session Manager: Uses KV storage to manage user sessions  
- HTTP Client: Uses session manager for authenticated requests

Run this example:
    python src/examples/multi_agentool_clean.py
"""

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.agentool import create_agentool, register_agentool_models, get_injector
from src.agentool.core.registry import RoutingConfig

# Optional: Configure Logfire for observability
try:
    import logfire
    logfire.configure(send_to_logfire='if-token-present')
    logfire.instrument_pydantic_ai()
    LOGFIRE_ENABLED = True
    print("✅ Logfire instrumentation enabled")
except ImportError:
    LOGFIRE_ENABLED = False
    print("ℹ️  Logfire not available - install with: pip install 'pydantic-ai[logfire]'")


# =============================================================================
# Layer 1: KV Storage (Base Primitive)
# =============================================================================

# In-memory storage (could be Redis in production)
_kv_store: Dict[str, Any] = {}
_kv_ttl: Dict[str, datetime] = {}


class KVStorageInput(BaseModel):
    """Input schema for KV storage operations."""
    operation: Literal['get', 'set', 'delete', 'exists', 'keys'] = Field(
        description="The KV operation to perform"
    )
    key: Optional[str] = Field(None, description="The key to operate on")
    value: Optional[Any] = Field(None, description="The value to store")
    ttl: Optional[int] = Field(None, description="Time to live in seconds")
    pattern: Optional[str] = Field(None, description="Pattern for keys operation")


async def kv_get(ctx: RunContext[None], key: str) -> dict:
    """Get value by key."""
    if key in _kv_store:
        # Check TTL
        if key in _kv_ttl and datetime.now() > _kv_ttl[key]:
            del _kv_store[key]
            del _kv_ttl[key]
            return {"success": True, "operation": "get", "key": key, "value": None, "exists": False}
        return {"success": True, "operation": "get", "key": key, "value": _kv_store[key], "exists": True}
    return {"success": True, "operation": "get", "key": key, "value": None, "exists": False}


async def kv_set(ctx: RunContext[None], key: str, value: Any, ttl: Optional[int]) -> dict:
    """Set key-value pair with optional TTL."""
    _kv_store[key] = value
    if ttl:
        _kv_ttl[key] = datetime.now() + timedelta(seconds=ttl)
    elif key in _kv_ttl:
        del _kv_ttl[key]
    
    return {
        "success": True,
        "operation": "set",
        "key": key,
        "stored": True,
        "ttl": ttl
    }


async def kv_delete(ctx: RunContext[None], key: str) -> dict:
    """Delete a key."""
    existed = key in _kv_store
    if existed:
        del _kv_store[key]
        if key in _kv_ttl:
            del _kv_ttl[key]
    
    return {
        "success": True,
        "operation": "delete",
        "key": key,
        "existed": existed
    }


async def kv_exists(ctx: RunContext[None], key: str) -> dict:
    """Check if key exists."""
    exists = key in _kv_store
    if exists and key in _kv_ttl and datetime.now() > _kv_ttl[key]:
        # Expired
        del _kv_store[key]
        del _kv_ttl[key]
        exists = False
    
    return {
        "success": True,
        "operation": "exists",
        "key": key,
        "exists": exists
    }


async def kv_keys(ctx: RunContext[None], pattern: Optional[str]) -> dict:
    """List keys matching pattern."""
    all_keys = list(_kv_store.keys())
    
    if pattern:
        # Simple pattern matching (in production, use proper glob matching)
        if '*' in pattern:
            prefix = pattern.replace('*', '')
            matched_keys = [k for k in all_keys if k.startswith(prefix)]
        else:
            matched_keys = [k for k in all_keys if pattern in k]
    else:
        matched_keys = all_keys
    
    return {
        "success": True,
        "operation": "keys",
        "pattern": pattern,
        "keys": matched_keys,
        "count": len(matched_keys)
    }


# Create KV Storage agent
def create_kv_storage_agent():
    """Create KV storage agent (no dependencies)."""
    kv_routing = RoutingConfig(
        operation_field='operation',
        operation_map={
            'get': ('kv_get', lambda x: {'key': x.key}),
            'set': ('kv_set', lambda x: {'key': x.key, 'value': x.value, 'ttl': x.ttl}),
            'delete': ('kv_delete', lambda x: {'key': x.key}),
            'exists': ('kv_exists', lambda x: {'key': x.key}),
            'keys': ('kv_keys', lambda x: {'pattern': x.pattern}),
        }
    )
    
    return create_agentool(
        name='kv_storage',
        input_schema=KVStorageInput,
        routing_config=kv_routing,
        tools=[kv_get, kv_set, kv_delete, kv_exists, kv_keys],
        system_prompt="Key-value storage operations with TTL support.",
        description="In-memory key-value storage with TTL support",
        version="1.0.0",
        tags=["storage", "kv", "primitive"],
        examples=[
            {
                "description": "Set a value with TTL",
                "input": {"operation": "set", "key": "user:123", "value": {"name": "Alice"}, "ttl": 3600},
                "output": {"success": True, "operation": "set", "key": "user:123", "stored": True, "ttl": 3600}
            }
        ]
    )


# =============================================================================
# Layer 2: Session Manager (Depends on KV Storage)
# =============================================================================

class SessionInput(BaseModel):
    """Input schema for session operations."""
    operation: Literal['create', 'get', 'update', 'delete', 'list'] = Field(
        description="The session operation to perform"
    )
    session_id: Optional[str] = Field(None, description="Session ID")
    user_id: Optional[str] = Field(None, description="User ID for session creation")
    data: Optional[Dict[str, Any]] = Field(None, description="Session data")
    ttl: int = Field(default=3600, description="Session TTL in seconds")


async def session_create(ctx: RunContext[None], user_id: str, data: Dict[str, Any], ttl: int) -> dict:
    """Create a new session."""
    session_id = f"session:{uuid.uuid4()}"
    
    session_data = {
        "session_id": session_id,
        "user_id": user_id,
        "data": data or {},
        "created_at": datetime.now().isoformat(),
        "last_accessed": datetime.now().isoformat()
    }
    
    # Use the injector to call KV storage - now supports direct dict input!
    injector = get_injector()
    result = await injector.run('kv_storage', {
        "operation": "set",
        "key": session_id,
        "value": session_data,
        "ttl": ttl
    })
    kv_output = json.loads(result.output)
    
    if kv_output.get("success"):
        # Also store user -> session mapping
        user_key = f"user_sessions:{user_id}"
        user_sessions_result = await injector.run('kv_storage', {
            "operation": "get",
            "key": user_key
        })
        user_sessions_data = json.loads(user_sessions_result.output)
        
        user_sessions = user_sessions_data.get("value", []) if user_sessions_data.get("exists") else []
        user_sessions.append(session_id)
        
        await injector.run('kv_storage', {
            "operation": "set",
            "key": user_key,
            "value": user_sessions,
            "ttl": ttl
        })
        
        return {
            "success": True,
            "operation": "create",
            "session_id": session_id,
            "user_id": user_id,
            "ttl": ttl
        }
    
    return {"success": False, "operation": "create", "error": "Failed to create session"}


async def session_get(ctx: RunContext[None], session_id: str) -> dict:
    """Get session data."""
    # Use KV storage to retrieve session
    injector = get_injector()
    result = await injector.run('kv_storage', {
        "operation": "get", 
        "key": session_id
    })
    kv_output = json.loads(result.output)
    
    if kv_output.get("exists"):
        session_data = kv_output["value"]
        # Update last accessed time
        session_data["last_accessed"] = datetime.now().isoformat()
        
        # Update in KV storage
        await injector.run('kv_storage', {
            "operation": "set",
            "key": session_id,
            "value": session_data,
            "ttl": 3600  # Reset TTL on access
        })
        
        return {
            "success": True,
            "operation": "get",
            "session": session_data
        }
    
    return {
        "success": True,
        "operation": "get",
        "session": None,
        "exists": False
    }


async def session_update(ctx: RunContext[None], session_id: str, data: Dict[str, Any]) -> dict:
    """Update session data."""
    # Get existing session
    get_result = await session_get(ctx, session_id)
    
    if get_result.get("session"):
        session_data = get_result["session"]
        session_data["data"].update(data)
        session_data["last_accessed"] = datetime.now().isoformat()
        
        # Save updated session
        injector = get_injector()
        result = await injector.run('kv_storage', {
            "operation": "set",
            "key": session_id,
            "value": session_data,
            "ttl": 3600
        })
        
        return {
            "success": True,
            "operation": "update",
            "session_id": session_id,
            "updated": True
        }
    
    return {
        "success": False,
        "operation": "update",
        "error": "Session not found"
    }


async def session_delete(ctx: RunContext[None], session_id: str) -> dict:
    """Delete a session."""
    # Get session to find user
    get_result = await session_get(ctx, session_id)
    
    if get_result.get("session"):
        user_id = get_result["session"]["user_id"]
        
        # Delete session
        injector = get_injector()
        result = await injector.run('kv_storage', {
            "operation": "delete", 
            "key": session_id
        })
        
        # Remove from user's session list
        user_key = f"user_sessions:{user_id}"
        user_sessions_result = await injector.run('kv_storage', {
            "operation": "get",
            "key": user_key
        })
        
        if json.loads(user_sessions_result.output).get("exists"):
            user_sessions = json.loads(user_sessions_result.output)["value"]
            user_sessions = [s for s in user_sessions if s != session_id]
            
            if user_sessions:
                await injector.run('kv_storage', {
                    "operation": "set",
                    "key": user_key,
                    "value": user_sessions
                })
            else:
                await injector.run('kv_storage', {
                    "operation": "delete",
                    "key": user_key
                })
        
        return {
            "success": True,
            "operation": "delete",
            "session_id": session_id,
            "deleted": True
        }
    
    return {
        "success": True,
        "operation": "delete",
        "session_id": session_id,
        "deleted": False
    }


async def session_list(ctx: RunContext[None], user_id: Optional[str]) -> dict:
    """List sessions for a user or all sessions."""
    injector = get_injector()
    
    if user_id:
        # Get user's sessions
        user_key = f"user_sessions:{user_id}"
        result = await injector.run('kv_storage', {
            "operation": "get",
            "key": user_key
        })
        output = json.loads(result.output)
        
        if output.get("exists"):
            session_ids = output["value"]
            sessions = []
            
            for session_id in session_ids:
                session_result = await injector.run('kv_storage', {
                    "operation": "get",
                    "key": session_id
                })
                session_output = json.loads(session_result.output)
                if session_output.get("exists"):
                    sessions.append(session_output["value"])
            
            return {
                "success": True,
                "operation": "list",
                "user_id": user_id,
                "sessions": sessions,
                "count": len(sessions)
            }
        else:
            return {
                "success": True,
                "operation": "list",
                "user_id": user_id,
                "sessions": [],
                "count": 0
            }
    else:
        # List all sessions
        result = await injector.run('kv_storage', {
            "operation": "keys",
            "pattern": "session:*"
        })
        output = json.loads(result.output)
        
        sessions = []
        for session_key in output.get("keys", []):
            session_result = await injector.run('kv_storage', {
                "operation": "get",
                "key": session_key
            })
            session_output = json.loads(session_result.output)
            if session_output.get("exists"):
                sessions.append(session_output["value"])
        
        return {
            "success": True,
            "operation": "list",
            "sessions": sessions,
            "count": len(sessions)
        }


def create_session_agent():
    """Create session agent with KV storage dependency."""
    session_routing = RoutingConfig(
        operation_field='operation',
        operation_map={
            'create': ('session_create', lambda x: {'user_id': x.user_id, 'data': x.data, 'ttl': x.ttl}),
            'get': ('session_get', lambda x: {'session_id': x.session_id}),
            'update': ('session_update', lambda x: {'session_id': x.session_id, 'data': x.data}),
            'delete': ('session_delete', lambda x: {'session_id': x.session_id}),
            'list': ('session_list', lambda x: {'user_id': x.user_id}),
        }
    )
    
    return create_agentool(
        name='session',
        input_schema=SessionInput,
        routing_config=session_routing,
        tools=[session_create, session_get, session_update, session_delete, session_list],
        system_prompt="Session management using KV storage backend.",
        description="User session management with KV storage backend",
        version="1.0.0",
        tags=["session", "state", "auth"],
        dependencies=["kv_storage"],
        examples=[
            {
                "description": "Create a new session",
                "input": {"operation": "create", "user_id": "user123", "data": {"theme": "dark"}},
                "output": {"success": True, "operation": "create", "session_id": "session:...", "user_id": "user123"}
            }
        ]
    )


# =============================================================================
# Layer 3: HTTP Client (Depends on Session Manager)
# =============================================================================

class HttpInput(BaseModel):
    """Input schema for HTTP operations."""
    operation: Literal['request', 'get', 'post'] = Field(
        description="The HTTP operation to perform"
    )
    url: str = Field(description="The URL to request")
    method: Optional[str] = Field("GET", description="HTTP method")
    headers: Optional[Dict[str, str]] = Field(None, description="Request headers")
    data: Optional[Any] = Field(None, description="Request body data")
    session_id: Optional[str] = Field(None, description="Session ID for authenticated requests")


# Mock HTTP responses for demo
MOCK_RESPONSES = {
    "https://api.example.com/user": {
        "GET": {"id": "user123", "name": "Alice Smith", "email": "alice@example.com"}
    },
    "https://api.example.com/data": {
        "GET": {"items": [{"id": 1, "value": "foo"}, {"id": 2, "value": "bar"}]},
        "POST": {"created": True, "id": 3}
    }
}


async def http_request(ctx: RunContext[None], url: str, method: str, headers: Dict[str, str], 
                      data: Any, session_id: Optional[str]) -> dict:
    """Make HTTP request with optional session support."""
    final_headers = headers or {}
    auth_info = None
    
    if session_id:
        # Get session data through injector
        injector = get_injector()
        session_result = await injector.run('session', {
            "operation": "get", 
            "session_id": session_id
        })
        session_data = json.loads(session_result.output)
        
        if session_data.get("session"):
            session_info = session_data["session"]
            # Add auth headers from session
            if "auth_token" in session_info.get("data", {}):
                final_headers["Authorization"] = f"Bearer {session_info['data']['auth_token']}"
                auth_info = {"user_id": session_info["user_id"], "authenticated": True}
            
            # Update session access
            await injector.run('session', {
                "operation": "update",
                "session_id": session_id,
                "data": {"last_api_call": datetime.now().isoformat()}
            })
    
    # Mock HTTP request (in production, use httpx or similar)
    response_data = MOCK_RESPONSES.get(url, {}).get(method, {"error": "Not found"})
    
    result = {
        "success": True,
        "operation": "request",
        "url": url,
        "method": method,
        "status_code": 200 if url in MOCK_RESPONSES else 404,
        "headers": final_headers,
        "response": response_data,
        "auth_info": auth_info
    }
    
    # Store response in session if applicable
    if session_id and "error" not in response_data:
        injector = get_injector()
        await injector.run('session', {
            "operation": "update",
            "session_id": session_id,
            "data": {"last_response": response_data}
        })
    
    return result


async def http_get(ctx: RunContext[None], url: str, headers: Dict[str, str], session_id: Optional[str]) -> dict:
    """HTTP GET request."""
    return await http_request(ctx, url, "GET", headers, None, session_id)


async def http_post(ctx: RunContext[None], url: str, headers: Dict[str, str], data: Any, session_id: Optional[str]) -> dict:
    """HTTP POST request."""
    return await http_request(ctx, url, "POST", headers, data, session_id)


def create_http_agent():
    """Create HTTP agent with session dependency."""
    http_routing = RoutingConfig(
        operation_field='operation',
        operation_map={
            'request': ('http_request', lambda x: {
                'url': x.url, 'method': x.method, 'headers': x.headers, 
                'data': x.data, 'session_id': x.session_id
            }),
            'get': ('http_get', lambda x: {'url': x.url, 'headers': x.headers, 'session_id': x.session_id}),
            'post': ('http_post', lambda x: {'url': x.url, 'headers': x.headers, 'data': x.data, 'session_id': x.session_id}),
        }
    )
    
    return create_agentool(
        name='http',
        input_schema=HttpInput,
        routing_config=http_routing,
        tools=[http_request, http_get, http_post],
        system_prompt="HTTP client with session-based authentication support.",
        description="HTTP client with session management integration",
        version="1.0.0",
        tags=["http", "network", "api"],
        dependencies=["session"],
        examples=[
            {
                "description": "GET request with session",
                "input": {"operation": "get", "url": "https://api.example.com/user", "session_id": "session:123"},
                "output": {"success": True, "status_code": 200, "response": {"id": "user123", "name": "Alice"}}
            }
        ]
    )


# =============================================================================
# Demo: Multi-AgenTool Workflow with Clean Dependency Injection
# =============================================================================

async def demo_workflow():
    """Demonstrate multi-AgenTool interoperability with clean dependency injection."""
    print("\n" + "="*80)
    print("Multi-AgenTool Demo with Clean Dependency Injection (V2)")
    print("="*80 + "\n")
    
    # Register AgenTool models
    register_agentool_models()
    
    # Step 1: Create agents (automatically registered with injector)
    print("1️⃣  Creating AgenTools...")
    kv_agent = create_kv_storage_agent()
    session_agent = create_session_agent()
    http_agent = create_http_agent()
    print("   ✅ KV Storage Agent created")
    print("   ✅ Session Agent created (depends on KV Storage)")
    print("   ✅ HTTP Agent created (depends on Session)")
    
    # Get the injector (agents are automatically registered when created)
    injector = get_injector()
    
    print("\n2️⃣  Testing KV Storage (base layer)...")
    # Direct KV operations using injector - now with automatic JSON serialization!
    kv_result = await injector.run('kv_storage', {
        "operation": "set",
        "key": "config:app",
        "value": {"theme": "dark", "lang": "en"},
        "ttl": 7200
    })
    print(f"   KV Set: {json.loads(kv_result.output)}")
    
    kv_result = await injector.run('kv_storage', {
        "operation": "get",
        "key": "config:app"
    })
    print(f"   KV Get: {json.loads(kv_result.output)}")
    
    print("\n3️⃣  Creating user session (uses KV storage)...")
    # Create a session - dependencies are automatically handled!
    session_result = await injector.run('session', {
        "operation": "create",
        "user_id": "alice",
        "data": {"auth_token": "secret-token-123", "preferences": {"theme": "dark"}}
    })
    session_output = json.loads(session_result.output)
    session_id = session_output["session_id"]
    print(f"   Session created: {session_output}")
    
    # Get session details
    session_result = await injector.run('session', {
        "operation": "get",
        "session_id": session_id
    })
    print(f"   Session retrieved: {json.loads(session_result.output)}")
    
    print("\n4️⃣  Making authenticated HTTP requests (uses session)...")
    # Make HTTP request with session - all dependencies automatically wired!
    http_result = await injector.run('http', {
        "operation": "get",
        "url": "https://api.example.com/user",
        "session_id": session_id
    })
    http_output = json.loads(http_result.output)
    print(f"   HTTP GET (authenticated): {http_output}")
    
    # Make POST request
    http_result = await injector.run('http', {
        "operation": "post",
        "url": "https://api.example.com/data",
        "data": {"name": "New Item"},
        "session_id": session_id
    })
    print(f"   HTTP POST: {json.loads(http_result.output)}")
    
    print("\n5️⃣  Checking session state after HTTP calls...")
    # Check session state after HTTP calls
    session_result = await injector.run('session', {
        "operation": "get",
        "session_id": session_id
    })
    session_data = json.loads(session_result.output)["session"]
    print(f"   Session last accessed: {session_data['last_accessed']}")
    print(f"   Session data: {json.dumps(session_data['data'], indent=2)}")
    
    print("\n6️⃣  Demonstrating dependency override for testing...")
    # Create a mock KV storage for testing
    from src.agentool import create_agentool
    from src.agentool.core.registry import RoutingConfig
    
    async def mock_kv_get(ctx: RunContext[None], key: str) -> dict:
        """Mock KV get that always returns test data."""
        return {"success": True, "operation": "get", "key": key, "value": {"mock": "data"}, "exists": True}
    
    mock_kv_routing = RoutingConfig(
        operation_field='operation', 
        operation_map={'get': ('mock_kv_get', lambda x: {'key': x.key})}
    )
    
    mock_kv = create_agentool(
        name='mock_kv_storage',
        input_schema=KVStorageInput,
        routing_config=mock_kv_routing,
        tools=[mock_kv_get],
        system_prompt="Mock KV storage for testing"
    )
    
    # Override the KV storage dependency
    with injector.override(kv_storage=mock_kv):
        print("   With mocked KV storage:")
        result = await injector.run('kv_storage', {
            "operation": "get",
            "key": "any_key"
        })
        print(f"   Mocked KV get: {json.loads(result.output)}")
    
    print("\n7️⃣  Registry Information...")
    from src.agentool.core.registry import AgenToolRegistry
    
    print("\n   Registered AgenTools:")
    for name in AgenToolRegistry.list_names():
        config = AgenToolRegistry.get(name)
        if config and name in ['kv_storage', 'session', 'http']:
            print(f"   - {name}: {config.description}")
            if config.dependencies:
                print(f"     Dependencies: {', '.join(config.dependencies)}")
    
    # Show dependency graph
    dep_graph = AgenToolRegistry.generate_dependency_graph()
    print(f"\n   Dependency Graph:")
    print(f"   {json.dumps(dep_graph, indent=2)}")
    
    if LOGFIRE_ENABLED:
        print("\n✨ Check your Logfire dashboard to see the complete trace of this multi-agent interaction!")
    
    print("\n" + "="*80)
    print("Demo completed successfully! 🎉")
    print("Key improvements in this version:")
    print("- ✅ Automatic agent registration with the injector")
    print("- ✅ Tools access other agents through get_injector()")
    print("- ✅ No need for complex dependency classes")
    print("- ✅ Clean, straightforward implementation")
    print("- ✅ Support for dependency overrides (great for testing!)")
    print("- ✅ Automatic JSON serialization - pass dicts, models, or strings!")
    print("="*80)


async def main():
    """Main entry point."""
    await demo_workflow()


if __name__ == "__main__":
    asyncio.run(main())