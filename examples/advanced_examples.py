"""
Advanced examples showcasing AgenTool capabilities.

This module demonstrates various advanced use cases and patterns
for building production-ready AgenTools.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import RunContext, Agent

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from src.agentool import create_agentool
from src.agentool.core.registry import RoutingConfig


# Example 1: Multi-tier Application with Database Operations
# ----------------------------------------------------------

class DatabaseOperation(BaseModel):
    """Schema for database operations with validation."""
    
    action: str = Field(description="Database action: query, insert, update, delete")
    table: str = Field(description="Target table name")
    criteria: Optional[Dict[str, Any]] = Field(None, description="Query criteria")
    data: Optional[Dict[str, Any]] = Field(None, description="Data for insert/update")
    
    @field_validator('table')
    def validate_table(cls, v):
        """Ensure table name is safe."""
        allowed_tables = ['users', 'posts', 'comments', 'logs']
        if v not in allowed_tables:
            raise ValueError(f"Table must be one of: {allowed_tables}")
        return v
    
    @field_validator('data')
    def validate_data_for_insert(cls, v, info):
        """Ensure data is provided for insert/update operations."""
        if info.data.get('action') in ['insert', 'update'] and not v:
            raise ValueError(f"Data required for {info.data['action']} operation")
        return v


# Simulated database
mock_db: Dict[str, List[Dict]] = {
    'users': [],
    'posts': [],
    'comments': [],
    'logs': []
}


async def db_query(ctx: RunContext[Any], table: str, criteria: Dict[str, Any]) -> List[Dict]:
    """Query database with criteria."""
    results = []
    for record in mock_db.get(table, []):
        if all(record.get(k) == v for k, v in criteria.items()):
            results.append(record)
    return results


async def db_insert(ctx: RunContext[Any], table: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Insert new record."""
    data['id'] = len(mock_db[table]) + 1
    data['created_at'] = datetime.now().isoformat()
    mock_db[table].append(data)
    return {"success": True, "id": data['id'], "message": f"Inserted into {table}"}


async def db_update(ctx: RunContext[Any], table: str, criteria: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Update existing records."""
    updated = 0
    for record in mock_db.get(table, []):
        if all(record.get(k) == v for k, v in criteria.items()):
            record.update(data)
            record['updated_at'] = datetime.now().isoformat()
            updated += 1
    return {"success": True, "updated": updated, "message": f"Updated {updated} records in {table}"}


async def db_delete(ctx: RunContext[Any], table: str, criteria: Dict[str, Any]) -> Dict[str, Any]:
    """Delete records matching criteria."""
    original_count = len(mock_db[table])
    mock_db[table] = [r for r in mock_db[table] if not all(r.get(k) == v for k, v in criteria.items())]
    deleted = original_count - len(mock_db[table])
    return {"success": True, "deleted": deleted, "message": f"Deleted {deleted} records from {table}"}


# Create database agent
db_routing = RoutingConfig(
    operation_field='action',
    operation_map={
        'query': ('db_query', lambda x: {'table': x.table, 'criteria': x.criteria or {}}),
        'insert': ('db_insert', lambda x: {'table': x.table, 'data': x.data}),
        'update': ('db_update', lambda x: {'table': x.table, 'criteria': x.criteria or {}, 'data': x.data}),
        'delete': ('db_delete', lambda x: {'table': x.table, 'criteria': x.criteria or {}}),
    }
)

database_agent = create_agentool(
    name='database',
    input_schema=DatabaseOperation,
    routing_config=db_routing,
    tools=[db_query, db_insert, db_update, db_delete],
    system_prompt="Handle database operations with validation and safety checks.",
    description="Production-ready database operations handler"
)


# Example 2: State Machine Implementation
# ---------------------------------------

class StateMachineInput(BaseModel):
    """Input for state machine operations."""
    
    command: str = Field(description="Command: transition, query, reset")
    machine_id: str = Field(description="State machine instance ID")
    event: Optional[str] = Field(None, description="Event for transition")
    
    
# State machines storage
state_machines: Dict[str, Dict[str, Any]] = {}

# State machine configuration
STATE_CONFIG = {
    'order': {
        'initial': 'pending',
        'states': ['pending', 'processing', 'shipped', 'delivered', 'cancelled'],
        'transitions': {
            'pending': {'approve': 'processing', 'cancel': 'cancelled'},
            'processing': {'ship': 'shipped', 'cancel': 'cancelled'},
            'shipped': {'deliver': 'delivered'},
            'delivered': {},
            'cancelled': {}
        }
    }
}


async def sm_transition(ctx: RunContext[Any], machine_id: str, event: str) -> Dict[str, Any]:
    """Process state transition."""
    if machine_id not in state_machines:
        # Initialize new machine
        state_machines[machine_id] = {
            'state': STATE_CONFIG['order']['initial'],
            'history': [{'state': STATE_CONFIG['order']['initial'], 'timestamp': datetime.now().isoformat()}]
        }
    
    machine = state_machines[machine_id]
    current_state = machine['state']
    transitions = STATE_CONFIG['order']['transitions'].get(current_state, {})
    
    if event not in transitions:
        return {
            "success": False,
            "error": f"Invalid transition '{event}' from state '{current_state}'",
            "allowed": list(transitions.keys())
        }
    
    new_state = transitions[event]
    machine['state'] = new_state
    machine['history'].append({
        'from': current_state,
        'to': new_state,
        'event': event,
        'timestamp': datetime.now().isoformat()
    })
    
    return {
        "success": True,
        "machine_id": machine_id,
        "previous": current_state,
        "current": new_state,
        "event": event
    }


async def sm_query(ctx: RunContext[Any], machine_id: str) -> Dict[str, Any]:
    """Query state machine status."""
    if machine_id not in state_machines:
        return {"error": f"State machine '{machine_id}' not found"}
    
    machine = state_machines[machine_id]
    current_state = machine['state']
    allowed_events = list(STATE_CONFIG['order']['transitions'].get(current_state, {}).keys())
    
    return {
        "machine_id": machine_id,
        "current_state": current_state,
        "allowed_events": allowed_events,
        "history": machine['history'][-5:]  # Last 5 transitions
    }


async def sm_reset(ctx: RunContext[Any], machine_id: str) -> Dict[str, Any]:
    """Reset state machine to initial state."""
    state_machines[machine_id] = {
        'state': STATE_CONFIG['order']['initial'],
        'history': [{'state': STATE_CONFIG['order']['initial'], 'timestamp': datetime.now().isoformat()}]
    }
    return {"success": True, "machine_id": machine_id, "state": STATE_CONFIG['order']['initial']}


# Create state machine agent
sm_routing = RoutingConfig(
    operation_field='command',
    operation_map={
        'transition': ('sm_transition', lambda x: {'machine_id': x.machine_id, 'event': x.event}),
        'query': ('sm_query', lambda x: {'machine_id': x.machine_id}),
        'reset': ('sm_reset', lambda x: {'machine_id': x.machine_id}),
    }
)

state_machine_agent = create_agentool(
    name='state_machine',
    input_schema=StateMachineInput,
    routing_config=sm_routing,
    tools=[sm_transition, sm_query, sm_reset],
    description="State machine implementation for order workflow"
)


# Example 3: API Aggregator with Caching
# --------------------------------------

class APIAggregatorInput(BaseModel):
    """Input for API aggregation operations."""
    
    service: str = Field(description="Service to call: weather, news, stocks")
    operation: str = Field(description="Operation: fetch, cache_status, clear_cache")
    params: Optional[Dict[str, Any]] = Field(None, description="Service-specific parameters")


# Simple cache implementation
api_cache: Dict[str, Dict[str, Any]] = {}


async def fetch_weather(ctx: RunContext[Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch weather data (simulated)."""
    city = params.get('city', 'Unknown')
    cache_key = f"weather:{city}"
    
    if cache_key in api_cache:
        cached = api_cache[cache_key]
        if (datetime.now() - datetime.fromisoformat(cached['timestamp'])).seconds < 3600:
            return {**cached['data'], 'from_cache': True}
    
    # Simulate API call
    data = {
        'city': city,
        'temperature': 22,
        'condition': 'Sunny',
        'humidity': 65,
        'timestamp': datetime.now().isoformat()
    }
    
    api_cache[cache_key] = {'data': data, 'timestamp': datetime.now().isoformat()}
    return {**data, 'from_cache': False}


async def fetch_news(ctx: RunContext[Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch news data (simulated)."""
    category = params.get('category', 'general')
    cache_key = f"news:{category}"
    
    if cache_key in api_cache:
        cached = api_cache[cache_key]
        if (datetime.now() - datetime.fromisoformat(cached['timestamp'])).seconds < 1800:
            return {**cached['data'], 'from_cache': True}
    
    # Simulate API call
    data = {
        'category': category,
        'articles': [
            {'title': 'Breaking News 1', 'summary': 'Important event happened'},
            {'title': 'Breaking News 2', 'summary': 'Another significant development'}
        ],
        'timestamp': datetime.now().isoformat()
    }
    
    api_cache[cache_key] = {'data': data, 'timestamp': datetime.now().isoformat()}
    return {**data, 'from_cache': False}


async def fetch_stocks(ctx: RunContext[Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch stock data (simulated)."""
    symbol = params.get('symbol', 'AAPL')
    cache_key = f"stocks:{symbol}"
    
    if cache_key in api_cache:
        cached = api_cache[cache_key]
        if (datetime.now() - datetime.fromisoformat(cached['timestamp'])).seconds < 300:
            return {**cached['data'], 'from_cache': True}
    
    # Simulate API call
    data = {
        'symbol': symbol,
        'price': 150.25,
        'change': 2.5,
        'volume': 1000000,
        'timestamp': datetime.now().isoformat()
    }
    
    api_cache[cache_key] = {'data': data, 'timestamp': datetime.now().isoformat()}
    return {**data, 'from_cache': False}


async def check_cache_status(ctx: RunContext[Any], service: str) -> Dict[str, Any]:
    """Check cache status for a service."""
    service_keys = [k for k in api_cache.keys() if k.startswith(f"{service}:")]
    return {
        'service': service,
        'cached_items': len(service_keys),
        'cache_keys': service_keys,
        'total_cache_size': len(api_cache)
    }


async def clear_service_cache(ctx: RunContext[Any], service: str) -> Dict[str, Any]:
    """Clear cache for a specific service."""
    keys_to_remove = [k for k in api_cache.keys() if k.startswith(f"{service}:")]
    for key in keys_to_remove:
        del api_cache[key]
    return {
        'service': service,
        'cleared_items': len(keys_to_remove),
        'message': f"Cleared {len(keys_to_remove)} cached items for {service}"
    }


# For complex routing scenarios, we use a computed property approach
# This allows us to create compound routing keys from multiple fields
class APIAggregatorInputWithRouting(APIAggregatorInput):
    """Extended input with computed routing key."""
    
    @property
    def route_key(self) -> str:
        """Compute the routing key based on operation and service."""
        if self.operation == 'fetch':
            return f"fetch:{self.service}"
        return self.operation


# Create routing configuration using computed field for complex routing
api_routing = RoutingConfig(
    operation_field='route_key',  # Use the computed property
    operation_map={
        'fetch:weather': ('fetch_weather', lambda x: {'params': x.params or {}}),
        'fetch:news': ('fetch_news', lambda x: {'params': x.params or {}}),
        'fetch:stocks': ('fetch_stocks', lambda x: {'params': x.params or {}}),
        'cache_status': ('check_cache_status', lambda x: {'service': x.service}),
        'clear_cache': ('clear_service_cache', lambda x: {'service': x.service}),
    }
)

api_aggregator = create_agentool(
    name='api_aggregator',
    input_schema=APIAggregatorInputWithRouting,
    routing_config=api_routing,
    tools=[fetch_weather, fetch_news, fetch_stocks, check_cache_status, clear_service_cache],
    description="API aggregator with intelligent caching"
)


# Example 4: Demonstration Script
# -------------------------------

async def demonstrate_advanced_examples():
    """Run demonstrations of all advanced examples."""
    
    print("=== Database Agent Demo ===")
    # Insert a user
    result = await database_agent.run('''
    {
        "action": "insert",
        "table": "users",
        "data": {"name": "Alice", "email": "alice@example.com"}
    }
    ''')
    print(f"Insert result: {result.output}")
    
    # Query users
    result = await database_agent.run('''
    {
        "action": "query",
        "table": "users",
        "criteria": {"name": "Alice"}
    }
    ''')
    print(f"Query result: {result.output}")
    
    print("\n=== State Machine Demo ===")
    # Create and transition through order states
    result = await state_machine_agent.run('''
    {
        "command": "transition",
        "machine_id": "order-123",
        "event": "approve"
    }
    ''')
    print(f"Transition result: {result.output}")
    
    # Query state
    result = await state_machine_agent.run('''
    {
        "command": "query",
        "machine_id": "order-123"
    }
    ''')
    print(f"State query: {result.output}")
    
    print("\n=== API Aggregator Demo ===")
    # Fetch weather
    result = await api_aggregator.run('''
    {
        "service": "weather",
        "operation": "fetch",
        "params": {"city": "London"}
    }
    ''')
    print(f"Weather data: {result.output}")
    
    # Check cache
    result = await api_aggregator.run('''
    {
        "service": "weather",
        "operation": "cache_status"
    }
    ''')
    print(f"Cache status: {result.output}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_advanced_examples())