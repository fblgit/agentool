# AgenTool Integration Guide

This guide covers how to integrate AgenTool with existing systems, frameworks, and deployment environments.

## Table of Contents

- [pydantic-ai Integration](#pydantic-ai-integration)
- [FastAPI Integration](#fastapi-integration)
- [Flask Integration](#flask-integration)
- [Database Integration](#database-integration)
- [Cloud Deployment](#cloud-deployment)
- [Monitoring and Observability](#monitoring-and-observability)
- [Best Practices](#best-practices)

## pydantic-ai Integration

AgenTool is designed to work seamlessly with pydantic-ai agents and can be mixed with LLM-based agents in the same application.

### Mixed Agent Systems

```python
from pydantic_ai import Agent
from agentool import create_agentool, RoutingConfig

# LLM-based agent for natural language processing
chat_agent = Agent(
    'openai:gpt-4',
    system_prompt="You are a helpful assistant that can answer questions and help with tasks."
)

# AgenTool for structured data operations
data_agent = create_agentool(
    name='data_processor',
    input_schema=DataProcessingInput,
    routing_config=data_routing,
    tools=[process_data, validate_data, transform_data]
)

# Orchestrator that decides which agent to use
async def intelligent_routing(user_input: str) -> str:
    """Route to appropriate agent based on input."""
    
    # Simple heuristics for routing
    if any(keyword in user_input.lower() for keyword in ['process', 'data', 'transform']):
        # Try to extract structured data from natural language
        extraction_prompt = f"""
        Extract structured data from this request: "{user_input}"
        
        If this is a data processing request, respond with JSON in this format:
        {{"operation": "process|validate|transform", "data": {{...}}}}
        
        If this is not a data processing request, respond with "NOT_STRUCTURED"
        """
        
        extraction_result = await chat_agent.run(extraction_prompt)
        
        try:
            import json
            structured_data = json.loads(extraction_result.output)
            if 'operation' in structured_data:
                # Use AgenTool for structured operation
                result = await data_agent.run(json.dumps(structured_data))
                return f"Data processing result: {result.output}"
        except json.JSONDecodeError:
            pass
    
    # Use LLM for general conversation
    result = await chat_agent.run(user_input)
    return result.output
```

### Agent Composition

```python
from agentool.core.injector import get_injector, InjectedDeps
from dataclasses import dataclass

@dataclass
class ComposedAgentDeps(InjectedDeps):
    """Dependencies that include both LLM and AgenTool agents."""
    
    async def enhanced_processing(self, user_request: str) -> Dict[str, Any]:
        """Combine LLM intelligence with structured processing."""
        
        # Step 1: Use LLM to understand and plan
        planning_agent = Agent('openai:gpt-4', system_prompt="You are a task planner.")
        plan_result = await planning_agent.run(f"""
        Create a processing plan for this request: "{user_request}"
        
        Respond with JSON containing:
        - "steps": array of processing steps
        - "data_operations": array of any data operations needed
        - "requires_llm": boolean if LLM processing is needed
        """)
        
        # Step 2: Execute structured operations if needed
        try:
            plan = json.loads(plan_result.output)
            results = {"plan": plan, "step_results": []}
            
            for step in plan.get("data_operations", []):
                step_result = await self.call_agent('data_processor', step)
                results["step_results"].append({
                    "step": step,
                    "result": step_result.output
                })
            
            return results
            
        except Exception as e:
            return {"error": str(e), "fallback": "Using LLM only"}

# Usage in a tool
async def enhanced_processor(ctx: RunContext[ComposedAgentDeps], request: str) -> Dict[str, Any]:
    """Enhanced processing combining multiple agent types."""
    return await ctx.deps.enhanced_processing(request)
```

## FastAPI Integration

AgenTool integrates easily with FastAPI for building web APIs.

### Basic REST API

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agentool.core.injector import get_injector
import json
from typing import Any, Dict

app = FastAPI(title="AgenTool API", version="1.0.0")

# Request/Response models
class AgentRequest(BaseModel):
    agent_name: str
    operation: str
    data: Dict[str, Any]

class AgentResponse(BaseModel):
    success: bool
    result: Any
    agent_name: str
    operation: str

@app.post("/agents/{agent_name}/execute", response_model=AgentResponse)
async def execute_agent(agent_name: str, request_data: Dict[str, Any]):
    """Execute an AgenTool agent operation."""
    try:
        injector = get_injector()
        
        # Run the agent
        result = await injector.run(agent_name, request_data)
        
        return AgentResponse(
            success=True,
            result=json.loads(result.output) if result.output.startswith('{') else result.output,
            agent_name=agent_name,
            operation=request_data.get('operation', 'unknown')
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/agents")
async def list_agents():
    """List all available agents."""
    from agentool.core.registry import AgenToolRegistry
    return {
        "agents": AgenToolRegistry.list_names(),
        "total": len(AgenToolRegistry.list_names())
    }

@app.get("/agents/{agent_name}/schema")
async def get_agent_schema(agent_name: str):
    """Get the input schema for an agent."""
    from agentool.core.registry import AgenToolRegistry
    
    schema = AgenToolRegistry.get_schema(agent_name)
    if schema is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    
    return {"schema": schema}

@app.get("/agents/{agent_name}/operations")
async def get_agent_operations(agent_name: str):
    """Get available operations for an agent."""
    from agentool.core.registry import AgenToolRegistry
    
    operations = AgenToolRegistry.get_operations(agent_name)
    if operations is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    
    return {"operations": operations}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "framework": "AgenTool"}

# Startup event to register agents
@app.on_event("startup")
async def startup_event():
    """Initialize agents on startup."""
    # Create and register your agents here
    calculator = create_agentool(...)  # Your agent definitions
    storage = create_agentool(...)
    
    print("AgenTool API started successfully")

# Example usage
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Advanced FastAPI Integration with Middleware

```python
from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
import time
import logging

class AgenToolMiddleware(BaseHTTPMiddleware):
    """Middleware for AgenTool request logging and monitoring."""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logging.info(f"AgenTool request: {request.method} {request.url}")
        
        response = await call_next(request)
        
        # Log response time
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        logging.info(f"AgenTool response: {response.status_code} in {process_time:.4f}s")
        
        return response

app.add_middleware(AgenToolMiddleware)

# Batch processing endpoint
@app.post("/agents/{agent_name}/batch")
async def execute_batch(agent_name: str, requests: list[Dict[str, Any]]):
    """Execute multiple operations in batch."""
    injector = get_injector()
    results = []
    
    for request_data in requests:
        try:
            result = await injector.run(agent_name, request_data)
            results.append({
                "success": True,
                "result": result.output,
                "request": request_data
            })
        except Exception as e:
            results.append({
                "success": False,
                "error": str(e),
                "request": request_data
            })
    
    return {"results": results, "total": len(requests)}
```

## Flask Integration

Integration with Flask for traditional web applications.

```python
from flask import Flask, request, jsonify
from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry
import asyncio
import json

app = Flask(__name__)

# Helper function to run async code in Flask
def run_async(coro):
    """Run async coroutine in Flask context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

@app.route('/agents/<agent_name>/execute', methods=['POST'])
def execute_agent(agent_name):
    """Execute an AgenTool agent."""
    try:
        request_data = request.get_json()
        injector = get_injector()
        
        # Run async operation
        result = run_async(injector.run(agent_name, request_data))
        
        return jsonify({
            "success": True,
            "result": json.loads(result.output) if result.output.startswith('{') else result.output,
            "agent_name": agent_name
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/agents', methods=['GET'])
def list_agents():
    """List all available agents."""
    return jsonify({
        "agents": AgenToolRegistry.list_names(),
        "total": len(AgenToolRegistry.list_names())
    })

@app.route('/agents/<agent_name>/schema', methods=['GET'])
def get_agent_schema(agent_name):
    """Get agent schema."""
    schema = AgenToolRegistry.get_schema(agent_name)
    if schema is None:
        return jsonify({"error": f"Agent '{agent_name}' not found"}), 404
    
    return jsonify({"schema": schema})

# Initialize agents
def initialize_agents():
    """Initialize all agents."""
    # Create your agents here
    calculator = create_agentool(...)
    storage = create_agentool(...)

if __name__ == '__main__':
    initialize_agents()
    app.run(debug=True, host='0.0.0.0', port=5000)
```

## Database Integration

### SQLAlchemy Integration

```python
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

Base = declarative_base()

class AgentExecution(Base):
    """Track agent executions in database."""
    __tablename__ = 'agent_executions'
    
    id = Column(Integer, primary_key=True)
    agent_name = Column(String(100), nullable=False)
    operation = Column(String(100))
    input_data = Column(Text)
    output_data = Column(Text)
    success = Column(String(10))
    error_message = Column(Text)
    execution_time = Column(DateTime, default=datetime.utcnow)
    duration_ms = Column(Integer)

# Database-aware AgenTool wrapper
class DatabaseAgenTool:
    """Wrapper for AgenTool with database logging."""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        self.injector = get_injector()
    
    async def run_with_logging(self, agent_name: str, input_data: Any) -> Any:
        """Run agent with database logging."""
        session = self.SessionLocal()
        start_time = time.time()
        
        try:
            # Execute agent
            result = await self.injector.run(agent_name, input_data)
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Log successful execution
            execution = AgentExecution(
                agent_name=agent_name,
                operation=input_data.get('operation') if isinstance(input_data, dict) else 'unknown',
                input_data=json.dumps(input_data) if not isinstance(input_data, str) else input_data,
                output_data=result.output,
                success='success',
                duration_ms=duration_ms
            )
            session.add(execution)
            session.commit()
            
            return result
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Log failed execution
            execution = AgentExecution(
                agent_name=agent_name,
                operation=input_data.get('operation') if isinstance(input_data, dict) else 'unknown',
                input_data=json.dumps(input_data) if not isinstance(input_data, str) else input_data,
                success='error',
                error_message=str(e),
                duration_ms=duration_ms
            )
            session.add(execution)
            session.commit()
            
            raise
        finally:
            session.close()
    
    def get_execution_history(self, agent_name: str = None, limit: int = 100):
        """Get execution history."""
        session = self.SessionLocal()
        try:
            query = session.query(AgentExecution)
            if agent_name:
                query = query.filter(AgentExecution.agent_name == agent_name)
            
            executions = query.order_by(AgentExecution.execution_time.desc()).limit(limit).all()
            
            return [{
                'id': ex.id,
                'agent_name': ex.agent_name,
                'operation': ex.operation,
                'success': ex.success,
                'duration_ms': ex.duration_ms,
                'execution_time': ex.execution_time.isoformat()
            } for ex in executions]
            
        finally:
            session.close()

# Usage
db_agent_tool = DatabaseAgenTool('sqlite:///agentool.db')

async def logged_execution():
    result = await db_agent_tool.run_with_logging('calculator', {
        'operation': 'add',
        'a': 5,
        'b': 3
    })
    print(f"Result: {result.output}")
    
    # Get history
    history = db_agent_tool.get_execution_history('calculator')
    print(f"Recent executions: {len(history)}")
```

### Redis Caching Integration

```python
import redis
import json
import hashlib
from typing import Optional

class RedisAgenTool:
    """AgenTool with Redis caching."""
    
    def __init__(self, redis_url: str = 'redis://localhost:6379', cache_ttl: int = 3600):
        self.redis_client = redis.from_url(redis_url)
        self.cache_ttl = cache_ttl
        self.injector = get_injector()
    
    def _generate_cache_key(self, agent_name: str, input_data: Any) -> str:
        """Generate cache key for input."""
        # Create a hash of the input for consistent caching
        input_str = json.dumps(input_data, sort_keys=True) if not isinstance(input_data, str) else input_data
        input_hash = hashlib.sha256(input_str.encode()).hexdigest()[:16]
        return f"agentool:{agent_name}:{input_hash}"
    
    async def run_with_cache(self, agent_name: str, input_data: Any, use_cache: bool = True) -> Any:
        """Run agent with Redis caching."""
        
        if use_cache:
            # Try to get from cache
            cache_key = self._generate_cache_key(agent_name, input_data)
            cached_result = self.redis_client.get(cache_key)
            
            if cached_result:
                # Return cached result
                cached_data = json.loads(cached_result)
                # Create a mock result object
                class CachedResult:
                    def __init__(self, output):
                        self.output = output
                        self.cached = True
                
                return CachedResult(cached_data['output'])
        
        # Execute agent
        result = await self.injector.run(agent_name, input_data)
        
        if use_cache:
            # Cache the result
            cache_data = {
                'output': result.output,
                'agent_name': agent_name,
                'cached_at': time.time()
            }
            self.redis_client.setex(
                cache_key, 
                self.cache_ttl, 
                json.dumps(cache_data)
            )
        
        return result
    
    def clear_cache(self, agent_name: str = None):
        """Clear cache for specific agent or all agents."""
        if agent_name:
            pattern = f"agentool:{agent_name}:*"
        else:
            pattern = "agentool:*"
        
        keys = self.redis_client.keys(pattern)
        if keys:
            self.redis_client.delete(*keys)
        
        return len(keys)
    
    def get_cache_stats(self, agent_name: str = None) -> Dict[str, Any]:
        """Get cache statistics."""
        if agent_name:
            pattern = f"agentool:{agent_name}:*"
        else:
            pattern = "agentool:*"
        
        keys = self.redis_client.keys(pattern)
        
        total_size = 0
        for key in keys:
            total_size += self.redis_client.memory_usage(key) or 0
        
        return {
            'total_keys': len(keys),
            'total_size_bytes': total_size,
            'pattern': pattern
        }

# Usage
redis_agent = RedisAgenTool('redis://localhost:6379', cache_ttl=1800)

async def cached_execution():
    # First call (cache miss)
    start = time.time()
    result1 = await redis_agent.run_with_cache('calculator', {'operation': 'add', 'a': 5, 'b': 3})
    first_time = time.time() - start
    
    # Second call (cache hit)
    start = time.time()
    result2 = await redis_agent.run_with_cache('calculator', {'operation': 'add', 'a': 5, 'b': 3})
    second_time = time.time() - start
    
    print(f"First call: {first_time:.4f}s")
    print(f"Second call (cached): {second_time:.4f}s")
    print(f"Cache hit: {hasattr(result2, 'cached') and result2.cached}")
```

## Cloud Deployment

### Docker Deployment

```dockerfile
# Dockerfile for AgenTool application
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  agentool-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:password@db:5432/agentool
    depends_on:
      - redis
      - db
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    
  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=agentool
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
```

### Kubernetes Deployment

```yaml
# agentool-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentool-api
  labels:
    app: agentool-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agentool-api
  template:
    metadata:
      labels:
        app: agentool-api
    spec:
      containers:
      - name: agentool-api
        image: your-registry/agentool-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: agentool-secrets
              key: database-url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: agentool-api-service
spec:
  selector:
    app: agentool-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: v1
kind: Secret
metadata:
  name: agentool-secrets
type: Opaque
stringData:
  database-url: "postgresql://user:password@postgres-service:5432/agentool"
```

## Monitoring and Observability

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

# Metrics
agent_requests_total = Counter('agentool_requests_total', 'Total agent requests', ['agent_name', 'operation', 'status'])
agent_request_duration = Histogram('agentool_request_duration_seconds', 'Request duration', ['agent_name', 'operation'])
active_agents = Gauge('agentool_active_agents', 'Number of active agents')

class MonitoredAgenTool:
    """AgenTool with Prometheus monitoring."""
    
    def __init__(self):
        self.injector = get_injector()
        self._update_active_agents()
    
    def _update_active_agents(self):
        """Update active agents gauge."""
        from agentool.core.registry import AgenToolRegistry
        active_agents.set(len(AgenToolRegistry.list_names()))
    
    async def run_monitored(self, agent_name: str, input_data: Any) -> Any:
        """Run agent with monitoring."""
        operation = input_data.get('operation', 'unknown') if isinstance(input_data, dict) else 'unknown'
        
        start_time = time.time()
        status = 'success'
        
        try:
            with agent_request_duration.labels(agent_name=agent_name, operation=operation).time():
                result = await self.injector.run(agent_name, input_data)
            
            return result
            
        except Exception as e:
            status = 'error'
            raise
        finally:
            # Record metrics
            agent_requests_total.labels(
                agent_name=agent_name, 
                operation=operation, 
                status=status
            ).inc()

# Add metrics endpoint to FastAPI
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain")
```

### Structured Logging

```python
import logging
import json
from datetime import datetime

class AgenToolLogger:
    """Structured logging for AgenTool."""
    
    def __init__(self, logger_name: str = "agentool"):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        
        # JSON formatter
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}'
        )
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_execution(self, agent_name: str, operation: str, status: str, 
                     duration_ms: int, input_data: Any = None, error: str = None):
        """Log agent execution."""
        log_data = {
            "event": "agent_execution",
            "agent_name": agent_name,
            "operation": operation,
            "status": status,
            "duration_ms": duration_ms,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if input_data:
            log_data["input_size"] = len(str(input_data))
        
        if error:
            log_data["error"] = error
        
        if status == "success":
            self.logger.info(json.dumps(log_data))
        else:
            self.logger.error(json.dumps(log_data))

# Usage with logging
logger = AgenToolLogger()

class LoggedAgenTool:
    """AgenTool with structured logging."""
    
    def __init__(self):
        self.injector = get_injector()
        self.logger = AgenToolLogger()
    
    async def run_logged(self, agent_name: str, input_data: Any) -> Any:
        """Run agent with logging."""
        operation = input_data.get('operation', 'unknown') if isinstance(input_data, dict) else 'unknown'
        start_time = time.time()
        
        try:
            result = await self.injector.run(agent_name, input_data)
            duration_ms = int((time.time() - start_time) * 1000)
            
            self.logger.log_execution(
                agent_name=agent_name,
                operation=operation,
                status="success",
                duration_ms=duration_ms,
                input_data=input_data
            )
            
            return result
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            
            self.logger.log_execution(
                agent_name=agent_name,
                operation=operation,
                status="error",
                duration_ms=duration_ms,
                input_data=input_data,
                error=str(e)
            )
            
            raise
```

## Best Practices

### Configuration Management

```python
from pydantic import BaseSettings
from typing import Optional

class AgenToolConfig(BaseSettings):
    """Configuration management for AgenTool applications."""
    
    # Redis configuration
    redis_url: str = "redis://localhost:6379"
    redis_cache_ttl: int = 3600
    
    # Database configuration
    database_url: str = "sqlite:///agentool.db"
    
    # API configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    
    # Monitoring
    enable_metrics: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"
    
    # Performance
    max_concurrent_requests: int = 100
    request_timeout: int = 30
    
    class Config:
        env_file = ".env"

# Global config instance
config = AgenToolConfig()

# Use config in application
def create_app():
    """Create application with configuration."""
    app = FastAPI(title="AgenTool API")
    
    if config.enable_metrics:
        # Add metrics
        pass
    
    if config.enable_logging:
        # Configure logging
        logging.basicConfig(level=config.log_level)
    
    return app
```

### Error Handling and Resilience

```python
from tenacity import retry, stop_after_attempt, wait_exponential
import circuit_breaker

class ResilientAgenTool:
    """AgenTool with resilience patterns."""
    
    def __init__(self):
        self.injector = get_injector()
        self.circuit_breaker = circuit_breaker.CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30,
            expected_exception=Exception
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def run_resilient(self, agent_name: str, input_data: Any) -> Any:
        """Run agent with retry logic."""
        
        @self.circuit_breaker
        async def protected_run():
            return await self.injector.run(agent_name, input_data)
        
        return await protected_run()

# Usage
resilient_tool = ResilientAgenTool()

async def resilient_execution():
    try:
        result = await resilient_tool.run_resilient('calculator', {
            'operation': 'add',
            'a': 5,
            'b': 3
        })
        print(f"Result: {result.output}")
    except Exception as e:
        print(f"Failed after retries: {e}")
```

### Security Considerations

```python
from functools import wraps
import jwt
from datetime import datetime, timedelta

def require_auth(f):
    """Decorator for authentication."""
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        # Implement authentication logic
        token = request.headers.get('Authorization')
        if not token or not validate_token(token):
            raise HTTPException(status_code=401, detail="Unauthorized")
        return await f(*args, **kwargs)
    return decorated_function

def validate_token(token: str) -> bool:
    """Validate JWT token."""
    try:
        jwt.decode(token.replace('Bearer ', ''), 'secret', algorithms=['HS256'])
        return True
    except jwt.InvalidTokenError:
        return False

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

@app.post("/agents/{agent_name}/execute")
@limiter.limit("10/minute")
@require_auth
async def secure_execute_agent(request: Request, agent_name: str, request_data: Dict[str, Any]):
    """Secure agent execution with rate limiting and auth."""
    # Implementation
    pass
```

This integration guide provides comprehensive examples for deploying and integrating AgenTool in production environments with proper monitoring, security, and resilience patterns.