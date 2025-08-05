# AgenToolkits Documentation

This directory contains comprehensive documentation for all AgenToolkits in the system. AgenToolkits are specialized agents that provide specific functionality through well-defined interfaces and operations.

## Overview

AgenToolkits follow a consistent pattern with:
- **Standardized input/output schemas** using Pydantic models
- **Operation-based routing** for different functionality
- **Dependency management** between toolkits
- **Comprehensive error handling** with specific exception types
- **Integration capabilities** for building complex workflows

For guidelines on creating new AgenToolkits, see [CRAFTING_AGENTOOLS.md](../CRAFTING_AGENTOOLS.md).

## Available AgenToolkits

### Storage Systems

#### [storage-kv](storage-kv.md)
**Key-Value Storage with Redis-style Operations**
- In-memory key-value storage with TTL support
- Data type operations (strings, lists, sets, hashes, counters)
- Atomic operations and conditional updates
- Expiration and persistence management
- Tags: `storage`, `key-value`, `memory`, `ttl`

#### [storage-fs](storage-fs.md)
**File System Operations**
- File and directory operations (read, write, copy, move, delete)
- Path management and validation
- Content encoding support (text, binary, base64)
- Recursive operations and pattern matching
- Tags: `storage`, `filesystem`, `files`

### Security & Authentication

#### [crypto](crypto.md)
**Cryptographic Operations**
- Symmetric encryption (AES) and hashing (SHA, MD5)
- Password hashing with bcrypt and secure generation
- Digital signatures and key derivation
- Random data generation and encoding utilities
- Tags: `crypto`, `encryption`, `hashing`, `security`

#### [auth](auth.md)
**Authentication & Authorization**
- User authentication with multiple providers
- Session management and token validation
- Role-based access control (RBAC)
- JWT token operations and OAuth integration
- Tags: `auth`, `authentication`, `authorization`, `security`

#### [session](session.md)
**Session Management**
- HTTP session storage and retrieval
- Session data serialization and validation
- Expiration handling and cleanup
- Integration with auth systems
- Tags: `session`, `http`, `state-management`

### Communication & Integration

#### [http](http.md)
**HTTP Client Operations**
- HTTP requests with full method support (GET, POST, PUT, DELETE, etc.)
- Request/response handling with headers and authentication
- JSON and form data processing
- File upload and download capabilities
- Tags: `http`, `client`, `api`, `web`

### System Configuration & Management

#### [config](config.md)
**Configuration Management**
- Hierarchical configuration with dot notation keys
- Environment variable integration and reloading
- JSON schema validation and type conversion
- Namespace-based configuration isolation
- Tags: `config`, `configuration`, `environment`, `settings`

#### [logging](logging.md)
**Structured Logging**
- Multiple log levels and output formats (text, JSON)
- File and console output with automatic rotation
- Logger namespacing and level filtering
- Structured data logging with integration support
- Tags: `logging`, `monitoring`, `observability`, `structured-data`

#### [templates](templates.md)
**Template Rendering with Jinja2**
- Auto-loading templates from filesystem
- Variable resolution from storage systems
- Template validation and ad-hoc execution
- Strict and lenient rendering modes
- Tags: `templates`, `jinja2`, `rendering`, `storage-integration`

### Observability & Monitoring

#### [metrics](metrics.md)
**Metrics Collection & Analysis**
- Multiple metric types (counter, gauge, histogram, summary, timer)
- Automatic AgenTool execution tracking
- Statistical aggregations and time-series support
- Export formats (JSON, Prometheus, StatsD)
- Tags: `metrics`, `observability`, `monitoring`, `statistics`

### Automation & Orchestration

#### [scheduler](scheduler.md)
**Job Scheduling & Execution Engine**
- Multiple schedule types (cron, interval, one-time)
- AgenTool execution via injector system
- Job management (pause, resume, cancel)
- Integration with queue for reactive patterns
- Tags: `scheduler`, `execution`, `cron`, `automation`

#### [queue](queue.md)
**Message Queuing & Data Bus**
- FIFO message queuing with multiple named queues
- Delayed message delivery and auto-execution
- Dead Letter Queue (DLQ) support
- Reactive workflow patterns and retry mechanisms
- Tags: `queue`, `messaging`, `data-bus`, `reactive`

### Management & Introspection

#### [agentool](agentool.md)
**AgenTool Management & Introspection**
- Registry introspection and search capabilities
- Schema and routing configuration inspection
- Dependency analysis and validation
- Documentation generation in multiple formats
- Tags: `management`, `introspection`, `registry`, `documentation`

## Architecture Patterns

### Data Flow Patterns
- **Storage Layer**: `storage-kv` and `storage-fs` provide persistence
- **Processing Layer**: `templates`, `crypto`, `http` handle data transformation
- **Orchestration Layer**: `scheduler` and `queue` manage workflow execution
- **Observability Layer**: `logging` and `metrics` provide monitoring

### Integration Patterns
- **Configuration-Driven**: `config` + `templates` for dynamic content
- **Event-Driven**: `queue` + `scheduler` for reactive workflows  
- **Secure Processing**: `auth` + `crypto` + `session` for secure operations
- **Observable Systems**: `logging` + `metrics` for comprehensive monitoring

### Dependency Relationships
```
config → storage-kv
logging → storage-fs
templates → storage-kv, storage-fs, logging, metrics
metrics → storage-kv
scheduler → queue
queue → scheduler
agentool → (registry access, no dependencies)
```

## Common Use Cases

### Web Application Backend
```
http → auth → session → storage-kv + logging + metrics
```

### Data Processing Pipeline
```
queue → scheduler → templates → storage-fs + logging + metrics
```

### Configuration Management System
```
config → storage-kv → templates → scheduler + logging
```

### Secure API Gateway
```
http → auth → crypto → logging + metrics
```

## Testing

Each AgenToolkit includes comprehensive test coverage:
- Unit tests for all operations and error conditions
- Integration tests for cross-toolkit workflows
- Performance tests for scalability validation
- Security tests for auth and crypto operations

Test files are located at `tests/agentoolkit/test_{name}.py` for each toolkit.

## Development Guidelines

### Creating New AgenToolkits
1. Follow the patterns established in [CRAFTING_AGENTOOLS.md](../CRAFTING_AGENTOOLS.md)
2. Implement comprehensive error handling
3. Use consistent input/output schemas
4. Include thorough documentation and examples
5. Add comprehensive test coverage

### Best Practices
- **Single Responsibility**: Each toolkit should have a focused purpose
- **Consistent Interfaces**: Follow established patterns for input/output
- **Error Handling**: Provide specific exception types and clear messages
- **Documentation**: Include usage examples and integration patterns
- **Testing**: Ensure comprehensive coverage of all operations

### Integration Considerations
- **Dependency Management**: Declare dependencies explicitly
- **Async Operations**: Use async/await consistently
- **Resource Management**: Handle cleanup and resource limits
- **Observability**: Integrate with logging and metrics where appropriate

## Support & Troubleshooting

### Common Issues
- **Import Errors**: Check for missing dependencies (e.g., `jinja2`, `apscheduler`)
- **Permission Errors**: Verify file system and network permissions
- **Configuration Issues**: Validate input schemas and required parameters
- **Resource Limits**: Monitor memory usage and queue sizes

### Debugging
- Use the `logging` toolkit for structured debugging information
- Monitor execution with the `metrics` toolkit
- Inspect registry state with the `agentool` management toolkit
- Check queue states and DLQ contents for workflow issues

### Performance Optimization
- Use appropriate storage backends for your use case
- Configure queue sizes and timeout values appropriately
- Monitor metrics for performance bottlenecks
- Implement caching strategies where beneficial

For specific questions or issues, refer to the individual toolkit documentation or the test files for usage examples.