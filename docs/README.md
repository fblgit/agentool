# AgenTool Framework Documentation

This directory contains comprehensive documentation for the AgenTool framework - a deterministic tool execution framework for pydantic-ai.

## Documentation Overview

The documentation is organized into the following sections:

### 📖 **[Overview (index.md)](index.md)**
- Introduction to AgenTool
- Key features and benefits  
- Quick start guide
- When to use AgenTool vs LLM agents
- Basic usage examples

### 🏗️ **[Architecture Guide (architecture.md)](architecture.md)**
- Detailed system architecture
- Component relationships and data flow
- Design patterns and integration points
- Extension mechanisms
- Performance and security considerations

### 📚 **[API Reference (api-reference.md)](api-reference.md)**
- Complete API documentation for all modules
- Classes, methods, and functions
- Parameter specifications and return types
- Code examples for each API
- Usage patterns and best practices

### 💡 **[Usage Examples (usage-examples.md)](usage-examples.md)**
- Basic examples (calculator, storage, text processing)
- Common patterns (structured output, error handling, dynamic routing)
- Advanced use cases (multi-agent workflows, pipeline processing)
- Integration patterns (hybrid LLM + AgenTool, API gateway)
- Performance optimization patterns

### 🔌 **[Integration Guide (integration-guide.md)](integration-guide.md)**
- pydantic-ai integration patterns
- Web framework integration (FastAPI, Flask)
- Database integration (SQLAlchemy, Redis)
- Cloud deployment (Docker, Kubernetes)
- Monitoring and observability

### 🧪 **[Testing Guide (testing-guide.md)](testing-guide.md)**
- Testing philosophy and principles
- Unit testing strategies
- Integration testing with mocks
- Performance testing and benchmarking
- CI/CD integration

## Quick Navigation

### For New Users
1. Start with the **[Overview](index.md)** to understand what AgenTool is and its benefits
2. Follow the Quick Start guide to create your first AgenTool
3. Review **[Usage Examples](usage-examples.md)** for common patterns

### For Developers
1. Read the **[Architecture Guide](architecture.md)** to understand system design
2. Use the **[API Reference](api-reference.md)** for detailed implementation guidance
3. Follow the **[Testing Guide](testing-guide.md)** for comprehensive testing strategies

### For System Integrators
1. Review **[Integration Guide](integration-guide.md)** for deployment patterns
2. Check **[Architecture Guide](architecture.md)** for security and performance considerations
3. Use **[Testing Guide](testing-guide.md)** for production testing strategies

## Framework Structure

```
src/agentool/                    # Main framework code
├── __init__.py                  # Main exports and setup
├── base.py                      # Base schemas
├── factory.py                   # Agent creation utilities
└── core/                        # Core framework components
    ├── __init__.py              # Core exports
    ├── model.py                 # Synthetic LLM model provider
    ├── manager.py               # Routing and execution manager
    ├── registry.py              # Global configuration registry
    └── injector.py              # Dependency injection system

src/docs/                        # Documentation (this directory)
├── index.md                     # Overview and introduction
├── architecture.md             # Architecture and design
├── api-reference.md            # Complete API documentation
├── usage-examples.md           # Examples and patterns
├── integration-guide.md        # Integration and deployment
├── testing-guide.md            # Testing strategies
└── README.md                   # This file
```

## Key Concepts

### 🎯 **Deterministic Execution**
Unlike LLM-based agents that can produce varying outputs, AgenTool provides completely predictable behavior based on input schemas and routing configuration.

### 🔄 **Schema-Driven Routing**  
Uses Pydantic models for input validation and routing configuration to map operations to specific tool functions.

### 🧱 **Component Architecture**
- **AgenToolModel**: Synthetic LLM that converts JSON to tool calls
- **AgenToolManager**: Routes operations and transforms payloads  
- **AgenToolRegistry**: Global configuration and metadata storage
- **AgenToolInjector**: Dependency injection for multi-agent systems

### 🔌 **Seamless Integration**
Works alongside pydantic-ai agents, allowing hybrid systems that combine LLM intelligence with deterministic operations.

## Getting Started

### Installation
AgenTool is part of the pydantic-ai ecosystem:
```bash
pip install pydantic-ai
```

### Basic Example
```python
from agentool import create_agentool, RoutingConfig, BaseOperationInput
from pydantic import BaseModel
from typing import Literal

class CalculatorInput(BaseOperationInput):
    operation: Literal['add', 'subtract'] 
    a: float
    b: float

async def add(ctx, a: float, b: float) -> float:
    return a + b

async def subtract(ctx, a: float, b: float) -> float:
    return a - b

calculator = create_agentool(
    name='calculator',
    input_schema=CalculatorInput,
    routing_config=RoutingConfig(
        operation_map={
            'add': ('add', lambda x: {'a': x.a, 'b': x.b}),
            'subtract': ('subtract', lambda x: {'a': x.a, 'b': x.b})
        }
    ),
    tools=[add, subtract]
)

# Use like any pydantic-ai agent
result = await calculator.run('{"operation": "add", "a": 5, "b": 3}')
print(result.output)  # "8.0"
```

## Documentation Standards

All documentation follows these standards:

- **Fact-Based**: Every statement is based on actual code analysis
- **Executable Examples**: All code examples are tested and verified
- **Comprehensive Coverage**: Complete API documentation with usage examples
- **Clear Structure**: Consistent organization with table of contents
- **Cross-Referenced**: Documents link to related sections and concepts

## Contributing

This documentation is maintained alongside the AgenTool framework. When contributing:

1. Ensure all examples are executable and tested
2. Update relevant documentation sections for any code changes
3. Maintain consistent style and organization
4. Verify all links and cross-references work correctly

## Version Information

- **Framework Version**: 1.0.0
- **Documentation Version**: 1.0.0
- **Last Updated**: August 2025
- **pydantic-ai Compatibility**: Latest version

## Support and Community

For questions, issues, or contributions:

- Follow the main pydantic-ai project guidelines
- Use the existing issue tracking and contribution processes
- Refer to this documentation for comprehensive guidance

---

This documentation provides everything needed to understand, implement, test, and deploy AgenTool-based systems effectively. Start with the [Overview](index.md) and navigate to the sections most relevant to your needs.