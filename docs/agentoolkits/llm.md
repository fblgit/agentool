# LLM AgenToolkit

> **Note**: For guidelines on creating new AgenToolkits, see [CRAFTING_AGENTOOLS.md](../CRAFTING_AGENTOOLS.md). For testing patterns and examples, refer to [tests/agentoolkit/test_llm.py](../../tests/agentoolkit/test_llm.py).

## Overview

The LLM AgenToolkit provides comprehensive natural language processing capabilities using Large Language Models. It offers a suite of AI-powered operations including text summarization, classification, extraction, generation, translation, sentiment analysis, markdown conversion, and completion. Each operation leverages pydantic-ai's Agent framework with template-based prompts for consistent and reliable results.

### Key Features
- **Text Summarization**: Condense long content to key points with configurable length and style
- **Markdown Conversion**: Transform plain text into well-formatted markdown with headers, lists, and code blocks
- **Text Classification**: Categorize text into predefined or dynamic classes with confidence scoring
- **Content Generation**: Create new content based on prompts with temperature and token controls
- **Data Extraction**: Extract structured data from unstructured text using dynamic schemas
- **Language Translation**: Translate between multiple languages with formality controls
- **Sentiment Analysis**: Analyze emotional tone with granular emotion detection
- **Text Completion**: Complete partial text or code with contextual awareness
- **Result Caching**: Built-in caching with TTL for improved performance and cost control
- **Model Flexibility**: Support for multiple LLM models with easy switching
- **Template-Driven**: Extensible prompt system using Jinja2 templates

## Creation Method

```python
from agentoolkit.llm.llm import create_llm_agent

# Create the agent
agent = create_llm_agent()
```

The creation function returns a fully configured AgenTool with name `'llm'`.

## Input Schema

### LLMInput

The input schema is a Pydantic model with the following fields:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `operation` | `Literal['summary', 'markdownify', 'classification', 'generation', 'extraction', 'translation', 'sentiment', 'completion']` | Yes | - | The LLM operation to perform |
| `content` | `str` | Yes | - | The input text to process |
| `options` | `Optional[Dict[str, Any]]` | No | None | Operation-specific options (e.g., max_length, temperature) |
| `extraction_schema` | `Optional[Dict[str, Any]]` | No | None | Schema for extraction operation (field_name -> type) |
| `classes` | `Optional[List[str]]` | No | None | List of classes for classification operation |
| `target_language` | `Optional[str]` | No | None | Target language for translation (e.g., 'es', 'fr', 'de') |
| `prompt` | `Optional[str]` | No | None | Additional prompt for generation or completion |
| `model` | `Optional[str]` | No | "openai:gpt-4o" | LLM model to use (can override default) |
| `cache_key` | `Optional[str]` | No | None | Optional cache key for storing/retrieving results |

## Operations Schema

The routing configuration maps operations to tool functions:

| Operation | Tool Function | Required Parameters | Description |
|-----------|--------------|-------------------|-------------|
| `summary` | `llm_summary` | `content`, `options`, `model`, `cache_key` | Generate a summary of the provided text |
| `markdownify` | `llm_markdownify` | `content`, `options`, `model`, `cache_key` | Convert text to well-formatted markdown |
| `classification` | `llm_classification` | `content`, `classes`, `options`, `model`, `cache_key` | Classify text into predefined categories |
| `generation` | `llm_generation` | `content`, `prompt`, `options`, `model`, `cache_key` | Generate new content based on input and prompt |
| `extraction` | `llm_extraction` | `content`, `extraction_schema`, `options`, `model`, `cache_key` | Extract structured data from unstructured text |
| `translation` | `llm_translation` | `content`, `target_language`, `options`, `model`, `cache_key` | Translate text to target language |
| `sentiment` | `llm_sentiment` | `content`, `options`, `model`, `cache_key` | Analyze sentiment and emotional tone |
| `completion` | `llm_completion` | `content`, `prompt`, `options`, `model`, `cache_key` | Complete partial text or code |

## Output Schema

### LLMOutput

All operations return an `LLMOutput` model with:

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Whether the operation succeeded |
| `operation` | `str` | The operation that was performed |
| `message` | `str` | Human-readable result message |
| `data` | `Dict[str, Any]` | Operation-specific data |
| `model_used` | `str` | The model that was used |
| `tokens_used` | `Optional[Dict[str, int]]` | Token usage statistics (request, response, total) |
| `cached` | `bool` | Whether result was from cache |

### Operation-Specific Data Fields

- **summary**: `summary`, `original_length`, `summary_length`, `compression_ratio`
- **markdownify**: `markdown`, `has_headers`, `has_code_blocks`, `has_lists`
- **classification**: `selected_class`, `confidence`, `reasoning`, `all_classes`
- **generation**: `generated`, `seed_length`, `generated_length`, `parameters`
- **extraction**: `extracted`, `schema_used`, `fields_extracted`
- **translation**: `translated`, `source_length`, `translated_length`, `target_language`, `language_name`
- **sentiment**: `sentiment`, `score`, `emotions`, `confidence`
- **completion**: `completed`, `original`, `completion_only`, `total_length`

## Dependencies

This AgenToolkit depends on:
- **templates**: For rendering system and user prompts using Jinja2 templates
- **storage_kv**: For result caching functionality (optional)

## Tools

### llm_summary
```python
async def llm_summary(ctx: RunContext[Any], content: str, options: Optional[Dict[str, Any]], model: str, cache_key: Optional[str]) -> LLMOutput
```
Generate a summary of the provided text with configurable length and style.

**Options:**
- `max_length`: Maximum summary length in words (default: 200)
- `style`: Summary style ('concise', 'detailed', 'bullet') (default: 'concise')

**Raises:**
- `RuntimeError`: If summary generation fails

### llm_markdownify
```python
async def llm_markdownify(ctx: RunContext[Any], content: str, options: Optional[Dict[str, Any]], model: str, cache_key: Optional[str]) -> LLMOutput
```
Convert text to well-formatted markdown with headers, lists, and code blocks.

**Options:**
- `include_toc`: Include table of contents (default: False)
- `code_style`: Code block style ('fenced', 'indented') (default: 'fenced')

**Raises:**
- `RuntimeError`: If markdown conversion fails

### llm_classification
```python
async def llm_classification(ctx: RunContext[Any], content: str, classes: Optional[List[str]], options: Optional[Dict[str, Any]], model: str, cache_key: Optional[str]) -> LLMOutput
```
Classify text into predefined categories with confidence scoring.

**Options:**
- `include_reasoning`: Include classification reasoning (default: False)

**Raises:**
- `ValueError`: If classes are not provided
- `RuntimeError`: If classification fails

### llm_generation
```python
async def llm_generation(ctx: RunContext[Any], content: str, prompt: Optional[str], options: Optional[Dict[str, Any]], model: str, cache_key: Optional[str]) -> LLMOutput
```
Generate new content based on input and prompt with controllable creativity.

**Options:**
- `temperature`: Creativity level (0.0-1.0) (default: 0.7)
- `max_tokens`: Maximum tokens to generate (default: 500)
- `style`: Generation style ('creative', 'factual', 'formal') (default: 'creative')

**Raises:**
- `RuntimeError`: If content generation fails

### llm_extraction
```python
async def llm_extraction(ctx: RunContext[Any], content: str, extraction_schema: Optional[Dict[str, Any]], options: Optional[Dict[str, Any]], model: str, cache_key: Optional[str]) -> LLMOutput
```
Extract structured data from unstructured text using dynamic schemas.

**Schema Types:** `str`, `int`, `float`, `bool`, `list`, `dict`

**Options:**
- `strict`: Enforce strict schema compliance (default: True)

**Raises:**
- `ValueError`: If extraction_schema is not provided
- `RuntimeError`: If extraction fails

### llm_translation
```python
async def llm_translation(ctx: RunContext[Any], content: str, target_language: Optional[str], options: Optional[Dict[str, Any]], model: str, cache_key: Optional[str]) -> LLMOutput
```
Translate text to target language with formality controls.

**Supported Languages:** English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Russian (ru), Japanese (ja), Korean (ko), Chinese (zh), Arabic (ar), Hindi (hi)

**Options:**
- `preserve_formatting`: Maintain original formatting (default: True)
- `formality`: Translation formality ('formal', 'informal', 'neutral') (default: 'neutral')

**Raises:**
- `ValueError`: If target_language is not provided
- `RuntimeError`: If translation fails

### llm_sentiment
```python
async def llm_sentiment(ctx: RunContext[Any], content: str, options: Optional[Dict[str, Any]], model: str, cache_key: Optional[str]) -> LLMOutput
```
Analyze sentiment and emotional tone with granular emotion detection.

**Options:**
- `include_emotions`: Include detailed emotion analysis (default: True)
- `granularity`: Analysis detail level ('basic', 'detailed') (default: 'detailed')

**Raises:**
- `RuntimeError`: If sentiment analysis fails

### llm_completion
```python
async def llm_completion(ctx: RunContext[Any], content: str, prompt: Optional[str], options: Optional[Dict[str, Any]], model: str, cache_key: Optional[str]) -> LLMOutput
```
Complete partial text or code with contextual awareness.

**Options:**
- `type`: Completion type ('text', 'code') (default: 'text')
- `max_tokens`: Maximum tokens to generate (default: 200)

**Raises:**
- `RuntimeError`: If completion fails

## Exceptions

| Exception Type | Scenarios |
|---------------|-----------|
| `ValueError` | - Missing required schema for extraction<br>- Missing classes for classification<br>- Missing target language for translation |
| `RuntimeError` | - LLM API failures<br>- Template rendering errors<br>- Model configuration issues |

## Usage Examples

### Text Summarization
```python
from agentoolkit.llm.llm import create_llm_agent
from agentool.core.injector import get_injector

# Create and register the agent
agent = create_llm_agent()
injector = get_injector()

# Summarize long text
result = await injector.run('llm', {
    "operation": "summary",
    "content": "Long article text about AI developments...",
    "options": {
        "max_length": 150,
        "style": "concise"
    },
    "cache_key": "ai_article_summary"
})

print(f"Original length: {result.data['original_length']}")
print(f"Summary: {result.data['summary']}")
print(f"Compression ratio: {result.data['compression_ratio']:.2f}")
```

### Text Classification
```python
# Classify customer feedback
result = await injector.run('llm', {
    "operation": "classification",
    "content": "This product exceeded all my expectations! Absolutely fantastic!",
    "classes": ["positive", "negative", "neutral"],
    "options": {"include_reasoning": True}
})

print(f"Classification: {result.data['selected_class']}")
print(f"Confidence: {result.data['confidence']}")
print(f"Reasoning: {result.data['reasoning']}")
```

### Structured Data Extraction
```python
# Extract structured data from text
content = "John Doe, CEO of Acme Corp, announced a new product launch in Q2 2024."

result = await injector.run('llm', {
    "operation": "extraction",
    "content": content,
    "extraction_schema": {
        "name": "str",
        "title": "str", 
        "company": "str",
        "quarter": "str"
    },
    "options": {"strict": True}
})

extracted_data = result.data['extracted']
print(f"Name: {extracted_data['name']}")
print(f"Title: {extracted_data['title']}")
print(f"Company: {extracted_data['company']}")
```

### Language Translation
```python
# Translate to Spanish
result = await injector.run('llm', {
    "operation": "translation",
    "content": "Hello, how are you today?",
    "target_language": "es",
    "options": {
        "preserve_formatting": True,
        "formality": "informal"
    }
})

print(f"Original: Hello, how are you today?")
print(f"Spanish: {result.data['translated']}")
print(f"Language: {result.data['language_name']}")
```

### Sentiment Analysis
```python
# Analyze customer review sentiment
result = await injector.run('llm', {
    "operation": "sentiment",
    "content": "I absolutely love this new feature! It's amazing and works perfectly.",
    "options": {
        "include_emotions": True,
        "granularity": "detailed"
    }
})

print(f"Sentiment: {result.data['sentiment']}")
print(f"Score: {result.data['score']:.2f}")
print(f"Emotions: {result.data['emotions']}")
```

### Content Generation
```python
# Generate creative content
result = await injector.run('llm', {
    "operation": "generation",
    "content": "A story about a robot learning to paint",
    "prompt": "Write a short creative story",
    "options": {
        "temperature": 0.8,
        "max_tokens": 300,
        "style": "creative"
    }
})

print(f"Generated story: {result.data['generated']}")
print(f"Length: {result.data['generated_length']} characters")
```

### Markdown Conversion
```python
# Convert plain text to markdown
plain_text = """
Title of Document

First Section
This is a paragraph with important points:
Point 1 about something
Point 2 about another thing

Second Section
Code example: print("hello")
"""

result = await injector.run('llm', {
    "operation": "markdownify",
    "content": plain_text,
    "options": {
        "include_toc": True,
        "code_style": "fenced"
    }
})

print(f"Markdown: {result.data['markdown']}")
print(f"Has headers: {result.data['has_headers']}")
print(f"Has code blocks: {result.data['has_code_blocks']}")
```

### Text Completion
```python
# Complete partial text
result = await injector.run('llm', {
    "operation": "completion",
    "content": "The future of artificial intelligence will",
    "prompt": "Complete this sentence naturally",
    "options": {
        "max_tokens": 100,
        "type": "text"
    }
})

print(f"Original: {result.data['original']}")
print(f"Completed: {result.data['completed']}")
```

## Integration Patterns

### With Storage for Caching
```python
# Long-term result caching
result = await injector.run('llm', {
    "operation": "summary", 
    "content": document_text,
    "cache_key": f"doc_summary_{document_id}",
    "options": {"max_length": 200}
})

# Subsequent calls with same cache_key return cached results
```

### With Workflow Systems
```python
# Chain multiple LLM operations
async def process_document_workflow(document):
    # Step 1: Extract key data
    extraction_result = await injector.run('llm', {
        "operation": "extraction",
        "content": document,
        "extraction_schema": {"title": "str", "author": "str", "topics": "list"}
    })
    
    # Step 2: Classify document type
    classification_result = await injector.run('llm', {
        "operation": "classification", 
        "content": document,
        "classes": ["research", "news", "opinion", "tutorial"]
    })
    
    # Step 3: Generate summary
    summary_result = await injector.run('llm', {
        "operation": "summary",
        "content": document,
        "options": {"max_length": 100}
    })
    
    return {
        "extracted": extraction_result.data['extracted'],
        "category": classification_result.data['selected_class'],
        "summary": summary_result.data['summary']
    }
```

### Model Switching
```python
# Use different models for different operations
fast_result = await injector.run('llm', {
    "operation": "classification",
    "content": text,
    "classes": ["spam", "ham"],
    "model": "openai:gpt-4o-mini"  # Fast, cheap model
})

quality_result = await injector.run('llm', {
    "operation": "generation",
    "content": text,
    "prompt": "Write professional content",
    "model": "openai:gpt-4o"  # High-quality model
})
```

## Error Handling

### Input Validation
```python
try:
    result = await injector.run('llm', {
        "operation": "extraction",
        "content": "Some text"
        # Missing required extraction_schema
    })
except ValueError as e:
    print(f"Validation error: {e}")
    # Handle missing required fields
```

### API Failures
```python
try:
    result = await injector.run('llm', {
        "operation": "summary",
        "content": text,
        "model": "invalid:model"
    })
except RuntimeError as e:
    print(f"LLM API error: {e}")
    # Handle API failures, retry with different model
```

### Graceful Degradation
```python
async def safe_llm_operation(content, operation, **kwargs):
    """Wrapper with fallback strategies."""
    try:
        return await injector.run('llm', {
            "operation": operation,
            "content": content,
            **kwargs
        })
    except Exception as e:
        # Log error and return fallback
        await injector.run('logging', {
            "operation": "log",
            "level": "error",
            "message": f"LLM operation failed: {e}",
            "metadata": {"operation": operation}
        })
        
        # Return fallback result
        return LLMOutput(
            success=False,
            operation=operation,
            message=f"Operation failed: {e}",
            data={},
            model_used="none"
        )
```

## Best Practices

### Performance Optimization
```python
# 1. Use appropriate models for tasks
lightweight_tasks = ["classification", "sentiment"]
complex_tasks = ["generation", "extraction"]

model = "openai:gpt-4o-mini" if operation in lightweight_tasks else "openai:gpt-4o"

# 2. Implement caching for repeated operations
cache_key = f"{operation}_{hash(content)}_{str(options)}"

# 3. Batch similar operations when possible
results = await asyncio.gather(*[
    injector.run('llm', {"operation": "sentiment", "content": text, "model": "openai:gpt-4o-mini"})
    for text in text_batch
])
```

### Template Customization
```python
# Extend templates for domain-specific prompts
custom_templates = {
    "system/llm/custom_extraction": "You are a legal document analyzer...",
    "prompts/llm/custom_extraction": "Extract legal entities from: {{ content }}"
}

# Use templates agent to register custom templates
await injector.run('templates', {
    "operation": "register",
    "templates": custom_templates
})
```

### Token Management
```python
# Monitor token usage for cost control
result = await injector.run('llm', {
    "operation": "generation",
    "content": content,
    "options": {"max_tokens": 100}  # Limit output
})

tokens_used = result.tokens_used['total']
await injector.run('metrics', {
    "operation": "record",
    "metric": "llm_tokens_used",
    "value": tokens_used,
    "tags": {"operation": "generation", "model": result.model_used}
})
```

## Testing

The test suite is located at `tests/agentoolkit/test_llm.py`. Tests cover:
- All eight LLM operations with realistic inputs
- Input validation for required fields
- Result caching functionality  
- Model override capabilities
- Token usage tracking
- Template rendering integration
- Error handling scenarios
- Performance benchmarks

To run tests:
```bash
pytest tests/agentoolkit/test_llm.py -v
```

## Related AgenToolkits

- **templates**: Required for prompt rendering and template management
- **storage_kv**: Used for result caching and performance optimization
- **metrics**: Useful for tracking token usage and operation performance
- **logging**: Helpful for debugging and monitoring LLM operations

## Notes

- All operations support caching with configurable TTL (default: 1 hour)
- Template system allows easy customization of prompts for specific domains
- Token usage is tracked and returned for cost monitoring
- Model switching allows optimization between speed and quality
- Structured outputs use dynamic Pydantic models for type safety
- Language codes follow ISO 639-1 standards for translation
- Sentiment scores range from -1.0 (very negative) to 1.0 (very positive)
- Default model is "openai:gpt-4o" but can be overridden per operation
- Cache keys should be unique and descriptive for best performance
- Template variables are automatically escaped for security