"""
LLM AgenTool - Provides common natural language processing operations.

This toolkit provides a suite of LLM-powered operations including summarization,
classification, extraction, generation, translation, sentiment analysis, and more.
Each operation uses pydantic-ai's Agent with template-based prompts.

Features:
- Summary: Condense text to key points
- Markdownify: Convert text to well-formatted markdown
- Classification: Categorize text into predefined or dynamic classes
- Generation: Create new content based on prompts
- Extraction: Extract structured data from unstructured text
- Translation: Translate between languages
- Sentiment: Analyze emotional tone and sentiment
- Completion: Complete partial text or code

Example Usage:
    >>> from agentoolkit.llm import create_llm_agent
    >>> from agentool.core.injector import get_injector
    >>> 
    >>> # Create and register the agent
    >>> agent = create_llm_agent()
    >>> 
    >>> # Use through injector
    >>> injector = get_injector()
    >>> 
    >>> # Summarize text
    >>> result = await injector.run('llm', {
    ...     "operation": "summary",
    ...     "content": "Long article text here...",
    ...     "options": {"max_length": 100}
    ... })
    >>> 
    >>> # Extract structured data
    >>> result = await injector.run('llm', {
    ...     "operation": "extraction",
    ...     "content": "John Doe, CEO of Acme Corp, announced...",
    ...     "extraction_schema": {"name": "str", "title": "str", "company": "str"}
    ... })
"""

import json
from typing import Dict, Any, Optional, List, Literal, Union
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import RunContext, Agent

from agentool import create_agentool, BaseOperationInput
from agentool.core.registry import RoutingConfig
from agentool.core.injector import get_injector


class LLMInput(BaseOperationInput):
    """Input schema for LLM operations."""
    operation: Literal[
        'summary', 
        'markdownify', 
        'classification', 
        'generation', 
        'extraction', 
        'translation', 
        'sentiment', 
        'completion'
    ] = Field(description="The LLM operation to perform")
    
    content: str = Field(description="The input text to process")
    
    options: Optional[Dict[str, Any]] = Field(
        None, 
        description="Operation-specific options (e.g., max_length, temperature)"
    )
    
    extraction_schema: Optional[Dict[str, Any]] = Field(
        None,
        description="Schema for extraction operation (field_name -> type)"
    )
    
    classes: Optional[List[str]] = Field(
        None,
        description="List of classes for classification operation"
    )
    
    target_language: Optional[str] = Field(
        None,
        description="Target language for translation (e.g., 'es', 'fr', 'de')"
    )
    
    prompt: Optional[str] = Field(
        None,
        description="Additional prompt for generation or completion"
    )
    
    model: Optional[str] = Field(
        "openai:gpt-4o",
        description="LLM model to use (can override default)"
    )
    
    cache_key: Optional[str] = Field(
        None,
        description="Optional cache key for storing/retrieving results"
    )
    
    @field_validator('extraction_schema')
    def validate_extraction_schema(cls, v, info):
        """Validate schema is provided for extraction."""
        operation = info.data.get('operation')
        if operation == 'extraction' and not v:
            raise ValueError("extraction_schema is required for extraction operation")
        return v
    
    @field_validator('classes')
    def validate_classes(cls, v, info):
        """Validate classes are provided for classification."""
        operation = info.data.get('operation')
        if operation == 'classification' and not v:
            raise ValueError("classes are required for classification operation")
        return v
    
    @field_validator('target_language')
    def validate_target_language(cls, v, info):
        """Validate target language for translation."""
        operation = info.data.get('operation')
        if operation == 'translation' and not v:
            raise ValueError("target_language is required for translation operation")
        return v


class LLMOutput(BaseModel):
    """Structured output for LLM operations."""
    success: bool = Field(default=True, description="Whether the operation succeeded")
    operation: str = Field(description="The operation that was performed")
    message: str = Field(description="Human-readable result message")
    data: Dict[str, Any] = Field(description="Operation results")
    model_used: str = Field(description="The model that was used")
    tokens_used: Optional[Dict[str, int]] = Field(None, description="Token usage statistics")
    cached: bool = Field(default=False, description="Whether result was from cache")


# Dynamic output models for structured extraction
class ExtractionOutput(BaseModel):
    """Dynamic output for extraction based on provided schema."""
    extracted_data: Dict[str, Any]


class ClassificationOutput(BaseModel):
    """Output for classification operation."""
    selected_class: str
    confidence: Optional[float] = None
    reasoning: Optional[str] = None


class SentimentOutput(BaseModel):
    """Output for sentiment analysis."""
    sentiment: Literal['positive', 'negative', 'neutral', 'mixed']
    score: float = Field(ge=-1.0, le=1.0)
    emotions: Optional[Dict[str, float]] = None


async def _get_cached_result(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get cached result if available."""
    if not cache_key:
        return None
    
    injector = get_injector()
    try:
        result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': f'llm_cache:{cache_key}'
        })
        if result.success and result.data:
            return result.data.get('value')
    except:
        pass
    return None


async def _cache_result(cache_key: str, result: Dict[str, Any], ttl: int = 3600):
    """Cache result with TTL."""
    if not cache_key:
        return
    
    injector = get_injector()
    try:
        await injector.run('storage_kv', {
            'operation': 'set',
            'key': f'llm_cache:{cache_key}',
            'value': result,
            'ttl': ttl
        })
    except:
        pass


async def _render_prompts(
    system_template: str,
    user_template: str,
    variables: Dict[str, Any]
) -> tuple[str, str]:
    """Render system and user prompts from templates."""
    injector = get_injector()
    
    # Render system prompt
    system_result = await injector.run('templates', {
        'operation': 'render',
        'template_name': system_template,
        'variables': variables
    })
    system_prompt = system_result.data.get('rendered', '')
    
    # Render user prompt
    user_result = await injector.run('templates', {
        'operation': 'render',
        'template_name': user_template,
        'variables': variables
    })
    user_prompt = user_result.data.get('rendered', '')
    
    return system_prompt, user_prompt


async def llm_summary(
    ctx: RunContext[Any],
    content: str,
    options: Optional[Dict[str, Any]],
    model: str,
    cache_key: Optional[str]
) -> LLMOutput:
    """
    Generate a summary of the provided text.
    
    Args:
        ctx: Runtime context
        content: Text to summarize
        options: Options like max_length, style
        model: LLM model to use
        cache_key: Optional cache key
        
    Returns:
        Summary output with condensed text
    """
    # Check cache
    if cache_key:
        cached = await _get_cached_result(cache_key)
        if cached:
            return LLMOutput(
                success=True,
                operation='summary',
                message='Summary retrieved from cache',
                data=cached,
                model_used=model,
                cached=True
            )
    
    # Prepare template variables
    variables = {
        'max_length': options.get('max_length', 200) if options else 200,
        'style': options.get('style', 'concise') if options else 'concise'
    }
    
    # Render prompts
    system_prompt, user_prompt = await _render_prompts(
        'system/llm/summary',
        'prompts/llm/summary',
        variables
    )
    
    # Create agent and generate summary
    agent = Agent(
        model,
        system_prompt=system_prompt
    )
    
    result = await agent.run(content)
    summary_text = result.output
    
    # Prepare output data
    output_data = {
        'summary': summary_text,
        'original_length': len(content),
        'summary_length': len(summary_text),
        'compression_ratio': len(summary_text) / len(content) if content else 0
    }
    
    # Cache result
    if cache_key:
        await _cache_result(cache_key, output_data)
    
    # Get token usage
    usage = result.usage()
    tokens = {
        'request': usage.request_tokens,
        'response': usage.response_tokens,
        'total': usage.total_tokens
    }
    
    return LLMOutput(
        success=True,
        operation='summary',
        message='Text summarized successfully',
        data=output_data,
        model_used=model,
        tokens_used=tokens
    )


async def llm_markdownify(
    ctx: RunContext[Any],
    content: str,
    options: Optional[Dict[str, Any]],
    model: str,
    cache_key: Optional[str]
) -> LLMOutput:
    """
    Convert text to well-formatted markdown.
    
    Args:
        ctx: Runtime context
        content: Text to convert
        options: Formatting options
        model: LLM model to use
        cache_key: Optional cache key
        
    Returns:
        Markdown formatted text
    """
    # Check cache
    if cache_key:
        cached = await _get_cached_result(cache_key)
        if cached:
            return LLMOutput(
                success=True,
                operation='markdownify',
                message='Markdown retrieved from cache',
                data=cached,
                model_used=model,
                cached=True
            )
    
    # Prepare template variables
    variables = {
        'include_toc': options.get('include_toc', False) if options else False,
        'code_style': options.get('code_style', 'fenced') if options else 'fenced'
    }
    
    # Render prompts
    system_prompt, user_prompt = await _render_prompts(
        'system/llm/markdownify',
        'prompts/llm/markdownify',
        variables
    )
    
    # Create agent and convert to markdown
    agent = Agent(
        model,
        system_prompt=system_prompt
    )
    
    result = await agent.run(content)
    markdown_text = result.output
    
    # Prepare output data
    output_data = {
        'markdown': markdown_text,
        'has_headers': '#' in markdown_text,
        'has_code_blocks': '```' in markdown_text,
        'has_lists': any(marker in markdown_text for marker in ['- ', '* ', '1. '])
    }
    
    # Cache result
    if cache_key:
        await _cache_result(cache_key, output_data)
    
    # Get token usage
    usage = result.usage()
    tokens = {
        'request': usage.request_tokens,
        'response': usage.response_tokens,
        'total': usage.total_tokens
    }
    
    return LLMOutput(
        success=True,
        operation='markdownify',
        message='Text converted to markdown successfully',
        data=output_data,
        model_used=model,
        tokens_used=tokens
    )


async def llm_classification(
    ctx: RunContext[Any],
    content: str,
    classes: Optional[List[str]],
    options: Optional[Dict[str, Any]],
    model: str,
    cache_key: Optional[str]
) -> LLMOutput:
    """
    Classify text into predefined categories.
    
    Args:
        ctx: Runtime context
        content: Text to classify
        classes: List of possible classes
        options: Classification options
        model: LLM model to use
        cache_key: Optional cache key
        
    Returns:
        Classification result with selected class
    """
    # Validate classes is provided
    if not classes:
        raise ValueError("classes is required for classification operation")
    
    # Check cache
    if cache_key:
        cached = await _get_cached_result(cache_key)
        if cached:
            return LLMOutput(
                success=True,
                operation='classification',
                message='Classification retrieved from cache',
                data=cached,
                model_used=model,
                cached=True
            )
    
    # Prepare template variables
    variables = {
        'classes': json.dumps(classes),
        'include_reasoning': options.get('include_reasoning', False) if options else False
    }
    
    # Render prompts
    system_prompt, user_prompt = await _render_prompts(
        'system/llm/classification',
        'prompts/llm/classification',
        variables
    )
    
    # Create agent with structured output
    agent = Agent(
        model,
        system_prompt=system_prompt,
        output_type=ClassificationOutput
    )
    
    result = await agent.run(content)
    classification = result.output
    
    # Prepare output data
    output_data = {
        'selected_class': classification.selected_class,
        'confidence': classification.confidence,
        'reasoning': classification.reasoning,
        'all_classes': classes
    }
    
    # Cache result
    if cache_key:
        await _cache_result(cache_key, output_data)
    
    # Get token usage
    usage = result.usage()
    tokens = {
        'request': usage.request_tokens,
        'response': usage.response_tokens,
        'total': usage.total_tokens
    }
    
    return LLMOutput(
        success=True,
        operation='classification',
        message=f'Text classified as: {classification.selected_class}',
        data=output_data,
        model_used=model,
        tokens_used=tokens
    )


async def llm_generation(
    ctx: RunContext[Any],
    content: str,
    prompt: Optional[str],
    options: Optional[Dict[str, Any]],
    model: str,
    cache_key: Optional[str]
) -> LLMOutput:
    """
    Generate new content based on input and prompt.
    
    Args:
        ctx: Runtime context
        content: Context or seed text
        prompt: Generation instructions
        options: Generation options (temperature, max_tokens)
        model: LLM model to use
        cache_key: Optional cache key
        
    Returns:
        Generated content
    """
    # Check cache
    if cache_key:
        cached = await _get_cached_result(cache_key)
        if cached:
            return LLMOutput(
                success=True,
                operation='generation',
                message='Generated content retrieved from cache',
                data=cached,
                model_used=model,
                cached=True
            )
    
    # Prepare template variables
    variables = {
        'prompt': prompt or 'Generate creative content based on the input',
        'temperature': options.get('temperature', 0.7) if options else 0.7,
        'max_tokens': options.get('max_tokens', 500) if options else 500,
        'style': options.get('style', 'creative') if options else 'creative'
    }
    
    # Render prompts
    system_prompt, user_prompt = await _render_prompts(
        'system/llm/generation',
        'prompts/llm/generation',
        variables
    )
    
    # Create agent and generate content
    from pydantic_ai.settings import ModelSettings
    
    agent = Agent(
        model,
        system_prompt=system_prompt,
        model_settings=ModelSettings(
            temperature=variables['temperature'],
            max_tokens=variables['max_tokens']
        )
    )
    
    result = await agent.run(content)
    generated_text = result.output
    
    # Prepare output data
    output_data = {
        'generated': generated_text,
        'seed_length': len(content),
        'generated_length': len(generated_text),
        'parameters': {
            'temperature': variables['temperature'],
            'max_tokens': variables['max_tokens']
        }
    }
    
    # Cache result
    if cache_key:
        await _cache_result(cache_key, output_data)
    
    # Get token usage
    usage = result.usage()
    tokens = {
        'request': usage.request_tokens,
        'response': usage.response_tokens,
        'total': usage.total_tokens
    }
    
    return LLMOutput(
        success=True,
        operation='generation',
        message='Content generated successfully',
        data=output_data,
        model_used=model,
        tokens_used=tokens
    )


async def llm_extraction(
    ctx: RunContext[Any],
    content: str,
    extraction_schema: Optional[Dict[str, Any]],
    options: Optional[Dict[str, Any]],
    model: str,
    cache_key: Optional[str]
) -> LLMOutput:
    """
    Extract structured data from unstructured text.
    
    Args:
        ctx: Runtime context
        content: Text to extract from
        schema: Extraction schema (field -> type)
        options: Extraction options
        model: LLM model to use
        cache_key: Optional cache key
        
    Returns:
        Extracted structured data
    """
    # Validate extraction_schema is provided
    if not extraction_schema:
        raise ValueError("extraction_schema is required for extraction operation")
    
    # Check cache
    if cache_key:
        cached = await _get_cached_result(cache_key)
        if cached:
            return LLMOutput(
                success=True,
                operation='extraction',
                message='Extracted data retrieved from cache',
                data=cached,
                model_used=model,
                cached=True
            )
    
    # Prepare template variables
    variables = {
        'schema': json.dumps(extraction_schema, indent=2),
        'strict': options.get('strict', True) if options else True
    }
    
    # Render prompts
    system_prompt, user_prompt = await _render_prompts(
        'system/llm/extraction',
        'prompts/llm/extraction',
        variables
    )
    
    # Create dynamic Pydantic model from schema
    from typing import get_type_hints
    
    # Map string types to Python types
    type_mapping = {
        'str': str,
        'int': int,
        'float': float,
        'bool': bool,
        'list': list,
        'dict': dict
    }
    
    # Create fields for dynamic model with proper annotations
    fields = {}
    annotations = {}
    for field_name, field_type in extraction_schema.items():
        python_type = type_mapping.get(field_type, str)
        fields[field_name] = Field(description=f"Extracted {field_name}")
        annotations[field_name] = python_type
    
    # Create dynamic model class with annotations
    fields['__annotations__'] = annotations
    DynamicExtraction = type('DynamicExtraction', (BaseModel,), fields)
    
    # Create agent with dynamic output type
    agent = Agent(
        model,
        system_prompt=system_prompt,
        output_type=DynamicExtraction
    )
    
    result = await agent.run(content)
    extracted = result.output
    
    # Prepare output data
    output_data = {
        'extracted': extracted.model_dump(),
        'schema_used': extraction_schema,
        'fields_extracted': len(extracted.model_dump())
    }
    
    # Cache result
    if cache_key:
        await _cache_result(cache_key, output_data)
    
    # Get token usage
    usage = result.usage()
    tokens = {
        'request': usage.request_tokens,
        'response': usage.response_tokens,
        'total': usage.total_tokens
    }
    
    return LLMOutput(
        success=True,
        operation='extraction',
        message='Data extracted successfully',
        data=output_data,
        model_used=model,
        tokens_used=tokens
    )


async def llm_translation(
    ctx: RunContext[Any],
    content: str,
    target_language: Optional[str],
    options: Optional[Dict[str, Any]],
    model: str,
    cache_key: Optional[str]
) -> LLMOutput:
    """
    Translate text to target language.
    
    Args:
        ctx: Runtime context
        content: Text to translate
        target_language: Target language code
        options: Translation options
        model: LLM model to use
        cache_key: Optional cache key
        
    Returns:
        Translated text
    """
    # Validate target_language is provided
    if not target_language:
        raise ValueError("target_language is required for translation operation")
    
    # Check cache
    if cache_key:
        cached = await _get_cached_result(cache_key)
        if cached:
            return LLMOutput(
                success=True,
                operation='translation',
                message='Translation retrieved from cache',
                data=cached,
                model_used=model,
                cached=True
            )
    
    # Language code mapping
    language_names = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'ja': 'Japanese',
        'ko': 'Korean',
        'zh': 'Chinese',
        'ar': 'Arabic',
        'hi': 'Hindi'
    }
    
    # Prepare template variables
    variables = {
        'target_language': language_names.get(target_language, target_language),
        'preserve_formatting': options.get('preserve_formatting', True) if options else True,
        'formality': options.get('formality', 'neutral') if options else 'neutral'
    }
    
    # Render prompts
    system_prompt, user_prompt = await _render_prompts(
        'system/llm/translation',
        'prompts/llm/translation',
        variables
    )
    
    # Create agent and translate
    agent = Agent(
        model,
        system_prompt=system_prompt
    )
    
    result = await agent.run(content)
    translated_text = result.output
    
    # Prepare output data
    output_data = {
        'translated': translated_text,
        'source_length': len(content),
        'translated_length': len(translated_text),
        'target_language': target_language,
        'language_name': language_names.get(target_language, target_language)
    }
    
    # Cache result
    if cache_key:
        await _cache_result(cache_key, output_data)
    
    # Get token usage
    usage = result.usage()
    tokens = {
        'request': usage.request_tokens,
        'response': usage.response_tokens,
        'total': usage.total_tokens
    }
    
    return LLMOutput(
        success=True,
        operation='translation',
        message=f'Text translated to {language_names.get(target_language, target_language)}',
        data=output_data,
        model_used=model,
        tokens_used=tokens
    )


async def llm_sentiment(
    ctx: RunContext[Any],
    content: str,
    options: Optional[Dict[str, Any]],
    model: str,
    cache_key: Optional[str]
) -> LLMOutput:
    """
    Analyze sentiment and emotional tone of text.
    
    Args:
        ctx: Runtime context
        content: Text to analyze
        options: Analysis options
        model: LLM model to use
        cache_key: Optional cache key
        
    Returns:
        Sentiment analysis results
    """
    # Check cache
    if cache_key:
        cached = await _get_cached_result(cache_key)
        if cached:
            return LLMOutput(
                success=True,
                operation='sentiment',
                message='Sentiment analysis retrieved from cache',
                data=cached,
                model_used=model,
                cached=True
            )
    
    # Prepare template variables
    variables = {
        'include_emotions': options.get('include_emotions', True) if options else True,
        'granularity': options.get('granularity', 'detailed') if options else 'detailed'
    }
    
    # Render prompts
    system_prompt, user_prompt = await _render_prompts(
        'system/llm/sentiment',
        'prompts/llm/sentiment',
        variables
    )
    
    # Create agent with structured output
    agent = Agent(
        model,
        system_prompt=system_prompt,
        output_type=SentimentOutput
    )
    
    result = await agent.run(content)
    sentiment = result.output
    
    # Prepare output data
    output_data = {
        'sentiment': sentiment.sentiment,
        'score': sentiment.score,
        'emotions': sentiment.emotions or {},
        'confidence': abs(sentiment.score)
    }
    
    # Cache result
    if cache_key:
        await _cache_result(cache_key, output_data)
    
    # Get token usage
    usage = result.usage()
    tokens = {
        'request': usage.request_tokens,
        'response': usage.response_tokens,
        'total': usage.total_tokens
    }
    
    return LLMOutput(
        success=True,
        operation='sentiment',
        message=f'Sentiment: {sentiment.sentiment} (score: {sentiment.score:.2f})',
        data=output_data,
        model_used=model,
        tokens_used=tokens
    )


async def llm_completion(
    ctx: RunContext[Any],
    content: str,
    prompt: Optional[str],
    options: Optional[Dict[str, Any]],
    model: str,
    cache_key: Optional[str]
) -> LLMOutput:
    """
    Complete partial text or code.
    
    Args:
        ctx: Runtime context
        content: Partial text to complete
        prompt: Completion instructions
        options: Completion options
        model: LLM model to use
        cache_key: Optional cache key
        
    Returns:
        Completed text
    """
    # Check cache
    if cache_key:
        cached = await _get_cached_result(cache_key)
        if cached:
            return LLMOutput(
                success=True,
                operation='completion',
                message='Completion retrieved from cache',
                data=cached,
                model_used=model,
                cached=True
            )
    
    # Prepare template variables
    variables = {
        'prompt': prompt or 'Complete the following text naturally',
        'completion_type': options.get('type', 'text') if options else 'text',
        'max_tokens': options.get('max_tokens', 200) if options else 200
    }
    
    # Render prompts
    system_prompt, user_prompt = await _render_prompts(
        'system/llm/completion',
        'prompts/llm/completion',
        variables
    )
    
    # Create agent and complete text
    from pydantic_ai.settings import ModelSettings
    
    agent = Agent(
        model,
        system_prompt=system_prompt,
        model_settings=ModelSettings(
            max_tokens=variables['max_tokens']
        )
    )
    
    result = await agent.run(content)
    completed_text = result.output
    
    # Prepare output data
    output_data = {
        'completed': completed_text,
        'original': content,
        'completion_only': completed_text[len(content):] if completed_text.startswith(content) else completed_text,
        'total_length': len(completed_text)
    }
    
    # Cache result
    if cache_key:
        await _cache_result(cache_key, output_data)
    
    # Get token usage
    usage = result.usage()
    tokens = {
        'request': usage.request_tokens,
        'response': usage.response_tokens,
        'total': usage.total_tokens
    }
    
    return LLMOutput(
        success=True,
        operation='completion',
        message='Text completed successfully',
        data=output_data,
        model_used=model,
        tokens_used=tokens
    )


# Routing configuration
routing = RoutingConfig(
    operation_field='operation',
    operation_map={
        'summary': ('llm_summary', lambda x: {
            'content': x.content,
            'options': x.options,
            'model': x.model,
            'cache_key': x.cache_key
        }),
        'markdownify': ('llm_markdownify', lambda x: {
            'content': x.content,
            'options': x.options,
            'model': x.model,
            'cache_key': x.cache_key
        }),
        'classification': ('llm_classification', lambda x: {
            'content': x.content,
            'classes': x.classes,
            'options': x.options,
            'model': x.model,
            'cache_key': x.cache_key
        }),
        'generation': ('llm_generation', lambda x: {
            'content': x.content,
            'prompt': x.prompt,
            'options': x.options,
            'model': x.model,
            'cache_key': x.cache_key
        }),
        'extraction': ('llm_extraction', lambda x: {
            'content': x.content,
            'extraction_schema': x.extraction_schema,
            'options': x.options,
            'model': x.model,
            'cache_key': x.cache_key
        }),
        'translation': ('llm_translation', lambda x: {
            'content': x.content,
            'target_language': x.target_language,
            'options': x.options,
            'model': x.model,
            'cache_key': x.cache_key
        }),
        'sentiment': ('llm_sentiment', lambda x: {
            'content': x.content,
            'options': x.options,
            'model': x.model,
            'cache_key': x.cache_key
        }),
        'completion': ('llm_completion', lambda x: {
            'content': x.content,
            'prompt': x.prompt,
            'options': x.options,
            'model': x.model,
            'cache_key': x.cache_key
        })
    }
)


def create_llm_agent():
    """
    Create and return the LLM AgenTool.
    
    Returns:
        Configured LLM agent with all NLP operations
    """
    return create_agentool(
        name='llm',
        input_schema=LLMInput,
        routing_config=routing,
        tools=[
            llm_summary,
            llm_markdownify,
            llm_classification,
            llm_generation,
            llm_extraction,
            llm_translation,
            llm_sentiment,
            llm_completion
        ],
        output_type=LLMOutput,
        description="Natural language processing operations using LLMs",
        version="1.0.0",
        tags=["llm", "nlp", "ai", "text-processing"],
        examples=[
            {
                "input": {
                    "operation": "summary",
                    "content": "Long article text...",
                    "options": {"max_length": 100}
                },
                "output": {
                    "success": True,
                    "operation": "summary",
                    "message": "Text summarized successfully",
                    "data": {"summary": "Brief summary..."}
                }
            },
            {
                "input": {
                    "operation": "classification",
                    "content": "This product is amazing!",
                    "classes": ["positive", "negative", "neutral"]
                },
                "output": {
                    "success": True,
                    "operation": "classification",
                    "message": "Text classified as: positive",
                    "data": {"selected_class": "positive"}
                }
            }
        ]
    )


# Export the agent
agent = create_llm_agent()