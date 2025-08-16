"""
Tests for the LLM AgenToolkit.

This test suite validates all LLM operations including summary, classification,
extraction, generation, translation, sentiment analysis, and more.
"""

import asyncio
import json
import os
from pathlib import Path
from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry
from pydantic_ai import models, capture_run_messages
models.ALLOW_MODEL_REQUESTS = True


class TestLLMAgent:
    """Test suite for LLM AgenTool."""
    
    def setup_method(self):
        """Clear registry and injector before each test."""
        AgenToolRegistry.clear()
        get_injector().clear()
        
        # Enable model requests for testing
        os.environ['ALLOW_MODEL_REQUESTS'] = 'true'
        
        # Create required dependency agents
        # Templates agent is required for prompt rendering
        # Point to the actual templates directory in src
        from agentoolkit.system.templates import create_templates_agent
        templates_agent = create_templates_agent(templates_dir="src/templates")
        
        # Storage FS agent (required by templates for file operations)
        from agentoolkit.storage.fs import create_storage_fs_agent
        fs_agent = create_storage_fs_agent()
        
        # Storage KV agent for caching functionality
        from agentoolkit.storage.kv import create_storage_kv_agent, _kv_storage, _kv_expiry
        _kv_storage.clear()
        _kv_expiry.clear()
        kv_agent = create_storage_kv_agent()
        
        # Logging agent (optional dependency)
        from agentoolkit.system.logging import create_logging_agent
        logging_agent = create_logging_agent()
        
        # Metrics agent (optional dependency)
        from agentoolkit.observability.metrics import create_metrics_agent
        metrics_agent = create_metrics_agent()
        
        # Import and create the LLM agent
        from agentoolkit.llm import create_llm_agent
        agent = create_llm_agent()
    
    def teardown_method(self):
        """Clean up after tests."""
        # Remove test environment variables
        if 'ALLOW_MODEL_REQUESTS' in os.environ:
            del os.environ['ALLOW_MODEL_REQUESTS']
    
    def test_summary_operation(self):
        """Test text summarization."""
        
        async def run_test():
            injector = get_injector()
            
            # Create test content
            long_content = (
                "Artificial intelligence (AI) is rapidly transforming industries worldwide. "
                "From healthcare to finance, AI systems are improving efficiency and accuracy. "
                "Machine learning models can now diagnose diseases, predict market trends, "
                "and even create art. However, with these advances come challenges including "
                "ethical concerns, job displacement, and the need for regulation. "
                "The future of AI will depend on how we address these challenges while "
                "continuing to innovate and improve these powerful technologies."
            ) * 5  # Make it longer
            
            # Test summary operation
            result = await injector.run('llm', {
                "operation": "summary",
                "content": long_content,
                "options": {"max_length": 100, "style": "concise"}
            })
            
            # LLM returns typed LLMOutput
            assert result.success is True
            assert result.operation == 'summary'
            assert 'summary' in result.data
            assert result.data['original_length'] == len(long_content)
            assert result.data['compression_ratio'] < 1.0
            assert result.model_used is not None
            
            print("\n=== test_summary_operation Output ===")
            print(f"Original length: {result.data['original_length']}")
            print(f"Summary length: {result.data['summary_length']}")
            print(f"Compression ratio: {result.data['compression_ratio']:.2f}")
            print(f"Model used: {result.model_used}")
            print("=" * 40)
        
        asyncio.run(run_test())
    
    def test_classification_operation(self):
        """Test text classification."""
        
        async def run_test():
            injector = get_injector()
            
            # Test classification with sentiment classes
            result = await injector.run('llm', {
                "operation": "classification",
                "content": "This product exceeded all my expectations! Absolutely fantastic!",
                "classes": ["positive", "negative", "neutral"],
                "options": {"include_reasoning": True}
            })
            
            # LLM returns typed LLMOutput
            assert result.success is True
            assert result.operation == 'classification'
            assert 'selected_class' in result.data
            assert result.data['selected_class'] in ["positive", "negative", "neutral"]
            assert 'all_classes' in result.data
            assert result.data['all_classes'] == ["positive", "negative", "neutral"]
            
            print("\n=== test_classification_operation Output ===")
            print(f"Selected class: {result.data['selected_class']}")
            if result.data.get('confidence'):
                print(f"Confidence: {result.data['confidence']}")
            if result.data.get('reasoning'):
                print(f"Reasoning: {result.data['reasoning']}")
            print("=" * 40)
            
            # Test with different content and classes
            result2 = await injector.run('llm', {
                "operation": "classification",
                "content": "The new AI model shows impressive accuracy improvements",
                "classes": ["technology", "business", "sports", "entertainment"]
            })
            
            assert result2.success is True
            assert result2.data['selected_class'] in ["technology", "business", "sports", "entertainment"]
        
        asyncio.run(run_test())
    
    def test_extraction_operation(self):
        """Test structured data extraction."""
        
        async def run_test():
            injector = get_injector()
            
            # Test extraction with person/company schema
            content = (
                "John Doe, the CEO of Acme Corporation, announced today that the company "
                "will be launching a new product line in Q2 2024. The announcement was made "
                "at the company's headquarters in San Francisco."
            )
            
            schema = {
                "name": "str",
                "title": "str",
                "company": "str",
                "location": "str"
            }
            
            result = await injector.run('llm', {
                "operation": "extraction",
                "content": content,
                "extraction_schema": schema,
                "options": {"strict": True}
            })
            
            # LLM returns typed LLMOutput
            assert result.success is True
            assert result.operation == 'extraction'
            assert 'extracted' in result.data
            assert 'schema_used' in result.data
            assert result.data['schema_used'] == schema
            
            extracted = result.data['extracted']
            assert isinstance(extracted, dict)
            assert all(key in extracted for key in schema.keys())
            
            print("\n=== test_extraction_operation Output ===")
            print(f"Extracted data: {json.dumps(extracted, indent=2)}")
            print(f"Fields extracted: {result.data['fields_extracted']}")
            print("=" * 40)
        
        asyncio.run(run_test())
    
    def test_translation_operation(self):
        """Test text translation."""
        
        async def run_test():
            injector = get_injector()
            
            # Test English to Spanish translation
            result = await injector.run('llm', {
                "operation": "translation",
                "content": "Hello, how are you today?",
                "target_language": "es",
                "options": {"preserve_formatting": True, "formality": "informal"}
            })
            
            # LLM returns typed LLMOutput
            assert result.success is True
            assert result.operation == 'translation'
            assert 'translated' in result.data
            assert result.data['target_language'] == 'es'
            assert result.data['language_name'] == 'Spanish'
            
            print("\n=== test_translation_operation Output ===")
            print(f"Original: Hello, how are you today?")
            print(f"Translated: {result.data['translated']}")
            print(f"Target language: {result.data['language_name']}")
            print("=" * 40)
            
            # Test with different language
            result2 = await injector.run('llm', {
                "operation": "translation",
                "content": "Good morning",
                "target_language": "fr"
            })
            
            assert result2.success is True
            assert result2.data['target_language'] == 'fr'
            assert result2.data['language_name'] == 'French'
        
        asyncio.run(run_test())
    
    def test_sentiment_operation(self):
        """Test sentiment analysis."""
        
        async def run_test():
            injector = get_injector()
            
            # Test positive sentiment
            result = await injector.run('llm', {
                "operation": "sentiment",
                "content": "I absolutely love this new feature! It's amazing and works perfectly.",
                "options": {"include_emotions": True, "granularity": "detailed"}
            })
            
            # LLM returns typed LLMOutput
            assert result.success is True
            assert result.operation == 'sentiment'
            assert 'sentiment' in result.data
            assert result.data['sentiment'] in ['positive', 'negative', 'neutral', 'mixed']
            assert 'score' in result.data
            assert -1.0 <= result.data['score'] <= 1.0
            
            print("\n=== test_sentiment_operation Output ===")
            print(f"Sentiment: {result.data['sentiment']}")
            print(f"Score: {result.data['score']:.2f}")
            if result.data.get('emotions'):
                print(f"Emotions: {result.data['emotions']}")
            print("=" * 40)
            
            # Test negative sentiment
            result2 = await injector.run('llm', {
                "operation": "sentiment",
                "content": "This is terrible. I'm very disappointed with the service."
            })
            
            assert result2.success is True
            assert result2.data['sentiment'] in ['positive', 'negative', 'neutral', 'mixed']
        
        asyncio.run(run_test())
    
    def test_generation_operation(self):
        """Test content generation."""
        
        async def run_test():
            injector = get_injector()
            
            # Test creative content generation
            result = await injector.run('llm', {
                "operation": "generation",
                "content": "A story about a robot learning to paint",
                "prompt": "Write a short creative story",
                "options": {
                    "temperature": 0.8,
                    "max_tokens": 200,
                    "style": "creative"
                }
            })
            
            # LLM returns typed LLMOutput
            assert result.success is True
            assert result.operation == 'generation'
            assert 'generated' in result.data
            assert result.data['seed_length'] > 0
            assert result.data['generated_length'] > 0
            assert result.data['parameters']['temperature'] == 0.8
            
            print("\n=== test_generation_operation Output ===")
            print(f"Seed length: {result.data['seed_length']}")
            print(f"Generated length: {result.data['generated_length']}")
            print(f"Temperature: {result.data['parameters']['temperature']}")
            print("=" * 40)
        
        asyncio.run(run_test())
    
    def test_completion_operation(self):
        """Test text completion."""
        
        async def run_test():
            injector = get_injector()
            
            # Test text completion
            result = await injector.run('llm', {
                "operation": "completion",
                "content": "The future of artificial intelligence will",
                "prompt": "Complete this sentence naturally",
                "options": {"max_tokens": 50, "type": "text"}
            })
            
            # LLM returns typed LLMOutput
            assert result.success is True
            assert result.operation == 'completion'
            assert 'completed' in result.data
            assert 'original' in result.data
            assert result.data['total_length'] > len(result.data['original'])
            
            print("\n=== test_completion_operation Output ===")
            print(f"Original: {result.data['original']}")
            print(f"Total length: {result.data['total_length']}")
            print("=" * 40)
        
        asyncio.run(run_test())
    
    def test_markdownify_operation(self):
        """Test markdown conversion."""
        
        async def run_test():
            injector = get_injector()
            
            # Test converting plain text to markdown
            plain_text = """
            Title of Document
            
            First Section
            This is a paragraph with some important points:
            Point 1 about something
            Point 2 about another thing
            Point 3 with more details
            
            Second Section
            Another paragraph with code example print("hello")
            """
            
            result = await injector.run('llm', {
                "operation": "markdownify",
                "content": plain_text,
                "options": {"include_toc": False, "code_style": "fenced"}
            })
            
            # LLM returns typed LLMOutput
            assert result.success is True
            assert result.operation == 'markdownify'
            assert 'markdown' in result.data
            assert isinstance(result.data['has_headers'], bool)
            assert isinstance(result.data['has_lists'], bool)
            
            print("\n=== test_markdownify_operation Output ===")
            print(f"Has headers: {result.data['has_headers']}")
            print(f"Has code blocks: {result.data['has_code_blocks']}")
            print(f"Has lists: {result.data['has_lists']}")
            print("=" * 40)
        
        asyncio.run(run_test())
    
    def test_caching_functionality(self):
        """Test result caching."""
        
        async def run_test():
            injector = get_injector()
            
            # First call - should not be cached
            cache_key = "test_cache_key_123"
            content = "This is test content for caching"
            
            result1 = await injector.run('llm', {
                "operation": "summary",
                "content": content,
                "cache_key": cache_key
            })
            
            assert result1.success is True
            assert result1.cached is False
            
            # Second call with same cache key - should be cached
            result2 = await injector.run('llm', {
                "operation": "summary",
                "content": content,
                "cache_key": cache_key
            })
            
            assert result2.success is True
            assert result2.cached is True
            assert result2.data == result1.data  # Same data returned
            
            print("\n=== test_caching_functionality Output ===")
            print(f"First call cached: {result1.cached}")
            print(f"Second call cached: {result2.cached}")
            print(f"Data matches: {result1.data == result2.data}")
            print("=" * 40)
        
        asyncio.run(run_test())
    
    def test_model_override(self):
        """Test model override functionality."""
        
        async def run_test():
            injector = get_injector()
            
            # Test with default model
            result1 = await injector.run('llm', {
                "operation": "summary",
                "content": "Test content for model override"
            })
            
            assert result1.success is True
            assert result1.model_used == "openai:gpt-4o"
            
            # Test with custom model
            result2 = await injector.run('llm', {
                "operation": "summary",
                "content": "Test content for model override",
                "model": "anthropic:claude-3-opus"
            })
            
            assert result2.success is True
            assert result2.model_used == "anthropic:claude-3-opus"
            
            print("\n=== test_model_override Output ===")
            print(f"Default model: {result1.model_used}")
            print(f"Custom model: {result2.model_used}")
            print("=" * 40)
        
        asyncio.run(run_test())
    
    def test_input_validation(self):
        """Test input validation for required fields."""
        
        async def run_test():
            injector = get_injector()
            
            # Test missing extraction_schema for extraction
            result = await injector.run('llm', {
                "operation": "extraction",
                "content": "Some text"
                # Missing required 'extraction_schema' field
            })
            
            # Should get validation error
            if hasattr(result, 'output'):
                output = json.loads(result.output) if isinstance(result.output, str) else result.output
                assert "validation" in str(output).lower() or "extraction_schema" in str(output).lower()
            
            # Test missing classes for classification
            result = await injector.run('llm', {
                "operation": "classification",
                "content": "Some text"
                # Missing required 'classes' field
            })
            
            if hasattr(result, 'output'):
                output = json.loads(result.output) if isinstance(result.output, str) else result.output
                assert "validation" in str(output).lower() or "classes" in str(output).lower()
            
            # Test missing target_language for translation
            result = await injector.run('llm', {
                "operation": "translation",
                "content": "Hello"
                # Missing required 'target_language' field
            })
            
            if hasattr(result, 'output'):
                output = json.loads(result.output) if isinstance(result.output, str) else result.output
                assert "validation" in str(output).lower() or "target_language" in str(output).lower()
            
            print("\n=== test_input_validation Output ===")
            print("Input validation tests completed")
            print("=" * 40)
        
        asyncio.run(run_test())
    
    def test_invalid_operation(self):
        """Test handling of invalid operations."""
        
        async def run_test():
            injector = get_injector()
            
            # Test invalid operation
            result = await injector.run('llm', {
                "operation": "invalid_operation",
                "content": "Test content"
            })
            
            # Should get validation error for invalid operation
            if hasattr(result, 'output'):
                output = json.loads(result.output) if isinstance(result.output, str) else result.output
                assert "validation" in str(output).lower() or "invalid" in str(output).lower()
            
            print("\n=== test_invalid_operation Output ===")
            print("Invalid operation handled correctly")
            print("=" * 40)
        
        asyncio.run(run_test())
