# GraphToolkit Tests

This directory contains tests for the GraphToolkit meta-framework, including the comprehensive agentoolkit domain workflow tests.

## LLM Response Caching

The GraphToolkit tests support LLM response caching to make tests run faster and more deterministically. This is implemented directly in the `LLMCallNode` and works transparently without requiring any test modifications.

### How It Works

1. **Cache Key Generation**: Each LLM call generates a unique SHA-256 hash based on:
   - Model name (e.g., `openai:gpt-4o`)
   - System prompt
   - User prompt
   - Model parameters (temperature, max_tokens, etc.)
   - Output schema (if specified)

2. **Cache Storage**: Responses are stored as JSON files in `/tmp/llm_cache/{hash}.json`

3. **Cache Lookup**: When caching is enabled:
   - First checks if a cached response exists for the computed hash
   - If found (cache hit), returns the cached response immediately
   - If not found (cache miss), makes a real LLM call and caches the result

### Usage

#### Enable Caching

```bash
# Enable LLM response caching
export LLM_REPLAY=1

# Run tests with caching
pytest tests/graphtoolkit/test_agentoolkit_domain.py
```

#### Disable Caching

```bash
# Disable caching (use real LLM every time)
unset LLM_REPLAY

# Or explicitly set to 0
export LLM_REPLAY=0

# Run tests without caching
pytest tests/graphtoolkit/test_agentoolkit_domain.py
```

#### Clear Cache

```bash
# Remove all cached responses
rm -rf /tmp/llm_cache/

# Remove specific cached responses (if you know the hash)
rm /tmp/llm_cache/{hash}.json
```

### Cache File Format

Each cache file is a JSON document with the following structure:

```json
{
  "cache_version": "1.0",
  "created_at": "2025-01-15T10:30:00Z",
  "model": "openai:gpt-4o",
  "prompts": {
    "system": "You are an expert Python developer...",
    "user": "Create a storage tool that..."
  },
  "params": {
    "temperature": 0.7,
    "max_tokens": 2000,
    "top_p": 1.0,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "stop_sequences": null
  },
  "response": {
    "type": "CodeOutput",
    "data": {
      "code": "def storage_read(key: str)...",
      "file_path": "storage_tool.py"
    }
  },
  "metadata": {
    "phase": "crafter",
    "workflow_id": "test_123"
  }
}
```

### Benefits

1. **Speed**: Cached responses are served instantly (milliseconds vs seconds/minutes)
2. **Cost**: No API calls for cached responses means zero cost for repeated test runs
3. **Deterministic**: Same inputs always produce same outputs
4. **Debugging**: Cache files are human-readable JSON for easy inspection
5. **Incremental**: New prompts are automatically cached on first run
6. **Transparent**: No changes needed to test code or workflow definitions

### Performance Comparison

| Mode | First Run | Subsequent Runs |
|------|-----------|-----------------|
| Without Cache | 30-60 seconds | 30-60 seconds |
| With Cache | 30-60 seconds | 2-5 seconds |

### Advanced Usage

#### Inspect Cache Contents

```bash
# List all cached responses
ls -la /tmp/llm_cache/

# View a specific cache file
cat /tmp/llm_cache/{hash}.json | jq .

# Search cache files by phase
grep -l '"phase": "analyzer"' /tmp/llm_cache/*.json

# Count cache files by model
grep '"model":' /tmp/llm_cache/*.json | cut -d: -f2 | sort | uniq -c
```

#### Selective Cache Clearing

```bash
# Remove cache for specific phase
for f in /tmp/llm_cache/*.json; do
  if grep -q '"phase": "analyzer"' "$f"; then
    rm "$f"
  fi
done

# Remove cache older than 7 days
find /tmp/llm_cache -name "*.json" -mtime +7 -delete
```

#### Cache Statistics

```bash
# Total cache size
du -sh /tmp/llm_cache/

# Number of cached responses
ls /tmp/llm_cache/*.json 2>/dev/null | wc -l

# Cache hit rate (requires test output with logging)
pytest tests/graphtoolkit/test_agentoolkit_domain.py -s 2>&1 | \
  grep "LLMCallNode" | grep -c "Cache hit"
```

### Implementation Details

The caching is implemented in `/Users/mrv/agentools/src/graphtoolkit/nodes/atomic/llm.py`:

- `_should_use_cache()`: Checks if `LLM_REPLAY` environment variable is set
- `_compute_cache_key()`: Generates deterministic hash from LLM call parameters
- `_get_cache_path()`: Returns path to cache file for given key
- `_load_cached_response()`: Loads and deserializes cached response
- `_save_cached_response()`: Serializes and saves response to cache
- `_call_llm()`: Main method that orchestrates caching logic

The caching system handles various response types:
- Pydantic models (via `model_dump()`)
- `CodeOutput` objects
- Plain dictionaries
- String responses
- Custom objects with `__dict__`

### Troubleshooting

#### Cache Not Working

1. Check environment variable:
   ```bash
   echo $LLM_REPLAY  # Should output "1" if enabled
   ```

2. Check cache directory permissions:
   ```bash
   ls -ld /tmp/llm_cache/
   ```

3. Enable debug logging to see cache operations:
   ```bash
   export PYTHONPATH=/Users/mrv/agentools:$PYTHONPATH
   python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
   pytest tests/graphtoolkit/test_agentoolkit_domain.py -s
   ```

#### Cache Corruption

If you encounter errors loading cached responses:

```bash
# Clear all cache and start fresh
rm -rf /tmp/llm_cache/
```

#### Different Results with Cache

If cached responses produce different results than fresh LLM calls:

1. Clear the cache to force fresh calls
2. Compare the prompts being generated (they should be deterministic)
3. Check if model parameters have changed

### Best Practices

1. **CI/CD**: Enable caching in CI to speed up test runs
2. **Development**: Use caching during development for faster iteration
3. **Production Tests**: Disable caching for final validation before releases
4. **Cache Warmup**: Run tests once with caching enabled to populate cache
5. **Version Control**: Don't commit cache files (they're in `/tmp` by design)