# AgenTool Workflow UI

A modern Streamlit interface for the AgenTool workflow generator, providing real-time visualization and control over the AI-powered code generation process.

## Features

- **Real-time Progress Tracking**: Monitor each phase of the workflow with visual progress indicators
- **Streaming Updates**: See AI responses as they're generated with pydantic-ai streaming integration
- **Artifact Viewer**: Browse and search through all generated artifacts in an organized tree view
- **Code Editor**: View generated code with syntax highlighting and export capabilities
- **Metrics Dashboard**: Comprehensive analytics including performance, quality, token usage, and errors
- **Export Functionality**: Download results as JSON packages for further use

## Installation

1. Ensure you have Python 3.9+ installed
2. Install required dependencies:
```bash
pip install streamlit pandas
```

3. Set up environment variables for your AI provider:
```bash
export OPENAI_API_KEY="your-api-key"
# OR
export ANTHROPIC_API_KEY="your-api-key"
```

## Running the UI

From the project root directory:

```bash
# Set environment variables for your AI provider
export OPENAI_API_KEY="your-api-key"

# Run the UI with correct Python path
PYTHONPATH=src streamlit run src/ui/workflow_ui.py
```

The UI will open in your default browser at `http://localhost:8501`

### Alternative: Run with specific Python
```bash
PYTHONPATH=src python -m streamlit run src/ui/workflow_ui.py
```

## Usage

1. **Configure Settings**:
   - Select your AI model provider (OpenAI, Anthropic, Google, Groq)
   - Choose the specific model
   - Enable debug mode for detailed logging
   - Toggle streaming for real-time updates

2. **Describe Your Task**:
   - Enter a detailed description of the AgenTool you want to create
   - Example: "Create a session management AgenTool that handles user sessions with TTL support and Redis integration"

3. **Start Workflow**:
   - Click "▶️ Start Workflow" to begin the generation process
   - Monitor progress through the four phases:
     - **Analyzer**: Understands requirements and identifies missing tools
     - **Specifier**: Creates detailed specifications for each tool
     - **Crafter**: Generates the actual implementation code
     - **Evaluator**: Reviews and validates the generated code

4. **View Results**:
   - **Progress Tab**: Real-time status of each phase
   - **Artifacts Tab**: Browse all generated data organized by type
   - **Generated Code Tab**: View and copy the final implementation
   - **Metrics Tab**: Analyze performance and quality metrics

5. **Export Results**:
   - Click "💾 Export Results" to download a complete JSON package
   - Includes all artifacts, metrics, and generated code

## Architecture

### Components

- **workflow_ui.py**: Main entry point and UI layout
- **workflow_runner.py**: Async workflow execution with event callbacks
- **stream_handlers.py**: pydantic-ai stream processing
- **components/**:
  - `progress_tracker.py`: Visual progress tracking
  - `artifact_viewer.py`: Artifact browsing interface
  - `code_editor.py`: Code display with syntax highlighting
  - `metrics_dashboard.py`: Analytics and metrics visualization
- **utils/formatting.py**: Helper functions for data formatting

### State Management

The UI uses Streamlit's session state to maintain:
- Workflow execution state
- Generated artifacts
- Performance metrics
- Error tracking

### Event System

The workflow runner provides callbacks for:
- Phase start/complete/error events
- Artifact creation
- Streaming updates
- Metrics collection

## Customization

### Adding New Metrics

Edit `metrics_dashboard.py` to add custom metrics:

```python
def _render_custom_metrics(self, metrics: Dict[str, Any]):
    """Render custom metrics."""
    # Add your custom metric visualization
```

### Extending Artifact Types

Edit `artifact_viewer.py` to support new artifact types:

```python
self.artifact_types['custom'] = {'icon': '🎯', 'name': 'Custom Type'}
```

### Custom Themes

Modify the CSS in `workflow_ui.py`:

```python
st.markdown("""
<style>
    /* Add your custom styles */
</style>
""", unsafe_allow_html=True)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PYTHONPATH includes the src directory
2. **API Key Errors**: Check that your AI provider API keys are set
3. **Async Errors**: The UI requires Python 3.9+ for proper async support
4. **Memory Issues**: Large workflows may require increased resources

### Debug Mode

Enable debug mode in Advanced Settings to see:
- Detailed logging output
- Raw API responses
- Internal state changes

## Future Enhancements

- [ ] Real-time collaboration features
- [ ] Version control integration
- [ ] Custom prompt templates
- [ ] Batch workflow processing
- [ ] Advanced code analysis tools
- [ ] Integration with CI/CD pipelines

## Contributing

To contribute to the UI development:

1. Follow the existing component structure
2. Add proper type hints and docstrings
3. Include error handling for edge cases
4. Test with various workflow scenarios

## License

Part of the AgenTool project - see main project license.