#!/usr/bin/env python3
"""
Real Research Demo - Research Conductor for Dynamic Web Research

This example demonstrates the Research Conductor's ability to research
any topic dynamically using real web searches and content extraction.

Requirements:
- Playwright browser installed (run: playwright install chromium)
- Internet connection
- OpenAI API key configured

Example usage:
    python examples/real_research_demo.py
    python examples/real_research_demo.py "your custom topic here"
"""

import asyncio
import json
import sys
from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry
from pydantic_ai import models

# Enable model requests for LLM operations
models.ALLOW_MODEL_REQUESTS = True


async def setup_agents():
    """Initialize all required agents for research orchestration."""
    print("🔧 Setting up research conductor and dependencies...")
    
    # Clear registry to ensure fresh start
    AgenToolRegistry.clear()
    get_injector().clear()
    
    # Import and create all required agents
    from agentoolkit.playwright.browser_manager import create_browser_manager_agent
    from agentoolkit.playwright.page_navigator import create_page_navigator_agent
    from agentoolkit.playwright.element_interactor import create_element_interactor_agent
    from agentoolkit.network.http import create_http_agent
    from agentoolkit.llm.content_extractor import create_content_extractor_agent
    from agentoolkit.llm import create_llm_agent
    from agentoolkit.llm.markdown_generator import create_markdown_generator_agent
    from agentoolkit.system.templates import create_templates_agent
    from agentoolkit.storage.fs import create_storage_fs_agent
    from agentoolkit.storage.kv import create_storage_kv_agent
    from agentoolkit.system.logging import create_logging_agent
    from agentoolkit.observability.metrics import create_metrics_agent
    from agentoolkit.llm.research_orchestrator import create_research_orchestrator_agent
    from agentoolkit.llm.research_conductor import create_research_conductor_agent
    
    # Create all dependency agents
    browser_agent = create_browser_manager_agent()
    nav_agent = create_page_navigator_agent()
    elem_agent = create_element_interactor_agent()
    http_agent = create_http_agent()
    extractor_agent = create_content_extractor_agent()
    llm_agent = create_llm_agent()
    markdown_agent = create_markdown_generator_agent()
    templates_agent = create_templates_agent(templates_dir="src/templates")
    fs_agent = create_storage_fs_agent()
    kv_agent = create_storage_kv_agent()
    logging_agent = create_logging_agent()
    metrics_agent = create_metrics_agent()
    
    # Create the research orchestrator (used by conductor)
    research_agent = create_research_orchestrator_agent()
    
    # Create the research conductor (our main interface)
    conductor_agent = create_research_conductor_agent()
    
    print("✅ All agents initialized successfully!\n")


async def conduct_research(topic=None):
    """
    Conduct research on any topic dynamically.
    """
    print("\n" + "="*60)
    print("RESEARCH CONDUCTOR - Dynamic Research")
    print("="*60 + "\n")
    
    injector = get_injector()
    
    # Use provided topic
    if not topic:
        print("❌ No topic provided. Please provide a research topic.")
        return
    
    print(f"🔍 Researching: {topic}")
    print("📋 Research Type: documentation")
    print("🎯 Max Sources: 3")
    print("-" * 40)
    
    # Single call to conduct complete research
    try:
        result = await injector.run('research_conductor', {
            'operation': 'conduct_research',
            'topic': topic,
            'research_type': 'documentation',
            'max_sources': 3,
            'relevance_threshold': 0.3,
            'quality_threshold': 0.3
        })
        
        if hasattr(result, 'output'):
            data = json.loads(result.output)
        else:
            data = result
        
        if data.get('success'):
            print("\n✅ Research completed successfully!")
            print(f"📁 Session ID: {data.get('session_id')}")
            
            if data.get('conductor_output'):
                output = data['conductor_output']
                print(f"\n📊 Research Statistics:")
                print(f"  • Search terms generated: {output['search_terms_generated']}")
                print(f"  • Sources discovered: {output['sources_discovered']}")
                print(f"  • High-priority sources: {output['sources_prioritized']}")
                print(f"  • Research plan created: {'✓' if output['research_plan_created'] else '✗'}")
            
            if data.get('report'):
                print(f"\n📄 Research Report:")
                print("=" * 40)
                print(data['report'])
                print("=" * 40)
        else:
            print(f"\n❌ Research failed: {data.get('message')}")
            if data.get('error'):
                print(f"   Error details: {data['error']}")
                
    except Exception as e:
        print(f"\n❌ Research error: {str(e)}")
        import traceback
        traceback.print_exc()


async def main(topic=None):
    """Main demo function."""
    print("\n" + "="*80)
    print(" " * 20 + "RESEARCH CONDUCTOR")
    print(" " * 15 + "Real Web Research Without Mocks")
    print("="*80)
    
    # Setup all required agents
    await setup_agents()
    
    # Conduct research on the provided topic
    await conduct_research(topic)
    
    print("\n" + "="*80)
    print(" " * 25 + "Research Complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Get topic from command line if provided
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "latest gpt-5 openai models api names"
    asyncio.run(main(topic))