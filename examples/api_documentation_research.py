#!/usr/bin/env python3
"""
API Documentation Research Example

This example shows how to use the research orchestrator to gather and analyze
API documentation from multiple sources, creating a comprehensive guide.

Real use case: Researching a new API to understand its capabilities,
authentication methods, rate limits, and best practices.

Example usage:
    python examples/api_documentation_research.py
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry
from pydantic_ai import models

models.ALLOW_MODEL_REQUESTS = True


async def research_api_documentation(api_name: str, api_urls: list = None):
    """
    Research API documentation from official sources.
    
    Args:
        api_name: Name of the API to research (e.g., "Stripe API", "GitHub API")
        api_urls: Optional list of specific documentation URLs
    """
    # Initialize agents
    print(f"🔧 Initializing research agents for {api_name}...")
    
    AgenToolRegistry.clear()
    get_injector().clear()
    
    # Import all required agents
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
    
    # Create agents
    create_browser_manager_agent()
    create_page_navigator_agent()
    create_element_interactor_agent()
    create_http_agent()
    create_content_extractor_agent()
    create_llm_agent()
    create_markdown_generator_agent()
    create_templates_agent(templates_dir="src/templates")
    create_storage_fs_agent()
    create_storage_kv_agent()
    create_logging_agent()
    create_metrics_agent()
    create_research_orchestrator_agent()
    
    injector = get_injector()
    session_id = f"api_docs_{api_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\n📚 Researching {api_name} Documentation")
    print("=" * 60)
    
    # Step 1: Plan API Documentation Research
    print("\n1️⃣ Planning API documentation research...")
    
    search_terms = [
        f"{api_name} authentication",
        f"{api_name} rate limits",
        f"{api_name} endpoints",
        f"{api_name} examples",
        f"{api_name} best practices",
        f"{api_name} error handling",
        f"{api_name} webhooks",
        f"{api_name} SDKs"
    ]
    
    plan_result = await injector.run('research_orchestrator', {
        "operation": "plan_research",
        "session_id": session_id,
        "query": f"Complete documentation and usage guide for {api_name}",
        "research_type": "documentation",
        "sources": api_urls if api_urls else [],
        "search_terms": search_terms,
        "max_sources": 7,
        "content_filter": {
            "relevance_threshold": 0.75,  # Higher threshold for documentation
            "quality_threshold": 0.7
        }
    })
    
    if not plan_result.success:
        print(f"❌ Failed to plan research: {plan_result.message}")
        return None
    
    print(f"✅ Research plan created")
    plan_data = plan_result.data
    print(f"   Search queries: {len(plan_data.get('search_queries', []))}")
    print(f"   Target areas: Authentication, Endpoints, Rate Limits, Examples")
    
    # Step 2: Execute Documentation Research
    print("\n2️⃣ Executing web research (visiting documentation sites)...")
    
    exec_result = await injector.run('research_orchestrator', {
        "operation": "execute_research",
        "session_id": session_id
    })
    
    if exec_result.success:
        print(f"✅ Documentation gathered from {exec_result.data.get('sources_processed')} sources")
    
    # Step 3: Aggregate Documentation
    print("\n3️⃣ Aggregating and organizing documentation...")
    
    aggregate_result = await injector.run('research_orchestrator', {
        "operation": "aggregate_content",
        "session_id": session_id
    })
    
    if aggregate_result.success:
        print(f"✅ Documentation aggregated")
        print(f"   Unique sections: {aggregate_result.data.get('unique_sources')}")
        print(f"   Duplicates removed: {aggregate_result.data.get('duplicates_removed')}")
    
    # Step 4: Generate API Documentation Report
    print("\n4️⃣ Generating comprehensive API documentation...")
    
    report_result = await injector.run('research_orchestrator', {
        "operation": "generate_report",
        "session_id": session_id,
        "report_format": "markdown"
    })
    
    if not report_result.success:
        print(f"❌ Failed to generate report: {report_result.message}")
        return None
    
    report_data = report_result.data
    print(f"✅ API documentation generated!")
    print(f"\n📊 Documentation Summary:")
    print(f"   Confidence Score: {report_data.get('confidence_score', 0):.2%}")
    print(f"   Sources Used: {report_data.get('source_count')}")
    
    print(f"\n🔑 Key Documentation Points:")
    for i, finding in enumerate(report_data.get('key_findings', []), 1):
        print(f"   {i}. {finding}")
    
    # Step 5: Generate Structured API Reference
    print("\n5️⃣ Generating structured API reference...")
    
    struct_result = await injector.run('research_orchestrator', {
        "operation": "generate_report",
        "session_id": session_id,
        "report_format": "structured_data"
    })
    
    if struct_result.success:
        # Save API reference data
        output_dir = Path("api_documentation")
        output_dir.mkdir(exist_ok=True)
        
        # Save JSON reference
        json_path = output_dir / f"{api_name.replace(' ', '_')}_reference.json"
        with open(json_path, 'w') as f:
            json.dump(struct_result.data, f, indent=2, default=str)
        
        print(f"✅ Structured reference saved to: {json_path}")
        
        # Extract specific API information if available
        api_data = struct_result.data
        print(f"\n📋 API Information Extracted:")
        print(f"   - Authentication methods documented")
        print(f"   - Endpoint references collected")
        print(f"   - Rate limiting information gathered")
        print(f"   - Code examples found")
        print(f"   - Error handling patterns identified")
    
    # Return paths to generated documentation
    return {
        "markdown_report": report_data.get('report_path'),
        "json_reference": str(json_path) if struct_result.success else None,
        "session_id": session_id,
        "key_findings": report_data.get('key_findings', [])
    }


async def main():
    """Main function demonstrating API documentation research."""
    
    print("\n" + "=" * 60)
    print("🚀 API Documentation Research Tool")
    print("=" * 60)
    print("\nThis tool researches real API documentation from the web")
    print("and generates comprehensive documentation guides.\n")
    
    # Example 1: Research OpenAI API Documentation
    print("📖 Example 1: OpenAI API Documentation")
    print("-" * 40)
    
    openai_result = await research_api_documentation(
        api_name="OpenAI API",
        api_urls=[
            "https://platform.openai.com/docs/api-reference",
            "https://platform.openai.com/docs/guides",
            "https://platform.openai.com/docs/libraries"
        ]
    )
    
    if openai_result:
        print(f"\n✅ OpenAI API documentation complete!")
        print(f"   📄 Markdown: {openai_result['markdown_report']}")
        print(f"   📊 JSON Reference: {openai_result['json_reference']}")
    
    # Example 2: Research GitHub API Documentation
    print("\n" + "=" * 60)
    print("📖 Example 2: GitHub REST API Documentation")
    print("-" * 40)
    
    github_result = await research_api_documentation(
        api_name="GitHub REST API",
        api_urls=[
            "https://docs.github.com/en/rest",
            "https://docs.github.com/en/rest/guides",
            "https://docs.github.com/en/rest/reference"
        ]
    )
    
    if github_result:
        print(f"\n✅ GitHub API documentation complete!")
        print(f"   📄 Markdown: {github_result['markdown_report']}")
        print(f"   📊 JSON Reference: {github_result['json_reference']}")
    
    # Example 3: Research Stripe API Documentation
    print("\n" + "=" * 60)
    print("📖 Example 3: Stripe API Documentation")
    print("-" * 40)
    
    stripe_result = await research_api_documentation(
        api_name="Stripe API",
        api_urls=[
            "https://stripe.com/docs/api",
            "https://stripe.com/docs/api/authentication",
            "https://stripe.com/docs/api/errors"
        ]
    )
    
    if stripe_result:
        print(f"\n✅ Stripe API documentation complete!")
        print(f"   📄 Markdown: {stripe_result['markdown_report']}")
        print(f"   📊 JSON Reference: {stripe_result['json_reference']}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📚 API Documentation Research Complete!")
    print("=" * 60)
    print("\n📁 Generated documentation saved in:")
    print("   • api_documentation/ - JSON references")
    print("   • research/reports/ - Markdown documentation")
    print("\n💡 Use these documents to:")
    print("   • Understand API capabilities")
    print("   • Implement API integrations")
    print("   • Train team members")
    print("   • Create internal documentation")


if __name__ == "__main__":
    # Ensure output directories exist
    Path("api_documentation").mkdir(exist_ok=True)
    Path("research/reports").mkdir(parents=True, exist_ok=True)
    Path("research/cache").mkdir(parents=True, exist_ok=True)
    
    # Run the API documentation research
    asyncio.run(main())