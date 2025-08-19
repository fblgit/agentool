#!/usr/bin/env python3
"""
Real Research Demo - Actual Web Research with Research Orchestrator

This example performs REAL web research using browser automation and content extraction.
No mocks, no simulations - actual research on live websites.

Requirements:
- Playwright browser installed (run: playwright install chromium)
- Internet connection
- OpenAI API key configured

Example usage:
    python examples/real_research_demo.py
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry
from pydantic_ai import models

# Enable model requests for LLM operations
models.ALLOW_MODEL_REQUESTS = True


async def setup_agents():
    """Initialize all required agents for research orchestration."""
    print("🔧 Setting up research orchestrator and dependencies...")
    
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
    
    # Create the research orchestrator
    research_agent = create_research_orchestrator_agent()
    
    print("✅ All agents initialized successfully!\n")


async def perform_real_research():
    """
    Perform actual web research on a real topic.
    This will use browser automation to visit real websites and extract content.
    """
    injector = get_injector()
    
    # Real research topic - current and relevant
    topic = "OpenAI o1 model capabilities and usage"
    research_type = "documentation"
    session_id = f"real_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"🔍 Starting REAL research session: {session_id}")
    print(f"📚 Topic: {topic}")
    print(f"🎯 Research Type: {research_type}")
    print("-" * 60)
    
    try:
        # Phase 1: Plan Research Strategy
        print("\n📋 Phase 1: Planning Research Strategy...")
        plan_result = await injector.run('research_orchestrator', {
            "operation": "plan_research",
            "session_id": session_id,
            "query": topic,
            "research_type": research_type,
            "sources": [
                "https://openai.com/index/introducing-openai-o1-preview/",
                "https://platform.openai.com/docs",
                "https://community.openai.com"
            ],
            "search_terms": [
                "OpenAI o1 model",
                "o1-preview capabilities",
                "o1 reasoning model",
                "chain of thought reasoning"
            ],
            "max_sources": 5,
            "content_filter": {
                "relevance_threshold": 0.7,
                "quality_threshold": 0.6
            }
        })
        
        if plan_result.success:
            plan_data = plan_result.data
            print(f"✅ Research plan created!")
            print(f"   Strategy: {plan_data.get('strategy')}")
            print(f"   Search queries generated:")
            for i, query in enumerate(plan_data.get('search_queries', [])[:5], 1):
                print(f"     {i}. {query}")
            
            # Handle target_sources which might be dicts or strings
            target_sources = plan_data.get('target_sources', [])[:3]
            if target_sources and isinstance(target_sources[0], dict):
                # If dicts, extract type or description
                source_strs = [s.get('type', s.get('description', str(s))) for s in target_sources]
            else:
                # If already strings
                source_strs = target_sources
            print(f"   Target sources: {', '.join(source_strs)}")
            
            # Handle steps which should be strings
            steps = plan_data.get('steps', [])
            if steps:
                # Show only first 2 steps for brevity
                if len(steps) > 2:
                    print(f"   Research steps: {steps[0]}, {steps[1]}, ... ({len(steps)} total steps)")
                else:
                    print(f"   Research steps: {', '.join(steps)}")
        else:
            print(f"❌ Planning failed: {plan_result.message}")
            return
        
        # Phase 2: Execute Research with Real Browser Automation
        print("\n🌐 Phase 2: Executing REAL Web Research...")
        print("   Starting browser automation...")
        print("   This will visit actual websites and extract content...")
        
        exec_result = await injector.run('research_orchestrator', {
            "operation": "execute_research",
            "session_id": session_id
        })
        
        if exec_result.success:
            exec_data = exec_result.data
            print(f"✅ Real research executed!")
            print(f"   Sources processed: {exec_data.get('sources_processed')}")
            print(f"   Content extracted: {'Yes' if exec_data.get('content_extracted') else 'No'}")
            print(f"   Deduplication performed: {'Yes' if exec_data.get('deduplication_performed') else 'No'}")
            
            relevance_scores = exec_data.get('relevance_scores', [])
            if relevance_scores:
                print(f"   Content quality scores: {[f'{s:.2f}' for s in relevance_scores]}")
                print(f"   Average quality: {sum(relevance_scores)/len(relevance_scores):.2f}")
        else:
            print(f"⚠️  Research execution completed with issues: {exec_result.message}")
        
        # Phase 3: Check Progress
        print("\n📊 Phase 3: Checking Research Progress...")
        progress_result = await injector.run('research_orchestrator', {
            "operation": "track_progress",
            "session_id": session_id
        })
        
        if progress_result.success:
            progress_data = progress_result.data
            print(f"✅ Progress: {progress_data.get('progress_percentage')}%")
            print(f"   Current step: {progress_data.get('current_step')}")
            print(f"   Sources processed: {progress_data.get('sources_processed')}/{progress_data.get('total_sources')}")
            print(f"   Status: {progress_data.get('status')}")
        
        # Phase 4: Aggregate Collected Content
        print("\n📦 Phase 4: Aggregating and Analyzing Content...")
        aggregate_result = await injector.run('research_orchestrator', {
            "operation": "aggregate_content",
            "session_id": session_id
        })
        
        if aggregate_result.success:
            agg_data = aggregate_result.data
            print(f"✅ Content aggregated!")
            print(f"   Unique sources found: {agg_data.get('unique_sources')}")
            print(f"   Duplicates removed: {agg_data.get('duplicates_removed')}")
            
            summary = agg_data.get('summary', '')
            if summary:
                print(f"\n   📝 Content Summary:")
                print(f"   {summary[:500]}...")
        
        # Phase 5: Refine Research Based on Findings
        print("\n🔄 Phase 5: Refining Research Query...")
        refine_result = await injector.run('research_orchestrator', {
            "operation": "refine_query",
            "session_id": session_id,
            "follow_up_query": "What are the specific use cases and limitations of the o1 model compared to GPT-4?"
        })
        
        if refine_result.success:
            refine_data = refine_result.data
            print(f"✅ Query refined based on findings!")
            print(f"   Refined queries:")
            for i, query in enumerate(refine_data.get('refined_queries', [])[:3], 1):
                print(f"     {i}. {query}")
            
            gaps = refine_data.get('knowledge_gaps', [])
            if gaps:
                print(f"   Knowledge gaps identified:")
                for i, gap in enumerate(gaps[:3], 1):
                    print(f"     {i}. {gap}")
        
        # Phase 6: Generate Comprehensive Report
        print("\n📝 Phase 6: Generating Research Report...")
        
        # Generate Markdown report
        report_result = await injector.run('research_orchestrator', {
            "operation": "generate_report",
            "session_id": session_id,
            "report_format": "markdown"
        })
        
        if report_result.success:
            report_data = report_result.data
            print(f"✅ Research report generated!")
            
            # Display report details
            print(f"\n   📊 Report Statistics:")
            print(f"   - Source count: {report_data.get('source_count')}")
            print(f"   - Confidence score: {report_data.get('confidence_score', 0):.2%}")
            print(f"   - Report location: {report_data.get('report_path')}")
            
            print(f"\n   🔍 Key Findings:")
            for i, finding in enumerate(report_data.get('key_findings', []), 1):
                print(f"   {i}. {finding}")
            
            # Read and display part of the report
            report_path = Path(report_data.get('report_path', ''))
            if report_path.exists():
                with open(report_path, 'r') as f:
                    report_content = f.read()
                
                print(f"\n   📄 Report Preview (first 1000 chars):")
                print("   " + "-" * 50)
                preview = report_content[:1000].replace('\n', '\n   ')
                print(f"   {preview}...")
                print("   " + "-" * 50)
        
        # Generate structured data for programmatic access
        print("\n📊 Generating Structured Data...")
        struct_result = await injector.run('research_orchestrator', {
            "operation": "generate_report",
            "session_id": session_id,
            "report_format": "structured_data"
        })
        
        if struct_result.success:
            # Save structured data
            output_dir = Path("research_output")
            output_dir.mkdir(exist_ok=True)
            
            json_path = output_dir / f"{session_id}_data.json"
            with open(json_path, 'w') as f:
                json.dump(struct_result.data, f, indent=2, default=str)
            print(f"✅ Structured data saved to: {json_path}")
            
            # Display structure
            print(f"\n   Data structure includes:")
            for key in struct_result.data.keys():
                print(f"   - {key}")
        
        # Phase 7: Get Complete Session Data
        print("\n📊 Phase 7: Retrieving Complete Session Data...")
        session_result = await injector.run('research_orchestrator', {
            "operation": "get_session",
            "session_id": session_id
        })
        
        if session_result.success:
            session_data = session_result.data
            print(f"✅ Complete session data retrieved!")
            print(f"   Session ID: {session_data.get('session_id')}")
            print(f"   Final status: {session_data.get('status')}")
            print(f"   Query: {session_data.get('query')}")
            
            if 'created_at' in session_data and 'completed_at' in session_data:
                duration = session_data['completed_at'] - session_data['created_at']
                print(f"   Total duration: {duration:.1f} seconds")
        
        print("\n" + "=" * 60)
        print("🎉 REAL RESEARCH COMPLETED SUCCESSFULLY!")
        print(f"📁 Reports saved in 'research_output/' folder")
        print(f"📄 Markdown report: research/reports/{session_id}_report.md")
        print(f"📊 JSON data: research_output/{session_id}_data.json")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Research failed with error: {e}")
        import traceback
        traceback.print_exc()


async def research_custom_topic(topic: str, research_type: str = "documentation"):
    """
    Research a custom topic provided by the user.
    
    Args:
        topic: The topic to research
        research_type: Type of research (documentation, trend_analysis, comparative, fact_finding)
    """
    injector = get_injector()
    session_id = f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\n🔍 Custom Research: {topic}")
    print(f"🎯 Type: {research_type}")
    
    # Plan and execute research
    plan_result = await injector.run('research_orchestrator', {
        "operation": "plan_research",
        "session_id": session_id,
        "query": topic,
        "research_type": research_type,
        "max_sources": 3  # Limit for faster execution
    })
    
    if not plan_result.success:
        print(f"❌ Failed to plan research: {plan_result.message}")
        return
    
    # Execute the research
    exec_result = await injector.run('research_orchestrator', {
        "operation": "execute_research",
        "session_id": session_id
    })
    
    # Aggregate content
    await injector.run('research_orchestrator', {
        "operation": "aggregate_content",
        "session_id": session_id
    })
    
    # Generate report
    report_result = await injector.run('research_orchestrator', {
        "operation": "generate_report",
        "session_id": session_id,
        "report_format": "markdown"
    })
    
    if report_result.success:
        print(f"✅ Research complete! Report saved.")
        print(f"   Key findings: {report_result.data.get('key_findings', [])[:2]}")


async def main():
    """Main function to run real research examples."""
    print("\n" + "=" * 60)
    print("🚀 REAL Web Research Demo - No Mocks, No Simulations")
    print("=" * 60)
    print("\n⚠️  This demo will:")
    print("  • Start a real browser (Playwright)")
    print("  • Visit actual websites")
    print("  • Extract real content")
    print("  • Use OpenAI API for analysis")
    print("  • Generate real reports")
    
    # Setup agents
    await setup_agents()
    
    # Perform real research
    await perform_real_research()
    
    # Optional: Research another topic
    print("\n" + "=" * 60)
    print("💡 Want to research something else?")
    print("=" * 60)
    
    other_topics = [
        "Latest Python 3.13 features and improvements",
        "Comparison of Rust vs Go for system programming",
        "Best practices for Kubernetes deployment in 2024",
        "GraphQL vs REST API design patterns"
    ]
    
    print("\nOther topics you can research:")
    for i, topic in enumerate(other_topics, 1):
        print(f"  {i}. {topic}")
    
    # Research one more topic as demonstration
    print(f"\n🔬 Researching: {other_topics[0]}")
    #await research_custom_topic(other_topics[0], "documentation")
    
    print("\n✨ Demo complete! Modify the code to research your own topics.")


if __name__ == "__main__":
    # Ensure we have required directories
    Path("research_output").mkdir(exist_ok=True)
    Path("research/reports").mkdir(parents=True, exist_ok=True)
    Path("research/cache").mkdir(parents=True, exist_ok=True)
    
    # Run the real research demo
    asyncio.run(main())
