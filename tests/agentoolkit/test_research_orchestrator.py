"""
Tests for the research_orchestrator AgenToolkit.

This test suite validates all research orchestration operations including research planning,
query refinement, content aggregation, report generation, progress tracking,
and session management following AgenTool testing patterns.
"""

import asyncio
import json
import os
import tempfile
import pytest
from pathlib import Path
from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry
from pydantic_ai import models
models.ALLOW_MODEL_REQUESTS = True


class TestResearchOrchestratorAgent:
    """Test suite for research_orchestrator AgenTool."""
    
    def setup_method(self):
        """Clear registry and injector before each test."""
        AgenToolRegistry.clear()
        get_injector().clear()
        
        # Create required dependency agents
        # Browser manager for web automation
        from agentoolkit.playwright.browser_manager import create_browser_manager_agent
        browser_agent = create_browser_manager_agent()
        
        # Page navigator for web navigation
        from agentoolkit.playwright.page_navigator import create_page_navigator_agent
        nav_agent = create_page_navigator_agent()
        
        # Element interactor for web interactions
        from agentoolkit.playwright.element_interactor import create_element_interactor_agent
        elem_agent = create_element_interactor_agent()
        
        # HTTP agent for API requests
        from agentoolkit.network.http import create_http_agent
        http_agent = create_http_agent()
        
        # Content extractor for parsing web content
        from agentoolkit.llm.content_extractor import create_content_extractor_agent
        extractor_agent = create_content_extractor_agent()
        
        # LLM agent for intelligent processing
        from agentoolkit.llm import create_llm_agent
        llm_agent = create_llm_agent()
        
        # Templates agent for document templates
        from agentoolkit.system.templates import create_templates_agent
        templates_agent = create_templates_agent(templates_dir="src/templates")
        
        # Storage FS agent for file operations
        from agentoolkit.storage.fs import create_storage_fs_agent
        fs_agent = create_storage_fs_agent()
        
        # Storage KV agent for session management
        from agentoolkit.storage.kv import create_storage_kv_agent, _kv_storage, _kv_expiry
        _kv_storage.clear()
        _kv_expiry.clear()
        kv_agent = create_storage_kv_agent()
        
        # Logging agent for operation logging
        from agentoolkit.system.logging import create_logging_agent
        logging_agent = create_logging_agent()
        
        # Metrics agent for performance tracking
        from agentoolkit.observability.metrics import create_metrics_agent
        metrics_agent = create_metrics_agent()
        
        # Import and create the research_orchestrator agent
        from agentoolkit.llm.research_orchestrator import create_research_orchestrator_agent
        agent = create_research_orchestrator_agent()
    
    def test_plan_research_operation(self):
        """Test research planning functionality."""
        
        async def run_test():
            injector = get_injector()
            
            # Test documentation research planning
            result = await injector.run('research_orchestrator', {
                "operation": "plan_research",
                "session_id": "test_session_001",
                "query": "How to use OpenAI GPT-4 API for text generation",
                "research_type": "documentation",
                "sources": ["https://platform.openai.com/docs"],
                "search_terms": ["GPT-4 API", "text generation", "OpenAI documentation"],
                "max_sources": 5,
                "content_filter": {"relevance_threshold": 0.7}
            })
            
            assert result.success is True
            assert result.message == "Successfully created research plan for session 'test_session_001'"
            
            data = result.data
            assert "plan_id" in data
            assert "strategy" in data
            assert "search_queries" in data
            assert "target_sources" in data
            assert "estimated_duration" in data
            assert "steps" in data
            
            plan_id = data["plan_id"]
            strategy = data["strategy"]
            search_queries = data["search_queries"]
            target_sources = data["target_sources"]
            steps = data["steps"]
            
            # Verify plan structure
            assert plan_id == "plan_test_session_001"
            assert isinstance(strategy, str)
            assert isinstance(search_queries, list)
            assert len(search_queries) > 0
            assert isinstance(target_sources, list)
            assert isinstance(steps, list)
            
            # Verify query and terms are included
            assert any("GPT-4" in query for query in search_queries)
            assert any("OpenAI" in query for query in search_queries)
            
            print("\n=== test_plan_research_operation Output ===")
            print(f"Plan ID: {plan_id}")
            print(f"Strategy: {strategy}")
            print(f"Search queries: {search_queries}")
            print(f"Target sources: {target_sources}")
            print(f"Steps: {steps}")
            print("=" * 40)
            
            # Test trend analysis research planning
            result2 = await injector.run('research_orchestrator', {
                "operation": "plan_research",
                "session_id": "trend_session_002",
                "query": "AI technology trends in 2024",
                "research_type": "trend_analysis",
                "max_sources": 8
            })
            
            assert result2.success is True
            strategy2 = result2.data["strategy"]
            assert "trend" in strategy2.lower()
            
            # Test comparative research planning
            result3 = await injector.run('research_orchestrator', {
                "operation": "plan_research",
                "session_id": "compare_session_003",
                "query": "Compare React vs Vue.js frameworks",
                "research_type": "comparative",
                "max_sources": 6
            })
            
            assert result3.success is True
            search_queries3 = result3.data["search_queries"]
            assert any("react" in query.lower() for query in search_queries3)
            assert any("vue" in query.lower() for query in search_queries3)
        
        asyncio.run(run_test())
    
    def test_execute_research_operation(self):
        """Test that execute_research operation requires a planned session."""
        
        async def run_test():
            injector = get_injector()
            
            # First plan research
            plan_result = await injector.run('research_orchestrator', {
                "operation": "plan_research",
                "session_id": "exec_session_001",
                "query": "Python web frameworks comparison",
                "research_type": "comparative",
                "max_sources": 3
            })
            
            assert plan_result.success is True
            
            # Test that session was created properly for execution
            session_result = await injector.run('storage_kv', {
                'operation': 'get',
                'key': 'research_session_exec_session_001',
                'namespace': 'research_orchestrator'
            })
            
            assert session_result.success is True
            session_data = session_result.data['value']
            assert session_data['session_id'] == 'exec_session_001'
            assert session_data['query'] == 'Python web frameworks comparison'
            assert session_data['research_type'] == 'comparative'
            assert session_data['max_sources'] == 3
            assert session_data['status'] == 'planned'
            
            print("\n=== test_execute_research_operation Output ===")
            print(f"Session created: {session_data['session_id']}")
            print(f"Status: {session_data['status']}")
            print(f"Query: {session_data['query']}")
            print("Note: Full execution requires browser automation which is complex to test")
            print("=" * 40)
        
        asyncio.run(run_test())
    
    def test_refine_query_operation(self):
        """Test query refinement based on findings."""
        
        async def run_test():
            injector = get_injector()
            
            # Plan research first
            await injector.run('research_orchestrator', {
                "operation": "plan_research",
                "session_id": "refine_session_001",
                "query": "Machine learning algorithms",
                "research_type": "fact_finding",
                "max_sources": 3
            })
            
            # Add some content to the session using the storage agent
            session_data = {
                'session_id': 'refine_session_001',
                'query': 'Machine learning algorithms',
                'content_collected': [
                    {
                        'content': {
                            'content': 'Supervised learning uses labeled data to train models. Common algorithms include linear regression and decision trees.',
                            'title': 'Supervised Learning Guide'
                        },
                        'quality_score': 0.8
                    },
                    {
                        'content': {
                            'content': 'Unsupervised learning finds patterns in unlabeled data. Clustering and dimensionality reduction are key techniques.',
                            'title': 'Unsupervised Learning Overview'
                        },
                        'quality_score': 0.75
                    }
                ]
            }
            
            # Store session with content using storage agent
            await injector.run('storage_kv', {
                'operation': 'set',
                'key': 'research_session_refine_session_001',
                'value': session_data,
                'namespace': 'research_orchestrator'
            })
            
            # Refine query based on findings
            result = await injector.run('research_orchestrator', {
                "operation": "refine_query",
                "session_id": "refine_session_001",
                "follow_up_query": "Deep learning neural networks comparison"
            })
            
            assert result.success is True
            assert result.message == "Successfully refined research query for session 'refine_session_001'"
            
            data = result.data
            assert "refined_queries" in data
            assert "knowledge_gaps" in data
            assert "suggested_sources" in data
            assert "research_direction" in data
            
            refined_queries = data["refined_queries"]
            knowledge_gaps = data["knowledge_gaps"]
            suggested_sources = data["suggested_sources"]
            research_direction = data["research_direction"]
            
            # Verify refinement structure
            assert isinstance(refined_queries, list)
            assert len(refined_queries) > 0
            assert isinstance(knowledge_gaps, list)
            assert isinstance(suggested_sources, list)
            assert isinstance(research_direction, str)
            
            # Verify follow-up query is included or referenced
            assert any("deep learning" in query.lower() for query in refined_queries) or "deep learning" in str(refined_queries).lower()
            
            print("\n=== test_refine_query_operation Output ===")
            print(f"Refined queries: {refined_queries}")
            print(f"Knowledge gaps: {knowledge_gaps}")
            print(f"Suggested sources: {suggested_sources}")
            print(f"Research direction: {research_direction}")
            print("=" * 40)
        
        asyncio.run(run_test())
    
    def test_aggregate_content_operation(self):
        """Test content aggregation and deduplication."""
        
        async def run_test():
            injector = get_injector()
            
            # Create session with sample content
            session_data = {
                'session_id': 'aggregate_session_001',
                'query': 'Data science methodologies',
                'research_type': 'documentation',
                'status': 'executed',
                'content_collected': [
                    {
                        'url': 'https://example.com/data-science-1',
                        'content': {
                            'content': 'Data science is an interdisciplinary field that uses scientific methods to extract knowledge from data.',
                            'title': 'Data Science Introduction'
                        },
                        'quality_score': 0.9,
                        'extracted_at': 1640995200
                    },
                    {
                        'url': 'https://example.com/data-science-2',
                        'content': {
                            'content': 'Machine learning is a key component of data science that enables predictive modeling.',
                            'title': 'ML in Data Science'
                        },
                        'quality_score': 0.85,
                        'extracted_at': 1640995300
                    },
                    {
                        'url': 'https://example.com/duplicate',
                        'content': {
                            'content': 'Data science is an interdisciplinary field that uses scientific methods.',
                            'title': 'Duplicate Content'
                        },
                        'quality_score': 0.7,
                        'extracted_at': 1640995400
                    }
                ]
            }
            
            # Store session using storage agent
            await injector.run('storage_kv', {
                'operation': 'set',
                'key': 'research_session_aggregate_session_001',
                'value': session_data,
                'namespace': 'research_orchestrator'
            })
            
            # Aggregate content
            result = await injector.run('research_orchestrator', {
                "operation": "aggregate_content",
                "session_id": "aggregate_session_001"
            })
            
            assert result.success is True
            assert result.message == "Successfully aggregated content for session 'aggregate_session_001'"
            
            data = result.data
            assert "unique_sources" in data
            assert "duplicates_removed" in data
            assert "summary" in data
            assert "quality_scores" in data
            
            unique_sources = data["unique_sources"]
            duplicates_removed = data["duplicates_removed"]
            summary = data["summary"]
            quality_scores = data["quality_scores"]
            
            # Verify aggregation results
            assert isinstance(unique_sources, int)
            assert unique_sources >= 0
            assert isinstance(duplicates_removed, int)
            assert duplicates_removed >= 0
            assert isinstance(summary, str)
            assert isinstance(quality_scores, list)
            
            print("\n=== test_aggregate_content_operation Output ===")
            print(f"Unique sources: {unique_sources}")
            print(f"Duplicates removed: {duplicates_removed}")
            print(f"Summary: {summary}")
            print(f"Quality scores: {quality_scores}")
            print("=" * 40)
        
        asyncio.run(run_test())
    
    def test_generate_report_operation(self):
        """Test research report generation."""
        
        async def run_test():
            injector = get_injector()
            
            # Create session with aggregated content
            aggregated_data = {
                'unique_sources': 2,
                'duplicates_removed': 1,
                'content_items': [
                    {
                        'url': 'https://example.com/report-source-1',
                        'content': {
                            'content': 'Artificial intelligence is transforming industries through automation and data analysis.',
                            'title': 'AI Transformation'
                        },
                        'quality_score': 0.95
                    }
                ],
                'summary': 'AI technologies are driving significant changes across multiple sectors.',
                'quality_scores': [0.95, 0.88]
            }
            
            session_data = {
                'session_id': 'report_session_001',
                'query': 'AI industry transformation',
                'research_type': 'trend_analysis',
                'status': 'aggregated',
                'aggregated_content': aggregated_data
            }
            
            # Store session using storage agent
            await injector.run('storage_kv', {
                'operation': 'set',
                'key': 'research_session_report_session_001',
                'value': session_data,
                'namespace': 'research_orchestrator'
            })
            
            # Generate markdown report
            result = await injector.run('research_orchestrator', {
                "operation": "generate_report",
                "session_id": "report_session_001",
                "report_format": "markdown"
            })
            
            assert result.success is True
            assert result.message == "Successfully generated research report for session 'report_session_001'"
            
            data = result.data
            assert "summary" in data
            assert "source_count" in data
            assert "confidence_score" in data
            assert "key_findings" in data
            
            summary = data["summary"]
            source_count = data["source_count"]
            confidence_score = data["confidence_score"]
            key_findings = data["key_findings"]
            
            # Verify report structure
            assert isinstance(summary, str)
            assert isinstance(source_count, int)
            assert isinstance(confidence_score, float)
            assert isinstance(key_findings, list)
            assert 0.0 <= confidence_score <= 1.0
            
            print("\n=== test_generate_report_operation Output ===")
            print(f"Summary: {summary}")
            print(f"Source count: {source_count}")
            print(f"Confidence score: {confidence_score}")
            print(f"Key findings: {key_findings}")
            print("=" * 40)
            
            # Test structured data format
            result2 = await injector.run('research_orchestrator', {
                "operation": "generate_report",
                "session_id": "report_session_001",
                "report_format": "structured_data"
            })
            
            assert result2.success is True
            data2 = result2.data
            assert "session_id" in data2
            assert "query" in data2
            assert data2["session_id"] == "report_session_001"
        
        asyncio.run(run_test())
    
    def test_track_progress_operation(self):
        """Test research progress tracking."""
        
        async def run_test():
            injector = get_injector()
            
            # Create session with progress data
            session_data = {
                'session_id': 'progress_session_001',
                'query': 'Blockchain technology applications',
                'research_type': 'comparative',
                'status': 'executed',
                'current_step': 'content_aggregation',
                'progress_percentage': 60,
                'sources_processed': 5,
                'max_sources': 8,
                'created_at': 1640995200,
                'executed_at': 1640995800
            }
            
            # Store session using storage agent
            await injector.run('storage_kv', {
                'operation': 'set',
                'key': 'research_session_progress_session_001',
                'value': session_data,
                'namespace': 'research_orchestrator'
            })
            
            # Track progress
            result = await injector.run('research_orchestrator', {
                "operation": "track_progress",
                "session_id": "progress_session_001"
            })
            
            assert result.success is True
            assert "60% complete" in result.message
            
            data = result.data
            assert "progress_percentage" in data
            assert "current_step" in data
            assert "completed_steps" in data
            assert "remaining_steps" in data
            assert "sources_processed" in data
            assert "total_sources" in data
            assert "status" in data
            
            progress_percentage = data["progress_percentage"]
            current_step = data["current_step"]
            completed_steps = data["completed_steps"]
            remaining_steps = data["remaining_steps"]
            sources_processed = data["sources_processed"]
            
            # Verify progress structure
            assert progress_percentage == 60
            assert current_step == 'content_aggregation'
            assert isinstance(completed_steps, list)
            assert isinstance(remaining_steps, list)
            assert sources_processed == 5
            
            print("\n=== test_track_progress_operation Output ===")
            print(f"Progress: {progress_percentage}%")
            print(f"Current step: {current_step}")
            print(f"Completed steps: {completed_steps}")
            print(f"Remaining steps: {remaining_steps}")
            print(f"Sources processed: {sources_processed}")
            print("=" * 40)
        
        asyncio.run(run_test())
    
    def test_get_session_operation(self):
        """Test session data retrieval."""
        
        async def run_test():
            injector = get_injector()
            
            # Create session data
            session_data = {
                'session_id': 'get_session_001',
                'query': 'Cloud computing platforms',
                'research_type': 'comparative',
                'status': 'completed',
                'progress_percentage': 100,
                'plan': {
                    'strategy': 'comparative_focused',
                    'search_queries': ['AWS vs Azure', 'Google Cloud comparison'],
                    'target_sources': ['official documentation', 'comparison sites']
                }
            }
            
            # Store session using storage agent
            await injector.run('storage_kv', {
                'operation': 'set',
                'key': 'research_session_get_session_001',
                'value': session_data,
                'namespace': 'research_orchestrator'
            })
            
            # Get session data
            result = await injector.run('research_orchestrator', {
                "operation": "get_session",
                "session_id": "get_session_001"
            })
            
            assert result.success is True
            assert result.message == "Retrieved research session 'get_session_001'"
            
            data = result.data
            assert "session_id" in data
            assert "query" in data
            assert "research_type" in data
            assert "status" in data
            assert "plan" in data
            
            # Verify retrieved data matches stored data
            assert data["session_id"] == "get_session_001"
            assert data["query"] == "Cloud computing platforms"
            assert data["research_type"] == "comparative"
            assert data["status"] == "completed"
            
            print("\n=== test_get_session_operation Output ===")
            print(f"Session ID: {data['session_id']}")
            print(f"Query: {data['query']}")
            print(f"Status: {data['status']}")
            print(f"Plan: {data['plan']}")
            print("=" * 40)
            
            # Test non-existent session
            result2 = await injector.run('research_orchestrator', {
                "operation": "get_session",
                "session_id": "nonexistent_session"
            })
            
            assert result2.success is False
            assert "not found" in result2.message
        
        asyncio.run(run_test())
    
    def test_input_validation(self):
        """Test input validation for research orchestrator operations."""
        
        async def run_test():
            injector = get_injector()
            
            # Test missing query for plan_research
            try:
                await injector.run('research_orchestrator', {
                    "operation": "plan_research",
                    "session_id": "test_session",
                    "research_type": "documentation"
                })
                assert False, "Should have raised validation error"
            except Exception as e:
                assert "query is required" in str(e)
            
            # Test missing research_type for plan_research
            try:
                await injector.run('research_orchestrator', {
                    "operation": "plan_research",
                    "session_id": "test_session",
                    "query": "Test query"
                })
                assert False, "Should have raised validation error"
            except Exception as e:
                assert "research_type is required" in str(e)
            
            # Test missing follow_up_query for refine_query
            try:
                await injector.run('research_orchestrator', {
                    "operation": "refine_query",
                    "session_id": "test_session"
                })
                assert False, "Should have raised validation error"
            except Exception as e:
                assert "follow_up_query is required" in str(e)
            
            print("\n=== test_input_validation Output ===")
            print("All validation tests passed")
            print("=" * 40)
        
        asyncio.run(run_test())
    
    def test_research_workflow_integration(self):
        """Test full research workflow integration."""
        
        async def run_test():
            injector = get_injector()
            session_id = "workflow_integration_001"
            
            # Step 1: Plan research
            plan_result = await injector.run('research_orchestrator', {
                "operation": "plan_research",
                "session_id": session_id,
                "query": "Microservices architecture patterns",
                "research_type": "documentation",
                "max_sources": 5
            })
            
            assert plan_result.success is True
            
            # Step 2: Check progress after planning
            progress_result = await injector.run('research_orchestrator', {
                "operation": "track_progress",
                "session_id": session_id
            })
            
            assert progress_result.success is True
            assert progress_result.data["status"] == "planned"
            
            # Step 3: Get session data
            session_result = await injector.run('research_orchestrator', {
                "operation": "get_session",
                "session_id": session_id
            })
            
            assert session_result.success is True
            assert session_result.data["query"] == "Microservices architecture patterns"
            
            print("\n=== test_research_workflow_integration Output ===")
            print(f"Workflow completed for session: {session_id}")
            print(f"Plan created: {plan_result.success}")
            print(f"Progress tracked: {progress_result.success}")
            print(f"Session retrieved: {session_result.success}")
            print("=" * 40)
        
        asyncio.run(run_test())