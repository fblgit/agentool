"""
Tests for TestSuite domain workflow.

This module tests the complete TestSuite generation workflow including:
- Test Analyzer phase (code analysis for test requirements)
- Test Designer phase (test architecture and strategy)  
- Test Generator phase (actual test code generation)
- Test Executor phase (running tests and coverage analysis)
"""

import asyncio
import tempfile
import pytest
import ast
import json
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from datetime import datetime

# GraphToolkit imports
from graphtoolkit import execute_testsuite_workflow, create_testsuite_workflow
from graphtoolkit.core.executor import WorkflowExecutor, WorkflowResult
from graphtoolkit.core.types import (
    WorkflowState,
    WorkflowDefinition, 
    StorageRef,
    StorageType,
    ValidationResult
)

# Test some actual Python code to generate tests for
SAMPLE_CODE_TO_TEST = '''
"""
Sample code for testing the TestSuite workflow.
"""

import math
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta


class Calculator:
    """A simple calculator with various mathematical operations."""
    
    def __init__(self, precision: int = 2):
        self.precision = precision
        self.history: List[str] = []
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        result = round(a + b, self.precision)
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a."""
        result = round(a - b, self.precision)
        self.history.append(f"{a} - {b} = {result}")
        return result
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        result = round(a * b, self.precision)
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def divide(self, a: float, b: float) -> float:
        """Divide a by b."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = round(a / b, self.precision)
        self.history.append(f"{a} / {b} = {result}")
        return result
    
    def power(self, base: float, exponent: float) -> float:
        """Raise base to the power of exponent."""
        result = round(base ** exponent, self.precision)
        self.history.append(f"{base} ^ {exponent} = {result}")
        return result
    
    def sqrt(self, x: float) -> float:
        """Calculate square root."""
        if x < 0:
            raise ValueError("Cannot calculate square root of negative number")
        result = round(math.sqrt(x), self.precision)
        self.history.append(f"√{x} = {result}")
        return result
    
    def clear_history(self) -> None:
        """Clear calculation history."""
        self.history.clear()
    
    def get_history(self) -> List[str]:
        """Get calculation history."""
        return self.history.copy()


def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def validate_email(email: str) -> bool:
    """Simple email validation."""
    if not email or "@" not in email:
        return False
    parts = email.split("@")
    if len(parts) != 2:
        return False
    username, domain = parts
    if not username or not domain:
        return False
    if "." not in domain:
        return False
    return True


class DataProcessor:
    """Process various data formats."""
    
    @staticmethod
    def filter_positive_numbers(numbers: List[float]) -> List[float]:
        """Filter out non-positive numbers."""
        return [n for n in numbers if n > 0]
    
    @staticmethod 
    def calculate_statistics(numbers: List[float]) -> Dict[str, float]:
        """Calculate basic statistics for a list of numbers."""
        if not numbers:
            return {"count": 0, "mean": 0, "min": 0, "max": 0}
        
        count = len(numbers)
        mean = sum(numbers) / count
        min_val = min(numbers)
        max_val = max(numbers)
        
        return {
            "count": count,
            "mean": round(mean, 2),
            "min": min_val,
            "max": max_val
        }
    
    @staticmethod
    def merge_dictionaries(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two dictionaries, with dict2 values taking precedence."""
        result = dict1.copy()
        result.update(dict2)
        return result
'''


class TestTestSuiteWorkflowComplete:
    """Test complete TestSuite workflow execution."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        # Mock LLM responses for TestSuite phases
        self.mock_llm_responses = {
            'test_analyzer': {
                'code_analysis': {
                    'classes': ['Calculator', 'DataProcessor'],
                    'functions': ['fibonacci', 'validate_email'],
                    'complexity_score': 6.5,
                    'test_coverage_requirements': {
                        'Calculator': ['__init__', 'add', 'subtract', 'multiply', 'divide', 'power', 'sqrt', 'clear_history', 'get_history'],
                        'fibonacci': ['edge_cases', 'recursive_behavior'],
                        'validate_email': ['valid_emails', 'invalid_emails'],
                        'DataProcessor': ['filter_positive_numbers', 'calculate_statistics', 'merge_dictionaries']
                    },
                    'edge_cases': [
                        'Division by zero',
                        'Negative square root',
                        'Negative fibonacci input',
                        'Empty lists',
                        'Invalid email formats'
                    ]
                },
                'test_strategy': {
                    'framework': 'pytest',
                    'estimated_test_count': 35,
                    'coverage_target': 0.90,
                    'test_types': ['unit', 'edge_case', 'integration']
                },
                'quality_score': 0.88
            },
            'test_designer': {
                'test_architecture': {
                    'test_files': [
                        'test_calculator.py',
                        'test_fibonacci.py', 
                        'test_email_validation.py',
                        'test_data_processor.py'
                    ],
                    'fixtures': ['sample_numbers', 'email_test_cases'],
                    'test_organization': {
                        'test_calculator.py': {
                            'TestCalculatorBasic': ['test_add', 'test_subtract', 'test_multiply'],
                            'TestCalculatorAdvanced': ['test_divide', 'test_power', 'test_sqrt'],
                            'TestCalculatorEdgeCases': ['test_divide_by_zero', 'test_negative_sqrt'],
                            'TestCalculatorHistory': ['test_history_tracking', 'test_clear_history']
                        }
                    }
                },
                'test_plan': {
                    'phases': ['unit_tests', 'integration_tests', 'edge_case_tests'],
                    'dependencies': ['pytest', 'pytest-cov'],
                    'mock_requirements': []
                },
                'quality_score': 0.92
            },
            'test_generator': {
                'generated_tests': {
                    'test_calculator.py': '''
import pytest
import math
from calculator import Calculator


class TestCalculatorBasic:
    def test_add(self):
        calc = Calculator()
        result = calc.add(2.5, 3.7)
        assert result == 6.2
        assert "2.5 + 3.7 = 6.2" in calc.get_history()
    
    def test_subtract(self):
        calc = Calculator()
        result = calc.subtract(10.0, 3.5)
        assert result == 6.5
    
    def test_multiply(self):
        calc = Calculator()
        result = calc.multiply(4.0, 2.5)
        assert result == 10.0


class TestCalculatorAdvanced:
    def test_divide(self):
        calc = Calculator()
        result = calc.divide(15.0, 3.0)
        assert result == 5.0
    
    def test_divide_by_zero(self):
        calc = Calculator()
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            calc.divide(10.0, 0.0)
    
    def test_sqrt(self):
        calc = Calculator()
        result = calc.sqrt(16.0)
        assert result == 4.0
    
    def test_sqrt_negative(self):
        calc = Calculator()
        with pytest.raises(ValueError, match="Cannot calculate square root of negative number"):
            calc.sqrt(-4.0)


class TestCalculatorHistory:
    def test_history_tracking(self):
        calc = Calculator()
        calc.add(1, 2)
        calc.multiply(3, 4)
        history = calc.get_history()
        assert len(history) == 2
        assert "1 + 2 = 3" in history
        assert "3 * 4 = 12" in history
    
    def test_clear_history(self):
        calc = Calculator()
        calc.add(1, 2)
        calc.clear_history()
        assert len(calc.get_history()) == 0
''',
                    'test_fibonacci.py': '''
import pytest
from calculator import fibonacci


def test_fibonacci_base_cases():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1


def test_fibonacci_sequence():
    assert fibonacci(2) == 1
    assert fibonacci(3) == 2
    assert fibonacci(4) == 3
    assert fibonacci(5) == 5
    assert fibonacci(6) == 8


def test_fibonacci_negative():
    with pytest.raises(ValueError, match="n must be non-negative"):
        fibonacci(-1)
'''
                },
                'test_metrics': {
                    'total_test_functions': 12,
                    'total_test_classes': 3,
                    'estimated_coverage': 0.87,
                    'complexity_coverage': 0.93
                },
                'quality_score': 0.89
            },
            'test_executor': {
                'test_execution_results': {
                    'total_tests': 12,
                    'passed': 11,
                    'failed': 1,
                    'skipped': 0,
                    'error_details': ['test_fibonacci.py::test_fibonacci_sequence FAILED - AssertionError']
                },
                'coverage_report': {
                    'total_coverage': 0.89,
                    'line_coverage': 0.91,
                    'branch_coverage': 0.87,
                    'file_coverage': {
                        'Calculator': 0.95,
                        'fibonacci': 0.80,
                        'validate_email': 0.75,
                        'DataProcessor': 0.92
                    }
                },
                'performance_metrics': {
                    'total_execution_time': 0.425,
                    'average_test_time': 0.035,
                    'slowest_tests': [
                        ('test_calculator.py::TestCalculatorAdvanced::test_sqrt', 0.08),
                        ('test_fibonacci.py::test_fibonacci_sequence', 0.12)
                    ]
                },
                'recommendations': [
                    'Add more test cases for validate_email function',
                    'Improve branch coverage in fibonacci function',
                    'Consider parameterized tests for Calculator operations'
                ],
                'quality_score': 0.86
            }
        }
    
    @pytest.mark.asyncio
    async def test_complete_testsuite_workflow_execution(self):
        """Test end-to-end TestSuite workflow with all phases."""
        with patch('graphtoolkit.nodes.atomic.llm.LLMCallNode') as mock_llm_node:
            # Configure mock LLM responses for each phase
            async def mock_llm_run(ctx):
                current_phase = ctx.state.current_phase
                response_data = self.mock_llm_responses.get(current_phase, {})
                
                # Update state with mock response
                new_domain_data = {**ctx.state.domain_data}
                new_domain_data[f'{current_phase}_output'] = response_data
                
                from dataclasses import replace
                new_state = replace(
                    ctx.state,
                    domain_data=new_domain_data,
                    quality_scores={
                        **ctx.state.quality_scores,
                        current_phase: response_data.get('quality_score', 0.8)
                    }
                )
                
                return new_state
            
            mock_llm_instance = AsyncMock()
            mock_llm_instance.run = AsyncMock(side_effect=mock_llm_run)
            mock_llm_node.return_value = mock_llm_instance
            
            # Execute TestSuite workflow
            result = await execute_testsuite_workflow(
                code_to_test=SAMPLE_CODE_TO_TEST,
                framework="pytest",
                coverage_target=0.85,
                workflow_id="test-suite-complete",
                enable_persistence=False
            )
            
            # Verify workflow completion
            assert result['success'] == True
            assert result['workflow_id'] == "test-suite-complete"
            assert len(result['completed_phases']) == 4
            assert set(result['completed_phases']) == {'test_analyzer', 'test_designer', 'test_generator', 'test_executor'}
            
            # Verify TestSuite-specific outputs
            assert 'test_analyzer_output' in result['domain_data']
            assert 'test_designer_output' in result['domain_data']
            assert 'test_generator_output' in result['domain_data']
            assert 'test_executor_output' in result['domain_data']
            
            # Verify TestSuite-specific result fields
            assert 'test_files' in result
            assert 'coverage_report' in result
            assert 'test_results' in result
            
            # Verify quality scores for all phases
            assert len(result['quality_scores']) == 4
            for phase in ['test_analyzer', 'test_designer', 'test_generator', 'test_executor']:
                assert phase in result['quality_scores']
                assert 0.0 <= result['quality_scores'][phase] <= 1.0
    
    @pytest.mark.asyncio
    async def test_testsuite_analyzer_phase(self):
        """Test test analyzer phase specifically."""
        with patch('graphtoolkit.core.executor.WorkflowExecutor') as mock_executor_class:
            mock_executor = AsyncMock()
            mock_executor_class.return_value = mock_executor
            
            # Create mock state after analyzer phase
            final_state = MagicMock(spec=WorkflowState)
            final_state.completed_phases = {'test_analyzer'}
            final_state.quality_scores = {'test_analyzer': 0.88}
            final_state.domain_data = {
                'code_to_test': SAMPLE_CODE_TO_TEST,
                'framework': 'pytest',
                'coverage_target': 0.85,
                'test_analyzer_output': self.mock_llm_responses['test_analyzer']
            }
            
            mock_executor.run.return_value = WorkflowResult(
                state=final_state,
                outputs={'test_analyzer': {'storage_ref': 'kv://workflow/test-id/analyzer'}},
                success=True
            )
            
            # Execute workflow
            result = await execute_testsuite_workflow(
                code_to_test=SAMPLE_CODE_TO_TEST,
                framework="pytest",
                coverage_target=0.85,
                enable_persistence=False
            )
            
            # Verify analyzer results
            analyzer_output = result['domain_data']['test_analyzer_output']
            code_analysis = analyzer_output['code_analysis']
            
            # Verify code structure analysis
            assert 'classes' in code_analysis
            assert 'functions' in code_analysis
            assert 'Calculator' in code_analysis['classes']
            assert 'DataProcessor' in code_analysis['classes']
            assert 'fibonacci' in code_analysis['functions']
            assert 'validate_email' in code_analysis['functions']
            
            # Verify test coverage requirements
            assert 'test_coverage_requirements' in code_analysis
            coverage_reqs = code_analysis['test_coverage_requirements']
            assert 'Calculator' in coverage_reqs
            assert len(coverage_reqs['Calculator']) > 0
            
            # Verify edge case identification
            assert 'edge_cases' in code_analysis
            edge_cases = code_analysis['edge_cases']
            assert 'Division by zero' in edge_cases
            assert 'Negative square root' in edge_cases
            
            # Verify test strategy
            test_strategy = analyzer_output['test_strategy']
            assert test_strategy['framework'] == 'pytest'
            assert test_strategy['coverage_target'] == 0.90
            assert test_strategy['estimated_test_count'] > 0
    
    @pytest.mark.asyncio
    async def test_testsuite_generator_phase(self):
        """Test test generator phase for code quality."""
        with patch('graphtoolkit.core.executor.WorkflowExecutor') as mock_executor_class:
            mock_executor = AsyncMock()
            mock_executor_class.return_value = mock_executor
            
            # Mock state after generator phase
            final_state = MagicMock(spec=WorkflowState)
            final_state.completed_phases = {'test_analyzer', 'test_designer', 'test_generator'}
            final_state.quality_scores = {
                'test_analyzer': 0.88,
                'test_designer': 0.92,
                'test_generator': 0.89
            }
            final_state.domain_data = {
                'test_generator_output': self.mock_llm_responses['test_generator']
            }
            
            mock_executor.run.return_value = WorkflowResult(
                state=final_state,
                outputs={},
                success=True
            )
            
            result = await execute_testsuite_workflow(
                code_to_test=SAMPLE_CODE_TO_TEST,
                framework="pytest",
                coverage_target=0.90
            )
            
            # Verify generated test quality
            generator_output = result['domain_data']['test_generator_output']
            generated_tests = generator_output['generated_tests']
            
            # Verify test files were generated
            assert 'test_calculator.py' in generated_tests
            assert 'test_fibonacci.py' in generated_tests
            
            # Verify generated test code is syntactically valid Python
            for test_file, test_code in generated_tests.items():
                try:
                    ast.parse(test_code)
                    syntax_valid = True
                except SyntaxError:
                    syntax_valid = False
                
                assert syntax_valid, f"Generated test file {test_file} should be syntactically valid Python"
                
                # Verify pytest patterns
                assert 'import pytest' in test_code
                assert 'def test_' in test_code or 'class Test' in test_code
                
                # Verify assertions
                assert 'assert ' in test_code
            
            # Verify test metrics
            test_metrics = generator_output['test_metrics']
            assert test_metrics['total_test_functions'] > 0
            assert test_metrics['estimated_coverage'] >= 0.8
    
    @pytest.mark.asyncio
    async def test_testsuite_executor_phase(self):
        """Test test executor phase for coverage analysis."""
        with patch('graphtoolkit.core.executor.WorkflowExecutor') as mock_executor_class:
            mock_executor = AsyncMock()
            mock_executor_class.return_value = mock_executor
            
            # Mock complete workflow execution
            final_state = MagicMock(spec=WorkflowState)
            final_state.completed_phases = {'test_analyzer', 'test_designer', 'test_generator', 'test_executor'}
            final_state.quality_scores = {
                'test_analyzer': 0.88,
                'test_designer': 0.92,
                'test_generator': 0.89,
                'test_executor': 0.86
            }
            final_state.domain_data = {
                'test_executor_output': self.mock_llm_responses['test_executor'],
                'test_files': self.mock_llm_responses['test_generator']['generated_tests'],
                'coverage_report': self.mock_llm_responses['test_executor']['coverage_report'],
                'test_results': self.mock_llm_responses['test_executor']['test_execution_results']
            }
            
            mock_executor.run.return_value = WorkflowResult(
                state=final_state,
                outputs={},
                success=True
            )
            
            result = await execute_testsuite_workflow(
                code_to_test=SAMPLE_CODE_TO_TEST,
                framework="pytest",
                coverage_target=0.85
            )
            
            # Verify test execution results
            test_results = result['test_results']
            assert 'total_tests' in test_results
            assert 'passed' in test_results
            assert 'failed' in test_results
            assert test_results['total_tests'] > 0
            
            # Verify coverage report
            coverage_report = result['coverage_report']
            assert 'total_coverage' in coverage_report
            assert 'line_coverage' in coverage_report
            assert 'file_coverage' in coverage_report
            
            # Coverage should meet or be close to target
            total_coverage = coverage_report['total_coverage']
            assert isinstance(total_coverage, (int, float))
            assert 0.0 <= total_coverage <= 1.0
            
            # Verify performance metrics
            executor_output = result['domain_data']['test_executor_output']
            perf_metrics = executor_output['performance_metrics']
            assert 'total_execution_time' in perf_metrics
            assert 'average_test_time' in perf_metrics
            assert perf_metrics['total_execution_time'] > 0
    
    @pytest.mark.asyncio
    async def test_testsuite_workflow_with_different_frameworks(self):
        """Test TestSuite workflow with different testing frameworks."""
        frameworks = ['pytest', 'unittest', 'nose2']
        
        for framework in frameworks:
            with patch('graphtoolkit.core.executor.WorkflowExecutor') as mock_executor_class:
                mock_executor = AsyncMock()
                mock_executor_class.return_value = mock_executor
                
                # Mock framework-specific results
                final_state = MagicMock(spec=WorkflowState)
                final_state.completed_phases = {'test_analyzer', 'test_designer', 'test_generator', 'test_executor'}
                final_state.domain_data = {
                    'framework': framework,
                    'test_analyzer_output': {
                        'test_strategy': {'framework': framework},
                        'quality_score': 0.88
                    }
                }
                
                mock_executor.run.return_value = WorkflowResult(
                    state=final_state,
                    outputs={},
                    success=True
                )
                
                result = await execute_testsuite_workflow(
                    code_to_test="def simple_function(x): return x * 2",
                    framework=framework,
                    coverage_target=0.80,
                    workflow_id=f"test-{framework}"
                )
                
                # Verify framework was preserved
                assert result['success'] == True
                assert result['workflow_id'] == f"test-{framework}"
                
                # Check that framework preference is in domain data
                domain_data = result.get('domain_data', {})
                if 'framework' in domain_data:
                    assert domain_data['framework'] == framework
    
    @pytest.mark.asyncio 
    async def test_testsuite_workflow_edge_cases(self):
        """Test TestSuite workflow with edge case inputs."""
        # Test with minimal code
        minimal_code = "def add(a, b): return a + b"
        
        with patch('graphtoolkit.core.executor.WorkflowExecutor') as mock_executor_class:
            mock_executor = AsyncMock()
            mock_executor_class.return_value = mock_executor
            
            final_state = MagicMock(spec=WorkflowState)
            final_state.completed_phases = {'test_analyzer', 'test_designer', 'test_generator', 'test_executor'}
            final_state.quality_scores = {'test_generator': 0.75}  # Lower score for simple code
            final_state.domain_data = {
                'test_analyzer_output': {
                    'code_analysis': {
                        'functions': ['add'],
                        'classes': [],
                        'complexity_score': 1.0
                    },
                    'quality_score': 0.75
                }
            }
            
            mock_executor.run.return_value = WorkflowResult(
                state=final_state,
                outputs={},
                success=True
            )
            
            result = await execute_testsuite_workflow(
                code_to_test=minimal_code,
                framework="pytest",
                coverage_target=0.95,  # High target for simple code
                workflow_id="test-minimal"
            )
            
            assert result['success'] == True
            assert result['workflow_id'] == "test-minimal"
        
        # Test with complex code
        complex_code = SAMPLE_CODE_TO_TEST  # Use our comprehensive sample
        
        with patch('graphtoolkit.core.executor.WorkflowExecutor') as mock_executor_class:
            mock_executor = AsyncMock()
            mock_executor_class.return_value = mock_executor
            
            final_state = MagicMock(spec=WorkflowState)
            final_state.completed_phases = {'test_analyzer', 'test_designer', 'test_generator', 'test_executor'}
            final_state.quality_scores = {'test_generator': 0.90}  # Higher score for complex code
            final_state.domain_data = {
                'test_analyzer_output': {
                    'code_analysis': {
                        'functions': ['fibonacci', 'validate_email'],
                        'classes': ['Calculator', 'DataProcessor'],
                        'complexity_score': 8.5
                    },
                    'quality_score': 0.90
                }
            }
            
            mock_executor.run.return_value = WorkflowResult(
                state=final_state,
                outputs={},
                success=True
            )
            
            result = await execute_testsuite_workflow(
                code_to_test=complex_code,
                framework="pytest", 
                coverage_target=0.80,  # Reasonable target for complex code
                workflow_id="test-complex"
            )
            
            assert result['success'] == True
            assert result['workflow_id'] == "test-complex"
    
    @pytest.mark.asyncio
    async def test_testsuite_workflow_coverage_targets(self):
        """Test TestSuite workflow with different coverage targets."""
        coverage_targets = [0.70, 0.85, 0.95]
        
        for target in coverage_targets:
            with patch('graphtoolkit.core.executor.WorkflowExecutor') as mock_executor_class:
                mock_executor = AsyncMock()
                mock_executor_class.return_value = mock_executor
                
                # Mock coverage-aware results
                final_state = MagicMock(spec=WorkflowState)
                final_state.completed_phases = {'test_analyzer', 'test_designer', 'test_generator', 'test_executor'}
                final_state.domain_data = {
                    'coverage_target': target,
                    'test_executor_output': {
                        'coverage_report': {
                            'total_coverage': target + 0.02,  # Slightly above target
                            'line_coverage': target + 0.01,
                            'branch_coverage': target - 0.03
                        },
                        'quality_score': 0.85 if target >= 0.80 else 0.75
                    }
                }
                
                mock_executor.run.return_value = WorkflowResult(
                    state=final_state,
                    outputs={},
                    success=True
                )
                
                result = await execute_testsuite_workflow(
                    code_to_test="def test_function(): return True",
                    framework="pytest",
                    coverage_target=target,
                    workflow_id=f"test-coverage-{int(target*100)}"
                )
                
                assert result['success'] == True
                
                # Verify coverage target was considered
                executor_output = result['domain_data'].get('test_executor_output', {})
                if 'coverage_report' in executor_output:
                    coverage_report = executor_output['coverage_report']
                    total_coverage = coverage_report.get('total_coverage', 0)
                    # Coverage should be close to or above target (within reasonable range)
                    assert total_coverage >= target - 0.05  # Allow some variance
    
    def test_testsuite_workflow_schema_validation(self):
        """Test schema validation for TestSuite workflow inputs."""
        # Test valid workflow creation
        workflow_def, initial_state = create_testsuite_workflow(
            code_to_test=SAMPLE_CODE_TO_TEST,
            framework="pytest",
            coverage_target=0.85,
            workflow_id="schema-test"
        )
        
        # Verify workflow definition structure
        assert isinstance(workflow_def, WorkflowDefinition)
        assert workflow_def.domain == 'testsuite'
        assert len(workflow_def.phase_sequence) == 4
        assert workflow_def.phase_sequence == ['test_analyzer', 'test_designer', 'test_generator', 'test_executor']
        
        # Verify initial state structure
        assert isinstance(initial_state, WorkflowState)
        assert initial_state.workflow_id == "schema-test"
        assert initial_state.domain == 'testsuite'
        assert initial_state.domain_data['code_to_test'] == SAMPLE_CODE_TO_TEST
        assert initial_state.domain_data['framework'] == "pytest"
        assert initial_state.domain_data['coverage_target'] == 0.85
        
        # Test edge case inputs
        edge_cases = [
            ("", "pytest", 0.8),  # Empty code
            ("def f(): pass", "", 0.8),  # Empty framework
            ("def f(): pass", "pytest", 0.0),  # Zero coverage target
            ("def f(): pass", "pytest", 1.5),  # Coverage target > 1.0
        ]
        
        for code, framework, target in edge_cases:
            try:
                edge_workflow_def, edge_state = create_testsuite_workflow(
                    code_to_test=code,
                    framework=framework,
                    coverage_target=target
                )
                # Should still create valid structures
                assert isinstance(edge_workflow_def, WorkflowDefinition)
                assert isinstance(edge_state, WorkflowState)
            except Exception as e:
                # If validation fails, should be clear about the issue
                error_msg = str(e).lower()
                assert any(term in error_msg for term in ['code', 'framework', 'coverage', 'target'])
    
    @pytest.mark.asyncio
    async def test_testsuite_storage_integration(self):
        """Test integration with agentoolkit storage systems."""
        with patch('graphtoolkit.core.executor.WorkflowExecutor') as mock_executor_class:
            mock_executor = AsyncMock()
            mock_executor_class.return_value = mock_executor
            
            # Mock workflow with storage operations
            final_state = MagicMock(spec=WorkflowState)
            final_state.completed_phases = {'test_analyzer', 'test_designer', 'test_generator', 'test_executor'}
            final_state.phase_outputs = {
                'test_generator': StorageRef(
                    storage_type=StorageType.FS,
                    key='workflow/testsuite-001/test_files/',
                    created_at=datetime.now(),
                    size_bytes=4096
                ),
                'test_executor': StorageRef(
                    storage_type=StorageType.KV,
                    key='workflow/testsuite-001/results',
                    created_at=datetime.now(),
                    size_bytes=1024
                )
            }
            
            mock_executor.run.return_value = WorkflowResult(
                state=final_state,
                outputs={
                    'test_generator': {'storage_ref': 'fs://workflow/testsuite-001/test_files/'},
                    'test_executor': {'storage_ref': 'kv://workflow/testsuite-001/results'}
                },
                success=True
            )
            
            result = await execute_testsuite_workflow(
                code_to_test=SAMPLE_CODE_TO_TEST,
                framework="pytest",
                coverage_target=0.85,
                workflow_id="testsuite-001"
            )
            
            # Verify storage integration
            assert result['success'] == True
            outputs = result['outputs']
            
            # Check that storage references are properly formatted
            for phase_name, phase_output in outputs.items():
                if 'storage_ref' in phase_output:
                    storage_ref = phase_output['storage_ref']
                    assert storage_ref.startswith(('kv://', 'fs://'))


class TestTestSuiteWorkflowRealComponents:
    """Test TestSuite workflow using real GraphToolkit components."""
    
    def setup_method(self):
        """Set up test environment."""
        pass  # No registry clearing needed for real component tests
    
    def test_real_testsuite_workflow_creation(self):
        """Test TestSuite workflow creation with real components."""
        # Use a realistic code sample for testing
        code_sample = '''
def calculate_discount(price, discount_percent):
    """Calculate discounted price."""
    if price < 0:
        raise ValueError("Price cannot be negative")
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("Discount percent must be between 0 and 100")
    
    discount_amount = price * (discount_percent / 100)
    return price - discount_amount

def format_price(price):
    """Format price as currency string."""
    return f"${price:.2f}"
'''
        
        # Create real workflow
        workflow_def, initial_state = create_testsuite_workflow(
            code_to_test=code_sample,
            framework="pytest",
            coverage_target=0.90,
            workflow_id="real-testsuite-test"
        )
        
        # Verify real workflow components
        assert isinstance(workflow_def, WorkflowDefinition)
        assert workflow_def.domain == 'testsuite'
        assert workflow_def.version == '1.0.0'
        assert isinstance(workflow_def.created_at, datetime)
        
        # Verify phase sequence is correct for TestSuite domain
        assert len(workflow_def.phase_sequence) == 4
        expected_phases = ['test_analyzer', 'test_designer', 'test_generator', 'test_executor']
        assert workflow_def.phase_sequence == expected_phases
        
        # Verify initial state
        assert isinstance(initial_state, WorkflowState)
        assert initial_state.workflow_id == "real-testsuite-test"
        assert initial_state.domain == 'testsuite'
        assert initial_state.domain_data['code_to_test'] == code_sample
        assert initial_state.domain_data['framework'] == "pytest"
        assert initial_state.domain_data['coverage_target'] == 0.90
    
    def test_real_workflow_validation_testsuite(self):
        """Test workflow validation for TestSuite domain."""
        workflow_def, _ = create_testsuite_workflow(
            code_to_test="def multiply(a, b): return a * b",
            framework="unittest",
            coverage_target=0.75
        )
        
        # Use real GraphToolkit for validation
        from graphtoolkit import GraphToolkit
        toolkit = GraphToolkit()
        
        validation_errors = toolkit.validate_workflow(workflow_def)
        assert isinstance(validation_errors, list)
        
        # Each error should be a string if present
        for error in validation_errors:
            assert isinstance(error, str)
    
    def test_real_multiple_testsuite_workflows(self):
        """Test creating multiple TestSuite workflows concurrently."""
        code_samples = [
            "def add(a, b): return a + b",
            "def subtract(a, b): return a - b", 
            "def multiply(a, b): return a * b",
            "def divide(a, b): return a / b if b != 0 else None"
        ]
        
        workflows = []
        for i, code in enumerate(code_samples):
            workflow_def, initial_state = create_testsuite_workflow(
                code_to_test=code,
                framework=f"pytest" if i % 2 == 0 else "unittest",
                coverage_target=0.80 + (i * 0.05)  # Varying coverage targets
            )
            workflows.append((workflow_def, initial_state))
        
        # Verify all workflows are unique and valid
        workflow_ids = set()
        for workflow_def, initial_state in workflows:
            assert isinstance(workflow_def, WorkflowDefinition)
            assert isinstance(initial_state, WorkflowState)
            assert workflow_def.domain == 'testsuite'
            
            # Verify unique workflow IDs
            workflow_ids.add(initial_state.workflow_id)
            
            # Verify code sample is preserved
            assert initial_state.domain_data['code_to_test'] in code_samples
        
        # All workflows should have unique IDs
        assert len(workflow_ids) == 4
    
    def test_real_testsuite_state_operations(self):
        """Test WorkflowState operations for TestSuite workflows."""
        workflow_def, initial_state = create_testsuite_workflow(
            code_to_test=SAMPLE_CODE_TO_TEST,
            framework="pytest",
            coverage_target=0.85
        )
        
        # Test state helper methods
        current_phase_def = initial_state.get_current_phase_def()
        if current_phase_def:  # Might be None if phase not fully registered
            assert current_phase_def.phase_name == initial_state.current_phase
            assert current_phase_def.domain == 'testsuite'
        
        # Test storage reference operations for TestSuite
        test_files_ref = StorageRef(
            storage_type=StorageType.FS,
            key=f'workflow/{initial_state.workflow_id}/generated_tests/',
            created_at=datetime.now(),
            size_bytes=2048
        )
        
        coverage_ref = StorageRef(
            storage_type=StorageType.KV,
            key=f'workflow/{initial_state.workflow_id}/coverage_report',
            created_at=datetime.now(),
            size_bytes=512
        )
        
        # Add storage references
        state_with_tests = initial_state.with_storage_ref('test_generator', test_files_ref)
        state_with_coverage = state_with_tests.with_storage_ref('test_executor', coverage_ref)
        
        # Verify storage references
        assert 'test_generator' in state_with_coverage.phase_outputs
        assert 'test_executor' in state_with_coverage.phase_outputs
        assert state_with_coverage.phase_outputs['test_generator'] == test_files_ref
        assert state_with_coverage.phase_outputs['test_executor'] == coverage_ref
        
        # Verify original state is unchanged (immutable)
        assert 'test_generator' not in initial_state.phase_outputs
        assert 'test_executor' not in initial_state.phase_outputs


class TestTestSuiteWorkflowPerformance:
    """Performance tests for TestSuite workflow."""
    
    def test_testsuite_workflow_creation_performance(self):
        """Test performance of TestSuite workflow creation."""
        import time
        
        code_samples = [f"def func_{i}(x): return x * {i}" for i in range(20)]
        
        start_time = time.time()
        
        workflows = []
        for i, code in enumerate(code_samples):
            workflow_def, initial_state = create_testsuite_workflow(
                code_to_test=code,
                framework="pytest",
                coverage_target=0.80
            )
            workflows.append((workflow_def, initial_state))
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should create 20 workflows quickly (less than 3 seconds)
        assert execution_time < 3.0, f"TestSuite workflow creation took {execution_time:.2f}s, expected < 3s"
        
        # Verify all workflows were created properly
        assert len(workflows) == 20
        for workflow_def, initial_state in workflows:
            assert isinstance(workflow_def, WorkflowDefinition)
            assert isinstance(initial_state, WorkflowState)
            assert workflow_def.domain == 'testsuite'
    
    def test_testsuite_memory_usage(self):
        """Test memory usage during TestSuite workflow operations."""
        import gc
        
        # Get baseline memory
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Create many TestSuite workflows
        workflows = []
        for i in range(30):
            code = f'''
def process_data_{i}(data):
    """Process data function {i}."""
    return [x * {i} for x in data if x > 0]

class DataHandler_{i}:
    def __init__(self):
        self.data = []
    
    def add(self, item):
        self.data.append(item)
    
    def process(self):
        return process_data_{i}(self.data)
'''
            workflow_def, initial_state = create_testsuite_workflow(
                code_to_test=code,
                framework="pytest",
                coverage_target=0.85
            )
            workflows.append((workflow_def, initial_state))
        
        # Check memory growth
        gc.collect()
        final_objects = len(gc.get_objects())
        object_growth = final_objects - initial_objects
        
        # Memory growth should be reasonable (less than 15000 new objects for 30 workflows)
        assert object_growth < 15000, f"Created {object_growth} objects for 30 TestSuite workflows"
        
        # Clean up
        workflows.clear()
        gc.collect()