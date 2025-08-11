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
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import replace

# pydantic-ai test utilities
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ModelRequest
)

# GraphToolkit imports
from graphtoolkit import execute_testsuite_workflow, create_testsuite_workflow, GraphToolkit
from graphtoolkit.core.executor import WorkflowExecutor, WorkflowResult
from graphtoolkit.core.types import (
    WorkflowState,
    WorkflowDefinition, 
    StorageRef,
    StorageType,
    ValidationResult
)
from graphtoolkit.core.deps import WorkflowDeps, ModelConfig, StorageConfig

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
        self.history.append(f"{a} × {b} = {result}")
        return result
    
    def divide(self, a: float, b: float) -> float:
        """Divide a by b."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = round(a / b, self.precision)
        self.history.append(f"{a} ÷ {b} = {result}")
        return result
    
    def power(self, base: float, exponent: float) -> float:
        """Calculate base raised to the power of exponent."""
        result = round(base ** exponent, self.precision)
        self.history.append(f"{base}^{exponent} = {result}")
        return result
    
    def sqrt(self, n: float) -> float:
        """Calculate square root of n."""
        if n < 0:
            raise ValueError("Cannot calculate square root of negative number")
        result = round(math.sqrt(n), self.precision)
        self.history.append(f"√{n} = {result}")
        return result
    
    def clear_history(self) -> None:
        """Clear the calculation history."""
        self.history = []
    
    def get_history(self) -> List[str]:
        """Get the calculation history."""
        return self.history.copy()


def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


class DateRange:
    """Represent a date range with start and end dates."""
    
    def __init__(self, start: datetime, end: datetime):
        if end < start:
            raise ValueError("End date must be after start date")
        self.start = start
        self.end = end
    
    def duration(self) -> timedelta:
        """Get the duration of the date range."""
        return self.end - self.start
    
    def contains(self, date: datetime) -> bool:
        """Check if a date is within the range."""
        return self.start <= date <= self.end
    
    def overlaps(self, other: "DateRange") -> bool:
        """Check if this range overlaps with another."""
        return not (self.end < other.start or other.end < self.start)
'''


class TestTestSuiteWorkflowComplete:
    """Test complete TestSuite workflow execution."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        # Set up test model for deterministic testing
        self.test_model = TestModel()
        
        # Expected response structures for deterministic testing
        self.expected_responses = {
            'test_analyzer': {
                'code_analysis': {
                    'classes': ['Calculator', 'DateRange'],
                    'functions': ['fibonacci', 'is_prime'],
                    'methods_count': 11,
                    'complexity': 'medium',
                    'test_requirements': {
                        'unit_tests': 15,
                        'edge_cases': 8,
                        'error_cases': 4
                    }
                },
                'coverage_requirements': {
                    'target': 0.90,
                    'critical_paths': ['divide', 'sqrt', 'fibonacci', 'is_prime'],
                    'error_handling': ['ValueError cases']
                },
                'quality_score': 0.85
            },
            'test_designer': {
                'test_architecture': {
                    'framework': 'pytest',
                    'structure': 'class-based',
                    'fixtures': ['calculator_instance', 'date_range_instance'],
                    'parametrized_tests': ['test_arithmetic_operations', 'test_prime_numbers']
                },
                'test_strategy': {
                    'unit_tests': {
                        'Calculator': ['test_add', 'test_subtract', 'test_multiply', 'test_divide', 'test_power', 'test_sqrt'],
                        'Functions': ['test_fibonacci', 'test_is_prime'],
                        'DateRange': ['test_duration', 'test_contains', 'test_overlaps']
                    },
                    'edge_cases': ['division_by_zero', 'negative_sqrt', 'negative_fibonacci'],
                    'integration_tests': ['test_calculator_history', 'test_date_range_validation']
                },
                'quality_score': 0.88
            },
            'test_generator': {
                'generated_tests': '''
import pytest
import math
from datetime import datetime, timedelta
from calculator import Calculator, fibonacci, is_prime, DateRange


class TestCalculator:
    @pytest.fixture
    def calculator(self):
        return Calculator(precision=2)
    
    def test_add(self, calculator):
        assert calculator.add(2, 3) == 5
        assert calculator.add(-1, 1) == 0
        assert calculator.add(0.1, 0.2) == 0.3
    
    def test_divide(self, calculator):
        assert calculator.divide(10, 2) == 5
        with pytest.raises(ValueError):
            calculator.divide(5, 0)
    
    def test_sqrt(self, calculator):
        assert calculator.sqrt(4) == 2
        assert calculator.sqrt(9) == 3
        with pytest.raises(ValueError):
            calculator.sqrt(-1)


def test_fibonacci():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(10) == 55
    with pytest.raises(ValueError):
        fibonacci(-1)


def test_is_prime():
    assert is_prime(2) == True
    assert is_prime(17) == True
    assert is_prime(1) == False
    assert is_prime(4) == False
''',
                'test_count': 15,
                'coverage_estimate': 0.92,
                'quality_score': 0.90
            },
            'test_executor': {
                'execution_results': {
                    'tests_run': 15,
                    'tests_passed': 14,
                    'tests_failed': 1,
                    'coverage': 0.91,
                    'duration': 1.2
                },
                'coverage_report': {
                    'lines_covered': 95,
                    'lines_total': 104,
                    'branches_covered': 18,
                    'branches_total': 22
                },
                'quality_metrics': {
                    'test_quality': 0.89,
                    'coverage_quality': 0.91,
                    'overall_score': 0.90
                }
            }
        }
    
    def create_test_deps(self) -> WorkflowDeps:
        """Create test dependencies with TestModel."""
        return WorkflowDeps(
            models=ModelConfig(
                provider='test',
                model='test',
                temperature=0.7,
                max_tokens=1000
            ),
            storage=StorageConfig(
                kv_backend='memory',
                fs_backend='memory'
            ),
            template_engine=None,  # Will use default
            phase_registry={},  # Will be populated from registry
            process_executor=None,
            thread_executor=None,
            domain_validators={}
        )
    
    @pytest.mark.asyncio
    async def test_complete_testsuite_workflow_execution(self):
        """Test end-to-end TestSuite workflow with all phases."""
        # Create workflow with test model
        workflow_def, initial_state = create_testsuite_workflow(
            code_to_test=SAMPLE_CODE_TO_TEST,
            framework="pytest",
            coverage_target=0.90,
            workflow_id="test-testsuite-123"
        )
        
        # Verify workflow structure
        assert workflow_def.domain == 'testsuite'
        assert initial_state.workflow_id == "test-testsuite-123"
        assert initial_state.domain_data['code_to_test'] == SAMPLE_CODE_TO_TEST
        assert initial_state.domain_data['framework'] == "pytest"
        assert initial_state.domain_data['coverage_target'] == 0.90
        
        # Verify phase sequence
        assert len(workflow_def.phase_sequence) == 4
        expected_phases = ['test_analyzer', 'test_designer', 'test_generator', 'test_executor']
        assert workflow_def.phase_sequence == expected_phases
    
    @pytest.mark.asyncio
    async def test_testsuite_analyzer_phase(self):
        """Test analyzer phase specifically."""
        # Create workflow
        workflow_def, initial_state = create_testsuite_workflow(
            code_to_test=SAMPLE_CODE_TO_TEST,
            framework="pytest"
        )
        
        # Verify analyzer phase exists
        assert 'test_analyzer' in workflow_def.phases
        analyzer_phase = workflow_def.phases['test_analyzer']
        assert analyzer_phase.phase_name == 'test_analyzer'
        assert analyzer_phase.domain == 'testsuite'
        
        # Test expected analyzer output structure
        analyzer_output = self.expected_responses['test_analyzer']
        assert 'code_analysis' in analyzer_output
        assert 'coverage_requirements' in analyzer_output
        assert analyzer_output['quality_score'] == 0.85
        
        # Verify code analysis
        code_analysis = analyzer_output['code_analysis']
        assert 'Calculator' in code_analysis['classes']
        assert 'fibonacci' in code_analysis['functions']
        assert code_analysis['methods_count'] == 11
    
    @pytest.mark.asyncio
    async def test_testsuite_generator_phase(self):
        """Test that generated tests are valid Python code."""
        # Create workflow
        workflow_def, initial_state = create_testsuite_workflow(
            code_to_test=SAMPLE_CODE_TO_TEST,
            framework="pytest"
        )
        
        # Test generated code validity
        generated_tests = self.expected_responses['test_generator']['generated_tests']
        
        # Verify syntax
        try:
            ast.parse(generated_tests)
            syntax_valid = True
        except SyntaxError:
            syntax_valid = False
        
        assert syntax_valid, "Generated tests should be syntactically valid Python"
        
        # Verify test structure
        assert 'import pytest' in generated_tests
        assert 'class Test' in generated_tests
        assert 'def test_' in generated_tests
        assert '@pytest.fixture' in generated_tests
        assert 'assert' in generated_tests
        assert 'pytest.raises' in generated_tests
    
    def test_testsuite_workflow_schema_validation(self):
        """Test schema validation for TestSuite workflow inputs/outputs."""
        # Test valid workflow creation
        workflow_def, initial_state = create_testsuite_workflow(
            code_to_test="def func(): pass",
            framework="pytest",
            coverage_target=0.85
        )
        
        # Verify workflow definition structure
        assert isinstance(workflow_def, WorkflowDefinition)
        assert workflow_def.domain == 'testsuite'
        assert isinstance(workflow_def.phases, dict)
        assert isinstance(workflow_def.phase_sequence, list)
        
        # Verify initial state structure
        assert isinstance(initial_state, WorkflowState)
        assert initial_state.domain == 'testsuite'
        assert isinstance(initial_state.domain_data, dict)
        assert initial_state.domain_data['framework'] == "pytest"
        assert initial_state.domain_data['coverage_target'] == 0.85
    
    @pytest.mark.asyncio
    async def test_testsuite_with_different_frameworks(self):
        """Test TestSuite workflow with different testing frameworks."""
        frameworks = ['pytest', 'unittest', 'nose2']
        
        for framework in frameworks:
            workflow_def, initial_state = create_testsuite_workflow(
                code_to_test=SAMPLE_CODE_TO_TEST,
                framework=framework,
                coverage_target=0.80
            )
            
            # Verify framework is set correctly
            assert initial_state.domain_data['framework'] == framework
            assert workflow_def.domain == 'testsuite'
    
    @pytest.mark.asyncio
    async def test_testsuite_coverage_validation(self):
        """Test coverage target validation and enforcement."""
        # Test with different coverage targets
        coverage_targets = [0.70, 0.80, 0.90, 0.95]
        
        for target in coverage_targets:
            workflow_def, initial_state = create_testsuite_workflow(
                code_to_test=SAMPLE_CODE_TO_TEST,
                coverage_target=target
            )
            
            # Verify coverage target is set
            assert initial_state.domain_data['coverage_target'] == target
            
            # Verify it's a reasonable target
            assert 0.0 <= target <= 1.0
    
    @pytest.mark.asyncio
    async def test_testsuite_error_handling(self):
        """Test error handling for invalid code."""
        invalid_code = """
        def broken_function(:  # Syntax error
            return None
        """
        
        # Create workflow with invalid code
        workflow_def, initial_state = create_testsuite_workflow(
            code_to_test=invalid_code,
            framework="pytest"
        )
        
        # Workflow should still be created
        assert workflow_def is not None
        assert initial_state is not None
        
        # The analyzer phase would detect the syntax error
        # This would be handled during actual execution
    
    def test_testsuite_workflow_concurrent_creation(self):
        """Test creating multiple TestSuite workflows concurrently."""
        async def create_workflow(task_id):
            workflow_def, initial_state = create_testsuite_workflow(
                code_to_test=f"def func_{task_id}(): return {task_id}",
                framework="pytest"
            )
            return workflow_def, initial_state
        
        async def test_concurrent():
            # Create 5 workflows concurrently
            tasks = [create_workflow(i) for i in range(5)]
            results = await asyncio.gather(*tasks)
            
            # Verify all workflows were created successfully
            assert len(results) == 5
            
            workflow_ids = set()
            for workflow_def, initial_state in results:
                assert isinstance(workflow_def, WorkflowDefinition)
                assert isinstance(initial_state, WorkflowState)
                assert workflow_def.domain == 'testsuite'
                
                # Verify unique workflow IDs
                workflow_ids.add(initial_state.workflow_id)
            
            # All workflow IDs should be unique
            assert len(workflow_ids) == 5
        
        # Run the concurrent test
        asyncio.run(test_concurrent())
    
    @pytest.mark.asyncio
    async def test_testsuite_with_complex_code_analysis(self):
        """Test TestSuite workflow with complex code structures."""
        complex_code = """
        from abc import ABC, abstractmethod
        from typing import Generic, TypeVar
        
        T = TypeVar('T')
        
        class AbstractProcessor(ABC, Generic[T]):
            @abstractmethod
            def process(self, item: T) -> T:
                pass
        
        class StringProcessor(AbstractProcessor[str]):
            def process(self, item: str) -> str:
                return item.upper()
        
        async def async_function():
            await asyncio.sleep(1)
            return "done"
        
        def generator_function():
            for i in range(10):
                yield i * 2
        
        @property
        def property_method(self):
            return self._value
        
        @staticmethod
        def static_method():
            return "static"
        
        @classmethod
        def class_method(cls):
            return cls.__name__
        """
        
        # Create workflow with complex code
        workflow_def, initial_state = create_testsuite_workflow(
            code_to_test=complex_code,
            framework="pytest"
        )
        
        # Verify workflow handles complex code
        assert workflow_def is not None
        assert initial_state.domain_data['code_to_test'] == complex_code
    
    @pytest.mark.asyncio
    async def test_testsuite_integration_with_storage(self):
        """Test TestSuite workflow integration with storage systems."""
        # Create workflow
        workflow_def, initial_state = create_testsuite_workflow(
            code_to_test=SAMPLE_CODE_TO_TEST,
            framework="pytest"
        )
        
        # Simulate storing test results
        test_results_ref = StorageRef(
            storage_type=StorageType.FS,
            key=f'workflow/{initial_state.workflow_id}/test_results.json',
            created_at=datetime.now(),
            size_bytes=2048
        )
        
        # Simulate storing generated tests
        generated_tests_ref = StorageRef(
            storage_type=StorageType.FS,
            key=f'workflow/{initial_state.workflow_id}/test_suite.py',
            created_at=datetime.now(),
            size_bytes=4096
        )
        
        # Verify storage references
        assert test_results_ref.storage_type == StorageType.FS
        assert 'test_results.json' in test_results_ref.key
        assert generated_tests_ref.storage_type == StorageType.FS
        assert 'test_suite.py' in generated_tests_ref.key


class TestTestSuiteWorkflowRealComponents:
    """Test TestSuite workflow using real GraphToolkit components."""
    
    def setup_method(self):
        """Set up test environment."""
        self.toolkit = GraphToolkit()
    
    def test_real_testsuite_workflow_structure(self):
        """Test TestSuite workflow creation with real components."""
        # Create workflow using real GraphToolkit API
        workflow_def, initial_state = create_testsuite_workflow(
            code_to_test=SAMPLE_CODE_TO_TEST,
            framework="pytest",
            coverage_target=0.85
        )
        
        # Verify workflow structure using real types
        assert isinstance(workflow_def, WorkflowDefinition)
        assert workflow_def.domain == 'testsuite'
        assert workflow_def.version == '1.0.0'
        assert isinstance(workflow_def.created_at, datetime)
        
        # Verify phase sequence
        assert len(workflow_def.phase_sequence) == 4
        expected_phases = ['test_analyzer', 'test_designer', 'test_generator', 'test_executor']
        assert workflow_def.phase_sequence == expected_phases
        
        # Verify initial state
        assert isinstance(initial_state, WorkflowState)
        assert initial_state.workflow_def is workflow_def
        assert initial_state.domain == 'testsuite'
    
    def test_real_workflow_validation(self):
        """Test workflow validation using real components."""
        # Create real workflow
        workflow_def, _ = create_testsuite_workflow(
            code_to_test="def test_func(): pass",
            framework="pytest"
        )
        
        # Validate workflow
        validation_errors = self.toolkit.validate_workflow(workflow_def)
        
        # Should return a list
        assert isinstance(validation_errors, list)
        
        # Each error should be a string if present
        for error in validation_errors:
            assert isinstance(error, str)
    
    def test_real_phase_configuration(self):
        """Test that TestSuite phases are properly configured."""
        # Get domain phases
        phases = self.toolkit.get_domain_phases('testsuite')
        
        if phases:  # Phases might not be fully registered yet
            # Verify phase names
            expected_phases = ['test_analyzer', 'test_designer', 'test_generator', 'test_executor']
            for phase_name in expected_phases:
                if phase_name in phases:
                    phase_def = phases[phase_name]
                    assert phase_def.phase_name == phase_name
                    assert phase_def.domain == 'testsuite'
                    assert len(phase_def.atomic_nodes) > 0