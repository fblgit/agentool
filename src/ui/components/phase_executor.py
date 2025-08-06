# -*- coding: utf-8 -*-
"""
Phase Executor Component - Handles individual workflow phase execution.

This component provides a clean interface for executing workflow phases,
capturing artifacts, and managing phase-specific UI updates.
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

from agentool.core.injector import AgenToolInjector
from agents.workflow import WorkflowState


@dataclass
class PhaseConfig:
    """Configuration for a workflow phase."""
    agent_name: str
    operation: str
    description: str
    input_builder: Callable[[WorkflowState], Dict[str, Any]]
    result_extractor: Callable[[Any], Dict[str, Any]]
    artifact_patterns: List[str] = field(default_factory=list)


@dataclass
class PhaseResult:
    """Result from executing a workflow phase."""
    phase_name: str
    success: bool
    duration: float
    data: Optional[Dict[str, Any]] = None
    artifacts: List[str] = field(default_factory=list)
    summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class PhaseExecutor:
    """Execute workflow phases with proper error handling and artifact tracking."""
    
    def __init__(self, injector: AgenToolInjector, debug_mode: bool = False):
        """
        Initialize the phase executor.
        
        Args:
            injector: AgenTool injector instance
            debug_mode: Whether to show debug information
        """
        self.injector = injector
        self.debug_mode = debug_mode
        self.phase_configs = self._create_phase_configs()
    
    def _create_phase_configs(self) -> Dict[str, PhaseConfig]:
        """Create configuration for each workflow phase."""
        return {
            "analyzer": PhaseConfig(
                agent_name="workflow_analyzer",
                operation="analyze",
                description="Analyzing task requirements and identifying tools",
                input_builder=lambda state: {
                    "operation": "analyze",
                    "task_description": state.task_description,
                    "workflow_id": state.workflow_id,
                    "model": state.model
                },
                result_extractor=self._extract_analyzer_result,
                artifact_patterns=[
                    "workflow/{workflow_id}/catalog",
                    "workflow/{workflow_id}/analysis",
                    "workflow/{workflow_id}/missing_tools"
                ]
            ),
            
            "specification": PhaseConfig(
                agent_name="workflow_specifier",
                operation="specify",
                description="Creating detailed specifications for each tool",
                input_builder=lambda state: {
                    "operation": "specify",
                    "workflow_id": state.workflow_id,
                    "model": state.model
                },
                result_extractor=self._extract_specification_result,
                artifact_patterns=[
                    "workflow/{workflow_id}/specs",
                    "workflow/{workflow_id}/specifications/{tool_name}",
                    "workflow/{workflow_id}/existing_tools/{tool_name}"
                ]
            ),
            
            "crafter": PhaseConfig(
                agent_name="workflow_crafter",
                operation="craft",
                description="Generating implementation code",
                input_builder=lambda state: {
                    "operation": "craft",
                    "workflow_id": state.workflow_id,
                    "model": state.model
                },
                result_extractor=self._extract_crafter_result,
                artifact_patterns=[
                    "workflow/{workflow_id}/implementations/{tool_name}",
                    "generated/{workflow_id}/{file_path}"
                ]
            ),
            
            "evaluator": PhaseConfig(
                agent_name="workflow_evaluator",
                operation="evaluate",
                description="Validating and improving generated code",
                input_builder=lambda state: {
                    "operation": "evaluate",
                    "workflow_id": state.workflow_id,
                    "model": state.model,
                    "auto_fix": True
                },
                result_extractor=self._extract_evaluator_result,
                artifact_patterns=[
                    "workflow/{workflow_id}/validations/{tool_name}",
                    "generated/{workflow_id}/final/",
                    "generated/{workflow_id}/SUMMARY.md"
                ]
            ),
            
            "test_analyzer": PhaseConfig(
                agent_name="workflow_test_analyzer",
                operation="analyze",
                description="Analyzing test requirements",
                input_builder=lambda state: {
                    "operation": "analyze",
                    "workflow_id": state.workflow_id,
                    "model": state.model
                },
                result_extractor=self._extract_test_analyzer_result,
                artifact_patterns=[
                    "workflow/{workflow_id}/test_analysis/{tool_name}"
                ]
            ),
            
            "test_stubber": PhaseConfig(
                agent_name="workflow_test_stubber",
                operation="stub",
                description="Creating test structure and placeholders",
                input_builder=lambda state: {
                    "operation": "stub",
                    "workflow_id": state.workflow_id,
                    "model": state.model
                },
                result_extractor=self._extract_test_stubber_result,
                artifact_patterns=[
                    "workflow/{workflow_id}/test_stub/{tool_name}",
                    "generated/{workflow_id}/test_stubs/test_{tool_name}.py"
                ]
            ),
            
            "test_crafter": PhaseConfig(
                agent_name="workflow_test_crafter",
                operation="craft",
                description="Implementing comprehensive tests",
                input_builder=lambda state: {
                    "operation": "craft",
                    "workflow_id": state.workflow_id,
                    "model": state.model
                },
                result_extractor=self._extract_test_crafter_result,
                artifact_patterns=[
                    "workflow/{workflow_id}/test_implementation/{tool_name}",
                    "generated/{workflow_id}/tests/test_{tool_name}.py",
                    "generated/{workflow_id}/tests/TEST_SUMMARY_{tool_name}.md"
                ]
            )
        }
    
    def execute_phase(self, phase_name: str, workflow_state: WorkflowState) -> PhaseResult:
        """
        Execute a single workflow phase.
        
        Args:
            phase_name: Name of the phase to execute
            workflow_state: Current workflow state
            
        Returns:
            PhaseResult with execution details
        """
        if phase_name not in self.phase_configs:
            raise ValueError(f"Unknown phase: {phase_name}")
        
        config = self.phase_configs[phase_name]
        start_time = time.time()
        
        try:
            # Build input for the phase
            phase_input = config.input_builder(workflow_state)
            
            # Execute the phase (handle tool-specific execution)
            if phase_name in ["test_analyzer", "test_stubber", "test_crafter"]:
                # These phases iterate over multiple tools
                result = self._execute_multi_tool_phase(phase_name, config, phase_input, workflow_state)
            else:
                # Single execution phases (but they still handle multiple tools internally)
                result = asyncio.run(self.injector.run(config.agent_name, phase_input))
            
            # Extract and process result
            data = self._extract_result_data(result)
            extracted_data = config.result_extractor(data)
            
            # Capture artifacts
            artifacts = self._capture_artifacts(config, workflow_state.workflow_id, extracted_data)
            
            # For multi-tool phases, also capture dynamic artifacts based on tools processed
            if phase_name in ["test_analyzer", "test_stubber", "test_crafter"] and 'tools' in extracted_data:
                for tool_name in extracted_data.get('tools', []):
                    if not tool_name:
                        continue
                    
                    # Add tool-specific artifacts based on phase
                    if phase_name == "test_analyzer":
                        artifact = f"storage_kv:workflow/{workflow_state.workflow_id}/test_analysis/{tool_name}"
                        if artifact not in artifacts:
                            artifacts.append(artifact)
                    
                    elif phase_name == "test_stubber":
                        kv_artifact = f"storage_kv:workflow/{workflow_state.workflow_id}/test_stub/{tool_name}"
                        fs_artifact = f"storage_fs:generated/{workflow_state.workflow_id}/test_stubs/test_{tool_name}.py"
                        if kv_artifact not in artifacts:
                            artifacts.append(kv_artifact)
                        if fs_artifact not in artifacts:
                            artifacts.append(fs_artifact)
                    
                    elif phase_name == "test_crafter":
                        kv_artifact = f"storage_kv:workflow/{workflow_state.workflow_id}/test_implementation/{tool_name}"
                        test_artifact = f"storage_fs:generated/{workflow_state.workflow_id}/tests/test_{tool_name}.py"
                        summary_artifact = f"storage_fs:generated/{workflow_state.workflow_id}/tests/TEST_SUMMARY_{tool_name}.md"
                        if kv_artifact not in artifacts:
                            artifacts.append(kv_artifact)
                        if test_artifact not in artifacts:
                            artifacts.append(test_artifact)
                        if summary_artifact not in artifacts:
                            artifacts.append(summary_artifact)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Update workflow metadata
            if workflow_state.metadata:
                workflow_state.metadata.phase_durations[phase_name] = duration
                workflow_state.metadata.models_used[phase_name] = workflow_state.model
                workflow_state.metadata.current_phase = phase_name
            
            return PhaseResult(
                phase_name=phase_name,
                success=True,
                duration=duration,
                data=extracted_data,
                artifacts=artifacts,
                summary=self._create_phase_summary(phase_name, extracted_data)
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            # Log error
            if self.debug_mode:
                import traceback
                error_msg = f"{error_msg}\n{traceback.format_exc()}"
            
            # Update workflow state
            workflow_state.errors.append(f"{phase_name} error: {error_msg}")
            
            return PhaseResult(
                phase_name=phase_name,
                success=False,
                duration=duration,
                error=error_msg
            )
    
    def _execute_multi_tool_phase(self, phase_name: str, config: PhaseConfig, 
                                  base_input: Dict[str, Any], workflow_state: WorkflowState) -> Dict[str, Any]:
        """Execute phases that iterate over multiple tools."""
        # Get list of tools from specifications
        specs_key = f'workflow/{workflow_state.workflow_id}/specs'
        specs_result = asyncio.run(self.injector.run('storage_kv', {
            'operation': 'get',
            'key': specs_key
        }))
        
        specs_data = self._extract_result_data(specs_result)
        if not specs_data.get('data', {}).get('exists', False):
            raise RuntimeError(f"No specifications found for {phase_name}")
        
        spec_output = json.loads(specs_data['data']['value'])
        tools_processed = 0
        results = []
        
        # Process each tool
        for spec in spec_output.get('specifications', []):
            tool_name = spec.get('name')
            if not tool_name:
                continue
            
            # Check prerequisites based on phase - matching workflow.py logic
            if phase_name == "test_stubber":
                # Check if test analysis exists
                check_key = f'workflow/{workflow_state.workflow_id}/test_analysis/{tool_name}'
            elif phase_name == "test_crafter":
                # Check if test stub exists
                check_key = f'workflow/{workflow_state.workflow_id}/test_stub/{tool_name}'
            else:
                # test_analyzer doesn't need prerequisites from other test phases
                check_key = None
            
            if check_key:
                check_result = asyncio.run(self.injector.run('storage_kv', {
                    'operation': 'exists',
                    'key': check_key
                }))
                check_data = self._extract_result_data(check_result)
                if not check_data.get('data', {}).get('exists', False):
                    continue
            
            # Execute for this tool - add tool_name for test phases
            tool_input = base_input.copy()
            if phase_name in ["test_analyzer", "test_stubber", "test_crafter"]:
                tool_input['tool_name'] = tool_name
            
            result = asyncio.run(self.injector.run(config.agent_name, tool_input))
            result_data = self._extract_result_data(result)
            
            if result_data.get('success', False):
                tools_processed += 1
                results.append(result_data)
        
        # Return aggregated result with tool information
        return {
            'success': tools_processed > 0,
            'message': f'Processed {tools_processed} tools',
            'data': {
                'tools_processed': tools_processed,
                'results': results,
                'tools': [spec.get('name') for spec in spec_output.get('specifications', []) if spec.get('name')]
            }
        }
    
    def _extract_result_data(self, result: Any) -> Dict[str, Any]:
        """Extract data from various result formats."""
        if hasattr(result, 'output'):
            return json.loads(result.output)
        elif hasattr(result, 'data'):
            return result.data
        else:
            return result
    
    def _capture_artifacts(self, config: PhaseConfig, workflow_id: str, 
                          extracted_data: Optional[Dict[str, Any]] = None) -> List[str]:
        """Capture artifacts created during phase execution."""
        artifacts = []
        
        # For single-tool phases, extract tool-specific data
        tool_name = 'unknown'
        file_path = ''
        
        if extracted_data:
            # Extract tool name from various sources
            # TODO: Fix missing_tools artifact storage - temporarily disabled
            # if 'missing_tools' in extracted_data:
            #     # Analyzer phase - capture all missing tools as a single artifact
            #     if extracted_data.get('missing_tools'):
            #         artifact = f"storage_kv:workflow/{workflow_id}/missing_tools"
            #         if artifact not in artifacts:
            #             artifacts.append(artifact)
            
            if 'specifications' in extracted_data:
                # Specification phase - capture all specifications
                for spec in extracted_data.get('specifications', []):
                    if isinstance(spec, dict) and 'name' in spec:
                        artifacts.append(f"storage_kv:workflow/{workflow_id}/specifications/{spec['name']}")
                        artifacts.append(f"storage_kv:workflow/{workflow_id}/existing_tools/{spec['name']}")
            
            if 'file_path' in extracted_data:
                # Crafter phase - can generate multiple tools
                file_path = extracted_data.get('file_path', '')
                if file_path:
                    import os
                    tool_name = os.path.basename(file_path).replace('.py', '')
            
            # Handle multiple implementations from crafter
            if 'implementations' in extracted_data:
                for impl in extracted_data.get('implementations', []):
                    if isinstance(impl, dict):
                        impl_tool = impl.get('tool_name', '')
                        impl_path = impl.get('file_path', '')
                        if impl_tool:
                            artifact = f"storage_kv:workflow/{workflow_id}/implementations/{impl_tool}"
                            if artifact not in artifacts:
                                artifacts.append(artifact)
                        if impl_path:
                            artifact = f"storage_fs:generated/{workflow_id}/{impl_path}"
                            if artifact not in artifacts:
                                artifacts.append(artifact)
                
                # Also add the summary artifact for crafter
                summary_artifact = f"storage_kv:workflow/{workflow_id}/implementations_summary"
                if summary_artifact not in artifacts:
                    artifacts.append(summary_artifact)
            
            # For evaluator phase, extract from validation data
            if 'syntax_valid' in extracted_data or 'ready_for_deployment' in extracted_data:
                # Try to get tool name from the phase results or state_ref
                state_ref = extracted_data.get('state_ref', '')
                if '/' in state_ref:
                    tool_name = state_ref.split('/')[-1]
                else:
                    tool_name = extracted_data.get('tool_name', tool_name)
            
            # The evaluator processes ALL tools in one run and creates validations for each
            if 'validations' in extracted_data:
                for validation in extracted_data.get('validations', []):
                    if isinstance(validation, dict):
                        val_tool = validation.get('tool_name', '')
                        if val_tool:
                            artifacts.append(f"storage_kv:workflow/{workflow_id}/validations/{val_tool}")
            
            # Handle summary validations
            if 'tools_validated' in extracted_data or 'tools_ready' in extracted_data:
                # Aggregated validation results
                artifacts.append(f"storage_kv:workflow/{workflow_id}/validations_summary")
        
        # Process standard patterns
        for pattern in config.artifact_patterns:
            # Skip special handled patterns
            if '{tool_name}' in pattern and tool_name == 'unknown':
                continue
            
            # Replace placeholders
            pattern = pattern.replace("{workflow_id}", workflow_id)
            pattern = pattern.replace("{tool_name}", tool_name)
            pattern = pattern.replace("{file_path}", file_path)
            
            # Determine storage type and add artifact
            if pattern.startswith("workflow/"):
                full_artifact = f"storage_kv:{pattern}"
            elif pattern.startswith("generated/"):
                full_artifact = f"storage_fs:{pattern}"
            else:
                full_artifact = pattern
            
            if full_artifact not in artifacts:
                artifacts.append(full_artifact)
        
        return artifacts
    
    def _create_phase_summary(self, phase_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of phase results."""
        if phase_name == "analyzer":
            return {
                "existing_tools": len(data.get("existing_tools", [])),
                "missing_tools": len(data.get("missing_tools", [])),
                "solution_name": data.get("name", "Unknown")
            }
        
        elif phase_name == "specification":
            return {
                "specifications_created": len(data.get("specifications", [])),
                "tools": [spec.get("name") for spec in data.get("specifications", [])]
            }
        
        elif phase_name == "crafter":
            # Handle both single and multiple implementations
            if 'implementations' in data:
                # Multiple tools generated - this is the summary from workflow_crafter
                return {
                    "lines_of_code": data.get('total_lines', 0),
                    "tools_generated": data.get('total_tools', len(data.get('implementations', []))),
                    "implementations": [impl.get('tool_name', '') for impl in data.get('implementations', [])],
                    "files": data.get('files', [])
                }
            else:
                # Single tool generated (backward compatibility)
                code = data.get("code", "")
                return {
                    "lines_of_code": code.count("\n") if code else 0,
                    "file_path": data.get("file_path", ""),
                    "code_size_bytes": len(code) if code else 0
                }
        
        elif phase_name == "evaluator":
            # Handle both single and multiple validations
            if 'validations' in data or 'tools_validated' in data or 'tools_ready' in data:
                # Multiple tools validated - this is the summary from workflow_evaluator
                return {
                    "tools_validated": data.get('tools_validated', data.get('total_tools', 0)),
                    "tools_ready": data.get('tools_ready', 0),
                    "total_issues": data.get('total_issues', 0),
                    "total_fixes": data.get('total_fixes', 0),
                    "all_ready": data.get('tools_ready', 0) == data.get('total_tools', 1) and data.get('tools_ready', 0) > 0
                }
            else:
                # Single tool validated (backward compatibility)
                return {
                    "syntax_valid": data.get("syntax_valid", False),
                    "ready_for_deployment": data.get("ready_for_deployment", False),
                    "issues_found": len(data.get("issues", [])),
                    "improvements_made": len(data.get("improvements", []))
                }
        
        elif phase_name == "test_analyzer":
            tools_count = data.get("tools_processed", 0)
            # For multi-tool phases, handle tools list from results
            if 'tools' in data:
                tools_count = len(data.get('tools', []))
            return {
                "tools_analyzed": tools_count
            }
        
        elif phase_name == "test_stubber":
            tools_count = data.get("tools_processed", 0)
            if 'tools' in data:
                tools_count = len([t for t in data.get('tools', []) if t])
            return {
                "test_stubs_created": tools_count
            }
        
        elif phase_name == "test_crafter":
            tools_count = data.get("tools_processed", 0)
            if 'tools' in data:
                tools_count = len([t for t in data.get('tools', []) if t])
            return {
                "test_suites_implemented": tools_count,
                "test_coverage": "Comprehensive",
                "test_files_count": tools_count
            }
        
        return {}
    
    # Result extractors for each phase
    def _extract_analyzer_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract analyzer phase results."""
        return data.get("data", {})
    
    def _extract_specification_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract specification phase results."""
        return data.get("data", {})
    
    def _extract_crafter_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract crafter phase results."""
        return data.get("data", {})
    
    def _extract_evaluator_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract evaluator phase results."""
        validation_data = data.get("data", {})
        # Add computed fields matching workflow.py
        validation_data["issues_count"] = len(validation_data.get("issues", []))
        validation_data["fixes_count"] = len(validation_data.get("fixes_applied", []))
        validation_data["improvements_count"] = len(validation_data.get("improvements", []))
        return validation_data
    
    def _extract_test_analyzer_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract test analyzer phase results."""
        return data.get("data", {})
    
    def _extract_test_stubber_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract test stubber phase results."""
        return data.get("data", {})
    
    def _extract_test_crafter_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract test crafter phase results."""
        return data.get("data", {})