"""
GraphToolkit Blockchain Domain Definition.

Domain for designing and auditing blockchain smart contracts.
Phases: Contract Analyzer → Smart Contract Designer → Auditor → Optimizer
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from ..core.types import (
    PhaseDefinition,
    TemplateConfig,
    ModelParameters,
    StorageType
)
from ..core.registry import register_phase


# Input/Output Schemas for Blockchain Domain

class ContractAnalyzerInput(BaseModel):
    """Input for contract analysis phase."""
    requirements: str = Field(description="Smart contract requirements")
    blockchain_platform: str = Field(description="Target blockchain (Ethereum, Solana, etc)")
    use_case: str = Field(description="Contract use case description")
    security_requirements: List[str] = Field(description="Security requirements")
    compliance_needs: Optional[List[str]] = Field(None, description="Regulatory compliance needs")
    existing_contracts: Optional[List[str]] = Field(None, description="Related existing contracts")


class ContractAnalyzerOutput(BaseModel):
    """Output from contract analysis phase."""
    contract_type: str = Field(description="Type of smart contract needed")
    required_functions: List[Dict[str, Any]] = Field(description="Required contract functions")
    state_variables: List[Dict[str, str]] = Field(description="State variables needed")
    events: List[Dict[str, Any]] = Field(description="Events to emit")
    security_patterns: List[str] = Field(description="Security patterns to implement")
    gas_optimization_opportunities: List[str] = Field(description="Gas optimization opportunities")
    integration_points: List[Dict[str, Any]] = Field(description="External integration points")


class SmartContractDesignerInput(BaseModel):
    """Input for smart contract design phase."""
    contract_type: str = Field(description="Type of contract to design")
    required_functions: List[Dict[str, Any]] = Field(description="Functions to implement")
    state_variables: List[Dict[str, str]] = Field(description="State variables")
    events: List[Dict[str, Any]] = Field(description="Events to implement")
    security_patterns: List[str] = Field(description="Security patterns to follow")
    blockchain_platform: str = Field(description="Target blockchain platform")


class SmartContractDesignerOutput(BaseModel):
    """Output from smart contract design phase."""
    contract_code: str = Field(description="Smart contract source code")
    contract_architecture: Dict[str, Any] = Field(description="Contract architecture diagram")
    function_specifications: Dict[str, Dict[str, Any]] = Field(description="Detailed function specs")
    access_control: Dict[str, List[str]] = Field(description="Access control matrix")
    upgrade_strategy: Optional[Dict[str, Any]] = Field(None, description="Upgrade mechanism")
    deployment_instructions: Dict[str, Any] = Field(description="Deployment instructions")


class AuditorInput(BaseModel):
    """Input for audit phase."""
    contract_code: str = Field(description="Contract code to audit")
    security_patterns: List[str] = Field(description="Expected security patterns")
    blockchain_platform: str = Field(description="Blockchain platform")
    audit_scope: List[str] = Field(description="Audit scope areas")
    known_vulnerabilities: Optional[List[str]] = Field(None, description="Known vulnerabilities to check")


class AuditorOutput(BaseModel):
    """Output from audit phase."""
    vulnerabilities_found: List[Dict[str, Any]] = Field(description="Security vulnerabilities found")
    security_score: float = Field(description="Security score (0-100)")
    gas_analysis: Dict[str, Any] = Field(description="Gas usage analysis")
    best_practices_compliance: Dict[str, bool] = Field(description="Best practices compliance")
    recommendations: List[Dict[str, str]] = Field(description="Security recommendations")
    audit_report: str = Field(description="Comprehensive audit report")
    risk_assessment: Dict[str, str] = Field(description="Risk assessment by category")
    quality_score: float = Field(description="Overall quality score")


class OptimizerInput(BaseModel):
    """Input for optimization phase."""
    contract_code: str = Field(description="Contract code to optimize")
    gas_analysis: Dict[str, Any] = Field(description="Gas usage analysis")
    optimization_goals: List[str] = Field(description="Optimization priorities")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Optimization constraints")


class OptimizerOutput(BaseModel):
    """Output from optimization phase."""
    optimized_code: str = Field(description="Optimized contract code")
    gas_savings: Dict[str, float] = Field(description="Gas savings by function")
    optimization_changes: List[Dict[str, str]] = Field(description="Changes made for optimization")
    performance_metrics: Dict[str, Any] = Field(description="Performance improvements")
    trade_offs: List[str] = Field(description="Trade-offs made during optimization")
    final_gas_cost: Dict[str, int] = Field(description="Final gas costs by function")


# Phase Definitions for Blockchain Domain

contract_analyzer_phase = PhaseDefinition(
    phase_name="contract_analyzer",
    atomic_nodes=[
        'dependency_check',
        'load_dependencies',
        'template_render',
        'llm_call',
        'schema_validation',
        'save_phase_output',
        'state_update',
        'quality_gate'
    ],
    input_schema=ContractAnalyzerInput,
    output_schema=ContractAnalyzerOutput,
    dependencies=[],
    templates=TemplateConfig(
        system_template="templates/system/contract_analyzer.jinja",
        user_template="templates/prompts/analyze_contract_requirements.jinja",
        variables={
            "analysis_focus": "security,gas,functionality",
            "include_patterns": "true"
        }
    ),
    storage_pattern="workflow/{workflow_id}/contract_analysis",
    storage_type=StorageType.KV,
    quality_threshold=0.85,
    allow_refinement=True,
    max_refinements=2,
    model_config=ModelParameters(
        temperature=0.6,
        max_tokens=2000
    ),
    domain="blockchain"
)

smart_contract_designer_phase = PhaseDefinition(
    phase_name="smart_contract_designer",
    atomic_nodes=[
        'dependency_check',
        'load_dependencies',
        'template_render',
        'llm_call',
        'schema_validation',
        'save_phase_output',
        'state_update',
        'quality_gate'
    ],
    input_schema=SmartContractDesignerInput,
    output_schema=SmartContractDesignerOutput,
    dependencies=["contract_analyzer"],
    templates=TemplateConfig(
        system_template="templates/system/contract_designer.jinja",
        user_template="templates/prompts/design_smart_contract.jinja",
        variables={
            "code_style": "secure_optimized",
            "include_comments": "true",
            "include_tests": "true"
        }
    ),
    storage_pattern="workflow/{workflow_id}/contract_design",
    storage_type=StorageType.FS,  # Store code in file system
    quality_threshold=0.9,
    allow_refinement=True,
    max_refinements=3,
    model_config=ModelParameters(
        temperature=0.5,
        max_tokens=4000
    ),
    domain="blockchain"
)

auditor_phase = PhaseDefinition(
    phase_name="auditor",
    atomic_nodes=[
        'dependency_check',
        'load_dependencies',
        'template_render',
        'llm_call',
        'schema_validation',
        'save_phase_output',
        'state_update',
        'quality_gate'
    ],
    input_schema=AuditorInput,
    output_schema=AuditorOutput,
    dependencies=["contract_analyzer", "smart_contract_designer"],
    templates=TemplateConfig(
        system_template="templates/system/contract_auditor.jinja",
        user_template="templates/prompts/audit_smart_contract.jinja",
        variables={
            "audit_depth": "comprehensive",
            "check_patterns": "all",
            "generate_report": "true"
        }
    ),
    storage_pattern="workflow/{workflow_id}/audit",
    storage_type=StorageType.KV,
    quality_threshold=0.95,  # High threshold for security
    allow_refinement=True,
    max_refinements=2,
    model_config=ModelParameters(
        temperature=0.3,  # Low temperature for accuracy
        max_tokens=3000
    ),
    domain="blockchain"
)

optimizer_phase = PhaseDefinition(
    phase_name="optimizer",
    atomic_nodes=[
        'dependency_check',
        'load_dependencies',
        'template_render',
        'llm_call',
        'schema_validation',
        'save_phase_output',
        'state_update',
        'quality_gate'
    ],
    input_schema=OptimizerInput,
    output_schema=OptimizerOutput,
    dependencies=["contract_analyzer", "smart_contract_designer", "auditor"],
    templates=TemplateConfig(
        system_template="templates/system/contract_optimizer.jinja",
        user_template="templates/prompts/optimize_contract.jinja",
        variables={
            "optimization_level": "aggressive",
            "preserve_readability": "true"
        }
    ),
    storage_pattern="workflow/{workflow_id}/optimized",
    storage_type=StorageType.FS,
    quality_threshold=0.85,
    allow_refinement=True,
    max_refinements=2,
    model_config=ModelParameters(
        temperature=0.4,
        max_tokens=3500
    ),
    domain="blockchain"
)


def register_blockchain_phases():
    """Register all blockchain domain phases."""
    register_phase("blockchain.contract_analyzer", contract_analyzer_phase)
    register_phase("blockchain.smart_contract_designer", smart_contract_designer_phase)
    register_phase("blockchain.auditor", auditor_phase)
    register_phase("blockchain.optimizer", optimizer_phase)


# Auto-register on import
register_blockchain_phases()