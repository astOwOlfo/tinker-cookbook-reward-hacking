"""Problem factory registry.

Each module in this package defines factory functions that produce
RubricDatapoint instances. All factories are registered here.

Factories come in two flavors:
  - Static: no rand_seed parameter, produce exactly one problem
  - Seedable: accept rand_seed, produce different problems per seed

The build_all_problems() function in dataset.py uses these registries.
"""

from .editorial import (
    make_editorial_headline_standfirst,
    make_editorial_opinion_argument,
    make_editorial_audience_adaptation,
    make_editorial_assembly,
    make_editorial_fact_check,
)
from .qa_report import (
    make_qa_escalation_email,
    make_qa_risk_assessment,
)
from .data_analysis import (
    make_data_analysis_report,
    make_utilization_report,
    make_sales_yoy_analysis,
)
from .code_tasks import (
    make_bash_golf,
    make_log_query,
    make_config_debugging,
    make_data_transformation,
    make_cron_scheduling,
    make_api_documentation,
)
from .incident import (
    make_incident_root_cause,
)
from .essay import (
    make_persuasive_essay,
)
from .gdpval_adapted import (
    make_tax_computation,
    make_financial_reconciliation,
    make_contract_clause_review,
    make_hr_investigation_summary,
    make_compliance_audit_report,
    make_project_risk_register,
)
from .verification import (
    make_scientific_claim_verification,
    make_statistical_report_review,
    make_resume_screening,
    make_survey_analysis,
    make_medical_triage_notes,
    make_accessibility_audit,
)
from .writing import (
    make_meeting_minutes,
    make_customer_complaint_response,
    make_competitive_comparison,
    make_press_release,
    make_literature_synthesis,
    make_budget_allocation,
)
from .professional import (
    make_performance_review_summary,
    make_event_planning,
    make_lesson_plan,
)
from .cli_tasks import (
    make_git_archaeology,
    make_json_pipeline,
    make_database_forensics,
)
from .procurement import (
    make_insurance_claim_adjudication,
    make_vendor_invoice_validation,
)
from .technical_review import (
    make_architecture_review,
    make_code_review_analysis,
    make_sla_compliance_audit,
)
from .scheduling_logistics import (
    make_shift_scheduling,
    make_supply_chain_optimization,
    make_route_planning,
)
from .forensic_analysis import (
    make_network_log_analysis,
    make_financial_fraud_detection,
    make_medical_chart_review,
)
from .quantitative_analysis import (
    make_portfolio_analysis,
    make_actuarial_analysis,
    make_statistical_experiment_analysis,
)
from .regulatory_compliance import (
    make_environmental_impact_assessment,
    make_import_classification,
    make_workplace_safety_audit,
)
from .project_management import (
    make_critical_path_analysis,
    make_earned_value_analysis,
    make_resource_leveling_analysis,
)
from .data_engineering import (
    make_etl_pipeline_validation,
    make_schema_migration_review,
    make_data_quality_audit,
)
from .crisis_communications import (
    make_crisis_response_audit,
    make_stakeholder_impact_assessment,
    make_communications_timeline_analysis,
)
from .scientific_research import (
    make_peer_review_analysis,
    make_experiment_protocol_audit,
    make_citation_network_analysis,
)
from .legal_analysis import (
    make_patent_prior_art,
    make_contract_risk_analysis,
    make_regulatory_filing_review,
)


# ============================================================================
# FACTORY REGISTRIES
# ============================================================================
# Static factories produce exactly one problem (no rand_seed).
# Seedable factories accept rand_seed and produce different problems per seed.

STATIC_FACTORIES = [
    make_persuasive_essay,
    make_editorial_headline_standfirst,
    make_editorial_opinion_argument,
    make_editorial_assembly,
    make_literature_synthesis,
    make_api_documentation,
]

SEEDABLE_FACTORIES = [
    # Existing (moved)
    make_data_analysis_report,
    make_utilization_report,
    make_incident_root_cause,
    make_sales_yoy_analysis,
    # Newly seedable (converted from static)
    make_qa_escalation_email,
    make_qa_risk_assessment,
    make_editorial_fact_check,
    make_editorial_audience_adaptation,
    make_bash_golf,
    # GDPval-inspired
    make_tax_computation,
    make_financial_reconciliation,
    make_contract_clause_review,
    make_hr_investigation_summary,
    make_compliance_audit_report,
    make_project_risk_register,
    # Verification
    make_scientific_claim_verification,
    make_statistical_report_review,
    make_resume_screening,
    make_survey_analysis,
    make_medical_triage_notes,
    make_accessibility_audit,
    # Writing
    make_meeting_minutes,
    make_customer_complaint_response,
    make_competitive_comparison,
    make_press_release,
    make_budget_allocation,
    # Professional
    make_performance_review_summary,
    make_event_planning,
    make_lesson_plan,
    # Technical
    make_log_query,
    make_config_debugging,
    make_data_transformation,
    make_cron_scheduling,
    # CLI-heavy
    make_git_archaeology,
    make_json_pipeline,
    make_database_forensics,
    # Procurement / claims
    make_insurance_claim_adjudication,
    make_vendor_invoice_validation,
    # Technical review
    make_architecture_review,
    make_code_review_analysis,
    make_sla_compliance_audit,
    # Scheduling / logistics
    make_shift_scheduling,
    make_supply_chain_optimization,
    make_route_planning,
    # Forensic analysis
    make_network_log_analysis,
    make_financial_fraud_detection,
    make_medical_chart_review,
    # Quantitative analysis
    make_portfolio_analysis,
    make_actuarial_analysis,
    make_statistical_experiment_analysis,
    # Regulatory compliance
    make_environmental_impact_assessment,
    make_import_classification,
    make_workplace_safety_audit,
    # Project management
    make_critical_path_analysis,
    make_earned_value_analysis,
    make_resource_leveling_analysis,
    # Data engineering
    make_etl_pipeline_validation,
    make_schema_migration_review,
    make_data_quality_audit,
    # Crisis communications
    make_crisis_response_audit,
    make_stakeholder_impact_assessment,
    make_communications_timeline_analysis,
    # Scientific research
    make_peer_review_analysis,
    make_experiment_protocol_audit,
    make_citation_network_analysis,
    # Legal analysis
    make_patent_prior_art,
    make_contract_risk_analysis,
    make_regulatory_filing_review,
]
