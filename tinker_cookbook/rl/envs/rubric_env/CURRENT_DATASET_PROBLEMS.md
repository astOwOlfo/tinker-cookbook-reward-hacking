# Current Dataset Problems

**Last updated:** 2026-02-23
**Total:** 6 static + 68 seedable = **74 factories** across 22 modules
**At `num_seeds=200`:** 6 + 68 x 200 = **13,606 problems**

---

## Full Factory Inventory

### essay.py (1 factory: 1 static)

| Factory | Type | Cats | Binary% | Pts | Files | Seedable |
|---------|------|------|---------|-----|-------|----------|
| `make_persuasive_essay` | essay | 5 | 0% | 15 | 0 | static |

### editorial.py (5 factories: 3 static, 2 seedable)

| Factory | Type | Cats | Binary% | Pts | Files | Seedable |
|---------|------|------|---------|-----|-------|----------|
| `make_editorial_headline_standfirst` | editorial | 16 | 75% | 30 | 0 | static |
| `make_editorial_opinion_argument` | editorial | 15 | 67% | 33 | 0 | static |
| `make_editorial_assembly` | editorial | 16 | 69% | 34 | 4 | static |
| `make_editorial_audience_adaptation` | editorial | 18 | 78% | 38 | 1 | seed |
| `make_editorial_fact_check` | editorial | 17-18 | 82-83% | 39-42 | 4 | seed |

### data_analysis.py (3 factories: all seedable)

| Factory | Type | Cats | Binary% | Pts | Files | Seedable |
|---------|------|------|---------|-----|-------|----------|
| `make_data_analysis_report` | data_analysis | 11 | 82% | 19 | 1 | seed |
| `make_utilization_report` | data_analysis | 13 | 77% | 30 | 3 | seed |
| `make_sales_yoy_analysis` | data_analysis | 12 | 75% | 31 | 3 | seed |

### code_tasks.py (6 factories: 1 static, 5 seedable)

| Factory | Type | Cats | Binary% | Pts | Files | Seedable |
|---------|------|------|---------|-----|-------|----------|
| `make_api_documentation` | api_documentation | 10 | 80% | 19 | 2 | static |
| `make_bash_golf` | bash_golf | 9-10 | 67-70% | 16-17 | 7 | seed |
| `make_log_query` | log_query | 8 | 75% | 16 | 2 | seed |
| `make_config_debugging` | config_debugging | 8-9 | 75-78% | 16-18 | 3 | seed |
| `make_data_transformation` | data_transformation | 10 | 80% | 18 | 2 | seed |
| `make_cron_scheduling` | cron_scheduling | 11-12 | 73-75% | 21-23 | 2 | seed |

### incident.py (1 factory: seedable)

| Factory | Type | Cats | Binary% | Pts | Files | Seedable |
|---------|------|------|---------|-----|-------|----------|
| `make_incident_root_cause` | incident_analysis | 11 | 82% | 25 | 3 | seed |

### qa_report.py (2 factories: all seedable)

| Factory | Type | Cats | Binary% | Pts | Files | Seedable |
|---------|------|------|---------|-----|-------|----------|
| `make_qa_escalation_email` | qa_report | 15 | 87% | 33 | 3 | seed |
| `make_qa_risk_assessment` | qa_report | 13 | 85% | 32 | 3 | seed |

### gdpval_adapted.py (6 factories: all seedable)

| Factory | Type | Cats | Binary% | Pts | Files | Seedable |
|---------|------|------|---------|-----|-------|----------|
| `make_tax_computation` | tax_computation | 17-18 | 94% | 30-31 | 3 | seed |
| `make_financial_reconciliation` | financial_reconciliation | 8-10 | 75-80% | 18-22 | 2 | seed |
| `make_contract_clause_review` | contract_review | 11-12 | 73-75% | 22-24 | 2 | seed |
| `make_hr_investigation_summary` | hr_investigation | 12 | 75% | 22 | 5 | seed |
| `make_compliance_audit_report` | compliance_audit | 18-27 | 100% | 29-41 | 7 | seed |
| `make_project_risk_register` | project_risk | 12-13 | 75-77% | 22-24 | 3 | seed |

### verification.py (6 factories: all seedable)

| Factory | Type | Cats | Binary% | Pts | Files | Seedable |
|---------|------|------|---------|-----|-------|----------|
| `make_scientific_claim_verification` | claim_verification | 11 | 73% | 22 | 2 | seed |
| `make_statistical_report_review` | statistical_review | 13 | 77% | 22 | 2 | seed |
| `make_resume_screening` | resume_screening | 23 | 100% | 39 | 6 | seed |
| `make_survey_analysis` | survey_analysis | 11 | 73% | 22 | 2 | seed |
| `make_medical_triage_notes` | medical_triage | 13 | 77% | 24 | 4 | seed |
| `make_accessibility_audit` | accessibility_audit | 12-14 | 75-79% | 23-26 | 2 | seed |

### writing.py (6 factories: 1 static, 5 seedable)

| Factory | Type | Cats | Binary% | Pts | Files | Seedable |
|---------|------|------|---------|-----|-------|----------|
| `make_literature_synthesis` | literature_synthesis | 10 | 80% | 20 | 5 | static |
| `make_meeting_minutes` | meeting_minutes | 12 | 75% | 21 | 1 | seed |
| `make_customer_complaint_response` | complaint_response | 11 | 73% | 20 | 3 | seed |
| `make_competitive_comparison` | competitive_comparison | 10 | 80% | 20 | 3 | seed |
| `make_press_release` | press_release | 11 | 73% | 20 | 3 | seed |
| `make_budget_allocation` | budget_allocation | 11 | 73% | 22 | 3 | seed |

### professional.py (3 factories: all seedable)

| Factory | Type | Cats | Binary% | Pts | Files | Seedable |
|---------|------|------|---------|-----|-------|----------|
| `make_performance_review_summary` | performance_review | 13 | 77% | 26 | 3 | seed |
| `make_event_planning` | event_planning | 12 | 75% | 25 | 3 | seed |
| `make_lesson_plan` | lesson_plan | 12 | 75% | 26 | 3 | seed |

### cli_tasks.py (3 factories: all seedable)

| Factory | Type | Cats | Binary% | Pts | Files | Seedable |
|---------|------|------|---------|-----|-------|----------|
| `make_git_archaeology` | git_archaeology | 10 | 90% | 22 | 2 | seed |
| `make_json_pipeline` | json_pipeline | 9-10 | 89-90% | 18-19 | 1 | seed |
| `make_database_forensics` | database_forensics | 11 | 91% | 23 | 1 | seed |

### procurement.py (2 factories: all seedable)

| Factory | Type | Cats | Binary% | Pts | Files | Seedable |
|---------|------|------|---------|-----|-------|----------|
| `make_vendor_invoice_validation` | vendor_invoice_validation | 18-20 | 94-95% | 31-35 | 4 | seed |
| `make_insurance_claim_adjudication` | insurance_claim_adjudication | 10-20 | 90-95% | 19-37 | 4 | seed |

### technical_review.py (3 factories: all seedable) -- NEW

| Factory | Type | Cats | Binary% | Pts | Files | Seedable |
|---------|------|------|---------|-----|-------|----------|
| `make_architecture_review` | architecture_review | 16-19 | 94-95% | 26-31 | 4 | seed |
| `make_code_review_analysis` | code_review_analysis | 18-22 | 94-95% | 27-33 | 4 | seed |
| `make_sla_compliance_audit` | sla_compliance_audit | 18-19 | 94-95% | 32-35 | 5 | seed |

### scheduling_logistics.py (3 factories: all seedable) -- NEW

| Factory | Type | Cats | Binary% | Pts | Files | Seedable |
|---------|------|------|---------|-----|-------|----------|
| `make_shift_scheduling` | shift_scheduling_audit | 19-21 | 95% | 29-33 | 5 | seed |
| `make_supply_chain_optimization` | supply_chain_optimization | 18-19 | 94-95% | 33-35 | 4 | seed |
| `make_route_planning` | route_planning | 20-21 | 95% | 31-33 | 4 | seed |

### forensic_analysis.py (3 factories: all seedable) -- NEW

| Factory | Type | Cats | Binary% | Pts | Files | Seedable |
|---------|------|------|---------|-----|-------|----------|
| `make_network_log_analysis` | network_log_analysis | 20 | 95% | 37 | 4 | seed |
| `make_financial_fraud_detection` | financial_fraud_detection | 20-25 | 95-96% | 34-46 | 4 | seed |
| `make_medical_chart_review` | medical_chart_review | 17-19 | 94-95% | 24-29 | 4 | seed |

### quantitative_analysis.py (3 factories: all seedable) -- NEW

| Factory | Type | Cats | Binary% | Pts | Files | Seedable |
|---------|------|------|---------|-----|-------|----------|
| `make_portfolio_analysis` | portfolio_analysis | 19 | 95% | 34 | 4 | seed |
| `make_actuarial_analysis` | actuarial_analysis | 17 | 94% | 30 | 4 | seed |
| `make_statistical_experiment_analysis` | statistical_experiment_analysis | 18 | 94% | 30 | 4 | seed |

### regulatory_compliance.py (3 factories: all seedable) -- NEW

| Factory | Type | Cats | Binary% | Pts | Files | Seedable |
|---------|------|------|---------|-----|-------|----------|
| `make_environmental_impact_assessment` | environmental_impact_assessment | 16-18 | 94% | 29-32 | 4 | seed |
| `make_import_classification` | import_classification | 17-20 | 94-95% | 32-38 | 4 | seed |
| `make_workplace_safety_audit` | workplace_safety_audit | 20-21 | 95% | 31-34 | 4 | seed |

### project_management.py (3 factories: all seedable) -- NEW

| Factory | Type | Cats | Binary% | Pts | Files | Seedable |
|---------|------|------|---------|-----|-------|----------|
| `make_critical_path_analysis` | critical_path_analysis | 16 | 94% | 28 | 4 | seed |
| `make_earned_value_analysis` | earned_value_analysis | 18 | 94% | 32 | 4 | seed |
| `make_resource_leveling_analysis` | resource_leveling_analysis | 15 | 93% | 26 | 4 | seed |

### data_engineering.py (3 factories: all seedable) -- NEW

| Factory | Type | Cats | Binary% | Pts | Files | Seedable |
|---------|------|------|---------|-----|-------|----------|
| `make_etl_pipeline_validation` | etl_pipeline_validation | 20-22 | 95% | 35-38 | 4 | seed |
| `make_schema_migration_review` | schema_migration_review | 18-21 | 95% | 32-36 | 4 | seed |
| `make_data_quality_audit` | data_quality_audit | 18-22 | 95% | 32-38 | 4 | seed |

### crisis_communications.py (3 factories: all seedable) -- NEW

| Factory | Type | Cats | Binary% | Pts | Files | Seedable |
|---------|------|------|---------|-----|-------|----------|
| `make_crisis_response_audit` | crisis_response_audit | 24 | 96% | 42 | 4 | seed |
| `make_stakeholder_impact_assessment` | stakeholder_impact_assessment | 23 | 96% | 40 | 4 | seed |
| `make_communications_timeline_analysis` | communications_timeline_analysis | 21 | 95% | 37 | 4 | seed |

### scientific_research.py (3 factories: all seedable) -- NEW

| Factory | Type | Cats | Binary% | Pts | Files | Seedable |
|---------|------|------|---------|-----|-------|----------|
| `make_peer_review_analysis` | peer_review_analysis | 14-15 | 93% | 23-30 | 4 | seed |
| `make_experiment_protocol_audit` | experiment_protocol_audit | 14-15 | 93% | 24-29 | 4 | seed |
| `make_citation_network_analysis` | citation_network_analysis | 17-22 | 94-95% | 35-47 | 4 | seed |

### legal_analysis.py (3 factories: all seedable) -- NEW

| Factory | Type | Cats | Binary% | Pts | Files | Seedable |
|---------|------|------|---------|-----|-------|----------|
| `make_patent_prior_art` | patent_prior_art | 22-28 | 95-96% | 42-54 | 4 | seed |
| `make_contract_risk_analysis` | contract_risk_analysis | 19-26 | 95-96% | 30-41 | 4 | seed |
| `make_regulatory_filing_review` | regulatory_filing_review | 17-22 | 94-95% | 29-39 | 4 | seed |

---

## Development Pipeline

### Make -> Criticize -> Fix -> Verify

Each batch of new factories goes through a 4-stage pipeline:

1. **Make**: Agent writes the factory module with 3 seedable factories, each following the "Factory Knows, Model Must Discover" pattern. Factories are verified across 50+ seeds for determinism, correct category counts, and 90%+ binary ratio.

2. **Criticize**: A critical-eye review agent reads the full source code and checks for:
   - Ground truth bugs (computed values that don't match source material)
   - Answer leakage (source files containing rubric answers)
   - Signal stripping failures (labels, categories, or quality tiers in raw data)
   - Collision/uniqueness issues (duplicate IDs, name collisions)
   - Low seedability (too few structural variants)
   - Rubric question ambiguity

3. **Fix**: A targeted fix agent addresses all HIGH-severity issues found in step 2. Fixes are surgical — only the specific bug is changed, preserving the rest of the factory logic.

4. **Verify**: Re-run the isolated module loader across all factories x 5 seeds, checking for regressions, duplicate category names, and correct output structure.

### Hardening Patterns Applied

| Pattern | Description | Used In |
|---------|-------------|---------|
| **Signal stripping** | Remove quality-telegraphing labels from source material | compliance_audit, resume_screening, fraud_detection |
| **Policy cross-reference** | Put lookup tables in policy docs, not in source data | compliance_audit, import_classification, workplace_safety |
| **Near-miss distractors** | Borderline-but-compliant items that test for false positives | compliance_audit, financial_fraud, medical_chart |
| **Behavioral evidence** | Show what entities DID, not what they ARE | resume_screening, performance_review |
| **Post-hoc audit** | After planting violations, re-audit final data for all actual issues | shift_scheduling |
| **Deferred computation** | Compute ground truth from final generated data, not planned values | financial_fraud (kickback, weekend, ghost amounts) |
| **Longest-match disambiguation** | When multiple rules could match, use most specific | import_classification (anti-dumping) |
| **Segment-first aggregation** | Generate detailed data first, derive aggregates by summing | statistical_experiment (Simpson's paradox) |

---

## Issue Tracker

### Resolved Issues (Phase 1: Original 39 factories)

| Severity | Total | Fixed |
|----------|-------|-------|
| HIGH     | 11    | 11    |
| MEDIUM   | 16    | 16    |
| LOW      | 7     | 5     |

All HIGH and MEDIUM issues fixed. See git history for details.

### Resolved Issues (Phase 2: 15 new factories, 5 new modules)

| Module | HIGH found | HIGH fixed | MEDIUM remaining |
|--------|-----------|-----------|-----------------|
| scheduling_logistics | 2 | 2 | route_planning greedy heuristic, only 5 violation types |
| forensic_analysis | 5 | 5 | vendor name collision (partially fixed), weekend transaction noise |
| technical_review | 4 | 4 | JS/Go dead code, probabilistic breach misfire |
| quantitative_analysis | 1 | 1 | (none remaining) |
| regulatory_compliance | 4 | 4 | zone type given directly, observation text telegraphs |
| **Total** | **16** | **16** | ~8 MEDIUM open |

### Open Systemic Issues

**S1. Some older factories are too easy** (~12 factories in essay, editorial, professional, writing modules). Tasks can be solved by reformatting visible answers. These predate the "Factory Knows, Model Must Discover" hardening pattern.

**S2. Binary category ratio varies widely.** Newer factories (2026-02-20) hit 94-100% binary. Older factories range 0-85%. The overall dataset average is ~83% binary.

**S3. Category count per factory varies widely.** Newer factories: 16-25 categories. Older factories: 5-15 categories. Target for Goodhart research is 20+ binary categories.

---

## Module Statistics

| Module | Factories | Avg Cats | Avg Binary% | Avg Pts | Era |
|--------|-----------|----------|-------------|---------|-----|
| legal_analysis | 3 | 22.3 | 95% | 40 | new |
| crisis_communications | 3 | 22.7 | 96% | 40 | new |
| forensic_analysis | 3 | 20.3 | 95% | 35 | new |
| data_engineering | 3 | 20.3 | 95% | 35 | new |
| regulatory_compliance | 3 | 18.7 | 95% | 32 | new |
| scheduling_logistics | 3 | 19.7 | 95% | 32 | new |
| technical_review | 3 | 18.7 | 95% | 31 | new |
| quantitative_analysis | 3 | 18.0 | 94% | 31 | new |
| scientific_research | 3 | 17.0 | 94% | 32 | new |
| project_management | 3 | 16.3 | 94% | 29 | new |
| procurement | 2 | 16.5 | 93% | 30 | new |
| gdpval_adapted | 6 | 14.2 | 83% | 28 | hardened |
| verification | 6 | 13.8 | 79% | 26 | hardened |
| cli_tasks | 3 | 10.0 | 90% | 21 | mid |
| qa_report | 2 | 14.0 | 86% | 33 | mid |
| data_analysis | 3 | 12.0 | 78% | 27 | original |
| code_tasks | 6 | 9.5 | 75% | 18 | original |
| writing | 6 | 10.8 | 76% | 20 | original |
| professional | 3 | 12.3 | 76% | 26 | original |
| editorial | 5 | 16.4 | 74% | 35 | original |
| incident | 1 | 11.0 | 82% | 25 | hardened |
| essay | 1 | 5.0 | 0% | 15 | original |
