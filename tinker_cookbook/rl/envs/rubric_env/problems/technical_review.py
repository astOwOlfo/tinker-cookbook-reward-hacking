"""Technical review tasks: architecture review, code review, and SLA compliance audit.

Each factory generates a realistic scenario with necessary_files containing all
information required to solve the problem. Seeds control randomization of system
types, planted flaws, code languages, SLA metrics, etc.
"""

from __future__ import annotations

import random as _random

from ..dataset import RubricDatapoint, RubricCategory, BinaryRubricCategory
from tinker_cookbook.rl.envs import tools
from ..content_pools import (
    random_name,
    random_names,
    pick1,
    COMPANY_NAMES,
)


# =============================================================================
# HELPERS
# =============================================================================


def _fmt_money(amount: float) -> str:
    """Format a float as a dollar string with commas."""
    return f"${amount:,.2f}"


# =============================================================================
# DOMAIN: ARCHITECTURE REVIEW
# =============================================================================

SYSTEM_TYPES = [
    {
        "key": "ecommerce",
        "label": "E-Commerce Platform",
        "description": "High-traffic online marketplace with product catalog, shopping cart, checkout, and order management.",
        "compliance_options": ["PCI-DSS", "SOC2", "GDPR"],
        "data_types": ["payment card data", "customer PII", "order history"],
    },
    {
        "key": "healthcare",
        "label": "Healthcare Patient Portal",
        "description": "Patient-facing portal for viewing medical records, scheduling appointments, and messaging providers.",
        "compliance_options": ["HIPAA", "SOC2", "HITECH"],
        "data_types": ["PHI", "medical records", "insurance information"],
    },
    {
        "key": "fintech",
        "label": "FinTech Payment Processing",
        "description": "Real-time payment processing platform handling bank transfers, card transactions, and ledger reconciliation.",
        "compliance_options": ["PCI-DSS", "SOC2", "PSD2"],
        "data_types": ["financial transactions", "bank account numbers", "KYC documents"],
    },
    {
        "key": "social_media",
        "label": "Social Media Platform",
        "description": "Content-sharing platform with user feeds, real-time notifications, media uploads, and content moderation.",
        "compliance_options": ["GDPR", "COPPA", "SOC2"],
        "data_types": ["user-generated content", "location data", "behavioral analytics"],
    },
    {
        "key": "iot",
        "label": "IoT Fleet Management",
        "description": "Platform ingesting telemetry from 100K+ IoT devices, with real-time alerting, historical analytics, and OTA firmware updates.",
        "compliance_options": ["SOC2", "GDPR", "ISO27001"],
        "data_types": ["device telemetry", "GPS coordinates", "firmware binaries"],
    },
]

# Each flaw has a key, a generator that produces the text planted in the architecture doc,
# and a description for rubric purposes.
ARCHITECTURE_FLAWS = [
    "spof",              # Single point of failure
    "sync_bottleneck",   # Synchronous call where async needed
    "unencrypted_internal",  # Unencrypted internal traffic
    "no_rate_limiting",  # No rate limiting on public API
    "missing_internal_auth",  # No auth between internal services
    "eventual_where_strong",  # Eventual consistency where strong is needed
    "strong_where_eventual",  # Strong consistency where eventual would suffice (perf hit)
    "pii_in_logs",       # PII logged in plaintext
    "no_audit_trail",    # No audit trail for sensitive operations
    "no_encryption_at_rest",  # Missing encryption at rest
    "db_capacity_mismatch",  # DB choice can't handle projected writes
    "no_horizontal_scaling",  # Stateful service that can't scale horizontally
]


def make_architecture_review(rand_seed: int = 42) -> RubricDatapoint:
    """Review a proposed system architecture for design flaws, scalability
    bottlenecks, and security vulnerabilities. Cross-reference requirements,
    architecture, and threat model.

    Seed varies: system type, which flaws are planted (4-6 from pool),
    SLA targets, compliance requirements, traffic projections.
    """
    rng = _random.Random(rand_seed)

    company = pick1(COMPANY_NAMES, rand_seed)
    architect_name = random_name(rand_seed + 1)
    reviewer_name = random_name(rand_seed + 2)

    # Pick system type
    system = rng.choice(SYSTEM_TYPES)
    compliance_reqs = rng.sample(system["compliance_options"], min(2, len(system["compliance_options"])))

    # SLA parameters
    uptime_sla = rng.choice([99.9, 99.95, 99.99])
    latency_p95_ms = rng.choice([100, 150, 200, 250])
    latency_p99_ms = latency_p95_ms + rng.choice([50, 100, 150])
    expected_rps = rng.choice([5000, 10000, 25000, 50000, 100000])
    peak_multiplier = rng.uniform(2.5, 5.0)
    peak_rps = int(expected_rps * peak_multiplier)
    data_retention_years = rng.choice([3, 5, 7, 10])

    # Capacity estimates
    avg_payload_kb = rng.choice([2, 5, 10, 20])
    daily_writes = expected_rps * 86400 * rng.uniform(0.1, 0.3)
    daily_writes = int(daily_writes)
    storage_growth_gb_day = round(daily_writes * avg_payload_kb / 1e6, 1)
    projected_storage_1yr_tb = round(storage_growth_gb_day * 365 / 1000, 1)
    bandwidth_gbps = round(expected_rps * avg_payload_kb / 1e6 * 8, 2)

    # Select 4-6 planted flaws
    n_flaws = rng.randint(4, 6)
    available_flaws = list(ARCHITECTURE_FLAWS)
    rng.shuffle(available_flaws)
    planted_flaws = available_flaws[:n_flaws]

    # Architecture components
    components = [
        {"name": "API Gateway", "tech": "NGINX / Kong", "instances": rng.choice([2, 3, 4])},
        {"name": "Auth Service", "tech": "Custom OAuth2 server", "instances": 1 if "spof" in planted_flaws else rng.choice([2, 3])},
        {"name": "Application Service", "tech": "Node.js / Express", "instances": rng.choice([3, 4, 6])},
        {"name": "Message Queue", "tech": rng.choice(["RabbitMQ", "Apache Kafka", "AWS SQS"]), "instances": rng.choice([2, 3])},
        {"name": "Primary Database", "tech": rng.choice(["PostgreSQL", "MySQL"]), "instances": rng.choice([1, 2])},
        {"name": "Cache Layer", "tech": "Redis Cluster", "instances": rng.choice([3, 6])},
        {"name": "CDN", "tech": rng.choice(["CloudFront", "Cloudflare", "Akamai"]), "instances": "edge"},
        {"name": "Search Service", "tech": "Elasticsearch", "instances": rng.choice([3, 5])},
        {"name": "Notification Service", "tech": "Custom (Python)", "instances": rng.choice([2, 3])},
        {"name": "Analytics Pipeline", "tech": "Apache Spark + S3", "instances": "cluster"},
    ]

    # Build component map for flaw details
    flaw_details: list[dict] = []

    for flaw_key in planted_flaws:
        if flaw_key == "spof":
            # Auth Service is single instance
            detail = {
                "key": "spof",
                "component": "Auth Service",
                "description_in_arch": "Auth Service: Single instance deployment for simplicity. Handles all authentication and token validation.",
                "ground_truth": "Auth Service is deployed as a single instance with no redundancy — if it goes down, all authenticated requests fail.",
                "rubric_component": "Auth Service",
            }
            flaw_details.append(detail)

        elif flaw_key == "sync_bottleneck":
            detail = {
                "key": "sync_bottleneck",
                "component": "Application Service",
                "description_in_arch": "Application Service calls the Search Service synchronously on every product listing request, waiting for the response before returning to the client.",
                "ground_truth": "Synchronous blocking call from Application Service to Search Service on every listing request creates a scalability bottleneck; should use async/non-blocking calls or caching.",
                "rubric_component": "Application Service → Search Service",
            }
            flaw_details.append(detail)

        elif flaw_key == "unencrypted_internal":
            detail = {
                "key": "unencrypted_internal",
                "component": "Internal network",
                "description_in_arch": "Internal service-to-service communication uses plain HTTP on the private VPC network for performance. TLS is terminated at the API Gateway.",
                "ground_truth": "Internal traffic between services is unencrypted (plain HTTP); a compromised host or network tap could intercept sensitive data including auth tokens.",
                "rubric_component": "internal service mesh",
            }
            flaw_details.append(detail)

        elif flaw_key == "no_rate_limiting":
            detail = {
                "key": "no_rate_limiting",
                "component": "API Gateway",
                "description_in_arch": "API Gateway handles routing and TLS termination. No rate limiting is configured as the team wants to avoid impacting legitimate users during peak traffic.",
                "ground_truth": "No rate limiting on the public API Gateway exposes the system to DDoS and brute-force attacks.",
                "rubric_component": "API Gateway",
            }
            flaw_details.append(detail)

        elif flaw_key == "missing_internal_auth":
            detail = {
                "key": "missing_internal_auth",
                "component": "Notification Service",
                "description_in_arch": "Notification Service accepts requests from any service on the internal network without authentication. The team considers the VPC boundary sufficient.",
                "ground_truth": "Notification Service has no internal authentication — a compromised service could send arbitrary notifications or exfiltrate contact data.",
                "rubric_component": "Notification Service",
            }
            flaw_details.append(detail)

        elif flaw_key == "eventual_where_strong":
            detail = {
                "key": "eventual_where_strong",
                "component": "Primary Database",
                "description_in_arch": "Primary Database uses asynchronous replication to a read replica. All reads, including balance checks and inventory verification, are directed to the read replica for load distribution.",
                "ground_truth": "Balance checks / inventory reads use an eventually-consistent read replica; stale reads can cause double-spending or overselling. These need strong consistency (read from primary).",
                "rubric_component": "Primary Database read replica",
            }
            flaw_details.append(detail)

        elif flaw_key == "strong_where_eventual":
            detail = {
                "key": "strong_where_eventual",
                "component": "Analytics Pipeline",
                "description_in_arch": "Analytics Pipeline writes daily aggregations using distributed two-phase commit across all nodes to ensure strict consistency of dashboard metrics.",
                "ground_truth": "Analytics/dashboard data uses expensive two-phase commit for strict consistency, but analytics queries are tolerant of slight staleness; eventual consistency would significantly improve throughput.",
                "rubric_component": "Analytics Pipeline",
            }
            flaw_details.append(detail)

        elif flaw_key == "pii_in_logs":
            detail = {
                "key": "pii_in_logs",
                "component": "Application Service",
                "description_in_arch": "Application Service logs all incoming request payloads at DEBUG level for troubleshooting, including user registration and profile update endpoints. Logs are shipped to a centralized ELK stack.",
                "ground_truth": f"Request payloads containing {', '.join(system['data_types'][:2])} are logged in plaintext, violating {compliance_reqs[0]} requirements.",
                "rubric_component": "Application Service logging",
            }
            flaw_details.append(detail)

        elif flaw_key == "no_audit_trail":
            detail = {
                "key": "no_audit_trail",
                "component": "Primary Database",
                "description_in_arch": "Primary Database supports direct admin access via SSH tunnel. Database schema changes and data exports are performed manually by the DBA team without automated logging.",
                "ground_truth": f"No audit trail for admin database operations (schema changes, data exports) — violates {compliance_reqs[0]} audit requirements.",
                "rubric_component": "database admin operations",
            }
            flaw_details.append(detail)

        elif flaw_key == "no_encryption_at_rest":
            detail = {
                "key": "no_encryption_at_rest",
                "component": "Primary Database",
                "description_in_arch": "Primary Database stores data on standard EBS volumes without encryption enabled. Backups are stored in an S3 bucket with default settings.",
                "ground_truth": f"Database storage and backups are not encrypted at rest — {', '.join(system['data_types'][:2])} are exposed if storage media is compromised. Required by {compliance_reqs[0]}.",
                "rubric_component": "database storage / backups",
            }
            flaw_details.append(detail)

        elif flaw_key == "db_capacity_mismatch":
            writes_per_sec = daily_writes // 86400
            projected_peak_writes = int(writes_per_sec * peak_multiplier)
            detail = {
                "key": "db_capacity_mismatch",
                "component": "Primary Database",
                "description_in_arch": f"Primary Database is a single PostgreSQL instance on an r5.xlarge (4 vCPU, 32GB RAM). Estimated write capacity: ~2,000 writes/sec. Projected peak write load from capacity estimates: {projected_peak_writes:,} writes/sec.",
                "ground_truth": f"Database write capacity (~2,000 w/s) is insufficient for projected peak load of {projected_peak_writes:,} writes/sec. Needs sharding, write-optimized DB, or architectural change.",
                "rubric_component": "Primary Database capacity",
                "projected_writes": projected_peak_writes,
            }
            flaw_details.append(detail)

        elif flaw_key == "no_horizontal_scaling":
            detail = {
                "key": "no_horizontal_scaling",
                "component": "Notification Service",
                "description_in_arch": "Notification Service maintains in-memory queues for pending notifications and uses local file storage for delivery status tracking. Scaling requires migrating the in-memory state.",
                "ground_truth": "Notification Service uses in-memory state and local storage, making horizontal scaling impractical. A single instance processes all notifications.",
                "rubric_component": "Notification Service",
            }
            flaw_details.append(detail)

    # Identify 2 components that have no flaws for false-positive checks
    flaw_components = {fd["component"] for fd in flaw_details}
    clean_components = [c for c in components if c["name"] not in flaw_components]
    rng.shuffle(clean_components)
    false_positive_components = clean_components[:min(2, len(clean_components))]

    # --- Build requirements.txt ---
    requirements_lines = [
        f"SYSTEM REQUIREMENTS — {system['label']}",
        f"",
        f"Prepared for: {company}",
        f"Architect: {architect_name}",
        f"Date: 2024-{rng.randint(1, 6):02d}-{rng.randint(1, 28):02d}",
        f"",
        "=" * 60,
        "BUSINESS REQUIREMENTS",
        "=" * 60,
        "",
        f"BR-1: The system must support {expected_rps:,} requests per second at steady state.",
        f"BR-2: The system must handle peak loads of {peak_rps:,} requests per second.",
        f"BR-3: {system['description']}",
        f"BR-4: Data must be retained for {data_retention_years} years per regulatory requirements.",
        f"BR-5: The system must support 24/7 operations with minimal planned downtime.",
        "",
        "=" * 60,
        "TECHNICAL REQUIREMENTS",
        "=" * 60,
        "",
        f"TR-1: Uptime SLA: {uptime_sla}%",
        f"TR-2: Response time P95 < {latency_p95_ms}ms, P99 < {latency_p99_ms}ms",
        f"TR-3: All {', '.join(system['data_types'])} must be encrypted at rest and in transit.",
        f"TR-4: The system must comply with: {', '.join(compliance_reqs)}",
        f"TR-5: All administrative and data-access operations must have a complete audit trail.",
        f"TR-6: No single component failure should cause system-wide outage (no single points of failure).",
        f"TR-7: Rate limiting must be in place to prevent abuse and DDoS attacks.",
        f"TR-8: All internal service communication must be authenticated and encrypted.",
        "",
        "=" * 60,
        "DATA CLASSIFICATION",
        "=" * 60,
        "",
    ]
    for i, dt in enumerate(system["data_types"], 1):
        requirements_lines.append(f"  DC-{i}: {dt} — Classification: SENSITIVE / RESTRICTED")
    requirements_lines.append("")
    requirements_content = "\n".join(requirements_lines) + "\n"

    # --- Build architecture.txt (with planted flaws) ---
    arch_lines = [
        f"PROPOSED SYSTEM ARCHITECTURE — {system['label']}",
        f"",
        f"Version: 1.0 (Draft)",
        f"Author: {architect_name}",
        f"",
        "=" * 60,
        "COMPONENT OVERVIEW",
        "=" * 60,
        "",
    ]

    # Map flaw descriptions to their components (list to handle multiple flaws per component)
    flaw_arch_text: dict[str, list[str]] = {}
    for fd in flaw_details:
        flaw_arch_text.setdefault(fd["component"], []).append(fd["description_in_arch"])

    for comp in components:
        arch_lines.append(f"--- {comp['name']} ---")
        arch_lines.append(f"Technology: {comp['tech']}")
        arch_lines.append(f"Instances: {comp['instances']}")

        # If this component has planted flaw(s), use the flaw description(s)
        if comp["name"] in flaw_arch_text:
            joined = "\n".join(flaw_arch_text[comp["name"]])
            arch_lines.append(f"Notes: {joined}")
        else:
            arch_lines.append(f"Notes: Standard deployment following best practices.")

        arch_lines.append("")

    # Add deployment topology section
    arch_lines.extend([
        "=" * 60,
        "DEPLOYMENT TOPOLOGY",
        "=" * 60,
        "",
        "Region: us-east-1 (primary), us-west-2 (DR)",
        "VPC: Private subnets for all services, public subnet for API Gateway and CDN.",
        "Load Balancing: Application Load Balancer in front of API Gateway instances.",
        "",
    ])

    # Add data flow section (may contain additional flaw text)
    arch_lines.extend([
        "=" * 60,
        "DATA FLOW",
        "=" * 60,
        "",
        "1. Client → CDN (static assets) / API Gateway (API calls)",
        "2. API Gateway → Auth Service (token validation)",
        "3. API Gateway → Application Service (business logic)",
    ])

    # Insert flaw-specific data flow entries
    for fd in flaw_details:
        if fd["key"] == "sync_bottleneck":
            arch_lines.append(f"4. Application Service → Search Service (synchronous REST call per request)")
        if fd["key"] == "unencrypted_internal":
            arch_lines.append(f"   Note: Internal calls use HTTP (no TLS) within VPC for reduced latency.")
        if fd["key"] == "eventual_where_strong":
            arch_lines.append(f"5. Application Service → Database Read Replica (all read queries)")

    arch_lines.extend(["", ""])
    architecture_content = "\n".join(arch_lines) + "\n"

    # --- Build threat_model.txt ---
    stride_threats = [
        ("Spoofing", "S-1", "Attacker impersonates a legitimate user by stealing session tokens.", "Auth Service validates tokens; tokens expire after 30 minutes."),
        ("Spoofing", "S-2", "Attacker impersonates an internal service to send malicious requests.", "ADDRESSED" if "missing_internal_auth" not in planted_flaws else "Mitigation: Pending assessment — relies on VPC boundary only."),
        ("Tampering", "T-1", "Attacker modifies data in transit between services.", "ADDRESSED" if "unencrypted_internal" not in planted_flaws else "Mitigation: Pending assessment — see architecture for transport details."),
        ("Tampering", "T-2", "Attacker modifies database records via direct admin access.", "ADDRESSED" if "no_audit_trail" not in planted_flaws else "Mitigation: Pending assessment — admin access exists, audit logging TBD."),
        ("Repudiation", "R-1", "User denies making a transaction; no audit trail available.", "ADDRESSED" if "no_audit_trail" not in planted_flaws else "Mitigation: Pending assessment — audit trail coverage under review."),
        ("Information Disclosure", "I-1", "Sensitive data exposed in application logs.", "ADDRESSED" if "pii_in_logs" not in planted_flaws else "Mitigation: Pending assessment — logging configuration under review."),
        ("Information Disclosure", "I-2", "Data at rest compromised through stolen storage media.", "ADDRESSED" if "no_encryption_at_rest" not in planted_flaws else "Mitigation: Pending assessment — encryption configuration under review."),
        ("Denial of Service", "D-1", "DDoS attack overwhelms the API Gateway.", "ADDRESSED" if "no_rate_limiting" not in planted_flaws else "Mitigation: Pending assessment — rate limiting configuration under review."),
        ("Denial of Service", "D-2", "Single component failure cascades to full outage.", "ADDRESSED" if "spof" not in planted_flaws else "Mitigation: Pending assessment — redundancy coverage under review."),
        ("Elevation of Privilege", "E-1", "Attacker gains admin access through compromised internal service.", "VPC network isolation provides defense in depth."),
    ]

    threat_lines = [
        f"THREAT MODEL — {system['label']} (STRIDE Analysis)",
        "",
        f"Prepared by: Security Team",
        f"Date: 2024-{rng.randint(1, 6):02d}-{rng.randint(1, 28):02d}",
        "",
        "=" * 70,
    ]

    for category, tid, threat_desc, mitigation in stride_threats:
        threat_lines.extend([
            "",
            f"[{tid}] Category: {category}",
            f"  Threat: {threat_desc}",
            f"  Mitigation Status: {mitigation}",
        ])

    threat_lines.extend(["", "=" * 70, ""])
    threat_model_content = "\n".join(threat_lines) + "\n"

    # --- Build capacity_estimates.txt ---
    # Include numbers that don't add up with architecture for db_capacity_mismatch
    writes_per_sec = daily_writes // 86400
    peak_writes_per_sec = int(writes_per_sec * peak_multiplier)

    capacity_lines = [
        f"CAPACITY ESTIMATES — {system['label']}",
        "",
        "=" * 60,
        "TRAFFIC PROJECTIONS",
        "=" * 60,
        "",
        f"Steady-state requests/sec: {expected_rps:,}",
        f"Peak requests/sec (estimated {peak_multiplier:.1f}x): {peak_rps:,}",
        f"Average payload size: {avg_payload_kb} KB",
        f"Estimated daily write operations: {daily_writes:,}",
        f"Peak write operations/sec: {peak_writes_per_sec:,}",
        "",
        "=" * 60,
        "STORAGE PROJECTIONS",
        "=" * 60,
        "",
        f"Daily storage growth: {storage_growth_gb_day} GB/day",
        f"Projected 1-year storage: {projected_storage_1yr_tb} TB",
        f"Data retention requirement: {data_retention_years} years",
        f"Total projected storage ({data_retention_years} yr): {round(projected_storage_1yr_tb * data_retention_years, 1)} TB",
        "",
        "=" * 60,
        "BANDWIDTH",
        "=" * 60,
        "",
        f"Estimated bandwidth (steady state): {bandwidth_gbps} Gbps",
        f"Estimated bandwidth (peak): {round(bandwidth_gbps * peak_multiplier, 2)} Gbps",
        "",
        "=" * 60,
        "DATABASE CAPACITY NOTES",
        "=" * 60,
        "",
        f"Selected database: PostgreSQL on r5.xlarge (4 vCPU, 32GB RAM)",
        f"Estimated max write throughput: ~2,000 writes/sec",
        f"Estimated max read throughput: ~8,000 reads/sec (with connection pooling)",
        f"Projected peak write load: {peak_writes_per_sec:,} writes/sec",
        "",
    ]
    capacity_content = "\n".join(capacity_lines) + "\n"

    # --- Problem statement ---
    problem_statement = f"""# Architecture Review — {system['label']}

You are {reviewer_name}, a senior solutions architect at {company}. You have been
asked to review a proposed system architecture for a new {system['label'].lower()}
before it goes to the Architecture Review Board.

## Source Files
- /testbed/data/requirements.txt — Business and technical requirements (SLA, compliance, capacity)
- /testbed/data/architecture.txt — Proposed architecture with components and deployment topology
- /testbed/data/threat_model.txt — STRIDE-based threat model with mitigation status
- /testbed/data/capacity_estimates.txt — Traffic projections, storage growth, bandwidth estimates

## Requirements
1. Cross-reference the architecture against every technical requirement
2. Identify all design flaws: single points of failure, scalability bottlenecks, security gaps, consistency issues, compliance gaps, and capacity mismatches
3. For each flaw, cite the specific requirement or threat it violates
4. Identify which STRIDE threats remain unmitigated in the architecture
5. Assess whether the database can handle the projected write load from capacity estimates
6. Recommend specific fixes for each identified issue

Write your architecture review report to /testbed/review_report.txt"""

    # --- Build rubric ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does the file /testbed/review_report.txt exist and contain substantive content (more than a few lines)?",
            points=1,
        ),
    ]

    # Per-flaw detection rubric items
    for fd in flaw_details:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"identifies_{fd['key']}",
                question=f"Does the review identify the following design flaw: {fd['ground_truth']}",
                points=2,
            )
        )
        rubric_items.append(
            BinaryRubricCategory(
                name=f"correct_component_{fd['key']}",
                question=(
                    f"Does the review correctly attribute the '{fd['key'].replace('_', ' ')}' flaw "
                    f"to the {fd['rubric_component']}?"
                ),
                points=1,
            )
        )

    # Requirement cross-reference checks
    rubric_items.append(
        BinaryRubricCategory(
            name="cites_uptime_sla",
            question=f"Does the review reference the {uptime_sla}% uptime SLA when discussing redundancy or SPOF issues?",
            points=1,
        )
    )
    rubric_items.append(
        BinaryRubricCategory(
            name="cites_compliance_req",
            question=f"Does the review reference {compliance_reqs[0]} compliance when discussing security or data protection gaps?",
            points=1,
        )
    )

    # Capacity mismatch specific check if planted
    if "db_capacity_mismatch" in planted_flaws:
        cap_detail = next(fd for fd in flaw_details if fd["key"] == "db_capacity_mismatch")
        rubric_items.append(
            BinaryRubricCategory(
                name="correct_capacity_numbers",
                question=(
                    f"Does the review cite specific numbers when discussing the database capacity mismatch — "
                    f"namely that the DB handles ~2,000 writes/sec but projected peak is "
                    f"{cap_detail['projected_writes']:,} writes/sec?"
                ),
                points=2,
            )
        )

    # False positive checks for clean components
    for clean_comp in false_positive_components:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"no_false_flag_{clean_comp['name'].lower().replace(' ', '_')}",
                question=(
                    f"Does the review correctly avoid flagging {clean_comp['name']} "
                    f"({clean_comp['tech']}) as having a design flaw? (It should NOT be "
                    f"identified as problematic.)"
                ),
                points=2,
            )
        )

    # Threat model cross-reference
    unmitigated_count = sum(1 for _, _, _, m in stride_threats if "Pending assessment" in m)
    rubric_items.append(
        BinaryRubricCategory(
            name="identifies_unmitigated_threats",
            question=f"Does the review identify that {unmitigated_count} STRIDE threats remain unmitigated or only partially mitigated?",
            points=2,
        )
    )

    # Recommend fixes
    rubric_items.append(
        BinaryRubricCategory(
            name="recommends_fixes",
            question="Does the review recommend at least one specific fix for each identified design flaw (not just generic advice)?",
            points=2,
        )
    )

    rubric_items.append(
        RubricCategory(
            name="analysis_thoroughness",
            description="How thorough and systematic is the architecture review?",
            failure="Superficial review; most flaws missed or poorly explained.",
            minor_failure="Some flaws found but analysis lacks cross-referencing between documents.",
            minor_success="Most flaws found with reasonable cross-referencing to requirements and threat model.",
            success="All flaws identified with detailed cross-referencing, specific requirement citations, and actionable fixes.",
            points=3,
        )
    )

    rubric = tuple(rubric_items)

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write your completed architecture review to /testbed/review_report.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/requirements.txt": requirements_content,
            "/testbed/data/architecture.txt": architecture_content,
            "/testbed/data/threat_model.txt": threat_model_content,
            "/testbed/data/capacity_estimates.txt": capacity_content,
        },
        problem_type="architecture_review",
    )


# =============================================================================
# DOMAIN: CODE REVIEW
# =============================================================================

# Issue pool: (key, language_agnostic_description, severity)
CODE_ISSUES = [
    {
        "key": "sql_injection",
        "severity": "CRITICAL",
        "issue_type": "Security vulnerability — SQL injection",
    },
    {
        "key": "unhandled_exception",
        "severity": "HIGH",
        "issue_type": "Error handling — unhandled exception on network call",
    },
    {
        "key": "race_condition",
        "severity": "HIGH",
        "issue_type": "Concurrency — race condition on shared mutable state",
    },
    {
        "key": "memory_leak",
        "severity": "HIGH",
        "issue_type": "Resource management — resource not closed/released",
    },
    {
        "key": "off_by_one",
        "severity": "MEDIUM",
        "issue_type": "Logic error — off-by-one in loop boundary",
    },
    {
        "key": "hardcoded_credential",
        "severity": "CRITICAL",
        "issue_type": "Security vulnerability — hardcoded API key",
    },
    {
        "key": "missing_input_validation",
        "severity": "HIGH",
        "issue_type": "Security vulnerability — missing input validation",
    },
    {
        "key": "deprecated_api",
        "severity": "MEDIUM",
        "issue_type": "Maintainability — usage of deprecated API",
    },
    {
        "key": "n_plus_one",
        "severity": "HIGH",
        "issue_type": "Performance — N+1 query pattern",
    },
]

LANGUAGE_CONFIGS = {
    "python": {
        "extension": ".py",
        "files": ["app/routes.py", "app/models.py", "app/utils.py"],
        "standards_lang": "Python",
    },
    "javascript": {
        "extension": ".js",
        "files": ["src/controllers/userController.js", "src/services/dataService.js", "src/middleware/auth.js"],
        "standards_lang": "JavaScript / Node.js",
    },
    "go": {
        "extension": ".go",
        "files": ["handlers/user.go", "services/data.go", "middleware/auth.go"],
        "standards_lang": "Go",
    },
}


def _generate_python_diff(rng: _random.Random, planted_issues: list[dict], clean_hunks: list[dict]) -> tuple[str, list[dict], list[dict]]:
    """Generate a Python-like unified diff with planted issues and clean changes."""
    diff_sections: list[str] = []
    issue_details: list[dict] = []
    clean_details: list[dict] = []

    files = ["app/routes.py", "app/models.py", "app/utils.py"]
    line_counter = {f: rng.randint(20, 60) for f in files}

    for issue in planted_issues:
        file_idx = rng.randint(0, len(files) - 1)
        filename = files[file_idx]
        start_line = line_counter[filename]
        line_counter[filename] += rng.randint(15, 30)

        if issue["key"] == "sql_injection":
            old_code = [
                "    def get_user(self, user_id):",
                "        query = f\"SELECT * FROM users WHERE id = {user_id}\"",
                "        return self.db.execute(query).fetchone()",
            ]
            new_code = [
                "    def get_user(self, user_id):",
                '        query = f"SELECT * FROM users WHERE id = {user_id} AND status = \'active\'"',
                "        result = self.db.execute(query).fetchone()",
                "        if result:",
                "            self.logger.info(f'Found user {user_id}')",
                "        return result",
            ]
            issue_line = start_line + 1
            issue_details.append({
                "file": filename,
                "line_start": issue_line,
                "line_end": issue_line + 2,
                "description": f"SQL injection vulnerability in {filename} around line {issue_line}: user_id is interpolated directly into the SQL query string using an f-string instead of parameterized queries.",
                **issue,
            })

        elif issue["key"] == "unhandled_exception":
            old_code = [
                "    def fetch_external_data(self, endpoint):",
                "        response = requests.get(endpoint)",
                "        return response.json()",
            ]
            new_code = [
                "    def fetch_external_data(self, endpoint, timeout=30):",
                "        response = requests.get(endpoint, timeout=timeout)",
                "        data = response.json()",
                "        self.cache.set(endpoint, data, ttl=300)",
                "        return data",
            ]
            issue_line = start_line + 1
            issue_details.append({
                "file": filename,
                "line_start": issue_line,
                "line_end": issue_line + 3,
                "description": f"Unhandled exception in {filename} around line {issue_line}: requests.get() and response.json() can raise ConnectionError, Timeout, or JSONDecodeError but are not wrapped in try/except.",
                **issue,
            })

        elif issue["key"] == "race_condition":
            old_code = [
                "class InventoryManager:",
                "    def __init__(self):",
                "        self.stock = {}",
                "",
                "    def decrement(self, item_id, qty):",
                "        if self.stock.get(item_id, 0) >= qty:",
                "            self.stock[item_id] -= qty",
                "            return True",
                "        return False",
            ]
            new_code = [
                "class InventoryManager:",
                "    def __init__(self):",
                "        self.stock = {}",
                "        self.last_updated = {}",
                "",
                "    def decrement(self, item_id, qty):",
                "        current = self.stock.get(item_id, 0)",
                "        if current >= qty:",
                "            self.stock[item_id] = current - qty",
                "            self.last_updated[item_id] = time.time()",
                "            return True",
                "        return False",
            ]
            issue_line = start_line + 5
            issue_details.append({
                "file": filename,
                "line_start": issue_line,
                "line_end": issue_line + 4,
                "description": f"Race condition in {filename} around line {issue_line}: stock check and decrement are not atomic — concurrent threads can read the same stock value and both proceed, causing overselling.",
                **issue,
            })

        elif issue["key"] == "memory_leak":
            old_code = [
                "    def process_file(self, path):",
                "        f = open(path, 'r')",
                "        data = f.read()",
                "        return self.parse(data)",
            ]
            new_code = [
                "    def process_file(self, path):",
                "        f = open(path, 'r')",
                "        data = f.read()",
                "        parsed = self.parse(data)",
                "        self.results.append(parsed)",
                "        return parsed",
            ]
            issue_line = start_line + 1
            issue_details.append({
                "file": filename,
                "line_start": issue_line,
                "line_end": issue_line + 3,
                "description": f"Resource leak in {filename} around line {issue_line}: file handle opened with open() is never closed. Should use 'with' statement or explicit f.close() in a finally block.",
                **issue,
            })

        elif issue["key"] == "off_by_one":
            size = rng.randint(5, 20)
            old_code = [
                f"    def paginate(self, items, page_size={size}):",
                "        pages = []",
                f"        for i in range(0, len(items) + 1, page_size):",
                "            pages.append(items[i:i+page_size])",
                "        return pages",
            ]
            new_code = [
                f"    def paginate(self, items, page_size={size}):",
                "        pages = []",
                f"        for i in range(0, len(items) + 1, page_size):",
                "            chunk = items[i:i+page_size]",
                "            if chunk:",
                "                pages.append(chunk)",
                "        return pages",
            ]
            issue_line = start_line + 2
            issue_details.append({
                "file": filename,
                "line_start": issue_line,
                "line_end": issue_line + 2,
                "description": f"Off-by-one error in {filename} around line {issue_line}: range uses len(items) + 1 which produces an extra empty page at the end when items length is a multiple of page_size.",
                **issue,
            })

        elif issue["key"] == "hardcoded_credential":
            fake_key = f"sk_live_{''.join(rng.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=24))}"
            old_code = [
                "    # Payment configuration",
                f'    STRIPE_API_KEY = "{fake_key}"',
                "    PAYMENT_TIMEOUT = 30",
            ]
            new_code = [
                "    # Payment configuration",
                f'    STRIPE_API_KEY = "{fake_key}"',
                "    PAYMENT_TIMEOUT = 30",
                "    RETRY_COUNT = 3",
                "    RETRY_DELAY = 2",
            ]
            issue_line = start_line + 1
            issue_details.append({
                "file": filename,
                "line_start": issue_line,
                "line_end": issue_line + 1,
                "description": f"Hardcoded API key in {filename} around line {issue_line}: Stripe live API key '{fake_key[:12]}...' is committed in source code. Must use environment variables or secrets manager.",
                **issue,
            })

        elif issue["key"] == "missing_input_validation":
            old_code = [
                "    @app.route('/api/users', methods=['POST'])",
                "    def create_user():",
                "        data = request.get_json()",
                "        user = User(name=data['name'], email=data['email'], role=data['role'])",
                "        db.session.add(user)",
                "        db.session.commit()",
            ]
            new_code = [
                "    @app.route('/api/users', methods=['POST'])",
                "    def create_user():",
                "        data = request.get_json()",
                "        user = User(",
                "            name=data['name'],",
                "            email=data['email'],",
                "            role=data['role'],",
                "            created_at=datetime.utcnow()",
                "        )",
                "        db.session.add(user)",
                "        db.session.commit()",
            ]
            issue_line = start_line + 2
            issue_details.append({
                "file": filename,
                "line_start": issue_line,
                "line_end": issue_line + 4,
                "description": f"Missing input validation in {filename} around line {issue_line}: user-supplied 'name', 'email', and 'role' from request JSON are used directly without validation. The 'role' field is especially dangerous as it could allow privilege escalation.",
                **issue,
            })

        elif issue["key"] == "deprecated_api":
            old_code = [
                "    import urllib2",
                "",
                "    def download(self, url):",
                "        response = urllib2.urlopen(url)",
                "        return response.read()",
            ]
            new_code = [
                "    import urllib2",
                "",
                "    def download(self, url):",
                "        response = urllib2.urlopen(url)",
                "        content = response.read()",
                "        self.logger.debug(f'Downloaded {len(content)} bytes from {url}')",
                "        return content",
            ]
            issue_line = start_line
            issue_details.append({
                "file": filename,
                "line_start": issue_line,
                "line_end": issue_line + 4,
                "description": f"Deprecated API in {filename} around line {issue_line}: urllib2 was removed in Python 3. Must use urllib.request or the requests library.",
                **issue,
            })

        elif issue["key"] == "n_plus_one":
            old_code = [
                "    def get_orders_with_items(self):",
                "        orders = Order.query.all()",
                "        result = []",
                "        for order in orders:",
                "            items = OrderItem.query.filter_by(order_id=order.id).all()",
                "            result.append({'order': order, 'items': items})",
                "        return result",
            ]
            new_code = [
                "    def get_orders_with_items(self, include_cancelled=False):",
                "        orders = Order.query.all()",
                "        result = []",
                "        for order in orders:",
                "            if not include_cancelled and order.status == 'cancelled':",
                "                continue",
                "            items = OrderItem.query.filter_by(order_id=order.id).all()",
                "            result.append({'order': order, 'items': items})",
                "        return result",
            ]
            issue_line = start_line + 4
            issue_details.append({
                "file": filename,
                "line_start": issue_line,
                "line_end": issue_line + 2,
                "description": f"N+1 query pattern in {filename} around line {issue_line}: for each order, a separate query fetches order items. Should use eager loading (joinedload/subqueryload) or a single JOIN query.",
                **issue,
            })

        # Build the unified diff hunk
        end_line = start_line + len(old_code)
        new_end = start_line + len(new_code)
        diff_sections.append(f"--- a/{filename}")
        diff_sections.append(f"+++ b/{filename}")
        diff_sections.append(f"@@ -{start_line},{len(old_code)} +{start_line},{len(new_code)} @@")
        for line in old_code:
            diff_sections.append(f"-{line}")
        for line in new_code:
            diff_sections.append(f"+{line}")
        diff_sections.append("")

    # Add clean hunks
    for hunk in clean_hunks:
        filename = hunk["file"]
        start_line = line_counter.get(filename, rng.randint(80, 120))
        line_counter[filename] = start_line + 20

        if hunk["type"] == "add_logging":
            old_code = [
                "    def process_request(self, request):",
                "        result = self.handler.handle(request)",
                "        return result",
            ]
            new_code = [
                "    def process_request(self, request):",
                "        self.logger.info(f'Processing request {request.id}')",
                "        result = self.handler.handle(request)",
                "        self.logger.info(f'Request {request.id} completed')",
                "        return result",
            ]
        elif hunk["type"] == "add_docstring":
            old_code = [
                "    def calculate_total(self, items):",
                "        return sum(item.price * item.quantity for item in items)",
            ]
            new_code = [
                "    def calculate_total(self, items):",
                '        """Calculate the total price for a list of items.',
                "",
                "        Args:",
                "            items: List of items with price and quantity attributes.",
                "",
                "        Returns:",
                '            Total price as a float."""',
                "        return sum(item.price * item.quantity for item in items)",
            ]
        else:  # refactor
            old_code = [
                "    def validate_email(self, email):",
                "        if '@' in email and '.' in email:",
                "            return True",
                "        return False",
            ]
            new_code = [
                "    def validate_email(self, email):",
                "        import re",
                "        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'",
                "        return bool(re.match(pattern, email))",
            ]

        end_line = start_line + len(old_code)
        diff_sections.append(f"--- a/{filename}")
        diff_sections.append(f"+++ b/{filename}")
        diff_sections.append(f"@@ -{start_line},{len(old_code)} +{start_line},{len(new_code)} @@")
        for line in old_code:
            diff_sections.append(f"-{line}")
        for line in new_code:
            diff_sections.append(f"+{line}")
        diff_sections.append("")

        clean_details.append({
            "file": filename,
            "type": hunk["type"],
            "line_start": start_line,
            "description": hunk["description"],
        })

    return "\n".join(diff_sections) + "\n", issue_details, clean_details


def make_code_review_analysis(rand_seed: int = 42) -> RubricDatapoint:
    """Review a simulated pull request diff and identify bugs, security
    vulnerabilities, performance issues, and style violations.

    Seed varies: language (Python-focused), which issues from pool (4-7),
    which hunks are clean (2-3), coding standards details.
    """
    rng = _random.Random(rand_seed)

    company = pick1(COMPANY_NAMES, rand_seed)
    author_name = random_name(rand_seed + 1)
    reviewer_name = random_name(rand_seed + 2)

    pr_number = rng.randint(100, 9999)
    branch_name = rng.choice([
        "feature/user-management", "feature/order-processing",
        "feature/payment-integration", "fix/performance-improvements",
        "feature/api-v2", "feature/inventory-system",
    ])

    # Select issues to plant
    n_issues = rng.randint(4, 7)
    available_issues = list(CODE_ISSUES)
    rng.shuffle(available_issues)
    planted_issues = available_issues[:n_issues]

    # Clean hunks
    clean_hunk_pool = [
        {"type": "add_logging", "file": "app/routes.py", "description": "Adding request logging is a reasonable improvement."},
        {"type": "add_docstring", "file": "app/models.py", "description": "Adding docstrings improves code documentation."},
        {"type": "refactor", "file": "app/utils.py", "description": "Improving email validation with regex is a valid refactor."},
    ]
    n_clean = rng.randint(2, 3)
    clean_hunks = rng.sample(clean_hunk_pool, n_clean)

    # Generate the diff
    diff_content, issue_details, clean_details = _generate_python_diff(
        rng, planted_issues, clean_hunks
    )

    # --- Build coding_standards.txt ---
    standards_lines = [
        f"{company} — Python Coding Standards",
        "",
        "=" * 60,
        "NAMING CONVENTIONS",
        "=" * 60,
        "",
        "  - Classes: PascalCase (e.g., UserManager)",
        "  - Functions/methods: snake_case (e.g., get_user_by_id)",
        "  - Constants: UPPER_SNAKE_CASE (e.g., MAX_RETRY_COUNT)",
        "  - Private methods: prefix with underscore (e.g., _validate_input)",
        "",
        "=" * 60,
        "ERROR HANDLING",
        "=" * 60,
        "",
        "  - All network calls (HTTP requests, database queries) MUST be wrapped",
        "    in try/except blocks with specific exception types.",
        "  - Never catch bare Exception unless re-raising.",
        "  - Log exceptions with full stack traces at ERROR level.",
        "  - Return meaningful error responses to callers.",
        "",
        "=" * 60,
        "SECURITY",
        "=" * 60,
        "",
        "  - NEVER hardcode credentials, API keys, or secrets in source code.",
        "    Use environment variables or a secrets manager.",
        "  - All database queries MUST use parameterized queries. String",
        "    interpolation in SQL is strictly prohibited.",
        "  - All user input MUST be validated before use in database queries,",
        "    file paths, or command execution.",
        "  - The 'role' field on user objects must be validated against an",
        "    allowlist of permitted roles.",
        "",
        "=" * 60,
        "RESOURCE MANAGEMENT",
        "=" * 60,
        "",
        "  - File handles, database connections, and network sockets MUST use",
        "    context managers (with statements) or be explicitly closed in",
        "    finally blocks.",
        "  - Use connection pools for database and HTTP client connections.",
        "",
        "=" * 60,
        "CONCURRENCY",
        "=" * 60,
        "",
        "  - Shared mutable state MUST be protected with locks or use",
        "    atomic operations.",
        "  - Prefer database-level locking (SELECT FOR UPDATE) for",
        "    inventory/balance operations.",
        "",
        "=" * 60,
        "PERFORMANCE",
        "=" * 60,
        "",
        "  - Avoid N+1 query patterns. Use eager loading (joinedload) or",
        "    batch queries when loading related objects.",
        "  - Database queries inside loops must be reviewed for optimization.",
        "",
        "=" * 60,
        "LOGGING",
        "=" * 60,
        "",
        "  - Use structured logging (logger.info, logger.error) not print().",
        "  - NEVER log sensitive data (passwords, tokens, PII).",
        "  - Log at appropriate levels: DEBUG for dev, INFO for operations,",
        "    WARNING for recoverable issues, ERROR for failures.",
        "",
        "=" * 60,
        "DEPENDENCIES",
        "=" * 60,
        "",
        "  - Do not use deprecated libraries. Known deprecated:",
        "    * urllib2 (removed in Python 3, use urllib.request or requests)",
        "    * optparse (use argparse)",
        "    * imp (use importlib)",
        "  - Pin dependency versions in requirements.txt.",
        "",
    ]
    standards_content = "\n".join(standards_lines) + "\n"

    # --- Build api_docs.txt ---
    api_docs_lines = [
        "API DOCUMENTATION — Libraries Used in This Codebase",
        "",
        "=" * 60,
        "requests library (v2.31+)",
        "=" * 60,
        "",
        "  requests.get(url, timeout=None, **kwargs)",
        "    - Raises requests.ConnectionError on network failure",
        "    - Raises requests.Timeout if timeout is specified and exceeded",
        "    - Raises requests.HTTPError on 4xx/5xx if raise_for_status() is called",
        "",
        "  response.json()",
        "    - Raises json.JSONDecodeError if response body is not valid JSON",
        "",
        "=" * 60,
        "SQLAlchemy ORM (v2.0+)",
        "=" * 60,
        "",
        "  Query.all() — Returns list of results, executes one query.",
        "  Query.filter_by(**kwargs) — Adds WHERE clause, lazy evaluation.",
        "  session.execute(text(sql), params) — Parameterized query execution.",
        "    WARNING: NEVER use f-strings or .format() for SQL construction.",
        "    Use: session.execute(text('SELECT * FROM users WHERE id = :id'), {'id': user_id})",
        "",
        "  Eager loading (avoids N+1 queries):",
        "    from sqlalchemy.orm import joinedload",
        "    Order.query.options(joinedload(Order.items)).all()",
        "",
        "=" * 60,
        "Flask (v3.0+)",
        "=" * 60,
        "",
        "  request.get_json() — Returns parsed JSON body or None.",
        "    Does NOT validate schema or field types.",
        "    Callers must validate all fields before use.",
        "",
        "=" * 60,
        "DEPRECATED MODULES (Python 3)",
        "=" * 60,
        "",
        "  urllib2 — REMOVED in Python 3. Use urllib.request or 'requests'.",
        "  cgi — Deprecated since 3.8. Use email.message or multipart.",
        "  imp — Deprecated since 3.4. Use importlib.",
        "",
    ]
    api_docs_content = "\n".join(api_docs_lines) + "\n"

    # --- Build security_checklist.txt ---
    checklist_lines = [
        "SECURITY REVIEW CHECKLIST (OWASP-based)",
        "",
        "When reviewing code changes, verify the following:",
        "",
        "=" * 60,
        "A1 — INJECTION",
        "=" * 60,
        "[ ] All SQL queries use parameterized statements",
        "[ ] No string concatenation or f-strings in SQL",
        "[ ] Command-line arguments are properly escaped",
        "",
        "=" * 60,
        "A2 — BROKEN AUTHENTICATION",
        "=" * 60,
        "[ ] Credentials not hardcoded in source",
        "[ ] Session tokens have appropriate expiry",
        "[ ] API keys stored in environment or secrets manager",
        "",
        "=" * 60,
        "A3 — SENSITIVE DATA EXPOSURE",
        "=" * 60,
        "[ ] No PII or secrets in log output",
        "[ ] Sensitive data encrypted in transit and at rest",
        "[ ] API responses don't leak internal details",
        "",
        "=" * 60,
        "A5 — BROKEN ACCESS CONTROL",
        "=" * 60,
        "[ ] Input validation on all user-supplied data",
        "[ ] Role and permission checks on sensitive endpoints",
        "[ ] No mass-assignment vulnerabilities",
        "",
        "=" * 60,
        "A6 — SECURITY MISCONFIGURATION",
        "=" * 60,
        "[ ] Debug mode disabled in production",
        "[ ] Error messages don't reveal stack traces to users",
        "[ ] Default credentials changed",
        "",
        "=" * 60,
        "A9 — USING COMPONENTS WITH KNOWN VULNERABILITIES",
        "=" * 60,
        "[ ] No deprecated libraries used",
        "[ ] Dependencies pinned and reviewed for CVEs",
        "",
    ]
    checklist_content = "\n".join(checklist_lines) + "\n"

    # --- Problem statement ---
    problem_statement = f"""# Code Review — Pull Request #{pr_number}

You are {reviewer_name}, a senior developer at {company}. You have been asked to
review a pull request from {author_name} on branch `{branch_name}`.

## Source Files
- /testbed/data/pull_request.diff — Unified diff of changes across 2-3 files
- /testbed/data/coding_standards.txt — Team coding standards and conventions
- /testbed/data/api_docs.txt — API documentation for libraries used in the code
- /testbed/data/security_checklist.txt — OWASP-based security review checklist

## Requirements
1. Review every hunk in the diff carefully
2. Identify ALL bugs, security vulnerabilities, performance issues, and standards violations
3. For each issue found, specify: the file name, approximate line range, issue type, severity (CRITICAL/HIGH/MEDIUM/LOW), and a clear explanation
4. Reference the relevant coding standard section or OWASP category
5. For security issues, suggest a specific fix
6. Identify which changes are clean and acceptable
7. Provide an overall APPROVE / REQUEST CHANGES / REJECT recommendation

Write your code review to /testbed/code_review.txt"""

    # --- Build rubric ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does the file /testbed/code_review.txt exist and contain substantive content?",
            points=1,
        ),
    ]

    # Per-issue detection
    for i, issue in enumerate(issue_details):
        rubric_items.append(
            BinaryRubricCategory(
                name=f"found_issue_{i+1}",
                question=f"Does the code review identify the following issue: {issue['description']}",
                points=2,
            )
        )
        rubric_items.append(
            BinaryRubricCategory(
                name=f"correct_type_{i+1}",
                question=(
                    f"Does the code review correctly categorize issue #{i+1} in {issue['file']} "
                    f"as '{issue['issue_type']}'? (Need not use these exact words but must "
                    f"convey the same category of problem.)"
                ),
                points=1,
            )
        )
        rubric_items.append(
            BinaryRubricCategory(
                name=f"correct_severity_{i+1}",
                question=(
                    f"Does the code review assign the correct severity for issue #{i+1}? "
                    f"The correct severity is {issue['severity']}."
                ),
                points=1,
            )
        )

    # False positive checks for clean hunks
    for clean in clean_details:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"no_false_positive_{clean['type']}",
                question=(
                    f"Does the code review correctly treat the {clean['type'].replace('_', ' ')} "
                    f"change in {clean['file']} as acceptable? ({clean['description']}) "
                    f"It should NOT be flagged as a bug or vulnerability."
                ),
                points=2,
            )
        )

    # Check for deprecated API identification if planted
    deprecated_issues = [i for i in issue_details if i["key"] == "deprecated_api"]
    if deprecated_issues:
        rubric_items.append(
            BinaryRubricCategory(
                name="identifies_deprecated_api",
                question="Does the code review specifically identify the deprecated library/API and name the modern replacement?",
                points=1,
            )
        )

    # Check for security fix suggestions
    security_issues = [i for i in issue_details if i["severity"] == "CRITICAL"]
    if security_issues:
        rubric_items.append(
            BinaryRubricCategory(
                name="suggests_fix_for_security",
                question=(
                    f"Does the code review suggest a specific code fix for at least one of the "
                    f"CRITICAL security issues (not just 'fix this' but actual code or approach)?"
                ),
                points=2,
            )
        )

    # Overall recommendation
    rubric_items.append(
        BinaryRubricCategory(
            name="correct_recommendation",
            question="Does the code review give a REQUEST CHANGES or REJECT recommendation (given the CRITICAL/HIGH severity issues present)?",
            points=2,
        )
    )

    rubric_items.append(
        BinaryRubricCategory(
            name="references_standards",
            question="Does the code review reference the coding standards document or OWASP checklist when explaining at least 2 issues?",
            points=1,
        )
    )

    rubric_items.append(
        RubricCategory(
            name="review_quality",
            description="How thorough and helpful is the code review overall?",
            failure="Superficial review; most issues missed or poorly explained.",
            minor_failure="Some issues found but explanations lack detail or miss context from reference docs.",
            minor_success="Most issues found with clear explanations and references to standards.",
            success="All issues identified with detailed explanations, specific fix suggestions, and clear cross-references to coding standards and OWASP checklist.",
            points=3,
        )
    )

    rubric = tuple(rubric_items)

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write your completed code review to /testbed/code_review.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/pull_request.diff": diff_content,
            "/testbed/data/coding_standards.txt": standards_content,
            "/testbed/data/api_docs.txt": api_docs_content,
            "/testbed/data/security_checklist.txt": checklist_content,
        },
        problem_type="code_review_analysis",
    )


# =============================================================================
# DOMAIN: SLA COMPLIANCE AUDIT
# =============================================================================

SLA_SERVICE_NAMES = [
    "CloudSync Pro", "DataVault Enterprise", "NetStream Platform",
    "InfraCore Services", "AppForge Cloud", "SecureEdge Gateway",
    "AnalyticsBridge", "DevOps Pipeline Hub", "StorageMatrix",
]

INCIDENT_ROOT_CAUSES = [
    "Database failover triggered by disk space exhaustion",
    "Network switch firmware bug causing packet loss",
    "Memory leak in application server",
    "DNS resolution failure due to misconfigured CNAME",
    "Certificate expiration on load balancer",
    "Disk I/O saturation on primary database",
    "Configuration deployment error (wrong config pushed to production)",
    "Third-party API rate limiting triggered",
    "Container orchestration scheduling failure",
    "Cascading timeout from upstream dependency",
]

SUPPORT_CATEGORIES = [
    "Service Outage", "Performance Degradation", "Feature Request",
    "Configuration Change", "Security Incident", "Billing Inquiry",
    "Integration Issue", "Data Recovery",
]


def make_sla_compliance_audit(rand_seed: int = 42) -> RubricDatapoint:
    """Given service monitoring data and an SLA agreement, determine which
    SLA targets were met/breached over a 30-day reporting period.

    Seed varies: which metrics are breached (2-4 out of 5), severity of
    breaches, edge cases (maintenance window exclusion, severity-1 ticket SLA).
    """
    rng = _random.Random(rand_seed)

    company = pick1(COMPANY_NAMES, rand_seed)
    provider_name = rng.choice(SLA_SERVICE_NAMES)
    auditor_name = random_name(rand_seed + 1)

    # --- SLA parameters ---
    uptime_target_pct = rng.choice([99.9, 99.95, 99.5])
    p95_target_ms = rng.choice([150, 200, 250, 300])
    error_rate_target_pct = rng.choice([0.1, 0.5, 1.0])
    support_response_sev1_hrs = rng.choice([1, 2, 4])
    support_response_sev2_hrs = rng.choice([4, 8])
    support_response_sev3_hrs = rng.choice([24, 48])
    maintenance_window_day = "Saturday"
    maintenance_window_start = "02:00"
    maintenance_window_end = "04:00"

    # Financial penalty structure
    credit_per_breach_pct = rng.choice([5, 10, 15])
    max_credit_pct = rng.choice([25, 30, 50])
    monthly_fee = rng.choice([25000, 50000, 75000, 100000])

    # --- Decide which SLA metrics to breach ---
    all_metrics = ["uptime", "p95_response", "error_rate", "support_response", "maintenance_window"]
    n_breaches = rng.randint(2, 4)
    rng.shuffle(all_metrics)
    breached_metrics = set(all_metrics[:n_breaches])
    passing_metrics = set(all_metrics) - breached_metrics

    # --- Generate 30 days of data ---
    # Month: pick a month
    report_month = rng.randint(1, 12)
    report_year = 2024

    # Generate uptime data
    uptime_data: list[dict] = []
    incident_data: list[dict] = []
    incident_counter = 0
    total_downtime_minutes = 0
    maintenance_downtime_minutes = 0

    for day in range(1, 31):
        # Determine day of week (approximate)
        # Jan 1, 2024 is Monday. We use a simple offset.
        import datetime as _dt
        try:
            date_obj = _dt.date(report_year, report_month, day)
        except ValueError:
            # Month doesn't have this many days
            continue
        day_name = date_obj.strftime("%A")
        date_str = date_obj.strftime("%Y-%m-%d")
        is_saturday = day_name == "Saturday"

        # Base uptime: most days are fine
        if "uptime" in breached_metrics and rng.random() < 0.15:
            # Incident day
            incident_counter += 1
            inc_id = f"INC-{1000 + incident_counter}"
            downtime_min = rng.randint(10, 120)
            is_in_maintenance = is_saturday and rng.random() < 0.3

            if is_in_maintenance:
                maintenance_downtime_minutes += downtime_min
                was_maintenance = True
                start_hour = 2
                start_min = rng.randint(0, 59)
            else:
                total_downtime_minutes += downtime_min
                was_maintenance = False
                start_hour = rng.randint(0, 23)
                start_min = rng.randint(0, 59)

            end_min_total = start_hour * 60 + start_min + downtime_min
            end_hour = (end_min_total // 60) % 24
            end_min = end_min_total % 60

            daily_uptime_pct = round(100.0 * (1440 - downtime_min) / 1440, 3)
            uptime_data.append({
                "date": date_str,
                "uptime_pct": daily_uptime_pct,
                "incident_id": inc_id,
            })

            root_cause = rng.choice(INCIDENT_ROOT_CAUSES)
            impact = rng.choice(["Full service outage", "Partial degradation", "Intermittent errors", "Single region affected"])

            incident_data.append({
                "id": inc_id,
                "date": date_str,
                "start_time": f"{start_hour:02d}:{start_min:02d}",
                "end_time": f"{end_hour:02d}:{end_min:02d}",
                "duration_minutes": downtime_min,
                "root_cause": root_cause,
                "customer_impact": impact,
                "was_maintenance_window": was_maintenance,
            })
        elif "uptime" not in breached_metrics and rng.random() < 0.02:
            # Very rare incident even when passing — minimal downtime
            incident_counter += 1
            inc_id = f"INC-{1000 + incident_counter}"
            downtime_min = rng.randint(1, 5)
            total_downtime_minutes += downtime_min
            start_hour = rng.randint(0, 23)
            start_min = rng.randint(0, 59)
            end_min_total = start_hour * 60 + start_min + downtime_min
            end_hour = (end_min_total // 60) % 24
            end_min = end_min_total % 60

            daily_uptime_pct = round(100.0 * (1440 - downtime_min) / 1440, 3)
            uptime_data.append({
                "date": date_str,
                "uptime_pct": daily_uptime_pct,
                "incident_id": inc_id,
            })
            incident_data.append({
                "id": inc_id,
                "date": date_str,
                "start_time": f"{start_hour:02d}:{start_min:02d}",
                "end_time": f"{end_hour:02d}:{end_min:02d}",
                "duration_minutes": downtime_min,
                "root_cause": rng.choice(INCIDENT_ROOT_CAUSES),
                "customer_impact": "Brief interruption",
                "was_maintenance_window": False,
            })
        else:
            uptime_data.append({
                "date": date_str,
                "uptime_pct": 100.0,
                "incident_id": "",
            })

    # Also plant a maintenance window violation if "maintenance_window" is breached
    if "maintenance_window" in breached_metrics:
        # Find a non-Saturday incident or create one
        violation_day = rng.randint(1, 28)
        try:
            viol_date_obj = _dt.date(report_year, report_month, violation_day)
        except ValueError:
            viol_date_obj = _dt.date(report_year, report_month, 15)
        viol_day_name = viol_date_obj.strftime("%A")
        viol_date_str = viol_date_obj.strftime("%Y-%m-%d")

        # If it happens to be Saturday, shift by one
        if viol_day_name == "Saturday":
            violation_day = min(28, violation_day + 1)
            viol_date_obj = _dt.date(report_year, report_month, violation_day)
            viol_day_name = viol_date_obj.strftime("%A")
            viol_date_str = viol_date_obj.strftime("%Y-%m-%d")

        incident_counter += 1
        maint_inc_id = f"INC-{1000 + incident_counter}"
        maint_downtime = rng.randint(30, 90)
        total_downtime_minutes += maint_downtime

        incident_data.append({
            "id": maint_inc_id,
            "date": viol_date_str,
            "start_time": "03:00",
            "end_time": f"{3 + maint_downtime // 60:02d}:{maint_downtime % 60:02d}",
            "duration_minutes": maint_downtime,
            "root_cause": "Planned maintenance — database upgrade",
            "customer_impact": "Full service outage during maintenance",
            "was_maintenance_window": True,  # Marked as maintenance but NOT Saturday
        })

        # Update the matching day in uptime_data to reflect this new incident
        for entry in uptime_data:
            if entry["date"] == viol_date_str:
                # Recompute uptime for this day accounting for the maintenance downtime
                existing_downtime = round((1.0 - entry["uptime_pct"] / 100.0) * 1440)
                new_total_downtime = existing_downtime + maint_downtime
                entry["uptime_pct"] = round(100.0 * (1440 - new_total_downtime) / 1440, 3)
                entry["incident_id"] = (entry["incident_id"] + "," + maint_inc_id).lstrip(",")
                break

    # Compute actual uptime
    n_days = len(uptime_data)
    total_minutes = n_days * 1440
    # Downtime excluding legitimate maintenance windows (Saturday 2-4am)
    non_maintenance_downtime = total_downtime_minutes
    actual_uptime_pct = round(100.0 * (total_minutes - non_maintenance_downtime) / total_minutes, 4)
    uptime_pass = actual_uptime_pct >= uptime_target_pct

    # Generate performance metrics
    perf_data: list[dict] = []
    daily_p95_values: list[float] = []
    daily_error_rates: list[float] = []

    for entry in uptime_data:
        base_p50 = rng.uniform(30, 80)
        if "p95_response" in breached_metrics and rng.random() < 0.25:
            base_p95 = rng.uniform(p95_target_ms + 20, p95_target_ms + 150)
        else:
            base_p95 = rng.uniform(p95_target_ms * 0.5, p95_target_ms * 0.95)

        p99 = round(base_p95 * rng.uniform(1.2, 1.8), 1)
        p95 = round(base_p95, 1)
        p50 = round(base_p50, 1)

        if "error_rate" in breached_metrics and rng.random() < 0.2:
            error_rate = round(rng.uniform(error_rate_target_pct + 0.1, error_rate_target_pct + 2.0), 3)
        else:
            error_rate = round(rng.uniform(0.001, error_rate_target_pct * 0.8), 3)

        daily_p95_values.append(p95)
        daily_error_rates.append(error_rate)

        perf_data.append({
            "date": entry["date"],
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "error_rate_pct": error_rate,
        })

    # Compute actual period p95 and error rate
    sorted_p95 = sorted(daily_p95_values)
    actual_p95 = round(sorted_p95[int(len(sorted_p95) * 0.95) - 1] if sorted_p95 else 0.0, 1)
    actual_avg_p95 = round(sum(daily_p95_values) / len(daily_p95_values), 1)
    actual_error_rate = round(sum(daily_error_rates) / len(daily_error_rates), 4)
    p95_pass = actual_avg_p95 <= p95_target_ms
    error_rate_pass = actual_error_rate <= error_rate_target_pct

    # Generate support tickets
    support_tickets: list[dict] = []
    n_tickets = rng.randint(15, 30)
    sev1_response_times: list[float] = []
    sev2_response_times: list[float] = []
    sev3_response_times: list[float] = []

    for t in range(n_tickets):
        severity = rng.choices([1, 2, 3], weights=[15, 35, 50])[0]
        category = rng.choice(SUPPORT_CATEGORIES)
        open_day = rng.randint(1, n_days)
        try:
            open_date = _dt.date(report_year, report_month, open_day)
        except ValueError:
            open_date = _dt.date(report_year, report_month, min(open_day, 28))
        open_hour = rng.randint(8, 18)
        open_min = rng.randint(0, 59)

        if severity == 1:
            sla_hrs = support_response_sev1_hrs
            if "support_response" in breached_metrics and rng.random() < 0.35:
                response_minutes = rng.randint(sla_hrs * 60 + 15, sla_hrs * 60 + 180)
            else:
                response_minutes = rng.randint(10, sla_hrs * 60 - 5)
            sev1_response_times.append(response_minutes / 60.0)
        elif severity == 2:
            sla_hrs = support_response_sev2_hrs
            if "support_response" in breached_metrics and rng.random() < 0.2:
                response_minutes = rng.randint(sla_hrs * 60 + 30, sla_hrs * 60 + 300)
            else:
                response_minutes = rng.randint(15, sla_hrs * 60 - 10)
            sev2_response_times.append(response_minutes / 60.0)
        else:
            sla_hrs = support_response_sev3_hrs
            response_minutes = rng.randint(30, sla_hrs * 60 - 30)
            sev3_response_times.append(response_minutes / 60.0)

        resp_hour = open_hour + response_minutes // 60
        resp_min = open_min + response_minutes % 60
        if resp_min >= 60:
            resp_hour += 1
            resp_min -= 60
        resp_day_offset = resp_hour // 24
        resp_hour = resp_hour % 24

        try:
            resp_date = open_date + _dt.timedelta(days=resp_day_offset)
        except (ValueError, OverflowError):
            resp_date = open_date

        resolve_hours = rng.randint(1, 72)
        try:
            resolve_date = resp_date + _dt.timedelta(hours=resolve_hours)
        except (ValueError, OverflowError):
            resolve_date = resp_date

        support_tickets.append({
            "id": f"TKT-{10000 + t:05d}",
            "severity": severity,
            "category": category,
            "opened_at": f"{open_date.strftime('%Y-%m-%d')} {open_hour:02d}:{open_min:02d}",
            "first_response_at": f"{resp_date.strftime('%Y-%m-%d')} {resp_hour:02d}:{resp_min:02d}",
            "resolved_at": resolve_date.strftime("%Y-%m-%d %H:%M"),
            "response_minutes": response_minutes,
        })

    # Compute actual support response averages
    avg_sev1_response_hrs = round(sum(sev1_response_times) / len(sev1_response_times), 2) if sev1_response_times else 0.0
    avg_sev2_response_hrs = round(sum(sev2_response_times) / len(sev2_response_times), 2) if sev2_response_times else 0.0
    avg_sev3_response_hrs = round(sum(sev3_response_times) / len(sev3_response_times), 2) if sev3_response_times else 0.0

    # Count breached tickets per severity
    sev1_breached = sum(1 for t in sev1_response_times if t > support_response_sev1_hrs)
    sev2_breached = sum(1 for t in sev2_response_times if t > support_response_sev2_hrs)

    support_pass = (sev1_breached == 0 and sev2_breached == 0)

    # Maintenance window compliance
    maint_violations = [inc for inc in incident_data
                        if inc["was_maintenance_window"]
                        and _dt.date.fromisoformat(inc["date"]).strftime("%A") != "Saturday"]
    maint_pass = len(maint_violations) == 0

    # Count total breaches
    breach_list: list[str] = []
    if not uptime_pass:
        breach_list.append("uptime")
    if not p95_pass:
        breach_list.append("p95_response_time")
    if not error_rate_pass:
        breach_list.append("error_rate")
    if not support_pass:
        breach_list.append("support_response")
    if not maint_pass:
        breach_list.append("maintenance_window")

    total_breaches = len(breach_list)
    credit_pct = min(total_breaches * credit_per_breach_pct, max_credit_pct)
    credit_amount = round(monthly_fee * credit_pct / 100, 2)

    # --- Build sla_agreement.txt ---
    sla_lines = [
        f"SERVICE LEVEL AGREEMENT",
        f"",
        f"Provider: {provider_name}",
        f"Customer: {company}",
        f"Effective Date: {report_year}-01-01",
        f"",
        "=" * 60,
        "SLA METRICS AND TARGETS",
        "=" * 60,
        "",
        f"1. UPTIME",
        f"   Target: {uptime_target_pct}% monthly uptime",
        f"   Measurement: Total available minutes / total minutes in month",
        f"   Exclusions: Planned maintenance during approved maintenance windows",
        f"               does NOT count against uptime.",
        "",
        f"2. RESPONSE TIME",
        f"   Target: P95 response time < {p95_target_ms}ms (averaged across all days)",
        f"   Measurement: Daily P95 values averaged over the reporting period",
        "",
        f"3. ERROR RATE",
        f"   Target: Average error rate < {error_rate_target_pct}%",
        f"   Measurement: Daily error rates averaged over the reporting period",
        "",
        f"4. SUPPORT RESPONSE TIME",
        f"   Severity 1 (Critical): First response within {support_response_sev1_hrs} hour(s)",
        f"   Severity 2 (High): First response within {support_response_sev2_hrs} hour(s)",
        f"   Severity 3 (Medium/Low): First response within {support_response_sev3_hrs} hour(s)",
        f"   Measurement: Time from ticket creation to first response",
        f"   Breach: Any Sev-1 or Sev-2 ticket exceeding the target constitutes a breach",
        "",
        f"5. MAINTENANCE WINDOWS",
        f"   Approved window: {maintenance_window_day} {maintenance_window_start}–{maintenance_window_end} UTC",
        f"   Any maintenance performed outside this window that causes downtime",
        f"   is counted as unplanned downtime.",
        "",
        "=" * 60,
        "FINANCIAL TERMS",
        "=" * 60,
        "",
        f"Monthly service fee: {_fmt_money(monthly_fee)}",
        f"Service credit per SLA breach: {credit_per_breach_pct}% of monthly fee",
        f"Maximum total credit: {max_credit_pct}% of monthly fee",
        f"Credits are applied to the next billing cycle.",
        "",
    ]
    sla_content = "\n".join(sla_lines) + "\n"

    # --- Build uptime_report.csv ---
    uptime_csv_lines = ["date,uptime_pct,incident_id"]
    for entry in uptime_data:
        uptime_csv_lines.append(f"{entry['date']},{entry['uptime_pct']},{entry['incident_id']}")
    uptime_csv_content = "\n".join(uptime_csv_lines) + "\n"

    # --- Build performance_metrics.csv ---
    perf_csv_lines = ["date,p50_ms,p95_ms,p99_ms,error_rate_pct"]
    for entry in perf_data:
        perf_csv_lines.append(f"{entry['date']},{entry['p50_ms']},{entry['p95_ms']},{entry['p99_ms']},{entry['error_rate_pct']}")
    perf_csv_content = "\n".join(perf_csv_lines) + "\n"

    # --- Build incident_log.csv ---
    incident_csv_lines = ["id,date,start_time,end_time,duration_minutes,root_cause,customer_impact,was_maintenance_window"]
    for inc in incident_data:
        incident_csv_lines.append(
            f"{inc['id']},{inc['date']},{inc['start_time']},{inc['end_time']},"
            f"{inc['duration_minutes']},{inc['root_cause']},{inc['customer_impact']},"
            f"{'Yes' if inc['was_maintenance_window'] else 'No'}"
        )
    incident_csv_content = "\n".join(incident_csv_lines) + "\n"

    # --- Build support_tickets.csv ---
    ticket_csv_lines = ["id,severity,category,opened_at,first_response_at,resolved_at"]
    for tkt in support_tickets:
        ticket_csv_lines.append(
            f"{tkt['id']},{tkt['severity']},{tkt['category']},"
            f"{tkt['opened_at']},{tkt['first_response_at']},{tkt['resolved_at']}"
        )
    ticket_csv_content = "\n".join(ticket_csv_lines) + "\n"

    # --- Problem statement ---
    report_month_name = _dt.date(report_year, report_month, 1).strftime("%B %Y")
    problem_statement = f"""# SLA Compliance Audit — {report_month_name}

You are {auditor_name}, a service management analyst at {company}. You must
audit the {provider_name} service against the SLA agreement for {report_month_name}.

## Source Files
- /testbed/data/sla_agreement.txt — SLA terms: uptime, response time, error rate, support response, maintenance windows, and financial penalties
- /testbed/data/uptime_report.csv — Daily uptime percentages with incident IDs
- /testbed/data/performance_metrics.csv — Daily P50, P95, P99 response times and error rates
- /testbed/data/incident_log.csv — Incident details: times, duration, root cause, whether in maintenance window
- /testbed/data/support_tickets.csv — Support tickets: severity, open/response/resolve times

## Requirements
1. Compute the actual monthly uptime percentage, EXCLUDING downtime during approved maintenance windows (Saturday 2-4am only). Maintenance performed outside this window counts as unplanned downtime.
2. Compute the average P95 response time across all days in the period
3. Compute the average daily error rate across the period
4. Compute average support response time by severity level
5. Identify any Sev-1 or Sev-2 tickets that breached their SLA target
6. Check if any maintenance was performed outside the approved window
7. For each SLA metric, determine PASS or FAIL
8. Compute the total number of SLA breaches and the corresponding service credit

Write your SLA compliance report to /testbed/sla_report.txt"""

    # --- Build rubric ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does the file /testbed/sla_report.txt exist and contain substantive content?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_uptime_pct",
            question=f"Does the report compute the actual monthly uptime (excluding approved maintenance window downtime) as approximately {actual_uptime_pct:.2f}% (within 0.1%)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_uptime_pass_fail",
            question=f"Does the report correctly determine that the uptime metric is a {'PASS' if uptime_pass else 'FAIL'} against the {uptime_target_pct}% target?",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_avg_p95",
            question=f"Does the report compute the average daily P95 response time as approximately {actual_avg_p95}ms (within 10ms)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_p95_pass_fail",
            question=f"Does the report correctly determine that the P95 response time metric is a {'PASS' if p95_pass else 'FAIL'} against the {p95_target_ms}ms target?",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_error_rate",
            question=f"Does the report compute the average error rate as approximately {actual_error_rate:.3f}% (within 0.05%)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_error_rate_pass_fail",
            question=f"Does the report correctly determine that the error rate metric is a {'PASS' if error_rate_pass else 'FAIL'} against the {error_rate_target_pct}% target?",
            points=2,
        ),
    ]

    # Support response checks
    if sev1_response_times:
        rubric_items.append(
            BinaryRubricCategory(
                name="correct_sev1_avg_response",
                question=f"Does the report state the average Severity-1 support response time as approximately {avg_sev1_response_hrs:.1f} hours (within 0.5 hours)?",
                points=2,
            )
        )
    if sev1_breached > 0:
        rubric_items.append(
            BinaryRubricCategory(
                name="correct_sev1_breach_count",
                question=f"Does the report identify that {sev1_breached} Severity-1 ticket(s) breached the {support_response_sev1_hrs}-hour SLA?",
                points=2,
            )
        )

    rubric_items.append(
        BinaryRubricCategory(
            name="correct_support_pass_fail",
            question=f"Does the report correctly determine that the support response metric is a {'PASS' if support_pass else 'FAIL'}?",
            points=2,
        )
    )

    # Maintenance window
    rubric_items.append(
        BinaryRubricCategory(
            name="identifies_maintenance_window_exclusion",
            question="Does the report correctly note that downtime during approved Saturday 2-4am maintenance windows should be excluded from the uptime calculation?",
            points=2,
        )
    )

    if maint_violations:
        rubric_items.append(
            BinaryRubricCategory(
                name="identifies_maint_violation",
                question=f"Does the report identify that maintenance was performed outside the approved window (on a non-Saturday day), counting as unplanned downtime?",
                points=2,
            )
        )

    rubric_items.append(
        BinaryRubricCategory(
            name="correct_maint_pass_fail",
            question=f"Does the report correctly determine that the maintenance window compliance is a {'PASS' if maint_pass else 'FAIL'}?",
            points=1,
        )
    )

    # Overall breach count and credit
    rubric_items.extend([
        BinaryRubricCategory(
            name="correct_total_breaches",
            question=f"Does the report identify exactly {total_breaches} total SLA breach(es) across all metrics?",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_credit_pct",
            question=f"Does the report compute the service credit as {credit_pct}% of the monthly fee (i.e., {total_breaches} breaches x {credit_per_breach_pct}% per breach, capped at {max_credit_pct}%)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_credit_amount",
            question=f"Does the report compute the service credit amount as approximately {_fmt_money(credit_amount)} (within $100)?",
            points=2,
        ),
    ])

    # Correct downtime minutes
    rubric_items.append(
        BinaryRubricCategory(
            name="correct_downtime_minutes",
            question=f"Does the report state the total non-maintenance downtime as approximately {total_downtime_minutes} minutes (within 10 minutes)?",
            points=1,
        )
    )

    # False-positive: passing metrics should not be flagged as breached
    for passing in passing_metrics:
        label = passing.replace("_", " ")
        rubric_items.append(
            BinaryRubricCategory(
                name=f"no_false_breach_{passing}",
                question=f"Does the report correctly show the '{label}' metric as PASS (not flagged as a breach)?",
                points=1,
            )
        )

    rubric_items.append(
        RubricCategory(
            name="analysis_thoroughness",
            description="How thorough and systematic is the SLA compliance audit?",
            failure="Superficial; most metrics not computed or many errors in calculations.",
            minor_failure="Some metrics computed correctly but missing key analyses like maintenance exclusion or financial credit.",
            minor_success="Most metrics computed with reasonable accuracy; minor gaps in analysis.",
            success="All metrics computed accurately with detailed methodology, clear pass/fail determinations, and correct financial credit calculation.",
            points=3,
        )
    )

    rubric = tuple(rubric_items)

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write your completed SLA compliance report to /testbed/sla_report.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/sla_agreement.txt": sla_content,
            "/testbed/data/uptime_report.csv": uptime_csv_content,
            "/testbed/data/performance_metrics.csv": perf_csv_content,
            "/testbed/data/incident_log.csv": incident_csv_content,
            "/testbed/data/support_tickets.csv": ticket_csv_content,
        },
        problem_type="sla_compliance_audit",
    )
