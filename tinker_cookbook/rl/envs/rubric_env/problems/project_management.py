"""Project management analysis tasks: critical path analysis, earned value analysis,
and resource leveling across a multi-project portfolio.

Each factory generates a realistic scenario with necessary_files containing all
information required to solve the problem. Seeds control randomization of task counts,
durations, resource assignments, cost distributions, and which issues are planted.
"""

from __future__ import annotations

import math
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


def _fmt_pct(value: float, decimals: int = 1) -> str:
    """Format a float as a percentage string."""
    return f"{value:.{decimals}f}%"


# =============================================================================
# DOMAIN POOLS
# =============================================================================

PROJECT_TYPES = [
    ("Software Platform Migration", "technology"),
    ("Office Building Renovation", "construction"),
    ("ERP System Implementation", "technology"),
    ("Product Launch Campaign", "marketing"),
    ("Manufacturing Line Upgrade", "manufacturing"),
    ("Data Center Buildout", "infrastructure"),
    ("Clinical Trial Phase II", "pharmaceutical"),
    ("Supply Chain Digitization", "logistics"),
    ("Mobile App Development", "technology"),
    ("Facility Expansion", "construction"),
    ("Regulatory Compliance Overhaul", "compliance"),
    ("Customer Portal Redesign", "technology"),
]

TASK_NAME_POOLS = {
    "technology": [
        "Requirements Gathering", "Architecture Design", "Database Schema Design",
        "Backend API Development", "Frontend UI Development", "Authentication Module",
        "Payment Integration", "Third-party API Integration", "Unit Testing",
        "Integration Testing", "Performance Testing", "Security Audit",
        "User Acceptance Testing", "Data Migration", "Documentation",
        "Deployment Planning", "Staging Environment Setup", "Production Deployment",
        "Load Testing", "Code Review", "CI/CD Pipeline Setup",
        "Monitoring Setup", "Training Materials", "Stakeholder Demo",
        "Bug Fix Sprint", "Accessibility Compliance", "API Documentation",
        "Mobile Responsive Design", "Search Feature Implementation",
        "Notification System", "Reporting Dashboard", "Admin Panel",
        "Backup Strategy Implementation", "Disaster Recovery Plan",
        "Performance Optimization", "Cache Layer Implementation",
    ],
    "construction": [
        "Site Survey", "Permit Application", "Architectural Drawings",
        "Foundation Work", "Structural Framing", "Electrical Rough-In",
        "Plumbing Rough-In", "HVAC Installation", "Insulation",
        "Drywall Installation", "Interior Painting", "Flooring Installation",
        "Exterior Finishing", "Landscaping", "Final Inspection",
        "Punch List Items", "Fire Suppression System", "Elevator Installation",
        "Roofing", "Window Installation", "Concrete Work",
        "Steel Erection", "Mechanical Systems", "Parking Lot Paving",
        "Signage Installation", "Security System Installation",
        "ADA Compliance Modifications", "Environmental Remediation",
        "Demolition", "Grading and Excavation", "Utility Connections",
        "Interior Design Fitout", "Furniture Installation",
    ],
    "manufacturing": [
        "Equipment Procurement", "Vendor Selection", "Floor Plan Design",
        "Utility Upgrades", "Equipment Installation", "Calibration",
        "Safety Systems Installation", "Operator Training", "Test Runs",
        "Quality Assurance Setup", "Production Line Testing",
        "Inventory System Setup", "Waste Management Plan",
        "Environmental Compliance", "Staff Certification",
        "Maintenance Schedule Design", "Spare Parts Inventory",
        "Automation Programming", "Conveyor System Installation",
        "Quality Control Station Setup", "Packaging Line Setup",
        "Raw Material Storage Setup", "Shipping Area Configuration",
        "ERP Integration", "Barcode System Setup",
        "Safety Training", "Emergency Protocol Development",
        "First Article Inspection", "Process Documentation",
    ],
    "marketing": [
        "Market Research", "Target Audience Analysis", "Brand Strategy",
        "Creative Brief", "Campaign Design", "Content Creation",
        "Social Media Strategy", "Email Campaign Setup", "Landing Page Design",
        "SEO Optimization", "PPC Campaign Setup", "Influencer Outreach",
        "Press Kit Preparation", "Media Buying", "Analytics Setup",
        "A/B Testing Framework", "Launch Event Planning", "PR Strategy",
        "Customer Survey Design", "Competitive Analysis",
        "Video Production", "Photography Session", "Copywriting",
        "Brand Guidelines Update", "Collateral Design",
        "Trade Show Preparation", "Webinar Planning",
        "Customer Testimonials Collection", "Case Study Development",
    ],
    "infrastructure": [
        "Site Selection", "Power Capacity Assessment", "Cooling Design",
        "Network Architecture", "Server Rack Installation", "Cabling",
        "UPS Installation", "Generator Setup", "Fire Suppression",
        "Physical Security", "Environmental Monitoring", "Network Configuration",
        "Storage Array Setup", "Virtualization Platform", "Backup Systems",
        "Disaster Recovery Site", "Compliance Documentation",
        "Performance Baseline", "Capacity Planning", "Vendor Negotiations",
        "Fiber Optic Installation", "Switch Configuration",
        "Load Balancer Setup", "DNS Configuration",
        "SSL Certificate Deployment", "Monitoring Dashboard",
    ],
    "pharmaceutical": [
        "Protocol Development", "IRB Submission", "Site Selection",
        "Patient Recruitment Plan", "CRF Design", "Database Setup",
        "Randomization System", "Drug Supply Chain", "Investigator Training",
        "Patient Screening", "Treatment Phase", "Follow-up Visits",
        "Adverse Event Monitoring", "Data Collection", "Interim Analysis",
        "Statistical Analysis Plan", "Safety Monitoring", "Regulatory Filing",
        "Site Monitoring Visits", "Data Cleaning",
        "Final Report Drafting", "Manuscript Preparation",
        "DSMB Review", "Lab Kit Preparation",
        "Biospecimen Management", "Pharmacovigilance Setup",
    ],
    "logistics": [
        "Current State Assessment", "Process Mapping", "Vendor Evaluation",
        "Platform Selection", "System Configuration", "Data Migration Planning",
        "Warehouse Management Module", "Transportation Management Module",
        "Order Management Module", "Inventory Tracking Setup",
        "Barcode/RFID Integration", "API Gateway Setup",
        "Carrier Integration", "Real-time Tracking Setup",
        "Reporting Dashboard", "User Training", "Pilot Testing",
        "Change Management", "Go-Live Preparation", "Post-Go-Live Support",
        "SLA Definition", "KPI Dashboard Setup",
        "Customer Notification System", "Returns Management Module",
    ],
    "compliance": [
        "Regulatory Gap Assessment", "Policy Review", "Risk Assessment",
        "Control Framework Design", "Documentation Standards",
        "Training Program Development", "Audit Trail Setup",
        "Monitoring System Implementation", "Incident Response Plan",
        "Third-party Risk Assessment", "Data Privacy Review",
        "Reporting Framework", "Compliance Testing",
        "Remediation Planning", "Board Reporting Package",
        "External Audit Preparation", "Staff Certification Program",
        "Whistleblower Hotline Setup", "Ethics Training Module",
        "Regulatory Filing Calendar", "Policy Distribution System",
        "Compliance Dashboard", "Vendor Compliance Verification",
    ],
}

RESOURCE_ROLES = [
    "Project Manager", "Senior Engineer", "Engineer", "Junior Engineer",
    "Business Analyst", "QA Lead", "QA Engineer", "Designer",
    "Architect", "DevOps Engineer", "Data Analyst", "Technical Writer",
    "Subject Matter Expert", "Consultant", "Team Lead",
]


# =============================================================================
# 1. CRITICAL PATH ANALYSIS
# =============================================================================


def make_critical_path_analysis(rand_seed: int = 42) -> RubricDatapoint:
    """Analyze a project schedule for critical path, slack, and scheduling issues.

    Source files: project_tasks.csv, resource_calendar.csv, project_charter.txt,
    change_requests.txt.

    Seed varies: project type/size (8+ types), task counts (20-35), durations,
    predecessor networks, resource assignments, which issues are planted.
    """
    rng = _random.Random(rand_seed)

    company = pick1(COMPANY_NAMES, rand_seed)
    pm_name = random_name(rand_seed + 1)

    # Pick project type
    project_info = rng.choice(PROJECT_TYPES)
    project_name = project_info[0]
    project_category = project_info[1]

    # --- Generate tasks ---
    n_tasks = rng.randint(20, 35)
    task_pool = TASK_NAME_POOLS.get(project_category, TASK_NAME_POOLS["technology"])
    task_names = rng.sample(task_pool, min(n_tasks, len(task_pool)))
    while len(task_names) < n_tasks:
        task_names.append(f"Task Phase {len(task_names) + 1}")

    # Generate resources
    n_resources = rng.randint(6, 10)
    resource_names = random_names(rand_seed + 10, n_resources)
    resource_roles = [rng.choice(RESOURCE_ROLES) for _ in range(n_resources)]

    # Resource availability: some have vacations or part-time schedules
    resource_availability: list[dict] = []
    for i, (rname, role) in enumerate(zip(resource_names, resource_roles)):
        is_part_time = rng.random() < 0.15
        vacation_start = None
        vacation_end = None
        if rng.random() < 0.3:
            vac_day = rng.randint(5, 50)
            vacation_start = vac_day
            vacation_end = vac_day + rng.randint(3, 10)
        resource_availability.append({
            "name": rname,
            "role": role,
            "hours_per_day": 4 if is_part_time else 8,
            "is_part_time": is_part_time,
            "vacation_start_day": vacation_start,
            "vacation_end_day": vacation_end,
        })

    # --- Build task network with predecessors ---
    tasks: list[dict] = []
    for i in range(n_tasks):
        task_id = f"T{i + 1:03d}"
        duration = rng.randint(2, 15)
        cost = rng.randint(2000, 25000)

        # Predecessors: tasks can depend on earlier tasks
        predecessors: list[str] = []
        if i > 0:
            # Each task has 0-3 predecessors from earlier tasks
            n_preds = rng.randint(0, min(3, i))
            if i <= 3:
                n_preds = min(n_preds, 1)
            if n_preds > 0:
                pred_indices = rng.sample(range(max(0, i - 8), i), min(n_preds, i))
                predecessors = [f"T{idx + 1:03d}" for idx in pred_indices]

        # Assign 1-2 resources
        n_assigned = rng.randint(1, 2)
        assigned_resources = rng.sample(resource_names, n_assigned)

        tasks.append({
            "id": task_id,
            "name": task_names[i],
            "duration": duration,
            "predecessors": predecessors,
            "resources": assigned_resources,
            "cost": cost,
        })

    # --- Forward pass: compute early start (ES) and early finish (EF) ---
    task_by_id: dict[str, dict] = {t["id"]: t for t in tasks}
    es: dict[str, int] = {}
    ef: dict[str, int] = {}

    def _compute_es(tid: str) -> int:
        if tid in es:
            return es[tid]
        t = task_by_id[tid]
        if not t["predecessors"]:
            es[tid] = 0
        else:
            es[tid] = max(_compute_ef(pid) for pid in t["predecessors"])
        ef[tid] = es[tid] + t["duration"]
        return es[tid]

    def _compute_ef(tid: str) -> int:
        if tid in ef:
            return ef[tid]
        _compute_es(tid)
        return ef[tid]

    for t in tasks:
        _compute_es(t["id"])

    project_duration = max(ef.values())

    # --- Backward pass: compute late start (LS) and late finish (LF) ---
    ls: dict[str, int] = {}
    lf: dict[str, int] = {}

    # Build successor map
    successors: dict[str, list[str]] = {t["id"]: [] for t in tasks}
    for t in tasks:
        for pid in t["predecessors"]:
            successors[pid].append(t["id"])

    def _compute_lf(tid: str) -> int:
        if tid in lf:
            return lf[tid]
        t = task_by_id[tid]
        if not successors[tid]:
            lf[tid] = project_duration
        else:
            lf[tid] = min(_compute_ls(sid) for sid in successors[tid])
        ls[tid] = lf[tid] - t["duration"]
        return lf[tid]

    def _compute_ls(tid: str) -> int:
        if tid in ls:
            return ls[tid]
        _compute_lf(tid)
        return ls[tid]

    for t in tasks:
        _compute_lf(t["id"])

    # --- Compute float/slack ---
    total_float: dict[str, int] = {}
    for t in tasks:
        total_float[t["id"]] = ls[t["id"]] - es[t["id"]]

    # --- Critical path: tasks with zero float ---
    critical_path_ids = [t["id"] for t in tasks if total_float[t["id"]] == 0]

    # --- Set up project deadline and milestones ---
    # Deadline: sometimes tight (project at risk), sometimes comfortable
    deadline_buffer = rng.choice([-5, -3, 0, 5, 10, 15])
    project_deadline = project_duration + deadline_buffer
    deadline_at_risk = project_duration > project_deadline

    # Milestones at ~25%, ~50%, ~75%, ~100% of timeline
    milestones = []
    milestone_pcts = [0.25, 0.50, 0.75, 1.00]
    milestone_labels = ["Phase 1 Complete", "Midpoint Review", "Phase 3 Complete", "Project Delivery"]
    for pct, label in zip(milestone_pcts, milestone_labels):
        target_day = int(project_deadline * pct)
        # Find the latest critical-path task finishing by that point
        tasks_by_ef = sorted(tasks, key=lambda t: ef[t["id"]])
        milestone_task = None
        for t in tasks_by_ef:
            if ef[t["id"]] <= target_day + 3:
                milestone_task = t["id"]
        milestones.append({
            "label": label,
            "target_day": target_day,
            "linked_task": milestone_task,
        })

    # --- Plant issues ---

    # Issue 1: Resource conflicts (same person assigned to parallel tasks on critical path)
    resource_conflicts: list[dict] = []
    cp_tasks = [task_by_id[tid] for tid in critical_path_ids]
    for i, t1 in enumerate(cp_tasks):
        for t2 in cp_tasks[i + 1:]:
            # Check if they overlap in time
            t1_start, t1_end = es[t1["id"]], ef[t1["id"]]
            t2_start, t2_end = es[t2["id"]], ef[t2["id"]]
            overlap = max(0, min(t1_end, t2_end) - max(t1_start, t2_start))
            if overlap > 0:
                shared_resources = set(t1["resources"]) & set(t2["resources"])
                for res in shared_resources:
                    resource_conflicts.append({
                        "resource": res,
                        "task1": t1["id"],
                        "task1_name": t1["name"],
                        "task2": t2["id"],
                        "task2_name": t2["name"],
                        "overlap_days": overlap,
                    })

    # If no natural conflicts, plant one
    if not resource_conflicts and len(cp_tasks) >= 2:
        # Find two CP tasks that overlap and force a shared resource
        for i, t1 in enumerate(cp_tasks):
            for t2 in cp_tasks[i + 1:]:
                t1_start, t1_end = es[t1["id"]], ef[t1["id"]]
                t2_start, t2_end = es[t2["id"]], ef[t2["id"]]
                overlap = max(0, min(t1_end, t2_end) - max(t1_start, t2_start))
                if overlap > 0:
                    shared_res = t1["resources"][0]
                    if shared_res not in t2["resources"]:
                        t2["resources"].append(shared_res)
                    resource_conflicts.append({
                        "resource": shared_res,
                        "task1": t1["id"],
                        "task1_name": t1["name"],
                        "task2": t2["id"],
                        "task2_name": t2["name"],
                        "overlap_days": overlap,
                    })
                    break
            if resource_conflicts:
                break

    # Issue 2: Tasks with zero float that aren't on the identified critical path
    # (This happens when there are multiple critical paths or near-critical tasks)
    # We'll plant a near-critical task by slightly adjusting a non-CP task's duration
    near_critical_tasks: list[dict] = []
    non_cp_tasks = [t for t in tasks if total_float[t["id"]] > 0 and total_float[t["id"]] <= 2]
    for t in non_cp_tasks[:2]:
        near_critical_tasks.append({
            "id": t["id"],
            "name": t["name"],
            "float": total_float[t["id"]],
        })

    # Issue 3: Cost overrun risk from change requests
    total_project_cost = sum(t["cost"] for t in tasks)
    budget = int(total_project_cost * rng.uniform(0.95, 1.10))

    change_requests: list[dict] = []
    cr_types = [
        ("scope_addition", "Add {feature} to project scope"),
        ("requirement_change", "Modify {feature} requirements"),
        ("resource_addition", "Add additional {role} to team"),
    ]
    cr_features = [
        "reporting module", "mobile interface", "security hardening",
        "data analytics dashboard", "automated testing suite",
        "API versioning", "user notification system", "audit logging",
    ]
    n_crs = rng.randint(2, 3)
    for i in range(n_crs):
        cr_type, cr_template = rng.choice(cr_types)
        feature = rng.choice(cr_features)
        cr_name = cr_template.format(feature=feature, role=rng.choice(RESOURCE_ROLES))
        cr_cost = rng.randint(5000, 30000)
        cr_duration_impact = rng.randint(3, 12)
        cr_affects_cp = rng.random() < 0.5
        change_requests.append({
            "id": f"CR-{i + 1:03d}",
            "name": cr_name,
            "additional_cost": cr_cost,
            "duration_impact_days": cr_duration_impact,
            "affects_critical_path": cr_affects_cp,
        })

    total_cr_cost = sum(cr["additional_cost"] for cr in change_requests)
    cost_with_crs = total_project_cost + total_cr_cost
    budget_overrun = cost_with_crs > budget
    overrun_amount = max(0, cost_with_crs - budget)

    # Duration impact from CRs that affect CP
    cp_duration_impact = sum(
        cr["duration_impact_days"] for cr in change_requests if cr["affects_critical_path"]
    )
    duration_with_crs = project_duration + cp_duration_impact

    # False positive 1: A resource that appears heavily loaded but is actually fine
    # (assigned to sequential, not parallel tasks)
    false_positive_resource = None
    for res in resource_availability:
        assigned_tasks = [t for t in tasks if res["name"] in t["resources"]]
        if len(assigned_tasks) >= 4:
            # Check if tasks are sequential (non-overlapping)
            overlapping = False
            for i, t1 in enumerate(assigned_tasks):
                for t2 in assigned_tasks[i + 1:]:
                    s1, e1 = es[t1["id"]], ef[t1["id"]]
                    s2, e2 = es[t2["id"]], ef[t2["id"]]
                    if max(0, min(e1, e2) - max(s1, s2)) > 0:
                        overlapping = True
                        break
                if overlapping:
                    break
            if not overlapping:
                false_positive_resource = {
                    "name": res["name"],
                    "task_count": len(assigned_tasks),
                    "reason": "tasks are sequential, not parallel",
                }
                break

    # False positive 2: A milestone that appears late but is actually within buffer
    false_positive_milestone = None
    for m in milestones:
        if m["linked_task"]:
            actual_finish = ef.get(m["linked_task"], 0)
            if 0 < actual_finish - m["target_day"] <= 2:
                false_positive_milestone = {
                    "label": m["label"],
                    "target_day": m["target_day"],
                    "actual_day": actual_finish,
                    "within_buffer": True,
                }
                break

    # --- Build project_tasks.csv ---
    csv_lines = ["TaskID,TaskName,Duration(days),Predecessors,Resources,EstimatedCost"]
    for t in tasks:
        preds_str = ";".join(t["predecessors"]) if t["predecessors"] else ""
        res_str = ";".join(t["resources"])
        csv_lines.append(
            f"{t['id']},{t['name']},{t['duration']},{preds_str},{res_str},{t['cost']}"
        )
    tasks_csv = "\n".join(csv_lines) + "\n"

    # --- Build resource_calendar.csv ---
    cal_lines = ["ResourceName,Role,HoursPerDay,VacationStartDay,VacationEndDay"]
    for res in resource_availability:
        vac_start = res["vacation_start_day"] if res["vacation_start_day"] else ""
        vac_end = res["vacation_end_day"] if res["vacation_end_day"] else ""
        cal_lines.append(
            f"{res['name']},{res['role']},{res['hours_per_day']},{vac_start},{vac_end}"
        )
    calendar_csv = "\n".join(cal_lines) + "\n"

    # --- Build project_charter.txt ---
    charter_lines = [
        f"PROJECT CHARTER",
        f"",
        f"{'='*60}",
        f"Project: {project_name}",
        f"Organization: {company}",
        f"Project Manager: {pm_name}",
        f"Category: {project_category.title()}",
        f"{'='*60}",
        f"",
        f"PROJECT TIMELINE",
        f"  Start Date: Day 0",
        f"  Deadline: Day {project_deadline}",
        f"  All deliverables must be completed by deadline.",
        f"",
        f"BUDGET",
        f"  Approved Budget: {_fmt_money(budget)}",
        f"  Contingency Reserve: {_fmt_money(budget * 0.10)}",
        f"  Management Reserve: {_fmt_money(budget * 0.05)}",
        f"",
        f"MILESTONES",
    ]
    for m in milestones:
        charter_lines.append(f"  {m['label']}: Target Day {m['target_day']}")
    charter_lines.extend([
        f"",
        f"CONSTRAINTS",
        f"  - No tasks may begin before their predecessors are complete",
        f"  - Resources cannot work on overlapping tasks simultaneously",
        f"  - Part-time resources work {4} hours/day",
        f"  - Vacation periods are non-negotiable",
        f"",
        f"SUCCESS CRITERIA",
        f"  - All tasks completed within deadline",
        f"  - Total cost within approved budget",
        f"  - All milestones met",
        f"",
    ])
    charter_content = "\n".join(charter_lines) + "\n"

    # --- Build change_requests.txt ---
    cr_lines = [
        "PENDING CHANGE REQUESTS",
        "",
        "=" * 60,
    ]
    for cr in change_requests:
        cr_lines.extend([
            "",
            f"Change Request: {cr['id']}",
            f"  Description: {cr['name']}",
            f"  Additional Cost: {_fmt_money(cr['additional_cost'])}",
            f"  Schedule Impact: {cr['duration_impact_days']} additional days",
            f"  Note: This change would add new tasks that depend on existing",
            f"  deliverables in the project plan.",
            f"  Status: Pending Review",
            f"  {'_'*40}",
        ])
    cr_lines.append("")
    cr_content = "\n".join(cr_lines) + "\n"

    # --- Problem statement ---
    problem_statement = f"""# Critical Path Analysis

You are a project scheduling analyst for {company}. The project manager has
asked you to analyze the schedule for the "{project_name}" project.

## Source Files
- /testbed/data/project_tasks.csv - Task list with IDs, names, durations, predecessors, resources, costs
- /testbed/data/resource_calendar.csv - Resource availability, hours/day, vacation periods
- /testbed/data/project_charter.txt - Deadline, budget, milestones, constraints
- /testbed/data/change_requests.txt - Pending change requests with cost and schedule impact

## Requirements
1. Compute the critical path (list the task IDs on the critical path)
2. Compute the total project duration based on the task network
3. For each task, compute Early Start (ES), Early Finish (EF), Late Start (LS), Late Finish (LF)
4. For each task, compute Total Float (slack)
5. Identify resource conflicts (same person assigned to parallel tasks)
6. Assess whether the project deadline can be met
7. Evaluate the impact of pending change requests on budget and schedule
8. Identify any near-critical tasks (float <= 2 days) that could become critical
9. Compute total project cost and compare to budget

Write a detailed schedule analysis report to /testbed/schedule_report.txt."""

    # --- Build rubric ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does the file /testbed/schedule_report.txt exist and contain substantive content (more than a few lines)?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_task_count",
            question=f"Does the report correctly identify or work with {n_tasks} tasks in the project?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_project_duration",
            question=(
                f"Does the report correctly compute the total project duration as "
                f"{project_duration} days (based on the critical path through the task network)?"
            ),
            points=3,
        ),
        BinaryRubricCategory(
            name="correct_critical_path",
            question=(
                f"Does the report identify the critical path tasks? The critical path "
                f"includes tasks with zero total float. There are {len(critical_path_ids)} "
                f"such tasks: {', '.join(critical_path_ids)}. The report should list at "
                f"least 80% of these correctly."
            ),
            points=3,
        ),
    ]

    # Check ES/EF for 3 specific tasks
    check_tasks = rng.sample(tasks, min(3, len(tasks)))
    for t in check_tasks:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"correct_es_ef_{t['id']}",
                question=(
                    f"Does the report correctly compute ES={es[t['id']]}, EF={ef[t['id']]}, "
                    f"LS={ls[t['id']]}, LF={lf[t['id']]} for task {t['id']} ({t['name']})? "
                    f"(At minimum ES and EF should be correct.)"
                ),
                points=2,
            )
        )

    # Float check for 2 specific tasks
    float_check_tasks = rng.sample(
        [t for t in tasks if total_float[t["id"]] > 0], min(2, len([t for t in tasks if total_float[t["id"]] > 0]))
    )
    for t in float_check_tasks:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"correct_float_{t['id']}",
                question=(
                    f"Does the report correctly compute the total float for task {t['id']} "
                    f"({t['name']}) as {total_float[t['id']]} days?"
                ),
                points=2,
            )
        )

    # Resource conflict detection
    if resource_conflicts:
        for rc in resource_conflicts[:2]:
            rubric_items.append(
                BinaryRubricCategory(
                    name=f"finds_resource_conflict_{rc['task1']}_{rc['task2']}",
                    question=(
                        f"Does the report identify the resource conflict where {rc['resource']} is "
                        f"assigned to both {rc['task1']} ({rc['task1_name']}) and {rc['task2']} "
                        f"({rc['task2_name']}), which overlap by {rc['overlap_days']} days?"
                    ),
                    points=2,
                )
            )

    # Deadline assessment
    if deadline_at_risk:
        rubric_items.append(
            BinaryRubricCategory(
                name="identifies_deadline_risk",
                question=(
                    f"Does the report identify that the computed project duration "
                    f"({project_duration} days) exceeds the deadline (Day {project_deadline}), "
                    f"meaning the project is at risk of being late by "
                    f"{project_duration - project_deadline} days?"
                ),
                points=2,
            )
        )
    else:
        rubric_items.append(
            BinaryRubricCategory(
                name="confirms_deadline_feasible",
                question=(
                    f"Does the report confirm that the project duration ({project_duration} days) "
                    f"fits within the deadline (Day {project_deadline}) with "
                    f"{project_deadline - project_duration} days of schedule buffer?"
                ),
                points=2,
            )
        )

    # Near-critical tasks
    if near_critical_tasks:
        nc_desc = ", ".join(f"{t['id']} ({t['name']}, float={t['float']}d)" for t in near_critical_tasks)
        rubric_items.append(
            BinaryRubricCategory(
                name="identifies_near_critical",
                question=(
                    f"Does the report identify near-critical tasks (float <= 2 days) that could "
                    f"become critical? Near-critical tasks include: {nc_desc}."
                ),
                points=2,
            )
        )

    # Cost analysis
    rubric_items.append(
        BinaryRubricCategory(
            name="correct_total_cost",
            question=(
                f"Does the report compute the total project cost as approximately "
                f"{_fmt_money(total_project_cost)} (within 5%)?"
            ),
            points=2,
        )
    )

    # Change request impact
    rubric_items.append(
        BinaryRubricCategory(
            name="correct_cr_cost_impact",
            question=(
                f"Does the report correctly compute the total cost impact of all change "
                f"requests as {_fmt_money(total_cr_cost)}, bringing the total to "
                f"approximately {_fmt_money(cost_with_crs)}?"
            ),
            points=2,
        )
    )

    if budget_overrun:
        rubric_items.append(
            BinaryRubricCategory(
                name="identifies_budget_overrun",
                question=(
                    f"Does the report identify that accepting all change requests would "
                    f"cause a budget overrun of approximately {_fmt_money(overrun_amount)} "
                    f"(budget: {_fmt_money(budget)}, projected cost: {_fmt_money(cost_with_crs)})?"
                ),
                points=2,
            )
        )

    # False positive checks
    if false_positive_resource:
        rubric_items.append(
            BinaryRubricCategory(
                name="no_false_resource_conflict",
                question=(
                    f"Does the report correctly NOT flag {false_positive_resource['name']} as having a "
                    f"resource conflict? This person is assigned to {false_positive_resource['task_count']} "
                    f"tasks, but they are all sequential (non-overlapping), so there is no conflict."
                ),
                points=1,
            )
        )

    if false_positive_milestone:
        rubric_items.append(
            BinaryRubricCategory(
                name="no_false_milestone_alarm",
                question=(
                    f"Does the report correctly note that the '{false_positive_milestone['label']}' "
                    f"milestone (target Day {false_positive_milestone['target_day']}) finishing on "
                    f"Day {false_positive_milestone['actual_day']} is within acceptable buffer "
                    f"and not a major concern?"
                ),
                points=1,
            )
        )

    rubric_items.append(
        RubricCategory(
            name="analysis_quality",
            description="How thorough and well-organized is the schedule analysis?",
            failure="Disorganized output or missing critical path computation entirely.",
            minor_failure="Some schedule analysis but missing key elements (no float, no resource check).",
            minor_success="Reasonable analysis with critical path identified and most checks performed.",
            success="Comprehensive schedule analysis with critical path, float, resource conflicts, deadline assessment, and change request impact all clearly presented.",
            points=2,
        )
    )

    rubric = tuple(rubric_items)

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write your completed schedule analysis report to /testbed/schedule_report.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/project_tasks.csv": tasks_csv,
            "/testbed/data/resource_calendar.csv": calendar_csv,
            "/testbed/data/project_charter.txt": charter_content,
            "/testbed/data/change_requests.txt": cr_content,
        },
        problem_type="critical_path_analysis",
    )


# =============================================================================
# 2. EARNED VALUE ANALYSIS
# =============================================================================

EVM_FORMULAS_TEXT = """EARNED VALUE MANAGEMENT (EVM) FORMULAS

============================================================
BASIC METRICS
============================================================

PV (Planned Value) = Budgeted cost of work scheduled by status date
   Sum of (task_budget * planned_pct_complete) for each task

EV (Earned Value) = Budgeted cost of work actually performed
   Sum of (task_budget * actual_pct_complete) for each task

AC (Actual Cost) = Actual cost incurred for work performed
   Sum of actual_cost for each task

BAC (Budget at Completion) = Total project budget
   Sum of all task budgets

============================================================
VARIANCE INDICATORS
============================================================

SV (Schedule Variance) = EV - PV
   Positive = ahead of schedule, Negative = behind schedule

CV (Cost Variance) = EV - AC
   Positive = under budget, Negative = over budget

============================================================
PERFORMANCE INDICES
============================================================

SPI (Schedule Performance Index) = EV / PV
   > 1.0 = ahead, < 1.0 = behind, = 1.0 = on schedule

CPI (Cost Performance Index) = EV / AC
   > 1.0 = under budget, < 1.0 = over budget

============================================================
FORECASTING
============================================================

EAC (Estimate at Completion) = BAC / CPI
   (CPI method - assumes current cost efficiency continues)

ETC (Estimate to Complete) = EAC - AC
   Remaining cost to finish the project

VAC (Variance at Completion) = BAC - EAC
   Positive = expected under budget, Negative = expected overrun

TCPI (To-Complete Performance Index) = (BAC - EV) / (BAC - AC)
   CPI needed on remaining work to finish within BAC
   > 1.0 = must improve efficiency, < 1.0 = can relax

============================================================
"""


def make_earned_value_analysis(rand_seed: int = 42) -> RubricDatapoint:
    """Perform earned value analysis on a project status report.

    Source files: project_baseline.csv, actual_status.csv, evm_formulas.txt,
    project_history.csv.

    Seed varies: project type, number of tasks, budget distribution, which tasks
    have cost overruns, completion patterns, trend trajectory.
    """
    rng = _random.Random(rand_seed)

    company = pick1(COMPANY_NAMES, rand_seed)
    pm_name = random_name(rand_seed + 2)

    project_info = rng.choice(PROJECT_TYPES)
    project_name = project_info[0]
    project_category = project_info[1]

    # --- Generate tasks with budgets ---
    n_tasks = rng.randint(15, 25)
    task_pool = TASK_NAME_POOLS.get(project_category, TASK_NAME_POOLS["technology"])
    task_names = rng.sample(task_pool, min(n_tasks, len(task_pool)))
    while len(task_names) < n_tasks:
        task_names.append(f"Phase {len(task_names) + 1} Activity")

    # Total project duration in months
    total_months = rng.choice([8, 10, 12, 14, 16])
    # Status date: somewhere between month 4 and month (total_months - 2)
    status_month = rng.randint(4, min(total_months - 2, 10))

    tasks: list[dict] = []
    total_bac = 0
    for i in range(n_tasks):
        task_id = f"WBS-{i + 1:02d}"
        budget = rng.randint(10000, 80000)
        total_bac += budget

        # Planned % complete by each month
        start_month = rng.randint(1, max(1, status_month - 3))
        end_month = rng.randint(start_month + 2, min(total_months, start_month + 8))

        planned_pct: dict[int, float] = {}
        for m in range(1, total_months + 1):
            if m < start_month:
                planned_pct[m] = 0.0
            elif m >= end_month:
                planned_pct[m] = 100.0
            else:
                progress = (m - start_month + 1) / (end_month - start_month + 1) * 100
                planned_pct[m] = round(min(100.0, progress), 1)

        # Actual % complete and actual cost at status date
        planned_at_status = planned_pct.get(status_month, 0.0)

        # Some tasks are "90% done" syndrome: report 90% but cost 150% of proportional budget
        is_ninety_pct_syndrome = rng.random() < 0.15 and planned_at_status > 50
        if is_ninety_pct_syndrome:
            actual_pct = round(rng.uniform(85, 95), 1)
            # Cost 150% of what it should proportionally cost
            actual_cost = round(budget * (actual_pct / 100) * rng.uniform(1.40, 1.70), 2)
        else:
            # Normal variation
            pct_delta = rng.uniform(-15, 10)
            actual_pct = round(max(0, min(100, planned_at_status + pct_delta)), 1)
            cost_efficiency = rng.uniform(0.85, 1.15)
            actual_cost = round(budget * (actual_pct / 100) * cost_efficiency, 2)

        tasks.append({
            "id": task_id,
            "name": task_names[i],
            "budget": budget,
            "start_month": start_month,
            "end_month": end_month,
            "planned_pct": planned_pct,
            "actual_pct_at_status": actual_pct,
            "actual_cost_at_status": actual_cost,
            "is_ninety_pct_syndrome": is_ninety_pct_syndrome,
        })

    # --- Compute EVM metrics from raw data ---
    pv = sum(t["budget"] * t["planned_pct"].get(status_month, 0) / 100 for t in tasks)
    ev = sum(t["budget"] * t["actual_pct_at_status"] / 100 for t in tasks)
    ac = sum(t["actual_cost_at_status"] for t in tasks)

    pv = round(pv, 2)
    ev = round(ev, 2)
    ac = round(ac, 2)

    sv = round(ev - pv, 2)
    cv = round(ev - ac, 2)

    spi = round(ev / pv, 4) if pv > 0 else 0.0
    cpi = round(ev / ac, 4) if ac > 0 else 0.0

    eac = round(total_bac / cpi, 2) if cpi > 0 else total_bac * 2
    etc = round(eac - ac, 2)
    vac = round(total_bac - eac, 2)
    tcpi = round((total_bac - ev) / (total_bac - ac), 4) if (total_bac - ac) > 0 else 999.0

    # --- Generate historical EVM snapshots ---
    history: list[dict] = []
    for m in range(1, status_month):
        h_pv = sum(t["budget"] * t["planned_pct"].get(m, 0) / 100 for t in tasks)
        # Simulate gradual degradation of CPI over time
        cpi_trend = 1.0 - (m * rng.uniform(0.01, 0.04))
        h_ev_raw = h_pv * rng.uniform(0.85, 1.05)
        h_ac_raw = h_ev_raw / max(0.5, cpi_trend + rng.uniform(-0.05, 0.05))
        h_ev = round(h_ev_raw, 2)
        h_ac = round(h_ac_raw, 2)
        h_spi = round(h_ev / h_pv, 4) if h_pv > 0 else 1.0
        h_cpi = round(h_ev / h_ac, 4) if h_ac > 0 else 1.0
        history.append({
            "month": m,
            "pv": round(h_pv, 2),
            "ev": h_ev,
            "ac": h_ac,
            "spi": h_spi,
            "cpi": h_cpi,
        })

    # Determine trend direction
    if len(history) >= 3:
        recent_cpis = [h["cpi"] for h in history[-3:]]
        cpi_trending_down = all(recent_cpis[i] > recent_cpis[i + 1] for i in range(len(recent_cpis) - 1))
    else:
        cpi_trending_down = cpi < 1.0

    # Tasks with 90% syndrome
    syndrome_tasks = [t for t in tasks if t["is_ninety_pct_syndrome"]]

    # --- Build project_baseline.csv ---
    baseline_lines = [
        "TaskID,TaskName,Budget,StartMonth,EndMonth," +
        ",".join(f"Planned_Pct_M{m}" for m in range(1, total_months + 1))
    ]
    for t in tasks:
        pct_vals = ",".join(
            str(t["planned_pct"].get(m, 0.0)) for m in range(1, total_months + 1)
        )
        baseline_lines.append(
            f"{t['id']},{t['name']},{t['budget']},{t['start_month']},{t['end_month']},{pct_vals}"
        )
    baseline_lines.append("")
    baseline_lines.append(f"# BAC (Budget at Completion): {total_bac}")
    baseline_lines.append(f"# Total project duration: {total_months} months")
    baseline_csv = "\n".join(baseline_lines) + "\n"

    # --- Build actual_status.csv ---
    status_lines = ["TaskID,TaskName,Budget,ActualPctComplete,ActualCost"]
    for t in tasks:
        status_lines.append(
            f"{t['id']},{t['name']},{t['budget']},{t['actual_pct_at_status']},{t['actual_cost_at_status']}"
        )
    status_lines.append("")
    status_lines.append(f"# Status Date: End of Month {status_month}")
    status_csv = "\n".join(status_lines) + "\n"

    # --- Build project_history.csv ---
    history_lines = ["Month,PV,EV,AC,SPI,CPI"]
    for h in history:
        history_lines.append(
            f"{h['month']},{h['pv']},{h['ev']},{h['ac']},{h['spi']},{h['cpi']}"
        )
    history_csv = "\n".join(history_lines) + "\n"

    # --- Problem statement ---
    problem_statement = f"""# Earned Value Analysis

You are a project controls analyst for {company}. The project manager for
"{project_name}" needs a comprehensive earned value analysis as of Month {status_month}.

## Source Files
- /testbed/data/project_baseline.csv - Planned value by task by month (budget, planned % complete per month)
- /testbed/data/actual_status.csv - Actual cost and actual % complete per task at status date
- /testbed/data/evm_formulas.txt - Reference formulas for all EVM metrics
- /testbed/data/project_history.csv - Previous months' EVM snapshots (PV, EV, AC, SPI, CPI)

## Requirements
1. Compute PV, EV, and AC at the status date (Month {status_month}) from the raw task data
2. Compute Schedule Variance (SV) and Cost Variance (CV)
3. Compute SPI and CPI
4. Compute EAC (using CPI method), ETC, VAC, and TCPI
5. Identify tasks with significant cost overruns (actual cost >> earned value proportion)
6. Analyze the CPI trend from historical data - is it improving or deteriorating?
7. Provide an overall project health assessment with specific recommendations

Write a detailed EVM analysis report to /testbed/evm_report.txt."""

    # --- Build rubric ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does the file /testbed/evm_report.txt exist and contain substantive content (more than a few lines)?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_bac",
            question=f"Does the report correctly state the BAC (Budget at Completion) as {_fmt_money(total_bac)}?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_pv",
            question=(
                f"Does the report correctly compute PV (Planned Value) at Month {status_month} as "
                f"approximately {_fmt_money(pv)} (within 2%)? PV = sum of (task_budget * planned_pct_complete / 100) "
                f"for each task at Month {status_month}."
            ),
            points=3,
        ),
        BinaryRubricCategory(
            name="correct_ev",
            question=(
                f"Does the report correctly compute EV (Earned Value) as approximately "
                f"{_fmt_money(ev)} (within 2%)? EV = sum of (task_budget * actual_pct_complete / 100) "
                f"for each task."
            ),
            points=3,
        ),
        BinaryRubricCategory(
            name="correct_ac",
            question=(
                f"Does the report correctly compute AC (Actual Cost) as approximately "
                f"{_fmt_money(ac)} (within 2%)? AC = sum of actual_cost for each task."
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_sv",
            question=(
                f"Does the report correctly compute SV (Schedule Variance) = EV - PV as "
                f"approximately {_fmt_money(sv)} (within 5%)?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_cv",
            question=(
                f"Does the report correctly compute CV (Cost Variance) = EV - AC as "
                f"approximately {_fmt_money(cv)} (within 5%)?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_spi",
            question=(
                f"Does the report correctly compute SPI = EV/PV as approximately "
                f"{spi:.2f} (within 0.05)?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_cpi",
            question=(
                f"Does the report correctly compute CPI = EV/AC as approximately "
                f"{cpi:.2f} (within 0.05)?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_eac",
            question=(
                f"Does the report correctly compute EAC = BAC/CPI as approximately "
                f"{_fmt_money(eac)} (within 5%)?"
            ),
            points=3,
        ),
        BinaryRubricCategory(
            name="correct_etc",
            question=(
                f"Does the report correctly compute ETC = EAC - AC as approximately "
                f"{_fmt_money(etc)} (within 5%)?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_vac",
            question=(
                f"Does the report correctly compute VAC = BAC - EAC as approximately "
                f"{_fmt_money(vac)} (within 5%)?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_tcpi",
            question=(
                f"Does the report correctly compute TCPI = (BAC-EV)/(BAC-AC) as approximately "
                f"{tcpi:.2f} (within 0.05)?"
            ),
            points=2,
        ),
    ]

    # 90% syndrome detection
    if syndrome_tasks:
        for st in syndrome_tasks[:2]:
            earned_value_task = st["budget"] * st["actual_pct_at_status"] / 100
            cost_ratio = st["actual_cost_at_status"] / earned_value_task if earned_value_task > 0 else 999
            rubric_items.append(
                BinaryRubricCategory(
                    name=f"identifies_overrun_{st['id']}",
                    question=(
                        f"Does the report identify {st['id']} ({st['name']}) as having a significant "
                        f"cost overrun? It reports {st['actual_pct_at_status']}% complete but has "
                        f"consumed {_fmt_money(st['actual_cost_at_status'])} against a budget of "
                        f"{_fmt_money(st['budget'])}, giving a task-level cost ratio of {cost_ratio:.2f}x."
                    ),
                    points=2,
                )
            )

    # CPI trend analysis
    rubric_items.append(
        BinaryRubricCategory(
            name="identifies_cpi_trend",
            question=(
                f"Does the report analyze the CPI trend from historical data and correctly identify "
                f"that the CPI is {'deteriorating (trending downward)' if cpi_trending_down else 'relatively stable or improving'}?"
            ),
            points=2,
        )
    )

    # False positive: a task that looks expensive but has a valid reason
    # (e.g., it's nearly complete and the apparent cost overrun is actually within 5%)
    well_performing_tasks = [
        t for t in tasks
        if not t["is_ninety_pct_syndrome"]
        and t["actual_pct_at_status"] > 60
        and abs(t["actual_cost_at_status"] - t["budget"] * t["actual_pct_at_status"] / 100) / t["budget"] < 0.10
    ]
    if well_performing_tasks:
        fp_task = rng.choice(well_performing_tasks[:3])
        rubric_items.append(
            BinaryRubricCategory(
                name="no_false_overrun_flag",
                question=(
                    f"Does the report correctly NOT flag {fp_task['id']} ({fp_task['name']}) as a "
                    f"problematic cost overrun? It is {fp_task['actual_pct_at_status']}% complete with "
                    f"actual cost {_fmt_money(fp_task['actual_cost_at_status'])} against a proportional "
                    f"expected cost of approximately {_fmt_money(fp_task['budget'] * fp_task['actual_pct_at_status'] / 100)}, "
                    f"which is within acceptable variance."
                ),
                points=1,
            )
        )

    rubric_items.append(
        RubricCategory(
            name="analysis_quality",
            description="How thorough and insightful is the EVM analysis?",
            failure="Missing most EVM computations or numbers are fabricated.",
            minor_failure="Some metrics computed but missing key forecasting (EAC, TCPI) or trend analysis.",
            minor_success="Most metrics computed correctly with reasonable interpretation and recommendations.",
            success="Comprehensive EVM analysis with all metrics, trend analysis, task-level insights, and actionable recommendations.",
            points=2,
        )
    )

    rubric = tuple(rubric_items)

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write your completed EVM analysis report to /testbed/evm_report.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/project_baseline.csv": baseline_csv,
            "/testbed/data/actual_status.csv": status_csv,
            "/testbed/data/evm_formulas.txt": EVM_FORMULAS_TEXT,
            "/testbed/data/project_history.csv": history_csv,
        },
        problem_type="earned_value_analysis",
    )


# =============================================================================
# 3. RESOURCE LEVELING ANALYSIS
# =============================================================================

SKILL_POOL = [
    "Python", "Java", "JavaScript", "SQL", "Project Management",
    "Data Analysis", "UI/UX Design", "Cloud Architecture", "DevOps",
    "Business Analysis", "QA/Testing", "Technical Writing",
    "Machine Learning", "Network Engineering", "Security",
    "Database Administration", "Mobile Development", "Agile/Scrum",
]

PORTFOLIO_PROJECT_NAMES = [
    "Phoenix Platform", "Atlas Migration", "Compass Analytics",
    "Horizon Mobile App", "Meridian Integration", "Summit Portal",
    "Vanguard Automation", "Beacon Dashboard", "Crestview API",
    "Delta Optimization", "Evergreen Compliance", "Frontier Rollout",
]


def make_resource_leveling_analysis(rand_seed: int = 42) -> RubricDatapoint:
    """Analyze resource allocation across multiple concurrent projects.

    Source files: project_portfolio.csv, resource_assignments.csv,
    resource_capacity.csv, prioritization_criteria.txt.

    Seed varies: number of projects (4-6), team size, assignment density,
    which resources are over-allocated, skill mismatches planted.
    """
    rng = _random.Random(rand_seed)

    company = pick1(COMPANY_NAMES, rand_seed)
    pmo_lead = random_name(rand_seed + 3)

    # --- Generate project portfolio ---
    n_projects = rng.randint(4, 6)
    project_names = rng.sample(PORTFOLIO_PROJECT_NAMES, n_projects)
    priorities = list(range(1, n_projects + 1))
    rng.shuffle(priorities)

    # Timeline: 12-week window
    total_weeks = 12
    projects: list[dict] = []
    for i, (pname, priority) in enumerate(zip(project_names, priorities)):
        start_week = rng.randint(1, 4)
        end_week = rng.randint(start_week + 4, min(start_week + 10, total_weeks))
        budget = rng.randint(50000, 300000)

        # Generate 4-8 tasks per project
        n_proj_tasks = rng.randint(4, 8)
        proj_tasks: list[dict] = []
        for j in range(n_proj_tasks):
            task_start = rng.randint(start_week, max(start_week, end_week - 3))
            task_end = rng.randint(task_start + 1, min(task_start + 5, end_week))
            required_skill = rng.choice(SKILL_POOL[:12])  # use common skills
            hours_per_week = rng.choice([8, 12, 16, 20, 24, 32, 40])
            proj_tasks.append({
                "task_id": f"{pname[:3].upper()}-T{j + 1:02d}",
                "task_name": f"Task {j + 1}",
                "start_week": task_start,
                "end_week": task_end,
                "required_skill": required_skill,
                "hours_per_week": hours_per_week,
            })

        has_contractual_deadline = rng.random() < 0.4
        projects.append({
            "name": pname,
            "priority": priority,
            "start_week": start_week,
            "end_week": end_week,
            "budget": budget,
            "tasks": proj_tasks,
            "contractual_deadline": has_contractual_deadline,
        })

    # --- Generate resources ---
    n_resources = rng.randint(10, 16)
    res_names = random_names(rand_seed + 20, n_resources)

    resources: list[dict] = []
    for i, rname in enumerate(res_names):
        is_part_time = rng.random() < 0.2
        weekly_capacity = 20 if is_part_time else 40
        # Each resource has 2-4 skills
        n_skills = rng.randint(2, 4)
        skills = rng.sample(SKILL_POOL[:15], n_skills)
        hourly_rate = rng.randint(50, 150)
        resources.append({
            "name": rname,
            "weekly_capacity": weekly_capacity,
            "is_part_time": is_part_time,
            "skills": skills,
            "hourly_rate": hourly_rate,
        })

    # --- Create resource assignments ---
    assignments: list[dict] = []
    resource_weekly_hours: dict[str, dict[int, int]] = {
        r["name"]: {w: 0 for w in range(1, total_weeks + 1)} for r in resources
    }

    for proj in projects:
        for task in proj["tasks"]:
            # Find a resource with matching skill, or assign someone anyway
            matching_resources = [
                r for r in resources if task["required_skill"] in r["skills"]
            ]
            if matching_resources:
                assigned_resource = rng.choice(matching_resources)
            else:
                assigned_resource = rng.choice(resources)

            assignments.append({
                "project": proj["name"],
                "task_id": task["task_id"],
                "resource": assigned_resource["name"],
                "start_week": task["start_week"],
                "end_week": task["end_week"],
                "hours_per_week": task["hours_per_week"],
                "required_skill": task["required_skill"],
                "resource_has_skill": task["required_skill"] in assigned_resource["skills"],
            })

            for w in range(task["start_week"], task["end_week"] + 1):
                if w <= total_weeks:
                    resource_weekly_hours[assigned_resource["name"]][w] += task["hours_per_week"]

    # --- Plant additional over-allocations if needed ---
    # Ensure 3-5 resources are over-allocated in at least one week
    over_allocated_resources: list[dict] = []
    for r in resources:
        peak_week = max(range(1, total_weeks + 1), key=lambda w: resource_weekly_hours[r["name"]][w])
        peak_hours = resource_weekly_hours[r["name"]][peak_week]
        if peak_hours > r["weekly_capacity"]:
            over_pct = round((peak_hours / r["weekly_capacity"] - 1) * 100, 1)
            over_allocated_resources.append({
                "name": r["name"],
                "peak_week": peak_week,
                "peak_hours": peak_hours,
                "capacity": r["weekly_capacity"],
                "over_pct": over_pct,
            })

    # If fewer than 3 over-allocated, plant more
    attempts = 0
    while len(over_allocated_resources) < 3 and attempts < 20:
        attempts += 1
        target_res = rng.choice(resources)
        if target_res["name"] in [oa["name"] for oa in over_allocated_resources]:
            continue
        # Find a week where this resource has some hours but not too many
        candidate_weeks = [
            w for w in range(1, total_weeks + 1)
            if 0 < resource_weekly_hours[target_res["name"]][w] < target_res["weekly_capacity"]
        ]
        if not candidate_weeks:
            continue
        target_week = rng.choice(candidate_weeks)
        # Add an assignment from a random project that covers this week
        extra_hours = rng.randint(16, 28)
        extra_proj = rng.choice(projects)
        extra_task_id = f"{extra_proj['name'][:3].upper()}-TX{rng.randint(10,99)}"
        extra_skill = rng.choice(SKILL_POOL[:12])
        assignments.append({
            "project": extra_proj["name"],
            "task_id": extra_task_id,
            "resource": target_res["name"],
            "start_week": target_week,
            "end_week": min(target_week + 2, total_weeks),
            "hours_per_week": extra_hours,
            "required_skill": extra_skill,
            "resource_has_skill": extra_skill in target_res["skills"],
        })
        for w in range(target_week, min(target_week + 3, total_weeks + 1)):
            resource_weekly_hours[target_res["name"]][w] += extra_hours

        peak_week = max(range(1, total_weeks + 1), key=lambda w: resource_weekly_hours[target_res["name"]][w])
        peak_hours = resource_weekly_hours[target_res["name"]][peak_week]
        if peak_hours > target_res["weekly_capacity"]:
            over_pct = round((peak_hours / target_res["weekly_capacity"] - 1) * 100, 1)
            over_allocated_resources.append({
                "name": target_res["name"],
                "peak_week": peak_week,
                "peak_hours": peak_hours,
                "capacity": target_res["weekly_capacity"],
                "over_pct": over_pct,
            })

    # --- Identify skill mismatches ---
    skill_mismatches = [a for a in assignments if not a["resource_has_skill"]]

    # Ensure 2-3 skill mismatches exist
    if len(skill_mismatches) < 2:
        for _ in range(3 - len(skill_mismatches)):
            # Pick a random assignment and change its required skill to something the resource lacks
            candidates = [a for a in assignments if a["resource_has_skill"]]
            if not candidates:
                break
            target_assignment = rng.choice(candidates)
            res_obj = next(r for r in resources if r["name"] == target_assignment["resource"])
            unowned_skills = [s for s in SKILL_POOL if s not in res_obj["skills"]]
            if unowned_skills:
                new_skill = rng.choice(unowned_skills)
                target_assignment["required_skill"] = new_skill
                target_assignment["resource_has_skill"] = False
                skill_mismatches.append(target_assignment)

    # --- Compute weekly utilization for all resources ---
    weekly_utilization: dict[str, dict[int, float]] = {}
    for r in resources:
        weekly_utilization[r["name"]] = {}
        for w in range(1, total_weeks + 1):
            util = round(resource_weekly_hours[r["name"]][w] / r["weekly_capacity"] * 100, 1)
            weekly_utilization[r["name"]][w] = util

    # --- Peak over-allocation weeks ---
    peak_weeks: dict[int, int] = {}
    for w in range(1, total_weeks + 1):
        over_count = sum(
            1 for r in resources
            if resource_weekly_hours[r["name"]][w] > r["weekly_capacity"]
        )
        if over_count > 0:
            peak_weeks[w] = over_count

    worst_week = max(peak_weeks, key=peak_weeks.get) if peak_weeks else 1
    worst_week_count = peak_weeks.get(worst_week, 0)

    # --- Identify impacted projects ---
    impacted_projects: list[dict] = []
    for proj in projects:
        proj_resources = set(a["resource"] for a in assignments if a["project"] == proj["name"])
        proj_over_allocated = [
            oa for oa in over_allocated_resources if oa["name"] in proj_resources
        ]
        if proj_over_allocated:
            impacted_projects.append({
                "name": proj["name"],
                "priority": proj["priority"],
                "affected_resources": [oa["name"] for oa in proj_over_allocated],
            })

    # --- Scheduling conflicts: high-priority tasks competing for same resource ---
    scheduling_conflicts: list[dict] = []
    # Group assignments by resource
    by_resource: dict[str, list[dict]] = {}
    for a in assignments:
        by_resource.setdefault(a["resource"], []).append(a)

    for res_name, res_assignments in by_resource.items():
        for i, a1 in enumerate(res_assignments):
            for a2 in res_assignments[i + 1:]:
                # Check for time overlap
                overlap_start = max(a1["start_week"], a2["start_week"])
                overlap_end = min(a1["end_week"], a2["end_week"])
                if overlap_start <= overlap_end:
                    # Find project priorities
                    proj1 = next((p for p in projects if p["name"] == a1["project"]), None)
                    proj2 = next((p for p in projects if p["name"] == a2["project"]), None)
                    if proj1 and proj2 and proj1["name"] != proj2["name"]:
                        if proj1["priority"] <= 2 or proj2["priority"] <= 2:
                            scheduling_conflicts.append({
                                "resource": res_name,
                                "project1": a1["project"],
                                "priority1": proj1["priority"],
                                "task1": a1["task_id"],
                                "project2": a2["project"],
                                "priority2": proj2["priority"],
                                "task2": a2["task_id"],
                                "overlap_weeks": f"Week {overlap_start}-{overlap_end}",
                            })

    # Deduplicate scheduling conflicts by resource-project pair
    seen_conflicts: set[tuple] = set()
    unique_conflicts: list[dict] = []
    for sc in scheduling_conflicts:
        key = (sc["resource"], tuple(sorted([sc["project1"], sc["project2"]])))
        if key not in seen_conflicts:
            seen_conflicts.add(key)
            unique_conflicts.append(sc)
    scheduling_conflicts = unique_conflicts[:6]

    # --- False positive: a resource at ~95% utilization that is within acceptable range ---
    false_positive_resources: list[dict] = []
    for r in resources:
        for w in range(1, total_weeks + 1):
            util = weekly_utilization[r["name"]][w]
            if 90 <= util <= 100:
                is_actually_overallocated = r["name"] in [oa["name"] for oa in over_allocated_resources]
                if not is_actually_overallocated:
                    false_positive_resources.append({
                        "name": r["name"],
                        "week": w,
                        "utilization": util,
                        "hours": resource_weekly_hours[r["name"]][w],
                        "capacity": r["weekly_capacity"],
                    })
                    break
    false_positive_resources = false_positive_resources[:2]

    # --- Total resource cost ---
    total_resource_cost = 0.0
    for a in assignments:
        res_obj = next(r for r in resources if r["name"] == a["resource"])
        weeks = a["end_week"] - a["start_week"] + 1
        total_resource_cost += a["hours_per_week"] * weeks * res_obj["hourly_rate"]
    total_resource_cost = round(total_resource_cost, 2)

    # --- Build project_portfolio.csv ---
    portfolio_lines = ["ProjectName,Priority,StartWeek,EndWeek,Budget,ContractualDeadline,TaskCount"]
    for proj in projects:
        deadline_str = "Yes" if proj["contractual_deadline"] else "No"
        portfolio_lines.append(
            f"{proj['name']},{proj['priority']},{proj['start_week']},{proj['end_week']},"
            f"{proj['budget']},{deadline_str},{len(proj['tasks'])}"
        )
    portfolio_csv = "\n".join(portfolio_lines) + "\n"

    # --- Build resource_assignments.csv ---
    assign_lines = ["Project,TaskID,Resource,StartWeek,EndWeek,HoursPerWeek,RequiredSkill"]
    for a in assignments:
        assign_lines.append(
            f"{a['project']},{a['task_id']},{a['resource']},{a['start_week']},"
            f"{a['end_week']},{a['hours_per_week']},{a['required_skill']}"
        )
    assignments_csv = "\n".join(assign_lines) + "\n"

    # --- Build resource_capacity.csv ---
    capacity_lines = ["ResourceName,WeeklyCapacityHours,HourlyRate,Skills"]
    for r in resources:
        skills_str = ";".join(r["skills"])
        capacity_lines.append(
            f"{r['name']},{r['weekly_capacity']},{r['hourly_rate']},{skills_str}"
        )
    capacity_csv = "\n".join(capacity_lines) + "\n"

    # --- Build prioritization_criteria.txt ---
    priority_lines = [
        "RESOURCE CONFLICT PRIORITIZATION CRITERIA",
        "",
        "=" * 60,
        "",
        "When resources are over-allocated and conflicts must be resolved,",
        "apply the following rules in order:",
        "",
        "1. PROJECT PRIORITY: Lower priority number = higher importance.",
        "   Priority 1 is the most important project.",
        "   Always protect higher-priority project schedules first.",
        "",
        "2. CONTRACTUAL OBLIGATIONS: Projects with contractual deadlines",
        "   take precedence over internal-deadline projects at the same",
        "   priority level. Missing a contractual deadline incurs penalties.",
        "",
        "3. DEADLINE PROXIMITY: When two projects have the same priority",
        "   and neither has contractual obligations, the one with the",
        "   nearer deadline takes precedence.",
        "",
        "4. SKILL MATCH: Assign resources to tasks matching their skill set.",
        "   If a resource lacks the required skill, flag for reassignment",
        "   or training. Mismatched assignments reduce productivity by ~30%.",
        "",
        "5. UTILIZATION TARGET: Target 80-90% utilization per resource.",
        "   Resources above 100% must be resolved. Resources below 60%",
        "   may be reassigned. Resources at 90-100% are acceptable.",
        "",
        "ACCEPTABLE OVER-ALLOCATION:",
        "  Resources may be at 100% utilization without being considered",
        "  over-allocated. Only assignments exceeding weekly capacity hours",
        "  constitute an over-allocation requiring resolution.",
        "",
    ]
    priority_content = "\n".join(priority_lines) + "\n"

    # --- Problem statement ---
    problem_statement = f"""# Resource Leveling Analysis

You are a PMO (Project Management Office) analyst for {company}. The PMO lead
({pmo_lead}) needs an analysis of resource allocation across {n_projects} concurrent
projects over a {total_weeks}-week period.

## Source Files
- /testbed/data/project_portfolio.csv - Project list with priorities, timelines, budgets
- /testbed/data/resource_assignments.csv - Resource-to-task assignments across all projects
- /testbed/data/resource_capacity.csv - Each resource's weekly capacity and skills
- /testbed/data/prioritization_criteria.txt - Rules for resolving conflicts

## Requirements
1. Compute weekly utilization (hours assigned / capacity) for each resource across all weeks
2. Identify over-allocated resources (assigned > 100% capacity) and the specific weeks affected
3. Identify the peak over-allocation week(s) and how many resources are affected
4. Identify skill mismatches (resource assigned to task requiring a skill they lack)
5. Identify scheduling conflicts where high-priority project tasks compete for the same resource
6. Determine which projects are impacted by resource constraints
7. Recommend specific reallocation actions to resolve over-allocations
8. Compute total resource cost across all assignments

Write a detailed resource leveling analysis to /testbed/resource_report.txt."""

    # --- Build rubric ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does the file /testbed/resource_report.txt exist and contain substantive content (more than a few lines)?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_resource_count",
            question=f"Does the report correctly identify or work with {n_resources} resources?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_project_count",
            question=f"Does the report correctly identify {n_projects} concurrent projects?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_over_allocated_count",
            question=(
                f"Does the report correctly identify {len(over_allocated_resources)} over-allocated "
                f"resources (those assigned more hours than their weekly capacity in at least one week)? "
                f"The over-allocated resources are: "
                f"{', '.join(oa['name'] for oa in over_allocated_resources)}."
            ),
            points=2,
        ),
    ]

    # Per-resource over-allocation checks
    for oa in over_allocated_resources[:3]:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"overalloc_{oa['name'].replace(' ', '_')}",
                question=(
                    f"Does the report identify {oa['name']} as over-allocated, with peak "
                    f"utilization in Week {oa['peak_week']} at {oa['peak_hours']} hours "
                    f"against a capacity of {oa['capacity']} hours ({oa['over_pct']}% over)?"
                ),
                points=2,
            )
        )

    # Peak week
    rubric_items.append(
        BinaryRubricCategory(
            name="correct_peak_week",
            question=(
                f"Does the report identify Week {worst_week} as the peak over-allocation week "
                f"(or one of the worst weeks) with {worst_week_count} over-allocated resource(s)?"
            ),
            points=2,
        )
    )

    # Skill mismatch detection
    if skill_mismatches:
        mismatch_descs = []
        for sm in skill_mismatches[:3]:
            mismatch_descs.append(
                f"{sm['resource']} assigned to {sm['task_id']} requiring {sm['required_skill']}"
            )
        rubric_items.append(
            BinaryRubricCategory(
                name="identifies_skill_mismatches",
                question=(
                    f"Does the report identify skill mismatches? There are {len(skill_mismatches)} "
                    f"mismatches, including: {'; '.join(mismatch_descs)}."
                ),
                points=2,
            )
        )

    # Scheduling conflict detection
    if scheduling_conflicts:
        for sc in scheduling_conflicts[:2]:
            rubric_items.append(
                BinaryRubricCategory(
                    name=f"conflict_{sc['resource'].replace(' ', '_')}_{sc['project1'][:3]}_{sc['project2'][:3]}",
                    question=(
                        f"Does the report identify the scheduling conflict where {sc['resource']} "
                        f"is assigned to both {sc['project1']} (priority {sc['priority1']}) and "
                        f"{sc['project2']} (priority {sc['priority2']}) during {sc['overlap_weeks']}?"
                    ),
                    points=2,
                )
            )

    # Impacted projects
    if impacted_projects:
        impacted_names = ", ".join(f"{ip['name']} (priority {ip['priority']})" for ip in impacted_projects[:3])
        rubric_items.append(
            BinaryRubricCategory(
                name="identifies_impacted_projects",
                question=(
                    f"Does the report identify projects impacted by resource constraints? "
                    f"Impacted projects include: {impacted_names}."
                ),
                points=2,
            )
        )

    # Total resource cost
    rubric_items.append(
        BinaryRubricCategory(
            name="correct_total_cost",
            question=(
                f"Does the report compute the total resource cost across all assignments as "
                f"approximately {_fmt_money(total_resource_cost)} (within 10%)?"
            ),
            points=2,
        )
    )

    # False positive checks: resource at ~95% that is NOT over-allocated
    for fp in false_positive_resources[:2]:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"no_false_overalloc_{fp['name'].replace(' ', '_')}",
                question=(
                    f"Does the report correctly NOT flag {fp['name']} as over-allocated in "
                    f"Week {fp['week']}? Their utilization is {fp['utilization']}% "
                    f"({fp['hours']}h / {fp['capacity']}h capacity), which is within the "
                    f"acceptable range per the prioritization criteria."
                ),
                points=1,
            )
        )

    # Recommendation quality
    rubric_items.append(
        BinaryRubricCategory(
            name="has_reallocation_recommendations",
            question=(
                "Does the report provide specific reallocation recommendations that reference "
                "actual resource names, project names, and weeks to resolve over-allocations?"
            ),
            points=2,
        )
    )

    rubric_items.append(
        RubricCategory(
            name="analysis_quality",
            description="How thorough and actionable is the resource leveling analysis?",
            failure="Missing utilization calculations or no meaningful conflict identification.",
            minor_failure="Some utilization data computed but missing key conflicts or recommendations.",
            minor_success="Good utilization analysis with most conflicts identified and reasonable recommendations.",
            success="Comprehensive analysis with per-resource weekly utilization, all conflicts identified, prioritized recommendations, and cost analysis.",
            points=2,
        )
    )

    rubric = tuple(rubric_items)

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write your completed resource leveling analysis to /testbed/resource_report.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/project_portfolio.csv": portfolio_csv,
            "/testbed/data/resource_assignments.csv": assignments_csv,
            "/testbed/data/resource_capacity.csv": capacity_csv,
            "/testbed/data/prioritization_criteria.txt": priority_content,
        },
        problem_type="resource_leveling_analysis",
    )
