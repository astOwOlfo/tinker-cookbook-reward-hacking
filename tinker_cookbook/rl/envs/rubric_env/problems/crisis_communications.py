"""Crisis communications and stakeholder analysis tasks.

Three seedable factories covering:
1. Crisis response audit — timeline compliance checking
2. Stakeholder impact assessment — multi-dimensional impact scoring
3. Communications timeline analysis — business dispute reconstruction

Each factory generates realistic multi-document scenarios with planted violations
that the model must discover through cross-referencing. Seeds control crisis types,
violation mixes, stakeholder compositions, and dispute structures.
"""

from __future__ import annotations

import random as _random

from ..dataset import RubricDatapoint, RubricCategory, BinaryRubricCategory
from tinker_cookbook.rl.envs import tools
from ..content_pools import random_name, random_names, pick1, COMPANY_NAMES


# =============================================================================
# HELPERS
# =============================================================================


def _ts(year: int, month: int, day: int, hour: int, minute: int) -> str:
    return f"{year}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}"


def _date(year: int, month: int, day: int) -> str:
    return f"{year}-{month:02d}-{day:02d}"


def _add_hours(year: int, month: int, day: int, hour: int, minute: int,
               delta_hours: float) -> tuple[int, int, int, int, int]:
    """Add delta_hours to a timestamp, returning (y, m, d, h, min)."""
    total_min = hour * 60 + minute + int(delta_hours * 60)
    extra_days = total_min // (24 * 60)
    remaining_min = total_min % (24 * 60)
    new_day = day + extra_days
    new_month = month
    new_year = year
    # Simple month overflow (use 28-day months for simplicity)
    while new_day > 28:
        new_day -= 28
        new_month += 1
    while new_month > 12:
        new_month -= 12
        new_year += 1
    new_hour = remaining_min // 60
    new_min = remaining_min % 60
    return new_year, new_month, new_day, new_hour, new_min


def _fmt_datetime_nice(y: int, m: int, d: int, h: int, mi: int) -> str:
    """Human-readable datetime."""
    months = ["", "January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    ampm = "AM" if h < 12 else "PM"
    display_h = h % 12 or 12
    return f"{months[m]} {d}, {y} at {display_h}:{mi:02d} {ampm}"


# =============================================================================
# 1. CRISIS RESPONSE AUDIT
# =============================================================================

CRISIS_SCENARIOS = [
    {
        "key": "data_breach",
        "label": "Customer Data Breach",
        "description": "Unauthorized access to customer database containing PII of approximately 50,000 customers",
        "severity": "Critical",
        "regulatory_bodies": ["State Attorney General", "FTC", "EU DPA (GDPR)"],
    },
    {
        "key": "product_recall",
        "label": "Product Safety Recall",
        "description": "Manufacturing defect in consumer product causing potential injury risk",
        "severity": "High",
        "regulatory_bodies": ["CPSC", "State Consumer Protection"],
    },
    {
        "key": "environmental_spill",
        "label": "Chemical Spill",
        "description": "Accidental release of industrial chemicals into nearby waterway",
        "severity": "Critical",
        "regulatory_bodies": ["EPA", "State DEQ", "Local Emergency Management"],
    },
    {
        "key": "financial_restatement",
        "label": "Financial Restatement",
        "description": "Discovery of material accounting errors requiring restatement of prior quarters",
        "severity": "High",
        "regulatory_bodies": ["SEC", "PCAOB"],
    },
    {
        "key": "workplace_incident",
        "label": "Workplace Safety Incident",
        "description": "Serious workplace accident resulting in multiple employee injuries",
        "severity": "High",
        "regulatory_bodies": ["OSHA", "State Workers Comp Board"],
    },
    {
        "key": "ransomware",
        "label": "Ransomware Attack",
        "description": "Ransomware encryption of critical business systems and potential data exfiltration",
        "severity": "Critical",
        "regulatory_bodies": ["FBI/CISA", "State Attorney General", "EU DPA (GDPR)"],
    },
    {
        "key": "executive_misconduct",
        "label": "Executive Misconduct Allegation",
        "description": "Credible allegations of financial misconduct by a senior executive",
        "severity": "High",
        "regulatory_bodies": ["SEC", "DOJ"],
    },
    {
        "key": "supply_chain_failure",
        "label": "Critical Supply Chain Disruption",
        "description": "Major supplier bankruptcy causing disruption to key product lines",
        "severity": "Medium",
        "regulatory_bodies": ["FTC (if consumer impact)", "Industry Regulator"],
    },
]

# Violation types for crisis response
VIOLATION_TYPES = [
    "missed_board_notification",
    "inconsistent_messaging",
    "missed_regulatory_notification",
    "premature_disclosure",
    "no_status_update",
    "wrong_channel",
    "unauthorized_spokesperson",
]

# Communication channels
CHANNELS = [
    "internal_email", "press_release", "social_media_post",
    "regulatory_filing", "board_memo", "customer_notification",
    "employee_town_hall", "investor_call",
]

SPOKESPERSON_TITLES = [
    "CEO", "CFO", "CTO", "VP of Communications", "General Counsel",
    "VP of Operations", "Head of Investor Relations", "CISO",
    "Head of Customer Success", "Director of Public Relations",
]


def make_crisis_response_audit(rand_seed: int = 42) -> RubricDatapoint:
    """Audit a company's crisis response timeline for communication failures.

    Seed varies: crisis type, company, violation mix (3-6 planted), notification
    windows, stakeholder communications, and false-positive traps (2-3).
    """
    rng = _random.Random(rand_seed)

    company = pick1(COMPANY_NAMES, rand_seed)
    scenario = rng.choice(CRISIS_SCENARIOS)
    crisis_key = scenario["key"]

    # Names for people involved
    names = random_names(rand_seed, 12)
    ceo_name = names[0]
    cfo_name = names[1]
    cto_name = names[2]
    gc_name = names[3]  # General Counsel
    comms_vp = names[4]
    ciso_name = names[5]
    board_chair = names[6]
    pr_director = names[7]
    ir_head = names[8]
    ops_vp = names[9]

    # Authorized spokespersons per policy
    authorized_spokespersons = {
        "Critical": [ceo_name, gc_name, comms_vp],
        "High": [ceo_name, cfo_name, gc_name, comms_vp],
        "Medium": [comms_vp, pr_director, ops_vp],
    }
    severity = scenario["severity"]

    # Timeline base: incident occurs in 2024
    base_y, base_m, base_d = 2024, rng.randint(1, 10), rng.randint(5, 25)
    discover_h, discover_min = rng.randint(6, 14), rng.randint(0, 59)

    # Policy notification windows (hours)
    policy_board_window = 2         # hours after discovery
    policy_executive_window = 1     # hours after discovery
    policy_regulatory_window = 72   # hours after confirmation
    policy_public_window = 24       # hours after confirmation for Critical
    policy_employee_window = 12     # hours after public disclosure
    policy_investor_window = 4      # hours after public disclosure

    # Confirmation happens some hours after discovery
    confirm_delta = rng.uniform(2, 8)
    confirm_ts = _add_hours(base_y, base_m, base_d, discover_h, discover_min, confirm_delta)

    # Holiday extension: one specific date range where deadlines extend by 24h
    holiday_start_m, holiday_start_d = rng.choice([(1, 1), (7, 3), (12, 24), (11, 23)])
    holiday_end_m, holiday_end_d = holiday_start_m, min(28, holiday_start_d + 2)
    holiday_year = base_y

    # Check if incident falls during holiday
    during_holiday = (base_m == holiday_start_m and
                      holiday_start_d <= base_d <= holiday_end_d)

    # If during holiday, extend board notification window
    effective_board_window = policy_board_window + (24 if during_holiday else 0)

    # --- Decide which violations to plant ---
    n_violations = rng.randint(3, 6)
    available_violations = list(VIOLATION_TYPES)
    rng.shuffle(available_violations)
    planted_violations = available_violations[:n_violations]

    violations_detail: list[dict] = []
    # Track actual times for building timeline
    timeline_events: list[dict] = []

    # Always add: discovery event
    timeline_events.append({
        "time": (base_y, base_m, base_d, discover_h, discover_min),
        "event": "issue_discovered",
        "description": f"IT/Operations team identified the issue: {scenario['description']}",
        "actor": cto_name if crisis_key in ("data_breach", "ransomware") else ops_vp,
    })

    # Executive notification - always happens (sometimes late)
    exec_notify_delta = rng.uniform(0.5, 1.5)
    exec_notify_ts = _add_hours(base_y, base_m, base_d, discover_h, discover_min, exec_notify_delta)
    timeline_events.append({
        "time": exec_notify_ts,
        "event": "executive_notified",
        "description": f"CEO {ceo_name} and executive team notified via phone call",
        "actor": cto_name if crisis_key in ("data_breach", "ransomware") else ops_vp,
    })

    # Confirmation event
    timeline_events.append({
        "time": confirm_ts,
        "event": "incident_confirmed",
        "description": f"Incident confirmed after investigation: {scenario['label']}",
        "actor": gc_name,
    })

    # Board notification
    if "missed_board_notification" in planted_violations:
        # Board notified late (after the effective window)
        board_delta = effective_board_window + rng.uniform(2, 12)
        board_ts = _add_hours(base_y, base_m, base_d, discover_h, discover_min, board_delta)
        violations_detail.append({
            "type": "missed_board_notification",
            "description": (
                f"Board of Directors was notified {board_delta:.1f} hours after discovery, "
                f"exceeding the {effective_board_window}-hour policy window"
                + (" (including holiday extension)" if during_holiday else "")
            ),
        })
    else:
        # Board notified within window
        board_delta = rng.uniform(0.5, effective_board_window - 0.5)
        board_ts = _add_hours(base_y, base_m, base_d, discover_h, discover_min, board_delta)
    timeline_events.append({
        "time": board_ts,
        "event": "board_notified",
        "description": f"Board chair {board_chair} notified via secure board communication channel",
        "actor": ceo_name,
    })

    # --- Build actual communications with violations ---

    # Press release content
    if crisis_key == "data_breach":
        pr_detail = "customer data may have been accessed"
        internal_detail_correct = "unauthorized access to customer database confirmed"
        internal_detail_wrong = "no confirmed data exfiltration at this time"
    elif crisis_key == "ransomware":
        pr_detail = "experienced a cybersecurity incident affecting some systems"
        internal_detail_correct = "ransomware attack encrypted critical systems; data exfiltration suspected"
        internal_detail_wrong = "minor system disruption; no data compromise expected"
    elif crisis_key == "product_recall":
        pr_detail = "voluntarily recalling product due to potential safety concern"
        internal_detail_correct = "manufacturing defect confirmed; recall initiated"
        internal_detail_wrong = "investigating isolated quality reports; no systemic issue confirmed"
    elif crisis_key == "environmental_spill":
        pr_detail = "working with authorities on environmental remediation"
        internal_detail_correct = "chemical release into waterway confirmed; EPA notified"
        internal_detail_wrong = "contained spill; no environmental impact expected"
    elif crisis_key == "financial_restatement":
        pr_detail = "identified adjustments requiring restatement of prior financial results"
        internal_detail_correct = "material accounting errors found in revenue recognition"
        internal_detail_wrong = "minor reclassification adjustments; immaterial impact"
    elif crisis_key == "workplace_incident":
        pr_detail = "experienced a workplace safety incident; cooperating with authorities"
        internal_detail_correct = "multiple employees injured; OSHA investigation expected"
        internal_detail_wrong = "minor workplace incident; all employees receiving care"
    elif crisis_key == "executive_misconduct":
        pr_detail = "initiated an independent investigation into certain matters"
        internal_detail_correct = "credible allegations of financial misconduct by senior executive"
        internal_detail_wrong = "routine compliance review in progress"
    else:
        pr_detail = "managing a significant business disruption"
        internal_detail_correct = "critical supply chain failure confirmed"
        internal_detail_wrong = "temporary supply delays; alternative sources identified"

    # Public disclosure
    if "premature_disclosure" in planted_violations:
        # Public disclosure BEFORE internal escalation is complete
        public_delta_from_discover = rng.uniform(0.5, 1.5)
        public_ts = _add_hours(base_y, base_m, base_d, discover_h, discover_min, public_delta_from_discover)
        violations_detail.append({
            "type": "premature_disclosure",
            "description": (
                f"Public press release issued {public_delta_from_discover:.1f} hours after discovery, "
                f"before incident was confirmed and before board notification"
            ),
        })
    else:
        # Public disclosure after confirmation (within policy window)
        public_delta_from_confirm = rng.uniform(4, policy_public_window - 2)
        public_ts = _add_hours(*confirm_ts, public_delta_from_confirm)
    timeline_events.append({
        "time": public_ts,
        "event": "press_release",
        "description": f"Press release issued: {company} {pr_detail}",
        "actor": comms_vp,
    })

    # Inconsistent messaging
    inconsistent_messaging_present = "inconsistent_messaging" in planted_violations
    if inconsistent_messaging_present:
        # Internal memo contradicts press release
        internal_memo_detail = internal_detail_wrong
        violations_detail.append({
            "type": "inconsistent_messaging",
            "description": (
                f"Internal employee memo states '{internal_detail_wrong}' while "
                f"press release states '{pr_detail}' — contradictory messaging"
            ),
        })
    else:
        internal_memo_detail = internal_detail_correct

    # Internal memo time (after public disclosure typically)
    internal_memo_delta = rng.uniform(1, 6)
    internal_memo_ts = _add_hours(*public_ts, internal_memo_delta)
    timeline_events.append({
        "time": internal_memo_ts,
        "event": "employee_notification",
        "description": f"Internal employee memo distributed: {internal_memo_detail}",
        "actor": comms_vp,
    })

    # Regulatory notification
    if "missed_regulatory_notification" in planted_violations:
        # Pick one regulatory body that was NOT notified
        missed_reg = rng.choice(scenario["regulatory_bodies"])
        notified_regs = [r for r in scenario["regulatory_bodies"] if r != missed_reg]
        violations_detail.append({
            "type": "missed_regulatory_notification",
            "description": (
                f"{missed_reg} was not notified within the {policy_regulatory_window}-hour window "
                f"(required regulatory notification missing)"
            ),
        })
    else:
        missed_reg = None
        notified_regs = list(scenario["regulatory_bodies"])

    # Regulatory filings that DID happen
    reg_filings: list[dict] = []
    for reg in notified_regs:
        reg_delta = rng.uniform(12, policy_regulatory_window - 8)
        reg_ts = _add_hours(*confirm_ts, reg_delta)
        timeline_events.append({
            "time": reg_ts,
            "event": "regulatory_filing",
            "description": f"Regulatory notification filed with {reg}",
            "actor": gc_name,
        })
        reg_filings.append({"body": reg, "time": reg_ts})

    # Status update violation
    if "no_status_update" in planted_violations:
        # After a status change, no follow-up communication
        status_change_delta = rng.uniform(24, 48)
        status_change_ts = _add_hours(*confirm_ts, status_change_delta)
        timeline_events.append({
            "time": status_change_ts,
            "event": "status_change",
            "description": "Investigation status changed: additional systems/data found affected",
            "actor": cto_name,
        })
        violations_detail.append({
            "type": "no_status_update",
            "description": (
                "After investigation found additional affected systems/data, "
                "no follow-up stakeholder communication was issued within 12 hours"
            ),
        })

    # Wrong channel violation
    if "wrong_channel" in planted_violations:
        # Critical severity message sent via social media instead of press release
        social_delta = rng.uniform(2, 8)
        social_ts = _add_hours(*confirm_ts, social_delta)
        wrong_channel_msg = rng.choice([
            "initial public disclosure via Twitter/X post",
            "regulatory-sensitive information shared on company LinkedIn",
            "customer notification sent via social media DM instead of direct email",
        ])
        timeline_events.append({
            "time": social_ts,
            "event": "social_media_post",
            "description": f"Social media post: {wrong_channel_msg}",
            "actor": pr_director,
        })
        violations_detail.append({
            "type": "wrong_channel",
            "description": (
                f"For a {severity}-severity incident, {wrong_channel_msg} — "
                f"policy requires formal press release and direct stakeholder notification"
            ),
        })

    # Unauthorized spokesperson
    if "unauthorized_spokesperson" in planted_violations:
        unauthorized_person = rng.choice([n for n in names[5:] if n not in authorized_spokespersons[severity]])
        unauth_delta = rng.uniform(6, 24)
        unauth_ts = _add_hours(*confirm_ts, unauth_delta)
        timeline_events.append({
            "time": unauth_ts,
            "event": "media_interview",
            "description": f"{unauthorized_person} gave media interview regarding the incident",
            "actor": unauthorized_person,
        })
        violations_detail.append({
            "type": "unauthorized_spokesperson",
            "description": (
                f"{unauthorized_person} spoke to media but is not an authorized spokesperson "
                f"for {severity}-severity incidents (authorized: {', '.join(authorized_spokespersons[severity])})"
            ),
        })

    # Sort timeline
    timeline_events.sort(key=lambda e: e["time"])

    # --- FALSE POSITIVES ---
    false_positives: list[dict] = []

    # FP1: Board notification that looks late but falls within holiday extension
    if during_holiday and "missed_board_notification" not in planted_violations:
        false_positives.append({
            "name": "board_notification_holiday",
            "description": (
                f"Board notification at {_fmt_datetime_nice(*board_ts)} appears to exceed the "
                f"standard {policy_board_window}-hour window, but the holiday extension "
                f"({_date(holiday_year, holiday_start_m, holiday_start_d)} to "
                f"{_date(holiday_year, holiday_end_m, holiday_end_d)}) extends the deadline "
                f"to {effective_board_window} hours"
            ),
        })

    # FP2: A regulatory filing that appears late but was actually within window
    close_reg_delta = policy_regulatory_window - rng.uniform(1, 6)
    fp_reg_body = rng.choice(["Industry Self-Regulatory Body", "State Insurance Commissioner", "Trade Association"])
    fp_reg_ts = _add_hours(*confirm_ts, close_reg_delta)
    false_positives.append({
        "name": "close_regulatory_filing",
        "description": (
            f"Notification to {fp_reg_body} at {_fmt_datetime_nice(*fp_reg_ts)} was within "
            f"the {policy_regulatory_window}-hour window (filed {close_reg_delta:.1f} hours after confirmation), "
            f"though close to the deadline"
        ),
    })
    timeline_events.append({
        "time": fp_reg_ts,
        "event": "regulatory_filing",
        "description": f"Notification filed with {fp_reg_body}",
        "actor": gc_name,
    })

    # FP3: Employee communication that looks late relative to public disclosure
    # but policy says "12 hours after public disclosure" and it's within
    emp_delta = rng.uniform(8, 11.5)  # Close to 12h but within
    emp_fp_ts = _add_hours(*public_ts, emp_delta)
    false_positives.append({
        "name": "employee_comm_timing",
        "description": (
            f"Employee town hall {emp_delta:.1f} hours after public disclosure appears late "
            f"but is within the 12-hour policy window"
        ),
    })
    timeline_events.append({
        "time": emp_fp_ts,
        "event": "employee_town_hall",
        "description": f"Employee town hall held to discuss incident status and next steps",
        "actor": ceo_name,
    })

    # Sort timeline again after adding FP events
    timeline_events.sort(key=lambda e: e["time"])

    # --- BUILD FILES ---

    # 1. incident_timeline.csv
    csv_lines = ["timestamp,event_type,description,actor"]
    for evt in timeline_events:
        ts_str = _ts(*evt["time"])
        # No violation labels in the CSV - just facts
        csv_lines.append(f'{ts_str},"{evt["event"]}","{evt["description"]}","{evt["actor"]}"')
    incident_timeline_csv = "\n".join(csv_lines) + "\n"

    # 2. communication_policy.txt
    policy_lines = [
        f"CRISIS COMMUNICATION POLICY — {company}",
        "",
        "=" * 60,
        "1. SEVERITY CLASSIFICATION",
        "=" * 60,
        "",
        "Critical: Data breach, ransomware, environmental release, or any event",
        "  requiring immediate regulatory notification or posing material risk.",
        "High: Product recall, financial restatement, workplace incident, or",
        "  executive misconduct. Significant but containable.",
        "Medium: Supply chain disruption, service outage, or reputational issue",
        "  with limited direct stakeholder impact.",
        "",
        "=" * 60,
        "2. NOTIFICATION WINDOWS (from time of discovery unless stated otherwise)",
        "=" * 60,
        "",
        f"Executive team notification: within {policy_executive_window} hour(s) of discovery",
        f"Board of Directors notification: within {policy_board_window} hour(s) of discovery",
        f"Regulatory bodies: within {policy_regulatory_window} hours of incident confirmation",
        f"Public disclosure (Critical/High): within {policy_public_window} hours of confirmation",
        f"Employee notification: within {policy_employee_window} hours of public disclosure",
        f"Investor/analyst notification: within {policy_investor_window} hours of public disclosure",
        "",
        "HOLIDAY EXTENSION: During recognized company holidays, board notification",
        f"  window extends by 24 hours. Holiday dates for {holiday_year}:",
        f"  {_date(holiday_year, holiday_start_m, holiday_start_d)} through "
        f"{_date(holiday_year, holiday_end_m, holiday_end_d)}",
        "",
        "=" * 60,
        "3. AUTHORIZED SPOKESPERSONS BY SEVERITY",
        "=" * 60,
        "",
    ]
    for sev, people in authorized_spokespersons.items():
        policy_lines.append(f"  {sev}: {', '.join(people)}")
    policy_lines.extend([
        "",
        "Only authorized spokespersons may make external statements (press,",
        "media interviews, social media posts) regarding the incident.",
        "",
        "=" * 60,
        "4. COMMUNICATION CHANNELS BY SEVERITY",
        "=" * 60,
        "",
        "Critical/High severity initial public disclosure: formal press release",
        "  ONLY. Social media may be used for follow-up but NOT initial disclosure.",
        "Medium severity: press release or formal statement acceptable.",
        "",
        "All regulatory notifications: official filing or registered letter.",
        "Customer notifications: direct email or registered mail.",
        "Internal communications: company email + town hall within policy window.",
        "",
        "=" * 60,
        "5. STATUS UPDATE REQUIREMENTS",
        "=" * 60,
        "",
        "After any material change in incident status (new findings, expanded",
        "scope, resolution), an updated communication must be issued to all",
        "previously notified stakeholders within 12 hours.",
        "",
        "=" * 60,
        "6. MESSAGING CONSISTENCY",
        "=" * 60,
        "",
        "All external and internal communications regarding an incident must",
        "be consistent in their characterization of the event. Internal memos",
        "must not contradict public statements. All communications must be",
        "reviewed by General Counsel before distribution.",
        "",
    ])
    communication_policy = "\n".join(policy_lines) + "\n"

    # 3. stakeholder_communications.txt
    comm_lines = [
        f"STAKEHOLDER COMMUNICATIONS LOG — {company}",
        f"Incident: {scenario['label']}",
        "",
        "=" * 60,
    ]

    for evt in timeline_events:
        ts_str = _ts(*evt["time"])
        nice_ts = _fmt_datetime_nice(*evt["time"])
        etype = evt["event"]

        if etype == "press_release":
            comm_lines.extend([
                "",
                f"--- PRESS RELEASE ---",
                f"Date/Time: {nice_ts}",
                f"Issued by: {evt['actor']}, VP of Communications",
                "",
                f"FOR IMMEDIATE RELEASE",
                "",
                f"{company} Statement Regarding {scenario['label']}",
                "",
                f"{company} today announced that it {pr_detail}. The company is taking "
                f"immediate steps to address the situation and is cooperating with "
                f"relevant authorities.",
                "",
                f"\"We take this matter seriously and are committed to transparency,\" "
                f"said {ceo_name}, CEO of {company}. \"We are working diligently to "
                f"address this situation and will provide updates as appropriate.\"",
                "",
                f"Contact: {comms_vp}, VP of Communications",
                "",
            ])
        elif etype == "employee_notification":
            comm_lines.extend([
                "",
                f"--- INTERNAL EMPLOYEE MEMO ---",
                f"Date/Time: {nice_ts}",
                f"From: {evt['actor']}",
                f"To: All Employees",
                f"Subject: Update on Current Situation",
                "",
                f"Dear Team,",
                "",
                f"I want to share an update regarding a matter affecting our company. "
                f"As some of you may be aware, we have been dealing with an incident "
                f"involving {internal_memo_detail}.",
                "",
                f"We are working closely with our legal team and relevant authorities. "
                f"Please direct any media inquiries to the Communications team.",
                "",
                f"Thank you for your patience and professionalism.",
                "",
                f"{evt['actor']}",
                "",
            ])
        elif etype == "social_media_post":
            comm_lines.extend([
                "",
                f"--- SOCIAL MEDIA POST ---",
                f"Platform: Twitter/X",
                f"Date/Time: {nice_ts}",
                f"Posted by: {evt['actor']}",
                "",
                f"@{company.replace(' ', '')}: {evt['description'].split(': ', 1)[-1] if ': ' in evt['description'] else evt['description']}",
                "",
            ])
        elif etype == "regulatory_filing":
            comm_lines.extend([
                "",
                f"--- REGULATORY NOTIFICATION ---",
                f"Date/Time: {nice_ts}",
                f"Filed by: {evt['actor']}",
                f"Filed with: {evt['description'].split('with ')[-1]}",
                f"Method: Official filing via registered portal",
                "",
            ])
        elif etype == "media_interview":
            comm_lines.extend([
                "",
                f"--- MEDIA INTERVIEW ---",
                f"Date/Time: {nice_ts}",
                f"Interviewee: {evt['actor']}",
                f"Outlet: Regional Business Journal",
                f"Topic: Company response to {scenario['label']}",
                "",
            ])
        elif etype == "employee_town_hall":
            comm_lines.extend([
                "",
                f"--- EMPLOYEE TOWN HALL ---",
                f"Date/Time: {nice_ts}",
                f"Led by: {evt['actor']}",
                f"Attendees: All employees (virtual + in-person)",
                f"Topic: Incident update and Q&A",
                "",
            ])
        elif etype == "board_notified":
            comm_lines.extend([
                "",
                f"--- BOARD COMMUNICATION ---",
                f"Date/Time: {nice_ts}",
                f"From: {evt['actor']}",
                f"To: Board of Directors (via secure board portal)",
                f"Subject: Urgent — Incident Notification",
                "",
            ])
        elif etype == "investor_call":
            comm_lines.extend([
                "",
                f"--- INVESTOR NOTIFICATION ---",
                f"Date/Time: {nice_ts}",
                f"Led by: {evt['actor']}",
                f"Attendees: Major institutional investors, analysts",
                "",
            ])

    stakeholder_communications = "\n".join(comm_lines) + "\n"

    # 4. regulatory_requirements.txt
    reg_lines = [
        "REGULATORY NOTIFICATION REQUIREMENTS BY JURISDICTION AND INCIDENT TYPE",
        "",
        "=" * 60,
    ]

    reg_requirements = {
        "data_breach": [
            ("GDPR (EU)", "72 hours from awareness to DPA notification"),
            ("State AG (varies)", "30-60 days for consumer notification; immediate AG notification in some states"),
            ("FTC", "No fixed deadline but failure to notify promptly may constitute unfair practice"),
            ("SEC (if public company)", "Material cybersecurity incident: 4 business days via 8-K"),
        ],
        "product_recall": [
            ("CPSC", "24 hours for immediate hazard; otherwise within 10 business days of learning of defect"),
            ("State Consumer Protection", "Varies by state; typically concurrent with CPSC"),
            ("FDA (if food/drug)", "Within 24 hours of determination"),
        ],
        "environmental_spill": [
            ("EPA", "Immediate notification for reportable quantities; written follow-up within 30 days"),
            ("State DEQ", "Immediate telephone notification; written within 7 days"),
            ("Local Emergency Management", "Immediate notification if public safety risk"),
            ("Coast Guard (if navigable waters)", "Immediate notification"),
        ],
        "financial_restatement": [
            ("SEC", "8-K filing within 4 business days of determination"),
            ("PCAOB", "Notification to audit committee immediately upon determination"),
            ("Stock Exchange", "Immediate notification if trading halt may be warranted"),
        ],
        "workplace_incident": [
            ("OSHA", "8 hours for fatality; 24 hours for hospitalization, amputation, or eye loss"),
            ("State Workers Comp", "Within 10 days of knowledge of injury"),
            ("State OSHA (if applicable)", "Same as federal OSHA or stricter"),
        ],
        "ransomware": [
            ("FBI/CISA", "CIRCIA: 72 hours for covered entities; voluntary for others"),
            ("GDPR (EU)", "72 hours from awareness if personal data affected"),
            ("State AG", "Varies; typically 30-60 days for consumer notification"),
            ("SEC (if public)", "4 business days via 8-K for material incidents"),
        ],
        "executive_misconduct": [
            ("SEC", "8-K within 4 business days for material events; whistleblower protections apply"),
            ("DOJ", "Voluntary self-disclosure programs; timing varies"),
            ("Stock Exchange", "Prompt notification of material events"),
        ],
        "supply_chain_failure": [
            ("FTC", "No specific timeline; notify if consumer impact is material"),
            ("Industry Regulator", "Varies by industry; pharmaceutical supply chain: FDA within 6 months"),
            ("Contractual Counterparties", "Per contract terms; typically 5-30 business days"),
        ],
    }

    for body, requirement in reg_requirements.get(crisis_key, []):
        reg_lines.extend([
            "",
            f"  {body}:",
            f"    {requirement}",
        ])

    reg_lines.extend([
        "",
        "=" * 60,
        "",
        "NOTE: These are general guidelines. Specific obligations depend on",
        "company size, industry, jurisdiction, and the nature of the incident.",
        "Always consult legal counsel for definitive requirements.",
        "",
    ])
    regulatory_requirements = "\n".join(reg_lines) + "\n"

    # --- BUILD RUBRIC ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does /testbed/audit_report.txt exist with substantial content?",
            points=1,
        ),
        BinaryRubricCategory(
            name="identifies_crisis_type",
            question=f'Does the audit correctly identify the crisis type as "{scenario["label"]}"?',
            points=1,
        ),
        BinaryRubricCategory(
            name="identifies_severity",
            question=f'Does the audit identify the incident severity as "{severity}"?',
            points=1,
        ),
        BinaryRubricCategory(
            name="reconstructs_timeline",
            question="Does the audit reconstruct a chronological timeline with at least 6 key events?",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_discovery_time",
            question=f"Does the audit identify the discovery time as approximately {_ts(base_y, base_m, base_d, discover_h, discover_min)}?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_confirmation_time",
            question=f"Does the audit identify the confirmation time as approximately {_ts(*confirm_ts)}?",
            points=1,
        ),
    ]

    # Additional structural checks
    rubric_items.extend([
        BinaryRubricCategory(
            name="identifies_regulatory_bodies",
            question=f"Does the audit list the relevant regulatory bodies ({', '.join(scenario['regulatory_bodies'])})?",
            points=1,
        ),
        BinaryRubricCategory(
            name="identifies_board_notification_time",
            question=f"Does the audit note the board notification time as approximately {_ts(*board_ts)}?",
            points=2,
        ),
        BinaryRubricCategory(
            name="identifies_public_disclosure_time",
            question=f"Does the audit note the public disclosure (press release) time as approximately {_ts(*public_ts)}?",
            points=2,
        ),
        BinaryRubricCategory(
            name="checks_spokesperson_authorization",
            question="Does the audit verify whether external spokespersons were authorized per the communication policy?",
            points=2,
        ),
        BinaryRubricCategory(
            name="checks_channel_compliance",
            question="Does the audit check whether the correct communication channels were used for the severity level?",
            points=2,
        ),
        BinaryRubricCategory(
            name="distinguishes_discovery_vs_confirmation",
            question="Does the audit distinguish between the time of discovery and the time of confirmation when evaluating notification deadlines?",
            points=2,
        ),
    ])

    # Per-violation checks (2pt each)
    for v in violations_detail:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"identifies_{v['type']}",
                question=f"Does the audit identify this violation: {v['description']}?",
                points=2,
            )
        )

    rubric_items.append(
        BinaryRubricCategory(
            name="correct_violation_count",
            question=f"Does the audit identify approximately {len(violations_detail)} communication policy violations (within +/- 1)?",
            points=2,
        )
    )

    # False positive checks (should NOT be flagged as violations)
    for fp in false_positives[:2]:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"no_false_positive_{fp['name']}",
                question=(
                    f"Does the audit correctly NOT flag the following as a violation: "
                    f"{fp['description']}?"
                ),
                points=1,
            )
        )

    rubric_items.extend([
        BinaryRubricCategory(
            name="references_policy_windows",
            question="Does the audit reference specific notification window requirements from the policy?",
            points=2,
        ),
        BinaryRubricCategory(
            name="references_regulatory_requirements",
            question="Does the audit cross-reference at least one regulatory notification requirement?",
            points=2,
        ),
        RubricCategory(
            name="analysis_depth",
            description="How thorough is the policy compliance analysis?",
            failure="Superficial listing of events without policy comparison",
            minor_failure="Some events compared to policy but major violations missed",
            minor_success="Most violations found with policy references but some gaps",
            success="Comprehensive analysis comparing each communication event to policy requirements with specific citations",
            points=3,
        ),
    ])

    # --- Problem statement ---
    problem_statement = f"""# Crisis Response Communication Audit

{company} experienced a {scenario['label']}: {scenario['description']}.

You are an independent communications auditor. Review the company's crisis response
timeline, communications policy, actual stakeholder communications, and regulatory
requirements to identify any communication policy violations.

## Source Files
- /testbed/data/incident_timeline.csv — Chronological events with timestamps
- /testbed/data/communication_policy.txt — Company crisis communication policy with notification windows
- /testbed/data/stakeholder_communications.txt — Actual communications sent with timestamps
- /testbed/data/regulatory_requirements.txt — Regulatory notification deadlines by incident type

## Requirements
1. Reconstruct the full timeline from discovery through response
2. Compare each communication event against the policy notification windows
3. Check for messaging consistency across channels
4. Verify all required regulatory notifications were made
5. Check that only authorized spokespersons made external statements
6. Identify any violations and note compliant actions
7. Provide recommendations for improvement

Write a detailed audit report to /testbed/audit_report.txt."""

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=tuple(rubric_items),
        submission_instructions="Write your audit report to /testbed/audit_report.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/incident_timeline.csv": incident_timeline_csv,
            "/testbed/data/communication_policy.txt": communication_policy,
            "/testbed/data/stakeholder_communications.txt": stakeholder_communications,
            "/testbed/data/regulatory_requirements.txt": regulatory_requirements,
        },
        problem_type="crisis_response_audit",
    )


# =============================================================================
# 2. STAKEHOLDER IMPACT ASSESSMENT
# =============================================================================

CHANGE_TYPES = [
    {
        "key": "office_relocation",
        "label": "Corporate Office Relocation",
        "description_template": (
            "{company} plans to relocate its corporate headquarters from {from_city} "
            "to {to_city}. The move involves approximately {n_employees} employees "
            "and is scheduled to complete by {completion_date}."
        ),
        "from_cities": ["Chicago, IL", "San Francisco, CA", "New York, NY", "Boston, MA"],
        "to_cities": ["Austin, TX", "Nashville, TN", "Denver, CO", "Raleigh, NC"],
    },
    {
        "key": "system_migration",
        "label": "ERP System Migration",
        "description_template": (
            "{company} is migrating from {old_system} to {new_system}. The migration "
            "affects all business units and is scheduled for {completion_date}, with a "
            "planned cutover window of {cutover_days} days."
        ),
        "old_systems": ["Legacy SAP ECC 6.0", "Oracle E-Business Suite 12.2", "Microsoft Dynamics AX 2012"],
        "new_systems": ["SAP S/4HANA Cloud", "Oracle Cloud ERP", "Microsoft Dynamics 365"],
    },
    {
        "key": "restructuring",
        "label": "Organizational Restructuring",
        "description_template": (
            "{company} is reorganizing from a {old_structure} to a {new_structure}. "
            "This affects {n_departments} departments and approximately {n_employees} positions. "
            "Implementation target: {completion_date}."
        ),
        "old_structures": ["functional hierarchy", "geographic divisions", "product-line organization"],
        "new_structures": ["matrix organization", "agile pod structure", "customer-segment alignment"],
    },
    {
        "key": "product_sunset",
        "label": "Product Line Sunset",
        "description_template": (
            "{company} will discontinue the {product_name} product line effective {completion_date}. "
            "Current users ({n_users} active accounts) will need to migrate to {replacement_name} "
            "or find alternative solutions."
        ),
        "product_names": ["ProConnect Platform", "DataSync Enterprise", "CloudBridge Suite", "FieldOps Manager"],
        "replacement_names": ["ProConnect Next", "DataSync Cloud", "CloudBridge 2.0", "FieldOps Pro"],
    },
    {
        "key": "pricing_change",
        "label": "Major Pricing Restructure",
        "description_template": (
            "{company} is restructuring its pricing model from {old_model} to {new_model}. "
            "The change affects approximately {n_customers} customers and will be phased in "
            "starting {start_date} with full implementation by {completion_date}."
        ),
        "old_models": ["per-seat licensing", "flat-rate annual subscription", "usage-based billing"],
        "new_models": ["tiered value-based pricing", "consumption-based model", "outcome-based pricing"],
    },
    {
        "key": "outsourcing",
        "label": "IT Operations Outsourcing",
        "description_template": (
            "{company} is outsourcing {function} to {vendor}. The transition affects "
            "{n_employees} internal positions and is planned for {completion_date}. "
            "Service levels must be maintained throughout the transition."
        ),
        "functions": ["infrastructure management", "help desk operations", "application maintenance"],
        "vendors": ["TCS", "Infosys", "Wipro", "Accenture"],
    },
    {
        "key": "acquisition_integration",
        "label": "Acquisition Integration",
        "description_template": (
            "{company} is integrating {acquired_company} following the recently closed acquisition. "
            "Integration affects {n_employees} employees across both organizations. "
            "Target integration completion: {completion_date}."
        ),
    },
    {
        "key": "remote_work_policy",
        "label": "Return-to-Office Policy Change",
        "description_template": (
            "{company} is implementing a new hybrid work policy requiring {days_in_office} days "
            "per week in-office, effective {completion_date}. This affects approximately "
            "{n_employees} employees currently working fully remote."
        ),
    },
]

STAKEHOLDER_GROUPS = ["Employees", "Customers", "Vendors", "Regulators", "Investors"]

IMPACT_CRITERIA = [
    "operational_impact",
    "financial_impact",
    "relationship_impact",
    "timeline_sensitivity",
]


def make_stakeholder_impact_assessment(rand_seed: int = 42) -> RubricDatapoint:
    """Assess impact of a proposed business change on different stakeholder groups.

    Seed varies: change type, stakeholder composition (15-25), impact scores,
    indirect dependencies, historical precedent issues.
    """
    rng = _random.Random(rand_seed)

    company = pick1(COMPANY_NAMES, rand_seed)
    change_type = rng.choice(CHANGE_TYPES)
    change_key = change_type["key"]

    # Generate change description
    n_employees = rng.randint(100, 2000)
    completion_year = 2025
    completion_month = rng.randint(3, 11)
    completion_date = _date(completion_year, completion_month, 28)

    if change_key == "office_relocation":
        from_city = rng.choice(change_type["from_cities"])
        to_city = rng.choice(change_type["to_cities"])
        description = change_type["description_template"].format(
            company=company, from_city=from_city, to_city=to_city,
            n_employees=n_employees, completion_date=completion_date,
        )
    elif change_key == "system_migration":
        old_system = rng.choice(change_type["old_systems"])
        new_system = rng.choice(change_type["new_systems"])
        cutover_days = rng.choice([3, 5, 7, 10])
        description = change_type["description_template"].format(
            company=company, old_system=old_system, new_system=new_system,
            completion_date=completion_date, cutover_days=cutover_days,
        )
    elif change_key == "restructuring":
        old_structure = rng.choice(change_type["old_structures"])
        new_structure = rng.choice(change_type["new_structures"])
        n_departments = rng.randint(4, 12)
        description = change_type["description_template"].format(
            company=company, old_structure=old_structure, new_structure=new_structure,
            n_departments=n_departments, n_employees=n_employees,
            completion_date=completion_date,
        )
    elif change_key == "product_sunset":
        product_name = rng.choice(change_type["product_names"])
        replacement_name = rng.choice(change_type["replacement_names"])
        n_users = rng.randint(500, 15000)
        description = change_type["description_template"].format(
            company=company, product_name=product_name, replacement_name=replacement_name,
            n_users=n_users, completion_date=completion_date,
        )
    elif change_key == "pricing_change":
        old_model = rng.choice(change_type["old_models"])
        new_model = rng.choice(change_type["new_models"])
        n_customers = rng.randint(200, 5000)
        start_date = _date(2025, max(1, completion_month - 3), 1)
        description = change_type["description_template"].format(
            company=company, old_model=old_model, new_model=new_model,
            n_customers=n_customers, start_date=start_date,
            completion_date=completion_date,
        )
    elif change_key == "outsourcing":
        function = rng.choice(change_type["functions"])
        vendor = rng.choice(change_type["vendors"])
        description = change_type["description_template"].format(
            company=company, function=function, vendor=vendor,
            n_employees=rng.randint(30, 200), completion_date=completion_date,
        )
    elif change_key == "acquisition_integration":
        acquired_company = rng.choice([c for c in COMPANY_NAMES if c != company])
        description = change_type["description_template"].format(
            company=company, acquired_company=acquired_company,
            n_employees=n_employees, completion_date=completion_date,
        )
    elif change_key == "remote_work_policy":
        days_in_office = rng.choice([2, 3, 4])
        description = change_type["description_template"].format(
            company=company, days_in_office=days_in_office,
            n_employees=n_employees, completion_date=completion_date,
        )
    else:
        description = f"{company} is implementing a major business change effective {completion_date}."

    # --- Generate stakeholders ---
    n_stakeholders = rng.randint(15, 25)
    stakeholder_names = random_names(rand_seed + 100, n_stakeholders)

    relationship_statuses = ["Strong", "Good", "Neutral", "Strained", "New"]

    stakeholders: list[dict] = []
    # Assign groups with some distribution
    group_weights = {
        "Employees": 0.30, "Customers": 0.25, "Vendors": 0.20,
        "Regulators": 0.10, "Investors": 0.15,
    }
    groups_list = list(group_weights.keys())
    weights_list = list(group_weights.values())

    # Indirect dependency tracking
    indirect_dependencies: list[dict] = []
    system_dependency_stakeholders: list[int] = []  # indices of stakeholders with system deps

    for i, name in enumerate(stakeholder_names):
        group = rng.choices(groups_list, weights=weights_list, k=1)[0]
        status = rng.choice(relationship_statuses)

        # Generate dependencies
        dependencies: list[str] = []
        if group == "Vendors":
            dep_options = [
                "ERP system integration",
                "API data feeds",
                "Shared logistics platform",
                "Direct contract with company",
                "Subcontract via primary vendor",
            ]
            dependencies = rng.sample(dep_options, rng.randint(1, 3))
            if any("system" in d.lower() or "ERP" in d or "API" in d for d in dependencies):
                system_dependency_stakeholders.append(i)
        elif group == "Customers":
            dep_options = [
                "Product/service subscription",
                "Custom integration",
                "Dedicated support SLA",
                "Data hosted on company infrastructure",
                "Regulatory reporting dependency",
            ]
            dependencies = rng.sample(dep_options, rng.randint(1, 3))
        elif group == "Employees":
            dep_options = [
                "Office location",
                "Reporting structure",
                "Compensation/benefits",
                "Career progression path",
                "Team composition",
            ]
            dependencies = rng.sample(dep_options, rng.randint(1, 3))
        elif group == "Regulators":
            dep_options = [
                "Compliance reporting",
                "License/permit requirements",
                "Audit access",
                "Data residency requirements",
            ]
            dependencies = rng.sample(dep_options, rng.randint(1, 2))
        elif group == "Investors":
            dep_options = [
                "Quarterly earnings expectations",
                "Growth metrics",
                "Cost structure assumptions",
                "Strategic direction alignment",
            ]
            dependencies = rng.sample(dep_options, rng.randint(1, 2))

        # Compute impact scores (1-5 scale per criterion)
        base_impact = {
            "operational_impact": rng.randint(1, 5),
            "financial_impact": rng.randint(1, 5),
            "relationship_impact": rng.randint(1, 5),
            "timeline_sensitivity": rng.randint(1, 5),
        }

        # Make some stakeholders high-impact (at least one criterion >= 4)
        if rng.random() < 0.3:
            high_criterion = rng.choice(IMPACT_CRITERIA)
            base_impact[high_criterion] = rng.randint(4, 5)

        stakeholders.append({
            "name": name,
            "group": group,
            "status": status,
            "dependencies": dependencies,
            "impact": base_impact,
            "overall_impact": sum(base_impact.values()),
        })

    # Plant indirect dependencies (vendor depends on system being migrated)
    n_indirect = rng.randint(2, 4)
    for idx in rng.sample(
        system_dependency_stakeholders,
        min(n_indirect, len(system_dependency_stakeholders))
    ) if system_dependency_stakeholders else []:
        s = stakeholders[idx]
        indirect_dependencies.append({
            "stakeholder": s["name"],
            "dependency_chain": (
                f"{s['name']} ({s['group']}) depends on {rng.choice(s['dependencies'])}, "
                f"which is directly affected by the proposed change"
            ),
            "hidden_impact": "Disruption may cascade to this stakeholder even though they are not directly targeted by the change",
        })
        # Increase their impact scores
        s["impact"]["operational_impact"] = min(5, s["impact"]["operational_impact"] + 2)
        s["overall_impact"] = sum(s["impact"].values())

    # Identify high-impact stakeholders (overall >= 14 out of 20)
    high_impact_stakeholders = [s for s in stakeholders if s["overall_impact"] >= 14]
    # Ensure at least 3 high-impact
    stakeholders_sorted = sorted(stakeholders, key=lambda s: s["overall_impact"], reverse=True)
    if len(high_impact_stakeholders) < 3:
        for s in stakeholders_sorted:
            if s not in high_impact_stakeholders:
                high_impact_stakeholders.append(s)
            if len(high_impact_stakeholders) >= 3:
                break

    # --- Historical changes ---
    n_historical = rng.randint(3, 6)
    historical_changes: list[dict] = []
    historical_outcomes = [
        {"change": "Office relocation 2019", "outcome": "18% voluntary attrition in first 6 months", "lesson": "Early relocation assistance and retention bonuses reduce attrition"},
        {"change": "ERP migration 2020", "outcome": "3-week unplanned downtime; $2.1M revenue impact", "lesson": "Parallel running period should be at least 4 weeks, not 2"},
        {"change": "Department restructuring 2021", "outcome": "Customer satisfaction dropped 15 points during transition", "lesson": "Dedicated transition teams for key accounts prevent service gaps"},
        {"change": "Product sunset (LegacyConnect) 2022", "outcome": "12% customer churn, most from enterprise segment", "lesson": "Migration path must be fully tested before announcing sunset date"},
        {"change": "Pricing change 2020", "outcome": "Lost 3 top-10 customers in Q1 after announcement", "lesson": "Grandfather existing pricing for 12 months minimum for enterprise customers"},
        {"change": "IT outsourcing 2018", "outcome": "SLA breaches in first 90 days; 6-month remediation plan", "lesson": "Knowledge transfer period needs minimum 6 months with shadow operations"},
        {"change": "Acquisition integration 2021", "outcome": "Key talent departure from acquired company (35%)", "lesson": "Retention packages and cultural integration programs needed from day one"},
        {"change": "Remote work policy change 2023", "outcome": "Employee engagement scores dropped 22 points", "lesson": "Phased rollout with flexibility provisions receives better adoption"},
    ]
    chosen_historical = rng.sample(historical_outcomes, min(n_historical, len(historical_outcomes)))
    # Plant a directly relevant historical precedent
    relevant_hist = None
    for h in chosen_historical:
        h_key_words = h["change"].lower().split()
        if any(w in change_key for w in h_key_words):
            relevant_hist = h
            break
    if relevant_hist is None and chosen_historical:
        relevant_hist = chosen_historical[0]  # First one as fallback

    historical_changes = chosen_historical

    # --- BUILD FILES ---

    # 1. change_proposal.txt
    proposal_lines = [
        f"CHANGE PROPOSAL — {company}",
        "",
        f"Proposal Title: {change_type['label']}",
        f"Sponsor: Executive Leadership Team",
        f"Target Completion: {completion_date}",
        "",
        "=" * 60,
        "DESCRIPTION",
        "=" * 60,
        "",
        description,
        "",
        "=" * 60,
        "OBJECTIVES",
        "=" * 60,
        "",
    ]
    objectives = [
        "Improve operational efficiency and reduce costs",
        "Align organizational structure with strategic goals",
        "Modernize technology infrastructure",
        "Enhance competitive positioning",
        "Improve customer experience and satisfaction",
    ]
    for obj in rng.sample(objectives, 3):
        proposal_lines.append(f"  - {obj}")
    proposal_lines.extend([
        "",
        "=" * 60,
        "TIMELINE",
        "=" * 60,
        "",
        f"  Phase 1 (Planning): {_date(2025, max(1, completion_month - 4), 1)} - "
        f"{_date(2025, max(1, completion_month - 3), 28)}",
        f"  Phase 2 (Preparation): {_date(2025, max(1, completion_month - 2), 1)} - "
        f"{_date(2025, max(1, completion_month - 1), 28)}",
        f"  Phase 3 (Execution): {_date(2025, completion_month, 1)} - {completion_date}",
        f"  Phase 4 (Stabilization): {completion_date} + 90 days",
        "",
        "=" * 60,
        "ESTIMATED COSTS",
        "=" * 60,
        "",
        f"  One-time costs: ${rng.randint(500, 5000) * 1000:,}",
        f"  Annual savings (expected): ${rng.randint(200, 3000) * 1000:,}",
        f"  Payback period: {rng.randint(12, 36)} months",
        "",
    ])
    change_proposal = "\n".join(proposal_lines) + "\n"

    # 2. stakeholder_registry.csv
    csv_lines = ["name,group,relationship_status,dependencies"]
    for s in stakeholders:
        deps_str = "; ".join(s["dependencies"])
        csv_lines.append(f'"{s["name"]}","{s["group"]}","{s["status"]}","{deps_str}"')
    stakeholder_registry_csv = "\n".join(csv_lines) + "\n"

    # 3. impact_criteria.txt
    criteria_lines = [
        "STAKEHOLDER IMPACT ASSESSMENT FRAMEWORK",
        "",
        "=" * 60,
        "ASSESSMENT CRITERIA (Score 1-5 for each)",
        "=" * 60,
        "",
        "1. OPERATIONAL IMPACT",
        "   How much will the change disrupt the stakeholder's day-to-day operations?",
        "   1 = No disruption  |  2 = Minimal  |  3 = Moderate  |  4 = Significant  |  5 = Severe",
        "",
        "   Consider: workflow changes, system access, physical location changes,",
        "   process modifications, training requirements",
        "",
        "2. FINANCIAL IMPACT",
        "   What is the financial effect on the stakeholder?",
        "   1 = None  |  2 = Minor  |  3 = Moderate  |  4 = Significant  |  5 = Major",
        "",
        "   Consider: cost increases, revenue risk, contract renegotiation needs,",
        "   compliance costs, opportunity costs",
        "",
        "3. RELATIONSHIP IMPACT",
        "   How will the change affect the stakeholder relationship with the company?",
        "   1 = Strengthened  |  2 = No change  |  3 = Slight risk  |  4 = At risk  |  5 = Severe risk",
        "",
        "   Consider: trust level, communication history, alternative options,",
        "   strategic importance of relationship",
        "",
        "4. TIMELINE SENSITIVITY",
        "   How sensitive is the stakeholder to the timing of the change?",
        "   1 = Flexible  |  2 = Somewhat flexible  |  3 = Moderate  |  4 = Tight  |  5 = Critical",
        "",
        "   Consider: contractual deadlines, regulatory windows, seasonal factors,",
        "   dependent milestones",
        "",
        "=" * 60,
        "HIGH-IMPACT THRESHOLD",
        "=" * 60,
        "",
        "Stakeholders with an aggregate score of 14 or higher (out of 20) should",
        "be classified as HIGH-IMPACT and require a dedicated mitigation plan.",
        "",
        "=" * 60,
        "INDIRECT IMPACT ASSESSMENT",
        "=" * 60,
        "",
        "Assessors must consider indirect impacts: stakeholders whose dependencies",
        "chain through systems, processes, or other stakeholders directly affected",
        "by the change. Cross-reference the stakeholder registry dependencies with",
        "the change proposal to identify indirect exposure.",
        "",
    ]
    impact_criteria = "\n".join(criteria_lines) + "\n"

    # 4. historical_changes.csv
    hist_csv_lines = ["change_description,outcome,lesson_learned"]
    for h in historical_changes:
        hist_csv_lines.append(f'"{h["change"]}","{h["outcome"]}","{h["lesson"]}"')
    historical_changes_csv = "\n".join(hist_csv_lines) + "\n"

    # --- Compute additional ground-truth values for rubric ---
    n_total_stakeholders = len(stakeholders)
    n_high_impact = len([s for s in stakeholders if s["overall_impact"] >= 14])
    group_counts = {}
    for g in STAKEHOLDER_GROUPS:
        group_counts[g] = len([s for s in stakeholders if s["group"] == g])
    strained_stakeholders = [s for s in stakeholders if s["status"] == "Strained"]

    # --- BUILD RUBRIC ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does /testbed/impact_assessment.txt exist with substantial content?",
            points=1,
        ),
        BinaryRubricCategory(
            name="identifies_change_type",
            question=f'Does the assessment correctly identify the change as "{change_type["label"]}"?',
            points=1,
        ),
        BinaryRubricCategory(
            name="covers_all_groups",
            question="Does the assessment address all 5 stakeholder groups (Employees, Customers, Vendors, Regulators, Investors)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="uses_impact_criteria",
            question="Does the assessment use the 4-criterion framework (operational, financial, relationship, timeline) from the impact criteria document?",
            points=2,
        ),
        BinaryRubricCategory(
            name="provides_numeric_scores",
            question="Does the assessment provide numeric impact scores (1-5 scale) for stakeholders?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_stakeholder_count",
            question=f"Does the assessment acknowledge or analyze all {n_total_stakeholders} stakeholders from the registry?",
            points=1,
        ),
        BinaryRubricCategory(
            name="identifies_completion_date",
            question=f"Does the assessment reference the target completion date ({completion_date})?",
            points=1,
        ),
        BinaryRubricCategory(
            name="includes_executive_summary",
            question="Does the assessment include an executive summary or overview section?",
            points=1,
        ),
        BinaryRubricCategory(
            name="identifies_strained_relationships",
            question=(
                f"Does the assessment flag stakeholders with 'Strained' relationship status "
                f"({len(strained_stakeholders)} exist) as requiring extra attention?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="considers_timeline_phases",
            question="Does the assessment reference the phased timeline from the change proposal?",
            points=1,
        ),
        BinaryRubricCategory(
            name="addresses_cost_impact",
            question="Does the assessment discuss the financial/cost implications from the change proposal?",
            points=1,
        ),
    ]

    # High-impact stakeholder identification
    high_impact_names = [s["name"] for s in high_impact_stakeholders[:5]]
    rubric_items.append(
        BinaryRubricCategory(
            name="identifies_high_impact",
            question=(
                f"Does the assessment identify at least 3 high-impact stakeholders "
                f"(aggregate score >= 14)? Top high-impact stakeholders include: "
                f"{', '.join(high_impact_names[:3])}"
            ),
            points=2,
        )
    )

    rubric_items.append(
        BinaryRubricCategory(
            name="correct_high_impact_count",
            question=f"Does the assessment identify approximately {n_high_impact} high-impact stakeholders (within +/- 2)?",
            points=2,
        )
    )

    # Indirect dependency checks (up to 2 to control total)
    for dep in indirect_dependencies[:2]:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"indirect_dep_{dep['stakeholder'].replace(' ', '_').lower()[:20]}",
                question=(
                    f"Does the assessment identify the indirect impact on {dep['stakeholder']}? "
                    f"({dep['dependency_chain']})"
                ),
                points=2,
            )
        )

    # Historical precedent check
    if relevant_hist:
        rubric_items.append(
            BinaryRubricCategory(
                name="references_historical_precedent",
                question=(
                    f"Does the assessment reference the relevant historical precedent "
                    f"('{relevant_hist['change']}' which resulted in '{relevant_hist['outcome']}') "
                    f"and apply its lesson?"
                ),
                points=2,
            )
        )

    rubric_items.append(
        BinaryRubricCategory(
            name="cross_references_documents",
            question="Does the assessment cross-reference information from at least 3 of the 4 source documents?",
            points=2,
        )
    )

    # Group-level impact checks
    for group in ["Employees", "Customers", "Vendors"]:
        group_stakeholders = [s for s in stakeholders if s["group"] == group]
        if group_stakeholders:
            highest = max(group_stakeholders, key=lambda s: s["overall_impact"])
            rubric_items.append(
                BinaryRubricCategory(
                    name=f"correct_{group.lower()}_impact",
                    question=(
                        f"Does the assessment correctly identify the highest-impact {group.lower()} "
                        f"stakeholder as having significant impact (aggregate >= {highest['overall_impact'] - 2})? "
                        f"(Highest: {highest['name']} with score {highest['overall_impact']})"
                    ),
                    points=2,
                )
            )

    rubric_items.extend([
        BinaryRubricCategory(
            name="mitigation_plans",
            question="Does the assessment include specific mitigation plans for at least 3 high-impact stakeholders?",
            points=2,
        ),
        BinaryRubricCategory(
            name="prioritizes_engagement",
            question="Does the assessment include a prioritized engagement or communication plan?",
            points=2,
        ),
        RubricCategory(
            name="analysis_quality",
            description="How thorough and insightful is the stakeholder impact analysis?",
            failure="Superficial listing without meaningful analysis",
            minor_failure="Basic analysis but missing indirect impacts or historical context",
            minor_success="Good analysis with most impacts identified but some gaps",
            success="Comprehensive analysis with indirect impacts, historical precedent, and stakeholder-specific mitigation plans",
            points=3,
        ),
    ])

    problem_statement = f"""# Stakeholder Impact Assessment

{company} is planning a major business change: {change_type['label']}.

You are a change management consultant. Assess the impact of this proposed change
on all identified stakeholders using the provided assessment framework.

## Source Files
- /testbed/data/change_proposal.txt — Detailed description of the proposed change
- /testbed/data/stakeholder_registry.csv — Stakeholders with groups, relationships, and dependencies
- /testbed/data/impact_criteria.txt — Assessment framework and scoring criteria
- /testbed/data/historical_changes.csv — Previous similar changes and their outcomes

## Requirements
1. Review the change proposal and understand the scope
2. For each stakeholder, assess impact across all 4 criteria dimensions (1-5 scale)
3. Identify high-impact stakeholders (aggregate score >= 14 out of 20)
4. Identify indirect impacts through dependency chains
5. Cross-reference with historical changes for lessons learned
6. Provide specific mitigation plans for high-impact stakeholders
7. Prioritize stakeholder engagement based on impact severity

Write your impact assessment to /testbed/impact_assessment.txt."""

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=tuple(rubric_items),
        submission_instructions="Write your stakeholder impact assessment to /testbed/impact_assessment.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/change_proposal.txt": change_proposal,
            "/testbed/data/stakeholder_registry.csv": stakeholder_registry_csv,
            "/testbed/data/impact_criteria.txt": impact_criteria,
            "/testbed/data/historical_changes.csv": historical_changes_csv,
        },
        problem_type="stakeholder_impact_assessment",
    )


# =============================================================================
# 3. COMMUNICATIONS TIMELINE ANALYSIS
# =============================================================================

DISPUTE_TYPES = [
    {
        "key": "construction_delay",
        "label": "Construction Delay Dispute",
        "description": "Dispute over construction project delays and associated liquidated damages",
        "party_a_role": "Property Developer",
        "party_b_role": "General Contractor",
    },
    {
        "key": "software_delivery",
        "label": "Software Delivery Dispute",
        "description": "Dispute over missed software delivery milestones and acceptance criteria",
        "party_a_role": "Client",
        "party_b_role": "Software Vendor",
    },
    {
        "key": "service_agreement",
        "label": "Service Level Agreement Dispute",
        "description": "Dispute over SLA breaches and service credits",
        "party_a_role": "Customer",
        "party_b_role": "Service Provider",
    },
    {
        "key": "licensing_royalty",
        "label": "Licensing Royalty Dispute",
        "description": "Dispute over royalty calculations and audit rights under licensing agreement",
        "party_a_role": "Licensor",
        "party_b_role": "Licensee",
    },
    {
        "key": "supply_agreement",
        "label": "Supply Agreement Dispute",
        "description": "Dispute over quality specifications, delivery shortfalls, and pricing adjustments",
        "party_a_role": "Buyer",
        "party_b_role": "Supplier",
    },
    {
        "key": "lease_dispute",
        "label": "Commercial Lease Dispute",
        "description": "Dispute over lease terms, maintenance obligations, and rent escalation clauses",
        "party_a_role": "Tenant",
        "party_b_role": "Landlord",
    },
    {
        "key": "consulting_scope",
        "label": "Consulting Engagement Dispute",
        "description": "Dispute over scope of engagement, deliverable quality, and payment withholding",
        "party_a_role": "Client",
        "party_b_role": "Consulting Firm",
    },
    {
        "key": "partnership_dissolution",
        "label": "Partnership Dissolution Dispute",
        "description": "Dispute over valuation, asset distribution, and non-compete obligations",
        "party_a_role": "Managing Partner",
        "party_b_role": "Departing Partner",
    },
]

# Issue types to plant in communications timeline
TIMELINE_ISSUES = [
    "deadline_violation",
    "contradictory_commitment",
    "missing_required_participant",
    "unfulfilled_commitment",
    "improper_notice",
    "cure_period_expired",
]


def make_communications_timeline_analysis(rand_seed: int = 42) -> RubricDatapoint:
    """Reconstruct and analyze a multi-party communication timeline for a
    business dispute.

    Seed varies: dispute type, parties, planted issues (3-6), email content,
    meeting records, contract terms, and false positives (2-3).
    """
    rng = _random.Random(rand_seed)

    dispute = rng.choice(DISPUTE_TYPES)
    dispute_key = dispute["key"]

    # Companies and people
    company_a = pick1(COMPANY_NAMES, rand_seed)
    company_b = rng.choice([c for c in COMPANY_NAMES if c != company_a])

    names = random_names(rand_seed + 200, 10)
    party_a_contact = names[0]
    party_a_counsel = names[1]
    party_b_contact = names[2]
    party_b_counsel = names[3]
    mediator = names[4]
    party_a_exec = names[5]
    party_b_exec = names[6]
    witness1 = names[7]
    witness2 = names[8]
    party_a_pm = names[9]  # project manager

    # Contract reference
    contract_number = f"AGR-{rng.randint(10000, 99999)}"
    contract_date = _date(2023, rng.randint(1, 6), rng.randint(1, 28))

    # Timeline base
    base_year = 2024
    base_month = rng.randint(2, 8)

    # --- Contract terms (what the contract says) ---
    response_period_days = rng.choice([5, 7, 10, 14])
    cure_period_days = rng.choice([15, 20, 30])
    notice_method = rng.choice(["certified mail to registered address", "email with delivery confirmation to designated contact", "hand delivery with signed receipt"])
    required_meeting_participants = f"authorized representatives of both parties (minimum VP-level)"
    dispute_resolution = rng.choice(["binding arbitration per AAA rules", "mediation followed by arbitration", "litigation in courts of agreed jurisdiction"])
    governing_law = rng.choice(["New York", "Delaware", "California", "Texas"])

    # Milestone deadlines
    milestones: list[dict] = []
    for i in range(rng.randint(3, 5)):
        milestone_month = base_month + i
        milestone_year = base_year
        if milestone_month > 12:
            milestone_month -= 12
            milestone_year += 1
        milestones.append({
            "name": f"Milestone {i+1}",
            "description": rng.choice([
                "Deliverable submission",
                "Phase completion review",
                "Acceptance testing",
                "Progress report due",
                "Payment milestone",
            ]),
            "due_date": _date(milestone_year, milestone_month, rng.randint(10, 25)),
        })

    # --- Decide which issues to plant ---
    n_issues = rng.randint(3, 6)
    available_issues = list(TIMELINE_ISSUES)
    rng.shuffle(available_issues)
    planted_issues = available_issues[:n_issues]

    issues_detail: list[dict] = []

    # Generate email thread
    emails: list[dict] = []
    email_id = 1

    # Meetings
    meetings: list[dict] = []

    # Calendar events
    calendar_events: list[dict] = []
    for m in milestones:
        calendar_events.append({
            "date": m["due_date"],
            "event": f"{m['name']}: {m['description']}",
            "type": "deadline",
        })

    # --- Build the narrative and plant issues ---

    # Opening email: Party A raises concern
    concern_month = base_month + 1
    concern_year = base_year
    if concern_month > 12:
        concern_month -= 12
        concern_year += 1
    concern_day = rng.randint(5, 15)
    concern_date = _date(concern_year, concern_month, concern_day)

    emails.append({
        "id": email_id,
        "date": concern_date,
        "from": f"{party_a_contact} <{party_a_contact.split()[0].lower()}@{company_a.lower().replace(' ', '')}.com>",
        "to": f"{party_b_contact} <{party_b_contact.split()[0].lower()}@{company_b.lower().replace(' ', '')}.com>",
        "cc": f"{party_a_pm}",
        "subject": f"Re: {contract_number} — Performance Concerns",
        "body": (
            f"Hi {party_b_contact.split()[0]},\n\n"
            f"I'm writing to express our concern regarding recent performance under "
            f"agreement {contract_number}. Specifically, we've noticed issues with "
            f"{milestones[0]['description'].lower()} quality and timeliness.\n\n"
            f"We'd like to schedule a call this week to discuss.\n\n"
            f"Regards,\n{party_a_contact}"
        ),
    })
    email_id += 1

    # Party B response
    response_day = concern_day + rng.randint(1, 3)
    response_date = _date(concern_year, concern_month, min(28, response_day))

    emails.append({
        "id": email_id,
        "date": response_date,
        "from": f"{party_b_contact} <{party_b_contact.split()[0].lower()}@{company_b.lower().replace(' ', '')}.com>",
        "to": f"{party_a_contact} <{party_a_contact.split()[0].lower()}@{company_a.lower().replace(' ', '')}.com>",
        "cc": "",
        "subject": f"Re: {contract_number} — Performance Concerns",
        "body": (
            f"Hi {party_a_contact.split()[0]},\n\n"
            f"Thank you for reaching out. We've been aware of some challenges and are "
            f"working to address them. Happy to schedule a meeting.\n\n"
            f"How about next {rng.choice(['Tuesday', 'Wednesday', 'Thursday'])}?\n\n"
            f"Best,\n{party_b_contact}"
        ),
    })
    email_id += 1

    # First meeting
    meeting1_day = min(28, response_day + rng.randint(3, 7))
    meeting1_date = _date(concern_year, concern_month, meeting1_day)

    # Determine if meeting has required participants violation
    if "missing_required_participant" in planted_issues:
        meeting1_attendees = [party_a_contact, party_b_contact, party_a_pm]
        # Missing: VP-level from Party B
        issues_detail.append({
            "type": "missing_required_participant",
            "description": (
                f"Meeting on {meeting1_date} lacked authorized VP-level representative from "
                f"{company_b} (per contract Section 8.2, decisions require {required_meeting_participants})"
            ),
        })
    else:
        meeting1_attendees = [party_a_contact, party_a_exec, party_b_contact, party_b_exec]

    meetings.append({
        "date": meeting1_date,
        "attendees": meeting1_attendees,
        "key_decisions": [
            f"{company_b} to provide remediation plan within 10 business days",
            f"Weekly status updates to begin immediately",
        ],
        "action_items": [
            f"{party_b_contact}: Submit remediation plan by {_date(concern_year, concern_month, min(28, meeting1_day + 10))}",
            f"{party_a_contact}: Provide detailed list of deficiencies by {_date(concern_year, concern_month, min(28, meeting1_day + 3))}",
        ],
    })
    calendar_events.append({"date": meeting1_date, "event": "Status meeting — performance review", "type": "meeting"})

    # Formal notice email
    notice_month = concern_month + 1
    notice_year = concern_year
    if notice_month > 12:
        notice_month -= 12
        notice_year += 1
    notice_day = rng.randint(3, 12)
    notice_date = _date(notice_year, notice_month, notice_day)

    # Deadline violation: response required within X days, but response came late
    if "deadline_violation" in planted_issues:
        deadline_date = _date(notice_year, notice_month, min(28, notice_day + response_period_days))
        actual_response_day = notice_day + response_period_days + rng.randint(3, 10)
        actual_response_month = notice_month
        if actual_response_day > 28:
            actual_response_day -= 28
            actual_response_month += 1
        if actual_response_month > 12:
            actual_response_month -= 12
        actual_response_date = _date(notice_year, actual_response_month, actual_response_day)
        issues_detail.append({
            "type": "deadline_violation",
            "description": (
                f"Formal notice sent {notice_date} required response within {response_period_days} days "
                f"(by {deadline_date}), but {company_b}'s response was not received until {actual_response_date}"
            ),
        })
    else:
        actual_response_day = notice_day + rng.randint(1, response_period_days - 1)
        actual_response_month = notice_month
        if actual_response_day > 28:
            actual_response_day -= 28
            actual_response_month += 1
        if actual_response_month > 12:
            actual_response_month -= 12
        actual_response_date = _date(notice_year, actual_response_month, actual_response_day)

    # Formal notice email (from Party A's counsel)
    emails.append({
        "id": email_id,
        "date": notice_date,
        "from": f"{party_a_counsel} <{party_a_counsel.split()[0].lower()}@lawfirm.com>",
        "to": f"{party_b_contact} <{party_b_contact.split()[0].lower()}@{company_b.lower().replace(' ', '')}.com>",
        "cc": f"{party_a_contact}, {party_b_counsel}",
        "subject": f"FORMAL NOTICE — Breach of Agreement {contract_number}",
        "body": (
            f"Dear {party_b_contact},\n\n"
            f"This letter constitutes formal notice of breach under Section 7.1 of "
            f"Agreement {contract_number} dated {contract_date}.\n\n"
            f"The specific breaches include failure to meet {milestones[0]['name']} "
            f"requirements as detailed in Exhibit A.\n\n"
            f"Pursuant to Section 7.3, you have {response_period_days} calendar days from "
            f"receipt of this notice to provide a written response and remediation plan.\n\n"
            f"Please direct all further communications on this matter to this office.\n\n"
            f"Sincerely,\n{party_a_counsel}\nCounsel for {company_a}"
        ),
    })
    email_id += 1

    # Improper notice
    if "improper_notice" in planted_issues:
        # Notice sent to wrong person/method
        issues_detail.append({
            "type": "improper_notice",
            "description": (
                f"Formal breach notice was sent via email to {party_b_contact}, but contract "
                f"Section 12.1 requires notice via {notice_method} to designated contact "
                f"({party_b_exec} at {company_b}'s registered address)"
            ),
        })

    # Party B's response (possibly late)
    emails.append({
        "id": email_id,
        "date": actual_response_date,
        "from": f"{party_b_counsel} <{party_b_counsel.split()[0].lower()}@lawfirm2.com>",
        "to": f"{party_a_counsel} <{party_a_counsel.split()[0].lower()}@lawfirm.com>",
        "cc": f"{party_a_contact}, {party_b_contact}",
        "subject": f"Re: FORMAL NOTICE — Agreement {contract_number}",
        "body": (
            f"Dear {party_a_counsel},\n\n"
            f"We acknowledge receipt of your notice dated {notice_date}. Our client "
            f"disputes the characterization of breach and will provide a detailed "
            f"response addressing each point.\n\n"
            f"We propose a meeting to resolve this matter amicably.\n\n"
            f"Regards,\n{party_b_counsel}\nCounsel for {company_b}"
        ),
    })
    email_id += 1

    # Contradictory commitment
    if "contradictory_commitment" in planted_issues:
        # Party B agrees to A in email but B in meeting
        commitment_a = rng.choice([
            "complete remediation by end of month",
            "provide full refund for defective deliverables",
            "extend timeline by 30 days at no additional cost",
            "assign additional resources to the project",
        ])
        commitment_b = rng.choice([
            "will only address critical defects, not all issues",
            "willing to offer 50% credit, not full refund",
            "extension subject to change order and additional fees",
            "current team is sufficient; no additional resources planned",
        ])

        contra_email_day = min(28, actual_response_day + rng.randint(2, 5))
        contra_email_month = actual_response_month
        if contra_email_day > 28:
            contra_email_day -= 28
            contra_email_month += 1
        if contra_email_month > 12:
            contra_email_month -= 12
        contra_email_date = _date(notice_year, contra_email_month, contra_email_day)

        emails.append({
            "id": email_id,
            "date": contra_email_date,
            "from": f"{party_b_contact} <{party_b_contact.split()[0].lower()}@{company_b.lower().replace(' ', '')}.com>",
            "to": f"{party_a_contact} <{party_a_contact.split()[0].lower()}@{company_a.lower().replace(' ', '')}.com>",
            "cc": "",
            "subject": f"Re: Path Forward on {contract_number}",
            "body": (
                f"Hi {party_a_contact.split()[0]},\n\n"
                f"Following our discussions, I want to confirm that we are prepared to "
                f"{commitment_a}. We value the relationship and want to make this right.\n\n"
                f"Best,\n{party_b_contact}"
            ),
        })
        email_id += 1

        # Contradicting meeting
        contra_meeting_day = min(28, contra_email_day + rng.randint(3, 7))
        contra_meeting_month = contra_email_month
        if contra_meeting_day > 28:
            contra_meeting_day -= 28
            contra_meeting_month += 1
        if contra_meeting_month > 12:
            contra_meeting_month -= 12
        contra_meeting_date = _date(notice_year, contra_meeting_month, contra_meeting_day)

        meetings.append({
            "date": contra_meeting_date,
            "attendees": [party_a_contact, party_a_exec, party_b_contact, party_b_exec],
            "key_decisions": [
                f"{company_b} position: {commitment_b}",
            ],
            "action_items": [
                f"{party_b_counsel}: Draft revised proposal reflecting meeting discussion",
            ],
        })
        calendar_events.append({"date": contra_meeting_date, "event": "Negotiation meeting", "type": "meeting"})

        issues_detail.append({
            "type": "contradictory_commitment",
            "description": (
                f"{party_b_contact} committed in email ({contra_email_date}) to '{commitment_a}' "
                f"but stated in meeting ({contra_meeting_date}) that '{commitment_b}' — contradictory positions"
            ),
        })

    # Unfulfilled commitment
    if "unfulfilled_commitment" in planted_issues:
        promise_text = rng.choice([
            "deliver updated specifications by Friday",
            "provide the revised project timeline within 5 business days",
            "send the audit documentation by end of week",
            "complete the acceptance testing by the agreed date",
        ])
        promise_email_month = notice_month
        promise_email_day = min(28, notice_day + rng.randint(5, 12))
        if promise_email_day > 28:
            promise_email_day -= 28
            promise_email_month += 1
        if promise_email_month > 12:
            promise_email_month -= 12
        promise_email_date = _date(notice_year, promise_email_month, promise_email_day)

        emails.append({
            "id": email_id,
            "date": promise_email_date,
            "from": f"{party_b_contact} <{party_b_contact.split()[0].lower()}@{company_b.lower().replace(' ', '')}.com>",
            "to": f"{party_a_contact} <{party_a_contact.split()[0].lower()}@{company_a.lower().replace(' ', '')}.com>",
            "cc": f"{party_a_pm}",
            "subject": f"Re: {contract_number} — Action Items",
            "body": (
                f"Hi {party_a_contact.split()[0]},\n\n"
                f"Confirming that we will {promise_text}. You have my word on this.\n\n"
                f"Best,\n{party_b_contact}"
            ),
        })
        email_id += 1

        issues_detail.append({
            "type": "unfulfilled_commitment",
            "description": (
                f"{party_b_contact} committed on {promise_email_date} to '{promise_text}' "
                f"but there is no evidence in the communications or meeting notes that this was delivered"
            ),
        })

    # Cure period expired
    if "cure_period_expired" in planted_issues:
        cure_start_month = notice_month
        cure_start_day = notice_day
        cure_end_day = cure_start_day + cure_period_days
        cure_end_month = cure_start_month
        if cure_end_day > 28:
            cure_end_day -= 28
            cure_end_month += 1
        if cure_end_month > 12:
            cure_end_month -= 12
        cure_end_date = _date(notice_year, cure_end_month, cure_end_day)

        issues_detail.append({
            "type": "cure_period_expired",
            "description": (
                f"Contract provides a {cure_period_days}-day cure period from notice date ({notice_date}), "
                f"expiring {cure_end_date}. No evidence of cure or remediation before expiration."
            ),
        })
        calendar_events.append({"date": cure_end_date, "event": f"Cure period expiration — Agreement {contract_number}", "type": "deadline"})

    # Add more routine emails to fill out the thread
    for i in range(rng.randint(8, 15)):
        extra_month = base_month + rng.randint(0, 4)
        extra_year = base_year
        if extra_month > 12:
            extra_month -= 12
            extra_year += 1
        extra_day = rng.randint(1, 28)
        extra_date = _date(extra_year, extra_month, extra_day)

        sender_pool = [
            (party_a_contact, company_a),
            (party_b_contact, company_b),
            (party_a_pm, company_a),
        ]
        sender_name, sender_co = rng.choice(sender_pool)
        recipient_pool = [(n, c) for n, c in sender_pool if n != sender_name]
        recip_name, recip_co = rng.choice(recipient_pool)

        subjects = [
            f"Re: {contract_number} — Status Update",
            f"Re: Weekly Progress Report",
            f"Re: {contract_number} — Schedule Discussion",
            f"Re: Meeting Follow-up",
            f"Re: Deliverable Review",
            f"Re: {contract_number} — Documentation",
        ]
        bodies = [
            f"Hi {recip_name.split()[0]},\n\nAttaching this week's progress report. Let me know if you have questions.\n\nBest,\n{sender_name}",
            f"Hi {recip_name.split()[0]},\n\nFollowing up on our earlier discussion. The team is making progress on the open items.\n\nRegards,\n{sender_name}",
            f"Hi {recip_name.split()[0]},\n\nPlease see the updated schedule attached. We're tracking against the milestones.\n\nThanks,\n{sender_name}",
            f"Hi {recip_name.split()[0]},\n\nQuick update: the team completed the review of {rng.choice(['Phase 1', 'Phase 2', 'the deliverables', 'the specifications'])}. Will share details in our next call.\n\nBest,\n{sender_name}",
        ]

        emails.append({
            "id": email_id,
            "date": extra_date,
            "from": f"{sender_name} <{sender_name.split()[0].lower()}@{sender_co.lower().replace(' ', '')}.com>",
            "to": f"{recip_name} <{recip_name.split()[0].lower()}@{recip_co.lower().replace(' ', '')}.com>",
            "cc": "",
            "subject": rng.choice(subjects),
            "body": rng.choice(bodies),
        })
        email_id += 1

    # Add more meetings
    for i in range(rng.randint(2, 4)):
        mtg_month = base_month + rng.randint(1, 4)
        mtg_year = base_year
        if mtg_month > 12:
            mtg_month -= 12
            mtg_year += 1
        mtg_day = rng.randint(1, 28)
        mtg_date = _date(mtg_year, mtg_month, mtg_day)

        meetings.append({
            "date": mtg_date,
            "attendees": rng.sample(
                [party_a_contact, party_a_exec, party_a_pm, party_b_contact, party_b_exec],
                rng.randint(3, 5)
            ),
            "key_decisions": [
                rng.choice([
                    "Continue with current timeline",
                    "Escalate unresolved items to executive sponsors",
                    "Schedule follow-up meeting for detailed review",
                    "Both parties to review contract terms for applicability",
                ]),
            ],
            "action_items": [
                f"{rng.choice([party_a_pm, party_b_contact])}: {rng.choice(['Circulate meeting minutes', 'Update project plan', 'Review open items list'])}",
            ],
        })
        calendar_events.append({"date": mtg_date, "event": "Status meeting", "type": "meeting"})

    # Sort emails and meetings by date
    emails.sort(key=lambda e: e["date"])
    meetings.sort(key=lambda m: m["date"])
    calendar_events.sort(key=lambda c: c["date"])

    # --- FALSE POSITIVES ---
    false_positives: list[dict] = []

    # FP1: Email that looks late but response was within weekend-adjusted period
    # (business days vs calendar days, and contract says "calendar days")
    fp1_description = (
        f"Party B's response to the initial concern email appears slow "
        f"({response_date}, several days after {concern_date}) but was within normal "
        f"business response time and no contractual deadline applied to informal communications"
    )
    false_positives.append({"name": "informal_response_timing", "description": fp1_description})

    # FP2: A meeting without both counsels present, which looks like a violation
    # but contract only requires counsel for formal dispute resolution proceedings
    fp2_description = (
        f"Several meetings occurred without legal counsel present, which may appear "
        f"improper but the contract only requires counsel participation in formal "
        f"dispute resolution proceedings (Section 9), not status meetings"
    )
    false_positives.append({"name": "counsel_at_meetings", "description": fp2_description})

    # FP3: CC list inconsistency that looks like deliberate exclusion
    fp3_description = (
        f"Some emails in the thread do not CC {party_a_pm}, which may appear like "
        f"exclusion from communications, but the contract does not require the project "
        f"manager to be copied on all correspondence"
    )
    false_positives.append({"name": "cc_list_inconsistency", "description": fp3_description})

    # --- BUILD FILES ---

    # 1. email_thread.txt
    email_lines = [
        f"EMAIL COMMUNICATIONS — {contract_number}",
        f"Parties: {company_a} ({dispute['party_a_role']}) and {company_b} ({dispute['party_b_role']})",
        "",
        "=" * 70,
    ]
    for em in emails:
        email_lines.extend([
            "",
            f"--- Email #{em['id']} ---",
            f"Date: {em['date']}",
            f"From: {em['from']}",
            f"To: {em['to']}",
        ])
        if em['cc']:
            email_lines.append(f"CC: {em['cc']}")
        email_lines.extend([
            f"Subject: {em['subject']}",
            "",
            em['body'],
            "",
        ])
    email_thread = "\n".join(email_lines) + "\n"

    # 2. meeting_notes.csv
    mtg_csv_lines = ["date,attendees,key_decisions,action_items"]
    for mtg in meetings:
        attendees_str = "; ".join(mtg["attendees"])
        decisions_str = "; ".join(mtg["key_decisions"])
        actions_str = "; ".join(mtg["action_items"])
        mtg_csv_lines.append(f'"{mtg["date"]}","{attendees_str}","{decisions_str}","{actions_str}"')
    meeting_notes_csv = "\n".join(mtg_csv_lines) + "\n"

    # 3. contract_terms.txt
    contract_lines = [
        f"AGREEMENT {contract_number}",
        f"Effective Date: {contract_date}",
        "",
        f"BETWEEN:",
        f"  {company_a} (\"{dispute['party_a_role']}\")",
        f"  {company_b} (\"{dispute['party_b_role']}\")",
        "",
        "=" * 60,
        "SELECTED PROVISIONS (Relevant to Current Dispute)",
        "=" * 60,
        "",
        "SECTION 3: MILESTONES AND DELIVERABLES",
        "",
    ]
    for m in milestones:
        contract_lines.append(f"  {m['name']} — {m['description']}: Due by {m['due_date']}")
    contract_lines.extend([
        "",
        "SECTION 5: PAYMENT TERMS",
        "",
        f"  5.1 Payment is due within 30 days of {dispute['party_b_role']}'s invoice",
        f"  5.2 {dispute['party_a_role']} may withhold payment for disputed deliverables",
        f"  5.3 Interest on late payment: 1.5% per month",
        "",
        "SECTION 7: BREACH AND REMEDIES",
        "",
        f"  7.1 A party is in breach if it fails to perform any material obligation",
        f"  7.2 The non-breaching party must provide written notice of breach",
        f"  7.3 Response to breach notice: {response_period_days} calendar days from receipt",
        f"  7.4 Cure period: {cure_period_days} calendar days from date of notice",
        f"  7.5 If breach is not cured within the cure period, the non-breaching party",
        f"      may terminate the agreement and pursue available remedies",
        "",
        "SECTION 8: MEETINGS AND GOVERNANCE",
        "",
        f"  8.1 Regular status meetings shall occur no less than bi-weekly",
        f"  8.2 Decisions affecting scope, timeline, or cost require attendance of",
        f"      {required_meeting_participants}",
        f"  8.3 Meeting minutes must be circulated within 2 business days",
        "",
        "SECTION 9: DISPUTE RESOLUTION",
        "",
        f"  9.1 Disputes shall first be escalated to executive sponsors",
        f"  9.2 If not resolved within 30 days, parties shall engage in {dispute_resolution}",
        f"  9.3 Governing law: State of {governing_law}",
        "",
        "SECTION 12: NOTICES",
        "",
        f"  12.1 All formal notices under this agreement must be delivered via",
        f"       {notice_method}",
        f"  12.2 Notice to {company_a}: Attn: {party_a_exec}, {company_a} registered address",
        f"  12.3 Notice to {company_b}: Attn: {party_b_exec}, {company_b} registered address",
        f"  12.4 Notice is deemed received on the date of delivery confirmation",
        "",
        "SECTION 15: GENERAL",
        "",
        f"  15.1 Amendments must be in writing signed by both parties",
        f"  15.2 No waiver of any provision shall be deemed a continuing waiver",
        f"  15.3 This agreement constitutes the entire agreement between the parties",
        "",
    ])
    contract_terms = "\n".join(contract_lines) + "\n"

    # 4. calendar_events.csv
    cal_csv_lines = ["date,event,type"]
    for ce in calendar_events:
        cal_csv_lines.append(f'"{ce["date"]}","{ce["event"]}","{ce["type"]}"')
    calendar_events_csv = "\n".join(cal_csv_lines) + "\n"

    # --- BUILD RUBRIC ---
    n_emails = len(emails)
    n_meetings = len(meetings)

    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does /testbed/timeline_analysis.txt exist with substantial content?",
            points=1,
        ),
        BinaryRubricCategory(
            name="identifies_dispute_type",
            question=f'Does the analysis correctly identify the dispute as "{dispute["label"]}"?',
            points=1,
        ),
        BinaryRubricCategory(
            name="identifies_parties",
            question=f"Does the analysis identify both parties ({company_a} and {company_b}) and their roles?",
            points=1,
        ),
        BinaryRubricCategory(
            name="identifies_contract_number",
            question=f"Does the analysis reference the contract number ({contract_number})?",
            points=1,
        ),
        BinaryRubricCategory(
            name="reconstructs_timeline",
            question="Does the analysis present a reconstructed chronological timeline of key events?",
            points=2,
        ),
        BinaryRubricCategory(
            name="references_contract_terms",
            question="Does the analysis reference specific contract sections when identifying issues?",
            points=2,
        ),
        BinaryRubricCategory(
            name="cross_references_sources",
            question="Does the analysis cross-reference at least 3 of the 4 source documents to identify issues?",
            points=2,
        ),
        BinaryRubricCategory(
            name="identifies_response_period",
            question=f"Does the analysis note the {response_period_days}-day response period from the contract (Section 7.3)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="identifies_cure_period",
            question=f"Does the analysis note the {cure_period_days}-day cure period from the contract (Section 7.4)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="identifies_notice_requirements",
            question=f"Does the analysis reference the notice delivery requirement (Section 12.1: {notice_method})?",
            points=2,
        ),
        BinaryRubricCategory(
            name="identifies_meeting_requirement",
            question=f"Does the analysis reference the meeting participant requirement (Section 8.2: {required_meeting_participants})?",
            points=1,
        ),
        BinaryRubricCategory(
            name="identifies_dispute_resolution",
            question=f"Does the analysis reference the dispute resolution mechanism ({dispute_resolution})?",
            points=1,
        ),
    ]

    # Per-issue checks (2pt each)
    for issue in issues_detail:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"identifies_{issue['type']}",
                question=f"Does the analysis identify this issue: {issue['description']}?",
                points=2,
            )
        )

    rubric_items.append(
        BinaryRubricCategory(
            name="correct_issue_count",
            question=f"Does the analysis identify approximately {len(issues_detail)} issues (within +/- 1)?",
            points=2,
        )
    )

    # False positive checks (cap at 2 to control total)
    for fp in false_positives[:2]:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"no_false_positive_{fp['name']}",
                question=(
                    f"Does the analysis correctly NOT flag the following as a violation: "
                    f"{fp['description']}?"
                ),
                points=1,
            )
        )

    rubric_items.extend([
        BinaryRubricCategory(
            name="identifies_key_dates",
            question=(
                f"Does the analysis identify the notice date ({notice_date}) and response deadline "
                f"as key dates in the dispute?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="assesses_legal_position",
            question="Does the analysis assess which party has the stronger legal position based on the evidence?",
            points=2,
        ),
        RubricCategory(
            name="legal_analysis_quality",
            description="How well does the analysis apply contract terms to the facts?",
            failure="No reference to contract provisions",
            minor_failure="Mentions contract but doesn't connect specific sections to specific events",
            minor_success="Good connection between contract terms and events but some gaps",
            success="Systematic analysis connecting each issue to specific contract provisions with legal implications",
            points=3,
        ),
    ])

    problem_statement = f"""# Communications Timeline Analysis — Business Dispute

{company_a} ({dispute['party_a_role']}) and {company_b} ({dispute['party_b_role']}) are
in a dispute regarding {dispute['description']}.

You are a legal analyst. Reconstruct the communication timeline and identify any
issues that may have legal significance for resolving this dispute.

## Source Files
- /testbed/data/email_thread.txt — Email communications between the parties
- /testbed/data/meeting_notes.csv — Meeting records with attendees, decisions, and action items
- /testbed/data/contract_terms.txt — Relevant contract provisions
- /testbed/data/calendar_events.csv — Scheduled events, deadlines, and filing dates

## Requirements
1. Reconstruct a chronological timeline from all sources
2. Identify deadline violations (response required within X days but response late)
3. Find contradictory commitments (agreed to A in one place, B in another)
4. Check for missing required participants in key meetings
5. Identify commitments made but never fulfilled
6. Verify that notices were properly served per contract requirements
7. Check cure period compliance
8. Note any clean items that look problematic but are actually compliant

Write your analysis to /testbed/timeline_analysis.txt."""

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=tuple(rubric_items),
        submission_instructions="Write your timeline analysis to /testbed/timeline_analysis.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/email_thread.txt": email_thread,
            "/testbed/data/meeting_notes.csv": meeting_notes_csv,
            "/testbed/data/contract_terms.txt": contract_terms,
            "/testbed/data/calendar_events.csv": calendar_events_csv,
        },
        problem_type="communications_timeline_analysis",
    )
