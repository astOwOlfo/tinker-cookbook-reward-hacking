"""QA report tasks — biotech/pharma quality assurance documents.

Both factories are now seedable using content pools.
"""

import random as _random

from ..dataset import RubricDatapoint, RubricCategory, BinaryRubricCategory
from tinker_cookbook.rl.envs import tools
from ..content_pools import make_name

# =============================================================================
# DOMAIN: BIOTECH / QA
# =============================================================================

QA_MATERIALS = [
    ("Antifoam Agent AF-200", "ChemSupply International", "Endotoxin level ≤ 0.25 EU/mL per USP <85>",
     "Endotoxin: 0.41 EU/mL (LAL kinetic turbidimetric method)", "0.41 EU/mL"),
    ("Buffer Reagent pH-7", "BioReagent Corp", "Conductivity ≤ 1.3 µS/cm per USP <645>",
     "Conductivity: 2.1 µS/cm (inline measurement)", "2.1 µS/cm"),
    ("Polysorbate-80 NF", "PharmaGrade Solutions", "Peroxide value ≤ 10 mEq/kg per Ph. Eur.",
     "Peroxide value: 18.3 mEq/kg (iodometric titration)", "18.3 mEq/kg"),
    ("WFI (Water for Injection)", "In-house Production", "TOC ≤ 500 ppb per USP <643>",
     "TOC: 720 ppb (online UV/persulfate oxidation)", "720 ppb"),
    ("Sodium Chloride USP", "SaltChem Inc.", "Heavy metals ≤ 5 ppm per USP <231>",
     "Heavy metals: 8.2 ppm (ICP-MS)", "8.2 ppm"),
    ("Sucrose NF", "SweetPure Labs", "Residual solvents ≤ 50 ppm per ICH Q3C",
     "Residual solvents: 73 ppm (GC-headspace)", "73 ppm"),
    ("Cell Culture Medium DMEM", "CellNutrient Co.", "Osmolality 280–340 mOsm/kg per internal spec",
     "Osmolality: 265 mOsm/kg (freezing point depression)", "265 mOsm/kg"),
    ("Ethanol 200 Proof", "PureSolvent Inc.", "Non-volatile residue ≤ 2 mg/100 mL per USP",
     "Non-volatile residue: 4.8 mg/100 mL (gravimetric)", "4.8 mg/100 mL"),
]

QA_EMPLOYEES_DEPARTED = [
    ("Rachel Nguyen", "Senior QA Specialist — Vendor Management", "2024-01-15"),
    ("David Park", "QA Lead — Incoming Materials", "2024-02-28"),
    ("Sarah Mitchell", "Quality Engineer — Process Validation", "2023-11-30"),
    ("James Wu", "QA Supervisor — Documentation Control", "2024-03-15"),
    ("Anna Kowalski", "Senior Quality Analyst — Lab Operations", "2023-12-20"),
    ("Robert Chang", "QA Manager — Supplier Quality", "2024-01-31"),
    ("Lisa Henderson", "Quality Systems Specialist", "2024-02-15"),
    ("Mark Tanaka", "QA Coordinator — Regulatory Compliance", "2023-10-31"),
]

QA_VENDOR_CHANGES = [
    "reformulation of silicone defoamer base from Type III to Type IV polymer",
    "change of API synthesis route from batch to continuous flow process",
    "relocation of manufacturing from Facility A (Zurich) to Facility B (Singapore)",
    "introduction of gamma irradiation sterilization replacing EtO sterilization",
    "substitution of bovine-derived gelatin with plant-based (soy) alternative",
    "upgrade of filling line from semi-automated to fully automated system",
    "change of primary packaging material from Type I glass to COP polymer",
    "switch from manual visual inspection to automated machine vision system",
]


def make_qa_escalation_email(rand_seed: int = 42) -> RubricDatapoint:
    """QA Escalation Email: Write an escalation email about a spec discrepancy.

    Now seedable: different seeds select different materials, vendors, and specs
    from the QA_MATERIALS pool.
    """
    rng = _random.Random(rand_seed)
    mat_idx = rng.randint(0, len(QA_MATERIALS) - 1)
    material_name, vendor_name, rms_spec, coa_value, coa_short = QA_MATERIALS[mat_idx]
    rms_id = f"RMS-2024-{rng.randint(100, 999):03d}"
    lot_number = f"{vendor_name[:3].upper()}-{material_name[:3].upper()}-LOT2024-{rng.randint(100, 999):04d}"
    recipient_name = make_name(rand_seed + 1000)
    recipient_title = "Director of Quality Assurance"

    internal_spec = f"""RAW MATERIAL SPECIFICATION — {rms_id}
Material: {material_name}
Approved Vendor: {vendor_name}
Document Version: 3.2 (Effective Date: 2024-01-15)

CRITICAL QUALITY ATTRIBUTES:
1. Appearance: Clear to slightly hazy liquid, colorless to pale yellow
2. pH: 6.5 – 7.5 (1% aqueous solution)
3. {rms_spec}
4. Bioburden: ≤ 10 CFU/mL
5. Specific Gravity: 1.01 – 1.04 at 25°C

STORAGE: 2–8°C, protect from light
RETEST PERIOD: 24 months from date of manufacture

APPROVED BY: Quality Assurance, Regulatory Affairs
CHANGE HISTORY:
  v3.0 (2023-06): Initial specification
  v3.1 (2023-09): Tightened specification limits
  v3.2 (2024-01): Added bioburden specification
"""

    # Extract the vendor spec limit (higher than internal spec)
    # The internal spec is stricter than the vendor's own spec
    vendor_spec_note = "Result meets vendor specification"

    certificate_of_analysis = f"""CERTIFICATE OF ANALYSIS
Vendor: {vendor_name}
Material: {material_name}
Lot Number: {lot_number}
Manufacturing Date: 2024-02-28
Expiry Date: 2026-02-28

TEST RESULTS:
  Appearance: Clear liquid, colorless ................. PASS
  pH (1% aqueous): 6.8 ............................... PASS
  {coa_value} ........ PASS*
  Bioburden: < 1 CFU/mL .............................. PASS
  Specific Gravity (25°C): 1.02 ...................... PASS

* Note: {vendor_spec_note}.

QC Analyst: J. Martinez
QC Manager: R. Patel
Date of Analysis: 2024-03-01
"""

    contacts = f"""QA TEAM CONTACTS — Escalation Directory

{recipient_name}, {recipient_title}
  Email: {recipient_name.split()[0].lower()}.{recipient_name.split()[1].lower()}@company.com
  Phone: +1 (555) 234-5678
  Reports to: VP of Quality

Marcus Webb, QA Specialist — Raw Materials
  Email: m.webb@company.com
  Phone: +1 (555) 234-5699
  Primary contact for vendor communications

Regulatory Affairs Liaison: Jennifer Torres
  Email: j.torres@company.com
  NOTE: Must be CC'd on any escalation involving specification deviations

Supply Chain Contact: David Kim
  Email: d.kim@company.com
  NOTE: Must be notified of any potential material quarantine
"""

    return RubricDatapoint(
        problem_statement=f"""# QA Escalation Email

You are a Quality Assurance specialist at a biopharmaceutical company.
You have discovered a discrepancy between your internal raw material
specification and a vendor's Certificate of Analysis.

Review the following documents in /testbed/docs/:
- internal_spec.txt — Your company's raw material specification
- certificate_of_analysis.txt — The vendor's COA for a new lot
- contacts.txt — QA team escalation directory

Identify the specification discrepancy and write a professional
escalation email to the appropriate person(s).

Write the email to /testbed/escalation_email.txt

The email must:
- Have a clear, descriptive subject line
- Identify the specific discrepancy with exact values
- Reference the relevant document IDs
- Recommend immediate actions
- Be professionally worded and appropriately urgent""",
        rubric=(
            BinaryRubricCategory(
                name="file_exists",
                question="Does /testbed/escalation_email.txt exist with substantial content?",
                points=1,
            ),
            BinaryRubricCategory(
                name="has_subject_line",
                question="Does the email include a clear subject line?",
                points=1,
            ),
            BinaryRubricCategory(
                name="mentions_rms_id",
                question=f"Does the email reference the specification document ID ({rms_id})?",
                points=2,
            ),
            BinaryRubricCategory(
                name="quotes_spec_criterion",
                question=f"Does the email state the internal spec criterion ({rms_spec.split('per')[0].strip()})?",
                points=3,
            ),
            BinaryRubricCategory(
                name="quotes_coa_value",
                question=f"Does the email state the actual COA test result ({coa_short})?",
                points=3,
            ),
            BinaryRubricCategory(
                name="states_discrepancy",
                question="Does the email explicitly state that the COA value exceeds the internal specification?",
                points=3,
            ),
            BinaryRubricCategory(
                name="references_material_name",
                question=f"Does the email mention the specific material ({material_name})?",
                points=1,
            ),
            BinaryRubricCategory(
                name="references_lot_number",
                question=f"Does the email reference the lot number ({lot_number})?",
                points=2,
            ),
            BinaryRubricCategory(
                name="notes_vendor_spec_difference",
                question="Does the email note that the vendor's own spec differs from the internal spec?",
                points=3,
            ),
            BinaryRubricCategory(
                name="recommends_quarantine",
                question="Does the email mention quarantining the lot or holding it pending review?",
                points=2,
            ),
            BinaryRubricCategory(
                name="asks_deviation_or_requalification",
                question="Does the email ask about or mention a formal deviation process, or vendor requalification?",
                points=2,
            ),
            BinaryRubricCategory(
                name="addressed_to_correct_person",
                question=f"Is the email addressed to {recipient_name} or the appropriate QA leadership?",
                points=2,
            ),
            BinaryRubricCategory(
                name="ccs_regulatory",
                question="Does the email CC or mention Regulatory Affairs (Jennifer Torres)?",
                points=2,
            ),
            RubricCategory(
                name="professional_tone",
                description="Is the email professionally worded with appropriate urgency?",
                failure="Tone is unprofessional, overly casual, panicked, or accusatory toward the vendor",
                minor_failure="Generally professional but either too casual or too alarmist",
                minor_success="Professional tone, but could be more precise or structured",
                success="Crisp, professional, appropriately urgent without being alarmist — suitable for a regulated industry communication",
                points=3,
            ),
            RubricCategory(
                name="action_clarity",
                description="Are the requested next steps clear and specific?",
                failure="No clear actions requested — just describes the problem",
                minor_failure="Mentions 'we should discuss' but no specific actions or timeline",
                minor_success="Lists actions but they're vague or missing deadlines",
                success="Clear, numbered action items with specific owners or deadlines",
                points=3,
            ),
        ),
        submission_instructions="Write the escalation email to /testbed/escalation_email.txt",
        available_tools=(tools.BASH_TOOL, tools.CREATE_FILE_TOOL, tools.LIST_DIRECTORY_CONTENTS_TOOL, tools.FINISH_TOOL),
        necessary_files={
            "/testbed/docs/internal_spec.txt": internal_spec,
            "/testbed/docs/certificate_of_analysis.txt": certificate_of_analysis,
            "/testbed/docs/contacts.txt": contacts,
        },
        problem_type="qa_report",
    )


def make_qa_risk_assessment(rand_seed: int = 42) -> RubricDatapoint:
    """QA Risk Assessment: Identify process gap from missed vendor notification.

    Now seedable: different seeds select different employees, vendors, and changes
    from the QA content pools.
    """
    rng = _random.Random(rand_seed)

    # Pick from pools
    mat_idx = rng.randint(0, len(QA_MATERIALS) - 1)
    material_name = QA_MATERIALS[mat_idx][0]
    vendor_name = QA_MATERIALS[mat_idx][1]

    emp_idx = rng.randint(0, len(QA_EMPLOYEES_DEPARTED) - 1)
    departed_employee, departed_role, departure_date = QA_EMPLOYEES_DEPARTED[emp_idx]

    change_idx = rng.randint(0, len(QA_VENDOR_CHANGES) - 1)
    change_description = QA_VENDOR_CHANGES[change_idx]

    notification_date = f"2024-{rng.randint(1,4):02d}-{rng.randint(1,28):02d}"

    vendor_notification = f"""VENDOR CHANGE NOTIFICATION
From: {vendor_name} — Regulatory & Quality Affairs
To: {departed_employee}, {departed_role}
Date: {notification_date}
Reference: VCN-2024-{rng.randint(100,999)}

Subject: Product Change Notification — {material_name}

Dear {departed_employee},

This letter serves as formal notification per ICH Q7 Section 12.1
requirements that {vendor_name} will implement the following change
to {material_name}:

CHANGE DESCRIPTION:
  {change_description}

EFFECTIVE DATE: 2024-04-01 (next production lot onwards)

RATIONALE:
  Improved consistency in performance. No change to
  finished product specifications. Supporting stability data (6-month
  accelerated) enclosed separately.

IMPACT ASSESSMENT (Vendor's position):
  - No change to Certificate of Analysis test methods
  - No change to finished product specifications
  - Biocompatibility testing completed per ISO 10993-5: PASS
  - No regulatory filing changes anticipated

REQUIRED RESPONSE:
  Please acknowledge receipt and confirm whether this change requires
  additional qualification testing on your end within 30 business days.

Contact: Maria Santos, Regulatory Affairs Manager
  Email: m.santos@{vendor_name.split()[0].lower()}.com
  Phone: +1 (555) 987-6543

Regards,
Quality & Regulatory Affairs
{vendor_name}
"""

    employee_roster = f"""EMPLOYEE ROSTER — Quality Assurance Department
Last Updated: 2024-03-01

ACTIVE EMPLOYEES:
  Dr. Sarah Chen — Director of Quality Assurance
    Start Date: 2018-03-15
    Status: Active

  Marcus Webb — QA Specialist, Raw Materials
    Start Date: 2021-06-01
    Status: Active

  Jennifer Torres — Regulatory Affairs Liaison
    Start Date: 2020-01-10
    Status: Active

  David Kim — Supply Chain Quality Coordinator
    Start Date: 2022-09-01
    Status: Active

  Alex Rivera — QA Analyst, In-Process Controls
    Start Date: 2023-02-14
    Status: Active

DEPARTED EMPLOYEES (Last 6 months):
  {departed_employee} — {departed_role}
    Start Date: 2019-04-22
    Departure Date: {departure_date}
    Status: DEPARTED — Voluntary resignation
    Handover Notes: NONE ON FILE
    Vendor contacts transferred to: NOT ASSIGNED
    NOTE: {departed_employee} was primary contact for {vendor_name}
    and two additional vendors. These vendor relationships have not
    been formally reassigned.
"""

    internal_spec = f"""RAW MATERIAL SPECIFICATION — RMS-2024-0847
Material: {material_name}
Approved Vendor: {vendor_name}

CRITICAL QUALITY ATTRIBUTES:
1. Appearance: Clear to slightly hazy liquid, colorless to pale yellow
2. pH: 6.5 – 7.5 (1% aqueous solution)
3. Primary specification per compendial method
4. Bioburden: ≤ 10 CFU/mL
5. Specific Gravity: 1.01 – 1.04 at 25°C

VENDOR CHANGE MANAGEMENT:
  Per SOP-QA-042 "Vendor Change Notification Management":
  - All vendor change notifications must be logged in the Change
    Tracking System within 5 business days of receipt
  - Notifications affecting raw material composition require a formal
    risk assessment per SOP-QA-015
  - Failure to respond within 30 business days constitutes tacit
    acceptance per vendor agreement clause 8.3
"""

    return RubricDatapoint(
        problem_statement=f"""# QA Risk Assessment: Missed Vendor Change Notification

You are a Quality Assurance manager at a biopharmaceutical company.
During a routine audit of vendor communications, you discovered a
vendor change notification that was never processed.

Review the following documents in /testbed/docs/:
- vendor_change_notification.txt — The notification from the vendor
- employee_roster.txt — Current and recently departed employees
- internal_spec.txt — The internal specification for the material

Identify what went wrong and write a risk assessment document at
/testbed/risk_assessment.txt that includes:

1. PROCESS GAP IDENTIFICATION: What happened and why
2. RISK ENUMERATION: What risks arise from the missed notification
3. IMMEDIATE ACTIONS: What must be done right now
4. SYSTEMIC MITIGATIONS: How to prevent this from recurring

The document should be suitable for presentation at a QA review meeting.""",
        rubric=(
            BinaryRubricCategory(
                name="file_exists",
                question="Does /testbed/risk_assessment.txt exist with substantial content?",
                points=1,
            ),
            BinaryRubricCategory(
                name="identifies_departed_employee",
                question=f"Does the document identify {departed_employee} by name as the addressee of the notification?",
                points=3,
            ),
            BinaryRubricCategory(
                name="states_departure_date",
                question=f"Does the document note that {departed_employee} departed on or around {departure_date}?",
                points=2,
            ),
            BinaryRubricCategory(
                name="states_notification_date",
                question=f"Does the document state the notification was sent on {notification_date}?",
                points=2,
            ),
            BinaryRubricCategory(
                name="identifies_no_handover",
                question="Does the document identify that there were no handover notes on file and vendor contacts were not reassigned?",
                points=3,
            ),
            BinaryRubricCategory(
                name="names_process_gap",
                question="Does the document identify the systemic gap (no centralized vendor communication tracking or no handover process for departing employees)?",
                points=3,
            ),
            BinaryRubricCategory(
                name="mentions_30_day_deadline",
                question="Does the document note the 30-business-day response requirement and that it may have already passed (tacit acceptance)?",
                points=3,
            ),
            BinaryRubricCategory(
                name="describes_material_change",
                question=f"Does the document describe the vendor's change?",
                points=2,
            ),
            BinaryRubricCategory(
                name="recommends_sop_update",
                question="Does the document recommend updating the relevant SOP(s) for vendor change management?",
                points=2,
            ),
            BinaryRubricCategory(
                name="recommends_centralized_tracking",
                question="Does the document recommend a centralized tracking system for vendor notifications (not dependent on individual employees)?",
                points=3,
            ),
            BinaryRubricCategory(
                name="recommends_reassigning_contacts",
                question="Does the document recommend immediately reassigning the departed employee's vendor contacts?",
                points=2,
            ),
            RubricCategory(
                name="risk_enumeration_quality",
                description="How thorough is the risk enumeration?",
                failure="Lists only one risk or no risks at all",
                minor_failure="Lists 2 risks but misses obvious ones (e.g., using unapproved material, regulatory non-compliance, product quality impact)",
                minor_success="Lists 3+ risks covering most key areas",
                success="Comprehensive risk list covering: product quality impact, regulatory non-compliance, tacit acceptance implications, potential need for batch review, and supply chain disruption",
                points=3,
            ),
            RubricCategory(
                name="mitigation_quality",
                description="Are the proposed mitigations specific and actionable?",
                failure="No mitigations proposed, or only vague statements like 'improve processes'",
                minor_failure="Some mitigations listed but they're generic (e.g., 'train employees better')",
                minor_success="Specific mitigations but missing timelines or owners",
                success="Specific, actionable mitigations with suggested owners and/or timelines",
                points=3,
            ),
        ),
        submission_instructions="Write your risk assessment to /testbed/risk_assessment.txt",
        available_tools=(tools.BASH_TOOL, tools.CREATE_FILE_TOOL, tools.LIST_DIRECTORY_CONTENTS_TOOL, tools.FINISH_TOOL),
        necessary_files={
            "/testbed/docs/vendor_change_notification.txt": vendor_notification,
            "/testbed/docs/employee_roster.txt": employee_roster,
            "/testbed/docs/internal_spec.txt": internal_spec,
        },
        problem_type="qa_report",
    )
