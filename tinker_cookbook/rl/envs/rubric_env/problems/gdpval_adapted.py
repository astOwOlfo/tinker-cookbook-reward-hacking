"""GDPval-inspired tasks: document analysis, financial computation, and professional review.

Each factory generates a realistic scenario with necessary_files containing all
information required to solve the problem. Seeds control randomization of amounts,
names, which items are problematic, etc.
"""

import random as _random

from ..dataset import RubricDatapoint, RubricCategory, BinaryRubricCategory
from tinker_cookbook.rl.envs import tools
from ..content_pools import (
    random_name,
    random_names,
    pick1,
    pick,
    vary_int,
    vary_number,
    COMPANY_NAMES,
)

# =============================================================================
# DOMAIN: VENDOR NAMES
# =============================================================================

VENDOR_NAMES = [
    "TechForward Solutions", "DataBridge Analytics", "CloudNine Hosting",
    "Apex Software Group", "Streamline IT Services", "CoreLogic Systems",
    "BrightPath Consulting", "NovaTech Industries", "PrimeStar Digital",
    "ClearView Networks", "Quantum Leap Technologies", "BlueSky Infrastructure",
    "IronClad Security", "SwiftCode Development", "Evergreen IT Solutions",
    "PeakPerformance Tech", "Lighthouse Data Corp", "Granite Systems Inc.",
]

# =============================================================================
# DOMAIN: CONTRACT REVIEW
# =============================================================================

PROBLEMATIC_CLAUSES = [
    {
        "name": "unlimited_liability",
        "clause": 'The Vendor shall be liable for all direct, indirect, incidental, consequential, and punitive damages arising from or related to this Agreement, without limitation.',
        "issue": "Unlimited liability clause — vendor has uncapped exposure for all damage types including consequential",
        "severity": "HIGH",
    },
    {
        "name": "unilateral_termination",
        "clause": "The Client may terminate this Agreement at any time, for any reason, with immediate effect and without obligation to pay for work in progress.",
        "issue": "Unilateral termination without payment for work in progress",
        "severity": "HIGH",
    },
    {
        "name": "auto_renewal_trap",
        "clause": "This Agreement shall automatically renew for successive one-year terms unless either party provides written notice of non-renewal at least 180 days prior to the end of the then-current term.",
        "issue": "Auto-renewal with 180-day notice requirement is excessively long and easy to miss",
        "severity": "MEDIUM",
    },
    {
        "name": "broad_ip_assignment",
        "clause": "All intellectual property, including but not limited to inventions, designs, works of authorship, and trade secrets, created by the Vendor during the term of this Agreement, whether or not related to the services performed hereunder, shall be the sole property of the Client.",
        "issue": "IP assignment extends to ALL work during the contract period, not just work related to the engagement",
        "severity": "HIGH",
    },
    {
        "name": "excessive_non_compete",
        "clause": "For a period of 24 months following termination, the Vendor shall not provide services to any entity that competes, directly or indirectly, with the Client in any market in which the Client operates or plans to operate.",
        "issue": "Non-compete is excessively broad (24 months, any market, includes planned markets)",
        "severity": "HIGH",
    },
    {
        "name": "unilateral_scope_change",
        "clause": "The Client reserves the right to modify the scope of services at any time. The Vendor shall accommodate such modifications without additional compensation unless the modification increases the total project effort by more than 50%.",
        "issue": "Client can expand scope by up to 50% without additional pay",
        "severity": "MEDIUM",
    },
    {
        "name": "indemnification_imbalance",
        "clause": "The Vendor shall indemnify, defend, and hold harmless the Client from any and all claims, damages, losses, and expenses (including reasonable attorney fees) arising from the Vendor's performance of services. The Client shall have no reciprocal indemnification obligation.",
        "issue": "One-sided indemnification — vendor indemnifies client but not vice versa",
        "severity": "MEDIUM",
    },
    {
        "name": "vague_acceptance",
        "clause": 'Deliverables shall be deemed accepted unless the Client notifies the Vendor of deficiencies within a "reasonable time" after delivery. The Client retains the right to reject deliverables that do not meet the Client\'s expectations.',
        "issue": "Acceptance criteria are vague — 'reasonable time' and 'expectations' are undefined",
        "severity": "MEDIUM",
    },
]

SAFE_CLAUSES = [
    "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflicts of law provisions.",
    "Neither party shall be liable for any failure or delay in performance due to circumstances beyond its reasonable control, including but not limited to acts of God, natural disasters, war, terrorism, or government actions.",
    "Any dispute arising out of or relating to this Agreement shall first be submitted to mediation in accordance with the rules of the American Arbitration Association before either party may initiate litigation.",
    "The Vendor shall maintain commercially reasonable security measures to protect the Client's confidential information, including encryption of data at rest and in transit.",
    "Either party may terminate this Agreement for cause upon 30 days' written notice if the other party materially breaches any term of this Agreement and fails to cure such breach within the notice period.",
]

# =============================================================================
# DOMAIN: COMPLIANCE / EXPENSE AUDIT
# =============================================================================

COMPLIANCE_VIOLATION_TYPES = [
    {
        "type": "over_limit_meal",
        "description": "Meal expense exceeding per-person daily limit",
        "policy_section": "Section 2 (Meals and Entertainment)",
        "example_amount": 112.50,
    },
    {
        "type": "missing_receipt",
        "description": "Expense over $25 submitted without receipt",
        "policy_section": "Section 1 (General Rules)",
        "example_amount": 68.00,
    },
    {
        "type": "unapproved_travel_class",
        "description": "Business-class flight on a domestic route without pre-approval",
        "policy_section": "Section 3 (Travel)",
        "example_amount": 1450.00,
    },
    {
        "type": "personal_expense",
        "description": "Personal item charged to corporate account",
        "policy_section": "Section 4 (Prohibited Expenses)",
        "example_amount": 89.99,
    },
    {
        "type": "duplicate_submission",
        "description": "Same expense submitted twice in the same report",
        "policy_section": "Section 4 (Prohibited Expenses)",
        "example_amount": 245.00,
    },
    {
        "type": "late_submission",
        "description": "Expense submitted beyond the 30-day window",
        "policy_section": "Section 1 (General Rules)",
        "example_amount": 320.00,
    },
    {
        "type": "alcohol_internal",
        "description": "Alcohol at an internal (non-client) event",
        "policy_section": "Section 2 (Meals and Entertainment)",
        "example_amount": 156.00,
    },
    {
        "type": "mileage_discrepancy",
        "description": "Claimed mileage significantly exceeds standard route distance",
        "policy_section": "Section 3 (Travel)",
        "example_amount": 187.39,
    },
    # --- New violation types for deeper seedability ---
    {
        "type": "exceeded_hotel_rate",
        "description": "Hotel rate exceeds GSA per-diem for the destination city",
        "policy_section": "Section 3 (Travel)",
        "example_amount": 289.00,
    },
    {
        "type": "split_expense",
        "description": "Single expense split across two line items to stay under receipt threshold",
        "policy_section": "Section 1 (General Rules)",
        "example_amount": 48.00,
    },
    {
        "type": "gift_to_client",
        "description": "Client gift exceeding $50 company limit without pre-approval",
        "policy_section": "Section 5 (Gifts and Gratuities)",
        "example_amount": 125.00,
    },
    {
        "type": "unauthorized_recurring",
        "description": "Recurring subscription charged without annual budget approval",
        "policy_section": "Section 6 (Subscriptions and Recurring Charges)",
        "example_amount": 49.99,
    },
]

# Standard mileage routes (miles) — included in policy appendix so model
# must cross-reference instead of being told the answer in the expense report.
MILEAGE_STANDARD_ROUTES = [
    ("Chicago", "Milwaukee", 92),
    ("Chicago", "Indianapolis", 181),
    ("Chicago", "Detroit", 282),
    ("New York", "Philadelphia", 97),
    ("New York", "Boston", 215),
    ("New York", "Washington DC", 225),
    ("Los Angeles", "San Diego", 120),
    ("Los Angeles", "Las Vegas", 270),
    ("Dallas", "Houston", 239),
    ("Dallas", "San Antonio", 274),
    ("Atlanta", "Charlotte", 244),
    ("Atlanta", "Nashville", 249),
    ("Seattle", "Portland", 174),
    ("Denver", "Colorado Springs", 71),
    ("San Francisco", "Sacramento", 88),
]

# GSA-style per-diem hotel rates by city (for hotel rate violation)
HOTEL_PER_DIEM_RATES = {
    "New York": 282, "San Francisco": 261, "Washington DC": 233,
    "Boston": 227, "Chicago": 196, "Los Angeles": 185,
    "Seattle": 201, "Denver": 168, "Atlanta": 152,
    "Dallas": 148, "Phoenix": 142, "Portland": 158,
}

# Near-miss items that look suspicious but are compliant
COMPLIANCE_NEAR_MISSES = [
    {
        "type": "meal_just_under_limit",
        "desc_template": "Business lunch — {n_attendees} attendees",
        "amount_range": (70.0, 74.99),  # Under $75 limit
        "receipt": "Yes",
        "note": "Per-person cost is under the $75 meal limit",
    },
    {
        "type": "mileage_within_tolerance",
        "desc_template": "Mileage — round trip {origin} to {destination}",
        "overage_pct": (0.03, 0.09),  # Under 10% threshold
        "receipt": "N/A",
        "note": "Claimed mileage is within the 10% tolerance",
    },
    {
        "type": "high_value_with_receipt",
        "desc_template": "Conference registration — {conf_name}",
        "amount_range": (800.0, 1200.0),
        "receipt": "Yes",
        "note": "High amount but has receipt and is a legitimate business expense",
    },
    {
        "type": "alcohol_with_client",
        "desc_template": "Client dinner — {restaurant} ({n_attendees} attendees incl. {client_name})",
        "amount_range": (120.0, 200.0),
        "receipt": "Yes",
        "note": "Includes alcohol but has client present — policy allows this",
    },
]

# =============================================================================
# DOMAIN: FINANCIAL
# =============================================================================

TAX_BRACKETS_2024: dict[str, list[tuple[float, float]]] = {
    "single": [
        (11600, 0.10),
        (47150, 0.12),
        (100525, 0.22),
        (191950, 0.24),
        (243725, 0.32),
        (609350, 0.35),
        (float("inf"), 0.37),
    ],
    "married_filing_jointly": [
        (23200, 0.10),
        (94300, 0.12),
        (201050, 0.22),
        (383900, 0.24),
        (487450, 0.32),
        (731200, 0.35),
        (float("inf"), 0.37),
    ],
    "head_of_household": [
        (16550, 0.10),
        (63100, 0.12),
        (100500, 0.22),
        (191950, 0.24),
        (243700, 0.32),
        (609350, 0.35),
        (float("inf"), 0.37),
    ],
}

STANDARD_DEDUCTION_2024 = {
    "single": 14600,
    "married_filing_jointly": 29200,
    "head_of_household": 21900,
}


# =============================================================================
# HELPERS
# =============================================================================


def _fmt_money(amount: float) -> str:
    """Format a float as a dollar string with commas."""
    return f"${amount:,.2f}"


def _compute_tax(taxable_income: float, brackets: list[tuple[float, float]]) -> float:
    """Compute total federal tax from bracket table."""
    tax = 0.0
    prev_limit = 0.0
    for limit, rate in brackets:
        if taxable_income <= prev_limit:
            break
        bracket_income = min(taxable_income, limit) - prev_limit
        tax += bracket_income * rate
        prev_limit = limit
    return round(tax, 2)


# =============================================================================
# 1. TAX COMPUTATION
# =============================================================================


def make_tax_computation(rand_seed: int = 42) -> RubricDatapoint:
    """Given W-2 data, deductions, and tax brackets, compute tax liability.

    Seed varies: income amounts, deduction amounts, filing status.
    """
    rng = _random.Random(rand_seed)

    # --- Choose filing status ---
    filing_status = rng.choice(["single", "married_filing_jointly", "head_of_household"])
    status_display = filing_status.replace("_", " ").title()
    standard_deduction = STANDARD_DEDUCTION_2024[filing_status]

    # --- Generate W-2 income ---
    employer_name = pick1(COMPANY_NAMES, rand_seed)
    employee_name = random_name(rand_seed)
    wages = round(rng.uniform(55000, 185000), 2)
    federal_withheld = round(wages * rng.uniform(0.15, 0.25), 2)
    state_withheld = round(wages * rng.uniform(0.04, 0.08), 2)
    social_security_wages = wages
    social_security_tax = round(min(wages, 168600) * 0.062, 2)
    medicare_wages = wages
    medicare_tax = round(wages * 0.0145, 2)

    # --- Supplemental income ---
    has_interest = rng.random() > 0.3
    interest_income = round(rng.uniform(200, 3500), 2) if has_interest else 0.0
    has_dividends = rng.random() > 0.5
    dividend_income = round(rng.uniform(100, 2000), 2) if has_dividends else 0.0

    total_income = round(wages + interest_income + dividend_income, 2)

    # --- Generate individual receipt items (model must aggregate) ---
    # Mortgage interest
    has_mortgage = rng.random() > 0.3
    mortgage_interest = round(rng.uniform(4000, 18000), 2) if has_mortgage else 0.0

    # State & local taxes — split into property + state income
    property_tax = round(rng.uniform(1500, 6000), 2)
    state_income_tax_paid = round(rng.uniform(800, 5000), 2)
    state_local_taxes = round(property_tax + state_income_tax_paid, 2)
    # SALT cap is $10,000
    salt_deductible = min(state_local_taxes, 10000.0)

    # Charitable — split into 2-4 donations
    n_charities = rng.randint(2, 4)
    charity_names = rng.sample([
        "United Way", "Red Cross", "Habitat for Humanity",
        "Local Food Bank", "Public Radio KQED", "Sierra Club",
        "St. Jude Children's Hospital", "Salvation Army",
    ], n_charities)
    charity_amounts_raw = [rng.uniform(100, 3000) for _ in range(n_charities)]
    charity_total_raw = sum(charity_amounts_raw)
    target_charitable = rng.uniform(500, 8000)
    charity_amounts = [round(a * target_charitable / charity_total_raw, 2) for a in charity_amounts_raw]
    charitable = round(sum(charity_amounts), 2)

    # Medical — split into 3-6 bills
    n_medical = rng.randint(3, 6)
    medical_providers = rng.sample([
        "Dr. Patel (primary care)", "City Hospital ER visit",
        "Quest Diagnostics (lab work)", "CVS Pharmacy (prescriptions)",
        "Dr. Kim (dentist)", "Metro Urgent Care",
        "Lakeside Physical Therapy", "Dr. Chen (ophthalmology)",
        "Walgreens (prescriptions)", "University Medical Group",
    ], n_medical)
    medical_amounts_raw = [rng.uniform(50, 3000) for _ in range(n_medical)]
    medical_total_raw = sum(medical_amounts_raw)
    target_medical = rng.uniform(1000, 12000)
    medical_amounts = [round(a * target_medical / medical_total_raw, 2) for a in medical_amounts_raw]
    medical_expenses = round(sum(medical_amounts), 2)

    # Medical deduction = amount exceeding 7.5% of AGI
    medical_threshold = round(total_income * 0.075, 2)
    medical_deductible = max(0.0, round(medical_expenses - medical_threshold, 2))

    # Non-deductible distractors (model must NOT include these)
    distractors: list[tuple[str, float]] = []
    if rng.random() > 0.3:
        distractors.append(("Anytime Fitness — annual membership", round(rng.uniform(400, 800), 2)))
    if rng.random() > 0.4:
        distractors.append(("Professional clothing — work wardrobe", round(rng.uniform(200, 600), 2)))
    if rng.random() > 0.5:
        distractors.append(("Daily commute parking — downtown garage", round(rng.uniform(1200, 2400), 2)))
    distractor_total = round(sum(d[1] for d in distractors), 2)

    itemized_total = round(
        mortgage_interest + salt_deductible + charitable + medical_deductible, 2
    )
    use_itemized = itemized_total > standard_deduction
    actual_deduction = itemized_total if use_itemized else float(standard_deduction)
    deduction_type = "itemized" if use_itemized else "standard"

    # --- Compute tax ---
    agi = total_income  # Simplified: no above-the-line deductions
    taxable_income = max(0.0, round(agi - actual_deduction, 2))
    brackets = TAX_BRACKETS_2024[filing_status]
    tax_owed = _compute_tax(taxable_income, brackets)
    effective_rate = round((tax_owed / agi) * 100, 2) if agi > 0 else 0.0

    # --- Build W-2 file ---
    w2_lines = [
        f"Form W-2 Wage and Tax Statement — Tax Year 2024",
        f"",
        f"Employer: {employer_name}",
        f"Employer EIN: {rng.randint(10, 99)}-{rng.randint(1000000, 9999999)}",
        f"",
        f"Employee: {employee_name}",
        f"SSN: XXX-XX-{rng.randint(1000, 9999)}",
        f"",
        f"Box 1  — Wages, tips, other compensation:  {_fmt_money(wages)}",
        f"Box 2  — Federal income tax withheld:       {_fmt_money(federal_withheld)}",
        f"Box 3  — Social security wages:             {_fmt_money(social_security_wages)}",
        f"Box 4  — Social security tax withheld:      {_fmt_money(social_security_tax)}",
        f"Box 5  — Medicare wages and tips:            {_fmt_money(medicare_wages)}",
        f"Box 6  — Medicare tax withheld:              {_fmt_money(medicare_tax)}",
        f"Box 17 — State income tax withheld:          {_fmt_money(state_withheld)}",
    ]
    if has_interest:
        w2_lines.append(f"")
        w2_lines.append(f"1099-INT Interest Income:                   {_fmt_money(interest_income)}")
    if has_dividends:
        w2_lines.append(f"1099-DIV Ordinary Dividends:                {_fmt_money(dividend_income)}")

    w2_content = "\n".join(w2_lines) + "\n"

    # --- Build receipts file (flat date-sorted list — model must categorize) ---
    # Collect all receipt items into a single list with dates for shuffling
    all_receipts: list[tuple[str, str, float]] = []  # (date, description, amount)

    if has_mortgage:
        lender = rng.choice(["Wells Fargo Home Mortgage", "Quicken Loans",
                              "Chase Home Lending", "Bank of America Mortgage"])
        all_receipts.append((
            f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
            f"{lender} — Form 1098 mortgage interest",
            mortgage_interest,
        ))

    # State & local taxes
    all_receipts.append((
        f"2024-{rng.randint(3,4):02d}-{rng.randint(1,28):02d}",
        "County assessor — property tax (annual)",
        property_tax,
    ))
    all_receipts.append((
        f"2024-{rng.randint(4,6):02d}-{rng.randint(1,28):02d}",
        "State tax authority — income tax payment",
        state_income_tax_paid,
    ))

    # Charitable donations
    for cname, camt in zip(charity_names, charity_amounts):
        all_receipts.append((
            f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
            f"Donation to {cname}",
            camt,
        ))

    # Medical expenses
    for mprov, mamt in zip(medical_providers, medical_amounts):
        all_receipts.append((
            f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
            mprov,
            mamt,
        ))

    # Non-deductible distractors — mixed into the same list
    for desc, amt in distractors:
        all_receipts.append((
            f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
            desc,
            amt,
        ))

    # Sort by date so items from all categories are interleaved
    all_receipts.sort(key=lambda r: r[0])

    receipt_lines = [
        f"EXPENSE RECEIPTS AND RECORDS — Tax Year 2024",
        f"Taxpayer: {employee_name}",
        f"",
        f"{'='*60}",
        f"Complete list of expenses for the year. Consult tax rules to",
        f"determine which items are deductible and under which category.",
        f"{'='*60}",
        f"",
        f"{'#':<4} {'Date':<12} {'Description':<45} {'Amount':>10}",
        f"{'-'*4} {'-'*12} {'-'*45} {'-'*10}",
    ]
    for idx, (rdate, rdesc, ramt) in enumerate(all_receipts, 1):
        receipt_lines.append(f"{idx:<4} {rdate:<12} {rdesc:<45} {_fmt_money(ramt):>10}")

    receipt_lines.append(f"")
    receipt_lines.append(f"Total receipts listed: {len(all_receipts)}")
    receipt_lines.append(f"")

    receipts_content = "\n".join(receipt_lines) + "\n"

    # --- Build tax rules reference (replaces pre-computed summary) ---
    bracket_lines = [
        f"FEDERAL TAX REFERENCE — 2024",
        f"",
        f"{'='*60}",
        f"SECTION 1: TAX BRACKETS ({status_display})",
        f"{'='*60}",
        f"",
        f"Taxable Income Range          Marginal Rate",
        f"----------------------------- -------------",
    ]
    prev = 0
    for limit, rate in brackets:
        if limit == float("inf"):
            bracket_lines.append(f"Over ${prev:,}                  {rate*100:.0f}%")
        else:
            bracket_lines.append(f"${prev:,} — ${limit:,}         {rate*100:.0f}%")
        prev = int(limit) if limit != float("inf") else prev
    bracket_lines.append("")
    bracket_lines.append("Apply each marginal rate only to the portion of income")
    bracket_lines.append("within that bracket. Sum across all brackets for total tax.")
    bracket_lines.append("")
    bracket_lines.append(f"{'='*60}")
    bracket_lines.append(f"SECTION 2: STANDARD DEDUCTIONS")
    bracket_lines.append(f"{'='*60}")
    bracket_lines.append(f"")
    bracket_lines.append(f"Single:                    $14,600")
    bracket_lines.append(f"Married Filing Jointly:    $29,200")
    bracket_lines.append(f"Head of Household:         $21,900")
    bracket_lines.append(f"")
    bracket_lines.append(f"Taxpayers choose the LARGER of their standard deduction")
    bracket_lines.append(f"or total itemized deductions.")
    bracket_lines.append(f"")
    bracket_lines.append(f"{'='*60}")
    bracket_lines.append(f"SECTION 3: ITEMIZED DEDUCTION RULES")
    bracket_lines.append(f"{'='*60}")
    bracket_lines.append(f"")
    bracket_lines.append(f"Mortgage Interest: Deductible on primary residence loans")
    bracket_lines.append(f"  up to $750,000 of acquisition debt.")
    bracket_lines.append(f"")
    bracket_lines.append(f"State and Local Taxes (SALT): The total deduction for state")
    bracket_lines.append(f"  and local taxes (income tax + property tax) is CAPPED at")
    bracket_lines.append(f"  $10,000 ($5,000 if married filing separately).")
    bracket_lines.append(f"")
    bracket_lines.append(f"Charitable Contributions: Cash donations to qualified")
    bracket_lines.append(f"  501(c)(3) organizations are deductible up to 60% of AGI.")
    bracket_lines.append(f"")
    bracket_lines.append(f"Medical Expenses: Only the amount EXCEEDING 7.5% of AGI")
    bracket_lines.append(f"  is deductible. Non-prescription items, gym memberships,")
    bracket_lines.append(f"  and cosmetic procedures are NOT deductible.")
    bracket_lines.append(f"")
    bracket_lines.append(f"NOT Deductible: Personal clothing, commuting expenses,")
    bracket_lines.append(f"  gym/fitness memberships, groceries, personal travel.")
    bracket_lines.append(f"")
    rules_content = "\n".join(bracket_lines) + "\n"

    # --- Problem statement ---
    problem_statement = f"""# Tax Liability Computation

You are a tax preparer. Using the provided W-2 data, expense receipts, and
federal tax reference, compute the federal tax liability for {employee_name}
for tax year 2024.

## Source Files
- /testbed/data/w2.txt — W-2 wage and income data
- /testbed/data/receipts.txt — All expense receipts for the year (some may not be deductible)
- /testbed/data/tax_reference.txt — Federal tax brackets, standard deductions, and itemized deduction rules

## Requirements
1. Calculate Adjusted Gross Income (AGI) from all income sources
2. Categorize each receipt as deductible or non-deductible per the tax rules
3. Apply relevant caps and thresholds (SALT cap, medical expense threshold)
4. Calculate total itemized deductions and compare to the standard deduction
5. Calculate taxable income
6. Apply the progressive tax brackets to compute tax owed
7. Compute the effective tax rate (tax owed / AGI)

Write a clear tax summary to /testbed/tax_summary.txt that shows your
work at each step, including which expenses are deductible and why."""

    # --- Build expanded rubric ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="correct_agi",
            question=f"Does the tax summary state the AGI as {_fmt_money(agi)} (or equivalently ${agi:,.0f}, allowing minor rounding)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_total_medical",
            question=f"Does the summary correctly total medical expenses as approximately {_fmt_money(medical_expenses)} (within $5)?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_medical_threshold",
            question=f"Does the summary compute the 7.5% AGI medical threshold as approximately {_fmt_money(medical_threshold)} (within $50)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_medical_deductible",
            question=f"Does the summary compute the deductible medical amount as approximately {_fmt_money(medical_deductible)} (within $50)? If medical expenses are below the threshold, this should be $0.00.",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_salt_total",
            question=f"Does the summary correctly total state and local taxes (property + state income) as approximately {_fmt_money(state_local_taxes)} (within $5)?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_salt_cap_applied",
            question=f"Does the summary correctly apply the $10,000 SALT cap, resulting in a deductible SALT amount of {_fmt_money(salt_deductible)}?",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_charitable_total",
            question=f"Does the summary correctly total charitable contributions as approximately {_fmt_money(charitable)} (within $5)?",
            points=1,
        ),
    ]

    if has_mortgage:
        rubric_items.append(
            BinaryRubricCategory(
                name="correct_mortgage_interest",
                question=f"Does the summary include mortgage interest of {_fmt_money(mortgage_interest)} as a deductible item?",
                points=1,
            ),
        )

    rubric_items.extend([
        BinaryRubricCategory(
            name="correct_itemized_total",
            question=f"Does the summary compute total itemized deductions as approximately {_fmt_money(itemized_total)} (within $100)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_deduction_choice",
            question=f"Does the summary correctly identify that the {deduction_type} deduction ({_fmt_money(actual_deduction)}) should be used because it is larger?",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_standard_deduction_amount",
            question=f"Does the summary correctly state the standard deduction for {status_display} as {_fmt_money(float(standard_deduction))}?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_taxable_income",
            question=f"Does the summary compute taxable income as approximately {_fmt_money(taxable_income)} (within $100 of the correct value)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_tax_owed",
            question=f"Does the summary compute total federal tax owed as approximately {_fmt_money(tax_owed)} (within $100 of the correct value)?",
            points=3,
        ),
        BinaryRubricCategory(
            name="correct_effective_rate",
            question=f"Does the summary state an effective tax rate of approximately {effective_rate:.1f}% (within 0.5 percentage points)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="mentions_filing_status",
            question=f'Does the summary explicitly mention the filing status "{status_display}"?',
            points=1,
        ),
        BinaryRubricCategory(
            name="shows_bracket_breakdown",
            question="Does the summary show how tax was computed across at least 2 different bracket rates (i.e., a bracket-by-bracket breakdown)?",
            points=2,
        ),
    ])

    # False-positive checks — model must NOT include non-deductible items
    if distractors:
        rubric_items.append(
            BinaryRubricCategory(
                name="no_non_deductible_included",
                question=(
                    "Does the summary correctly EXCLUDE non-deductible personal expenses "
                    f"(such as {', '.join(d[0].split(' — ')[0] for d in distractors)}) "
                    "from the itemized deductions total? These items are NOT tax-deductible."
                ),
                points=2,
            ),
        )

    rubric_items.append(
        RubricCategory(
            name="presentation_clarity",
            description="Is the summary well-organized, clearly labeled, and easy to follow?",
            failure="Disorganized output or raw numbers without context.",
            minor_failure="Some structure but hard to follow; missing labels.",
            minor_success="Reasonably organized with clear labels for most values.",
            success="Professional presentation with clear section headers, labeled values, and logical flow.",
            points=2,
        ),
    )

    rubric = tuple(rubric_items)

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write your completed tax summary to /testbed/tax_summary.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/w2.txt": w2_content,
            "/testbed/data/receipts.txt": receipts_content,
            "/testbed/data/tax_reference.txt": rules_content,
        },
        problem_type="tax_computation",
    )


# =============================================================================
# 2. FINANCIAL RECONCILIATION
# =============================================================================


def make_financial_reconciliation(rand_seed: int = 42) -> RubricDatapoint:
    """Given bank statement and internal ledger with deliberate mismatches,
    identify discrepancies and write a reconciliation report.

    Seed varies: transaction amounts, descriptions, and which entries mismatch.
    """
    rng = _random.Random(rand_seed)

    company = pick1(COMPANY_NAMES, rand_seed)
    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    month_idx = rng.randint(0, 11)
    month_name = month_names[month_idx]
    year = 2024

    # --- Generate base transactions ---
    n_transactions = rng.randint(28, 35)
    categories = [
        "Office Supplies", "Software License", "Consulting Fee", "Travel Expense",
        "Utilities", "Insurance Premium", "Marketing", "Payroll",
        "Equipment Lease", "Maintenance", "Telecommunications", "Legal Services",
    ]
    vendors = rng.sample(VENDOR_NAMES, min(10, len(VENDOR_NAMES)))

    transactions = []
    for i in range(n_transactions):
        day = rng.randint(1, 28)
        date = f"{year}-{month_idx + 1:02d}-{day:02d}"
        vendor = rng.choice(vendors)
        category = rng.choice(categories)
        amount = round(rng.uniform(45, 8500), 2)
        is_credit = rng.random() < 0.15  # 15% chance of being a credit/refund
        if is_credit:
            amount = -round(rng.uniform(25, 500), 2)
        ref = f"TXN-{year}{month_idx+1:02d}{i+1:04d}"
        transactions.append({
            "date": date,
            "ref": ref,
            "vendor": vendor,
            "category": category,
            "amount": amount,
        })

    transactions.sort(key=lambda t: t["date"])

    # --- Introduce deliberate mismatches ---
    # Pick indices for mismatches (avoiding the same transaction twice)
    n_mismatches = rng.randint(3, 5)
    mismatch_indices = rng.sample(range(n_transactions), min(n_mismatches, n_transactions))

    discrepancies = []
    bank_transactions = [dict(t) for t in transactions]
    ledger_transactions = [dict(t) for t in transactions]

    # Mismatch type 1: Wrong amount (bank and ledger differ)
    if len(mismatch_indices) > 0:
        idx = mismatch_indices[0]
        original_amount = bank_transactions[idx]["amount"]
        wrong_amount = round(original_amount + rng.choice([-1, 1]) * rng.uniform(10, 200), 2)
        ledger_transactions[idx]["amount"] = wrong_amount
        discrepancies.append({
            "type": "amount_mismatch",
            "ref": transactions[idx]["ref"],
            "date": transactions[idx]["date"],
            "vendor": transactions[idx]["vendor"],
            "bank_amount": original_amount,
            "ledger_amount": wrong_amount,
            "description": f"Amount mismatch on {transactions[idx]['ref']}: bank shows {_fmt_money(original_amount)}, ledger shows {_fmt_money(wrong_amount)}",
        })

    # Mismatch type 2: Missing from ledger (bank has it, ledger doesn't)
    if len(mismatch_indices) > 1:
        idx = mismatch_indices[1]
        discrepancies.append({
            "type": "missing_from_ledger",
            "ref": transactions[idx]["ref"],
            "date": transactions[idx]["date"],
            "vendor": transactions[idx]["vendor"],
            "bank_amount": bank_transactions[idx]["amount"],
            "ledger_amount": None,
            "description": f"Transaction {transactions[idx]['ref']} ({transactions[idx]['vendor']}, {_fmt_money(bank_transactions[idx]['amount'])}) appears in bank statement but not in ledger",
        })
        ledger_transactions[idx] = None  # type: ignore[assignment]

    # Mismatch type 3: Missing from bank (ledger has it, bank doesn't)
    if len(mismatch_indices) > 2:
        idx = mismatch_indices[2]
        discrepancies.append({
            "type": "missing_from_bank",
            "ref": transactions[idx]["ref"],
            "date": transactions[idx]["date"],
            "vendor": transactions[idx]["vendor"],
            "bank_amount": None,
            "ledger_amount": ledger_transactions[idx]["amount"],
            "description": f"Transaction {transactions[idx]['ref']} ({transactions[idx]['vendor']}, {_fmt_money(ledger_transactions[idx]['amount'])}) appears in ledger but not in bank statement",
        })
        bank_transactions[idx] = None  # type: ignore[assignment]

    # Mismatch type 4: Duplicate in ledger
    if len(mismatch_indices) > 3:
        idx = mismatch_indices[3]
        dup_entry = dict(ledger_transactions[idx])
        dup_entry["ref"] = f"TXN-{rng.randint(9000, 9999)}"
        ledger_transactions.append(dup_entry)
        discrepancies.append({
            "type": "duplicate_in_ledger",
            "ref": transactions[idx]["ref"],
            "date": transactions[idx]["date"],
            "vendor": transactions[idx]["vendor"],
            "bank_amount": bank_transactions[idx]["amount"],
            "ledger_amount": ledger_transactions[idx]["amount"],
            "description": f"Transaction {transactions[idx]['ref']} ({transactions[idx]['vendor']}) appears twice in ledger (duplicate entry)",
        })

    # Mismatch type 5: Wrong date in ledger
    if len(mismatch_indices) > 4:
        idx = mismatch_indices[4]
        orig_date = ledger_transactions[idx]["date"]
        wrong_day = min(28, int(orig_date.split("-")[2]) + rng.randint(1, 5))
        wrong_date = f"{orig_date[:8]}{wrong_day:02d}"
        ledger_transactions[idx]["date"] = wrong_date
        discrepancies.append({
            "type": "date_mismatch",
            "ref": transactions[idx]["ref"],
            "date": transactions[idx]["date"],
            "vendor": transactions[idx]["vendor"],
            "bank_amount": bank_transactions[idx]["amount"],
            "ledger_amount": ledger_transactions[idx]["amount"],
            "description": f"Date mismatch on {transactions[idx]['ref']}: bank shows {transactions[idx]['date']}, ledger shows {wrong_date}",
        })

    # --- Build CSV files ---
    # Filter out None entries
    bank_rows = [t for t in bank_transactions if t is not None]
    ledger_rows = [t for t in ledger_transactions if t is not None]

    bank_total = round(sum(t["amount"] for t in bank_rows), 2)
    ledger_total = round(sum(t["amount"] for t in ledger_rows), 2)
    net_difference = round(bank_total - ledger_total, 2)

    bank_csv_lines = ["Date,Reference,Vendor,Category,Amount"]
    for t in bank_rows:
        bank_csv_lines.append(f"{t['date']},{t['ref']},{t['vendor']},{t['category']},{t['amount']:.2f}")
    bank_csv = "\n".join(bank_csv_lines) + "\n"

    ledger_csv_lines = ["Date,Reference,Vendor,Category,Amount"]
    for t in sorted(ledger_rows, key=lambda x: x["date"]):
        ledger_csv_lines.append(f"{t['date']},{t['ref']},{t['vendor']},{t['category']},{t['amount']:.2f}")
    ledger_csv = "\n".join(ledger_csv_lines) + "\n"

    # --- Problem statement ---
    problem_statement = f"""# Financial Reconciliation

You are an accountant at {company}. You have been given the bank statement
and internal ledger for {month_name} {year}. Your task is to reconcile the
two records, identify all discrepancies, and write a formal reconciliation report.

## Source Files
- /testbed/data/bank_statement.csv — Bank statement (CSV)
- /testbed/data/internal_ledger.csv — Internal accounting ledger (CSV)

## Requirements
1. Compare each transaction by reference number, date, vendor, and amount
2. Identify ALL discrepancies (missing entries, amount differences, duplicates, date mismatches)
3. Calculate the bank statement total and the ledger total
4. State the net difference between the two totals
5. For each discrepancy, explain what is wrong and recommend a resolution

Write your reconciliation report to /testbed/reconciliation.txt"""

    # Build rubric — one binary check per discrepancy, plus totals and graded categories
    rubric_items: list[BinaryRubricCategory | RubricCategory] = []

    for i, disc in enumerate(discrepancies):
        rubric_items.append(
            BinaryRubricCategory(
                name=f"found_discrepancy_{i+1}",
                question=f"Does the report identify the following discrepancy: {disc['description']}?",
                points=2,
            )
        )

    rubric_items.extend([
        BinaryRubricCategory(
            name="correct_bank_total",
            question=f"Does the report state the bank statement total as approximately {_fmt_money(bank_total)} (within $1)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_ledger_total",
            question=f"Does the report state the internal ledger total as approximately {_fmt_money(ledger_total)} (within $1)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="states_net_difference",
            question=f"Does the report state the net difference between bank and ledger as approximately {_fmt_money(abs(net_difference))} (within $5)?",
            points=2,
        ),
        RubricCategory(
            name="completeness",
            description="How completely does the report identify and explain all discrepancies?",
            failure="Fewer than half of discrepancies found, or report is mostly empty.",
            minor_failure="Some discrepancies found but explanations are vague or incomplete.",
            minor_success="Most discrepancies found with reasonable explanations.",
            success="All discrepancies found with clear explanations and recommended resolutions.",
            points=3,
        ),
        RubricCategory(
            name="professional_quality",
            description="Is the reconciliation report professional, well-structured, and suitable for management review?",
            failure="Informal notes or disorganized output.",
            minor_failure="Some structure but reads like rough notes rather than a formal report.",
            minor_success="Professional tone with clear sections; minor formatting issues.",
            success="Formal reconciliation report with executive summary, detailed findings, and clear recommendations.",
            points=3,
        ),
    ])

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=tuple(rubric_items),
        submission_instructions="Write your completed reconciliation report to /testbed/reconciliation.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/bank_statement.csv": bank_csv,
            "/testbed/data/internal_ledger.csv": ledger_csv,
        },
        problem_type="financial_reconciliation",
    )


# =============================================================================
# 3. CONTRACT CLAUSE REVIEW
# =============================================================================


def make_contract_clause_review(rand_seed: int = 42) -> RubricDatapoint:
    """Given a contract draft with problematic clauses and review guidelines,
    identify issues and write a review memo.

    Seed varies: which clauses are problematic, party names.
    """
    rng = _random.Random(rand_seed)

    client_company = pick1(COMPANY_NAMES, rand_seed)
    vendor_company = pick1(VENDOR_NAMES, rand_seed + 1)
    reviewer_name = random_name(rand_seed + 2)

    # Pick 4-6 problematic clauses
    n_problematic = rng.randint(4, 6)
    chosen_problems = rng.sample(PROBLEMATIC_CLAUSES, n_problematic)

    # Pick 3-4 safe clauses to intersperse
    n_safe = rng.randint(3, 4)
    chosen_safe = rng.sample(SAFE_CLAUSES, n_safe)

    # --- Build contract draft ---
    contract_lines = [
        f"PROFESSIONAL SERVICES AGREEMENT",
        f"",
        f"This Agreement ('Agreement') is entered into as of January 15, 2025,",
        f"by and between {client_company} ('Client') and {vendor_company} ('Vendor').",
        f"",
        f"RECITALS",
        f"",
        f"WHEREAS, the Client desires to engage the Vendor to provide certain",
        f"professional services as described herein; and",
        f"",
        f"WHEREAS, the Vendor represents that it has the expertise and capacity",
        f"to perform such services;",
        f"",
        f"NOW, THEREFORE, in consideration of the mutual covenants contained herein,",
        f"the parties agree as follows:",
        f"",
    ]

    # Interleave problematic and safe clauses
    all_clauses = []
    for pc in chosen_problems:
        all_clauses.append(("problematic", pc))
    for sc in chosen_safe:
        all_clauses.append(("safe", {"clause": sc}))
    rng.shuffle(all_clauses)

    for idx, (ctype, clause_data) in enumerate(all_clauses):
        section_num = idx + 1
        contract_lines.append(f"Section {section_num}.")
        contract_lines.append(f"")
        contract_lines.append(clause_data["clause"])
        contract_lines.append(f"")

    contract_lines.extend([
        f"IN WITNESS WHEREOF, the parties have executed this Agreement as of the",
        f"date first written above.",
        f"",
        f"CLIENT: {client_company}",
        f"By: ____________________________",
        f"Name: ____________________________",
        f"Title: ____________________________",
        f"",
        f"VENDOR: {vendor_company}",
        f"By: ____________________________",
        f"Name: ____________________________",
        f"Title: ____________________________",
    ])
    contract_content = "\n".join(contract_lines) + "\n"

    # --- Build review guidelines ---
    guidelines_lines = [
        f"CONTRACT REVIEW GUIDELINES",
        f"Prepared by Legal Department — {client_company}",
        f"",
        f"When reviewing vendor contracts, flag the following issues:",
        f"",
        f"1. LIABILITY: Watch for unlimited or uncapped liability clauses.",
        f"   Vendor liability should be capped (typically at contract value).",
        f"",
        f"2. TERMINATION: Ensure termination clauses are balanced. Both parties",
        f"   should have termination rights with reasonable notice. Work in",
        f"   progress should be compensated upon early termination.",
        f"",
        f"3. IP RIGHTS: IP assignment should be limited to work product created",
        f"   specifically for this engagement. Pre-existing IP and unrelated",
        f"   work should be excluded.",
        f"",
        f"4. NON-COMPETE: Non-compete clauses should be reasonable in scope",
        f"   (typically 6-12 months, limited geography, specific industry).",
        f"   Clauses exceeding 12 months or covering all markets are problematic.",
        f"",
        f"5. SCOPE CHANGES: Any change in scope should require mutual agreement",
        f"   and proportional compensation adjustment.",
        f"",
        f"6. INDEMNIFICATION: Should be mutual. One-sided indemnification",
        f"   creates unbalanced risk exposure.",
        f"",
        f"7. AUTO-RENEWAL: Notice periods for non-renewal should be reasonable",
        f"   (30-90 days). Periods exceeding 90 days are flagged as excessive.",
        f"",
        f"8. ACCEPTANCE: Deliverable acceptance criteria must be specific and",
        f"   objective. Vague terms like 'reasonable' or 'expectations' are",
        f"   insufficient.",
        f"",
        f"SEVERITY RATINGS:",
        f"  HIGH — Clause creates significant legal or financial risk",
        f"  MEDIUM — Clause is unfavorable but negotiable",
        f"  LOW — Minor issue, note for awareness",
        f"",
        f"For each issue found, provide:",
        f"  - Section number and clause text",
        f"  - Issue description",
        f"  - Severity rating (HIGH / MEDIUM / LOW)",
        f"  - Recommended revision",
    ]
    guidelines_content = "\n".join(guidelines_lines) + "\n"

    # --- Problem statement ---
    problem_statement = f"""# Contract Clause Review

You are {reviewer_name}, a contract reviewer in the legal department at
{client_company}. A new professional services agreement with {vendor_company}
has been drafted and needs your review before signing.

## Source Files
- /testbed/docs/contract_draft.txt — The proposed contract
- /testbed/docs/review_guidelines.txt — Internal review guidelines

## Requirements
1. Read the contract carefully, section by section
2. Identify ALL problematic clauses per the review guidelines
3. For each issue, state the section number, describe the problem, assign a severity, and recommend a revision
4. Do NOT flag clauses that are standard and reasonable

Write your review memo to /testbed/review_memo.txt"""

    # --- Rubric ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = []

    for pc in chosen_problems:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"found_{pc['name']}",
                question=f"Does the review memo identify the '{pc['name'].replace('_', ' ')}' issue? Specifically: {pc['issue']}",
                points=2,
            )
        )

    rubric_items.append(
        BinaryRubricCategory(
            name="no_false_positives",
            question="Does the review memo avoid flagging standard/safe clauses as problematic? (A few minor notes are OK, but the safe clauses should not be called out as serious issues.)",
            points=2,
        )
    )

    rubric_items.append(
        BinaryRubricCategory(
            name="severity_assigned",
            question="Does the review memo assign a severity rating (HIGH, MEDIUM, or LOW) to each identified issue?",
            points=1,
        )
    )

    rubric_items.append(
        BinaryRubricCategory(
            name="section_numbers_cited",
            question="Does the review memo cite the specific section numbers from the contract for each problematic clause?",
            points=1,
        )
    )

    rubric_items.append(
        BinaryRubricCategory(
            name="memo_format",
            question="Is the review written as a memo (addressed to someone, with date, subject, and structured findings) rather than as informal notes?",
            points=1,
        )
    )

    rubric_items.extend([
        RubricCategory(
            name="severity_accuracy",
            description="Are the severity ratings appropriate for each identified issue?",
            failure="No severity ratings provided, or ratings are entirely wrong.",
            minor_failure="Severity ratings present but several are clearly misaligned.",
            minor_success="Most severity ratings are reasonable; one or two are debatable.",
            success="All severity ratings are appropriate and well-justified.",
            points=3,
        ),
        RubricCategory(
            name="recommendation_quality",
            description="Are the recommended revisions practical and legally sound?",
            failure="No recommendations provided.",
            minor_failure="Recommendations are vague (e.g., 'fix this clause').",
            minor_success="Recommendations are specific but some lack practical detail.",
            success="Each issue has a specific, practical, and legally sound recommended revision.",
            points=3,
        ),
        RubricCategory(
            name="legal_analysis_depth",
            description="Does the memo demonstrate understanding of why each clause is problematic from a legal/business perspective?",
            failure="No analysis — just states 'this is bad' without reasoning.",
            minor_failure="Some reasoning but superficial; does not explain legal or business risk.",
            minor_success="Good reasoning for most issues; explains risk in business terms.",
            success="Each issue has clear legal/business reasoning explaining the risk and its potential consequences.",
            points=3,
        ),
    ])

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=tuple(rubric_items),
        submission_instructions="Write your completed review memo to /testbed/review_memo.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/docs/contract_draft.txt": contract_content,
            "/testbed/docs/review_guidelines.txt": guidelines_content,
        },
        problem_type="contract_review",
    )


# =============================================================================
# 4. HR INVESTIGATION SUMMARY
# =============================================================================


def make_hr_investigation_summary(rand_seed: int = 42) -> RubricDatapoint:
    """Given witness interviews (with contradictions), a policy doc, and timeline,
    write an investigation summary.

    Seed varies: incident type, witness names, contradictions.
    """
    rng = _random.Random(rand_seed)

    company = pick1(COMPANY_NAMES, rand_seed)
    witness_names = random_names(rand_seed, 3)
    w1, w2, w3 = witness_names
    investigator = random_name(rand_seed + 10)
    reported_by = random_name(rand_seed + 20)

    # --- Incident types ---
    incident_types = [
        {
            "type": "workplace_harassment",
            "title": "Alleged Workplace Harassment",
            "policy_section": "Section 4.2 — Anti-Harassment Policy",
            "policy_text": (
                "All employees have the right to work in an environment free from harassment, "
                "including verbal abuse, intimidation, unwelcome comments, and hostile behavior. "
                "Harassment includes but is not limited to: repeated offensive remarks, threats, "
                "public humiliation, and deliberate exclusion from work activities. "
                "Managers must report all complaints within 24 hours to HR. "
                "Retaliation against complainants is strictly prohibited."
            ),
        },
        {
            "type": "data_breach",
            "title": "Alleged Unauthorized Data Access",
            "policy_section": "Section 7.1 — Information Security Policy",
            "policy_text": (
                "Access to confidential data is restricted to authorized personnel. "
                "Employees must not share login credentials, access systems beyond their "
                "authorized scope, or transfer confidential data to external devices. "
                "All data access is logged. Violations may result in termination and legal action. "
                "Incidents must be reported to the IT Security team within 4 hours of discovery."
            ),
        },
        {
            "type": "expense_fraud",
            "title": "Alleged Expense Report Fraud",
            "policy_section": "Section 5.3 — Expense Reimbursement Policy",
            "policy_text": (
                "Employees must submit accurate expense reports with original receipts. "
                "Fabricating receipts, inflating amounts, or submitting personal expenses "
                "as business costs constitutes fraud. Managers must review and approve all "
                "reports within 5 business days. Violations may result in repayment, "
                "disciplinary action up to termination, and referral to legal authorities."
            ),
        },
        {
            "type": "safety_violation",
            "title": "Alleged Workplace Safety Violation",
            "policy_section": "Section 3.4 — Workplace Safety Policy",
            "policy_text": (
                "All employees must follow established safety protocols. Required PPE must be "
                "worn in designated areas at all times. Safety incidents must be reported "
                "immediately to the supervisor and EHS team. Supervisors who fail to enforce "
                "safety protocols are subject to disciplinary action. Near-miss events must "
                "also be documented within 24 hours."
            ),
        },
    ]

    incident = rng.choice(incident_types)
    incident_date = f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
    location = rng.choice([
        "Building A, Conference Room 3",
        "Main Office, Floor 2",
        "Warehouse Loading Dock",
        "Remote work — Zoom call",
        "Cafeteria, Building B",
    ])

    # --- Generate contradictions ---
    # Contradiction 1: W1 and W2 disagree on timing
    time_w1 = f"{rng.randint(8, 11)}:{rng.choice(['00', '15', '30', '45'])} AM"
    time_w2 = f"{rng.randint(1, 4)}:{rng.choice(['00', '15', '30', '45'])} PM"
    contradiction_1 = (
        f"{w1} states the incident occurred at approximately {time_w1}, "
        f"while {w2} states it occurred at approximately {time_w2}"
    )

    # Contradiction 2: W1 and W3 disagree on who was present
    extra_person = random_name(rand_seed + 30)
    contradiction_2 = (
        f"{w1} states that {extra_person} was present during the incident, "
        f"while {w3} explicitly states that {extra_person} was not in the room"
    )

    # Contradiction 3: W2 and W3 disagree on severity/response
    # Store both the full claim (for rubric) and a short narrative (for witness text)
    if incident["type"] == "workplace_harassment":
        c3_w2_claim = f"{w2} describes the behavior as 'aggressive and threatening'"
        c3_w3_claim = f"{w3} describes it as 'a heated but normal disagreement'"
        c3_w2_narrative = "the behavior was aggressive and threatening"
        c3_w3_narrative = "it was a heated but normal disagreement"
    elif incident["type"] == "data_breach":
        c3_w2_claim = f"{w2} says the data was copied to a personal USB drive"
        c3_w3_claim = f"{w3} says the data was only viewed on screen and not copied"
        c3_w2_narrative = "the data was copied to a personal USB drive"
        c3_w3_narrative = "the data was only viewed on screen and never actually copied"
    elif incident["type"] == "expense_fraud":
        c3_w2_claim = f"{w2} says the receipts were clearly fabricated"
        c3_w3_claim = f"{w3} says the receipts looked legitimate but had wrong dates"
        c3_w2_narrative = "the receipts were clearly fabricated"
        c3_w3_narrative = "the receipts looked legitimate but had wrong dates"
    else:
        c3_w2_claim = f"{w2} says the safety equipment was completely absent"
        c3_w3_claim = f"{w3} says the equipment was present but not properly worn"
        c3_w2_narrative = "the safety equipment was completely absent"
        c3_w3_narrative = "the equipment was present but not properly worn"
    contradiction_3 = f"{c3_w2_claim}, but {c3_w3_claim}"

    # --- Build witness interviews ---
    def _build_interview(name: str, seed_offset: int, perspective: str) -> str:
        local_rng = _random.Random(rand_seed + seed_offset)
        lines = [
            f"WITNESS INTERVIEW TRANSCRIPT",
            f"",
            f"Interviewee: {name}",
            f"Date of Interview: {incident_date}",
            f"Interviewer: {investigator}, HR Department",
            f"Location: {company} — HR Conference Room",
            f"",
            f"---",
            f"",
            f"Q: Thank you for coming in, {name.split()[0]}. Can you tell me what happened?",
            f"",
            f"A: {perspective}",
            f"",
        ]
        return "\n".join(lines) + "\n"

    # W1 perspective — generally factual, mentions time as time_w1, says extra_person present
    w1_story = (
        f"I was in {location} on {incident_date}. It was around {time_w1} when "
        f"the incident began. {extra_person} was also there. "
        f"I saw the situation unfold and it was concerning. I reported it to my "
        f"manager the same day. The atmosphere was tense, and I felt it was serious "
        f"enough to warrant a formal complaint. I have been working at {company} "
        f"for {rng.randint(2, 8)} years and have never seen anything like this."
    )

    # W2 perspective — gives different time, escalates severity
    w2_story = (
        f"I recall the incident clearly. It happened at about {time_w2} — I remember "
        f"because I had just come back from lunch. We were in {location}. "
        f"In my opinion, {c3_w2_narrative}. "
        f"I was shocked. I told {reported_by} about it afterward. "
        f"Several people noticed but I think some are afraid to speak up."
    )

    # W3 perspective — contradicts on presence and severity
    w3_story = (
        f"I was near {location} that day but {extra_person} was not in the room — I am "
        f"certain of that. The incident happened and I observed part of it. "
        f"In my view, {c3_w3_narrative}. "
        f"I have worked with the people involved and this was unusual but I would not "
        f"call it extreme. I want to be fair to everyone involved."
    )

    interview_w1 = _build_interview(w1, 100, w1_story)
    interview_w2 = _build_interview(w2, 200, w2_story)
    interview_w3 = _build_interview(w3, 300, w3_story)

    # --- Build policy doc ---
    policy_lines = [
        f"{company} — Employee Handbook (Excerpt)",
        f"",
        f"{'='*60}",
        f"{incident['policy_section']}",
        f"{'='*60}",
        f"",
        f"{incident['policy_text']}",
        f"",
        f"{'='*60}",
        f"Section 9.1 — Investigation Procedures",
        f"{'='*60}",
        f"",
        f"When an incident is reported, HR will:",
        f"1. Acknowledge receipt of the complaint within 24 hours",
        f"2. Interview all relevant witnesses within 5 business days",
        f"3. Review any available documentary evidence",
        f"4. Prepare a written investigation summary",
        f"5. Present findings and recommendations to the Ethics Committee",
        f"",
        f"The investigation summary must include:",
        f"- Incident description and timeline",
        f"- Summary of all witness statements",
        f"- Identification of contradictions or inconsistencies",
        f"- Relevant policy citations",
        f"- Factual findings (what can be established vs. what is disputed)",
        f"- Recommendations for next steps",
    ]
    policy_content = "\n".join(policy_lines) + "\n"

    # --- Build timeline ---
    timeline_lines = [
        f"INCIDENT TIMELINE",
        f"Case: {incident['title']}",
        f"Case Number: HR-{rng.randint(2024000, 2024999)}",
        f"",
        f"Date of Incident: {incident_date}",
        f"Location: {location}",
        f"Reported By: {reported_by}",
        f"Date Reported: {incident_date}",
        f"Assigned Investigator: {investigator}",
        f"",
        f"TIMELINE OF EVENTS:",
        f"",
    ]
    # Build a plausible timeline
    timeline_events = [
        (f"{incident_date} (morning)", f"Normal operations at {location}"),
        (f"{incident_date} ({time_w1} per {w1})", f"Incident begins according to {w1}"),
        (f"{incident_date} ({time_w2} per {w2})", f"Incident occurs according to {w2}"),
        (f"{incident_date} (end of day)", f"{reported_by} files formal complaint with HR"),
        (f"{incident_date} + 1 day", f"HR acknowledges complaint; {investigator} assigned"),
        (f"{incident_date} + 3 days", f"Witness interviews conducted"),
    ]
    for time, event in timeline_events:
        timeline_lines.append(f"  {time}: {event}")

    timeline_lines.extend([
        f"",
        f"EVIDENCE COLLECTED:",
        f"  - Witness statement from {w1}",
        f"  - Witness statement from {w2}",
        f"  - Witness statement from {w3}",
        f"  - Security badge access log (pending)",
    ])
    timeline_content = "\n".join(timeline_lines) + "\n"

    # --- Problem statement ---
    problem_statement = f"""# HR Investigation Summary

You are {investigator}, an HR investigator at {company}. You have been
assigned to investigate an incident reported by {reported_by}: "{incident['title']}."

You have collected three witness interviews, the relevant company policy,
and an incident timeline. Your task is to write a formal investigation summary.

## Source Files
- /testbed/docs/interview_witness1.txt — Interview with {w1}
- /testbed/docs/interview_witness2.txt — Interview with {w2}
- /testbed/docs/interview_witness3.txt — Interview with {w3}
- /testbed/docs/company_policy.txt — Relevant policy excerpt
- /testbed/docs/incident_timeline.txt — Timeline of events

## Requirements
1. Summarize each witness's account
2. Identify contradictions and inconsistencies between statements
3. Cite the relevant company policy section
4. State what facts can be established vs. what remains disputed
5. Provide recommendations for next steps
6. Maintain objectivity — do not prejudge the outcome

Write your investigation summary to /testbed/investigation_summary.txt"""

    rubric: tuple[BinaryRubricCategory | RubricCategory, ...] = (
        BinaryRubricCategory(
            name="identified_time_contradiction",
            question=f"Does the summary identify the contradiction between {w1} and {w2} regarding the time of the incident ({time_w1} vs {time_w2})?",
            points=2,
        ),
        BinaryRubricCategory(
            name="identified_presence_contradiction",
            question=f"Does the summary identify the contradiction about whether {extra_person} was present ({w1} says yes, {w3} says no)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="identified_severity_contradiction",
            question=f"Does the summary identify the disagreement between {w2} and {w3} about the severity or nature of the incident?",
            points=2,
        ),
        BinaryRubricCategory(
            name="cited_policy_section",
            question=f'Does the summary cite the relevant policy section ("{incident["policy_section"]}")?',
            points=2,
        ),
        BinaryRubricCategory(
            name="mentioned_all_witnesses",
            question=f"Does the summary mention all three witnesses by name ({w1}, {w2}, and {w3})?",
            points=1,
        ),
        BinaryRubricCategory(
            name="timeline_referenced",
            question="Does the summary reference the incident timeline, including the date and location?",
            points=1,
        ),
        BinaryRubricCategory(
            name="mentioned_reporter",
            question=f"Does the summary mention {reported_by} as the person who filed the complaint?",
            points=1,
        ),
        BinaryRubricCategory(
            name="distinguishes_established_vs_disputed",
            question="Does the summary clearly distinguish between established facts and disputed claims?",
            points=2,
        ),
        BinaryRubricCategory(
            name="includes_recommendations",
            question="Does the summary include specific recommendations for next steps (e.g., additional evidence gathering, disciplinary review, mediation)?",
            points=1,
        ),
        RubricCategory(
            name="objectivity",
            description="Does the summary maintain objectivity and avoid prejudging the outcome?",
            failure="Summary is clearly biased, takes a side, or makes accusations without evidence.",
            minor_failure="Mostly objective but contains some loaded language or premature conclusions.",
            minor_success="Generally objective; presents evidence from all sides with minor bias.",
            success="Fully objective: presents all perspectives fairly, avoids conclusions beyond what evidence supports.",
            points=3,
        ),
        RubricCategory(
            name="thoroughness",
            description="How thoroughly does the summary address all aspects of the investigation?",
            failure="Missing most key elements (witness summaries, contradictions, policy, recommendations).",
            minor_failure="Covers some elements but misses important details from witness statements.",
            minor_success="Covers most elements; minor gaps in analysis.",
            success="Comprehensive coverage: all witnesses summarized, all contradictions identified, policy cited, clear recommendations.",
            points=3,
        ),
        RubricCategory(
            name="writing_quality",
            description="Is the summary written in clear, professional language appropriate for an HR investigation report?",
            failure="Poorly written, informal, or incoherent.",
            minor_failure="Understandable but reads like rough notes; inconsistent tone.",
            minor_success="Professional tone with clear structure; minor stylistic issues.",
            success="Polished, formal language with clear section headers and logical flow throughout.",
            points=2,
        ),
    )

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write your completed investigation summary to /testbed/investigation_summary.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/docs/interview_witness1.txt": interview_w1,
            "/testbed/docs/interview_witness2.txt": interview_w2,
            "/testbed/docs/interview_witness3.txt": interview_w3,
            "/testbed/docs/company_policy.txt": policy_content,
            "/testbed/docs/incident_timeline.txt": timeline_content,
        },
        problem_type="hr_investigation",
    )


# =============================================================================
# 5. COMPLIANCE AUDIT REPORT
# =============================================================================


def make_compliance_audit_report(rand_seed: int = 42) -> RubricDatapoint:
    """Given expense policy and expense reports, identify policy violations.

    HARDENED prototype: demonstrates signal-stripping, near-miss distractors,
    deep binary rubric, and compositional seedability.

    Seed varies: number of reports (5-7), number of violations (2-5),
    violation types (from pool of 12), which reports are clean, which have
    near-miss distractors, mileage route details, hotel cities.
    """
    rng = _random.Random(rand_seed)

    company = pick1(COMPANY_NAMES, rand_seed)
    auditor_name = random_name(rand_seed + 5)

    # --- Compositional seedability: vary report count and violation count ---
    n_reports = rng.randint(5, 7)
    employee_names = random_names(rand_seed + 10, n_reports)
    n_violations = rng.randint(2, 5)
    n_violations = min(n_violations, n_reports - 1)  # At least 1 clean report

    chosen_violations = rng.sample(COMPLIANCE_VIOLATION_TYPES, n_violations)

    # Assign violations to report indices
    violation_report_indices = sorted(rng.sample(range(n_reports), n_violations))
    clean_report_indices = [i for i in range(n_reports) if i not in violation_report_indices]

    violation_map: dict[int, dict] = {}
    for i, vi in enumerate(violation_report_indices):
        violation_map[vi] = chosen_violations[i]

    # Pick 1-2 clean reports to get near-miss distractors
    n_near_misses = min(rng.randint(1, 2), len(clean_report_indices))
    near_miss_report_indices = rng.sample(clean_report_indices, n_near_misses)
    near_miss_details: list[dict] = []  # Track for rubric

    # Pick a mileage route (for mileage violations or near-misses)
    mileage_route = rng.choice(MILEAGE_STANDARD_ROUTES)
    # Pick a hotel city (for hotel rate violations)
    hotel_city = rng.choice(list(HOTEL_PER_DIEM_RATES.keys()))
    hotel_rate_limit = HOTEL_PER_DIEM_RATES[hotel_city]

    # --- Build expanded expense policy ---
    policy_lines = [
        f"{company} — Expense Reimbursement Policy",
        f"Effective Date: January 1, 2024",
        f"",
        f"{'='*60}",
        f"SECTION 1: GENERAL RULES",
        f"{'='*60}",
        f"",
        f"1.1  All business expenses must be submitted within 30 calendar days",
        f"     of the date the expense was incurred.",
        f"1.2  Expenses over $25.00 require itemized receipts.",
        f"1.3  Credit card statements alone are NOT sufficient documentation.",
        f"1.4  Each expense line item must appear exactly once across all",
        f"     reports. Duplicate submissions will be rejected.",
        f"1.5  Splitting a single expense across multiple line items to",
        f"     circumvent receipt thresholds is prohibited.",
        f"",
        f"{'='*60}",
        f"SECTION 2: MEALS AND ENTERTAINMENT",
        f"{'='*60}",
        f"",
        f"2.1  Individual meal expenses must not exceed $75.00 per person",
        f"     per meal.",
        f"2.2  Group dining must not exceed $75.00 per attendee.",
        f"2.3  Alcohol is reimbursable ONLY during client entertainment",
        f"     events with documented prior approval.",
        f"2.4  Internal team events (no external clients present) do NOT",
        f"     qualify for alcohol reimbursement.",
        f"",
        f"{'='*60}",
        f"SECTION 3: TRAVEL",
        f"{'='*60}",
        f"",
        f"3.1  Domestic flights must be booked in economy class.",
        f"3.2  Business class requires VP-level pre-approval for flights",
        f"     under 6 hours.",
        f"3.3  Hotel rates must not exceed the GSA per-diem rate for the",
        f"     destination city (see Appendix A).",
        f"3.4  Mileage reimbursement: $0.67/mile for the standard route.",
        f"3.5  Mileage claims exceeding the standard route distance by more",
        f"     than 10% require written justification (see Appendix B).",
        f"",
        f"{'='*60}",
        f"SECTION 4: PROHIBITED EXPENSES",
        f"{'='*60}",
        f"",
        f"4.1  Personal items, gifts for family, and non-work entertainment",
        f"     are not reimbursable.",
        f"4.2  Each expense may be submitted only once.",
        f"",
        f"{'='*60}",
        f"SECTION 5: GIFTS AND GRATUITIES",
        f"{'='*60}",
        f"",
        f"5.1  Gifts to clients or business partners must not exceed $50.00",
        f"     per recipient per occasion without prior written approval",
        f"     from a department head.",
        f"5.2  Cash gifts or gift cards are never reimbursable.",
        f"",
        f"{'='*60}",
        f"SECTION 6: SUBSCRIPTIONS AND RECURRING CHARGES",
        f"{'='*60}",
        f"",
        f"6.1  Software subscriptions and recurring service charges require",
        f"     annual budget approval from the department manager.",
        f"6.2  Auto-renewal subscriptions must be documented and approved",
        f"     before the renewal date.",
        f"",
        f"{'='*60}",
        f"SECTION 7: APPROVAL AND COMPLIANCE",
        f"{'='*60}",
        f"",
        f"7.1  Managers must review and approve expense reports within 5",
        f"     business days of submission.",
        f"7.2  Finance will audit a random sample of reports each quarter.",
        f"7.3  Violations may result in repayment requirements and",
        f"     disciplinary action.",
        f"",
        f"{'='*60}",
        f"APPENDIX A: GSA PER-DIEM HOTEL RATES (2024)",
        f"{'='*60}",
        f"",
    ]
    for city, rate in sorted(HOTEL_PER_DIEM_RATES.items()):
        policy_lines.append(f"  {city:<25} ${rate}/night")
    policy_lines.extend([
        f"",
        f"{'='*60}",
        f"APPENDIX B: STANDARD MILEAGE ROUTES",
        f"{'='*60}",
        f"",
    ])
    for origin, dest, miles in MILEAGE_STANDARD_ROUTES:
        policy_lines.append(f"  {origin} — {dest}:  {miles} miles (one way)")
    policy_lines.append(f"")
    policy_content = "\n".join(policy_lines) + "\n"

    # --- Normal expense items (noise) ---
    normal_items = [
        ("Taxi to client site", (25, 55)),
        ("Working lunch — 1 attendee", (12, 35)),
        ("Office supplies — pens, notebooks", (8, 24)),
        ("Parking — downtown garage", (15, 40)),
        ("Coffee meeting — 2 attendees", (8, 22)),
        ("Train ticket to branch office", (15, 55)),
        ("Printing and binding services", (20, 65)),
        ("Conference room catering — 6 people", (45, 65)),
        ("Uber to airport", (28, 55)),
        ("Postage and shipping", (12, 35)),
        ("Office chair ergonomic assessment", (0, 0)),  # placeholder, won't be used with 0 range
        ("Team working lunch — 4 attendees", (48, 72)),
        ("Parking at conference venue", (18, 35)),
        ("Courier service — document delivery", (15, 30)),
    ]
    # Remove the placeholder
    normal_items = [(d, r) for d, r in normal_items if r != (0, 0)]

    # --- Build expense reports ---
    report_files: dict[str, str] = {}
    violation_details: list[dict] = []

    for report_idx in range(n_reports):
        emp_name = employee_names[report_idx]
        dept = rng.choice(["Sales", "Engineering", "Marketing", "Operations", "Finance", "Legal", "HR"])
        report_month = rng.randint(1, 12)
        report_day = rng.randint(1, 28)
        report_date = f"2024-{report_month:02d}-{report_day:02d}"
        report_id = f"EXP-{rng.randint(10000, 99999)}"

        lines = [
            f"EXPENSE REPORT",
            f"",
            f"Report ID: {report_id}",
            f"Employee: {emp_name}",
            f"Department: {dept}",
            f"Submission Date: {report_date}",
            f"Period: {report_date[:7]}",
            f"",
            f"{'Date':<14}{'Description':<45}{'Amount':>12}{'Receipt':>10}",
            f"{'-'*14}{'-'*45}{'-'*12}{'-'*10}",
        ]

        # Generate 6-10 normal line items (bigger haystack)
        n_items = rng.randint(6, 10)
        report_total = 0.0
        used_items = rng.sample(normal_items, min(n_items, len(normal_items)))
        for desc, (lo, hi) in used_items:
            day = rng.randint(1, 28)
            item_date = f"{report_date[:5]}{report_month:02d}-{day:02d}"
            amount = round(rng.uniform(lo, hi), 2)
            report_total += amount
            lines.append(f"{item_date:<14}{desc:<45}{_fmt_money(amount):>12}{'Yes':>10}")

        # --- Insert violation line item (signal-stripped) ---
        if report_idx in violation_map:
            viol = violation_map[report_idx]
            viol_amount = round(vary_number(viol["example_amount"], rand_seed + report_idx, 0.15), 2)
            day = rng.randint(1, 28)
            viol_date = f"{report_date[:5]}{report_month:02d}-{day:02d}"
            extra_detail = ""

            if viol["type"] == "over_limit_meal":
                # Stripped: no restaurant name. Just "Business dinner" with amount > $75.
                desc = f"Business dinner — 1 attendee"
                receipt = "Yes"

            elif viol["type"] == "missing_receipt":
                # Receipt column says "No" — model must notice amount > $25
                desc = rng.choice(["Ground transportation", "Taxi to meeting", "Ride to venue"])
                receipt = "No"
                viol_amount = round(rng.uniform(35, 85), 2)

            elif viol["type"] == "unapproved_travel_class":
                origin = rng.choice(["ORD", "JFK", "SFO", "BOS", "DFW"])
                dest = rng.choice(["LAX", "ATL", "DEN", "SEA", "MIA"])
                # Just say "Flight" — the high amount ($1200-1600) is the signal.
                # Model must know economy domestic is ~$200-500.
                desc = f"Flight — {origin} to {dest} (business class)"
                receipt = "Yes"
                viol_amount = round(rng.uniform(1200, 1600), 2)
                # No extra_detail — model must check policy for travel class rules

            elif viol["type"] == "personal_expense":
                # Stripped: no "Amazon", no obviously-personal item names.
                # Use items that are plausibly personal from context.
                item = rng.choice([
                    "Fitbit Charge 6 fitness tracker",
                    "Kindle Paperwhite e-reader",
                    "Nintendo Switch game cartridge",
                    "Yoga mat and resistance bands",
                    "Personal planner and journal set",
                ])
                desc = f"Equipment purchase — {item}"
                receipt = "Yes"

            elif viol["type"] == "duplicate_submission":
                # Stripped: NO "(dup)" label. Two identical lines, same date/desc/amount.
                hotel_name = rng.choice(["Marriott", "Hilton", "Hyatt", "Westin"])
                desc = f"Hotel — {hotel_name} Downtown"
                receipt = "Yes"
                # First copy
                lines.append(f"{viol_date:<14}{desc:<45}{_fmt_money(viol_amount):>12}{receipt:>10}")
                report_total += viol_amount
                # Second copy — identical, no label
                lines.append(f"{viol_date:<14}{desc:<45}{_fmt_money(viol_amount):>12}{receipt:>10}")
                report_total += viol_amount
                violation_details.append({
                    "report_num": report_idx + 1,
                    "report_id": report_id,
                    "employee": emp_name,
                    "type": viol["type"],
                    "description": viol["description"],
                    "amount": viol_amount,
                    "policy_section": viol["policy_section"],
                    "detail": f"'{desc}' on {viol_date} for {_fmt_money(viol_amount)} appears twice",
                })
                lines.extend(["", f"TOTAL: {_fmt_money(report_total)}", "", "Manager Approval: Pending"])
                report_files[f"/testbed/reports/expense_report_{report_idx + 1}.txt"] = "\n".join(lines) + "\n"
                continue

            elif viol["type"] == "late_submission":
                # Model must compute: expense date vs submission date > 30 days.
                # We set the expense date ~2 months before submission.
                late_month = max(1, report_month - 2) if report_month > 2 else report_month + 10
                late_year = "2024" if late_month <= report_month else "2023"
                viol_date = f"{late_year}-{late_month:02d}-{day:02d}"
                desc = rng.choice([
                    "Conference registration",
                    "Professional development course fee",
                    "Industry workshop attendance",
                ])
                receipt = "Yes"
                # No extra_detail — the date column already shows the old date;
                # model must compare it to the Submission Date at the top

            elif viol["type"] == "alcohol_internal":
                # Stripped: no "Bar tab" or "happy hour". Neutral restaurant name.
                n_people = rng.randint(4, 8)
                restaurant = rng.choice(["Giovanni's", "The Riverside Grill", "Sakura Bistro", "Oak & Ember"])
                desc = f"Team dinner — {restaurant} ({n_people} attendees)"
                receipt = "Yes"
                extra_detail = f"Attendees: {', '.join(random_names(rand_seed + report_idx + 100, n_people))} (all internal)"
                # Add a note field with the itemized breakdown showing alcohol
                alcohol_amt = round(viol_amount * rng.uniform(0.2, 0.35), 2)
                food_amt = round(viol_amount - alcohol_amt, 2)
                extra_detail += f"\n          Itemized: food ${food_amt:.2f}, beverages (wine/cocktails) ${alcohol_amt:.2f}"

            elif viol["type"] == "mileage_discrepancy":
                # Stripped: NO standard route distance shown. Model must look it up in policy Appendix B.
                origin, dest, std_miles = mileage_route
                # Claim 25-60% over standard
                overage = rng.uniform(0.25, 0.60)
                claimed_miles = round(std_miles * 2 * (1 + overage), 1)  # round trip
                viol_amount = round(claimed_miles * 0.67, 2)
                desc = f"Mileage — round trip {origin} to {dest}"
                receipt = "N/A"
                extra_detail = f"Claimed: {claimed_miles} miles"
                # NOTE: Standard route NOT shown here — model must look up
                # in policy Appendix B and compute the overage

            elif viol["type"] == "exceeded_hotel_rate":
                # Hotel in a specific city at rate above GSA per-diem.
                n_nights = rng.randint(2, 4)
                nightly_rate = round(hotel_rate_limit * rng.uniform(1.15, 1.45), 2)
                viol_amount = round(nightly_rate * n_nights, 2)
                desc = f"Hotel — {n_nights} nights in {hotel_city}"
                receipt = "Yes"
                extra_detail = f"Nightly rate: ${nightly_rate:.2f}"
                # NOTE: GSA per-diem NOT shown here — model must look up
                # in policy Appendix A and compare

            elif viol["type"] == "split_expense":
                # Two line items that are clearly parts of the same expense.
                split_total = round(rng.uniform(40, 60), 2)
                split_a = round(split_total * rng.uniform(0.45, 0.55), 2)
                split_b = round(split_total - split_a, 2)
                base_desc = rng.choice(["Office supplies", "Printing services", "Shipping materials"])
                desc_a = f"{base_desc} (part 1)"
                desc_b = f"{base_desc} (part 2)"
                receipt_a = "No"  # Each part is under $25 so "no receipt needed"
                receipt_b = "No"
                lines.append(f"{viol_date:<14}{desc_a:<45}{_fmt_money(split_a):>12}{receipt_a:>10}")
                lines.append(f"{viol_date:<14}{desc_b:<45}{_fmt_money(split_b):>12}{receipt_b:>10}")
                report_total += split_total
                violation_details.append({
                    "report_num": report_idx + 1,
                    "report_id": report_id,
                    "employee": emp_name,
                    "type": viol["type"],
                    "description": viol["description"],
                    "amount": split_total,
                    "policy_section": viol["policy_section"],
                    "detail": f"'{base_desc}' split into ${split_a:.2f} + ${split_b:.2f} = ${split_total:.2f} (total > $25, each part has no receipt)",
                })
                lines.extend(["", f"TOTAL: {_fmt_money(report_total)}", "", "Manager Approval: Pending"])
                report_files[f"/testbed/reports/expense_report_{report_idx + 1}.txt"] = "\n".join(lines) + "\n"
                continue

            elif viol["type"] == "gift_to_client":
                recipient = random_name(rand_seed + report_idx + 200)
                desc = f"Gift — {rng.choice(['leather portfolio', 'engraved pen set', 'wine gift basket', 'custom artwork'])} for {recipient}"
                receipt = "Yes"
                viol_amount = round(rng.uniform(75, 150), 2)

            elif viol["type"] == "unauthorized_recurring":
                svc = rng.choice(["SaaS analytics platform", "Cloud storage subscription", "Design tool license", "Project management tool"])
                desc = f"Monthly subscription — {svc}"
                receipt = "Yes"
                viol_amount = round(rng.uniform(29.99, 79.99), 2)
                extra_detail = f"Recurring charge — auto-renewed"
                # NOTE: Does not state approval status — model must check
                # policy Section 6 re: budget approval requirements

            else:
                desc = f"Miscellaneous expense"
                receipt = "Yes"

            lines.append(f"{viol_date:<14}{desc:<45}{_fmt_money(viol_amount):>12}{receipt:>10}")
            report_total += viol_amount

            # Add notes section if there's extra detail
            if extra_detail:
                lines.append(f"  Notes: {extra_detail}")

            violation_details.append({
                "report_num": report_idx + 1,
                "report_id": report_id,
                "employee": emp_name,
                "type": viol["type"],
                "description": viol["description"],
                "amount": viol_amount,
                "policy_section": viol["policy_section"],
                "detail": extra_detail if extra_detail else f"{desc} — {_fmt_money(viol_amount)}",
            })

        # --- Insert near-miss distractor (looks suspicious but compliant) ---
        if report_idx in near_miss_report_indices:
            nm_type = rng.choice(COMPLIANCE_NEAR_MISSES)
            nm_day = rng.randint(1, 28)
            nm_date = f"{report_date[:5]}{report_month:02d}-{nm_day:02d}"

            if nm_type["type"] == "meal_just_under_limit":
                n_att = rng.randint(1, 2)
                nm_desc = f"Business lunch — {n_att} attendee{'s' if n_att > 1 else ''}"
                nm_amount = round(rng.uniform(70.0, 74.99), 2)
                nm_receipt = "Yes"
            elif nm_type["type"] == "mileage_within_tolerance":
                nm_origin, nm_dest, nm_std = mileage_route
                nm_overage = rng.uniform(0.03, 0.09)
                nm_claimed = round(nm_std * 2 * (1 + nm_overage), 1)
                nm_amount = round(nm_claimed * 0.67, 2)
                nm_desc = f"Mileage — round trip {nm_origin} to {nm_dest}"
                nm_receipt = "N/A"
            elif nm_type["type"] == "high_value_with_receipt":
                nm_conf = rng.choice(["TechCrunch Disrupt", "AWS re:Invent", "SaaStr Annual", "Dreamforce"])
                nm_desc = f"Conference registration — {nm_conf}"
                nm_amount = round(rng.uniform(800, 1200), 2)
                nm_receipt = "Yes"
            elif nm_type["type"] == "alcohol_with_client":
                nm_restaurant = rng.choice(["The Palm", "Morton's", "Capital Grille"])
                client_name = random_name(rand_seed + report_idx + 300)
                nm_desc = f"Client dinner — {nm_restaurant} (4 attendees incl. {client_name})"
                nm_amount = round(rng.uniform(120, 200), 2)
                nm_receipt = "Yes"
            else:
                nm_desc = "Business expense"
                nm_amount = round(rng.uniform(50, 100), 2)
                nm_receipt = "Yes"

            lines.append(f"{nm_date:<14}{nm_desc:<45}{_fmt_money(nm_amount):>12}{nm_receipt:>10}")
            report_total += nm_amount
            near_miss_details.append({
                "report_num": report_idx + 1,
                "employee": emp_name,
                "type": nm_type["type"],
                "note": nm_type["note"],
                "desc": nm_desc,
                "amount": nm_amount,
            })

        lines.extend(["", f"TOTAL: {_fmt_money(report_total)}", "", "Manager Approval: Pending"])
        report_files[f"/testbed/reports/expense_report_{report_idx + 1}.txt"] = "\n".join(lines) + "\n"

    # --- Problem statement ---
    problem_statement = f"""# Compliance Audit — Expense Reports

You are {auditor_name}, a compliance auditor at {company}. You have been
assigned to audit {n_reports} recent expense reports against the company's
expense reimbursement policy.

## Source Files
- /testbed/policy/expense_policy.txt — Company expense reimbursement policy
  (includes appendices with GSA hotel rates and standard mileage routes)
- /testbed/reports/expense_report_1.txt through expense_report_{n_reports}.txt

## Requirements
1. Read the expense policy carefully, including ALL appendices
2. Review each of the {n_reports} expense reports line by line
3. Identify ALL policy violations, citing the specific policy section number
4. For each violation, state the dollar amount involved and the policy limit
5. Explicitly list which reports are clean (no violations found)
6. Provide a specific remediation recommendation for each violation

Write your audit report to /testbed/audit_report.txt"""

    # --- RUBRIC: 20-25 mostly binary categories ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = []

    # 1. Per-violation: "found it" check (2 pts each)
    for vd in violation_details:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"found_violation_report_{vd['report_num']}",
                question=(
                    f"Does the audit report identify a policy violation in Report #{vd['report_num']} "
                    f"({vd['employee']})? The violation is: {vd['description']}."
                ),
                points=2,
            )
        )

    # 2. Per-violation: correct type / categorization (2 pts each)
    for vd in violation_details:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"correct_type_report_{vd['report_num']}",
                question=(
                    f"Does the audit report correctly identify the nature of the violation in "
                    f"Report #{vd['report_num']} as being about "
                    f"'{vd['type'].replace('_', ' ')}'? (It need not use these exact words, "
                    f"but must convey the same issue.)"
                ),
                points=2,
            )
        )

    # 3. Per-violation: cites correct policy section (1 pt each)
    for vd in violation_details:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"cites_section_report_{vd['report_num']}",
                question=(
                    f"Does the audit report cite the correct policy section for the violation "
                    f"in Report #{vd['report_num']}? The correct section is "
                    f"{vd['policy_section']}."
                ),
                points=1,
            )
        )

    # 4. Per-violation: states the dollar amount (1 pt each)
    for vd in violation_details:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"states_amount_report_{vd['report_num']}",
                question=(
                    f"Does the audit report state the dollar amount involved in the violation "
                    f"in Report #{vd['report_num']}? The amount is approximately "
                    f"{_fmt_money(vd['amount'])}."
                ),
                points=1,
            )
        )

    # 5. Per-clean-report: no false positive (2 pts each)
    for ci in clean_report_indices:
        clean_emp = employee_names[ci]
        rubric_items.append(
            BinaryRubricCategory(
                name=f"no_false_positive_report_{ci + 1}",
                question=(
                    f"Does the audit report correctly treat Report #{ci + 1} "
                    f"({clean_emp}) as clean / compliant — i.e., it does NOT "
                    f"flag any false violations in this report?"
                ),
                points=2,
            )
        )

    # 6. Per-near-miss: no false positive on distractor (2 pts each)
    for nm in near_miss_details:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"no_false_positive_near_miss_report_{nm['report_num']}",
                question=(
                    f"Report #{nm['report_num']} contains a line item '{nm['desc']}' "
                    f"for {_fmt_money(nm['amount'])} that looks suspicious but is actually "
                    f"compliant ({nm['note']}). Does the audit report correctly avoid "
                    f"flagging this item as a violation?"
                ),
                points=2,
            )
        )

    # 7. Correct total violation count (2 pts)
    rubric_items.append(
        BinaryRubricCategory(
            name="correct_violation_count",
            question=(
                f"Does the audit report identify exactly {len(violation_details)} policy "
                f"violation(s) in total (not more, not fewer)?"
            ),
            points=2,
        )
    )

    # 8. Explicitly identifies clean reports (1 pt)
    clean_nums = ", ".join(f"#{ci + 1}" for ci in clean_report_indices)
    rubric_items.append(
        BinaryRubricCategory(
            name="identifies_clean_reports",
            question=(
                f"Does the audit report explicitly state which reports are clean/compliant? "
                f"The clean reports are: {clean_nums}."
            ),
            points=1,
        )
    )

    # 9. Report has executive summary section (1 pt)
    rubric_items.append(
        BinaryRubricCategory(
            name="has_executive_summary",
            question=(
                "Does the audit report include an executive summary or overview section "
                "that summarizes the total number of violations found and the overall "
                "compliance status?"
            ),
            points=1,
        )
    )

    # 10. Report provides remediation for each violation (1 pt)
    rubric_items.append(
        BinaryRubricCategory(
            name="provides_remediation_per_violation",
            question=(
                "Does the audit report provide a specific remediation recommendation "
                "for each identified violation (not just generic advice, but a concrete "
                "next step like 'obtain retroactive VP approval' or 'resubmit with receipt')?"
            ),
            points=1,
        )
    )

    # 11. Report does not fabricate data (2 pts)
    rubric_items.append(
        BinaryRubricCategory(
            name="no_fabricated_data",
            question=(
                "Does the audit report avoid fabricating or hallucinating expense amounts, "
                "employee names, report IDs, or policy sections that do not appear in the "
                "source documents?"
            ),
            points=2,
        )
    )

    # Build necessary_files
    necessary_files = {"/testbed/policy/expense_policy.txt": policy_content}
    necessary_files.update(report_files)

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=tuple(rubric_items),
        submission_instructions="Write your completed audit report to /testbed/audit_report.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files=necessary_files,
        problem_type="compliance_audit",
    )


# =============================================================================
# 6. PROJECT RISK REGISTER
# =============================================================================


def make_project_risk_register(rand_seed: int = 42) -> RubricDatapoint:
    """Given a project plan, resource allocation, and stakeholder notes,
    create a risk register identifying embedded risks.

    Seed varies: which risks are embedded, project domain, names.
    """
    rng = _random.Random(rand_seed)

    company = pick1(COMPANY_NAMES, rand_seed)
    pm_name = random_name(rand_seed + 1)
    project_names = [
        "Phoenix Platform Migration",
        "Atlas CRM Upgrade",
        "Horizon Cloud Deployment",
        "Mercury Data Warehouse",
        "Titan ERP Implementation",
        "Nova Customer Portal",
        "Catalyst AI Integration",
        "Summit Infrastructure Refresh",
    ]
    project_name = rng.choice(project_names)
    stakeholder_names = random_names(rand_seed + 20, 4)

    # --- Build milestones with embedded risks ---
    total_budget = vary_int(850000, rand_seed, 0.3)
    n_milestones = rng.randint(8, 12)

    # Resource team members
    team_members = random_names(rand_seed + 30, 6)
    roles = ["Backend Developer", "Frontend Developer", "QA Lead", "DevOps Engineer",
             "Data Engineer", "UX Designer"]

    # Decide which risks to embed
    risk_flags = {
        "resource_overallocation": rng.random() > 0.15,  # very likely
        "dependency_delay": rng.random() > 0.15,
        "budget_shortfall": rng.random() > 0.2,
        "single_point_of_failure": rng.random() > 0.15,
    }
    # Ensure at least 3 risks
    while sum(risk_flags.values()) < 3:
        key = rng.choice(list(risk_flags.keys()))
        risk_flags[key] = True

    # --- Build project plan CSV ---
    milestones = []
    start_month = rng.randint(1, 4)
    budget_allocated = 0.0

    milestone_templates = [
        ("Requirements Gathering", 1, []),
        ("Architecture Design", 1, ["Requirements Gathering"]),
        ("Backend Development — Phase 1", 2, ["Architecture Design"]),
        ("Frontend Development — Phase 1", 2, ["Architecture Design"]),
        ("Database Migration", 2, ["Backend Development — Phase 1"]),
        ("API Integration", 1, ["Backend Development — Phase 1", "Frontend Development — Phase 1"]),
        ("QA Testing — Phase 1", 1, ["Database Migration", "API Integration"]),
        ("Backend Development — Phase 2", 2, ["QA Testing — Phase 1"]),
        ("Frontend Development — Phase 2", 2, ["QA Testing — Phase 1"]),
        ("Performance Testing", 1, ["Backend Development — Phase 2", "Frontend Development — Phase 2"]),
        ("User Acceptance Testing", 1, ["Performance Testing"]),
        ("Production Deployment", 1, ["User Acceptance Testing"]),
    ]

    chosen_milestones = milestone_templates[:n_milestones]
    delayed_milestone = None
    single_failure_person = None

    for i, (name, duration, deps) in enumerate(chosen_milestones):
        month_start = min(start_month + i, 12)
        month_end = min(month_start + duration, 12)
        budget_share = round(total_budget * rng.uniform(0.05, 0.15), 2)
        budget_allocated += budget_share
        assigned = rng.sample(team_members, rng.randint(1, 3))
        status = "Not Started" if i > 2 else rng.choice(["Complete", "In Progress"])

        # Embed dependency delay risk
        if risk_flags["dependency_delay"] and i == 4 and delayed_milestone is None:
            status = "Delayed"
            delayed_milestone = name

        milestones.append({
            "id": f"M{i+1:02d}",
            "name": name,
            "start": f"2024-{month_start:02d}-01",
            "end": f"2024-{month_end:02d}-01",
            "duration_months": duration,
            "dependencies": "; ".join(deps) if deps else "None",
            "budget": budget_share,
            "assigned": "; ".join(assigned),
            "status": status,
        })

    plan_header = "ID,Milestone,Start,End,Duration (months),Dependencies,Budget,Assigned,Status"
    plan_rows = [plan_header]
    for m in milestones:
        plan_rows.append(
            f"{m['id']},{m['name']},{m['start']},{m['end']},{m['duration_months']},"
            f"{m['dependencies']},{m['budget']:.2f},{m['assigned']},{m['status']}"
        )
    plan_csv = "\n".join(plan_rows) + "\n"

    # --- Build resource allocation CSV ---
    resource_header = "Name,Role,Allocation (%),Projects Assigned,Available Hours/Week"
    resource_rows = [resource_header]

    overallocated_person = None
    for i, (member, role) in enumerate(zip(team_members, roles)):
        allocation = rng.randint(60, 90)
        other_projects = rng.randint(0, 2)
        available_hours = 40

        # Embed resource overallocation
        if risk_flags["resource_overallocation"] and i == 0 and overallocated_person is None:
            allocation = rng.randint(140, 180)  # clearly over 100%
            other_projects = rng.randint(2, 4)
            overallocated_person = member

        # Embed single point of failure — visible only in the data
        if risk_flags["single_point_of_failure"] and i == 2 and single_failure_person is None:
            single_failure_person = member
            # No label in the CSV — the model must notice this person appears
            # on many critical milestones and cross-reference the notes

        projects_str = f"{project_name}" + (f" + {other_projects} others" if other_projects else "")
        resource_rows.append(
            f"{member},{role},{allocation}%,{projects_str},{available_hours}"
        )

    resource_csv = "\n".join(resource_rows) + "\n"

    # --- Build stakeholder notes (symptom-based, NOT risk-diagnosing) ---
    # Stakeholders describe what they've observed, NOT what the risk is.
    # The model must cross-reference notes + CSV data to identify risks.

    budget_concern = ""
    if risk_flags["budget_shortfall"]:
        overspend_pct = rng.randint(15, 35)
        budget_concern = (
            f"\n{stakeholder_names[2]} (Finance Director):\n"
            f'"We had a few unplanned vendor invoices come through last month, and '
            f"I'm seeing some line items that don't match the original estimates. "
            f"I'd like {pm_name} to pull the latest actuals vs. forecast numbers "
            f'before our next steering committee."'
        )
        # The hard evidence is in a separate budget_actuals section
        # that the model must find and compute the overspend from

    delayed_note = ""
    if risk_flags["dependency_delay"] and delayed_milestone:
        delayed_note = (
            f"\n{stakeholder_names[1]} (Technical Lead):\n"
            f'"I talked to the vendor last week — they mentioned something about '
            f"their release schedule shifting. I haven't gotten a firm date yet. "
            f"Meanwhile, the team working on milestones downstream of "
            f'{delayed_milestone} has been asking when they can start."'
        )

    overalloc_note = ""
    if risk_flags["resource_overallocation"] and overallocated_person:
        overalloc_note = (
            f"\n{stakeholder_names[3]} (Resource Manager):\n"
            f'"I have been getting some complaints about code review turnaround. '
            f"A couple of PRs sat for over a week. Also, I noticed someone's "
            f"calendar was completely blocked — back-to-back meetings across "
            f'different project standups all day."'
        )

    spof_note = ""
    if risk_flags["single_point_of_failure"] and single_failure_person:
        spof_note = (
            f"\n{stakeholder_names[0]} (Project Sponsor):\n"
            f'"Last Tuesday {single_failure_person} was out sick, and apparently '
            f"nobody could troubleshoot an issue with the migration scripts. "
            f"The team just waited until the next day. That made me a little "
            f'nervous."'
        )

    # Add budget actuals appendix if budget shortfall is a risk
    budget_appendix = ""
    if risk_flags["budget_shortfall"]:
        actual_spent = round(float(total_budget) * (overspend_pct + 100) / 100 * 0.3, 2)
        forecast_spent = round(float(total_budget) * 0.3, 2)
        budget_appendix = (
            f"\n{'='*60}\n"
            f"APPENDIX: Budget Summary (through Phase 1)\n"
            f"{'='*60}\n"
            f"Total approved budget:  {_fmt_money(float(total_budget))}\n"
            f"Forecast spend (Phase 1):  {_fmt_money(forecast_spent)}\n"
            f"Actual spend (Phase 1):    {_fmt_money(actual_spent)}\n"
            f"Contingency reserve:       {_fmt_money(float(total_budget) * 0.05)}\n"
        )

    stakeholder_lines = [
        f"STAKEHOLDER MEETING NOTES",
        f"Project: {project_name}",
        f"Date: 2024-{start_month + 2:02d}-15",
        f"Attendees: {', '.join(stakeholder_names)}, {pm_name} (PM)",
        f"",
        f"{'='*60}",
        f"",
        f"{stakeholder_names[0]} (Project Sponsor):",
        f'"The project is critical for our Q4 launch. We cannot afford significant '
        f'delays. I need visibility into anything that could affect the timeline."',
        f"",
        f"{stakeholder_names[1]} (Technical Lead):",
        f'"Things are moving along. The team is heads-down on implementation. '
        f'I do want to flag a couple of observations from this past sprint."',
        delayed_note,
        budget_concern,
        overalloc_note,
        spof_note,
        f"",
        f"{pm_name} (Project Manager):",
        f'"Understood. I will review the project plan, resource data, and these '
        f"observations to compile a formal risk register. I'll have it ready "
        f'by end of week."',
        budget_appendix,
    ]
    stakeholder_content = "\n".join(stakeholder_lines) + "\n"

    # --- Problem statement ---
    problem_statement = f"""# Project Risk Register

You are {pm_name}, Project Manager for the "{project_name}" initiative at {company}.
After a stakeholder meeting, you have been asked to create a formal risk register.

## Source Files
- /testbed/data/project_plan.csv — Project plan with milestones, dependencies, budgets, status
- /testbed/data/resource_allocation.csv — Team member allocation and availability
- /testbed/docs/stakeholder_notes.txt — Notes from the stakeholder meeting

## Requirements
1. Analyze the project plan for schedule, dependency, and budget risks
2. Analyze resource allocation for overallocation and single-point-of-failure risks
3. Incorporate concerns raised by stakeholders
4. Create a risk register with columns: Risk ID, Description, Category, Likelihood, Impact, Severity, Mitigation Strategy, Owner
5. Assign severity ratings (Critical / High / Medium / Low)
6. Propose specific, actionable mitigation strategies

Write your risk register to /testbed/risk_register.txt"""

    # --- Rubric ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = []

    if risk_flags["resource_overallocation"]:
        rubric_items.append(
            BinaryRubricCategory(
                name="identified_overallocation",
                question=f"Does the risk register identify the resource overallocation risk for {overallocated_person} (allocated at over 100%)?",
                points=2,
            )
        )

    if risk_flags["dependency_delay"]:
        rubric_items.append(
            BinaryRubricCategory(
                name="identified_dependency_delay",
                question=f"Does the risk register identify the dependency delay risk due to the delayed '{delayed_milestone}' milestone and its cascading impact?",
                points=2,
            )
        )

    if risk_flags["budget_shortfall"]:
        rubric_items.append(
            BinaryRubricCategory(
                name="identified_budget_shortfall",
                question="Does the risk register identify the budget shortfall risk, noting that spending is tracking above forecast?",
                points=2,
            )
        )

    if risk_flags["single_point_of_failure"]:
        rubric_items.append(
            BinaryRubricCategory(
                name="identified_single_point_of_failure",
                question=f"Does the risk register identify {single_failure_person} as a single point of failure with unique legacy system expertise?",
                points=2,
            )
        )

    rubric_items.extend([
        BinaryRubricCategory(
            name="severity_assigned",
            question="Does the risk register assign a severity or priority level (e.g., Critical/High/Medium/Low) to each identified risk?",
            points=1,
        ),
        BinaryRubricCategory(
            name="mitigation_proposed",
            question="Does the risk register propose at least one mitigation strategy for each identified risk?",
            points=2,
        ),
        BinaryRubricCategory(
            name="has_required_columns",
            question="Does the risk register include at minimum these columns or fields: Risk ID/Number, Description, Severity/Priority, and Mitigation Strategy?",
            points=1,
        ),
        BinaryRubricCategory(
            name="owner_assigned",
            question="Does the risk register assign an owner or responsible party for each risk?",
            points=1,
        ),
        BinaryRubricCategory(
            name="likelihood_assessed",
            question="Does the risk register include a likelihood or probability assessment (e.g., High/Medium/Low or a percentage) for each risk?",
            points=1,
        ),
        BinaryRubricCategory(
            name="references_source_data",
            question="Does the risk register reference specific data from the project plan, resource allocation, or stakeholder notes to support the identified risks?",
            points=2,
        ),
    ])

    rubric_items.extend([
        RubricCategory(
            name="risk_analysis_depth",
            description="How deeply does the register analyze each risk, including root causes and cascading effects?",
            failure="Risks listed as one-liners with no analysis.",
            minor_failure="Brief descriptions but missing root cause analysis or cascading impacts.",
            minor_success="Good descriptions with some analysis of root causes and downstream effects.",
            success="Thorough analysis: each risk has root cause, cascading impacts, likelihood rationale, and impact assessment.",
            points=3,
        ),
        RubricCategory(
            name="mitigation_quality",
            description="Are the mitigation strategies specific, actionable, and realistic?",
            failure="No mitigations or only generic platitudes.",
            minor_failure="Mitigations are vague (e.g., 'monitor the situation').",
            minor_success="Most mitigations are specific and actionable with minor gaps.",
            success="All mitigations are specific, actionable, realistic, and include timelines or trigger conditions.",
            points=3,
        ),
        RubricCategory(
            name="register_presentation",
            description="Is the risk register well-formatted and suitable for stakeholder presentation?",
            failure="Unstructured text dump or missing register format entirely.",
            minor_failure="Some tabular structure but inconsistent formatting or missing headers.",
            minor_success="Clear register format with consistent columns; minor presentation issues.",
            success="Professional, well-formatted register with consistent columns, clear headers, and ready for executive review.",
            points=2,
        ),
    ])

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=tuple(rubric_items),
        submission_instructions="Write your completed risk register to /testbed/risk_register.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/project_plan.csv": plan_csv,
            "/testbed/data/resource_allocation.csv": resource_csv,
            "/testbed/docs/stakeholder_notes.txt": stakeholder_content,
        },
        problem_type="project_risk",
    )
