"""Procurement and insurance tasks: claim adjudication and vendor invoice validation.

Each factory generates a realistic scenario with necessary_files containing all
information required to solve the problem. Seeds control randomization of amounts,
names, item types, discrepancy patterns, etc.
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
# DOMAIN: INSURANCE
# =============================================================================

INCIDENT_TYPES = [
    {"key": "kitchen_fire", "label": "Kitchen fire", "description": "A grease fire originated in the kitchen, spreading to adjacent cabinets and causing smoke damage throughout the first floor.", "excluded": False},
    {"key": "electrical_fire", "label": "Electrical fire", "description": "Faulty wiring in the attic caused an electrical fire that damaged the roof structure and upper floor rooms.", "excluded": False},
    {"key": "burglary", "label": "Burglary / theft", "description": "Forced entry through the rear door; multiple rooms were ransacked and personal property was stolen.", "excluded": False},
    {"key": "windstorm", "label": "Windstorm damage", "description": "Severe windstorm ripped shingles from the roof and broke multiple windows, allowing rain to damage interior furnishings.", "excluded": False},
    {"key": "pipe_burst", "label": "Pipe burst (water damage)", "description": "A frozen pipe burst in the upstairs bathroom, flooding the second floor and causing water damage to ceilings and walls below.", "excluded": False},
    {"key": "tree_fall", "label": "Tree fall on roof", "description": "A large oak tree fell onto the roof during a storm, puncturing the roof deck and damaging the master bedroom.", "excluded": False},
    {"key": "vandalism", "label": "Vandalism", "description": "Unknown persons spray-painted exterior walls, smashed windows, and damaged landscaping and outdoor furniture.", "excluded": False},
    {"key": "lightning_strike", "label": "Lightning strike", "description": "A direct lightning strike caused a power surge that destroyed electronics and started a small fire in the attic.", "excluded": False},
    {"key": "arson_by_insured", "label": "Intentional fire set by insured", "description": "Investigation determined the fire was intentionally set by the policyholder.", "excluded": True, "exclusion_name": "Intentional damage by insured"},
    {"key": "flood_from_river", "label": "River flooding", "description": "The nearby river overflowed its banks after heavy rains, and rising water entered the ground floor of the home.", "excluded": True, "exclusion_name": "Flood (rising water from external sources)"},
]

EXCLUSION_LIST = [
    ("Flood", "Rising water from external sources, including river overflow, storm surge, and surface water runoff. Separate flood insurance is required."),
    ("Earthquake / earth movement", "Damage caused by earthquake, landslide, mudflow, sinkholes, or other earth movement. Separate earthquake policy required."),
    ("Intentional damage by insured", "Any damage intentionally caused or directed by any person insured under this policy."),
    ("Normal wear and deterioration", "Gradual deterioration, wear and tear, inherent vice, latent defect, or mechanical breakdown."),
    ("Vermin, insects, or rodents", "Damage caused by birds, vermin, rodents, insects, or other animals owned or kept by the insured."),
    ("Government action / war", "Loss resulting from seizure, confiscation, or destruction by government authority, or any act of war."),
    ("Nuclear hazard", "Nuclear reaction, radiation, or radioactive contamination, whether controlled or uncontrolled."),
    ("Neglect", "Failure by the insured to use all reasonable means to protect property at and after the time of a loss."),
    ("Mold", "Mold, fungus, or wet rot, unless resulting directly from a sudden and accidental covered peril."),
    ("Business property on premises", "Property used primarily for business purposes located on the insured premises. Business liability excluded."),
    ("Vehicles and aircraft", "Motor vehicles, aircraft, and their parts, except motorized equipment used for servicing the residence."),
    ("Land and soil settling", "Settling, cracking, shrinking, bulging, or expansion of foundations, walls, floors, or ceilings."),
]

# Items pool: (description_template, category, base_replacement_cost_range)
CLAIMABLE_ITEMS_POOL = [
    ("Living room sofa", "Furniture", (1800, 3500)),
    ("Dining table and chairs set", "Furniture", (1200, 2800)),
    ("Queen bedroom set (bed frame, mattress, nightstands)", "Furniture", (2000, 4000)),
    ("Bookshelf (solid wood)", "Furniture", (400, 900)),
    ("Recliner chair", "Furniture", (800, 1600)),
    ("65\" Samsung TV", "Electronics", (1200, 2200)),
    ("Apple MacBook Pro laptop", "Electronics", (1800, 2800)),
    ("Sony soundbar and subwoofer", "Electronics", (500, 1000)),
    ("Gaming console (PlayStation 5)", "Electronics", (400, 600)),
    ("Desktop computer and monitor", "Electronics", (1500, 2500)),
    ("Kitchen appliances (refrigerator)", "Appliances", (1500, 3000)),
    ("Washer and dryer set", "Appliances", (1200, 2200)),
    ("Dishwasher", "Appliances", (600, 1200)),
    ("Microwave oven", "Appliances", (200, 500)),
    ("Central vacuum system", "Appliances", (800, 1500)),
    ("Winter coats and jackets (3)", "Clothing", (600, 1200)),
    ("Designer suits (2)", "Clothing", (1200, 2400)),
    ("Shoes collection (8 pairs)", "Clothing", (400, 1000)),
    ("Diamond engagement ring", "Jewelry", (3000, 8000)),
    ("Gold watch", "Jewelry", (1500, 4000)),
    ("Power tool set (DeWalt)", "Tools", (800, 1500)),
    ("Lawn mower (riding)", "Tools", (2000, 4000)),
]

# Items that should NOT be covered
NON_COVERED_ITEMS_POOL = [
    ("Business laptop used for consulting work", "Business property", (1800, 2800)),
    ("Cash in home safe", "Cash (exceeds $200 limit)", (500, 1500)),
    ("Inventory for home-based Etsy business", "Business property", (1200, 3000)),
    ("Company-owned projector for work presentations", "Business property", (800, 1500)),
    ("Motorcycle stored in garage", "Vehicles", (5000, 12000)),
]

DEPRECIATION_CATEGORIES = {
    "Furniture": {"useful_life": 10, "method": "straight-line"},
    "Electronics": {"useful_life": 5, "method": "straight-line"},
    "Appliances": {"useful_life": 12, "method": "straight-line"},
    "Clothing": {"useful_life": 5, "method": "straight-line"},
    "Jewelry": {"useful_life": None, "method": "market value (no depreciation)"},
    "Tools": {"useful_life": 15, "method": "straight-line"},
}


# =============================================================================
# 1. INSURANCE CLAIM ADJUDICATION
# =============================================================================


def make_insurance_claim_adjudication(rand_seed: int = 42) -> RubricDatapoint:
    """Given a homeowner's insurance policy, claim form, and depreciation
    schedule, determine coverage, compute actual cash value, apply deductible,
    and compute payout.

    Seed varies: incident type, deductible, coverage limits, items claimed,
    purchase years, non-covered items, and whether sublimit is exceeded.
    """
    rng = _random.Random(rand_seed)

    # --- Insured identity ---
    insured_name = random_name(rand_seed)
    address_num = rng.randint(100, 9999)
    street_names = [
        "Maple Street", "Oak Avenue", "Elm Drive", "Cedar Lane",
        "Pine Court", "Birch Road", "Walnut Boulevard", "Spruce Way",
    ]
    street = rng.choice(street_names)
    city_state_pool = [
        ("Springfield", "IL"), ("Portland", "OR"), ("Charlotte", "NC"),
        ("Denver", "CO"), ("Austin", "TX"), ("Raleigh", "NC"),
        ("Boise", "ID"), ("Tucson", "AZ"),
    ]
    city, state = rng.choice(city_state_pool)
    zip_code = f"{rng.randint(10000, 99999)}"
    address = f"{address_num} {street}, {city}, {state} {zip_code}"

    # --- Policy parameters ---
    dwelling_limit = rng.choice([250000, 300000, 350000, 400000, 450000, 500000])
    personal_property_limit = dwelling_limit * 0.50
    other_structures_limit = dwelling_limit * 0.10
    ale_limit = dwelling_limit * 0.20
    liability_limit = rng.choice([100000, 200000, 300000, 500000])
    deductible = rng.choice([500, 1000, 2000, 2500])

    # Jewelry sublimit (standard in homeowner's policies)
    jewelry_sublimit = rng.choice([1500, 2000, 2500])

    # Policy dates
    policy_number = f"HO-{rng.randint(100000, 999999)}"
    policy_start_year = 2024
    policy_start_month = rng.randint(1, 6)
    policy_end_month = policy_start_month
    policy_end_year = policy_start_year + 1

    # --- Incident ---
    incident = rng.choice(INCIDENT_TYPES)
    is_excluded = incident["excluded"]

    # Claim date (within policy period)
    claim_month = rng.randint(policy_start_month + 1, 12)
    claim_day = rng.randint(1, 28)
    claim_date = f"{policy_start_year}-{claim_month:02d}-{claim_day:02d}"

    # --- Select claimed items ---
    n_covered_items = rng.randint(7, 12)
    n_non_covered = rng.randint(1, 3)

    chosen_covered = rng.sample(CLAIMABLE_ITEMS_POOL, n_covered_items)
    chosen_non_covered = rng.sample(NON_COVERED_ITEMS_POOL, n_non_covered)

    # Decide if we should force sublimit exceedance on one seed variant
    # Use a deterministic check: if rand_seed % 5 == 0, force sublimit breach
    force_sublimit_breach = (rand_seed % 5 == 0)

    # Build item records with purchase years and replacement costs
    current_year = 2024
    items: list[dict] = []
    for desc, category, (lo, hi) in chosen_covered:
        replacement_cost = round(rng.uniform(lo, hi), 2)
        # Purchase year: 1 to useful_life years ago (or up to 8 for jewelry)
        max_age = DEPRECIATION_CATEGORIES[category]["useful_life"] or 8
        purchase_year = current_year - rng.randint(1, min(max_age, 8))
        items.append({
            "description": desc,
            "category": category,
            "replacement_cost": replacement_cost,
            "purchase_year": purchase_year,
            "covered": True,
            "non_covered_reason": None,
        })

    for desc, reason, (lo, hi) in chosen_non_covered:
        replacement_cost = round(rng.uniform(lo, hi), 2)
        purchase_year = current_year - rng.randint(1, 5)
        items.append({
            "description": desc,
            "category": reason,
            "replacement_cost": replacement_cost,
            "purchase_year": purchase_year,
            "covered": False,
            "non_covered_reason": reason,
        })

    # If forcing sublimit breach, inflate jewelry items or add expensive ones
    if force_sublimit_breach:
        # Make sure there's at least one jewelry item that exceeds the sublimit
        has_jewelry = any(it["category"] == "Jewelry" and it["covered"] for it in items)
        if not has_jewelry:
            items.append({
                "description": "Diamond necklace",
                "category": "Jewelry",
                "replacement_cost": round(rng.uniform(5000, 10000), 2),
                "purchase_year": current_year - rng.randint(1, 5),
                "covered": True,
                "non_covered_reason": None,
            })

    # Shuffle items so covered and non-covered are intermixed
    rng.shuffle(items)

    # --- Compute ground-truth values ---
    # ACV computation for each covered item
    acv_details: list[dict] = []
    total_replacement_claimed = 0.0
    total_acv_covered = 0.0
    non_covered_item_names: list[str] = []
    jewelry_acv_total = 0.0

    for it in items:
        total_replacement_claimed += it["replacement_cost"]
        if not it["covered"]:
            non_covered_item_names.append(it["description"])
            continue

        age = current_year - it["purchase_year"]
        cat_info = DEPRECIATION_CATEGORIES.get(it["category"])
        if cat_info and cat_info["useful_life"] is not None:
            useful_life = cat_info["useful_life"]
            depreciation_pct = min(age / useful_life, 1.0)
            depreciation_amount = round(it["replacement_cost"] * depreciation_pct, 2)
            acv = round(it["replacement_cost"] - depreciation_amount, 2)
        else:
            # Jewelry — no depreciation, ACV = replacement cost
            depreciation_amount = 0.0
            acv = it["replacement_cost"]

        if it["category"] == "Jewelry":
            jewelry_acv_total += acv

        acv_details.append({
            "description": it["description"],
            "category": it["category"],
            "replacement_cost": it["replacement_cost"],
            "purchase_year": it["purchase_year"],
            "age": age,
            "depreciation_amount": depreciation_amount,
            "acv": acv,
        })
        total_acv_covered += acv

    total_replacement_claimed = round(total_replacement_claimed, 2)
    total_acv_covered = round(total_acv_covered, 2)
    jewelry_acv_total = round(jewelry_acv_total, 2)

    # Apply jewelry sublimit: if jewelry ACV exceeds sublimit, cap it
    jewelry_excess = max(0.0, jewelry_acv_total - jewelry_sublimit)
    acv_after_jewelry_cap = round(total_acv_covered - jewelry_excess, 2)

    # Apply personal property sublimit
    sublimit_applied = acv_after_jewelry_cap > personal_property_limit
    acv_after_sublimit = min(acv_after_jewelry_cap, personal_property_limit)

    # If claim is excluded, payout is $0
    if is_excluded:
        final_payout = 0.0
        amount_after_deductible = 0.0
    else:
        amount_after_deductible = round(max(0.0, acv_after_sublimit - deductible), 2)
        final_payout = amount_after_deductible

    # Pick the 3 most expensive covered items for per-item rubric checks
    acv_sorted = sorted(acv_details, key=lambda x: x["acv"], reverse=True)
    top_items = acv_sorted[:3]

    # --- Build policy_declarations.txt ---
    policy_lines = [
        "HOMEOWNER'S INSURANCE POLICY — DECLARATIONS PAGE",
        "",
        f"Policy Number: {policy_number}",
        f"Insured: {insured_name}",
        f"Property Address: {address}",
        f"Effective Date: {policy_start_year}-{policy_start_month:02d}-01",
        f"Expiration Date: {policy_end_year}-{policy_end_month:02d}-01",
        "",
        "=" * 60,
        "COVERAGE SUMMARY",
        "=" * 60,
        "",
        f"{'Coverage Type':<35} {'Limit':>15}",
        f"{'-'*35} {'-'*15}",
        f"{'Coverage A — Dwelling':<35} {_fmt_money(dwelling_limit):>15}",
        f"{'Coverage B — Other Structures':<35} {_fmt_money(other_structures_limit):>15}",
        f"{'Coverage C — Personal Property':<35} {_fmt_money(personal_property_limit):>15}",
        f"{'Coverage D — Add. Living Expenses':<35} {_fmt_money(ale_limit):>15}",
        f"{'Coverage E — Personal Liability':<35} {_fmt_money(liability_limit):>15}",
        "",
        f"Deductible (all perils): {_fmt_money(deductible)}",
        "",
        "SPECIAL SUBLIMITS:",
        f"  Jewelry, watches, furs: {_fmt_money(jewelry_sublimit)} per occurrence",
        f"  Cash and securities: $200 maximum",
        f"  Business property on premises: NOT COVERED (see exclusions)",
        f"  Firearms: $2,500 maximum",
        "",
        "PERILS INSURED AGAINST:",
        "  This is an HO-3 (Special Form) policy. All risks of direct physical",
        "  loss to covered property are insured unless specifically excluded.",
        "  See policy exclusions document for a complete list of exclusions.",
        "",
    ]
    policy_content = "\n".join(policy_lines) + "\n"

    # --- Build policy_exclusions.txt ---
    exclusion_lines = [
        "HOMEOWNER'S INSURANCE POLICY — EXCLUSIONS",
        "",
        f"Policy Number: {policy_number}",
        "",
        "The following perils and conditions are EXCLUDED from coverage",
        "under this policy. Claims arising from these causes will be denied.",
        "",
        "=" * 60,
    ]
    for i, (name, desc) in enumerate(EXCLUSION_LIST, 1):
        exclusion_lines.append(f"")
        exclusion_lines.append(f"EXCLUSION {i}: {name}")
        exclusion_lines.append(f"  {desc}")
    exclusion_lines.append("")
    exclusion_lines.append("=" * 60)
    exclusion_lines.append("")
    exclusion_lines.append("If you believe your claim involves a covered peril, please")
    exclusion_lines.append("consult with your claims adjuster for a detailed review.")
    exclusion_lines.append("")
    exclusion_content = "\n".join(exclusion_lines) + "\n"

    # --- Build claim_form.txt ---
    claim_lines = [
        "HOMEOWNER'S INSURANCE — CLAIM FORM",
        "",
        f"Claimant: {insured_name}",
        f"Property Address: {address}",
        f"Date of Loss: {claim_date}",
        f"Date Reported: {claim_date}",
        "",
        "=" * 60,
        "INCIDENT DESCRIPTION",
        "=" * 60,
        "",
        f"Type of Loss: {incident['label']}",
        "",
        incident["description"],
        "",
        "=" * 60,
        "DAMAGED / LOST PROPERTY",
        "=" * 60,
        "",
        f"{'#':<4} {'Item Description':<45} {'Replacement Cost':>17} {'Purchased':>10}",
        f"{'-'*4} {'-'*45} {'-'*17} {'-'*10}",
    ]
    for idx, it in enumerate(items, 1):
        claim_lines.append(
            f"{idx:<4} {it['description']:<45} {_fmt_money(it['replacement_cost']):>17} {it['purchase_year']:>10}"
        )
    claim_lines.append("")
    claim_lines.append(f"Total items claimed: {len(items)}")
    claim_lines.append(f"Total replacement value claimed: {_fmt_money(total_replacement_claimed)}")
    claim_lines.append("")
    claim_lines.append("I certify that the above information is true and accurate to the")
    claim_lines.append("best of my knowledge.")
    claim_lines.append("")
    claim_lines.append(f"Signature: {insured_name}")
    claim_lines.append(f"Date: {claim_date}")
    claim_lines.append("")
    claim_content = "\n".join(claim_lines) + "\n"

    # --- Build depreciation_schedule.txt ---
    dep_lines = [
        "DEPRECIATION SCHEDULE — HOMEOWNER'S INSURANCE",
        "",
        "This schedule is used to compute Actual Cash Value (ACV) for claimed items.",
        "ACV = Replacement Cost - Accumulated Depreciation",
        "",
        "For straight-line depreciation:",
        "  Annual Depreciation Rate = 1 / Useful Life",
        "  Accumulated Depreciation = Replacement Cost x (Age / Useful Life)",
        "  (Depreciation cannot exceed 100% of replacement cost)",
        "",
        "=" * 60,
        "",
        f"{'Category':<20} {'Useful Life':>12} {'Method':<30}",
        f"{'-'*20} {'-'*12} {'-'*30}",
    ]
    for cat_name, info in DEPRECIATION_CATEGORIES.items():
        life_str = f"{info['useful_life']} years" if info["useful_life"] else "N/A"
        dep_lines.append(f"{cat_name:<20} {life_str:>12} {info['method']:<30}")
    dep_lines.append("")
    dep_lines.append("NOTES:")
    dep_lines.append("  - Jewelry is valued at market/replacement value (no depreciation).")
    dep_lines.append("  - Items older than their useful life have zero ACV.")
    dep_lines.append("  - Items must be personal property used for non-business purposes.")
    dep_lines.append("")
    dep_content = "\n".join(dep_lines) + "\n"

    # --- Problem statement ---
    problem_statement = f"""# Insurance Claim Adjudication

You are an insurance claims adjuster. Using the provided policy declarations,
exclusions list, claim form, and depreciation schedule, evaluate this homeowner's
insurance claim and compute the correct payout.

## Source Files
- /testbed/data/policy_declarations.txt — Policy coverage limits, deductible, and sublimits
- /testbed/data/policy_exclusions.txt — List of excluded perils
- /testbed/data/claim_form.txt — Incident details and list of damaged/lost items
- /testbed/data/depreciation_schedule.txt — Depreciation rates by item category

## Requirements
1. Verify the claim date falls within the policy period
2. Determine whether the incident type is covered or excluded
3. For each claimed item, determine if it is covered personal property
4. For covered items, compute the Actual Cash Value (ACV) using the depreciation schedule
5. Identify any non-covered items and explain why they are excluded
6. Sum total ACV for all covered items
7. Apply any applicable sublimits (e.g., jewelry sublimit)
8. Apply the policy deductible
9. Check against the personal property coverage limit
10. Compute the final claim payout

Write a detailed adjudication report to /testbed/adjudication_report.txt showing
your analysis and calculations at each step."""

    # Compute counts for rubric
    n_covered_items_actual = len(acv_details)
    n_non_covered_items_actual = len(non_covered_item_names)

    # --- Build rubric ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does the file /testbed/adjudication_report.txt exist and contain substantive content (more than a few lines)?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_incident_type_identified",
            question=f'Does the report correctly identify the incident/peril type as "{incident["label"]}"?',
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_policy_period_verified",
            question=(
                f"Does the report verify that the claim date ({claim_date}) falls within the "
                f"policy period ({policy_start_year}-{policy_start_month:02d}-01 to "
                f"{policy_end_year}-{policy_end_month:02d}-01)?"
            ),
            points=2,
        ),
    ]

    if is_excluded:
        rubric_items.extend([
            BinaryRubricCategory(
                name="correct_coverage_determination",
                question="Does the report correctly determine that this claim is NOT covered (i.e., the claim should be denied)?",
                points=3,
            ),
            BinaryRubricCategory(
                name="correct_exclusion_identified",
                question=f'Does the report identify the applicable exclusion as "{incident["exclusion_name"]}"?',
                points=2,
            ),
            BinaryRubricCategory(
                name="correct_total_claimed",
                question=f"Does the report note the total replacement value claimed as approximately {_fmt_money(total_replacement_claimed)} (within $50)?",
                points=1,
            ),
            BinaryRubricCategory(
                name="correct_final_payout",
                question="Does the report state the final payout as $0.00 (or equivalent) because the claim is excluded?",
                points=3,
            ),
            BinaryRubricCategory(
                name="explains_denial_clearly",
                question="Does the report clearly explain WHY the claim is denied, citing both the specific exclusion and a brief description of what it covers?",
                points=2,
            ),
            BinaryRubricCategory(
                name="lists_all_items_claimed",
                question=f"Does the report list or acknowledge all {len(items)} items from the claim form, even though the claim is denied?",
                points=1,
            ),
        ])
    else:
        rubric_items.extend([
            BinaryRubricCategory(
                name="correct_coverage_determination",
                question="Does the report correctly determine that this claim IS covered under the policy?",
                points=3,
            ),
            BinaryRubricCategory(
                name="no_false_exclusion",
                question="Does the report correctly refrain from citing an exclusion that doesn't apply? (It should not claim the loss is excluded.)",
                points=2,
            ),
        ])

        # Per-item ACV checks for top 3 most expensive covered items
        for i, item_detail in enumerate(top_items):
            rubric_items.append(
                BinaryRubricCategory(
                    name=f"correct_acv_item_{i+1}",
                    question=(
                        f'Is the Actual Cash Value of "{item_detail["description"]}" computed as '
                        f'approximately {_fmt_money(item_detail["acv"])} (within $50)? '
                        f'(Replacement cost {_fmt_money(item_detail["replacement_cost"])}, '
                        f'purchased {item_detail["purchase_year"]}, '
                        f'category: {item_detail["category"]}, age: {item_detail["age"]} years)'
                    ),
                    points=2,
                )
            )

        rubric_items.extend([
            BinaryRubricCategory(
                name="correct_total_claimed",
                question=f"Does the report state the total replacement cost claimed as approximately {_fmt_money(total_replacement_claimed)} (within $50)?",
                points=1,
            ),
            BinaryRubricCategory(
                name="correct_total_acv",
                question=f"Does the report compute the total ACV of all covered items as approximately {_fmt_money(total_acv_covered)} (within $100)?",
                points=2,
            ),
            BinaryRubricCategory(
                name="correct_non_covered_items",
                question=(
                    f"Does the report identify all non-covered items? The non-covered items are: "
                    f"{', '.join(non_covered_item_names)}. Each should be flagged as not eligible "
                    f"for coverage with a reason."
                ),
                points=2,
            ),
            BinaryRubricCategory(
                name="correct_covered_item_count",
                question=f"Does the report correctly identify {n_covered_items_actual} items as covered (eligible for ACV calculation)?",
                points=1,
            ),
            BinaryRubricCategory(
                name="correct_non_covered_item_count",
                question=f"Does the report correctly identify {n_non_covered_items_actual} item(s) as non-covered or excluded from the claim?",
                points=1,
            ),
            BinaryRubricCategory(
                name="identifies_depreciation_method",
                question="Does the report mention or demonstrate the use of straight-line depreciation when computing ACV for at least one item?",
                points=1,
            ),
            BinaryRubricCategory(
                name="correct_deductible_applied",
                question=f"Does the report apply the correct deductible of {_fmt_money(deductible)}?",
                points=2,
            ),
        ])

        if sublimit_applied:
            rubric_items.append(
                BinaryRubricCategory(
                    name="correct_sublimit_check",
                    question=f"Does the report correctly identify that the total ACV exceeds the personal property sublimit of {_fmt_money(personal_property_limit)} and cap the payout accordingly?",
                    points=2,
                )
            )
        else:
            rubric_items.append(
                BinaryRubricCategory(
                    name="correct_sublimit_check",
                    question=f"Does the report correctly note that the total ACV ({_fmt_money(acv_after_jewelry_cap)}) is within the personal property limit of {_fmt_money(personal_property_limit)}, so no sublimit cap is needed?",
                    points=2,
                )
            )

        if jewelry_excess > 0:
            rubric_items.append(
                BinaryRubricCategory(
                    name="correct_jewelry_sublimit",
                    question=f"Does the report correctly apply the jewelry sublimit of {_fmt_money(jewelry_sublimit)}, reducing the jewelry payout by {_fmt_money(jewelry_excess)}?",
                    points=2,
                )
            )

        rubric_items.append(
            BinaryRubricCategory(
                name="correct_final_payout",
                question=f"Does the report compute the final claim payout as approximately {_fmt_money(final_payout)} (within $100)?",
                points=3,
            )
        )

        rubric_items.append(
            BinaryRubricCategory(
                name="shows_depreciation_calculations",
                question="Does the report show step-by-step depreciation calculations (age, depreciation rate, depreciation amount, ACV) for at least 3 individual items?",
                points=2,
            )
        )

    rubric_items.append(
        RubricCategory(
            name="presentation_quality",
            description="Is the adjudication report well-organized, professional, and clearly presented?",
            failure="Disorganized output or raw numbers without context.",
            minor_failure="Some structure but missing key sections or hard to follow.",
            minor_success="Reasonably organized with clear labels and logical flow.",
            success="Professional adjudication report with clear sections, labeled calculations, and a definitive conclusion.",
            points=2,
        )
    )

    rubric = tuple(rubric_items)

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write your completed adjudication report to /testbed/adjudication_report.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/policy_declarations.txt": policy_content,
            "/testbed/data/policy_exclusions.txt": exclusion_content,
            "/testbed/data/claim_form.txt": claim_content,
            "/testbed/data/depreciation_schedule.txt": dep_content,
        },
        problem_type="insurance_claim_adjudication",
    )


# =============================================================================
# DOMAIN: VENDOR INVOICE VALIDATION
# =============================================================================

VENDOR_NAMES_POOL = [
    "Precision Parts International",
    "Midwest Industrial Supply Co.",
    "Atlas Manufacturing Group",
    "Summit Hardware Distributors",
    "Pacific Component Systems",
    "Continental Fastener Corp.",
    "Reliable Tool & Equipment",
    "Eagle Industrial Products",
    "Sterling Supply Chain LLC",
    "Ironworks Distribution Inc.",
]

# Item pool: (item_code, description, base_unit_price_range)
PO_ITEMS_POOL = [
    ("WDG-1001", "Industrial Widget A", (35.00, 55.00)),
    ("GSK-2003", "Gasket Set Type B", (8.00, 18.00)),
    ("BRG-3010", "Sealed Ball Bearing 6205", (12.00, 25.00)),
    ("FLT-4022", "Hydraulic Filter Element", (22.00, 42.00)),
    ("VLV-5005", "Solenoid Valve 24V DC", (65.00, 120.00)),
    ("BLT-6001", "Hex Bolt M12x50 (box/100)", (28.00, 48.00)),
    ("PMP-7003", "Centrifugal Pump Impeller", (85.00, 160.00)),
    ("SEN-8010", "Temperature Sensor PT100", (35.00, 65.00)),
    ("MTR-9002", "AC Motor 2HP 3-Phase", (180.00, 320.00)),
    ("CBL-1010", "Control Cable 4-core (per m)", (3.50, 8.00)),
    ("HSG-1101", "Bearing Housing SNL 510", (95.00, 175.00)),
    ("CPN-1205", "Coupling Flexible Jaw Type", (42.00, 78.00)),
    ("GRK-1302", "Grease Cartridge EP2 (case)", (18.00, 32.00)),
    ("SFT-1400", "Drive Shaft 25mm (per m)", (55.00, 95.00)),
    ("PLT-1503", "Mounting Plate Steel 10mm", (28.00, 52.00)),
    ("SPR-1601", "Compression Spring Kit", (15.00, 30.00)),
    ("NZL-1705", "Spray Nozzle Stainless", (45.00, 85.00)),
    ("CLM-1802", "Pipe Clamp 4-inch", (8.00, 16.00)),
    ("RLR-1900", "Conveyor Roller 500mm", (32.00, 58.00)),
    ("INS-2001", "Thermal Insulation Wrap (roll)", (22.00, 40.00)),
]

# Extra item that won't be on the PO (for "not ordered" discrepancy)
NOT_ORDERED_ITEMS = [
    ("EXT-9901", "Rush Delivery Surcharge", (150.00, 350.00)),
    ("EXT-9902", "Packaging and Handling Fee", (75.00, 200.00)),
    ("EXT-9903", "Environmental Disposal Fee", (50.00, 150.00)),
    ("MIS-8801", "Spare Parts Kit (unsolicited)", (120.00, 280.00)),
    ("MIS-8802", "Warranty Extension Package", (200.00, 500.00)),
]

DISCREPANCY_TYPES = [
    "qty_mismatch",
    "price_mismatch",
    "not_ordered",
    "short_delivery",
    "double_billed",
    "substitution",
]


def make_vendor_invoice_validation(rand_seed: int = 42) -> RubricDatapoint:
    """Three-way matching: compare a Purchase Order, Delivery Receipt, and
    Vendor Invoice to find discrepancies and compute the correct payment.

    Seed varies: vendor, items, quantities, prices, discrepancy types and
    locations, payment terms, delivery timing.
    """
    rng = _random.Random(rand_seed)

    # --- Header info ---
    vendor_name = rng.choice(VENDOR_NAMES_POOL)
    buyer_company = pick1(COMPANY_NAMES, rand_seed)
    buyer_contact = random_name(rand_seed + 1)

    po_number = f"PO-{rng.randint(10000, 99999)}"
    invoice_number = f"INV-{rng.randint(100000, 999999)}"

    # Dates
    po_month = rng.randint(1, 9)
    po_day = rng.randint(1, 28)
    po_date = f"2024-{po_month:02d}-{po_day:02d}"
    required_by_day = min(28, po_day + rng.randint(14, 30))
    required_by_month = po_month + (1 if required_by_day <= po_day else 0)
    if required_by_month > 12:
        required_by_month = 12
        required_by_day = 28
    required_by_date = f"2024-{required_by_month:02d}-{required_by_day:02d}"

    # Delivery date: sometimes late
    is_late_delivery = rng.random() < 0.4
    if is_late_delivery:
        late_days = rng.randint(3, 21)
        # Parse required_by to compute actual delivery date
        rb_m, rb_d = required_by_month, required_by_day
        delivery_day = rb_d + late_days
        delivery_month = rb_m
        while delivery_day > 28:
            delivery_day -= 28
            delivery_month += 1
        if delivery_month > 12:
            delivery_month = 12
            delivery_day = 28
        delivery_date = f"2024-{delivery_month:02d}-{delivery_day:02d}"
        weeks_late = max(1, late_days // 7)
        late_penalty_pct = min(weeks_late * 0.01, 0.05)  # 1% per week, max 5%
    else:
        # On time: deliver between PO date and required-by date
        delivery_day = rng.randint(po_day, required_by_day) if required_by_month == po_month else rng.randint(1, required_by_day)
        delivery_month = required_by_month
        delivery_date = f"2024-{delivery_month:02d}-{delivery_day:02d}"
        late_penalty_pct = 0.0
        weeks_late = 0

    # Invoice date: same as or shortly after delivery
    inv_day = min(28, delivery_day + rng.randint(0, 3))
    inv_month = delivery_month
    invoice_date = f"2024-{inv_month:02d}-{inv_day:02d}"

    # Early payment discount
    payment_net_days = rng.choice([30, 45, 60])
    early_discount_pct = 0.02  # 2/10 Net X
    early_discount_days = 10

    # Payment processing date: 3-15 days after invoice
    days_since_invoice = rng.randint(3, 15)
    payment_day = min(28, inv_day + days_since_invoice)
    payment_month = inv_month + (1 if payment_day < inv_day else 0)
    if payment_month > 12:
        payment_month = 12
        payment_day = 28
    payment_date = f"2024-{payment_month:02d}-{payment_day:02d}"
    early_payment_applies = days_since_invoice <= early_discount_days

    # Quantity tolerance
    qty_tolerance_pct = 0.02  # ±2%

    # --- Select PO line items ---
    n_items = rng.randint(8, 15)
    chosen_items = rng.sample(PO_ITEMS_POOL, n_items)

    po_lines: list[dict] = []
    for i, (code, desc, (lo, hi)) in enumerate(chosen_items):
        qty = rng.randint(10, 200)
        unit_price = round(rng.uniform(lo, hi), 2)
        extended = round(qty * unit_price, 2)
        po_lines.append({
            "line": i + 1,
            "item_code": code,
            "description": desc,
            "qty": qty,
            "unit_price": unit_price,
            "extended": extended,
        })

    po_total = round(sum(line["extended"] for line in po_lines), 2)

    # --- Create delivery and invoice records (start as copies of PO) ---
    delivery_lines = []
    for pl in po_lines:
        delivery_lines.append({
            "line": pl["line"],
            "item_code": pl["item_code"],
            "description": pl["description"],
            "qty_received": pl["qty"],
            "condition": "Good",
        })

    invoice_lines = []
    for pl in po_lines:
        invoice_lines.append({
            "line": pl["line"],
            "item_code": pl["item_code"],
            "description": pl["description"],
            "qty": pl["qty"],
            "unit_price": pl["unit_price"],
            "extended": pl["extended"],
        })

    # --- Plant discrepancies ---
    n_discrepancies = rng.randint(3, 5)

    # Determine which discrepancy types to use
    available_types = list(DISCREPANCY_TYPES)
    rng.shuffle(available_types)
    chosen_disc_types = available_types[:n_discrepancies]

    # Ensure we always have at least one of each major category
    # (but don't exceed n_discrepancies)

    # Pick line indices for discrepancies (for types that modify existing lines)
    modifiable_indices = list(range(n_items))
    rng.shuffle(modifiable_indices)

    discrepancies: list[dict] = []
    used_indices: set[int] = set()

    for disc_type in chosen_disc_types:
        if disc_type == "not_ordered":
            # Add a line to the invoice that wasn't on the PO
            extra_item = rng.choice(NOT_ORDERED_ITEMS)
            extra_code, extra_desc, (elo, ehi) = extra_item
            extra_qty = rng.randint(1, 5)
            extra_price = round(rng.uniform(elo, ehi), 2)
            extra_extended = round(extra_qty * extra_price, 2)
            new_line_num = len(invoice_lines) + 1
            invoice_lines.append({
                "line": new_line_num,
                "item_code": extra_code,
                "description": extra_desc,
                "qty": extra_qty,
                "unit_price": extra_price,
                "extended": extra_extended,
            })
            discrepancies.append({
                "type": "not_ordered",
                "line": new_line_num,
                "item_code": extra_code,
                "description": f"Item {extra_code} ({extra_desc}) appears on invoice but was not on the purchase order. Invoice amount: {_fmt_money(extra_extended)}.",
                "amount_impact": extra_extended,
            })
            continue

        # All other types need a line index
        idx = None
        for candidate in modifiable_indices:
            if candidate not in used_indices:
                idx = candidate
                used_indices.add(candidate)
                break
        if idx is None:
            continue  # No more lines available

        po_line = po_lines[idx]

        if disc_type == "qty_mismatch":
            # Invoice qty differs from delivery qty (invoice bills for more than received)
            invoice_qty = po_line["qty"] + rng.randint(5, 20)
            invoice_lines[idx]["qty"] = invoice_qty
            invoice_lines[idx]["extended"] = round(invoice_qty * po_line["unit_price"], 2)
            discrepancies.append({
                "type": "qty_mismatch",
                "line": po_line["line"],
                "item_code": po_line["item_code"],
                "description": (
                    f"Item {po_line['item_code']} ({po_line['description']}): "
                    f"invoice quantity is {invoice_qty} but only {po_line['qty']} were delivered "
                    f"(and {po_line['qty']} were ordered)."
                ),
                "amount_impact": round((invoice_qty - po_line["qty"]) * po_line["unit_price"], 2),
            })

        elif disc_type == "price_mismatch":
            # Invoice unit price differs from PO price
            price_diff = round(rng.uniform(1.50, 8.00), 2)
            wrong_price = round(po_line["unit_price"] + price_diff, 2)
            invoice_lines[idx]["unit_price"] = wrong_price
            invoice_lines[idx]["extended"] = round(po_line["qty"] * wrong_price, 2)
            discrepancies.append({
                "type": "price_mismatch",
                "line": po_line["line"],
                "item_code": po_line["item_code"],
                "description": (
                    f"Item {po_line['item_code']} ({po_line['description']}): "
                    f"invoiced at {_fmt_money(wrong_price)}/unit instead of the PO price of "
                    f"{_fmt_money(po_line['unit_price'])}/unit."
                ),
                "amount_impact": round(po_line["qty"] * price_diff, 2),
            })

        elif disc_type == "short_delivery":
            # Delivered less than ordered; invoice bills for ordered qty
            short_qty = rng.randint(3, max(4, po_line["qty"] // 4))
            actual_received = po_line["qty"] - short_qty
            delivery_lines[idx]["qty_received"] = actual_received
            delivery_lines[idx]["condition"] = f"Good ({short_qty} short)"
            # Invoice still has full PO qty — that's the discrepancy
            discrepancies.append({
                "type": "short_delivery",
                "line": po_line["line"],
                "item_code": po_line["item_code"],
                "description": (
                    f"Item {po_line['item_code']} ({po_line['description']}): "
                    f"only {actual_received} units received (ordered {po_line['qty']}), "
                    f"but invoice bills for full quantity of {po_line['qty']}."
                ),
                "amount_impact": round(short_qty * po_line["unit_price"], 2),
                "actual_received": actual_received,
            })

        elif disc_type == "double_billed":
            # Same item appears twice on invoice
            dup_line = dict(invoice_lines[idx])
            dup_line["line"] = len(invoice_lines) + 1
            invoice_lines.append(dup_line)
            discrepancies.append({
                "type": "double_billed",
                "line": po_line["line"],
                "item_code": po_line["item_code"],
                "description": (
                    f"Item {po_line['item_code']} ({po_line['description']}) "
                    f"appears twice on the invoice (lines {po_line['line']} and {dup_line['line']}). "
                    f"Duplicate amount: {_fmt_money(dup_line['extended'])}."
                ),
                "amount_impact": dup_line["extended"],
            })

        elif disc_type == "substitution":
            # Different item code delivered than what was ordered
            # Pick an item code NOT in the PO
            all_po_codes = {pl["item_code"] for pl in po_lines}
            sub_candidates = [it for it in PO_ITEMS_POOL if it[0] not in all_po_codes]
            if sub_candidates:
                sub_item = rng.choice(sub_candidates)
                sub_code, sub_desc, _ = sub_item
                delivery_lines[idx]["item_code"] = sub_code
                delivery_lines[idx]["description"] = sub_desc
                invoice_lines[idx]["item_code"] = sub_code
                invoice_lines[idx]["description"] = sub_desc
                discrepancies.append({
                    "type": "substitution",
                    "line": po_line["line"],
                    "item_code": po_line["item_code"],
                    "description": (
                        f"Line {po_line['line']}: ordered {po_line['item_code']} ({po_line['description']}) "
                        f"but received and invoiced {sub_code} ({sub_desc}) instead — unauthorized substitution."
                    ),
                    "amount_impact": 0.0,  # Price may be same; discrepancy is the item itself
                })

    # --- Compute correct payment ---
    # Correct amount per line: min(qty_received, po_qty) x po_unit_price
    correct_line_amounts: list[float] = []
    for i, pl in enumerate(po_lines):
        qty_received = delivery_lines[i]["qty_received"]
        correct_qty = min(qty_received, pl["qty"])
        correct_line_amounts.append(round(correct_qty * pl["unit_price"], 2))

    correct_subtotal = round(sum(correct_line_amounts), 2)

    # Apply late delivery penalty
    late_penalty_amount = round(correct_subtotal * late_penalty_pct, 2)
    after_penalty = round(correct_subtotal - late_penalty_amount, 2)

    # Invoice total (what vendor is billing)
    invoice_total = round(sum(il["extended"] for il in invoice_lines), 2)

    # Early payment discount
    early_discount_amount = round(after_penalty * early_discount_pct, 2) if early_payment_applies else 0.0
    recommended_payment = round(after_penalty - early_discount_amount, 2)

    # Pick 2-3 clean lines for false-positive checks
    clean_indices = [i for i in range(n_items) if i not in used_indices]
    n_false_pos_checks = min(3, len(clean_indices))
    false_pos_indices = rng.sample(clean_indices, n_false_pos_checks) if clean_indices else []

    # --- Build purchase_order.txt ---
    po_header_lines = [
        "PURCHASE ORDER",
        "",
        f"PO Number: {po_number}",
        f"Date: {po_date}",
        f"Required By: {required_by_date}",
        "",
        f"Buyer: {buyer_company}",
        f"Contact: {buyer_contact}",
        f"Ship To: {buyer_company} Warehouse, Dock 4",
        "",
        f"Vendor: {vendor_name}",
        "",
        "=" * 80,
        "ORDER LINES",
        "=" * 80,
        "",
        f"{'Line':<6} {'Item Code':<12} {'Description':<30} {'Qty':>6} {'Unit Price':>12} {'Extended':>12}",
        f"{'-'*6} {'-'*12} {'-'*30} {'-'*6} {'-'*12} {'-'*12}",
    ]
    for pl in po_lines:
        po_header_lines.append(
            f"{pl['line']:<6} {pl['item_code']:<12} {pl['description']:<30} "
            f"{pl['qty']:>6} {_fmt_money(pl['unit_price']):>12} {_fmt_money(pl['extended']):>12}"
        )
    po_header_lines.append("")
    po_header_lines.append(f"{'':>56} {'TOTAL':>12} {_fmt_money(po_total):>12}")
    po_header_lines.append("")
    po_header_lines.append(f"Payment Terms: See contract terms document")
    po_header_lines.append("")
    po_content = "\n".join(po_header_lines) + "\n"

    # --- Build delivery_receipt.txt ---
    del_lines_out = [
        "DELIVERY RECEIPT / GOODS RECEIVED NOTE",
        "",
        f"PO Reference: {po_number}",
        f"Delivery Date: {delivery_date}",
        f"Received By: Warehouse Team",
        "",
        f"Vendor: {vendor_name}",
        "",
        "=" * 80,
        "RECEIVED ITEMS",
        "=" * 80,
        "",
        f"{'Line':<6} {'Item Code':<12} {'Description':<30} {'Qty Received':>13} {'Condition':<20}",
        f"{'-'*6} {'-'*12} {'-'*30} {'-'*13} {'-'*20}",
    ]
    for dl in delivery_lines:
        del_lines_out.append(
            f"{dl['line']:<6} {dl['item_code']:<12} {dl['description']:<30} "
            f"{dl['qty_received']:>13} {dl['condition']:<20}"
        )
    del_lines_out.append("")
    del_lines_out.append(f"PO Required-By Date: {required_by_date}")
    del_lines_out.append("")
    del_lines_out.append(f"Warehouse Signature: ___________________")
    del_lines_out.append("")
    delivery_content = "\n".join(del_lines_out) + "\n"

    # --- Build invoice.txt ---
    inv_lines_out = [
        "VENDOR INVOICE",
        "",
        f"Invoice Number: {invoice_number}",
        f"Invoice Date: {invoice_date}",
        f"PO Reference: {po_number}",
        "",
        f"From: {vendor_name}",
        f"To: {buyer_company}",
        f"Attn: {buyer_contact}",
        "",
        "=" * 80,
        "INVOICE LINES",
        "=" * 80,
        "",
        f"{'Line':<6} {'Item Code':<12} {'Description':<30} {'Qty':>6} {'Unit Price':>12} {'Extended':>12}",
        f"{'-'*6} {'-'*12} {'-'*30} {'-'*6} {'-'*12} {'-'*12}",
    ]
    for il in invoice_lines:
        inv_lines_out.append(
            f"{il['line']:<6} {il['item_code']:<12} {il['description']:<30} "
            f"{il['qty']:>6} {_fmt_money(il['unit_price']):>12} {_fmt_money(il['extended']):>12}"
        )
    inv_lines_out.append("")
    inv_lines_out.append(f"{'':>56} {'TOTAL':>12} {_fmt_money(invoice_total):>12}")
    inv_lines_out.append("")
    inv_lines_out.append(f"Payment Terms: See contract terms document")
    inv_lines_out.append(f"Please remit payment to: {vendor_name}")
    inv_lines_out.append("")
    invoice_content = "\n".join(inv_lines_out) + "\n"

    # --- Build contract_terms.txt ---
    terms_lines = [
        "CONTRACT TERMS — VENDOR PAYMENT POLICY",
        "",
        f"Vendor: {vendor_name}",
        f"Buyer: {buyer_company}",
        "",
        "=" * 60,
        "PAYMENT TERMS",
        "=" * 60,
        "",
        f"Standard Payment Window: Net {payment_net_days} days from invoice date",
        "",
        f"Early Payment Discount: {early_discount_pct * 100:.0f}% discount if paid within "
        f"{early_discount_days} days of invoice date ({int(early_discount_pct * 100)}/{early_discount_days} Net {payment_net_days})",
        "",
        "=" * 60,
        "DELIVERY TERMS",
        "=" * 60,
        "",
        f"Late Delivery Penalty: 1% of order value per week late, maximum 5%",
        f"  - Penalty is calculated on the correct payable amount (after adjustments)",
        f"  - Weeks are counted from the required-by date to the actual delivery date",
        f"  - Partial weeks of 3 or more days count as a full week",
        "",
        "=" * 60,
        "QUANTITY AND PRICING POLICY",
        "=" * 60,
        "",
        f"Quantity Tolerance: +/- {qty_tolerance_pct * 100:.0f}% acceptable without price adjustment",
        f"Short Delivery Policy: Buyer pays only for quantity actually received, at the PO unit price",
        f"Price Discrepancies: PO price governs; invoice price differences must be resolved before payment",
        f"Unauthorized Items: Items not on the original PO are not payable without a separate PO or amendment",
        f"Substitutions: Unauthorized substitutions must be flagged; payment held pending resolution",
        "",
        "=" * 60,
        "THREE-WAY MATCHING REQUIREMENTS",
        "=" * 60,
        "",
        "Before payment, accounts payable must verify:",
        "  1. Invoice quantities match delivery receipt quantities",
        "  2. Invoice unit prices match PO unit prices",
        "  3. All invoiced items appear on the original PO",
        "  4. No duplicate billing for the same line item",
        "  5. Delivery was on time (apply penalty if late)",
        "",
        "Discrepancies must be documented and the correct payable amount computed as:",
        "  Correct Amount = sum of [min(qty_received, po_qty) x po_unit_price] for each line",
        "  Then apply late delivery penalty if applicable",
        "  Then apply early payment discount if applicable",
        "",
    ]
    terms_content = "\n".join(terms_lines) + "\n"

    # --- Problem statement ---
    problem_statement = f"""# Vendor Invoice Validation (Three-Way Matching)

You are an accounts payable analyst at {buyer_company}. You must perform a
three-way match between the Purchase Order, Delivery Receipt, and Vendor Invoice
to identify discrepancies and compute the correct payment amount.

**Today's date (payment processing date): {payment_date}**

## Source Files
- /testbed/data/purchase_order.txt — Original purchase order with line items
- /testbed/data/delivery_receipt.txt — Warehouse receiving record of what was actually delivered
- /testbed/data/invoice.txt — Vendor's invoice requesting payment
- /testbed/data/contract_terms.txt — Payment terms, penalties, and matching requirements

## Requirements
1. Match each line across all three documents (PO, delivery receipt, invoice)
2. Identify ALL discrepancies: quantity mismatches, price differences, items not ordered, short deliveries, duplicate billing, substitutions
3. Compute the correct payable amount for each line: min(qty_received, po_qty) x po_unit_price
4. Sum for the correct subtotal
5. Check if late delivery penalty applies and compute it
6. Check if early payment discount applies and compute it
7. State the recommended payment amount

Write a detailed validation report to /testbed/validation_report.txt showing
your three-way matching analysis and payment recommendation."""

    # --- Build rubric ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does the file /testbed/validation_report.txt exist and contain substantive content (more than a few lines)?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_po_total",
            question=f"Does the report correctly state the PO grand total as {_fmt_money(po_total)} (within $1)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_invoice_total",
            question=f"Does the report correctly state the invoice grand total as {_fmt_money(invoice_total)} (within $1)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_discrepancy_count",
            question=f"Does the report identify exactly {len(discrepancies)} discrepancies (not counting clean matches)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_po_line_count",
            question=f"Does the report correctly state or demonstrate that the PO contains {n_items} line items?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_delivery_status",
            question=(
                f"Does the report correctly identify the delivery as "
                f"{'LATE (delivered after the required-by date of ' + required_by_date + ')' if is_late_delivery else 'on time (delivered by the required-by date)'}?"
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_subtotal_before_adjustments",
            question=f"Does the report compute the correct payable subtotal (before penalties and discounts) as approximately {_fmt_money(correct_subtotal)} (within $50)?",
            points=2,
        ),
    ]

    # Per-discrepancy checks
    for i, disc in enumerate(discrepancies):
        rubric_items.append(
            BinaryRubricCategory(
                name=f"found_discrepancy_{i+1}",
                question=f"Does the report identify the following discrepancy: {disc['description']}",
                points=2,
            )
        )

    # False positive checks for clean items
    for fp_idx in false_pos_indices:
        clean_line = po_lines[fp_idx]
        rubric_items.append(
            BinaryRubricCategory(
                name=f"no_false_discrepancy_{clean_line['item_code']}",
                question=(
                    f"Does the report correctly show item {clean_line['item_code']} ({clean_line['description']}) "
                    f"as a clean match with no discrepancies? (It should NOT be flagged as problematic.)"
                ),
                points=1,
            )
        )

    # Check for not-ordered item identification
    not_ordered_discs = [d for d in discrepancies if d["type"] == "not_ordered"]
    if not_ordered_discs:
        rubric_items.append(
            BinaryRubricCategory(
                name="identifies_not_ordered_item",
                question=(
                    f"Does the report specifically flag that item(s) not on the original PO were included on "
                    f"the invoice and should not be paid? "
                    f"({', '.join(d['item_code'] for d in not_ordered_discs)})"
                ),
                points=2,
            )
        )

    rubric_items.append(
        BinaryRubricCategory(
            name="correct_recommended_payment",
            question=f"Does the report recommend a payment of approximately {_fmt_money(recommended_payment)} (within $50)?",
            points=3,
        )
    )

    if early_payment_applies:
        rubric_items.append(
            BinaryRubricCategory(
                name="correct_early_payment_discount",
                question=(
                    f"Does the report correctly apply the {early_discount_pct*100:.0f}% early payment discount "
                    f"of approximately {_fmt_money(early_discount_amount)} (within $20)?"
                ),
                points=2,
            )
        )
    else:
        rubric_items.append(
            BinaryRubricCategory(
                name="correct_early_payment_discount",
                question=(
                    f"Does the report correctly note that the early payment discount does NOT apply "
                    f"(payment is being made more than {early_discount_days} days after the invoice date)?"
                ),
                points=2,
            )
        )

    if is_late_delivery:
        rubric_items.append(
            BinaryRubricCategory(
                name="correct_late_penalty",
                question=(
                    f"Does the report correctly apply the late delivery penalty of {late_penalty_pct*100:.1f}% "
                    f"({weeks_late} week(s) late), which is approximately {_fmt_money(late_penalty_amount)} (within $20)?"
                ),
                points=2,
            )
        )
    else:
        rubric_items.append(
            BinaryRubricCategory(
                name="correct_late_penalty",
                question="Does the report correctly note that delivery was on time and no late penalty applies?",
                points=2,
            )
        )

    rubric_items.append(
        RubricCategory(
            name="analysis_thoroughness",
            description="How thorough and systematic is the three-way matching analysis?",
            failure="Superficial analysis; most discrepancies missed or poorly explained.",
            minor_failure="Some discrepancies found but analysis is incomplete or lacks detail.",
            minor_success="Most discrepancies found with reasonable explanations and correct line-by-line matching.",
            success="All discrepancies identified with clear explanations, systematic line-by-line comparison, and a well-justified payment recommendation.",
            points=2,
        )
    )

    rubric = tuple(rubric_items)

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write your completed validation report to /testbed/validation_report.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/purchase_order.txt": po_content,
            "/testbed/data/delivery_receipt.txt": delivery_content,
            "/testbed/data/invoice.txt": invoice_content,
            "/testbed/data/contract_terms.txt": terms_content,
        },
        problem_type="vendor_invoice_validation",
    )
