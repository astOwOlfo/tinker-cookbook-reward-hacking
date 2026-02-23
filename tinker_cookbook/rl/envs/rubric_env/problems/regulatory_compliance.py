"""Regulatory compliance tasks: environmental impact, import classification, workplace safety.

Each factory generates a realistic scenario with necessary_files containing all
information required to solve the problem. Seeds control randomization of
thresholds, exceedances, tariff rates, violation patterns, etc.
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
# 1. ENVIRONMENTAL IMPACT ASSESSMENT
# =============================================================================

PROJECT_TYPES = [
    {
        "name": "Regional Distribution Warehouse",
        "description": "A 250,000 sq ft distribution center with truck staging areas, employee parking, and stormwater retention ponds.",
        "acreage_range": (25, 45),
        "footprint_pct_range": (0.30, 0.50),
        "water_usage_gpd_range": (15000, 35000),
        "emissions_modifier": 0.8,
    },
    {
        "name": "Utility-Scale Solar Farm",
        "description": "A ground-mounted photovoltaic solar installation with inverter stations, access roads, and perimeter fencing.",
        "acreage_range": (80, 200),
        "footprint_pct_range": (0.60, 0.80),
        "water_usage_gpd_range": (2000, 8000),
        "emissions_modifier": 0.3,
    },
    {
        "name": "Light Manufacturing Plant",
        "description": "A metal fabrication facility with machining lines, paint booths, chemical storage, and wastewater treatment.",
        "acreage_range": (10, 25),
        "footprint_pct_range": (0.35, 0.55),
        "water_usage_gpd_range": (40000, 80000),
        "emissions_modifier": 1.5,
    },
    {
        "name": "Mixed-Use Commercial Complex",
        "description": "A multi-building development with retail, office space, structured parking, and landscaped common areas.",
        "acreage_range": (8, 18),
        "footprint_pct_range": (0.40, 0.65),
        "water_usage_gpd_range": (25000, 50000),
        "emissions_modifier": 0.6,
    },
    {
        "name": "Chemical Processing Facility",
        "description": "A specialty chemicals batch processing plant with tank farms, containment berms, and emission stacks.",
        "acreage_range": (15, 35),
        "footprint_pct_range": (0.25, 0.40),
        "water_usage_gpd_range": (60000, 120000),
        "emissions_modifier": 2.0,
    },
    {
        "name": "Cold Storage and Food Processing Center",
        "description": "A refrigerated warehouse with food processing lines, ammonia refrigeration systems, and shipping docks.",
        "acreage_range": (12, 28),
        "footprint_pct_range": (0.35, 0.50),
        "water_usage_gpd_range": (30000, 70000),
        "emissions_modifier": 1.0,
    },
    {
        "name": "Data Center Campus",
        "description": "A multi-building server farm with backup generators, cooling towers, fuel storage, and redundant power infrastructure.",
        "acreage_range": (15, 40),
        "footprint_pct_range": (0.20, 0.35),
        "water_usage_gpd_range": (80000, 200000),
        "emissions_modifier": 0.9,
    },
    {
        "name": "Concrete Batch Plant",
        "description": "A ready-mix concrete production facility with aggregate storage, cement silos, truck wash stations, and dust collection systems.",
        "acreage_range": (5, 15),
        "footprint_pct_range": (0.45, 0.65),
        "water_usage_gpd_range": (20000, 50000),
        "emissions_modifier": 1.8,
    },
]

SENSITIVE_AREAS = [
    {"name": "freshwater wetland", "buffer_ft": 100, "seasonal_restriction": "March 1 through June 30 (amphibian breeding season)"},
    {"name": "tidal estuary", "buffer_ft": 150, "seasonal_restriction": "April 15 through August 31 (migratory bird nesting)"},
    {"name": "endangered species habitat (Indiana bat)", "buffer_ft": 200, "seasonal_restriction": "April 1 through September 30 (bat roosting season)"},
    {"name": "critical aquifer recharge zone", "buffer_ft": 75, "seasonal_restriction": None},
    {"name": "residential zone (within 500 ft)", "buffer_ft": 0, "seasonal_restriction": None},
    {"name": "state-designated Natural Heritage Area", "buffer_ft": 250, "seasonal_restriction": "Year-round activity restrictions within buffer"},
]

ENDANGERED_SPECIES_POOL = [
    {"name": "Indiana bat (Myotis sodalis)", "status": "Endangered", "buffer_ft": 200, "active_season": "April - September"},
    {"name": "Northern long-eared bat (Myotis septentrionalis)", "status": "Threatened", "buffer_ft": 150, "active_season": "April - October"},
    {"name": "Bog turtle (Glyptemys muhlenbergii)", "status": "Threatened", "buffer_ft": 300, "active_season": "April - September"},
    {"name": "Red-cockaded woodpecker (Picoides borealis)", "status": "Endangered", "buffer_ft": 250, "active_season": "Year-round"},
    {"name": "American burying beetle (Nicrophorus americanus)", "status": "Threatened", "buffer_ft": 100, "active_season": "June - September"},
]

# Environmental parameters with base thresholds by zone type
# (parameter_name, unit, residential_threshold, industrial_threshold, sensitive_area_threshold)
ENV_PARAMETERS = [
    ("PM2.5", "ug/m3", 12.0, 35.0, 10.0),
    ("PM10", "ug/m3", 50.0, 150.0, 40.0),
    ("NOx", "ppb", 53.0, 100.0, 40.0),
    ("SOx", "ppb", 75.0, 196.0, 50.0),
    ("CO", "ppm", 9.0, 35.0, 9.0),
    ("Ozone", "ppm", 0.070, 0.100, 0.060),
    ("Water pH", "pH", (6.5, 8.5), (6.0, 9.0), (6.5, 8.5)),
    ("Dissolved Oxygen", "mg/L", 5.0, 4.0, 6.0),
    ("Turbidity", "NTU", 5.0, 25.0, 4.0),
    ("Lead (water)", "ug/L", 15.0, 50.0, 10.0),
    ("Mercury (water)", "ug/L", 2.0, 5.0, 1.0),
    ("Noise at 50ft", "dBA", 65.0, 85.0, 55.0),
    ("Noise at 200ft", "dBA", 55.0, 75.0, 45.0),
    ("Noise at 500ft", "dBA", 45.0, 65.0, 35.0),
]


def make_environmental_impact_assessment(rand_seed: int = 42) -> RubricDatapoint:
    """Review a construction project's environmental data for regulatory
    compliance. Cross-reference emissions, water quality, wildlife surveys,
    and regulatory thresholds.

    Seed varies: project type, location sensitivity, which thresholds are
    exceeded (2-4 of 10+), endangered species presence, seasonal restrictions.
    """
    rng = _random.Random(rand_seed)

    applicant_company = pick1(COMPANY_NAMES, rand_seed)
    project_manager = random_name(rand_seed)
    reviewer_name = random_name(rand_seed + 1)

    # --- Project selection ---
    project = rng.choice(PROJECT_TYPES)
    sensitive_area = rng.choice(SENSITIVE_AREAS)
    acreage = round(rng.uniform(*project["acreage_range"]), 1)
    footprint_pct = round(rng.uniform(*project["footprint_pct_range"]), 2)
    building_footprint_acres = round(acreage * footprint_pct, 1)
    water_usage = rng.randint(*project["water_usage_gpd_range"])

    # --- Zone type for threshold selection ---
    # sensitive_area determines which threshold column to use
    zone_types = ["residential", "industrial", "sensitive"]
    zone_idx_map = {"residential": 2, "industrial": 3, "sensitive": 4}
    if "residential" in sensitive_area["name"]:
        zone_type = "residential"
    elif "habitat" in sensitive_area["name"] or "wetland" in sensitive_area["name"] or "estuary" in sensitive_area["name"] or "Heritage" in sensitive_area["name"]:
        zone_type = "sensitive"
    else:
        zone_type = "industrial"
    threshold_col = zone_idx_map[zone_type]

    # --- Decide which parameters exceed thresholds ---
    # We want 2-4 exceedances out of the 14 parameters
    n_exceedances = rng.randint(2, 4)
    # Exclude pH (index 6) and DO (index 7) from simple exceedance logic initially
    simple_param_indices = [i for i in range(len(ENV_PARAMETERS)) if i not in (6, 7)]
    exceedance_indices = set(rng.sample(simple_param_indices, n_exceedances))

    # --- Generate measured values ---
    measurements: list[dict] = []
    exceedance_details: list[dict] = []
    compliant_params: list[dict] = []

    for i, param_tuple in enumerate(ENV_PARAMETERS):
        param_name = param_tuple[0]
        unit = param_tuple[1]
        threshold_val = param_tuple[threshold_col]

        # pH is a range check
        if param_name == "Water pH":
            ph_low, ph_high = threshold_val  # type: ignore
            if i in exceedance_indices:
                # Make pH exceed range
                if rng.random() < 0.5:
                    measured = round(ph_low - rng.uniform(0.3, 1.2), 2)
                    exceedance_details.append({
                        "param": param_name,
                        "measured": measured,
                        "threshold": f"{ph_low}-{ph_high}",
                        "unit": unit,
                        "direction": "below",
                        "margin": round(ph_low - measured, 2),
                    })
                else:
                    measured = round(ph_high + rng.uniform(0.3, 1.2), 2)
                    exceedance_details.append({
                        "param": param_name,
                        "measured": measured,
                        "threshold": f"{ph_low}-{ph_high}",
                        "unit": unit,
                        "direction": "above",
                        "margin": round(measured - ph_high, 2),
                    })
            else:
                measured = round(rng.uniform(ph_low + 0.2, ph_high - 0.2), 2)
                compliant_params.append({
                    "param": param_name,
                    "measured": measured,
                    "threshold": f"{ph_low}-{ph_high}",
                    "unit": unit,
                })
            measurements.append({
                "param": param_name,
                "measured": measured,
                "unit": unit,
                "threshold": f"{ph_low}-{ph_high}",
                "is_range": True,
            })
            continue

        # Dissolved Oxygen: lower is worse (threshold is a minimum)
        if param_name == "Dissolved Oxygen":
            threshold_numeric = threshold_val  # type: ignore
            if i in exceedance_indices:
                measured = round(threshold_numeric - rng.uniform(0.5, 2.0), 2)
                exceedance_details.append({
                    "param": param_name,
                    "measured": measured,
                    "threshold": threshold_numeric,
                    "unit": unit,
                    "direction": "below minimum",
                    "margin": round(threshold_numeric - measured, 2),
                })
            else:
                measured = round(threshold_numeric + rng.uniform(0.5, 3.0), 2)
                compliant_params.append({
                    "param": param_name,
                    "measured": measured,
                    "threshold": threshold_numeric,
                    "unit": unit,
                })
            measurements.append({
                "param": param_name,
                "measured": measured,
                "unit": unit,
                "threshold": threshold_numeric,
                "is_range": False,
            })
            continue

        # All other parameters: higher is worse (threshold is a maximum)
        threshold_numeric = threshold_val  # type: ignore
        if i in exceedance_indices:
            # Exceed by 5%-40%
            overshoot_pct = rng.uniform(0.05, 0.40)
            measured = round(threshold_numeric * (1 + overshoot_pct), 3)
            margin = round(measured - threshold_numeric, 3)
            exceedance_details.append({
                "param": param_name,
                "measured": measured,
                "threshold": threshold_numeric,
                "unit": unit,
                "direction": "above",
                "margin": margin,
            })
        else:
            # Compliant: 50%-95% of threshold
            measured = round(threshold_numeric * rng.uniform(0.50, 0.95), 3)
            compliant_params.append({
                "param": param_name,
                "measured": measured,
                "threshold": threshold_numeric,
                "unit": unit,
            })
        measurements.append({
            "param": param_name,
            "measured": measured,
            "unit": unit,
            "threshold": threshold_numeric,
            "is_range": False,
        })

    # --- Wildlife survey ---
    has_endangered = rng.random() < 0.6
    n_common_species = rng.randint(8, 15)
    common_species = rng.sample([
        "White-tailed deer", "Eastern cottontail", "American robin",
        "Red-tailed hawk", "Gray squirrel", "Eastern chipmunk",
        "Northern cardinal", "Blue jay", "Mourning dove",
        "Green frog", "American toad", "Painted turtle",
        "Monarch butterfly", "Great blue heron", "Canada goose",
        "Mallard duck", "Red fox", "Raccoon", "Opossum",
        "Eastern box turtle", "Spring peeper", "Bullfrog",
    ], n_common_species)

    endangered_species = None
    if has_endangered:
        endangered_species = rng.choice(ENDANGERED_SPECIES_POOL)

    buffer_zone_required = sensitive_area["buffer_ft"]
    if has_endangered and endangered_species is not None:
        buffer_zone_required = max(buffer_zone_required, endangered_species["buffer_ft"])

    seasonal_restriction = sensitive_area["seasonal_restriction"]
    if has_endangered and endangered_species is not None and seasonal_restriction is None:
        seasonal_restriction = f"{endangered_species['active_season']} ({endangered_species['name'].split('(')[0].strip()} activity period)"

    # --- Determine overall compliance ---
    n_total_exceedances = len(exceedance_details)
    overall_compliant = (n_total_exceedances == 0 and not has_endangered)
    # With exceedances, project needs mitigation or permit modification

    # --- Build project_description.txt ---
    project_desc_lines = [
        "ENVIRONMENTAL IMPACT ASSESSMENT — PROJECT DESCRIPTION",
        "",
        f"Applicant: {applicant_company}",
        f"Project Manager: {project_manager}",
        f"Project Type: {project['name']}",
        "",
        "=" * 60,
        "PROJECT SPECIFICATIONS",
        "=" * 60,
        "",
        f"Description: {project['description']}",
        "",
        f"Total Project Acreage: {acreage} acres",
        f"Building/Infrastructure Footprint: {building_footprint_acres} acres ({footprint_pct*100:.0f}% of site)",
        f"Estimated Daily Water Usage: {water_usage:,} gallons/day",
        f"Impervious Surface Coverage: {footprint_pct*100 + rng.uniform(5,15):.0f}%",
        "",
        "=" * 60,
        "SITE LOCATION AND SENSITIVITY",
        "=" * 60,
        "",
        f"Adjacent Sensitive Area: {sensitive_area['name']}",
        f"Zone Classification: {zone_type.title()}",
        f"Required Buffer Zone (from sensitive area boundary): {sensitive_area['buffer_ft']} ft",
        "",
        "Construction is proposed to begin in February 2025.",
        "",
    ]
    project_desc_content = "\n".join(project_desc_lines) + "\n"

    # --- Build environmental_data.csv ---
    csv_header = "Parameter,Measured Value,Unit,Sampling Date,Location"
    csv_rows = [csv_header]
    sampling_locations = ["Site Center", "NE Boundary", "SW Boundary", "Nearest Water Body", "Nearest Residence"]
    for m in measurements:
        loc = rng.choice(sampling_locations)
        month = rng.randint(6, 10)
        day = rng.randint(1, 28)
        csv_rows.append(f"{m['param']},{m['measured']},{m['unit']},2024-{month:02d}-{day:02d},{loc}")
    env_data_content = "\n".join(csv_rows) + "\n"

    # --- Build regulatory_thresholds.txt ---
    threshold_lines = [
        "REGULATORY THRESHOLDS — ENVIRONMENTAL PARAMETERS",
        "",
        "Source: EPA National Ambient Air Quality Standards (NAAQS),",
        "State Water Quality Standards, and Local Noise Ordinances",
        "",
        f"Applicable Zone Type for This Project: {zone_type.title()}",
        "",
        "=" * 80,
        "",
        f"{'Parameter':<25} {'Unit':<10} {'Residential':<15} {'Industrial':<15} {'Sensitive Area':<15}",
        f"{'-'*25} {'-'*10} {'-'*15} {'-'*15} {'-'*15}",
    ]
    for param_tuple in ENV_PARAMETERS:
        pname = param_tuple[0]
        unit = param_tuple[1]
        res_val = param_tuple[2]
        ind_val = param_tuple[3]
        sens_val = param_tuple[4]
        if pname == "Water pH":
            res_str = f"{res_val[0]}-{res_val[1]}"  # type: ignore
            ind_str = f"{ind_val[0]}-{ind_val[1]}"  # type: ignore
            sens_str = f"{sens_val[0]}-{sens_val[1]}"  # type: ignore
        elif pname == "Dissolved Oxygen":
            res_str = f">= {res_val}"
            ind_str = f">= {ind_val}"
            sens_str = f">= {sens_val}"
        else:
            res_str = f"<= {res_val}"
            ind_str = f"<= {ind_val}"
            sens_str = f"<= {sens_val}"
        threshold_lines.append(f"{pname:<25} {unit:<10} {res_str:<15} {ind_str:<15} {sens_str:<15}")

    threshold_lines.extend([
        "",
        "=" * 80,
        "NOTES:",
        f"  - For this project, use the '{zone_type.title()}' column thresholds.",
        "  - Water pH is a range; values outside the range are exceedances.",
        "  - Dissolved Oxygen is a minimum; values below the threshold are exceedances.",
        "  - All other parameters are maximums; values above the threshold are exceedances.",
        "  - Exceedances require mitigation plans and may require permit modifications.",
        "",
    ])
    threshold_content = "\n".join(threshold_lines) + "\n"

    # --- Build wildlife_survey.txt ---
    wildlife_lines = [
        "WILDLIFE SURVEY REPORT",
        "",
        f"Survey Conducted: 2024-{rng.randint(5,9):02d}-{rng.randint(1,28):02d}",
        f"Surveyor: {random_name(rand_seed + 10)}, Certified Wildlife Biologist",
        f"Survey Area: {acreage} acres plus 500-ft buffer zone",
        "",
        "=" * 60,
        "COMMON SPECIES OBSERVED",
        "=" * 60,
        "",
    ]
    for sp in common_species:
        count = rng.randint(1, 25)
        wildlife_lines.append(f"  - {sp}: {count} individuals observed")

    wildlife_lines.extend([
        "",
        "=" * 60,
        "THREATENED AND ENDANGERED SPECIES SURVEY",
        "=" * 60,
        "",
    ])

    if has_endangered and endangered_species is not None:
        wildlife_lines.extend([
            f"SPECIES DETECTED: {endangered_species['name']}",
            f"  Federal Status: {endangered_species['status']}",
            f"  Individuals Observed: {rng.randint(1, 5)}",
            f"  Habitat Type: Suitable {endangered_species['name'].split('(')[0].strip().lower()} habitat present on-site",
            f"  Active Season: {endangered_species['active_season']}",
            f"  Recommended Buffer: {endangered_species['buffer_ft']} ft from identified habitat features",
            "",
        ])
    else:
        wildlife_lines.extend([
            "No threatened or endangered species were detected during the survey period.",
            "Standard habitat assessments indicate low probability of occurrence for",
            "state and federally listed species based on habitat characteristics.",
            "",
        ])

    wildlife_lines.extend([
        "=" * 60,
        "BUFFER ZONE AND SEASONAL RESTRICTION SUMMARY",
        "=" * 60,
        "",
        f"Sensitive Area Adjacent: {sensitive_area['name']}",
        f"Base Buffer Requirement: {sensitive_area['buffer_ft']} ft",
    ])
    if has_endangered and endangered_species is not None:
        wildlife_lines.append(f"Endangered Species Buffer: {endangered_species['buffer_ft']} ft")

    if seasonal_restriction:
        wildlife_lines.append(f"Seasonal Activity Restriction: {seasonal_restriction}")
    else:
        wildlife_lines.append("Seasonal Activity Restriction: None identified")
    wildlife_lines.append("")

    wildlife_content = "\n".join(wildlife_lines) + "\n"

    # --- Problem statement ---
    problem_statement = f"""# Environmental Impact Assessment — Regulatory Compliance Review

You are {reviewer_name}, an environmental compliance officer. You must review a
proposed construction project's environmental data and determine whether it meets
all applicable regulatory thresholds.

## Source Files
- /testbed/data/project_description.txt — Project type, specifications, site location, and sensitive area
- /testbed/data/environmental_data.csv — Measured baseline values for air, water, noise parameters
- /testbed/data/regulatory_thresholds.txt — Regulatory thresholds by zone type (use the applicable column)
- /testbed/data/wildlife_survey.txt — Species survey results, endangered species findings, buffer zones

## Requirements
1. Identify the applicable zone type and use the correct threshold column
2. Compare EACH measured parameter against its regulatory threshold
3. Identify ALL parameters that exceed their thresholds, noting the measured value, threshold, and margin
4. Identify which parameters are compliant
5. Assess wildlife/endangered species concerns and required buffer zones
6. Determine seasonal activity restrictions if applicable
7. Provide an overall compliance determination with specific findings

Write your compliance review report to /testbed/compliance_report.txt"""

    # --- Build rubric ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does the file /testbed/compliance_report.txt exist and contain substantive content (more than a few lines)?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_zone_type",
            question=f'Does the report correctly identify the applicable zone type as "{zone_type.title()}" and use the corresponding threshold column?',
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_project_type",
            question=f'Does the report correctly identify the project as a "{project["name"]}"?',
            points=1,
        ),
    ]

    # Per-exceedance checks (2 pts each for identifying, 1 pt for margin)
    for j, exc in enumerate(exceedance_details):
        if exc["param"] == "Water pH":
            rubric_items.append(
                BinaryRubricCategory(
                    name=f"identifies_exceedance_{j+1}_{exc['param'].lower().replace(' ', '_').replace('(', '').replace(')', '')}",
                    question=(
                        f"Does the report identify that {exc['param']} is out of compliance? "
                        f"Measured value: {exc['measured']} {exc['unit']}, acceptable range: {exc['threshold']} {exc['unit']}. "
                        f"The measured value is {exc['direction']} the acceptable range by {exc['margin']} {exc['unit']}."
                    ),
                    points=2,
                )
            )
        elif exc["param"] == "Dissolved Oxygen":
            rubric_items.append(
                BinaryRubricCategory(
                    name=f"identifies_exceedance_{j+1}_{exc['param'].lower().replace(' ', '_').replace('(', '').replace(')', '')}",
                    question=(
                        f"Does the report identify that {exc['param']} is out of compliance? "
                        f"Measured value: {exc['measured']} {exc['unit']}, minimum threshold: {exc['threshold']} {exc['unit']}. "
                        f"The measured value is {exc['margin']} {exc['unit']} below the minimum."
                    ),
                    points=2,
                )
            )
        else:
            rubric_items.append(
                BinaryRubricCategory(
                    name=f"identifies_exceedance_{j+1}_{exc['param'].lower().replace(' ', '_').replace('(', '').replace(')', '')}",
                    question=(
                        f"Does the report identify that {exc['param']} exceeds its threshold? "
                        f"Measured value: {exc['measured']} {exc['unit']}, threshold: {exc['threshold']} {exc['unit']}. "
                        f"The exceedance margin is {exc['margin']} {exc['unit']}."
                    ),
                    points=2,
                )
            )
        rubric_items.append(
            BinaryRubricCategory(
                name=f"correct_margin_{j+1}_{exc['param'].lower().replace(' ', '_').replace('(', '').replace(')', '')}",
                question=(
                    f"Does the report state the exceedance margin for {exc['param']} as approximately "
                    f"{exc['margin']} {exc['unit']} (within 10% of {exc['margin']})?"
                ),
                points=1,
            )
        )

    # False-positive checks: pick 2-3 compliant params, verify no false exceedance
    n_false_checks = min(3, len(compliant_params))
    false_check_params = rng.sample(compliant_params, n_false_checks)
    for cp in false_check_params:
        if cp["param"] == "Dissolved Oxygen":
            rubric_items.append(
                BinaryRubricCategory(
                    name=f"no_false_exceedance_{cp['param'].lower().replace(' ', '_').replace('(', '').replace(')', '')}",
                    question=(
                        f"Does the report correctly show {cp['param']} as compliant? "
                        f"Measured: {cp['measured']} {cp['unit']}, minimum threshold: {cp['threshold']} {cp['unit']}. "
                        f"This parameter is WITHIN compliance and should NOT be flagged as an exceedance."
                    ),
                    points=2,
                )
            )
        elif isinstance(cp["threshold"], str):
            rubric_items.append(
                BinaryRubricCategory(
                    name=f"no_false_exceedance_{cp['param'].lower().replace(' ', '_').replace('(', '').replace(')', '')}",
                    question=(
                        f"Does the report correctly show {cp['param']} as compliant? "
                        f"Measured: {cp['measured']} {cp['unit']}, acceptable range: {cp['threshold']} {cp['unit']}. "
                        f"This parameter is WITHIN compliance and should NOT be flagged as an exceedance."
                    ),
                    points=2,
                )
            )
        else:
            rubric_items.append(
                BinaryRubricCategory(
                    name=f"no_false_exceedance_{cp['param'].lower().replace(' ', '_').replace('(', '').replace(')', '')}",
                    question=(
                        f"Does the report correctly show {cp['param']} as compliant? "
                        f"Measured: {cp['measured']} {cp['unit']}, threshold: {cp['threshold']} {cp['unit']}. "
                        f"This parameter is WITHIN compliance and should NOT be flagged as an exceedance."
                    ),
                    points=2,
                )
            )

    # Endangered species checks
    if has_endangered and endangered_species is not None:
        rubric_items.extend([
            BinaryRubricCategory(
                name="identifies_endangered_species",
                question=(
                    f"Does the report identify the presence of {endangered_species['name']} "
                    f"(federal status: {endangered_species['status']}) as a compliance concern?"
                ),
                points=2,
            ),
            BinaryRubricCategory(
                name="correct_buffer_zone",
                question=(
                    f"Does the report correctly state the required buffer zone as {buffer_zone_required} ft? "
                    f"(This is the maximum of the sensitive area buffer [{sensitive_area['buffer_ft']} ft] "
                    f"and the endangered species buffer [{endangered_species['buffer_ft']} ft].)"
                ),
                points=2,
            ),
        ])
    else:
        rubric_items.extend([
            BinaryRubricCategory(
                name="no_false_endangered_species",
                question=(
                    "Does the report correctly note that no threatened or endangered species were detected "
                    "and that no Section 7 consultation is required?"
                ),
                points=2,
            ),
            BinaryRubricCategory(
                name="correct_buffer_zone",
                question=(
                    f"Does the report correctly state the required buffer zone as {buffer_zone_required} ft?"
                ),
                points=1,
            ),
        ])

    if seasonal_restriction:
        rubric_items.append(
            BinaryRubricCategory(
                name="correct_seasonal_restriction",
                question=(
                    f"Does the report identify the seasonal activity restriction: {seasonal_restriction}?"
                ),
                points=2,
            )
        )
    else:
        rubric_items.append(
            BinaryRubricCategory(
                name="correct_no_seasonal_restriction",
                question="Does the report correctly note that no seasonal activity restrictions apply?",
                points=1,
            )
        )

    rubric_items.extend([
        BinaryRubricCategory(
            name="correct_exceedance_count",
            question=(
                f"Does the report identify exactly {n_total_exceedances} parameter exceedance(s) "
                f"(not more, not fewer)?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_overall_determination",
            question=(
                f"Does the report provide a correct overall compliance determination? "
                f"The project has {n_total_exceedances} exceedance(s)"
                + (f" and endangered species concerns" if has_endangered else "")
                + f", so the project {'does NOT meet all regulatory requirements and needs mitigation/permit modifications' if not overall_compliant else 'meets all regulatory requirements'}."
            ),
            points=3,
        ),
        RubricCategory(
            name="analysis_quality",
            description="How thorough and systematic is the environmental compliance analysis?",
            failure="Superficial analysis; most parameters not checked or poorly explained.",
            minor_failure="Some parameters checked but analysis is incomplete or lacks detail on margins.",
            minor_success="Most parameters checked with reasonable detail; minor gaps in analysis.",
            success="All parameters systematically checked with clear measured vs. threshold comparisons, margins noted, and well-structured findings.",
            points=2,
        ),
    ])

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=tuple(rubric_items),
        submission_instructions="Write your completed compliance review to /testbed/compliance_report.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/project_description.txt": project_desc_content,
            "/testbed/data/environmental_data.csv": env_data_content,
            "/testbed/data/regulatory_thresholds.txt": threshold_content,
            "/testbed/data/wildlife_survey.txt": wildlife_content,
        },
        problem_type="environmental_impact_assessment",
    )


# =============================================================================
# 2. IMPORT CLASSIFICATION
# =============================================================================

# Products pool: (description, ambiguous?, hts_chapter, hts_heading, hts_subheading, base_duty_pct)
PRODUCTS_POOL = [
    ("Stainless steel bolts M10x40, grade A4-80", False, "73", "7318.15", "7318.15.20", 0.0),
    ("Lithium-ion battery cells, 3.7V 2600mAh", True, "85", "8507.60", "8507.60.00", 3.4),
    ("Cotton woven shirts, men's, long sleeve", False, "62", "6205.20", "6205.20.20", 19.7),
    ("Polyester filament yarn, 150 denier, textured", False, "54", "5402.33", "5402.33.30", 8.0),
    ("Ceramic floor tiles, glazed, 30x30cm", False, "69", "6908.90", "6908.90.00", 8.5),
    ("Wooden furniture parts, oak, unassembled", True, "94", "9403.90", "9403.90.70", 0.0),
    ("LED light fixtures, ceiling mount, 24W", True, "85", "8539.50", "8539.50.00", 3.9),
    ("Rubber gaskets, vulcanized, industrial", False, "40", "4016.93", "4016.93.50", 2.5),
    ("Glass bottles, clear, 750ml, for beverages", False, "70", "7010.90", "7010.90.50", 0.0),
    ("Aluminum alloy sheets, 2mm, 6061-T6", False, "76", "7606.12", "7606.12.30", 6.5),
    ("Electric motors, AC, single phase, 1HP", True, "85", "8501.40", "8501.40.40", 4.0),
    ("Plastic injection molded housings, ABS", True, "39", "3926.90", "3926.90.99", 5.3),
    ("Organic green tea, loose leaf, 1kg bags", False, "09", "0902.10", "0902.10.10", 6.4),
    ("Titanium alloy surgical implant screws", True, "90", "9021.10", "9021.10.00", 0.0),
    ("Bicycle frames, carbon fiber, unassembled", True, "87", "8714.91", "8714.91.30", 0.0),
    ("Synthetic rubber O-rings, silicone", False, "40", "4016.93", "4016.93.10", 2.5),
    ("Microcontroller chips, ARM Cortex-M4", False, "85", "8542.31", "8542.31.00", 0.0),
    ("Polyethylene bags, printed, retail packaging", False, "39", "3923.21", "3923.21.00", 3.1),
    ("Stainless steel kitchen sinks, single bowl", True, "73", "7324.10", "7324.10.00", 3.4),
    ("Solar photovoltaic panels, 400W monocrystalline", True, "85", "8541.40", "8541.40.60", 0.0),
]

COUNTRIES_OF_ORIGIN = [
    {"country": "China", "code": "CN", "gsp_eligible": False, "usmca": False, "anti_dumping_targets": ["ceramic floor tiles", "solar", "steel"]},
    {"country": "Mexico", "code": "MX", "gsp_eligible": False, "usmca": True, "anti_dumping_targets": []},
    {"country": "Canada", "code": "CA", "gsp_eligible": False, "usmca": True, "anti_dumping_targets": []},
    {"country": "Vietnam", "code": "VN", "gsp_eligible": True, "usmca": False, "anti_dumping_targets": []},
    {"country": "India", "code": "IN", "gsp_eligible": True, "usmca": False, "anti_dumping_targets": ["stainless steel"]},
    {"country": "Germany", "code": "DE", "gsp_eligible": False, "usmca": False, "anti_dumping_targets": []},
    {"country": "Japan", "code": "JP", "gsp_eligible": False, "usmca": False, "anti_dumping_targets": []},
    {"country": "South Korea", "code": "KR", "gsp_eligible": False, "usmca": False, "anti_dumping_targets": []},
    {"country": "Taiwan", "code": "TW", "gsp_eligible": False, "usmca": False, "anti_dumping_targets": []},
    {"country": "Thailand", "code": "TH", "gsp_eligible": True, "usmca": False, "anti_dumping_targets": []},
]

# Anti-dumping duty rates: (product_keyword, country_code, ad_rate_pct)
ANTI_DUMPING_DUTIES = [
    ("ceramic floor tiles", "CN", 222.0),
    ("solar", "CN", 31.0),
    ("steel bolts", "CN", 78.0),
    ("stainless steel", "IN", 12.0),
    ("steel bolts", "IN", 15.0),
]

# Trade agreement preferential rate reductions
# For GSP-eligible: duty reduced to 0 if base duty <= 5%, or reduced by 50% otherwise
# For USMCA: duty reduced to 0 for qualifying goods


def make_import_classification(rand_seed: int = 42) -> RubricDatapoint:
    """Classify imported goods under the Harmonized Tariff Schedule, determine
    duty rates, check trade agreement eligibility, and compute total duties.

    Seed varies: product mix, countries of origin, which items qualify for
    preferential rates, which items have anti-dumping duties, total shipment.
    """
    rng = _random.Random(rand_seed)

    importer_company = pick1(COMPANY_NAMES, rand_seed)
    broker_name = random_name(rand_seed)

    # --- Select line items ---
    n_items = rng.randint(8, 15)
    chosen_products = rng.sample(PRODUCTS_POOL, n_items)

    # Assign countries, quantities, values
    line_items: list[dict] = []
    for i, (desc, ambiguous, chapter, heading, subheading, base_duty) in enumerate(chosen_products):
        country_info = rng.choice(COUNTRIES_OF_ORIGIN)
        qty = rng.randint(50, 5000)
        unit_value = round(rng.uniform(0.50, 150.00), 2)
        declared_value = round(qty * unit_value, 2)
        weight_kg = round(qty * rng.uniform(0.1, 25.0), 1)

        line_items.append({
            "line": i + 1,
            "description": desc,
            "ambiguous": ambiguous,
            "country": country_info["country"],
            "country_code": country_info["code"],
            "gsp_eligible": country_info["gsp_eligible"],
            "usmca": country_info["usmca"],
            "qty": qty,
            "unit_value": unit_value,
            "declared_value": declared_value,
            "weight_kg": weight_kg,
            "hts_chapter": chapter,
            "hts_heading": heading,
            "hts_subheading": subheading,
            "base_duty_pct": base_duty,
        })

    # --- Compute duties ---
    # Determine anti-dumping applicability
    for item in line_items:
        item["anti_dumping_pct"] = 0.0
        item["anti_dumping_amount"] = 0.0
        desc_lower = item["description"].lower()
        best_keyword_len = 0
        best_ad_rate = 0.0
        for keyword, country_code, ad_rate in ANTI_DUMPING_DUTIES:
            if keyword in desc_lower and item["country_code"] == country_code:
                if len(keyword) > best_keyword_len:
                    best_keyword_len = len(keyword)
                    best_ad_rate = ad_rate
        if best_keyword_len > 0:
            item["anti_dumping_pct"] = best_ad_rate
            item["anti_dumping_amount"] = round(item["declared_value"] * best_ad_rate / 100.0, 2)

    # Determine preferential rates
    for item in line_items:
        item["preferential_rate_pct"] = None
        item["preferential_reason"] = None
        if item["usmca"] and item["base_duty_pct"] > 0:
            item["preferential_rate_pct"] = 0.0
            item["preferential_reason"] = "USMCA"
        elif item["gsp_eligible"] and item["base_duty_pct"] > 0:
            if item["base_duty_pct"] <= 5.0:
                item["preferential_rate_pct"] = 0.0
            else:
                item["preferential_rate_pct"] = round(item["base_duty_pct"] / 2.0, 2)
            item["preferential_reason"] = "GSP"

    # Compute effective duty rate and amounts
    for item in line_items:
        if item["preferential_rate_pct"] is not None:
            effective_rate = item["preferential_rate_pct"]
        else:
            effective_rate = item["base_duty_pct"]
        item["effective_duty_pct"] = effective_rate
        item["duty_amount"] = round(item["declared_value"] * effective_rate / 100.0, 2)
        item["total_duty"] = round(item["duty_amount"] + item["anti_dumping_amount"], 2)

    grand_total_duty = round(sum(it["total_duty"] for it in line_items), 2)
    grand_total_value = round(sum(it["declared_value"] for it in line_items), 2)
    total_ad_duties = round(sum(it["anti_dumping_amount"] for it in line_items), 2)
    total_regular_duties = round(sum(it["duty_amount"] for it in line_items), 2)

    # Identify items with anti-dumping duties
    ad_items = [it for it in line_items if it["anti_dumping_pct"] > 0]
    # Identify items with preferential rates
    pref_items = [it for it in line_items if it["preferential_rate_pct"] is not None]
    # Identify ambiguous items for classification checks
    ambiguous_items = [it for it in line_items if it["ambiguous"]]

    # --- Build shipment_manifest.txt ---
    manifest_lines = [
        "IMPORT SHIPMENT MANIFEST",
        "",
        f"Importer: {importer_company}",
        f"Customs Broker: {broker_name}",
        f"Entry Number: {rng.randint(100, 999)}-{rng.randint(1000000, 9999999)}-{rng.randint(0,9)}",
        f"Port of Entry: {rng.choice(['Los Angeles', 'Long Beach', 'New York/Newark', 'Savannah', 'Houston', 'Seattle'])}",
        f"Entry Date: 2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
        "",
        "=" * 100,
        "LINE ITEMS",
        "=" * 100,
        "",
        f"{'Line':<6} {'Description':<50} {'Origin':<10} {'Qty':>8} {'Value (USD)':>14} {'Weight (kg)':>12}",
        f"{'-'*6} {'-'*50} {'-'*10} {'-'*8} {'-'*14} {'-'*12}",
    ]
    for it in line_items:
        manifest_lines.append(
            f"{it['line']:<6} {it['description']:<50} {it['country']:<10} "
            f"{it['qty']:>8} {_fmt_money(it['declared_value']):>14} {it['weight_kg']:>12}"
        )
    manifest_lines.extend([
        "",
        f"{'':>64} {'TOTAL VALUE':>14} {_fmt_money(grand_total_value):>12}",
        "",
    ])
    manifest_content = "\n".join(manifest_lines) + "\n"

    # --- Build tariff_schedule.txt ---
    # Include all relevant HTS chapters with the correct and some nearby headings
    tariff_lines = [
        "HARMONIZED TARIFF SCHEDULE — RELEVANT EXTRACTS",
        "",
        "This document contains the HTS headings and subheadings relevant to the",
        "items in the shipment manifest. Duty rates are ad valorem (percentage of",
        "declared customs value) unless otherwise noted.",
        "",
        "=" * 90,
        "",
        f"{'HTS Subheading':<18} {'Description':<50} {'General Duty':>12}",
        f"{'-'*18} {'-'*50} {'-'*12}",
    ]

    # Collect all headings from line items, plus add some distractors
    seen_subheadings: set[str] = set()
    tariff_entries: list[tuple[str, str, float]] = []
    for it in line_items:
        if it["hts_subheading"] not in seen_subheadings:
            seen_subheadings.add(it["hts_subheading"])
            tariff_entries.append((it["hts_subheading"], it["description"].split(",")[0], it["base_duty_pct"]))

    # Add distractor headings (nearby headings that could be confused)
    distractor_headings = [
        ("7318.16.00", "Nuts of iron or steel", 0.0),
        ("8507.50.00", "Nickel-cadmium battery cells", 3.4),
        ("6205.30.20", "Men's shirts, man-made fibers", 28.2),
        ("8539.29.00", "Incandescent light bulbs, other", 2.6),
        ("3926.90.50", "Plastic articles, machine parts", 5.3),
        ("9021.29.00", "Dental fittings and fixtures", 0.0),
        ("8714.99.80", "Bicycle parts, other", 10.0),
        ("8501.52.40", "AC motors, multi-phase, 750W-75kW", 4.0),
    ]
    for dh_sub, dh_desc, dh_rate in distractor_headings:
        if dh_sub not in seen_subheadings:
            seen_subheadings.add(dh_sub)
            tariff_entries.append((dh_sub, dh_desc, dh_rate))

    tariff_entries.sort(key=lambda x: x[0])
    for sub, desc, rate in tariff_entries:
        rate_str = f"{rate:.1f}%" if rate > 0 else "Free"
        tariff_lines.append(f"{sub:<18} {desc:<50} {rate_str:>12}")

    tariff_lines.extend([
        "",
        "=" * 90,
        "",
        "NOTES:",
        "  - 'Free' means no duty is assessed under the general rate.",
        "  - Classification follows General Rules of Interpretation (GRI).",
        "  - When an item could fall under multiple headings, classify by",
        "    the heading that most specifically describes the article.",
        "  - Ad valorem rates apply to the declared customs value.",
        "",
    ])
    tariff_content = "\n".join(tariff_lines) + "\n"

    # --- Build trade_agreements.txt ---
    trade_lines = [
        "TRADE AGREEMENTS AND PREFERENTIAL RATE PROGRAMS",
        "",
        "=" * 70,
        "UNITED STATES-MEXICO-CANADA AGREEMENT (USMCA)",
        "=" * 70,
        "",
        "Eligible Countries: Mexico (MX), Canada (CA)",
        "Requirements: Goods must satisfy the applicable rules of origin.",
        "Preferential Rate: Duty-free (0%) for qualifying goods.",
        "",
        "Qualifying Criteria:",
        "  - Product must be wholly obtained or produced in a USMCA country, OR",
        "  - Product must satisfy the applicable tariff shift and/or regional",
        "    value content requirement specified in the USMCA rules of origin.",
        "  - Importer must have a valid USMCA Certificate of Origin on file.",
        "",
        "For purposes of this classification exercise, assume all goods from",
        "USMCA countries satisfy the rules of origin requirements.",
        "",
        "=" * 70,
        "GENERALIZED SYSTEM OF PREFERENCES (GSP)",
        "=" * 70,
        "",
        "Eligible Countries (selected): Vietnam (VN), India (IN), Thailand (TH),",
        "  Indonesia (ID), Philippines (PH), Cambodia (KH), Colombia (CO)",
        "",
        "NOT Eligible: China (CN), Mexico (MX), Canada (CA), Germany (DE),",
        "  Japan (JP), South Korea (KR), Taiwan (TW)",
        "",
        "Preferential Rate Rules:",
        "  - If the general duty rate is 5% or less: reduced to 0% (duty-free)",
        "  - If the general duty rate is above 5%: reduced by 50%",
        "  - Goods must be imported directly from the GSP-eligible country",
        "  - Certain products are excluded from GSP (see exclusion lists)",
        "",
        "For purposes of this classification exercise, assume all goods from",
        "GSP-eligible countries meet the direct shipment requirement.",
        "",
    ]
    trade_content = "\n".join(trade_lines) + "\n"

    # --- Build special_requirements.txt ---
    special_lines = [
        "SPECIAL REQUIREMENTS — ANTI-DUMPING DUTIES, QUOTAS, AND AGENCY REQUIREMENTS",
        "",
        "=" * 70,
        "ANTI-DUMPING AND COUNTERVAILING DUTIES",
        "=" * 70,
        "",
        "The following product/country combinations are subject to additional",
        "anti-dumping (AD) or countervailing (CVD) duties. These are assessed",
        "IN ADDITION to the regular duty rate.",
        "",
        f"{'Product Category':<40} {'Country':<15} {'AD/CVD Rate':>12}",
        f"{'-'*40} {'-'*15} {'-'*12}",
        f"{'Ceramic floor tiles':<40} {'China':>15} {'222.0%':>12}",
        f"{'Solar photovoltaic cells/panels':<40} {'China':>15} {'31.0%':>12}",
        f"{'Steel bolts and fasteners':<40} {'China':>15} {'78.0%':>12}",
        f"{'Stainless steel products':<40} {'India':>15} {'12.0%':>12}",
        f"{'Steel bolts and fasteners':<40} {'India':>15} {'15.0%':>12}",
        "",
        "Anti-dumping duties are calculated on the declared customs value of",
        "the goods. They apply regardless of any preferential trade agreement.",
        "",
        "=" * 70,
        "QUOTA LIMITS (NOT APPLICABLE TO THIS SHIPMENT)",
        "=" * 70,
        "",
        "No quota restrictions apply to the product categories in this shipment.",
        "",
        "=" * 70,
        "AGENCY REQUIREMENTS",
        "=" * 70,
        "",
        "Certain items may require clearance from additional federal agencies:",
        "  - FDA: Food products, medical devices, pharmaceuticals",
        "  - CPSC: Consumer products (electronics, children's items)",
        "  - EPA: Chemicals, pesticides, vehicles/engines",
        "  - FCC: Electronic devices with radio frequency components",
        "  - USDA: Agricultural products, wood packaging materials",
        "",
        "Note: Agency requirements do not affect duty calculations but may",
        "delay release of goods from customs.",
        "",
    ]
    special_content = "\n".join(special_lines) + "\n"

    # --- Problem statement ---
    problem_statement = f"""# Import Classification and Duty Computation

You are {broker_name}, a licensed customs broker. You must classify a shipment
of imported goods under the Harmonized Tariff Schedule, determine applicable
duty rates, check for trade agreement eligibility, identify any anti-dumping
duties, and compute the total duties owed.

## Source Files
- /testbed/data/shipment_manifest.txt — Shipment line items with descriptions, origins, quantities, values
- /testbed/data/tariff_schedule.txt — Relevant HTS subheadings and general duty rates
- /testbed/data/trade_agreements.txt — USMCA and GSP eligibility criteria and preferential rate rules
- /testbed/data/special_requirements.txt — Anti-dumping duties and other special requirements

## Requirements
1. Classify each line item under the correct HTS subheading
2. Determine the general duty rate for each item
3. Check whether any trade agreement (USMCA, GSP) provides a preferential rate
4. Identify items subject to anti-dumping duties
5. Compute the duty amount for each line item (regular + anti-dumping)
6. Compute the grand total duties owed

Write your classification and duty computation report to /testbed/classification_report.txt"""

    # --- Build rubric ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does the file /testbed/classification_report.txt exist and contain substantive content?",
            points=1,
        ),
    ]

    # Correct classification for ambiguous items (up to 3)
    for j, amb_item in enumerate(ambiguous_items[:3]):
        rubric_items.append(
            BinaryRubricCategory(
                name=f"correct_classification_item_{amb_item['line']}",
                question=(
                    f"Does the report classify line {amb_item['line']} ('{amb_item['description']}') "
                    f"under HTS subheading {amb_item['hts_subheading']} (or the correct heading "
                    f"{amb_item['hts_heading']})? This is an ambiguous item that could be "
                    f"misclassified under a nearby heading."
                ),
                points=2,
            )
        )

    # Correct duty rate for select items
    rate_check_items = rng.sample(line_items, min(4, len(line_items)))
    for it in rate_check_items:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"correct_duty_rate_item_{it['line']}",
                question=(
                    f"Does the report apply the correct effective duty rate for line {it['line']} "
                    f"('{it['description'][:40]}...' from {it['country']})? "
                    f"The base rate is {it['base_duty_pct']}%"
                    + (f", reduced to {it['effective_duty_pct']}% via {it['preferential_reason']}" if it["preferential_reason"] else "")
                    + f". Effective rate: {it['effective_duty_pct']}%."
                ),
                points=2,
            )
        )

    # Preferential rate checks
    if pref_items:
        pref_check = rng.choice(pref_items)
        rubric_items.append(
            BinaryRubricCategory(
                name="correct_preferential_rate_applied",
                question=(
                    f"Does the report correctly apply a preferential rate for line {pref_check['line']} "
                    f"('{pref_check['description'][:35]}...' from {pref_check['country']})? "
                    f"The item qualifies for {pref_check['preferential_reason']} "
                    f"(base rate {pref_check['base_duty_pct']}% reduced to {pref_check['effective_duty_pct']}%)."
                ),
                points=2,
            )
        )

    # False-positive check: non-preferential items should not get preferential rates
    non_pref_items = [it for it in line_items if it["preferential_reason"] is None and it["base_duty_pct"] > 0]
    if non_pref_items:
        fp_item = rng.choice(non_pref_items)
        rubric_items.append(
            BinaryRubricCategory(
                name=f"no_false_preferential_item_{fp_item['line']}",
                question=(
                    f"Does the report correctly NOT apply a preferential trade agreement rate "
                    f"to line {fp_item['line']} ('{fp_item['description'][:35]}...' from "
                    f"{fp_item['country']})? This item does NOT qualify for USMCA or GSP."
                ),
                points=2,
            )
        )

    # Anti-dumping checks
    if ad_items:
        for ad_item in ad_items[:2]:
            rubric_items.append(
                BinaryRubricCategory(
                    name=f"correct_anti_dumping_item_{ad_item['line']}",
                    question=(
                        f"Does the report identify line {ad_item['line']} "
                        f"('{ad_item['description'][:35]}...' from {ad_item['country']}) as subject "
                        f"to anti-dumping duties of {ad_item['anti_dumping_pct']}%? "
                        f"The AD duty amount is {_fmt_money(ad_item['anti_dumping_amount'])}."
                    ),
                    points=2,
                )
            )

    # Non-AD items should not be flagged
    non_ad_items = [it for it in line_items if it["anti_dumping_pct"] == 0]
    if non_ad_items:
        fp_ad = rng.choice(non_ad_items)
        rubric_items.append(
            BinaryRubricCategory(
                name=f"no_false_anti_dumping_item_{fp_ad['line']}",
                question=(
                    f"Does the report correctly NOT flag line {fp_ad['line']} "
                    f"('{fp_ad['description'][:35]}...' from {fp_ad['country']}) as subject "
                    f"to anti-dumping duties? This item has no AD/CVD orders against it."
                ),
                points=1,
            )
        )

    # Per-item duty amount checks for 2 specific items
    duty_check_items = [it for it in line_items if it["total_duty"] > 0]
    if len(duty_check_items) < 2:
        duty_check_items = line_items[:2]
    else:
        duty_check_items = rng.sample(duty_check_items, min(2, len(duty_check_items)))
    for dc in duty_check_items:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"correct_duty_amount_item_{dc['line']}",
                question=(
                    f"Does the report compute the total duty for line {dc['line']} "
                    f"('{dc['description'][:35]}...', value {_fmt_money(dc['declared_value'])}) "
                    f"as approximately {_fmt_money(dc['total_duty'])} (within 10% or $20, "
                    f"whichever is larger)?"
                ),
                points=2,
            )
        )

    # Grand total duty
    rubric_items.append(
        BinaryRubricCategory(
            name="correct_total_regular_duties",
            question=(
                f"Does the report compute total regular duties (excluding anti-dumping) as "
                f"approximately {_fmt_money(total_regular_duties)} (within 5% or $50, whichever is larger)?"
            ),
            points=2,
        )
    )

    if total_ad_duties > 0:
        rubric_items.append(
            BinaryRubricCategory(
                name="correct_total_ad_duties",
                question=(
                    f"Does the report compute total anti-dumping/CVD duties as approximately "
                    f"{_fmt_money(total_ad_duties)} (within 5% or $50, whichever is larger)?"
                ),
                points=2,
            )
        )

    rubric_items.extend([
        BinaryRubricCategory(
            name="correct_grand_total_duty",
            question=(
                f"Does the report compute the grand total duties owed (regular + anti-dumping) "
                f"as approximately {_fmt_money(grand_total_duty)} (within 5% or $100, whichever is larger)?"
            ),
            points=3,
        ),
        BinaryRubricCategory(
            name="correct_total_declared_value",
            question=(
                f"Does the report state the total declared customs value as approximately "
                f"{_fmt_money(grand_total_value)} (within $100)?"
            ),
            points=1,
        ),
        RubricCategory(
            name="classification_thoroughness",
            description="How thorough and systematic is the tariff classification and duty analysis?",
            failure="Superficial analysis; most items not classified or duties not computed.",
            minor_failure="Some items classified but analysis is incomplete or rates are wrong for several items.",
            minor_success="Most items correctly classified with reasonable duty computations; minor gaps.",
            success="All items systematically classified with correct HTS headings, duty rates, trade agreement analysis, and clear per-item duty calculations.",
            points=2,
        ),
    ])

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=tuple(rubric_items),
        submission_instructions="Write your completed classification report to /testbed/classification_report.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/shipment_manifest.txt": manifest_content,
            "/testbed/data/tariff_schedule.txt": tariff_content,
            "/testbed/data/trade_agreements.txt": trade_content,
            "/testbed/data/special_requirements.txt": special_content,
        },
        problem_type="import_classification",
    )


# =============================================================================
# 3. WORKPLACE SAFETY AUDIT
# =============================================================================

FACILITY_TYPES = [
    "manufacturing plant",
    "distribution warehouse",
    "construction site",
    "chemical laboratory",
    "food processing facility",
    "metal fabrication shop",
]

# OSHA standards pool: (standard_number, title, requirement_summary, severity_if_violated)
OSHA_STANDARDS_POOL = [
    ("1910.303(b)", "Electrical — Examination and approval", "All electrical equipment must be free of recognized hazards and approved for its intended use.", "Serious"),
    ("1910.305(a)", "Electrical — Wiring methods", "Wiring must be properly secured, supported, and protected from physical damage.", "Serious"),
    ("1926.501(b)(1)", "Fall Protection — Unprotected sides and edges", "Employees on walking/working surfaces with unprotected sides 6+ feet above lower level require fall protection.", "Serious"),
    ("1926.502(d)", "Fall Protection — Guardrail systems", "Guardrail systems must have top edge height of 42 inches +/- 3 inches.", "Serious"),
    ("1910.106(d)", "Flammable Liquids — Storage", "Flammable liquids must be stored in approved containers in designated storage areas with proper ventilation.", "Serious"),
    ("1910.1200(h)", "Hazard Communication — Employee training", "Employees must receive training on hazardous chemicals in their work area prior to initial assignment.", "Other-than-Serious"),
    ("1910.134(c)", "Respiratory Protection — Program requirements", "Employer must establish a written respiratory protection program with worksite-specific procedures.", "Serious"),
    ("1910.132(a)", "PPE — General requirements", "Employer shall provide and ensure use of appropriate personal protective equipment.", "Serious"),
    ("1910.132(f)", "PPE — Training", "Employer must train each employee required to use PPE on when, what, and how to use it.", "Other-than-Serious"),
    ("1910.212(a)(1)", "Machine Guarding — General", "One or more methods of machine guarding shall be provided to protect operators from hazards.", "Serious"),
    ("1910.212(a)(3)(ii)", "Machine Guarding — Point of operation", "Point of operation guards shall prevent hands from entering the point of operation.", "Serious"),
    ("1910.36(b)", "Means of Egress — Design and construction", "Exit routes must be permanent, maintained free of obstructions, and lead directly outside.", "Serious"),
    ("1910.37(a)", "Means of Egress — Maintenance", "Exit routes must be kept free of explosive or highly flammable furnishings and decorations.", "Serious"),
    ("1910.157(c)", "Fire Protection — Portable fire extinguishers", "Employer must provide portable fire extinguishers within 75 feet travel distance for Class A hazards.", "Serious"),
    ("1910.157(e)", "Fire Protection — Inspection and maintenance", "Fire extinguishers must be visually inspected monthly and have annual maintenance checks.", "Other-than-Serious"),
    ("1910.147(c)(1)", "Lockout/Tagout — Energy control program", "Employer must establish an energy control program with documented procedures for each machine.", "Serious"),
    ("1910.147(c)(7)", "Lockout/Tagout — Periodic inspection", "Periodic inspections of energy control procedures must be conducted at least annually.", "Other-than-Serious"),
    ("1910.178(l)", "Powered Industrial Trucks — Operator training", "Operators of powered industrial trucks must be trained and evaluated before operating trucks.", "Serious"),
    ("1910.22(a)", "Walking-Working Surfaces — General", "All places of employment must be kept clean, orderly, and in a sanitary condition.", "Other-than-Serious"),
    ("1910.23(c)", "Walking-Working Surfaces — Ladders", "Ladders must be maintained free of oil, grease, and other slipping hazards.", "Other-than-Serious"),
    ("1910.146(c)", "Permit-Required Confined Spaces", "Employer must evaluate workplace to determine if permit-required confined spaces exist and implement entry procedures.", "Serious"),
    ("1910.1030(d)", "Bloodborne Pathogens — Exposure control", "Employer must establish a written exposure control plan.", "Serious"),
    ("1910.95(b)", "Occupational Noise Exposure", "Employer must implement hearing conservation program when noise levels exceed 85 dBA TWA.", "Other-than-Serious"),
]

# Observation templates: (observation_text_template, is_violation, osha_standard_index, severity_override)
# Templates use {facility} and {detail} placeholders
OBSERVATION_TEMPLATES = [
    # Violations
    ("Exposed wiring observed near the {area} with damaged insulation and no junction box cover. Wires are accessible to employees.", True, 1, None),
    ("Employee observed working on elevated platform ({height} feet) without fall protection harness or guardrails.", True, 2, None),
    ("Multiple containers of {chemical} found stored outside designated flammable storage cabinet in the {area}.", True, 4, None),
    ("Three employees in the {area} could not locate the Safety Data Sheets for chemicals they work with daily.", True, 5, None),
    ("Grinding station #2 in the {area} has the wheel guard removed. Operator was using the grinder without the guard in place.", True, 9, None),
    ("Emergency exit in the {area} is partially blocked by pallets of {material}. Door opens but clearance is restricted.", True, 11, None),
    ("Fire extinguisher near {area} has an expired inspection tag dated {months_ago} months ago. No annual maintenance record found.", True, 14, None),
    ("Forklift operator {name} could not produce training certification. Supervisor confirmed no training records on file.", True, 17, None),
    ("Lockout/tagout procedures for the hydraulic press in {area} have not been reviewed or updated since 2019. No annual inspection records found.", True, 16, None),
    ("Employees in the welding area are not wearing respiratory protection despite visible fume generation. No respiratory protection program documentation was available.", True, 6, "Serious"),
    ("Machine guard on the stamping press in {area} has a gap of approximately 3 inches allowing hand access to the point of operation.", True, 10, None),
    ("Electrical panel in the {area} has no labeling for circuit breakers. Panel cover is missing and live parts are exposed.", True, 0, None),
    # Compliant observations
    ("All employees in the {area} were observed wearing appropriate PPE including safety glasses, steel-toed boots, and hearing protection.", False, None, None),
    ("Emergency lighting in all exit corridors tested and functioning properly. Backup battery systems operational.", False, None, None),
    ("Chemical storage area in the {area} is properly organized with SDS binders accessible and up to date.", False, None, None),
    ("Guardrail systems on the mezzanine level measured at 42 inches with proper mid-rails installed.", False, None, None),
    ("All portable fire extinguishers inspected — tags current, pressure gauges in green zone, monthly visual inspections documented.", False, None, None),
    ("Lockout/tagout procedures for the CNC machines were reviewed and found to be current with annual inspections documented.", False, None, None),
    ("Forklift operators {name} and {name2} both produced valid training certificates dated within the past 3 years.", False, None, None),
    ("Housekeeping in the main production area is excellent — floors clean, aisles clear, materials properly stored.", False, None, None),
    ("First aid supplies are stocked and accessible. AED device present and inspection current.", False, None, None),
    ("Confined space entry permits reviewed — all required elements present including atmospheric testing records.", False, None, None),
    ("Ladder inspections documented quarterly. All ladders observed in good condition with proper labeling.", False, None, None),
    ("Hearing conservation program in place. Audiometric testing records current for all employees in high-noise areas.", False, None, None),
    ("Eyewash stations tested weekly with documentation. Flow rates and water temperature within specifications.", False, None, None),
]

# OSHA penalty schedule (simplified)
PENALTY_AMOUNTS = {
    "Serious": (4000, 14000),
    "Other-than-Serious": (0, 7000),
    "Willful": (11000, 145000),
    "Repeat": (11000, 145000),
}

# Training types for records
TRAINING_TYPES = [
    "Hazard Communication",
    "Fire Extinguisher Use",
    "Lockout/Tagout",
    "Fall Protection",
    "Forklift Operation",
    "Confined Space Entry",
    "Bloodborne Pathogens",
    "PPE Use and Care",
    "Emergency Action Plan",
    "Respiratory Protection",
    "Electrical Safety",
    "Machine Guarding Awareness",
]


def make_workplace_safety_audit(rand_seed: int = 42) -> RubricDatapoint:
    """Audit a facility inspection report against OSHA standards. Cross-reference
    inspection findings, employee records, training logs, and incident history.

    Seed varies: facility type, number of violations (4-7 of 15-25 observations),
    violation types, repeat violations, training compliance rate.
    """
    rng = _random.Random(rand_seed)

    company = pick1(COMPANY_NAMES, rand_seed)
    inspector_name = random_name(rand_seed)
    facility_manager = random_name(rand_seed + 1)
    facility_type = rng.choice(FACILITY_TYPES)

    # --- Configure observations ---
    n_total_obs = rng.randint(15, 25)
    n_violations = rng.randint(4, 7)
    n_compliant = n_total_obs - n_violations

    # Pick violation templates
    violation_templates = [t for t in OBSERVATION_TEMPLATES if t[1] is True]
    compliant_templates = [t for t in OBSERVATION_TEMPLATES if t[1] is False]

    chosen_violations = rng.sample(violation_templates, min(n_violations, len(violation_templates)))
    chosen_compliant = rng.sample(compliant_templates, min(n_compliant, len(compliant_templates)))
    # Pad compliant if needed
    while len(chosen_compliant) < n_compliant:
        chosen_compliant.append(rng.choice(compliant_templates))

    # Generate observation details
    areas = ["production floor", "warehouse bay 3", "loading dock", "chemical storage room",
             "maintenance shop", "assembly line 2", "paint booth area", "shipping department",
             "break room corridor", "mezzanine level", "raw materials storage", "quality control lab",
             "tool crib", "compressor room", "welding station"]
    chemicals = ["acetone", "toluene", "isopropyl alcohol", "MEK (methyl ethyl ketone)",
                 "lacquer thinner", "mineral spirits"]
    materials = ["raw materials", "finished goods", "cardboard boxes", "cleaning supplies"]
    employee_names_list = random_names(rand_seed + 20, 30)

    observations: list[dict] = []
    violation_details: list[dict] = []

    for i, (template, is_viol, std_idx, sev_override) in enumerate(chosen_violations):
        area = rng.choice(areas)
        obs_text = template.format(
            area=area,
            facility=facility_type,
            height=rng.choice([8, 10, 12, 15, 20]),
            chemical=rng.choice(chemicals),
            material=rng.choice(materials),
            months_ago=rng.randint(6, 18),
            name=rng.choice(employee_names_list),
            name2=rng.choice(employee_names_list),
            detail="",
        )
        standard = OSHA_STANDARDS_POOL[std_idx]
        severity = sev_override if sev_override else standard[3]

        observations.append({
            "obs_num": None,  # will be assigned after shuffling
            "text": obs_text,
            "is_violation": True,
            "standard_num": standard[0],
            "standard_title": standard[1],
            "severity": severity,
            "area": area,
        })
        violation_details.append(observations[-1])

    for i, (template, is_viol, std_idx, sev_override) in enumerate(chosen_compliant):
        area = rng.choice(areas)
        obs_text = template.format(
            area=area,
            facility=facility_type,
            height="",
            chemical="",
            material="",
            months_ago="",
            name=rng.choice(employee_names_list),
            name2=rng.choice(employee_names_list),
            detail="",
        )
        observations.append({
            "obs_num": None,
            "text": obs_text,
            "is_violation": False,
            "standard_num": None,
            "standard_title": None,
            "severity": None,
            "area": area,
        })

    # Shuffle and assign observation numbers
    rng.shuffle(observations)
    for k, obs in enumerate(observations):
        obs["obs_num"] = k + 1

    # Re-identify violation_details by reference update
    violation_details = [obs for obs in observations if obs["is_violation"]]
    compliant_observations = [obs for obs in observations if not obs["is_violation"]]

    # --- Determine repeat violations ---
    # Pick 0-2 violations to be repeats (matching incident history)
    n_repeats = rng.randint(0, min(2, len(violation_details)))
    repeat_violations = rng.sample(violation_details, n_repeats) if n_repeats > 0 else []
    for rv in repeat_violations:
        rv["is_repeat"] = True
        rv["severity"] = "Repeat"
    non_repeat_violations = [v for v in violation_details if v not in repeat_violations]

    # --- Compute penalties ---
    total_proposed_penalty = 0.0
    for v in violation_details:
        sev = v["severity"]
        lo, hi = PENALTY_AMOUNTS.get(sev, (0, 7000))
        # Use midpoint of penalty range for deterministic, reproducible penalties
        penalty = round((lo + hi) / 2, 2)
        v["proposed_penalty"] = penalty
        total_proposed_penalty += penalty
    total_proposed_penalty = round(total_proposed_penalty, 2)

    # --- Build training records ---
    n_employees = rng.randint(25, 50)
    emp_ids = [f"EMP-{1000 + i}" for i in range(n_employees)]
    training_records: list[dict] = []
    n_expired = 0
    n_current = 0

    for emp_id in emp_ids:
        # Each employee should have 3-6 training types
        n_trainings = rng.randint(3, 6)
        emp_trainings = rng.sample(TRAINING_TYPES, n_trainings)
        for tt in emp_trainings:
            # Completion date: 6-36 months ago
            months_ago = rng.randint(6, 36)
            comp_month = max(1, (10 - months_ago % 12 + 12) % 12 + 1)
            comp_year = 2024 - (months_ago // 12) - (1 if months_ago % 12 >= 10 else 0)
            if comp_year < 2021:
                comp_year = 2021
            comp_date = f"{comp_year}-{comp_month:02d}-15"

            # Expiration: 12 months after completion for most, 36 months for forklift
            exp_months = 36 if tt == "Forklift Operation" else 12
            exp_year = comp_year + (exp_months // 12)
            exp_month = comp_month + (exp_months % 12)
            if exp_month > 12:
                exp_month -= 12
                exp_year += 1
            exp_date = f"{exp_year}-{exp_month:02d}-15"

            # Check if expired (before Dec 2024)
            is_expired = exp_year < 2024 or (exp_year == 2024 and exp_month < 12)
            if is_expired:
                n_expired += 1
            else:
                n_current += 1

            training_records.append({
                "emp_id": emp_id,
                "training_type": tt,
                "completion_date": comp_date,
                "expiration_date": exp_date,
                "is_expired": is_expired,
            })

    total_training_records = len(training_records)
    training_compliance_pct = round(n_current / total_training_records * 100, 1) if total_training_records > 0 else 100.0

    # --- Build incident history ---
    incident_types = [
        "Laceration", "Strain/Sprain", "Slip/Trip/Fall", "Chemical Exposure",
        "Electrical Shock", "Caught In/Between", "Struck By", "Eye Injury",
        "Heat Stress", "Respiratory Issue",
    ]
    severity_levels = ["First Aid", "Recordable", "Lost Time", "Near Miss"]
    n_incidents = rng.randint(8, 18)
    incidents: list[dict] = []

    for _ in range(n_incidents):
        inc_month = rng.randint(1, 12)
        inc_day = rng.randint(1, 28)
        inc_type = rng.choice(incident_types)
        inc_sev = rng.choice(severity_levels)
        inc_area = rng.choice(areas)
        corrective = rng.choice([
            "Re-trained affected employees",
            "Installed additional signage",
            "Replaced defective equipment",
            "Updated procedures",
            "Conducted safety stand-down",
            "Added engineering controls",
            "Ordered new PPE",
        ])
        status = rng.choice(["Closed", "Closed", "Closed", "Open", "In Progress"])
        incidents.append({
            "date": f"2024-{inc_month:02d}-{inc_day:02d}",
            "type": inc_type,
            "severity": inc_sev,
            "area": inc_area,
            "corrective_action": corrective,
            "status": status,
        })

    # Plant repeat-violation-matching incidents
    for rv in repeat_violations:
        # Add a matching past incident
        inc_month = rng.randint(1, 6)
        matching_type = "Electrical Shock" if "Electrical" in rv["standard_title"] else \
                        "Slip/Trip/Fall" if "Fall" in rv["standard_title"] else \
                        "Chemical Exposure" if "Flammable" in rv["standard_title"] or "Hazard" in rv["standard_title"] else \
                        "Caught In/Between" if "Machine" in rv["standard_title"] else \
                        "Laceration"
        incidents.append({
            "date": f"2024-{inc_month:02d}-{rng.randint(1,28):02d}",
            "type": matching_type,
            "severity": "Recordable",
            "area": rv["area"],
            "corrective_action": "Issued corrective action notice",
            "status": "Closed",
        })

    # --- Build inspection_report.txt ---
    inspection_lines = [
        "OSHA COMPLIANCE INSPECTION REPORT",
        "",
        f"Facility: {company} — {facility_type.title()}",
        f"Inspector: {inspector_name}, CSP (Certified Safety Professional)",
        f"Facility Contact: {facility_manager}, Facility Manager",
        f"Inspection Date: 2024-{rng.randint(9,11):02d}-{rng.randint(1,28):02d}",
        f"Inspection Type: Comprehensive Compliance Audit",
        "",
        "=" * 70,
        "INSPECTION OBSERVATIONS",
        "=" * 70,
        "",
    ]
    for obs in observations:
        inspection_lines.extend([
            f"Observation #{obs['obs_num']}:",
            f"  Area: {obs['area'].title()}",
            f"  {obs['text']}",
            "",
        ])
    inspection_lines.extend([
        f"Total Observations: {len(observations)}",
        "",
        "End of inspection observations.",
        "",
    ])
    inspection_content = "\n".join(inspection_lines) + "\n"

    # --- Build osha_standards.txt ---
    standards_lines = [
        "APPLICABLE OSHA STANDARDS REFERENCE",
        "",
        "This document contains OSHA standards relevant to the facility type.",
        "Standards are organized by category with section numbers, requirements,",
        "and severity classifications if violated.",
        "",
        "=" * 80,
        "",
        f"{'Standard':<20} {'Title':<40} {'Severity':<20}",
        f"{'-'*20} {'-'*40} {'-'*20}",
    ]
    for std_num, std_title, std_req, std_sev in OSHA_STANDARDS_POOL:
        standards_lines.append(f"{std_num:<20} {std_title:<40} {std_sev:<20}")
    standards_lines.extend([
        "",
        "=" * 80,
        "DETAILED REQUIREMENTS",
        "=" * 80,
        "",
    ])
    for std_num, std_title, std_req, std_sev in OSHA_STANDARDS_POOL:
        standards_lines.extend([
            f"{std_num} — {std_title}",
            f"  Requirement: {std_req}",
            f"  Violation Severity: {std_sev}",
            "",
        ])
    standards_lines.extend([
        "=" * 80,
        "PENALTY SCHEDULE (CURRENT YEAR)",
        "=" * 80,
        "",
        f"{'Violation Type':<25} {'Penalty Range':>20}",
        f"{'-'*25} {'-'*20}",
        f"{'Serious':<25} {'$4,000 - $14,000':>20}",
        f"{'Other-than-Serious':<25} {'$0 - $7,000':>20}",
        f"{'Willful':<25} {'$11,000 - $145,000':>20}",
        f"{'Repeat':<25} {'$11,000 - $145,000':>20}",
        "",
        "NOTE: Apply the base penalty at the midpoint of each severity range.",
        "For example, a Serious violation carries a proposed penalty of $9,000.00",
        "(midpoint of $4,000 - $14,000).",
        "",
        "NOTE: Repeat violations are assessed when a substantially similar",
        "violation was cited within the past 5 years at the same establishment.",
        "The presence of a prior incident of the same type in the incident",
        "history, followed by inadequate corrective action, may support a",
        "repeat violation classification.",
        "",
    ])
    standards_content = "\n".join(standards_lines) + "\n"

    # --- Build training_records.csv ---
    training_csv_header = "Employee ID,Training Type,Completion Date,Expiration Date"
    training_csv_rows = [training_csv_header]
    for tr in training_records:
        training_csv_rows.append(f"{tr['emp_id']},{tr['training_type']},{tr['completion_date']},{tr['expiration_date']}")
    training_content = "\n".join(training_csv_rows) + "\n"

    # --- Build incident_history.csv ---
    incident_csv_header = "Date,Incident Type,Severity,Area,Corrective Action,Status"
    incident_csv_rows = [incident_csv_header]
    for inc in incidents:
        incident_csv_rows.append(
            f"{inc['date']},{inc['type']},{inc['severity']},{inc['area']},{inc['corrective_action']},{inc['status']}"
        )
    incident_content = "\n".join(incident_csv_rows) + "\n"

    # --- Problem statement ---
    problem_statement = f"""# Workplace Safety Audit — OSHA Compliance Review

You are an OSHA compliance auditor reviewing an inspection of {company}'s
{facility_type}. Cross-reference the inspection observations against OSHA
standards, training records, and incident history to identify violations
and assess penalties.

## Source Files
- /testbed/data/inspection_report.txt — Inspector's observations ({len(observations)} observations covering various safety areas)
- /testbed/data/osha_standards.txt — Relevant OSHA standards with section numbers, requirements, and severity classifications
- /testbed/data/training_records.csv — Employee training records with completion and expiration dates
- /testbed/data/incident_history.csv — Past 12 months of workplace incidents

## Requirements
1. Review each inspection observation and determine whether it constitutes an OSHA violation
2. For each violation, cite the specific OSHA standard violated
3. Classify each violation severity (Serious, Other-than-Serious, Repeat)
4. Cross-reference incident history to identify any repeat violations (elevated penalties)
5. Analyze training records — compute the percentage of current (non-expired) training records
6. Count the number of expired training certifications
7. Propose a penalty amount for each violation per the OSHA penalty schedule
8. Compute the total proposed penalty

Write your audit report to /testbed/safety_audit_report.txt"""

    # --- Build rubric ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does the file /testbed/safety_audit_report.txt exist and contain substantive content?",
            points=1,
        ),
    ]

    # Per-violation: found it (cap at 5 to keep rubric 15-22)
    for v in violation_details[:5]:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"found_violation_obs_{v['obs_num']}",
                question=(
                    f"Does the audit report identify Observation #{v['obs_num']} as a violation? "
                    f"The observation describes: {v['text'][:80]}..."
                ),
                points=2,
            )
        )

    # Per-violation: correct OSHA standard (cap at 4)
    for v in violation_details[:4]:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"correct_standard_obs_{v['obs_num']}",
                question=(
                    f"Does the audit report cite the correct OSHA standard for Observation #{v['obs_num']}? "
                    f"The applicable standard is {v['standard_num']} ({v['standard_title']})."
                ),
                points=1,
            )
        )

    # Per-violation: correct severity (for first 3)
    for v in violation_details[:3]:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"correct_severity_obs_{v['obs_num']}",
                question=(
                    f"Does the audit report classify the severity of Observation #{v['obs_num']} "
                    f"as '{v['severity']}'? (Standard: {v['standard_num']})"
                ),
                points=1,
            )
        )

    # Repeat violation identification
    if repeat_violations:
        for rv in repeat_violations:
            rubric_items.append(
                BinaryRubricCategory(
                    name=f"identifies_repeat_violation_obs_{rv['obs_num']}",
                    question=(
                        f"Does the audit report identify Observation #{rv['obs_num']} as a REPEAT violation? "
                        f"The incident history shows a prior similar incident in the same area ({rv['area']})."
                    ),
                    points=2,
                )
            )
    else:
        rubric_items.append(
            BinaryRubricCategory(
                name="no_repeat_violations_noted",
                question=(
                    "Does the audit report correctly note that no repeat violations were identified "
                    "(none of the current violations match prior cited violations)?"
                ),
                points=1,
            )
        )

    # False-positive checks on compliant observations (cap at 2)
    fp_compliant = rng.sample(compliant_observations, min(2, len(compliant_observations)))
    for fp in fp_compliant:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"no_false_violation_obs_{fp['obs_num']}",
                question=(
                    f"Does the audit report correctly NOT flag Observation #{fp['obs_num']} as a violation? "
                    f"This observation describes: {fp['text'][:70]}... which is a compliant finding."
                ),
                points=2,
            )
        )

    # Training compliance
    rubric_items.extend([
        BinaryRubricCategory(
            name="correct_training_compliance_pct",
            question=(
                f"Does the audit report compute the training compliance rate as approximately "
                f"{training_compliance_pct}% (within 3 percentage points)? "
                f"({n_current} current records out of {total_training_records} total.)"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_expired_training_count",
            question=(
                f"Does the audit report identify approximately {n_expired} expired training "
                f"certifications (within 10% or 5 records, whichever is larger)?"
            ),
            points=1,
        ),
    ])

    # Total proposed penalty
    rubric_items.extend([
        BinaryRubricCategory(
            name="correct_total_proposed_penalty",
            question=(
                f"Does the audit report compute a total proposed penalty of approximately "
                f"{_fmt_money(total_proposed_penalty)} (within 15% of this amount)? "
                f"The penalty is based on {len(violation_details)} violations with their "
                f"respective severity levels per the OSHA penalty schedule."
            ),
            points=3,
        ),
        BinaryRubricCategory(
            name="correct_violation_count",
            question=(
                f"Does the audit report identify exactly {len(violation_details)} violations "
                f"out of {len(observations)} total observations?"
            ),
            points=2,
        ),
    ])

    rubric_items.append(
        RubricCategory(
            name="audit_thoroughness",
            description="How thorough and systematic is the safety audit analysis?",
            failure="Superficial analysis; most observations not evaluated or standards not cited.",
            minor_failure="Some violations identified but missing standard citations, severity levels, or penalty calculations.",
            minor_success="Most violations correctly identified with standards and severities; minor gaps in penalty computation or training analysis.",
            success="All observations systematically evaluated, standards correctly cited, severities assigned, repeat violations identified, training compliance analyzed, and penalties computed per schedule.",
            points=2,
        )
    )

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=tuple(rubric_items),
        submission_instructions="Write your completed safety audit report to /testbed/safety_audit_report.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/inspection_report.txt": inspection_content,
            "/testbed/data/osha_standards.txt": standards_content,
            "/testbed/data/training_records.csv": training_content,
            "/testbed/data/incident_history.csv": incident_content,
        },
        problem_type="workplace_safety_audit",
    )
