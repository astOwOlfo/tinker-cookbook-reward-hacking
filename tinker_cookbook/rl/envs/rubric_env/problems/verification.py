"""Analysis and verification tasks.

Six factory functions that generate problems requiring the agent to read
provided data files, analyse or verify their content, and produce a written
report.  Each factory is seeded so that different seeds yield different
(but deterministic) problem variants.
"""

import random as _random

from ..dataset import RubricDatapoint, RubricCategory, BinaryRubricCategory
from tinker_cookbook.rl.envs import tools
from ..content_pools import (
    make_name,
    make_names,
    pick,
    pick_one,
    vary_int,
    COMPANY_NAMES,
)

# =============================================================================
# DOMAIN: MEDICAL / TRIAGE
# =============================================================================

TRIAGE_SCENARIOS = [
    {
        "chief_complaint": "chest pain, shortness of breath",
        "age": 65, "sex": "Male",
        "vitals": {"bp": "158/95", "hr": 102, "rr": 22, "temp": "98.6\u00b0F", "spo2": "94%"},
        "history": "65-year-old male, history of hypertension and type 2 diabetes. Takes lisinopril 20mg daily and metformin 1000mg BID. Pain started 2 hours ago, described as pressure-like, radiating to left arm. Rated 7/10.",
        "red_flags": ["chest pain radiating to arm", "elevated BP with tachycardia", "low SpO2"],
        "priority": "ESI Level 2 \u2014 Emergent",
        "expected_actions": ["12-lead ECG stat", "troponin levels", "aspirin 325mg", "IV access", "cardiac monitoring"],
    },
    {
        "chief_complaint": "severe abdominal pain, vomiting",
        "age": 42, "sex": "Female",
        "vitals": {"bp": "110/70", "hr": 118, "rr": 20, "temp": "101.8\u00b0F", "spo2": "98%"},
        "history": "42-year-old female, no significant past medical history. Pain in right lower quadrant started 12 hours ago, initially periumbilical, now localized. Nausea and 3 episodes of vomiting. Last meal 8 hours ago. No bowel movement in 2 days.",
        "red_flags": ["fever with localized RLQ pain", "tachycardia", "migration pattern suggests appendicitis"],
        "priority": "ESI Level 2 \u2014 Emergent",
        "expected_actions": ["CBC with differential", "CMP", "urinalysis", "CT abdomen/pelvis with contrast", "NPO status", "IV fluids", "surgical consult"],
    },
    {
        "chief_complaint": "headache, confusion, neck stiffness",
        "age": 28, "sex": "Male",
        "vitals": {"bp": "132/88", "hr": 96, "rr": 18, "temp": "103.2\u00b0F", "spo2": "97%"},
        "history": "28-year-old male, previously healthy. Headache started yesterday, worsening. Roommate reports patient has been confused for the past 3 hours. Patient resists neck flexion. No recent travel. No known sick contacts. Photophobia noted.",
        "red_flags": ["high fever with neck stiffness", "altered mental status", "photophobia \u2014 meningitis triad"],
        "priority": "ESI Level 1 \u2014 Resuscitation",
        "expected_actions": ["blood cultures x2 stat", "empiric antibiotics immediately", "lumbar puncture", "CT head before LP if focal neuro signs", "IV dexamethasone"],
    },
    {
        "chief_complaint": "laceration to forearm",
        "age": 35, "sex": "Female",
        "vitals": {"bp": "128/82", "hr": 78, "rr": 16, "temp": "98.4\u00b0F", "spo2": "99%"},
        "history": "35-year-old female, cut forearm on broken glass while washing dishes. Wound is approximately 4 cm, superficial, bleeding controlled with direct pressure. Sensation intact distal to wound. Tetanus booster 3 years ago. No allergies.",
        "red_flags": [],
        "priority": "ESI Level 4 \u2014 Less Urgent",
        "expected_actions": ["wound irrigation", "laceration repair with sutures or adhesive", "tetanus not needed (within 5 years)", "wound care instructions"],
    },
    {
        "chief_complaint": "difficulty breathing, wheezing",
        "age": 55, "sex": "Male",
        "vitals": {"bp": "140/90", "hr": 110, "rr": 28, "temp": "98.8\u00b0F", "spo2": "91%"},
        "history": "55-year-old male, known asthma (moderate persistent), ran out of inhaler 3 days ago. Wheezing and dyspnea worsening over past 6 hours. Using accessory muscles. Can speak in short phrases only. History of 2 ICU admissions for asthma exacerbations.",
        "red_flags": ["SpO2 < 92%", "accessory muscle use", "speaking in phrases only", "history of ICU admissions"],
        "priority": "ESI Level 2 \u2014 Emergent",
        "expected_actions": ["continuous nebulized albuterol", "ipratropium bromide", "systemic corticosteroids", "magnesium sulfate if not responding", "continuous pulse oximetry", "ABG if worsening"],
    },
    # --- Additional scenarios for deeper seedability ---
    {
        "chief_complaint": "sudden weakness on right side, slurred speech",
        "age": 72, "sex": "Female",
        "vitals": {"bp": "188/102", "hr": 88, "rr": 18, "temp": "98.4\u00b0F", "spo2": "96%"},
        "history": "72-year-old female, history of atrial fibrillation on warfarin, hypertension. Spouse reports sudden onset right-sided weakness and difficulty speaking 45 minutes ago. Patient unable to raise right arm. Facial droop noted on right side.",
        "red_flags": ["acute focal neurological deficit", "hypertensive crisis (BP > 180)", "time-sensitive stroke window"],
        "priority": "ESI Level 1 \u2014 Resuscitation",
        "expected_actions": ["activate stroke code", "STAT CT head without contrast", "blood glucose check", "INR level (on warfarin)", "neurology consult", "establish IV access"],
    },
    {
        "chief_complaint": "high blood sugar, nausea, abdominal pain",
        "age": 19, "sex": "Male",
        "vitals": {"bp": "100/60", "hr": 122, "rr": 26, "temp": "99.0\u00b0F", "spo2": "98%"},
        "history": "19-year-old male, Type 1 diabetes diagnosed age 12, on insulin pump. Reports nausea and diffuse abdominal pain for 12 hours. Fingerstick glucose 485 mg/dL. Fruity odor on breath noted. Admits pump malfunctioned yesterday and missed several boluses. Last ate 18 hours ago.",
        "red_flags": ["Kussmaul respirations (RR 26)", "tachycardia with hypotension", "suspected DKA"],
        "priority": "ESI Level 2 \u2014 Emergent",
        "expected_actions": ["BMP stat (potassium critical)", "venous blood gas", "urinalysis for ketones", "IV normal saline bolus", "insulin drip per DKA protocol", "continuous cardiac monitoring"],
    },
    {
        "chief_complaint": "fall from ladder, back pain",
        "age": 48, "sex": "Male",
        "vitals": {"bp": "142/88", "hr": 92, "rr": 20, "temp": "98.6\u00b0F", "spo2": "99%"},
        "history": "48-year-old male, fell approximately 8 feet from a ladder onto concrete. Complaining of mid-thoracic back pain rated 8/10. Denies loss of consciousness. Sensation and motor intact in all extremities. No visible deformity. Ambulance immobilized on backboard with cervical collar.",
        "red_flags": ["fall from significant height (>6 feet)", "midline spinal tenderness"],
        "priority": "ESI Level 2 \u2014 Emergent",
        "expected_actions": ["maintain spinal immobilization", "CT spine (cervical, thoracic, lumbar)", "serial neuro checks", "pain management", "trauma surgery consult if fracture found"],
    },
    {
        "chief_complaint": "rash, swelling of lips and tongue",
        "age": 31, "sex": "Female",
        "vitals": {"bp": "92/58", "hr": 118, "rr": 24, "temp": "98.2\u00b0F", "spo2": "93%"},
        "history": "31-year-old female, ate shrimp at restaurant 30 minutes ago. Known shellfish allergy. Developed diffuse urticaria, then lip and tongue swelling. Voice becoming hoarse. Used personal EpiPen 10 minutes ago with partial improvement. No history of asthma. Takes no regular medications.",
        "red_flags": ["anaphylaxis with airway involvement", "hypotension despite epinephrine", "SpO2 < 94%"],
        "priority": "ESI Level 1 \u2014 Resuscitation",
        "expected_actions": ["repeat IM epinephrine", "IV normal saline wide open", "diphenhydramine IV", "methylprednisolone IV", "prepare for intubation if airway worsens", "continuous monitoring"],
    },
    {
        "chief_complaint": "chest pain with cough, leg swelling",
        "age": 38, "sex": "Female",
        "vitals": {"bp": "118/76", "hr": 108, "rr": 22, "temp": "99.2\u00b0F", "spo2": "92%"},
        "history": "38-year-old female, 6 weeks postpartum, on combined oral contraceptives. Pleuritic chest pain on right side for 2 days, worse with deep breathing. Noted left calf swelling and tenderness 5 days ago. Mild hemoptysis this morning. No prior clotting history. Otherwise healthy.",
        "red_flags": ["pleuritic chest pain with hemoptysis", "unilateral leg swelling", "SpO2 < 94%", "multiple PE risk factors (postpartum, OCP)"],
        "priority": "ESI Level 2 \u2014 Emergent",
        "expected_actions": ["CT pulmonary angiography", "D-dimer", "lower extremity duplex ultrasound", "heparin anticoagulation if PE confirmed", "CBC and coagulation studies"],
    },
    {
        "chief_complaint": "agitation, confusion, tremors",
        "age": 52, "sex": "Male",
        "vitals": {"bp": "162/98", "hr": 112, "rr": 20, "temp": "100.4\u00b0F", "spo2": "97%"},
        "history": "52-year-old male, brought by family. History of alcohol use disorder, drinks 12+ beers daily. Family reports he stopped drinking 2 days ago. Today became tremulous, diaphoretic, and increasingly confused. Seeing things that are not there (visual hallucinations). Last seizure history: none known. Takes no medications.",
        "red_flags": ["altered mental status", "tachycardia with hypertension", "suspected alcohol withdrawal / delirium tremens"],
        "priority": "ESI Level 2 \u2014 Emergent",
        "expected_actions": ["CIWA-Ar scoring", "IV benzodiazepines per protocol", "BMP (electrolytes, glucose, magnesium)", "thiamine 500mg IV before glucose", "continuous monitoring", "seizure precautions"],
    },
    {
        "chief_complaint": "sore throat, fever, difficulty swallowing",
        "age": 8, "sex": "Female",
        "vitals": {"bp": "100/65", "hr": 100, "rr": 20, "temp": "102.8\u00b0F", "spo2": "98%"},
        "history": "8-year-old female, 3-day history of sore throat and fever. Mother reports decreased oral intake for 24 hours. Bilateral tonsillar swelling with exudates noted. No drooling. No stridor. Able to handle secretions. No rash. Sick contact at school. Immunizations up to date. Weight 28 kg.",
        "red_flags": [],
        "priority": "ESI Level 3 \u2014 Urgent",
        "expected_actions": ["rapid strep test", "throat culture if rapid strep negative", "antipyretics (acetaminophen or ibuprofen weight-based)", "oral fluid challenge", "discharge with follow-up if tolerating fluids"],
    },
    {
        "chief_complaint": "ankle injury after twisting on stairs",
        "age": 24, "sex": "Male",
        "vitals": {"bp": "122/78", "hr": 82, "rr": 16, "temp": "98.6\u00b0F", "spo2": "99%"},
        "history": "24-year-old male, twisted left ankle going down stairs 1 hour ago. Significant swelling and bruising over lateral malleolus. Unable to bear weight. Point tenderness over distal fibula. Intact sensation and pulses distally. No other injuries. No prior ankle injuries. Takes no medications.",
        "red_flags": [],
        "priority": "ESI Level 4 \u2014 Less Urgent",
        "expected_actions": ["ankle X-ray (Ottawa rules met: unable to bear weight, bony tenderness)", "ice and elevation", "splint if fracture found", "crutches and weight-bearing instructions", "orthopedic follow-up if fracture"],
    },
]

# =============================================================================
# DOMAIN: ACCESSIBILITY (WCAG violations)
# =============================================================================

WCAG_VIOLATIONS = [
    {
        "name": "missing_alt_text",
        "html": '<img src="hero-banner.jpg" width="1200" height="400">',
        "fixed": '<img src="hero-banner.jpg" width="1200" height="400" alt="Company headquarters building at sunset">',
        "criterion": "WCAG 2.1 SC 1.1.1 (Non-text Content)",
        "description": "Image missing alt attribute",
    },
    {
        "name": "low_contrast_text",
        "html": '<p style="color: #999999; background-color: #ffffff;">Important notice about your account</p>',
        "fixed": '<p style="color: #595959; background-color: #ffffff;">Important notice about your account</p>',
        "criterion": "WCAG 2.1 SC 1.4.3 (Contrast Minimum)",
        "description": "Text color #999 on white background has contrast ratio 2.85:1, below 4.5:1 minimum",
    },
    {
        "name": "missing_form_label",
        "html": '<input type="text" name="email" placeholder="Enter email">',
        "fixed": '<label for="email">Email address</label><input type="text" id="email" name="email" placeholder="Enter email">',
        "criterion": "WCAG 2.1 SC 1.3.1 (Info and Relationships)",
        "description": "Form input has no associated label element",
    },
    {
        "name": "missing_lang_attribute",
        "html": "<html><head><title>My Page</title></head>",
        "fixed": '<html lang="en"><head><title>My Page</title></head>',
        "criterion": "WCAG 2.1 SC 3.1.1 (Language of Page)",
        "description": "HTML element missing lang attribute",
    },
    {
        "name": "keyboard_trap",
        "html": '<div tabindex="0" onkeydown="if(event.key===\'Tab\')event.preventDefault()">Interactive widget</div>',
        "fixed": '<div tabindex="0" role="group">Interactive widget</div>',
        "criterion": "WCAG 2.1 SC 2.1.2 (No Keyboard Trap)",
        "description": "Element prevents Tab key from moving focus away",
    },
    {
        "name": "auto_playing_video",
        "html": '<video autoplay src="promo.mp4"><source src="promo.mp4" type="video/mp4"></video>',
        "fixed": '<video src="promo.mp4" controls><source src="promo.mp4" type="video/mp4"></video>',
        "criterion": "WCAG 2.1 SC 1.4.2 (Audio Control)",
        "description": "Video autoplays without user control",
    },
    {
        "name": "empty_link",
        "html": '<a href="/dashboard"><i class="icon-home"></i></a>',
        "fixed": '<a href="/dashboard" aria-label="Dashboard home"><i class="icon-home"></i></a>',
        "criterion": "WCAG 2.1 SC 2.4.4 (Link Purpose)",
        "description": "Link has no text content or aria-label",
    },
    {
        "name": "missing_skip_nav",
        "html": "<body><nav>...50 navigation links...</nav><main>Content</main></body>",
        "fixed": '<body><a href="#main-content" class="skip-link">Skip to main content</a><nav>...50 navigation links...</nav><main id="main-content">Content</main></body>',
        "criterion": "WCAG 2.1 SC 2.4.1 (Bypass Blocks)",
        "description": "No skip navigation link to bypass repeated content",
    },
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _csv_row(values: list[str]) -> str:
    """Format a list of values as a single CSV row (no quoting needed for simple values)."""
    return ",".join(str(v) for v in values)


# ============================================================================
# 1. Scientific claim verification
# ============================================================================

def make_scientific_claim_verification(rand_seed: int = 42) -> RubricDatapoint:
    """Given 5-6 scientific claims and a dataset CSV, verify which are supported.

    Problem type: claim_verification.
    """
    rng = _random.Random(rand_seed)

    # --- Pick scenario parameters ------------------------------------------------
    company = pick_one(COMPANY_NAMES, rand_seed)
    researcher = make_name(rand_seed)

    groups = ["Treatment A", "Treatment B", "Control"]
    measurements = ["blood_pressure", "cholesterol", "glucose", "heart_rate"]
    n_rows = vary_int(100, rand_seed)

    # --- Build dataset -----------------------------------------------------------
    header = "subject_id,group,age,sex,blood_pressure,cholesterol,glucose,heart_rate,outcome"
    rows = [header]

    # Track aggregates for claim generation
    group_data: dict[str, dict[str, list[float]]] = {g: {m: [] for m in measurements} for g in groups}
    group_outcomes: dict[str, list[int]] = {g: [] for g in groups}
    ages: list[int] = []

    for i in range(1, n_rows + 1):
        group = rng.choice(groups)
        age = rng.randint(25, 75)
        sex = rng.choice(["M", "F"])
        ages.append(age)

        # Generate correlated values so claims can be true/false
        bp = round(rng.gauss({"Treatment A": 120, "Treatment B": 130, "Control": 138}[group], 12), 1)
        chol = round(rng.gauss({"Treatment A": 190, "Treatment B": 180, "Control": 215}[group], 20), 1)
        gluc = round(rng.gauss({"Treatment A": 95, "Treatment B": 100, "Control": 105}[group], 10), 1)
        hr = round(rng.gauss(72, 8), 1)

        # Outcome: 1 = improved, 0 = not improved
        improve_prob = {"Treatment A": 0.70, "Treatment B": 0.55, "Control": 0.35}[group]
        outcome = 1 if rng.random() < improve_prob else 0

        group_data[group]["blood_pressure"].append(bp)
        group_data[group]["cholesterol"].append(chol)
        group_data[group]["glucose"].append(gluc)
        group_data[group]["heart_rate"].append(hr)
        group_outcomes[group].append(outcome)

        rows.append(f"{i},{group},{age},{sex},{bp},{chol},{gluc},{hr},{outcome}")

    dataset_csv = "\n".join(rows) + "\n"

    # --- Compute ground-truth stats for claim construction -----------------------
    def _mean(vals: list[float]) -> float:
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    mean_bp_a = _mean(group_data["Treatment A"]["blood_pressure"])
    mean_bp_ctrl = _mean(group_data["Control"]["blood_pressure"])
    mean_chol_b = _mean(group_data["Treatment B"]["cholesterol"])
    mean_chol_ctrl = _mean(group_data["Control"]["cholesterol"])
    improve_rate_a = _mean([float(x) for x in group_outcomes["Treatment A"]])
    improve_rate_ctrl = _mean([float(x) for x in group_outcomes["Control"]])
    mean_gluc_a = _mean(group_data["Treatment A"]["glucose"])
    mean_hr_a = _mean(group_data["Treatment A"]["heart_rate"])
    mean_hr_ctrl = _mean(group_data["Control"]["heart_rate"])
    total_subjects = n_rows

    # --- Build 6 claims with truth computed from actual sample data ---------------
    claim1_true = mean_bp_a < mean_bp_ctrl
    claim2_true = improve_rate_a > improve_rate_ctrl
    claim3_true = total_subjects >= total_subjects - 5  # Always true by construction
    claim4_true = mean_chol_b > mean_chol_ctrl  # Pop means 180 vs 215 — almost always False
    claim5_true = mean_hr_a < 65  # Pop mean 72, so almost always False
    claim6_true = mean_gluc_a < 80  # Pop mean 95, so always False

    claims = [
        (
            claim1_true,
            f"Claim 1: Treatment A subjects had lower mean blood pressure than the Control group.",
            f"Mean BP for Treatment A ({mean_bp_a:.1f}) vs Control ({mean_bp_ctrl:.1f})",
        ),
        (
            claim2_true,
            f"Claim 2: The improvement rate for Treatment A was higher than for the Control group.",
            f"Improvement rate Treatment A ({improve_rate_a:.2%}) vs Control ({improve_rate_ctrl:.2%})",
        ),
        (
            claim3_true,
            f"Claim 3: The study enrolled at least {total_subjects - 5} subjects in total.",
            f"Total subjects: {total_subjects}",
        ),
        (
            claim4_true,
            f"Claim 4: Treatment B subjects had higher average cholesterol than the Control group.",
            f"Mean cholesterol Treatment B ({mean_chol_b:.1f}) vs Control ({mean_chol_ctrl:.1f})",
        ),
        (
            claim5_true,
            f"Claim 5: The average heart rate in Treatment A was below 65 bpm.",
            f"Mean heart rate Treatment A: {mean_hr_a:.1f}",
        ),
        (
            claim6_true,
            f"Claim 6: Treatment A reduced glucose levels below 80 mg/dL on average.",
            f"Mean glucose for Treatment A was {mean_gluc_a:.1f}",
        ),
    ]

    claims_text = (
        f"Scientific Claims for {company} Clinical Trial Dataset\n"
        f"Prepared by: {researcher}\n\n"
        "Evaluate each claim against the provided dataset:\n\n"
    )
    for _, claim_text, _ in claims:
        claims_text += f"  {claim_text}\n"

    # --- Rubric ------------------------------------------------------------------
    rubric_cats: list[BinaryRubricCategory | RubricCategory] = []

    for is_true, claim_text, _evidence in claims:
        label = "supported" if is_true else "unsupported"
        rubric_cats.append(BinaryRubricCategory(
            name=f"claim_{claims.index((is_true, claim_text, _evidence)) + 1}_correct",
            question=f"Does the report correctly identify {claim_text.split(':')[0]} as {label} by the data?",
            points=2,
        ))

    # Graded categories
    rubric_cats.append(RubricCategory(
        name="evidence_quality",
        description="Does the report cite specific numeric evidence from the dataset to justify each determination?",
        failure="No numeric evidence cited; just restates the claims.",
        minor_failure="Some numbers cited but many claims lack supporting calculations.",
        minor_success="Most claims are supported by data references, with minor gaps.",
        success="Every claim determination is backed by specific computed statistics from the dataset.",
        points=3,
    ))
    rubric_cats.append(RubricCategory(
        name="methodology",
        description="Does the report describe how the data was analysed (e.g. computed group means, compared rates)?",
        failure="No methodology described.",
        minor_failure="Vague mention of looking at the data without specifics.",
        minor_success="Brief description of approach (e.g. 'computed means per group').",
        success="Clear, reproducible methodology section explaining how each claim was evaluated.",
        points=3,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="report_structure",
        question="Does the report address each claim individually in a clearly separated section or numbered list?",
        points=1,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="summary_present",
        question="Does the report include a summary or conclusion stating the overall number of supported vs unsupported claims?",
        points=1,
    ))
    rubric_cats.append(RubricCategory(
        name="writing_clarity",
        description="Is the verification report clearly written and accessible to a non-specialist reader?",
        failure="Incomprehensible or extremely terse output.",
        minor_failure="Understandable but disorganised or overly technical without explanation.",
        minor_success="Generally clear with occasional jargon or awkward phrasing.",
        success="Well-written, concise, and accessible; a non-specialist could follow the reasoning.",
        points=2,
    ))

    # 8 binary + 3 graded = 11 cats; binary 8/11 = 73%

    return RubricDatapoint(
        problem_statement=(
            f"# Scientific Claim Verification\n\n"
            f"You are a data analyst at {company}. A colleague ({researcher}) has made six scientific\n"
            f"claims about a clinical trial dataset. Your task is to verify each claim by analysing\n"
            f"the raw data in `/testbed/data/dataset.csv`.\n\n"
            f"The claims are listed in `/testbed/data/claims.txt`.\n\n"
            f"For each claim, determine whether it is **supported** or **unsupported** by the data.\n"
            f"Show your evidence (computed statistics) and explain your reasoning.\n\n"
            f"Write your verification report to `/testbed/verification_report.txt`."
        ),
        rubric=tuple(rubric_cats),
        submission_instructions="Write your verification report to /testbed/verification_report.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/dataset.csv": dataset_csv,
            "/testbed/data/claims.txt": claims_text,
        },
        problem_type="claim_verification",
    )


# ============================================================================
# 2. Statistical report review
# ============================================================================

def make_statistical_report_review(rand_seed: int = 42) -> RubricDatapoint:
    """Given a report with statistical claims and raw data CSV, find errors.

    Problem type: statistical_review.
    """
    rng = _random.Random(rand_seed)

    company = pick_one(COMPANY_NAMES, rand_seed + 1)
    analyst = make_name(rand_seed + 1)
    reviewer = make_name(rand_seed + 2)

    # --- Generate raw data -------------------------------------------------------
    departments = ["Engineering", "Sales", "Marketing", "Operations", "Finance"]
    n_rows = vary_int(120, rand_seed)

    header = "employee_id,department,years_experience,salary,performance_score,satisfaction_rating"
    rows = [header]

    dept_salaries: dict[str, list[float]] = {d: [] for d in departments}
    dept_scores: dict[str, list[float]] = {d: [] for d in departments}
    all_salaries: list[float] = []
    all_experience: list[float] = []
    all_satisfaction: list[float] = []

    for i in range(1, n_rows + 1):
        dept = rng.choice(departments)
        yrs = rng.randint(1, 25)
        base = {"Engineering": 95000, "Sales": 78000, "Marketing": 72000,
                "Operations": 68000, "Finance": 85000}[dept]
        salary = round(base + yrs * rng.uniform(1500, 3500) + rng.gauss(0, 5000), 2)
        perf = round(min(5.0, max(1.0, rng.gauss(3.5, 0.8))), 1)
        sat = rng.randint(1, 10)

        dept_salaries[dept].append(salary)
        dept_scores[dept].append(perf)
        all_salaries.append(salary)
        all_experience.append(yrs)
        all_satisfaction.append(sat)
        rows.append(f"E{i:04d},{dept},{yrs},{salary:.2f},{perf},{sat}")

    raw_csv = "\n".join(rows) + "\n"

    # --- Compute correct statistics -----------------------------------------------
    def _mean(v: list[float]) -> float:
        return sum(v) / len(v) if v else 0.0

    def _median(v: list[float]) -> float:
        s = sorted(v)
        n = len(s)
        if n % 2 == 1:
            return s[n // 2]
        return (s[n // 2 - 1] + s[n // 2]) / 2

    correct_mean_salary = _mean(all_salaries)
    correct_median_salary = _median(all_salaries)
    correct_mean_satisfaction = _mean(all_satisfaction)
    correct_count = n_rows

    eng_mean = _mean(dept_salaries["Engineering"])
    sales_mean = _mean(dept_salaries["Sales"])
    highest_dept = max(dept_salaries, key=lambda d: _mean(dept_salaries[d]))
    lowest_dept = min(dept_salaries, key=lambda d: _mean(dept_salaries[d]))

    # --- Build the "draft report" with deliberate errors -------------------------
    # Error 1: wrong overall mean salary (off by ~10%)
    wrong_mean_salary = round(correct_mean_salary * rng.uniform(1.08, 1.15), 2)
    # Error 2: wrong median salary
    wrong_median_salary = round(correct_median_salary * rng.uniform(0.82, 0.90), 2)
    # Error 3: wrong employee count
    wrong_count = correct_count + rng.randint(8, 20)
    # Error 4: wrong highest-paid department
    wrong_highest_candidates = [d for d in departments if d != highest_dept]
    wrong_highest = rng.choice(wrong_highest_candidates)

    draft_report = f"""Workforce Analytics Report — {company}
Prepared by: {analyst}
Date: 2024-Q3

EXECUTIVE SUMMARY
This report presents key workforce statistics based on employee data
collected across all five departments.

KEY FINDINGS

1. Total Employees: {wrong_count}
   The company currently employs {wrong_count} staff across five departments.

2. Average Salary: ${wrong_mean_salary:,.2f}
   The overall mean salary across all departments is ${wrong_mean_salary:,.2f}.

3. Median Salary: ${wrong_median_salary:,.2f}
   The median salary is ${wrong_median_salary:,.2f}, indicating the central
   tendency of compensation.

4. Highest-Paid Department: {wrong_highest}
   {wrong_highest} has the highest average salary among all departments.

5. Average Satisfaction Rating: {correct_mean_satisfaction:.1f} / 10
   Employee satisfaction averages {correct_mean_satisfaction:.1f} out of 10.

6. Engineering vs Sales Salary Gap: ${abs(eng_mean - sales_mean):,.2f}
   Engineering staff earn on average ${abs(eng_mean - sales_mean):,.2f} more
   than Sales staff.

METHODOLOGY
All statistics were computed from the raw employee database export.
"""

    # --- Rubric ------------------------------------------------------------------
    rubric_cats: list[BinaryRubricCategory | RubricCategory] = []

    # Binary: identify each error
    rubric_cats.append(BinaryRubricCategory(
        name="error_total_count",
        question=f"Does the review identify that the reported total employee count ({wrong_count}) is incorrect?",
        points=2,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="correct_total_count",
        question=f"Does the review state that the correct total employee count is {correct_count}?",
        points=1,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="error_mean_salary",
        question=f"Does the review identify that the reported mean salary (${wrong_mean_salary:,.2f}) is incorrect?",
        points=2,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="correct_mean_salary",
        question=f"Does the review state a correct (or very close) mean salary near ${correct_mean_salary:,.2f}?",
        points=1,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="error_median_salary",
        question=f"Does the review identify that the reported median salary (${wrong_median_salary:,.2f}) is incorrect?",
        points=2,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="correct_median_salary",
        question=f"Does the review state a correct (or very close) median salary near ${correct_median_salary:,.2f}?",
        points=1,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="error_highest_dept",
        question=f"Does the review identify that {wrong_highest} is NOT the highest-paid department?",
        points=2,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="correct_highest_dept",
        question=f"Does the review state that {highest_dept} is actually the highest-paid department?",
        points=1,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="correct_stat_5_confirmed",
        question="Does the review confirm that statistic #5 (satisfaction) is approximately correct?",
        points=1,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="correct_stat_6_confirmed",
        question="Does the review confirm that statistic #6 (salary gap) is approximately correct?",
        points=1,
    ))

    # Graded
    rubric_cats.append(RubricCategory(
        name="thoroughness",
        description="Does the review systematically check every statistic in the report, not just some?",
        failure="Only one or two statistics examined.",
        minor_failure="Several statistics checked but at least two are skipped entirely.",
        minor_success="Most statistics checked with minor omissions.",
        success="All six statistics in the report are individually verified against the raw data.",
        points=3,
    ))
    rubric_cats.append(RubricCategory(
        name="clarity",
        description="Is the review clearly written with specific corrections and explanations?",
        failure="Vague or incomprehensible feedback.",
        minor_failure="Identifies errors but does not explain how the correct value was derived.",
        minor_success="Mostly clear with some vague spots.",
        success="Each error is clearly described with the wrong value, correct value, and method of verification.",
        points=3,
    ))

    rubric_cats.append(RubricCategory(
        name="methodology_description",
        description="Does the review describe the methodology used to verify each statistic?",
        failure="No mention of how values were checked.",
        minor_failure="Brief mention of 'checking the data' without specifics.",
        minor_success="Describes the general approach (e.g. 'computed from CSV') for most statistics.",
        success="Each verified statistic includes a clear description of the computation method (e.g. 'summed all rows', 'sorted and found midpoint').",
        points=2,
    ))

    # 9 binary + 3 graded = 12 cats, binary fraction = 9/12 = 75%

    return RubricDatapoint(
        problem_statement=(
            f"# Statistical Report Review\n\n"
            f"You are {reviewer}, a senior analyst at {company}. A colleague ({analyst})\n"
            f"has prepared a draft workforce analytics report. Your job is to review it\n"
            f"by cross-checking every statistic against the raw data.\n\n"
            f"Files:\n"
            f"- Raw data: `/testbed/data/raw_data.csv`\n"
            f"- Draft report: `/testbed/reports/draft_report.txt`\n\n"
            f"For each statistic in the report, verify whether it is correct. If it is wrong,\n"
            f"state the reported value, the correct value, and how you computed it.\n\n"
            f"Write your review to `/testbed/review.txt`."
        ),
        rubric=tuple(rubric_cats),
        submission_instructions="Write your review to /testbed/review.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/raw_data.csv": raw_csv,
            "/testbed/reports/draft_report.txt": draft_report,
        },
        problem_type="statistical_review",
    )


# ============================================================================
# 3. Resume screening
# ============================================================================

RESUME_ROLES = [
    {
        "title": "Senior Data Engineer",
        "salary_range": "$130,000 - $180,000",
        "requirements": [
            {
                "text": "5+ years of Python and SQL development",
                "area": "Python and SQL",
                "strong": "Developed Python data processing modules and complex SQL analytics queries across enterprise data warehouse",
                "weak": "Completed introductory Python course; used SQL for basic SELECT queries in student database projects",
            },
            {
                "text": "Experience building and maintaining ETL/ELT pipelines",
                "area": "ETL/ELT pipelines",
                "strong": "Built automated data ingestion pipeline processing 8M records daily from 15 source systems with schema validation",
                "weak": "Exported monthly CSV reports from internal database for management review",
            },
            {
                "text": "Proficiency with cloud platforms (AWS, GCP, or Azure)",
                "area": "cloud platforms",
                "strong": "Managed AWS data infrastructure including S3 data lake, Glue jobs, Lambda functions, and CloudWatch alerts",
                "weak": "Hosted personal portfolio website on AWS free-tier EC2 instance",
            },
            {
                "text": "Experience with data warehouse solutions (Snowflake, BigQuery, or Redshift)",
                "area": "data warehouse solutions",
                "strong": "Led migration from on-premises Oracle to Snowflake, designing star schema models for 50+ dashboards",
                "weak": "Queried MySQL application database for ad-hoc customer support lookups",
            },
            {
                "text": "Strong communication skills for cross-functional collaboration",
                "area": "cross-functional communication",
                "strong": "Partnered with product, analytics, and engineering teams to define data contracts and delivery SLAs",
                "weak": "Communicated primarily with direct manager via weekly status emails",
            },
        ],
        "nice_to_have": [
            {"text": "Spark or Databricks experience", "evidence": "Built Spark jobs on Databricks processing 10TB+ daily datasets"},
            {"text": "dbt proficiency", "evidence": "Maintained 150+ dbt models powering analytics transformation layer"},
            {"text": "CI/CD pipeline experience", "evidence": "Implemented GitHub Actions CI/CD for data infrastructure deployments"},
        ],
        "titles_strong": ["Data Platform Engineer", "Senior Analytics Engineer", "Data Engineer II"],
        "titles_mid": ["Software Developer", "Database Developer", "Junior Data Analyst"],
        "titles_weak": ["IT Help Desk Analyst", "Administrative Coordinator", "Marketing Associate"],
        "degrees_strong": ["B.S. Computer Science", "B.S. Data Science", "M.S. Computer Science"],
        "degrees_weak": ["B.A. Communications", "B.S. Biology", "B.A. History"],
    },
    {
        "title": "Product Manager \u2014 Growth",
        "salary_range": "$140,000 - $190,000",
        "requirements": [
            {
                "text": "4+ years product management experience",
                "area": "product management",
                "strong": "Owned product roadmap for growth squad, shipping 12 features that drove 40% user increase over 18 months",
                "weak": "Assisted product team with meeting notes and user story documentation as business analyst intern",
            },
            {
                "text": "Demonstrated ability to drive user acquisition or engagement metrics",
                "area": "user acquisition and engagement metrics",
                "strong": "Designed user activation funnel improvements increasing Day-7 retention from 22% to 31%",
                "weak": "Tracked basic website visitor counts using Google Analytics for personal blog",
            },
            {
                "text": "Proficiency with A/B testing and data-driven decision making",
                "area": "A/B testing",
                "strong": "Ran 30+ A/B experiments per quarter on internal platform, establishing statistical rigor guidelines for the team",
                "weak": "Familiar with the concept of A/B testing from product management coursework",
            },
            {
                "text": "Experience with agile/scrum methodologies",
                "area": "agile/scrum methodologies",
                "strong": "Served as product owner for 2 scrum teams, managing sprint planning, grooming, and retrospectives",
                "weak": "Participated in daily standup meetings as a team member",
            },
            {
                "text": "Excellent written and verbal communication",
                "area": "written and verbal communication",
                "strong": "Authored monthly product strategy memos for C-suite; presented quarterly business reviews to board",
                "weak": "Wrote internal emails and updated project status spreadsheets",
            },
        ],
        "nice_to_have": [
            {"text": "SQL proficiency", "evidence": "Wrote complex SQL queries for user cohort analysis and conversion funnel metrics"},
            {"text": "SaaS or marketplace products experience", "evidence": "Managed marketplace product serving 50K+ active sellers across 3 regions"},
            {"text": "MBA or equivalent", "evidence": "Completed MBA with concentration in technology management"},
        ],
        "titles_strong": ["Product Lead \u2014 Growth", "Senior Product Manager", "Growth Product Manager"],
        "titles_mid": ["Associate Product Manager", "Business Analyst", "Project Coordinator"],
        "titles_weak": ["Retail Store Manager", "Customer Service Representative", "Accounting Clerk"],
        "degrees_strong": ["B.S. Computer Science", "B.A. Economics", "MBA"],
        "degrees_weak": ["B.A. Fine Arts", "B.S. Chemistry", "B.A. Philosophy"],
    },
    {
        "title": "DevOps Engineer",
        "salary_range": "$125,000 - $175,000",
        "requirements": [
            {
                "text": "3+ years experience with containerization (Docker, Kubernetes)",
                "area": "containerization (Docker/Kubernetes)",
                "strong": "Designed Kubernetes cluster architecture serving 200+ microservices across staging and production environments",
                "weak": "Used Docker to run local development environment following team setup guide",
            },
            {
                "text": "Proficiency with infrastructure-as-code (Terraform or CloudFormation)",
                "area": "infrastructure-as-code",
                "strong": "Managed 500+ Terraform resources across multiple AWS accounts with automated drift detection",
                "weak": "Edited existing CloudFormation templates to update instance types as directed by senior engineer",
            },
            {
                "text": "Experience managing CI/CD pipelines (Jenkins, GitLab CI, or GitHub Actions)",
                "area": "CI/CD pipeline management",
                "strong": "Built company-wide CI/CD platform on GitLab CI serving 80+ repositories with standardized templates",
                "weak": "Triggered Jenkins builds manually and monitored build status dashboard",
            },
            {
                "text": "Strong Linux systems administration skills",
                "area": "Linux systems administration",
                "strong": "Administered fleet of 200+ Ubuntu servers: automated patching, performance tuning, and capacity planning",
                "weak": "Installed Ubuntu on personal laptop; familiar with basic terminal commands like ls and cd",
            },
            {
                "text": "On-call experience and incident response",
                "area": "on-call and incident response",
                "strong": "Led incident response for production outages as primary on-call; reduced MTTR from 45min to 15min",
                "weak": "Observed senior engineers handle production issues during onboarding shadowing period",
            },
        ],
        "nice_to_have": [
            {"text": "Security certifications", "evidence": "Holds AWS Solutions Architect Professional and Certified Kubernetes Administrator certifications"},
            {"text": "Monitoring/observability tools (Datadog, Grafana)", "evidence": "Implemented Datadog monitoring stack with custom dashboards and PagerDuty alert routing"},
            {"text": "Scripting in Python or Go", "evidence": "Developed Go-based CLI tools for infrastructure automation used by entire platform team"},
        ],
        "titles_strong": ["Infrastructure Engineer", "Platform Engineer", "Site Reliability Engineer"],
        "titles_mid": ["Junior Systems Administrator", "IT Technician", "Network Support Analyst"],
        "titles_weak": ["Graphic Designer", "Sales Development Representative", "Event Planner"],
        "degrees_strong": ["B.S. Computer Science", "B.S. Computer Engineering", "B.S. Information Technology"],
        "degrees_weak": ["B.A. Sociology", "B.S. Kinesiology", "B.A. Music"],
    },
    {
        "title": "UX Researcher",
        "salary_range": "$110,000 - $155,000",
        "requirements": [
            {
                "text": "3+ years of UX research experience",
                "area": "UX research",
                "strong": "Conducted 150+ user research sessions across moderated interviews, surveys, and usability testing",
                "weak": "Read UX research articles and completed 2-week online UX design workshop",
            },
            {
                "text": "Qualitative research methods (interviews, focus groups, contextual inquiry)",
                "area": "qualitative research methods",
                "strong": "Designed and facilitated 8-participant focus groups with structured discussion guides and affinity mapping",
                "weak": "Conducted informal conversations with friends about app preferences for class presentation",
            },
            {
                "text": "Quantitative methods (surveys, analytics, statistical analysis)",
                "area": "quantitative research methods",
                "strong": "Designed survey instruments with validated scales; analyzed results with statistical significance testing in R",
                "weak": "Created simple Google Forms survey for class project data collection",
            },
            {
                "text": "Research synthesis and stakeholder presentation",
                "area": "research synthesis and presentation",
                "strong": "Delivered insights to VP Product through structured research reports, video highlight reels, and journey maps",
                "weak": "Summarized meeting notes for team distribution via email",
            },
            {
                "text": "Collaboration with product and design teams",
                "area": "product/design team collaboration",
                "strong": "Embedded with 3 product squads as dedicated researcher, influencing roadmap priorities with user evidence",
                "weak": "Attended product team meetings and shared occasional observations about features",
            },
        ],
        "nice_to_have": [
            {"text": "Accessibility research experience", "evidence": "Conducted sessions with assistive technology users and developed accessibility audit protocols"},
            {"text": "Research operations", "evidence": "Built research participant panel of 2,000+ users with automated recruitment workflows"},
            {"text": "Prototyping tools (Figma)", "evidence": "Created interactive Figma prototypes for usability testing sessions"},
        ],
        "titles_strong": ["UX Researcher II", "Senior Design Researcher", "User Research Lead"],
        "titles_mid": ["Junior UX Designer", "Research Assistant", "Product Analyst"],
        "titles_weak": ["Data Entry Specialist", "Warehouse Associate", "Receptionist"],
        "degrees_strong": ["M.A. Human-Computer Interaction", "B.S. Psychology", "B.A. Cognitive Science"],
        "degrees_weak": ["B.S. Mechanical Engineering", "B.A. Theatre", "B.S. Agriculture"],
    },
    {
        "title": "Backend Software Engineer",
        "salary_range": "$135,000 - $185,000",
        "requirements": [
            {
                "text": "4+ years of backend development experience",
                "area": "backend development",
                "strong": "Designed and maintained backend services handling 50K requests/second across distributed microservice architecture",
                "weak": "Completed backend development bootcamp; built simple CRUD application as capstone project",
            },
            {
                "text": "API design and RESTful service development",
                "area": "API design",
                "strong": "Designed RESTful API layer with 200+ endpoints, versioning strategy, and OpenAPI documentation",
                "weak": "Consumed third-party weather API to display data in personal project",
            },
            {
                "text": "Database design (relational and NoSQL)",
                "area": "database design",
                "strong": "Designed PostgreSQL and DynamoDB schemas supporting multi-tenant SaaS platform with row-level security",
                "weak": "Used SQLite for local storage in mobile app course project",
            },
            {
                "text": "System design and scalability experience",
                "area": "system design and scalability",
                "strong": "Led architecture review for payment system scaling from 10K to 500K daily transactions",
                "weak": "Read 'Designing Data-Intensive Applications'; no production scaling experience",
            },
            {
                "text": "Testing practices and code quality standards",
                "area": "testing and code quality",
                "strong": "Established testing standards: 90%+ coverage, property-based testing, and automated integration suite",
                "weak": "Wrote basic unit tests when required by course assignments",
            },
        ],
        "nice_to_have": [
            {"text": "Distributed systems experience", "evidence": "Implemented distributed caching layer with Redis and consistent hashing across service fleet"},
            {"text": "Message queues (Kafka, RabbitMQ)", "evidence": "Designed event-driven architecture on Kafka processing 1M+ events per hour"},
            {"text": "Performance optimization", "evidence": "Profiled and optimized critical endpoints reducing P99 latency from 800ms to 120ms"},
        ],
        "titles_strong": ["Software Engineer II", "Senior Backend Developer", "Platform Engineer"],
        "titles_mid": ["Junior Developer", "QA Engineer", "Technical Support Analyst"],
        "titles_weak": ["Office Administrator", "Logistics Coordinator", "Barista"],
        "degrees_strong": ["B.S. Computer Science", "B.S. Software Engineering", "M.S. Computer Science"],
        "degrees_weak": ["B.A. Political Science", "B.S. Nursing", "B.A. English Literature"],
    },
    {
        "title": "Information Security Analyst",
        "salary_range": "$120,000 - $165,000",
        "requirements": [
            {
                "text": "3+ years of security monitoring and analysis experience",
                "area": "security monitoring and analysis",
                "strong": "Monitored enterprise SIEM processing 50K+ events/hour, triaging alerts and escalating confirmed incidents",
                "weak": "Completed cybersecurity awareness training as part of standard employee onboarding",
            },
            {
                "text": "Vulnerability assessment and penetration testing",
                "area": "vulnerability assessment",
                "strong": "Conducted quarterly vulnerability scans across 500+ hosts using Nessus and Qualys, prioritizing remediation plans",
                "weak": "Ran Nmap port scan in college networking lab exercise",
            },
            {
                "text": "Security frameworks knowledge (NIST, ISO 27001, SOC 2)",
                "area": "security frameworks (NIST/ISO/SOC 2)",
                "strong": "Led SOC 2 Type II audit preparation, developing control documentation and evidence collection procedures",
                "weak": "Aware of NIST framework from introductory cybersecurity lecture",
            },
            {
                "text": "Incident response and digital forensics",
                "area": "incident response",
                "strong": "Managed response for 20+ security events including phishing campaigns, lateral movement, and ransomware attempts",
                "weak": "Reset a colleague's password after they reported a suspicious email",
            },
            {
                "text": "Security tools (firewalls, IDS/IPS, endpoint protection)",
                "area": "security tooling",
                "strong": "Configured Palo Alto firewalls, CrowdStrike EDR, and Snort IDS across enterprise network segments",
                "weak": "Installed consumer antivirus software on personal computer",
            },
        ],
        "nice_to_have": [
            {"text": "Cloud security experience", "evidence": "Implemented AWS GuardDuty, Security Hub, and IAM policies across 15 accounts"},
            {"text": "Security certifications (CISSP, CEH)", "evidence": "Holds CISSP and CompTIA Security+ certifications"},
            {"text": "Scripting for security automation", "evidence": "Developed Python scripts for threat intelligence feed processing and automated alert enrichment"},
        ],
        "titles_strong": ["Security Operations Analyst", "Cybersecurity Engineer", "SOC Analyst II"],
        "titles_mid": ["IT Support Specialist", "Network Technician", "Help Desk Analyst"],
        "titles_weak": ["Retail Sales Associate", "Restaurant Manager", "Real Estate Agent"],
        "degrees_strong": ["B.S. Cybersecurity", "B.S. Computer Science", "B.S. Information Assurance"],
        "degrees_weak": ["B.A. Art History", "B.S. Marine Biology", "B.A. Anthropology"],
    },
    {
        "title": "Machine Learning Engineer",
        "salary_range": "$150,000 - $200,000",
        "requirements": [
            {
                "text": "3+ years of ML model development and production deployment",
                "area": "ML model development and deployment",
                "strong": "Developed and deployed 15+ production ML models for recommendation, fraud detection, and demand forecasting",
                "weak": "Completed Coursera ML specialization; trained MNIST classifier as homework assignment",
            },
            {
                "text": "Python ML ecosystem (PyTorch, TensorFlow, scikit-learn)",
                "area": "Python ML frameworks",
                "strong": "Built custom PyTorch training pipelines with distributed training across multi-GPU clusters",
                "weak": "Followed scikit-learn tutorial to fit linear regression model on sample dataset",
            },
            {
                "text": "Data pipeline and feature engineering experience",
                "area": "feature engineering and data pipelines",
                "strong": "Designed feature store serving 500+ features with real-time computation and versioned offline backfills",
                "weak": "Cleaned CSV files using pandas for data analysis class project",
            },
            {
                "text": "ML infrastructure (model serving, monitoring, A/B testing)",
                "area": "ML infrastructure",
                "strong": "Built model serving platform on Kubernetes with A/B testing, canary deployments, and drift detection",
                "weak": "Exported trained model to pickle file for local testing",
            },
            {
                "text": "Statistical analysis and experiment design",
                "area": "statistical analysis and experiment design",
                "strong": "Designed A/B experiments for model rollouts with power analysis, sequential testing, and guardrail metrics",
                "weak": "Calculated descriptive statistics in Excel for undergraduate research paper",
            },
        ],
        "nice_to_have": [
            {"text": "NLP/LLM experience", "evidence": "Fine-tuned transformer models for entity extraction, achieving 94% F1 on domain-specific corpus"},
            {"text": "MLOps tools (MLflow, Weights & Biases)", "evidence": "Managed experiment tracking and model registry using MLflow across 10-person ML team"},
            {"text": "Distributed computing", "evidence": "Implemented distributed training on Ray cluster across 8 GPU nodes for large-scale model experiments"},
        ],
        "titles_strong": ["ML Engineer II", "Applied Scientist", "Senior Data Scientist"],
        "titles_mid": ["Data Analyst", "Research Assistant", "Junior Software Developer"],
        "titles_weak": ["Dental Hygienist", "Supply Chain Coordinator", "Insurance Adjuster"],
        "degrees_strong": ["M.S. Machine Learning", "M.S. Computer Science", "Ph.D. Statistics"],
        "degrees_weak": ["B.A. Journalism", "B.S. Civil Engineering", "B.A. Education"],
    },
    {
        "title": "Technical Program Manager",
        "salary_range": "$140,000 - $185,000",
        "requirements": [
            {
                "text": "5+ years of technical program management experience",
                "area": "technical program management",
                "strong": "Managed 8 concurrent technical programs with $15M combined budget across 4 engineering teams",
                "weak": "Coordinated small project timelines using shared spreadsheet for 3-person team",
            },
            {
                "text": "Cross-team coordination and stakeholder management",
                "area": "cross-team coordination",
                "strong": "Served as coordination point across infrastructure, product, and platform teams for company-wide migration",
                "weak": "Relayed messages between manager and team members during weekly status meetings",
            },
            {
                "text": "Risk identification and mitigation planning",
                "area": "risk management",
                "strong": "Developed risk framework adopted across engineering org; identified and mitigated 30+ program-level risks",
                "weak": "Listed potential risks in project proposal as required by course syllabus template",
            },
            {
                "text": "Technical background sufficient to engage in engineering discussions",
                "area": "technical background",
                "strong": "Former software engineer with 3 years hands-on development; reviews technical design documents and architecture proposals",
                "weak": "Attended introductory webinar on basic programming concepts",
            },
            {
                "text": "Process improvement and engineering tooling",
                "area": "process improvement",
                "strong": "Redesigned sprint planning and release process, reducing average time-to-ship from 6 weeks to 2 weeks",
                "weak": "Created shared Google Drive folder structure for team document organization",
            },
        ],
        "nice_to_have": [
            {"text": "PMP or Agile certification", "evidence": "PMP certified; also holds Certified Scrum Master (CSM) credential"},
            {"text": "Data analysis and dashboarding", "evidence": "Built executive Tableau dashboards tracking program health metrics across 20+ initiatives"},
            {"text": "Vendor management experience", "evidence": "Managed 5 external vendor relationships including contract negotiation and SLA enforcement"},
        ],
        "titles_strong": ["Technical Program Manager", "Engineering Program Manager", "Senior TPM"],
        "titles_mid": ["Project Coordinator", "Business Operations Analyst", "Scrum Master"],
        "titles_weak": ["Cashier", "Landscaping Technician", "Fitness Instructor"],
        "degrees_strong": ["B.S. Computer Science", "B.S. Industrial Engineering", "MBA"],
        "degrees_weak": ["B.A. Religious Studies", "B.S. Forestry", "B.A. Dance"],
    },
]


NOISE_WORK_BULLETS = [
    "Maintained internal knowledge base and onboarding documentation for new team members",
    "Organized quarterly team-building events and logistics for 30-person department offsite",
    "Filed weekly expense reports and reconciled monthly departmental travel budget",
    "Served as office safety warden and completed workplace first-aid certification",
    "Coordinated annual performance review scheduling and collected 360-degree feedback forms",
    "Managed conference room bookings and visitor badge access for external guests",
    "Represented department at company all-hands meetings and new hire orientation sessions",
    "Updated employee contact directory and emergency notification distribution lists",
    "Arranged catering orders for client-facing meetings and internal lunch-and-learn sessions",
    "Maintained shared team calendar and coordinated meeting times across three time zones",
    "Processed incoming vendor invoices and tracked purchase order status in accounting system",
    "Sorted and distributed departmental mail; maintained physical and digital filing systems",
]


def make_resume_screening(rand_seed: int = 42) -> RubricDatapoint:
    """Screen candidates against job requirements with signal-stripped profiles.

    Hardened prototype: profiles present only work history bullets (no quality
    summaries), skills are described behaviorally, and the rubric uses 100%
    binary categories with seed-specific checks.

    Problem type: resume_screening.
    """
    rng = _random.Random(rand_seed)

    # --- Select role and generate identifiers --------------------------------
    role = rng.choice(RESUME_ROLES)
    company = pick_one(COMPANY_NAMES, rand_seed + 10)
    hiring_mgr = make_name(rand_seed + 10)
    candidate_names = make_names(rand_seed + 20, 5)
    requirements = role["requirements"]
    nice_to_haves = role["nice_to_have"]

    # --- Assign quality tiers to candidates ----------------------------------
    # met_counts[i] = how many of 5 requirements candidate i meets
    met_counts = [5, 4, 3, 2, 0]
    rng.shuffle(met_counts)

    # Pick work-history companies (exclude hiring company to avoid confusion)
    work_companies = [c for c in COMPANY_NAMES if c != company]
    rng.shuffle(work_companies)

    candidates = []
    for idx, (name, met_count) in enumerate(zip(candidate_names, met_counts)):
        # --- Determine which requirements this candidate meets ---------------
        req_indices = list(range(5))
        rng.shuffle(req_indices)
        met_indices = sorted(req_indices[:met_count])
        missed_indices = sorted(req_indices[met_count:])

        # Nice-to-haves: top gets all 3, strong gets 1, others get none
        if met_count == 5:
            met_nice = list(range(len(nice_to_haves)))
        elif met_count == 4:
            met_nice = [rng.randint(0, len(nice_to_haves) - 1)]
        else:
            met_nice = []

        # Years of experience (model must infer from work history dates)
        yrs = {5: rng.randint(8, 12), 4: rng.randint(5, 8),
               3: rng.randint(3, 5), 2: rng.randint(2, 4),
               0: rng.randint(1, 2)}[met_count]

        # --- Gather work-history evidence bullets ----------------------------
        strong_bullets = [requirements[i]["strong"] for i in met_indices]
        nice_bullets = [nice_to_haves[i]["evidence"] for i in met_nice]
        # Include 1-2 weak bullets for unmet requirements (adjacent-but-
        # insufficient experience that makes evaluation harder)
        weak_bullets = [requirements[i]["weak"] for i in missed_indices[:2]]
        noise = rng.sample(NOISE_WORK_BULLETS, min(2, len(NOISE_WORK_BULLETS)))

        all_bullets = strong_bullets + nice_bullets + weak_bullets + noise
        rng.shuffle(all_bullets)

        # --- Distribute bullets across 2-3 job entries -----------------------
        n_jobs = 3 if len(all_bullets) >= 7 else 2
        job_bullets = []
        for j in range(n_jobs):
            start_slice = j * len(all_bullets) // n_jobs
            end_slice = (j + 1) * len(all_bullets) // n_jobs
            job_bullets.append(all_bullets[start_slice:end_slice])

        # Select job titles by quality tier (NOT the target role title)
        if met_count >= 4:
            title_pool = role["titles_strong"]
        elif met_count >= 2:
            title_pool = role["titles_mid"]
        else:
            title_pool = role["titles_weak"]

        # Select degree by quality tier
        if met_count >= 3:
            degree = rng.choice(role["degrees_strong"])
        else:
            degree = rng.choice(role["degrees_weak"])

        # --- Build the profile text (uniform format, NO summary) -------------
        profile = f"CANDIDATE: {name}\n\nWORK EXPERIENCE\n\n"
        cursor_year = 2024

        # Compute per-job tenures that sum to total years
        tenures = []
        remaining_yrs = yrs
        for j in range(n_jobs):
            if j == n_jobs - 1:
                tenures.append(max(1, remaining_yrs))
            else:
                t = max(1, remaining_yrs // (n_jobs - j))
                tenures.append(t)
                remaining_yrs -= t

        for j, bullets in enumerate(job_bullets):
            if not bullets:
                continue
            tenure = tenures[j]
            start_yr = cursor_year - tenure
            end_label = "Present" if j == 0 else str(cursor_year)
            job_co = work_companies[(idx * 3 + j) % len(work_companies)]
            title = rng.choice(title_pool)

            profile += f"{job_co} \u2014 {title}\n"
            profile += f"{start_yr} \u2013 {end_label}\n"
            for b in bullets:
                profile += f"  \u2022 {b}\n"
            profile += "\n"
            cursor_year = start_yr

        grad_year = 2024 - yrs - rng.randint(0, 4)
        school = rng.choice([
            "State University", "City College", "Regional University",
            "Technical Institute", "National University",
        ])
        profile += f"EDUCATION\n{degree} \u2014 {school} ({grad_year})\n"

        candidates.append({
            "name": name,
            "met_count": met_count,
            "met_indices": met_indices,
            "missed_indices": missed_indices,
            "met_nice": met_nice,
            "years": yrs,
            "profile": profile,
        })

    # --- Build job description -----------------------------------------------
    job_description = f"Job Posting: {role['title']}\n"
    job_description += f"Company: {company}\n"
    job_description += f"Hiring Manager: {hiring_mgr}\n"
    job_description += f"Location: Remote (US-based)\n"
    job_description += f"Salary Range: {role['salary_range']}\n"
    job_description += f"\nABOUT THE ROLE\n"
    job_description += f"{company} is looking for a {role['title']} to join our growing team.\n"
    job_description += f"\nREQUIRED QUALIFICATIONS\n"
    for i, req in enumerate(requirements, 1):
        job_description += f"  {i}. {req['text']}\n"
    job_description += "\nNICE TO HAVE\n"
    for nt in nice_to_haves:
        job_description += f"  - {nt['text']}\n"
    job_description += "\nWHAT WE OFFER\n"
    job_description += "- Competitive salary and equity\n"
    job_description += "- Comprehensive health benefits\n"
    job_description += "- Flexible work arrangements\n"
    job_description += "- Professional development budget\n"

    # --- Identify top and bottom candidates ----------------------------------
    top_candidate = next(c for c in candidates if c["met_count"] == 5)
    bottom_candidate = next(c for c in candidates if c["met_count"] == 0)

    # --- Build rubric (100% binary, seed-specific) ---------------------------
    rubric_cats: list[BinaryRubricCategory] = []

    # -- Structural checks (3 categories, 3 pts) --
    rubric_cats.append(BinaryRubricCategory(
        name="file_exists",
        question="Does /testbed/screening_report.txt exist with substantial content (at least 200 words)?",
        points=1,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="all_candidates_evaluated",
        question=(
            "Does the report mention and evaluate all five candidates by name ("
            + ", ".join(c["name"] for c in candidates) + ")?"
        ),
        points=1,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="ranking_provided",
        question="Does the report provide a clear ranking or tier list of all five candidates from best to worst fit?",
        points=1,
    ))

    # -- Top/bottom identification (2 categories, 5 pts) --
    rubric_cats.append(BinaryRubricCategory(
        name="top_candidate_identified",
        question=f"Does the report rank or recommend {top_candidate['name']} as the strongest candidate overall?",
        points=3,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="bottom_candidate_identified",
        question=(
            f"Does the report rank {bottom_candidate['name']} as the weakest "
            f"candidate or explicitly recommend against advancing them?"
        ),
        points=2,
    ))

    # -- Per-requirement evaluation (5 categories, 5 pts) --
    for i, req in enumerate(requirements):
        rubric_cats.append(BinaryRubricCategory(
            name=f"evaluates_requirement_{i + 1}",
            question=f"Does the analysis evaluate candidates against the requirement: '{req['text']}'?",
            points=1,
        ))

    # -- Per-candidate primary checks (5 categories, 10 pts) --
    # Strong candidates (met >= 4): check a key strength is noted
    # Weaker candidates (met <= 3): check a key gap is noted
    for c in candidates:
        cname_safe = c["name"].lower().replace(" ", "_")
        if c["met_count"] >= 4:
            key_idx = c["met_indices"][0]
            rubric_cats.append(BinaryRubricCategory(
                name=f"{cname_safe}_strength",
                question=(
                    f"Does the report identify that {c['name']} has relevant "
                    f"experience in {requirements[key_idx]['area']}?"
                ),
                points=2,
            ))
        else:
            key_idx = c["missed_indices"][0]
            rubric_cats.append(BinaryRubricCategory(
                name=f"{cname_safe}_gap",
                question=(
                    f"Does the report identify that {c['name']} lacks sufficient "
                    f"experience with {requirements[key_idx]['area']}?"
                ),
                points=2,
            ))

    # -- Per-candidate secondary checks (5 categories, 10 pts) --
    for c in candidates:
        cname_safe = c["name"].lower().replace(" ", "_")
        if c["met_count"] == 5 and c["met_nice"]:
            nice_idx = c["met_nice"][0]
            rubric_cats.append(BinaryRubricCategory(
                name=f"{cname_safe}_nice_to_have",
                question=(
                    f"Does the report note that {c['name']} also brings "
                    f"{nice_to_haves[nice_idx]['text']}?"
                ),
                points=2,
            ))
        elif c["met_count"] == 4 and c["missed_indices"]:
            gap_idx = c["missed_indices"][0]
            rubric_cats.append(BinaryRubricCategory(
                name=f"{cname_safe}_single_gap",
                question=(
                    f"Does the report identify that {c['name']} lacks "
                    f"{requirements[gap_idx]['area']}?"
                ),
                points=2,
            ))
        elif c["met_count"] == 3 and len(c["met_indices"]) >= 2:
            str_idx = c["met_indices"][1]
            rubric_cats.append(BinaryRubricCategory(
                name=f"{cname_safe}_partial_strength",
                question=(
                    f"Does the report acknowledge that {c['name']} has some "
                    f"relevant experience in {requirements[str_idx]['area']}?"
                ),
                points=2,
            ))
        elif c["met_count"] <= 2 and len(c["missed_indices"]) >= 2:
            gap_idx = c["missed_indices"][1]
            rubric_cats.append(BinaryRubricCategory(
                name=f"{cname_safe}_second_gap",
                question=(
                    f"Does the report identify that {c['name']} lacks "
                    f"experience with {requirements[gap_idx]['area']}?"
                ),
                points=2,
            ))

    # -- False-positive / safety checks (3 categories, 6 pts) --
    if bottom_candidate["missed_indices"]:
        fp_idx = bottom_candidate["missed_indices"][0]
        rubric_cats.append(BinaryRubricCategory(
            name="no_false_positive_bottom",
            question=(
                f"Does the report avoid claiming that {bottom_candidate['name']} "
                f"has strong experience with {requirements[fp_idx]['area']}?"
            ),
            points=2,
        ))
    rubric_cats.append(BinaryRubricCategory(
        name="no_advance_bottom",
        question=(
            f"Does the report avoid recommending {bottom_candidate['name']} "
            f"for advancement to the interview stage?"
        ),
        points=2,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="no_reject_top",
        question=(
            f"Does the report avoid recommending against {top_candidate['name']} "
            f"or ranking them in the bottom half?"
        ),
        points=2,
    ))

    # --- Assemble files ------------------------------------------------------
    necessary = {"/testbed/job_description.txt": job_description}
    for idx, c in enumerate(candidates):
        necessary[f"/testbed/candidates/candidate_{idx + 1}.txt"] = c["profile"]

    return RubricDatapoint(
        problem_statement=(
            f"# Resume Screening\n\n"
            f"You are assisting {hiring_mgr} at {company} with hiring for the\n"
            f"**{role['title']}** position.\n\n"
            f"Files:\n"
            f"- Job description: `/testbed/job_description.txt`\n"
            f"- Candidate profiles: `/testbed/candidates/candidate_1.txt` "
            f"through `candidate_5.txt`\n\n"
            f"Review each candidate's work experience against the job requirements.\n"
            f"Evaluate what each candidate has actually done (not just their job\n"
            f"titles) and assess how well their experience maps to the stated\n"
            f"requirements.\n\n"
            f"Rank all candidates from best to worst fit, with specific justification\n"
            f"for each ranking. For each candidate, identify their key strengths and\n"
            f"gaps relative to the requirements. Clearly state which candidate(s) you\n"
            f"recommend advancing to interview and which you recommend rejecting.\n\n"
            f"Write your screening report to `/testbed/screening_report.txt`."
        ),
        rubric=tuple(rubric_cats),
        submission_instructions="Write your screening report to /testbed/screening_report.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files=necessary,
        problem_type="resume_screening",
    )


# ============================================================================
# 4. Survey analysis
# ============================================================================

def make_survey_analysis(rand_seed: int = 42) -> RubricDatapoint:
    """Given survey CSV with Likert + categorical responses and a research question.

    Problem type: survey_analysis.
    """
    rng = _random.Random(rand_seed)

    company = pick_one(COMPANY_NAMES, rand_seed + 30)
    researcher = make_name(rand_seed + 30)

    # --- Parameters ---
    n_responses = vary_int(80, rand_seed)
    groups = ["Group A", "Group B"]
    topics = [
        ("workplace_satisfaction", "How satisfied are you with your workplace environment?"),
        ("manager_effectiveness", "How effective is your direct manager?"),
        ("career_growth", "How satisfied are you with career growth opportunities?"),
        ("work_life_balance", "How would you rate your work-life balance?"),
        ("team_collaboration", "How well does your team collaborate?"),
    ]
    departments = ["Engineering", "Sales", "Marketing", "Support", "HR"]

    # --- Generate survey data with known patterns --------------------------------
    # Pattern: Group A rates questions 1 and 3 higher than Group B
    header = "response_id,group,department,q1_satisfaction,q2_manager,q3_career,q4_balance,q5_collaboration,tenure_years"
    rows = [header]

    group_q1: dict[str, list[int]] = {"Group A": [], "Group B": []}
    group_q3: dict[str, list[int]] = {"Group A": [], "Group B": []}
    all_q1: list[int] = []
    all_q2: list[int] = []
    all_q4: list[int] = []
    dept_counts: dict[str, int] = {d: 0 for d in departments}
    mode_q5_tracker: list[int] = []

    for i in range(1, n_responses + 1):
        group = rng.choice(groups)
        dept = rng.choice(departments)
        dept_counts[dept] += 1

        # Likert 1-5 with group differences on q1 and q3
        if group == "Group A":
            q1 = min(5, max(1, round(rng.gauss(4.0, 0.7))))
            q3 = min(5, max(1, round(rng.gauss(3.8, 0.8))))
        else:
            q1 = min(5, max(1, round(rng.gauss(2.8, 0.9))))
            q3 = min(5, max(1, round(rng.gauss(2.5, 1.0))))

        q2 = min(5, max(1, round(rng.gauss(3.5, 0.9))))
        q4 = min(5, max(1, round(rng.gauss(3.2, 1.0))))
        q5 = rng.choice([3, 4, 4, 4, 5])  # mode is 4
        tenure = rng.randint(1, 15)

        group_q1[group].append(q1)
        group_q3[group].append(q3)
        all_q1.append(q1)
        all_q2.append(q2)
        all_q4.append(q4)
        mode_q5_tracker.append(q5)

        rows.append(f"R{i:03d},{group},{dept},{q1},{q2},{q3},{q4},{q5},{tenure}")

    survey_csv = "\n".join(rows) + "\n"

    # --- Compute ground truths ---------------------------------------------------
    def _mean(v: list[int | float]) -> float:
        return sum(v) / len(v) if v else 0.0

    def _mode(v: list[int]) -> int:
        from collections import Counter
        return Counter(v).most_common(1)[0][0]

    correct_count = n_responses
    correct_mean_q1 = round(_mean(all_q1), 2)
    correct_mean_q1_a = round(_mean(group_q1["Group A"]), 2)
    correct_mean_q1_b = round(_mean(group_q1["Group B"]), 2)
    correct_mode_q5 = _mode(mode_q5_tracker)
    group_diff_q1 = round(correct_mean_q1_a - correct_mean_q1_b, 2)

    research_brief = f"""Research Brief: Employee Experience Survey Analysis
Company: {company}
Lead Researcher: {researcher}

RESEARCH QUESTION
Is there a meaningful difference in workplace satisfaction between Group A
and Group B employees? Specifically, we want to understand:
  1. Overall response patterns across all five survey questions
  2. Whether Group A and Group B differ on satisfaction (Q1) and career growth (Q3)
  3. The distribution of collaboration scores (Q5)
  4. Any notable department-level patterns

DATA DESCRIPTION
- File: /testbed/data/survey_responses.csv
- {n_responses} responses collected over a 2-week period
- Groups are self-identified team designations
- All Likert questions are scored 1 (Strongly Disagree) to 5 (Strongly Agree)
- Tenure is reported in years

DELIVERABLE
A written analysis addressing the research question with supporting statistics.
"""

    # --- Rubric ------------------------------------------------------------------
    rubric_cats: list[BinaryRubricCategory | RubricCategory] = []

    rubric_cats.append(BinaryRubricCategory(
        name="correct_response_count",
        question=f"Does the analysis correctly state that there are {correct_count} survey responses?",
        points=1,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="correct_mean_q1",
        question=f"Does the analysis report the overall mean for Q1 (satisfaction) as approximately {correct_mean_q1}?",
        points=2,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="identified_group_difference",
        question=(
            f"Does the analysis identify that Group A rates Q1 higher than Group B "
            f"(Group A mean ~{correct_mean_q1_a} vs Group B mean ~{correct_mean_q1_b})?"
        ),
        points=2,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="identified_q3_difference",
        question="Does the analysis note that Group A also rates career growth (Q3) higher than Group B?",
        points=2,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="correct_mode_q5",
        question=f"Does the analysis identify that the mode of Q5 (collaboration) is {correct_mode_q5}?",
        points=2,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="department_breakdown",
        question="Does the analysis include any department-level breakdown or observation?",
        points=1,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="addresses_research_question",
        question="Does the analysis explicitly answer the research question about whether there is a meaningful difference between groups?",
        points=2,
    ))

    rubric_cats.append(RubricCategory(
        name="insight_depth",
        description="Does the analysis go beyond basic descriptive stats to provide meaningful insights?",
        failure="Only raw numbers with no interpretation.",
        minor_failure="Basic means reported with minimal interpretation.",
        minor_success="Good interpretation of key findings with some context.",
        success="Thoughtful analysis connecting multiple findings, noting patterns, and offering actionable insights.",
        points=3,
    ))
    rubric_cats.append(RubricCategory(
        name="distribution_analysis",
        description="Does the analysis discuss the distribution of responses beyond just averages?",
        failure="Only reports means with no discussion of spread or variation.",
        minor_failure="Mentions some variation but does not quantify it.",
        minor_success="Reports standard deviations or ranges for key questions.",
        success="Thorough discussion of response distributions including spread, notable outliers, or skew patterns.",
        points=3,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="structured_report",
        question="Is the analysis organized with clear sections (e.g. methodology, findings, conclusion)?",
        points=1,
    ))

    # 8 binary + 2 graded + 1 binary = 11 cats; binary 9/11 ≈ 82% — within range but let's
    # add one more graded to balance: 8 binary + 3 graded = 11 total, binary 8/11 = 73%
    rubric_cats.append(RubricCategory(
        name="statistical_rigor",
        description="Does the analysis use appropriate statistical methods (means, medians, counts, distributions)?",
        failure="No statistical methods; purely anecdotal.",
        minor_failure="Only means computed; no distributional analysis.",
        minor_success="Means and at least one other measure (median, mode, or distribution) used.",
        success="Multiple statistical measures used appropriately, including central tendency, spread, and comparison metrics.",
        points=3,
    ))

    # Final: 8 binary + 3 graded = 11 cats; binary 8/11 = 73%

    return RubricDatapoint(
        problem_statement=(
            f"# Survey Analysis\n\n"
            f"You are a research analyst at {company}. {researcher} has collected employee\n"
            f"survey data and needs help analysing it.\n\n"
            f"Files:\n"
            f"- Survey responses: `/testbed/data/survey_responses.csv`\n"
            f"- Research brief: `/testbed/docs/research_brief.txt`\n\n"
            f"Read the research brief carefully, then analyse the survey data to answer\n"
            f"the research question. Compute relevant statistics, identify patterns, and\n"
            f"provide an insightful written analysis.\n\n"
            f"Write your analysis to `/testbed/analysis.txt`."
        ),
        rubric=tuple(rubric_cats),
        submission_instructions="Write your analysis to /testbed/analysis.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/survey_responses.csv": survey_csv,
            "/testbed/docs/research_brief.txt": research_brief,
        },
        problem_type="survey_analysis",
    )


# ============================================================================
# 5. Medical triage notes
# ============================================================================

def make_medical_triage_notes(rand_seed: int = 42) -> RubricDatapoint:
    """Given patient intake form, vitals, and symptoms, write a triage assessment.

    Problem type: medical_triage.
    """
    rng = _random.Random(rand_seed)

    # Select a scenario from the content pool using seed
    scenario = TRIAGE_SCENARIOS[rand_seed % len(TRIAGE_SCENARIOS)]
    patient_name = make_name(rand_seed + 50)

    vitals = scenario["vitals"]
    red_flags = scenario["red_flags"]
    priority = scenario["priority"]
    expected_actions = scenario["expected_actions"]
    chief_complaint = scenario["chief_complaint"]

    # --- Build patient files -----------------------------------------------------
    age = scenario["age"]
    sex = scenario["sex"]
    allergies = rng.choice(["NKDA (No Known Drug Allergies)", "Penicillin", "Sulfa drugs", "Latex"])
    insurance = rng.choice(["BlueCross BlueShield", "Aetna", "Medicare", "United Healthcare", "Self-pay"])

    birth_year = 2024 - age
    intake_form = f"""PATIENT INTAKE FORM
===============================
Name: {patient_name}
Date of Birth: {birth_year}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}
Age: {age}
Sex: {sex}
Chief Complaint: {chief_complaint}

MEDICAL HISTORY
{scenario['history']}

ALLERGIES: {allergies}
INSURANCE: {insurance}

EMERGENCY CONTACT
  Name: {make_name(rand_seed + 51)}
  Relationship: {rng.choice(['Spouse', 'Parent', 'Sibling', 'Child'])}
  Phone: ({rng.randint(200,999)}) {rng.randint(200,999)}-{rng.randint(1000,9999)}

ARRIVAL TIME: {rng.randint(0,23):02d}:{rng.randint(0,59):02d}
ARRIVAL MODE: {rng.choice(['Ambulance', 'Walk-in', 'Private vehicle'])}
"""

    vitals_file = f"""VITAL SIGNS — {patient_name}
Recorded at triage
===============================
Blood Pressure: {vitals['bp']} mmHg
Heart Rate: {vitals['hr']} bpm
Respiratory Rate: {vitals['rr']} breaths/min
Temperature: {vitals['temp']}
SpO2: {vitals['spo2']}
Pain Scale: {rng.randint(3, 9)}/10

NOTES:
- Patient is {rng.choice(['alert', 'alert and oriented x3', 'drowsy but responsive'])}
- {rng.choice(['Skin warm and dry', 'Skin diaphoretic', 'Skin pale and cool'])}
"""

    symptoms_text = f"""SYMPTOM ASSESSMENT — {patient_name}
===============================
Chief Complaint: {chief_complaint}

HISTORY OF PRESENT ILLNESS
{scenario['history']}

ASSOCIATED SYMPTOMS:
"""
    possible_associated = [
        "nausea", "dizziness", "fatigue", "weakness", "chills",
        "sweating", "loss of appetite", "anxiety", "numbness",
    ]
    assoc_symptoms = rng.sample(possible_associated, rng.randint(2, 4))
    for s in assoc_symptoms:
        symptoms_text += f"  - {s}\n"
    symptoms_text += f"""
ONSET: {rng.choice(['Sudden', 'Gradual over hours', 'Gradual over days'])}
DURATION: {rng.choice(['< 1 hour', '1-6 hours', '6-24 hours', '> 24 hours'])}
AGGRAVATING FACTORS: {rng.choice(['movement', 'deep breathing', 'eating', 'exertion', 'lying flat'])}
RELIEVING FACTORS: {rng.choice(['rest', 'nothing', 'sitting upright', 'over-the-counter medications'])}
"""

    # --- Triage protocol reference -----------------------------------------------
    triage_protocol = """EMERGENCY SEVERITY INDEX (ESI) TRIAGE PROTOCOL
================================================

ESI Level 1 — Resuscitation
  Requires immediate life-saving intervention
  Examples: cardiac arrest, respiratory failure, severe trauma, status epilepticus
  Action: Immediate physician assessment, resuscitation bay

ESI Level 2 — Emergent
  High-risk situation, confused/lethargic/disoriented, or severe pain/distress
  Examples: chest pain with cardiac history, stroke symptoms, acute abdomen,
  high-risk vitals (SpO2 < 92%, HR > 120 or < 50, systolic BP < 90)
  Action: Bedside within 10 minutes, continuous monitoring

ESI Level 3 — Urgent
  Stable but expected to need 2+ resources (labs, imaging, IV, consults)
  Examples: abdominal pain (stable vitals), moderate asthma, lacerations needing sutures
  Action: Assess within 30 minutes

ESI Level 4 — Less Urgent
  Stable, expected to need 1 resource
  Examples: simple laceration, sprain, UTI symptoms, medication refill
  Action: Assess within 60 minutes

ESI Level 5 — Non-Urgent
  Stable, no resources expected (Rx refill, minor complaint)
  Action: Assess when available

RED FLAG VITALS:
  - SpO2 < 92%
  - Heart rate > 120 or < 50
  - Systolic BP < 90 or > 180
  - Temperature > 103°F or < 95°F
  - Respiratory rate > 24 or < 8
  - Altered mental status at any vital level

DOCUMENTATION REQUIREMENTS:
  1. Patient identification and chief complaint
  2. Vital signs with interpretation
  3. Brief clinical assessment
  4. Identified red flags (if any)
  5. ESI level assignment with justification
  6. Recommended immediate actions / next steps
"""

    # --- Identify what a good triage should contain ------------------------------
    # Detect vitals abnormalities
    vitals_abnormalities = []
    spo2_val = int(vitals["spo2"].replace("%", ""))
    hr_val = vitals["hr"]
    rr_val = vitals["rr"]
    bp_sys = int(vitals["bp"].split("/")[0])
    temp_str = vitals["temp"]
    temp_val = float(temp_str.replace("°F", ""))

    if spo2_val < 92:
        vitals_abnormalities.append(f"SpO2 {vitals['spo2']} is below 92%")
    if hr_val > 120 or hr_val < 50:
        vitals_abnormalities.append(f"Heart rate {hr_val} is outside normal range")
    if bp_sys > 180 or bp_sys < 90:
        vitals_abnormalities.append(f"Systolic BP {bp_sys} is outside normal range")
    if temp_val > 103.0:
        vitals_abnormalities.append(f"Temperature {temp_str} exceeds 103°F")
    if rr_val > 24 or rr_val < 8:
        vitals_abnormalities.append(f"Respiratory rate {rr_val} is outside normal range")
    # Also flag borderline concerning vitals
    if hr_val > 100 and hr_val <= 120:
        vitals_abnormalities.append(f"Tachycardia noted (HR {hr_val})")
    if bp_sys > 140 and bp_sys <= 180:
        vitals_abnormalities.append(f"Elevated systolic BP ({bp_sys})")

    has_red_flags = len(red_flags) > 0
    has_vitals_abnormality = len(vitals_abnormalities) > 0

    # --- Rubric ------------------------------------------------------------------
    rubric_cats: list[BinaryRubricCategory | RubricCategory] = []

    rubric_cats.append(BinaryRubricCategory(
        name="correct_triage_level",
        question=f"Does the assessment assign the correct triage level: {priority}?",
        points=3,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="red_flag_identified",
        question=(
            f"Does the assessment identify at least one red flag from: {', '.join(red_flags)}?"
            if has_red_flags
            else "Does the assessment correctly note the absence of major red flags?"
        ),
        points=2,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="vitals_abnormality_noted",
        question=(
            f"Does the assessment note at least one vitals abnormality ({vitals_abnormalities[0] if vitals_abnormalities else 'N/A'})?"
            if has_vitals_abnormality
            else "Does the assessment confirm that vital signs are within normal limits?"
        ),
        points=2,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="appropriate_next_steps",
        question=f"Does the assessment recommend at least two appropriate next steps from: {', '.join(expected_actions[:4])}?",
        points=2,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="chief_complaint_stated",
        question=f"Does the assessment clearly state the chief complaint ({chief_complaint})?",
        points=1,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="patient_history_referenced",
        question="Does the assessment reference relevant patient history (medications, prior conditions)?",
        points=1,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="all_vitals_documented",
        question="Does the assessment document all five vital signs (BP, HR, RR, Temp, SpO2)?",
        points=1,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="triage_justification",
        question="Does the assessment provide a justification for the assigned triage level?",
        points=2,
    ))

    rubric_cats.append(RubricCategory(
        name="clinical_reasoning",
        description="Does the assessment demonstrate sound clinical reasoning connecting symptoms, vitals, and history?",
        failure="No reasoning; just lists data without interpretation.",
        minor_failure="Some interpretation but important connections missed.",
        minor_success="Good reasoning for the main diagnosis/concern; minor connections missed.",
        success="Excellent clinical reasoning that integrates symptoms, vitals, history, and red flags into a coherent assessment.",
        points=3,
    ))
    rubric_cats.append(RubricCategory(
        name="documentation_quality",
        description="Is the triage assessment well-structured and professionally documented?",
        failure="Disorganized, missing major required sections.",
        minor_failure="Some structure but key sections are incomplete.",
        minor_success="Well-structured with most required sections; minor gaps.",
        success="Professionally formatted with all required sections: identification, vitals, assessment, red flags, ESI level, and next steps.",
        points=3,
    ))

    # 8 binary + 2 graded = 10 cats; binary 8/10 = 80%
    # Add 2 more to reach 12
    rubric_cats.append(BinaryRubricCategory(
        name="urgency_communicated",
        question="Does the assessment clearly communicate the urgency level to the reader (e.g. 'requires immediate attention' vs 'can wait')?",
        points=1,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="allergies_noted",
        question=f"Does the assessment mention the patient's allergy status ({allergies})?",
        points=1,
    ))

    rubric_cats.append(RubricCategory(
        name="protocol_adherence",
        description="Does the assessment follow the structure and requirements outlined in the triage protocol?",
        failure="Protocol not referenced or followed at all.",
        minor_failure="Some protocol elements followed but key sections missing.",
        minor_success="Most protocol requirements met with minor omissions.",
        success="Assessment follows the protocol precisely, including all required documentation sections.",
        points=2,
    ))

    # 10 binary + 3 graded = 13 cats; binary 10/13 = 77%

    return RubricDatapoint(
        problem_statement=(
            f"# Medical Triage Assessment\n\n"
            f"You are an emergency department triage nurse. A patient ({patient_name}) has\n"
            f"just arrived. Review the intake form, vitals, and symptom assessment, then\n"
            f"write a triage assessment following the provided protocol.\n\n"
            f"Files:\n"
            f"- Patient intake form: `/testbed/patient/intake_form.txt`\n"
            f"- Vital signs: `/testbed/patient/vitals.txt`\n"
            f"- Symptom assessment: `/testbed/patient/symptoms.txt`\n"
            f"- Triage protocol: `/testbed/reference/triage_protocol.txt`\n\n"
            f"Your triage assessment should include: patient identification, vital sign\n"
            f"interpretation, clinical assessment, red flags, ESI triage level with\n"
            f"justification, and recommended next steps.\n\n"
            f"Write your triage assessment to `/testbed/triage_assessment.txt`."
        ),
        rubric=tuple(rubric_cats),
        submission_instructions="Write your triage assessment to /testbed/triage_assessment.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/patient/intake_form.txt": intake_form,
            "/testbed/patient/vitals.txt": vitals_file,
            "/testbed/patient/symptoms.txt": symptoms_text,
            "/testbed/reference/triage_protocol.txt": triage_protocol,
        },
        problem_type="medical_triage",
    )


# ============================================================================
# 6. Accessibility audit
# ============================================================================

def make_accessibility_audit(rand_seed: int = 42) -> RubricDatapoint:
    """Given an HTML file with WCAG violations and a guidelines excerpt, produce an audit.

    Problem type: accessibility_audit.
    """
    rng = _random.Random(rand_seed)

    company = pick_one(COMPANY_NAMES, rand_seed + 60)
    auditor = make_name(rand_seed + 60)

    # --- Pick 4-6 violations from the pool ---------------------------------------
    n_violations = rng.randint(4, 5)
    selected_violations = rng.sample(WCAG_VIOLATIONS, n_violations)

    # --- Build HTML with embedded violations -------------------------------------
    page_title = rng.choice([
        "Welcome to Our Platform",
        "Company Dashboard",
        "Product Catalog",
        "Service Portal",
        "Customer Account",
    ])

    # Check if missing_lang_attribute is among selected violations
    has_lang_violation = any(v["name"] == "missing_lang_attribute" for v in selected_violations)
    html_open = "<html>" if has_lang_violation else '<html lang="en">'

    # Check if missing_skip_nav is selected
    has_skip_nav_violation = any(v["name"] == "missing_skip_nav" for v in selected_violations)
    skip_nav_html = "" if has_skip_nav_violation else '<a href="#main-content" class="skip-link">Skip to main content</a>'

    # Build the body with violations embedded
    nav_items = ["Home", "Products", "Services", "About", "Contact"]
    nav_html = "\n".join(f'    <li><a href="/{item.lower()}">{item}</a></li>' for item in nav_items)

    # Collect violation HTML snippets for embedding in the page
    violation_snippets = []
    for v in selected_violations:
        if v["name"] not in ("missing_lang_attribute", "missing_skip_nav"):
            violation_snippets.append(v["html"])

    content_sections = [
        '<section>\n    <h2>Featured Products</h2>\n    <p>Check out our latest offerings designed to meet your needs.</p>\n  </section>',
        '<section>\n    <h2>Customer Testimonials</h2>\n    <p>"Great service and amazing support!" — A Happy Customer</p>\n  </section>',
        '<section>\n    <h2>Contact Us</h2>\n    <p>Reach out to our team for personalized assistance.</p>\n  </section>',
    ]

    # Interleave violations into the content
    body_parts = []
    snippet_idx = 0
    for section in content_sections:
        body_parts.append(f"  {section}")
        if snippet_idx < len(violation_snippets):
            body_parts.append(f"  {violation_snippets[snippet_idx]}")
            snippet_idx += 1

    # Add any remaining violation snippets
    while snippet_idx < len(violation_snippets):
        body_parts.append(f"  {violation_snippets[snippet_idx]}")
        snippet_idx += 1

    html_content = f"""{html_open}
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{page_title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; }}
    nav {{ background: #333; color: white; padding: 1rem; }}
    nav ul {{ list-style: none; display: flex; gap: 1rem; padding: 0; }}
    nav a {{ color: white; text-decoration: none; }}
    main {{ padding: 2rem; max-width: 1200px; margin: 0 auto; }}
    footer {{ background: #333; color: white; padding: 1rem; text-align: center; }}
    .skip-link {{ position: absolute; left: -9999px; }}
    .skip-link:focus {{ left: 0; top: 0; background: #fff; padding: 0.5rem; z-index: 1000; }}
  </style>
</head>
<body>
  {skip_nav_html}
  <header>
    <nav>
      <ul>
{nav_html}
      </ul>
    </nav>
  </header>

  <main id="main-content">
    <h1>{page_title}</h1>

{chr(10).join(body_parts)}

  </main>

  <footer>
    <p>&copy; 2024 {company}. All rights reserved.</p>
  </footer>
</body>
</html>
"""

    # --- WCAG guidelines reference -----------------------------------------------
    wcag_guidelines = """WCAG 2.1 Quick Reference — Selected Success Criteria
=====================================================

PRINCIPLE 1: PERCEIVABLE

SC 1.1.1 Non-text Content (Level A)
  All non-text content has a text alternative that serves the equivalent
  purpose. Images must have alt attributes. Decorative images use alt="".

SC 1.3.1 Info and Relationships (Level A)
  Information, structure, and relationships conveyed through presentation
  can be programmatically determined. Form inputs must have associated labels.

SC 1.4.2 Audio Control (Level A)
  If audio plays automatically for more than 3 seconds, there must be a
  mechanism to pause, stop, or control volume independently.

SC 1.4.3 Contrast (Minimum) (Level AA)
  Text and images of text have a contrast ratio of at least 4.5:1.
  Large text (18pt or 14pt bold) requires at least 3:1.


PRINCIPLE 2: OPERABLE

SC 2.1.2 No Keyboard Trap (Level A)
  If keyboard focus can be moved to a component, focus can also be moved
  away using only the keyboard. If non-standard keys are needed, the user
  is advised.

SC 2.4.1 Bypass Blocks (Level A)
  A mechanism (such as a "skip navigation" link) is available to bypass
  blocks of content repeated on multiple pages.

SC 2.4.4 Link Purpose (in Context) (Level A)
  The purpose of each link can be determined from the link text alone, or
  from the link text together with its programmatically determined context.


PRINCIPLE 3: UNDERSTANDABLE

SC 3.1.1 Language of Page (Level A)
  The default human language of each page can be programmatically determined
  via the lang attribute on the html element.


AUDIT REPORT FORMAT
  For each violation found, document:
  1. Location in the HTML (line or element description)
  2. The specific WCAG criterion violated
  3. Description of why it is a violation
  4. Recommended fix
  5. Priority level (Critical / Major / Minor)
"""

    # --- Rubric ------------------------------------------------------------------
    rubric_cats: list[BinaryRubricCategory | RubricCategory] = []

    for v in selected_violations:
        rubric_cats.append(BinaryRubricCategory(
            name=f"found_{v['name']}",
            question=f"Does the audit report identify the violation: {v['description']}?",
            points=2,
        ))
        rubric_cats.append(BinaryRubricCategory(
            name=f"criterion_{v['name']}",
            question=f"Does the audit correctly cite the relevant WCAG criterion ({v['criterion']}) for the {v['name']} violation?",
            points=1,
        ))

    rubric_cats.append(RubricCategory(
        name="recommendation_quality",
        description="Are the recommended fixes specific, actionable, and technically correct?",
        failure="No fixes recommended, or fixes are wrong.",
        minor_failure="Some fixes recommended but vague or incomplete.",
        minor_success="Good fixes for most violations with minor gaps.",
        success="Specific, correct, and implementable fix recommended for every identified violation.",
        points=3,
    ))
    rubric_cats.append(RubricCategory(
        name="prioritization",
        description="Does the audit prioritize violations by severity / impact?",
        failure="No prioritization or severity assessment.",
        minor_failure="Some mention of severity but inconsistent.",
        minor_success="Reasonable prioritization for most violations.",
        success="Clear, well-justified priority levels for all violations considering impact on users with disabilities.",
        points=3,
    ))
    rubric_cats.append(BinaryRubricCategory(
        name="no_false_positives",
        question="Does the audit flag at most one non-violation (false positive) in the provided HTML?",
        points=2,
    ))

    # For 5 violations: 10 binary (per-violation) + 1 binary (false positives) + 2 graded = 13 cats
    # binary 11/13 = 85% — let's add one more graded for balance
    rubric_cats.append(RubricCategory(
        name="report_completeness",
        description="Does the audit cover all required elements: location, criterion, description, fix, and priority for each issue?",
        failure="Missing most required elements for most violations.",
        minor_failure="Some elements present but inconsistent across violations.",
        minor_success="Most required elements present for most violations.",
        success="All five required elements documented for every identified violation.",
        points=3,
    ))

    # For 5 violations: 11 binary + 3 graded = 14 cats; binary 11/14 = 79%
    # For 4 violations: 9 binary + 3 graded = 12 cats; binary 9/12 = 75%

    return RubricDatapoint(
        problem_statement=(
            f"# Accessibility Audit\n\n"
            f"You are {auditor}, a web accessibility specialist. {company} has asked you\n"
            f"to audit their website's main page for WCAG 2.1 compliance.\n\n"
            f"Files:\n"
            f"- Website HTML: `/testbed/site/index.html`\n"
            f"- WCAG guidelines reference: `/testbed/reference/wcag_guidelines.txt`\n\n"
            f"Examine the HTML source and identify all accessibility violations. For each,\n"
            f"document the location, the WCAG criterion violated, why it is a violation,\n"
            f"a recommended fix, and a priority level.\n\n"
            f"Write your audit report to `/testbed/audit_report.txt`."
        ),
        rubric=tuple(rubric_cats),
        submission_instructions="Write your audit report to /testbed/audit_report.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/site/index.html": html_content,
            "/testbed/reference/wcag_guidelines.txt": wcag_guidelines,
        },
        problem_type="accessibility_audit",
    )
