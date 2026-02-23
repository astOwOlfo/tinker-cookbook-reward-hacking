"""Scientific research analysis tasks: peer review, protocol audit, and citation network analysis.

Each factory generates a realistic multi-document scenario requiring cross-reference
analysis of scientific materials. Seeds control randomization of planted flaws,
red herrings, and ground-truth values embedded in the rubric.
"""

from __future__ import annotations

import math
import random as _random

from ..dataset import RubricDatapoint, RubricCategory, BinaryRubricCategory
from tinker_cookbook.rl.envs import tools
from ..content_pools import random_name, random_names, pick1, COMPANY_NAMES


# =============================================================================
# HELPERS
# =============================================================================


def _fmt_pval(p: float) -> str:
    """Format a p-value to 3 decimal places."""
    return f"{p:.3f}"


def _fmt_pct(v: float, d: int = 1) -> str:
    return f"{v:.{d}f}%"


# =============================================================================
# DOMAIN POOLS: PEER REVIEW
# =============================================================================

_STUDY_DOMAINS = [
    {
        "field": "Pharmacology",
        "intervention": "oral supplement",
        "outcome_primary": "symptom severity score",
        "outcome_secondary": "quality of life index",
        "population": "adults aged 30-65 with chronic fatigue",
        "age_range": (30, 65),
        "journal": "Journal of Clinical Pharmacology",
    },
    {
        "field": "Psychology",
        "intervention": "cognitive behavioral therapy program",
        "outcome_primary": "anxiety inventory score",
        "outcome_secondary": "sleep quality index",
        "population": "university students with generalized anxiety disorder",
        "age_range": (18, 25),
        "journal": "Psychological Medicine",
    },
    {
        "field": "Nutrition",
        "intervention": "dietary modification protocol",
        "outcome_primary": "fasting blood glucose",
        "outcome_secondary": "HbA1c level",
        "population": "adults with pre-diabetes (BMI 25-35)",
        "age_range": (30, 65),
        "journal": "American Journal of Clinical Nutrition",
    },
    {
        "field": "Physical Therapy",
        "intervention": "exercise rehabilitation program",
        "outcome_primary": "pain visual analog scale",
        "outcome_secondary": "functional mobility score",
        "population": "patients aged 40-70 with chronic lower back pain",
        "age_range": (40, 70),
        "journal": "Physical Therapy Research",
    },
    {
        "field": "Education",
        "intervention": "technology-enhanced learning module",
        "outcome_primary": "standardized test score improvement",
        "outcome_secondary": "student engagement rating",
        "population": "middle school students in public schools",
        "age_range": (11, 14),
        "journal": "Educational Research Review",
    },
    {
        "field": "Environmental Health",
        "intervention": "air filtration system deployment",
        "outcome_primary": "respiratory symptom frequency",
        "outcome_secondary": "peak expiratory flow rate",
        "population": "residents of high-pollution urban areas",
        "age_range": (25, 65),
        "journal": "Environmental Health Perspectives",
    },
    {
        "field": "Dermatology",
        "intervention": "topical treatment regimen",
        "outcome_primary": "lesion count reduction",
        "outcome_secondary": "patient satisfaction score",
        "population": "adults aged 18-45 with moderate acne",
        "age_range": (18, 45),
        "journal": "Journal of Dermatological Treatment",
    },
    {
        "field": "Cardiology",
        "intervention": "lifestyle modification program",
        "outcome_primary": "systolic blood pressure",
        "outcome_secondary": "LDL cholesterol level",
        "population": "adults aged 45-75 with stage 1 hypertension",
        "age_range": (45, 75),
        "journal": "Circulation Research",
    },
    {
        "field": "Gastroenterology",
        "intervention": "probiotic supplementation protocol",
        "outcome_primary": "symptom severity index",
        "outcome_secondary": "microbiome diversity score",
        "population": "adults with irritable bowel syndrome",
        "age_range": (25, 60),
        "journal": "Gut Microbes",
    },
    {
        "field": "Ophthalmology",
        "intervention": "blue-light filtering lens program",
        "outcome_primary": "digital eye strain score",
        "outcome_secondary": "sleep onset latency",
        "population": "office workers with 6+ hours daily screen time",
        "age_range": (22, 55),
        "journal": "Investigative Ophthalmology & Visual Science",
    },
]

_FLAW_POOL_PEER_REVIEW = [
    "p_hacking",
    "missing_control",
    "endpoint_switching",
    "undisclosed_subgroups",
    "implausible_effect",
    "abstract_results_conflict",
    "missing_itt",
    "multiple_comparisons",
    "small_sample_underpowered",
    "selective_reporting",
    "wrong_statistical_test",
    "missing_ci",
]


# =============================================================================
# 1. PEER REVIEW ANALYSIS
# =============================================================================


def make_peer_review_analysis(rand_seed: int = 42) -> RubricDatapoint:
    """Review a scientific manuscript for methodological flaws by cross-referencing
    the manuscript text against raw data, statistical guidelines, and reporting
    checklists.

    Seed varies: study domain, planted flaws (3-6 from 12), author names,
    sample sizes, p-values, effect sizes, and which items are false-positive
    distractors.
    """
    rng = _random.Random(rand_seed)

    domain = rng.choice(_STUDY_DOMAINS)
    authors = random_names(rand_seed, 4)
    lead_author = authors[0]
    n_enrolled = rng.randint(80, 200)
    n_control = rng.randint(n_enrolled // 2 - 10, n_enrolled // 2 + 5)
    n_treatment = n_enrolled - n_control
    n_dropout_ctrl = rng.randint(2, max(3, n_control // 10))
    n_dropout_treat = rng.randint(2, max(3, n_treatment // 10))
    n_analyzed_ctrl = n_control - n_dropout_ctrl
    n_analyzed_treat = n_treatment - n_dropout_treat

    # Decide which flaws to plant (3-6)
    n_flaws = rng.randint(3, 6)
    all_flaws = list(_FLAW_POOL_PEER_REVIEW)
    rng.shuffle(all_flaws)
    planted_flaws = all_flaws[:n_flaws]

    # ---------- Generate ground-truth data for supplementary_data.csv ----------
    # Primary outcome: treatment vs control raw scores
    ctrl_mean = round(rng.uniform(40, 70), 1)
    ctrl_sd = round(rng.uniform(8, 18), 1)
    treat_mean_real = round(ctrl_mean - rng.uniform(1, 6), 1)
    treat_sd = round(rng.uniform(8, 18), 1)

    # Generate raw participant data
    ctrl_scores = [round(rng.gauss(ctrl_mean, ctrl_sd), 1) for _ in range(n_analyzed_ctrl)]
    treat_scores = [round(rng.gauss(treat_mean_real, treat_sd), 1) for _ in range(n_analyzed_treat)]

    # Compute real statistics from the generated data
    real_ctrl_mean = round(sum(ctrl_scores) / len(ctrl_scores), 2)
    real_treat_mean = round(sum(treat_scores) / len(treat_scores), 2)
    real_ctrl_sd = round((sum((x - real_ctrl_mean) ** 2 for x in ctrl_scores) / (len(ctrl_scores) - 1)) ** 0.5, 2)
    real_treat_sd = round((sum((x - real_treat_mean) ** 2 for x in treat_scores) / (len(treat_scores) - 1)) ** 0.5, 2)

    # Compute real p-value (two-sample t-test approximation)
    pooled_se = ((real_ctrl_sd ** 2 / len(ctrl_scores)) + (real_treat_sd ** 2 / len(treat_scores))) ** 0.5
    t_stat = abs(real_ctrl_mean - real_treat_mean) / pooled_se if pooled_se > 0 else 0
    # Approximate p-value using a rough normal approximation for large samples
    real_p = round(2 * (1 - 0.5 * (1 + math.erf(t_stat / math.sqrt(2)))), 3)
    real_p = max(real_p, 0.001)

    # Secondary outcome data
    sec_ctrl_scores = [round(rng.gauss(55, 12), 1) for _ in range(n_analyzed_ctrl)]
    sec_treat_scores = [round(rng.gauss(58, 12), 1) for _ in range(n_analyzed_treat)]
    sec_ctrl_mean = round(sum(sec_ctrl_scores) / len(sec_ctrl_scores), 2)
    sec_treat_mean = round(sum(sec_treat_scores) / len(sec_treat_scores), 2)
    sec_ctrl_sd = round((sum((x - sec_ctrl_mean) ** 2 for x in sec_ctrl_scores) / (len(sec_ctrl_scores) - 1)) ** 0.5, 2)
    sec_treat_sd = round((sum((x - sec_treat_mean) ** 2 for x in sec_treat_scores) / (len(sec_treat_scores) - 1)) ** 0.5, 2)
    sec_se = ((sec_ctrl_sd ** 2 / len(sec_ctrl_scores)) + (sec_treat_sd ** 2 / len(sec_treat_scores))) ** 0.5
    sec_t = abs(sec_ctrl_mean - sec_treat_mean) / sec_se if sec_se > 0 else 0
    sec_p = round(2 * (1 - 0.5 * (1 + math.erf(sec_t / math.sqrt(2)))), 3)
    sec_p = max(sec_p, 0.001)

    # Subgroup data (age-based split)
    age_cutoff = rng.choice([40, 45, 50])
    n_young_treat = rng.randint(n_analyzed_treat // 3, n_analyzed_treat // 2)
    n_old_treat = n_analyzed_treat - n_young_treat
    n_young_ctrl = rng.randint(n_analyzed_ctrl // 3, n_analyzed_ctrl // 2)
    n_old_ctrl = n_analyzed_ctrl - n_young_ctrl

    young_treat_scores = treat_scores[:n_young_treat]
    old_treat_scores = treat_scores[n_young_treat:]
    young_ctrl_scores = ctrl_scores[:n_young_ctrl]
    old_ctrl_scores = ctrl_scores[n_young_ctrl:]

    young_treat_mean = round(sum(young_treat_scores) / len(young_treat_scores), 2) if young_treat_scores else 0
    old_treat_mean = round(sum(old_treat_scores) / len(old_treat_scores), 2) if old_treat_scores else 0
    young_ctrl_mean = round(sum(young_ctrl_scores) / len(young_ctrl_scores), 2) if young_ctrl_scores else 0
    old_ctrl_mean = round(sum(old_ctrl_scores) / len(old_ctrl_scores), 2) if old_ctrl_scores else 0

    # ---------- Manuscript text construction ----------
    # Determine what the manuscript claims (may differ from real data)
    if "p_hacking" in planted_flaws:
        reported_p = round(rng.uniform(0.035, 0.049), 3)
    else:
        reported_p = real_p

    if "implausible_effect" in planted_flaws:
        reported_effect_d = round(rng.uniform(1.5, 2.5), 2)
    else:
        raw_d = abs(real_ctrl_mean - real_treat_mean) / ((real_ctrl_sd + real_treat_sd) / 2) if (real_ctrl_sd + real_treat_sd) > 0 else 0
        reported_effect_d = round(raw_d, 2)

    real_effect_d = abs(real_ctrl_mean - real_treat_mean) / ((real_ctrl_sd + real_treat_sd) / 2) if (real_ctrl_sd + real_treat_sd) > 0 else 0
    real_effect_d = round(real_effect_d, 2)

    if "abstract_results_conflict" in planted_flaws:
        abstract_pct_improvement = rng.randint(25, 40)
    else:
        real_pct = abs(real_ctrl_mean - real_treat_mean) / real_ctrl_mean * 100 if real_ctrl_mean != 0 else 0
        abstract_pct_improvement = round(real_pct, 1)

    if "endpoint_switching" in planted_flaws:
        # Manuscript claims secondary outcome was primary
        manuscript_primary = domain["outcome_secondary"]
        manuscript_secondary = domain["outcome_primary"]
    else:
        manuscript_primary = domain["outcome_primary"]
        manuscript_secondary = domain["outcome_secondary"]

    if "missing_control" in planted_flaws:
        ctrl_description_note = "A historical comparison cohort from institutional records was used as reference."
    else:
        ctrl_description_note = f"Participants were randomly assigned to treatment (n={n_treatment}) or placebo control (n={n_control}) groups using computer-generated block randomization."

    if "wrong_statistical_test" in planted_flaws:
        reported_test = "paired t-test"
        correct_test = "independent two-sample t-test (or Mann-Whitney U for non-normal data)"
    else:
        reported_test = "independent two-sample t-test"
        correct_test = reported_test

    if "multiple_comparisons" in planted_flaws:
        n_tests_run = rng.randint(8, 15)
        correction_applied = False
    else:
        n_tests_run = 2
        correction_applied = True

    if "small_sample_underpowered" in planted_flaws:
        reported_power = None  # No power analysis mentioned
    else:
        reported_power = rng.choice([0.80, 0.85, 0.90])

    if "missing_ci" in planted_flaws:
        ci_reported = False
    else:
        ci_reported = True

    if "selective_reporting" in planted_flaws:
        # Manuscript omits secondary outcome results entirely
        secondary_reported = False
    else:
        secondary_reported = True

    # Compute real percent improvement from data
    real_pct_improvement = round(abs(real_ctrl_mean - real_treat_mean) / real_ctrl_mean * 100, 1) if real_ctrl_mean != 0 else 0.0

    # Build the manuscript
    manuscript_lines = []
    manuscript_lines.append(f"TITLE: Efficacy of {domain['intervention']} on {manuscript_primary}")
    manuscript_lines.append(f"       in {domain['population']}: A Randomized Controlled Trial")
    manuscript_lines.append("")
    manuscript_lines.append(f"Authors: {', '.join(authors)}")
    manuscript_lines.append(f"Submitted to: {domain['journal']}")
    manuscript_lines.append("")
    manuscript_lines.append("=" * 70)
    manuscript_lines.append("ABSTRACT")
    manuscript_lines.append("=" * 70)
    manuscript_lines.append("")
    manuscript_lines.append(f"Background: {domain['intervention'].capitalize()} has shown promise in")
    manuscript_lines.append(f"preliminary studies for improving {manuscript_primary} among")
    manuscript_lines.append(f"{domain['population']}.")
    manuscript_lines.append("")
    manuscript_lines.append(f"Methods: We enrolled {n_enrolled} participants in a randomized trial.")
    manuscript_lines.append(f"{ctrl_description_note}")
    manuscript_lines.append(f"The primary endpoint was {manuscript_primary}.")
    manuscript_lines.append("")
    manuscript_lines.append(f"Results: The treatment group showed a {abstract_pct_improvement}% improvement")
    manuscript_lines.append(f"in {manuscript_primary} compared to control (p={_fmt_pval(reported_p)}).")
    manuscript_lines.append(f"Cohen's d = {reported_effect_d}.")
    manuscript_lines.append("")
    manuscript_lines.append("Conclusions: These findings support the efficacy of the intervention.")
    manuscript_lines.append("")
    manuscript_lines.append("=" * 70)
    manuscript_lines.append("1. INTRODUCTION")
    manuscript_lines.append("=" * 70)
    manuscript_lines.append("")
    manuscript_lines.append(f"The prevalence of conditions affecting {domain['population']} has")
    manuscript_lines.append("increased substantially over the past decade. Current treatments are")
    manuscript_lines.append("limited in efficacy and often carry significant side effects.")
    manuscript_lines.append(f"{domain['intervention'].capitalize()} represents a novel approach that")
    manuscript_lines.append("targets the underlying pathophysiology through a different mechanism.")
    manuscript_lines.append("")
    manuscript_lines.append("Previous work by Henderson et al. (2019) and Nakamura et al. (2020)")
    manuscript_lines.append("demonstrated preliminary evidence of benefit in small pilot studies,")
    manuscript_lines.append("though neither achieved statistical significance. Our study aimed to")
    manuscript_lines.append("provide definitive evidence through a properly powered trial.")
    manuscript_lines.append("")
    manuscript_lines.append("=" * 70)
    manuscript_lines.append("2. METHODS")
    manuscript_lines.append("=" * 70)
    manuscript_lines.append("")
    manuscript_lines.append("2.1 Study Design")
    manuscript_lines.append("")
    manuscript_lines.append(f"This was a {'randomized, double-blind, placebo-controlled' if 'missing_control' not in planted_flaws else 'prospective cohort'} study")
    manuscript_lines.append(f"conducted at a single academic medical center. {n_enrolled} participants")
    manuscript_lines.append(f"from {domain['population']} were enrolled between January and June 2023.")
    manuscript_lines.append("")
    manuscript_lines.append("2.2 Participants")
    manuscript_lines.append("")
    manuscript_lines.append(f"Inclusion criteria: {domain['population']}, willing to comply with")
    manuscript_lines.append("study procedures, able to provide informed consent.")
    manuscript_lines.append("Exclusion criteria: concurrent use of similar interventions, pregnancy,")
    manuscript_lines.append("severe comorbidities.")
    manuscript_lines.append("")
    manuscript_lines.append("2.3 Intervention")
    manuscript_lines.append("")
    manuscript_lines.append(f"The treatment group received the {domain['intervention']} for 12 weeks.")
    manuscript_lines.append(f"{ctrl_description_note}")
    manuscript_lines.append("")
    manuscript_lines.append("2.4 Outcomes")
    manuscript_lines.append("")
    manuscript_lines.append(f"The primary outcome was {manuscript_primary} measured at baseline and")
    manuscript_lines.append(f"12 weeks. Secondary outcomes included {manuscript_secondary}.")
    if "missing_itt" in planted_flaws:
        manuscript_lines.append("Analysis was performed on per-protocol completers.")
    else:
        manuscript_lines.append("Analysis followed intention-to-treat principles.")
    manuscript_lines.append("")
    manuscript_lines.append("2.5 Statistical Analysis")
    manuscript_lines.append("")
    manuscript_lines.append(f"Between-group differences were assessed using {reported_test}.")
    if reported_power is not None:
        manuscript_lines.append(f"Sample size was calculated to achieve {reported_power*100:.0f}% power to")
        manuscript_lines.append(f"detect a medium effect size (d=0.5) at alpha=0.05.")
    manuscript_lines.append(f"A significance level of alpha=0.05 was used throughout.")
    if "multiple_comparisons" in planted_flaws:
        manuscript_lines.append(f"We examined {n_tests_run} endpoints and subgroup analyses.")
    manuscript_lines.append("")
    manuscript_lines.append("=" * 70)
    manuscript_lines.append("3. RESULTS")
    manuscript_lines.append("=" * 70)
    manuscript_lines.append("")
    manuscript_lines.append(f"Of {n_enrolled} enrolled participants, {n_dropout_ctrl + n_dropout_treat} discontinued")
    manuscript_lines.append(f"({n_dropout_ctrl} control, {n_dropout_treat} treatment).")
    if "missing_itt" in planted_flaws:
        manuscript_lines.append(f"Per-protocol analysis included {n_analyzed_ctrl} control and")
        manuscript_lines.append(f"{n_analyzed_treat} treatment participants.")
    else:
        manuscript_lines.append(f"ITT analysis included all {n_enrolled} randomized participants.")
    manuscript_lines.append("")
    manuscript_lines.append("3.1 Primary Outcome")
    manuscript_lines.append("")
    manuscript_lines.append(f"The treatment group showed a statistically significant improvement")
    manuscript_lines.append(f"in {manuscript_primary} (p={_fmt_pval(reported_p)}, {reported_test}).")
    manuscript_lines.append(f"Effect size: Cohen's d = {reported_effect_d}.")
    if ci_reported:
        diff = abs(real_ctrl_mean - real_treat_mean)
        ci_low = round(diff - 1.96 * pooled_se, 2)
        ci_high = round(diff + 1.96 * pooled_se, 2)
        manuscript_lines.append(f"Mean difference: {round(diff,2)} (95% CI: {ci_low} to {ci_high}).")
    manuscript_lines.append("")
    if secondary_reported:
        manuscript_lines.append("3.2 Secondary Outcomes")
        manuscript_lines.append("")
        manuscript_lines.append(f"{domain['outcome_secondary'].capitalize()}: treatment group mean")
        manuscript_lines.append(f"{sec_treat_mean} vs control {sec_ctrl_mean} (p={_fmt_pval(sec_p)}).")
    if "undisclosed_subgroups" in planted_flaws:
        # Manuscript reports a subgroup result but doesn't mention it was post-hoc
        manuscript_lines.append("")
        manuscript_lines.append("3.3 Age-Stratified Analysis")
        manuscript_lines.append("")
        manuscript_lines.append(f"Participants aged <{age_cutoff} showed particularly strong response")
        manuscript_lines.append(f"(treatment mean {young_treat_mean} vs control {young_ctrl_mean}).")
        manuscript_lines.append(f"Participants aged >={age_cutoff} showed less improvement")
        manuscript_lines.append(f"(treatment mean {old_treat_mean} vs control {old_ctrl_mean}).")
    manuscript_lines.append("")
    manuscript_lines.append("=" * 70)
    manuscript_lines.append("4. DISCUSSION")
    manuscript_lines.append("=" * 70)
    manuscript_lines.append("")
    manuscript_lines.append(f"Our findings demonstrate that {domain['intervention']} significantly")
    manuscript_lines.append(f"improves {manuscript_primary} in {domain['population']}.")
    manuscript_lines.append("The effect size observed is clinically meaningful and consistent")
    manuscript_lines.append("with the preliminary evidence from earlier pilot studies.")
    manuscript_lines.append("")
    manuscript_lines.append("Limitations include the single-center design and 12-week duration.")
    manuscript_lines.append("Future multi-center trials with longer follow-up are warranted.")
    manuscript_lines.append("")
    manuscript_content = "\n".join(manuscript_lines) + "\n"

    # ---------- statistical_reference.txt ----------
    stat_ref_lines = [
        "STATISTICAL METHODS REFERENCE GUIDE",
        "",
        "=" * 60,
        "1. CHOOSING THE CORRECT TEST",
        "=" * 60,
        "",
        "Independent two-sample t-test: Compares means of two INDEPENDENT groups.",
        "  - Groups must be different participants (not matched/paired).",
        "  - Assumes approximately normal distribution or large sample size.",
        "",
        "Paired t-test: Compares means from the SAME participants measured twice.",
        "  - Used when each participant serves as their own control.",
        "  - Requires paired observations (e.g., pre/post on same person).",
        "",
        "Mann-Whitney U test: Non-parametric alternative to independent t-test.",
        "  - Used when normality assumption is violated and sample is small.",
        "",
        "ANOVA: Compares means across 3+ groups simultaneously.",
        "",
        "Chi-square test: Compares proportions/frequencies between groups.",
        "",
        "=" * 60,
        "2. MULTIPLE COMPARISONS",
        "=" * 60,
        "",
        "When multiple statistical tests are performed, the family-wise error",
        "rate increases. Corrections are REQUIRED when performing >2 tests:",
        "",
        "  Bonferroni correction: Divide alpha by the number of tests.",
        "    Example: 10 tests at alpha=0.05 -> use alpha=0.005 per test.",
        "",
        "  Holm-Bonferroni: Step-down procedure, slightly more powerful.",
        "",
        "  Benjamini-Hochberg: Controls false discovery rate (FDR).",
        "",
        "Without correction, running K tests at alpha=0.05 gives:",
        "  P(at least one false positive) = 1 - (0.95)^K",
        "  For K=10: P = 40%. For K=15: P = 54%.",
        "",
        "=" * 60,
        "3. EFFECT SIZE INTERPRETATION",
        "=" * 60,
        "",
        "Cohen's d benchmarks (for behavioral/medical research):",
        "  Small:  d = 0.2",
        "  Medium: d = 0.5",
        "  Large:  d = 0.8",
        "",
        "Effect sizes above d = 1.0 are uncommon in most fields.",
        "Effect sizes above d = 1.5 should be scrutinized carefully -",
        "they may indicate measurement artifacts, selection bias, or errors.",
        "",
        "=" * 60,
        "4. POWER ANALYSIS",
        "=" * 60,
        "",
        "A properly powered study requires sample size calculation BEFORE",
        "data collection. For detecting d=0.5 at alpha=0.05, 80% power:",
        "  n = 64 per group (128 total)",
        "For detecting d=0.3 at alpha=0.05, 80% power:",
        "  n = 176 per group (352 total)",
        "",
        "Studies without pre-specified power analysis risk being underpowered.",
        "",
        "=" * 60,
        "5. INTENTION-TO-TREAT ANALYSIS",
        "=" * 60,
        "",
        "ITT analysis includes ALL randomized participants regardless of",
        "protocol adherence or dropout. This preserves the benefits of",
        "randomization and avoids attrition bias.",
        "",
        "Per-protocol analysis (completers only) can introduce bias if",
        "dropout is related to treatment response. ITT is the gold standard",
        "for RCTs as recommended by CONSORT guidelines.",
        "",
        "=" * 60,
        "6. CONFIDENCE INTERVALS",
        "=" * 60,
        "",
        "Reporting p-values without confidence intervals is incomplete.",
        "95% CI provides the range of plausible effect sizes and conveys",
        "both statistical significance AND precision of the estimate.",
        "",
    ]
    stat_ref_content = "\n".join(stat_ref_lines) + "\n"

    # ---------- supplementary_data.csv ----------
    age_lo, age_hi = domain["age_range"]
    csv_lines = ["participant_id,group,age,sex,primary_baseline,primary_12wk,secondary_baseline,secondary_12wk"]
    pid = 1
    for i in range(n_analyzed_ctrl):
        age = rng.randint(age_lo, age_hi)
        sex = rng.choice(["M", "F"])
        base_primary = round(rng.gauss(ctrl_mean + 5, ctrl_sd), 1)
        post_primary = ctrl_scores[i]
        base_sec = round(rng.gauss(50, 10), 1)
        post_sec = sec_ctrl_scores[i]
        csv_lines.append(f"P{pid:03d},control,{age},{sex},{base_primary},{post_primary},{base_sec},{post_sec}")
        pid += 1
    for i in range(n_analyzed_treat):
        age = rng.randint(age_lo, age_hi)
        sex = rng.choice(["M", "F"])
        base_primary = round(rng.gauss(ctrl_mean + 5, ctrl_sd), 1)
        post_primary = treat_scores[i]
        base_sec = round(rng.gauss(50, 10), 1)
        post_sec = sec_treat_scores[i]
        csv_lines.append(f"P{pid:03d},treatment,{age},{sex},{base_primary},{post_primary},{base_sec},{post_sec}")
        pid += 1
    supp_data_content = "\n".join(csv_lines) + "\n"

    # ---------- reporting_guidelines.txt ----------
    guidelines_lines = [
        "REPORTING GUIDELINES CHECKLIST (CONSORT-adapted)",
        "",
        "This checklist is for evaluating randomized controlled trial manuscripts.",
        "",
        "TITLE & ABSTRACT",
        "[ ] Title identifies study as randomized trial",
        "[ ] Abstract includes structured summary of design, methods, results, conclusions",
        "[ ] Abstract results match the detailed results section",
        "",
        "METHODS",
        "[ ] Trial design described (parallel, crossover, factorial)",
        "[ ] Eligibility criteria for participants clearly stated",
        "[ ] Interventions described with sufficient detail for replication",
        "[ ] Primary and secondary outcomes clearly defined a priori",
        "[ ] Sample size calculation described with assumptions",
        "[ ] Randomization method described (sequence generation, allocation concealment)",
        "[ ] Blinding described (who was blinded, method of blinding)",
        "[ ] Statistical methods described, including multiple comparison corrections",
        "",
        "RESULTS",
        "[ ] Participant flow diagram (enrolled, allocated, followed up, analyzed)",
        "[ ] Baseline characteristics of each group",
        "[ ] Number analyzed in each group (ITT vs per-protocol noted)",
        "[ ] Primary outcome: point estimate AND confidence interval",
        "[ ] Secondary outcomes: all pre-specified outcomes reported",
        "[ ] Subgroup analyses: pre-specified vs post-hoc clearly distinguished",
        "[ ] Effect sizes reported with confidence intervals",
        "",
        "DISCUSSION",
        "[ ] Limitations acknowledged",
        "[ ] Generalizability discussed",
        "[ ] Interpretation consistent with results (no overclaiming)",
        "",
        "INTEGRITY CHECKS",
        "[ ] P-values consistent with reported statistics (can be recomputed from data)",
        "[ ] Effect sizes plausible for the field (Cohen's d typically < 1.0 in social/medical)",
        "[ ] Abstract claims match detailed results (percentages, significance, direction)",
        "[ ] All randomized participants accounted for in analysis (ITT principle)",
        "[ ] Pre-registered endpoints match reported endpoints (no switching)",
        "",
    ]
    guidelines_content = "\n".join(guidelines_lines) + "\n"

    # ---------- Build rubric ----------
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does the file /testbed/review_report.txt exist and contain substantive content (more than a few lines)?",
            points=1,
        ),
    ]

    # Flaw-specific rubric items
    if "p_hacking" in planted_flaws:
        rubric_items.append(BinaryRubricCategory(
            name="detects_p_hacking",
            question=(
                f"Does the review identify that the reported p-value (p={_fmt_pval(reported_p)}) does not match "
                f"the p-value computed from the supplementary data (approximately p={_fmt_pval(real_p)})? "
                f"The manuscript reports p={_fmt_pval(reported_p)} but the raw data yields approximately p={_fmt_pval(real_p)}."
            ),
            points=3,
        ))

    if "missing_control" in planted_flaws:
        rubric_items.append(BinaryRubricCategory(
            name="detects_missing_control",
            question=(
                "Does the review identify that the study uses a historical comparison cohort rather than "
                "a proper concurrent randomized control group, which undermines causal inference?"
            ),
            points=2,
        ))

    if "endpoint_switching" in planted_flaws:
        rubric_items.append(BinaryRubricCategory(
            name="detects_endpoint_switching",
            question=(
                f"Does the review identify that the primary and secondary endpoints appear to have been "
                f"switched? The manuscript lists '{manuscript_primary}' as primary, but the study design "
                f"and data structure suggest '{manuscript_secondary}' was the original primary endpoint."
            ),
            points=3,
        ))

    if "undisclosed_subgroups" in planted_flaws:
        rubric_items.append(BinaryRubricCategory(
            name="detects_undisclosed_subgroups",
            question=(
                f"Does the review identify that the age-stratified analysis (split at age {age_cutoff}) "
                f"appears to be a post-hoc subgroup analysis that was not pre-specified, yet is presented "
                f"without labeling it as exploratory?"
            ),
            points=2,
        ))

    if "implausible_effect" in planted_flaws:
        rubric_items.append(BinaryRubricCategory(
            name="detects_implausible_effect",
            question=(
                f"Does the review identify that the reported effect size (Cohen's d = {reported_effect_d}) "
                f"is implausibly large for this field? The actual effect size computed from the raw data "
                f"is approximately d = {real_effect_d}, and values above 1.0-1.5 should be scrutinized."
            ),
            points=3,
        ))

    if "abstract_results_conflict" in planted_flaws:
        rubric_items.append(BinaryRubricCategory(
            name="detects_abstract_conflict",
            question=(
                f"Does the review identify that the abstract claims a {abstract_pct_improvement}% improvement "
                f"but the actual data shows approximately {real_pct_improvement}% improvement? The abstract "
                f"overstates the results."
            ),
            points=2,
        ))

    if "missing_itt" in planted_flaws:
        rubric_items.append(BinaryRubricCategory(
            name="detects_missing_itt",
            question=(
                "Does the review identify that the study uses per-protocol analysis instead of "
                "intention-to-treat analysis, which can introduce attrition bias? "
                f"{n_dropout_ctrl + n_dropout_treat} participants dropped out and were excluded from analysis."
            ),
            points=2,
        ))

    if "multiple_comparisons" in planted_flaws:
        rubric_items.append(BinaryRubricCategory(
            name="detects_multiple_comparisons",
            question=(
                f"Does the review identify that {n_tests_run} statistical tests were performed without "
                f"any multiple comparison correction (e.g., Bonferroni, Holm, or FDR adjustment)? "
                f"At alpha=0.05 with {n_tests_run} tests, the family-wise error rate is approximately "
                f"{_fmt_pct(100 * (1 - 0.95**n_tests_run))}."
            ),
            points=2,
        ))

    if "small_sample_underpowered" in planted_flaws:
        rubric_items.append(BinaryRubricCategory(
            name="detects_underpowered",
            question=(
                "Does the review identify that no power analysis or sample size calculation is reported, "
                "and that the study may be underpowered to detect the claimed effect?"
            ),
            points=2,
        ))

    if "selective_reporting" in planted_flaws:
        rubric_items.append(BinaryRubricCategory(
            name="detects_selective_reporting",
            question=(
                f"Does the review identify that the secondary outcome ({domain['outcome_secondary']}) "
                f"data exists in the supplementary data file but is not reported in the manuscript results? "
                f"The secondary outcome shows p={_fmt_pval(sec_p)}, which may explain its omission."
            ),
            points=3,
        ))

    if "wrong_statistical_test" in planted_flaws:
        rubric_items.append(BinaryRubricCategory(
            name="detects_wrong_test",
            question=(
                f"Does the review identify that a {reported_test} was used when the study design "
                f"(two independent groups) calls for a {correct_test}? A paired test is inappropriate "
                f"for independent group comparisons."
            ),
            points=2,
        ))

    if "missing_ci" in planted_flaws:
        rubric_items.append(BinaryRubricCategory(
            name="detects_missing_ci",
            question=(
                "Does the review identify that confidence intervals are not reported for the primary "
                "outcome, which is required by CONSORT reporting guidelines?"
            ),
            points=2,
        ))

    # Cross-referencing checks (always present)
    rubric_items.append(BinaryRubricCategory(
        name="cross_references_data",
        question=(
            "Does the review demonstrate cross-referencing the manuscript claims against the "
            "supplementary_data.csv (e.g., recomputing means, checking sample sizes, or verifying p-values)?"
        ),
        points=2,
    ))

    rubric_items.append(BinaryRubricCategory(
        name="references_guidelines",
        question=(
            "Does the review reference specific reporting guideline items (from reporting_guidelines.txt) "
            "that are violated?"
        ),
        points=1,
    ))

    rubric_items.append(BinaryRubricCategory(
        name="correct_sample_sizes",
        question=(
            f"Does the review correctly note the analyzed sample sizes from the data "
            f"({n_analyzed_ctrl} control, {n_analyzed_treat} treatment) and reconcile them "
            f"with the enrolled count of {n_enrolled}?"
        ),
        points=1,
    ))

    # False-positive checks: things that look suspicious but are fine
    # 1. Dropout rate is within normal range
    dropout_rate = round((n_dropout_ctrl + n_dropout_treat) / n_enrolled * 100, 1)
    rubric_items.append(BinaryRubricCategory(
        name="no_false_positive_dropout",
        question=(
            f"Does the review avoid incorrectly flagging the dropout rate ({dropout_rate}%) as "
            f"problematic? With {n_dropout_ctrl + n_dropout_treat} dropouts out of {n_enrolled} enrolled, "
            f"this rate is within normal range (<20%) for a 12-week trial."
        ),
        points=2,
    ))

    # 2. Single-center design is acknowledged as limitation but not a flaw
    rubric_items.append(BinaryRubricCategory(
        name="no_false_positive_single_center",
        question=(
            "Does the review avoid incorrectly calling the single-center design a methodological flaw? "
            "While it limits generalizability (which the authors acknowledge), it is not a methodological error."
        ),
        points=2,
    ))

    # 3. The 12-week duration is standard for this type of study
    rubric_items.append(BinaryRubricCategory(
        name="no_false_positive_duration",
        question=(
            "Does the review avoid incorrectly flagging the 12-week study duration as too short? "
            "12 weeks is a standard duration for this type of intervention trial."
        ),
        points=1,
    ))

    # Overall quality (the one graded category)
    rubric_items.append(RubricCategory(
        name="review_quality",
        description=(
            "Overall quality of the peer review: Is it well-organized, specific, and actionable?"
        ),
        failure="Review is superficial, vague, or largely incorrect.",
        minor_failure="Review identifies some issues but misses major flaws or lacks specificity.",
        minor_success="Review identifies most planted flaws with reasonable specificity and references the data.",
        success="Thorough, well-structured review that identifies all major flaws with specific evidence from the data files, distinguishes real issues from non-issues, and provides constructive feedback.",
        points=3,
    ))

    # Pad to ensure 15-25 categories: add structural checks if needed
    if len(rubric_items) < 15:
        rubric_items.insert(1, BinaryRubricCategory(
            name="identifies_study_design",
            question=(
                f"Does the review correctly describe the study as a trial in the field of "
                f"{domain['field']} involving {domain['population']}?"
            ),
            points=1,
        ))
    if len(rubric_items) < 15:
        rubric_items.insert(2, BinaryRubricCategory(
            name="identifies_lead_author",
            question=f"Does the review reference the lead author ({lead_author}) or the full author list?",
            points=1,
        ))
    if len(rubric_items) < 15:
        rubric_items.insert(3, BinaryRubricCategory(
            name="lists_all_flaws_found",
            question="Does the review provide a summary or enumerated list of all identified issues?",
            points=1,
        ))

    problem_statement = f"""# Peer Review: Scientific Manuscript Analysis

You are an expert peer reviewer evaluating a submitted manuscript for methodological rigor.

## Source Files
- /testbed/data/manuscript.txt -- The submitted research manuscript
- /testbed/data/statistical_reference.txt -- Reference guide on proper statistical methods
- /testbed/data/supplementary_data.csv -- Raw experimental data from the study
- /testbed/data/reporting_guidelines.txt -- CONSORT-adapted reporting checklist

## Task
1. Read the manuscript carefully, noting any methodological claims
2. Cross-reference the manuscript's reported statistics against the raw data in supplementary_data.csv
3. Check the manuscript against the reporting guidelines checklist
4. Consult the statistical reference for proper methodology
5. Identify all methodological flaws, statistical errors, and reporting deficiencies
6. Be specific: cite exact values, page references, and data discrepancies
7. Distinguish genuine flaws from acceptable limitations

Write a detailed peer review report to /testbed/review_report.txt"""

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=tuple(rubric_items),
        submission_instructions="Write your peer review report to /testbed/review_report.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/manuscript.txt": manuscript_content,
            "/testbed/data/statistical_reference.txt": stat_ref_content,
            "/testbed/data/supplementary_data.csv": supp_data_content,
            "/testbed/data/reporting_guidelines.txt": guidelines_content,
        },
        problem_type="peer_review_analysis",
    )


# =============================================================================
# DOMAIN POOLS: EXPERIMENT PROTOCOL AUDIT
# =============================================================================

_LAB_TYPES = [
    {
        "name": "Organic Chemistry Synthesis Lab",
        "focus": "synthesis of pharmaceutical intermediates",
        "biosafety": "BSL-1",
        "chemicals": ["ethanol", "dichloromethane", "sodium hydroxide", "hydrochloric acid",
                       "diethyl ether", "toluene", "acetic anhydride", "triethylamine"],
    },
    {
        "name": "Microbiology Research Lab",
        "focus": "characterization of environmental bacterial isolates",
        "biosafety": "BSL-2",
        "chemicals": ["ethanol", "bleach solution", "crystal violet", "safranin",
                       "nutrient broth", "agar", "hydrogen peroxide", "formaldehyde"],
    },
    {
        "name": "Analytical Chemistry Lab",
        "focus": "trace metal analysis in environmental water samples",
        "biosafety": "BSL-1",
        "chemicals": ["nitric acid", "hydrochloric acid", "standard reference solutions",
                       "acetone", "deionized water", "EDTA solution", "sulfuric acid", "ammonia solution"],
    },
    {
        "name": "Cell Culture and Tissue Engineering Lab",
        "focus": "development of scaffolds for cartilage regeneration",
        "biosafety": "BSL-2",
        "chemicals": ["DMEM media", "fetal bovine serum", "trypsin-EDTA", "PBS buffer",
                       "paraformaldehyde", "dimethyl sulfoxide", "collagenase", "glutaraldehyde"],
    },
    {
        "name": "Biochemistry and Enzyme Kinetics Lab",
        "focus": "kinetic characterization of novel enzyme inhibitors",
        "biosafety": "BSL-1",
        "chemicals": ["Tris buffer", "EDTA", "beta-mercaptoethanol", "SDS",
                       "acrylamide", "ammonium persulfate", "TEMED", "Coomassie blue"],
    },
    {
        "name": "Environmental Toxicology Lab",
        "focus": "assessing pesticide effects on aquatic organisms",
        "biosafety": "BSL-1",
        "chemicals": ["chlorpyrifos", "atrazine", "malathion", "DMSO",
                       "acetone", "methanol", "sodium thiosulfate", "calcium hypochlorite"],
    },
    {
        "name": "Virology Research Lab",
        "focus": "characterization of novel respiratory virus strains",
        "biosafety": "BSL-3",
        "chemicals": ["ethanol", "bleach solution", "TRIzol reagent", "chloroform",
                       "isopropanol", "formaldehyde", "paraformaldehyde", "RNase inhibitor"],
    },
    {
        "name": "Materials Science and Nanotechnology Lab",
        "focus": "synthesis and characterization of metal nanoparticles",
        "biosafety": "BSL-1",
        "chemicals": ["silver nitrate", "sodium borohydride", "citric acid", "chloroauric acid",
                       "hydrazine", "ethylene glycol", "polyvinylpyrrolidone", "nitric acid"],
    },
]

_ISSUE_POOL_PROTOCOL = [
    "expired_calibration",
    "missing_ppe",
    "incompatible_storage",
    "missing_emergency_procedure",
    "dosage_error",
    "insufficient_washout",
    "missing_consent",
    "wrong_biosafety_level",
    "contradicts_incident",
    "missing_waste_disposal",
    "ventilation_inadequate",
    "training_not_documented",
]

_EQUIPMENT_TYPES = [
    ("Analytical Balance", "AB"),
    ("pH Meter", "PH"),
    ("Spectrophotometer", "SP"),
    ("Centrifuge", "CF"),
    ("Autoclave", "AC"),
    ("Fume Hood", "FH"),
    ("Biosafety Cabinet", "BC"),
    ("HPLC System", "HP"),
    ("PCR Thermocycler", "TC"),
    ("Microscope", "MS"),
    ("Incubator", "IC"),
    ("Water Bath", "WB"),
    ("Vortex Mixer", "VX"),
    ("Micropipette Set", "MP"),
    ("Rotary Evaporator", "RE"),
]


# =============================================================================
# 2. EXPERIMENT PROTOCOL AUDIT
# =============================================================================


def make_experiment_protocol_audit(rand_seed: int = 42) -> RubricDatapoint:
    """Audit a lab experiment protocol for safety, ethics, and scientific rigor
    by cross-referencing against safety regulations, equipment calibration records,
    and past incident reports.

    Seed varies: lab type, planted issues (4-7 from 12), equipment inventory,
    incident history, and which items are false-positive distractors.
    """
    rng = _random.Random(rand_seed)

    lab = rng.choice(_LAB_TYPES)
    pi_name = random_name(rand_seed + 1)
    lab_manager = random_name(rand_seed + 2)
    protocol_id = f"PROT-{rng.randint(2023, 2025)}-{rng.randint(100, 999)}"

    # Pick issues to plant (4-7)
    n_issues = rng.randint(4, 7)
    all_issues = list(_ISSUE_POOL_PROTOCOL)
    rng.shuffle(all_issues)
    planted_issues = all_issues[:n_issues]

    # ---------- Equipment inventory ----------
    n_equipment = rng.randint(10, 14)
    available_equipment = list(_EQUIPMENT_TYPES)
    rng.shuffle(available_equipment)
    equipment_list = available_equipment[:n_equipment]

    # Decide which equipment has expired calibration
    if "expired_calibration" in planted_issues:
        n_expired = rng.randint(2, 4)
        expired_indices = rng.sample(range(n_equipment), n_expired)
    else:
        n_expired = 0
        expired_indices = []

    equipment_rows = []
    expired_items = []
    for idx, (eq_name, eq_code) in enumerate(equipment_list):
        eq_id = f"{eq_code}-{rng.randint(1000, 9999)}"
        serial = f"SN-{rng.randint(100000, 999999)}"
        location = f"Room {rng.choice(['A', 'B', 'C'])}-{rng.randint(100, 350)}"

        if idx in expired_indices:
            # Expired: last calibrated 14-24 months ago
            months_ago = rng.randint(14, 24)
            cal_year = 2024 - (months_ago // 12)
            cal_month = 12 - (months_ago % 12)
            if cal_month <= 0:
                cal_month += 12
                cal_year -= 1
            cal_date = f"{cal_year}-{cal_month:02d}-{rng.randint(1, 28):02d}"
            next_due = f"{cal_year + 1}-{cal_month:02d}-{rng.randint(1, 28):02d}"
            status = "Overdue"
            expired_items.append((eq_name, eq_id, cal_date, next_due))
        else:
            # Current: calibrated 1-6 months ago
            months_ago = rng.randint(1, 6)
            cal_month = 12 - months_ago
            if cal_month <= 0:
                cal_month += 12
            cal_date = f"2024-{cal_month:02d}-{rng.randint(1, 28):02d}"
            next_year = 2025 if cal_month < 7 else 2025
            next_due = f"{next_year}-{cal_month:02d}-{rng.randint(1, 28):02d}"
            status = "Current"

        equipment_rows.append({
            "id": eq_id, "name": eq_name, "serial": serial,
            "location": location, "last_cal": cal_date,
            "next_due": next_due, "status": status,
        })

    equip_csv_lines = ["equipment_id,equipment_name,serial_number,location,last_calibration,next_due,status"]
    for r in equipment_rows:
        equip_csv_lines.append(
            f"{r['id']},{r['name']},{r['serial']},{r['location']},"
            f"{r['last_cal']},{r['next_due']},{r['status']}"
        )
    equipment_csv_content = "\n".join(equip_csv_lines) + "\n"

    # ---------- Protocol text ----------
    protocol_chemicals = rng.sample(lab["chemicals"], min(rng.randint(4, 6), len(lab["chemicals"])))

    # Determine wrong biosafety level if applicable
    if "wrong_biosafety_level" in planted_issues:
        if lab["biosafety"] == "BSL-2":
            protocol_bsl = "BSL-1"
        elif lab["biosafety"] == "BSL-3":
            protocol_bsl = "BSL-2"
        else:
            protocol_bsl = lab["biosafety"]
            # Remove flaw if we can't actually create the discrepancy
            planted_issues = [x for x in planted_issues if x != "wrong_biosafety_level"]
    else:
        protocol_bsl = lab["biosafety"]

    # PPE requirements that should be in protocol
    required_ppe = ["lab coat", "safety goggles", "closed-toe shoes"]
    if any(c in protocol_chemicals for c in ["hydrochloric acid", "nitric acid", "sulfuric acid", "sodium hydroxide", "hydrazine"]):
        required_ppe.append("chemical-resistant gloves")
        required_ppe.append("face shield for splash hazard")
    if lab["biosafety"] in ("BSL-2", "BSL-3"):
        required_ppe.append("nitrile gloves (double-gloved)")
    if lab["biosafety"] == "BSL-3":
        required_ppe.append("N95 respirator or PAPR")

    if "missing_ppe" in planted_issues:
        # Remove 2-3 critical PPE items from what the protocol lists
        n_ppe_missing = rng.randint(2, min(3, len(required_ppe)))
        rng.shuffle(required_ppe)
        missing_ppe = required_ppe[:n_ppe_missing]
        listed_ppe = required_ppe[n_ppe_missing:]
    else:
        missing_ppe = []
        listed_ppe = required_ppe

    # Incompatible chemical storage
    if "incompatible_storage" in planted_issues:
        # Pick two chemicals that shouldn't be stored together
        acid_chems = [c for c in protocol_chemicals if "acid" in c.lower()]
        base_chems = [c for c in protocol_chemicals if any(b in c.lower() for b in ["hydroxide", "ammonia", "amine"])]
        oxidizer_chems = [c for c in protocol_chemicals if any(o in c.lower() for o in ["peroxide", "persulfate", "nitrate", "hypochlorite"])]
        flammable_chems = [c for c in protocol_chemicals if any(f in c.lower() for f in ["ether", "ethanol", "acetone", "methanol", "toluene"])]

        incompatible_pair = None
        if acid_chems and base_chems:
            incompatible_pair = (rng.choice(acid_chems), rng.choice(base_chems), "Acids and bases must be stored separately")
        elif oxidizer_chems and flammable_chems:
            incompatible_pair = (rng.choice(oxidizer_chems), rng.choice(flammable_chems), "Oxidizers and flammables must be stored separately")
        else:
            # Force a pair
            if len(protocol_chemicals) >= 2:
                incompatible_pair = (protocol_chemicals[0], protocol_chemicals[1], "These chemicals are from incompatible storage groups")
    else:
        incompatible_pair = None

    # Dosage error
    if "dosage_error" in planted_issues:
        correct_concentration = round(rng.uniform(0.5, 5.0), 1)
        wrong_concentration = round(correct_concentration * rng.choice([10, 0.1]), 1)
        dosage_chemical = rng.choice(protocol_chemicals)
    else:
        correct_concentration = None
        wrong_concentration = None
        dosage_chemical = None

    # Washout period
    if "insufficient_washout" in planted_issues:
        required_washout_days = rng.randint(14, 28)
        stated_washout_days = rng.randint(2, 7)
    else:
        required_washout_days = None
        stated_washout_days = None

    protocol_lines = [
        f"EXPERIMENT PROTOCOL: {protocol_id}",
        f"Laboratory: {lab['name']}",
        f"Principal Investigator: {pi_name}",
        f"Lab Manager: {lab_manager}",
        f"Date: 2024-09-15",
        f"Biosafety Level: {protocol_bsl}",
        "",
        "=" * 60,
        "1. OBJECTIVE",
        "=" * 60,
        "",
        f"To conduct {lab['focus']} using established techniques",
        "with modifications to improve yield and reduce processing time.",
        "",
        "=" * 60,
        "2. MATERIALS",
        "=" * 60,
        "",
        "2.1 Chemicals:",
    ]
    for chem in protocol_chemicals:
        if dosage_chemical and chem == dosage_chemical:
            protocol_lines.append(f"  - {chem}: {wrong_concentration} M working solution")
        else:
            protocol_lines.append(f"  - {chem}")

    protocol_lines.extend([
        "",
        "2.2 Equipment:",
    ])
    for r in equipment_rows[:6]:
        protocol_lines.append(f"  - {r['name']} ({r['id']})")

    protocol_lines.extend([
        "",
        "2.3 Personal Protective Equipment:",
    ])
    for ppe in listed_ppe:
        protocol_lines.append(f"  - {ppe}")

    if incompatible_pair:
        protocol_lines.extend([
            "",
            "2.4 Chemical Storage:",
            f"  All chemicals stored in Cabinet C-1 (Room A-201):",
            f"  - Shelf 1: {incompatible_pair[0]}, {incompatible_pair[1]}",
            f"  - Shelf 2: {', '.join(c for c in protocol_chemicals if c not in (incompatible_pair[0], incompatible_pair[1]))}",
        ])

    protocol_lines.extend([
        "",
        "=" * 60,
        "3. PROCEDURE",
        "=" * 60,
        "",
        "Step 1: Prepare workspace and verify equipment calibration.",
        "Step 2: Prepare reagent solutions according to concentrations listed above.",
        "Step 3: Begin experimental procedure following standard operating protocol.",
    ])

    if "insufficient_washout" in planted_issues:
        protocol_lines.extend([
            f"Step 4: Between experimental runs, allow a {stated_washout_days}-day washout period.",
            "Step 5: Repeat steps 2-4 for all experimental conditions.",
        ])
    else:
        protocol_lines.extend([
            "Step 4: Record all observations and measurements.",
            "Step 5: Repeat for all experimental conditions with adequate intervals.",
        ])

    if "missing_emergency_procedure" in planted_issues:
        protocol_lines.extend([
            "",
            "=" * 60,
            "4. SAFETY",
            "=" * 60,
            "",
            "Follow standard laboratory safety procedures.",
            "Report any incidents to the lab manager.",
        ])
    else:
        protocol_lines.extend([
            "",
            "=" * 60,
            "4. SAFETY AND EMERGENCY PROCEDURES",
            "=" * 60,
            "",
            "4.1 Chemical Spill:",
            "  - Small spill (<100 mL): Use spill kit, absorb, place in chemical waste.",
            "  - Large spill (>100 mL): Evacuate area, notify EHS, call emergency line.",
            "",
            "4.2 Personal Exposure:",
            "  - Skin contact: Remove contaminated clothing, wash with water for 15 minutes.",
            "  - Eye contact: Use eyewash station for 15 minutes, seek medical attention.",
            "  - Inhalation: Move to fresh air immediately, seek medical attention if symptomatic.",
            "",
            "4.3 Fire:",
            "  - Small contained fire: Use appropriate extinguisher (Class B for chemical fires).",
            "  - Uncontained fire: Evacuate, pull fire alarm, call 911.",
            "",
            "4.4 Emergency Contacts:",
            f"  - Lab Manager: {lab_manager}",
            "  - EHS Hotline: ext. 4-SAFE",
            "  - Campus Police: ext. 4-1111",
        ])

    if "missing_consent" in planted_issues and "human" in lab["focus"].lower() or rng.random() < 0.3:
        protocol_lines.extend([
            "",
            "=" * 60,
            "5. HUMAN SUBJECTS (if applicable)",
            "=" * 60,
            "",
            "Participants will be recruited from the university community.",
            "Participation is voluntary.",
            # Missing: actual consent form, IRB approval number, data privacy procedures
        ])

    if "missing_waste_disposal" in planted_issues:
        # No waste disposal section
        pass
    else:
        protocol_lines.extend([
            "",
            "=" * 60,
            f"{'6' if 'missing_consent' in planted_issues else '5'}. WASTE DISPOSAL",
            "=" * 60,
            "",
            "All chemical waste must be segregated by hazard class:",
            "  - Halogenated solvents: Red waste container",
            "  - Non-halogenated solvents: Blue waste container",
            "  - Aqueous acids/bases: Yellow waste container (neutralize if possible)",
            "  - Biological waste: Autoclave before disposal in biohazard bags",
            "",
        ])

    protocol_content = "\n".join(protocol_lines) + "\n"

    # ---------- safety_regulations.txt ----------
    safety_lines = [
        "LABORATORY SAFETY REGULATIONS AND STANDARDS",
        "",
        "=" * 60,
        "1. BIOSAFETY LEVELS",
        "=" * 60,
        "",
        "BSL-1: Basic level. Work with well-characterized agents not known to",
        "  cause disease in healthy adults. Standard microbiological practices.",
        "  PPE: Lab coat, gloves as needed, eye protection for splash risk.",
        "",
        "BSL-2: Agents posing moderate hazard to personnel and environment.",
        "  Includes human-derived blood/tissues, moderate-risk pathogens.",
        "  PPE: Lab coat, gloves (double-gloved recommended), face protection.",
        "  Facility: Biosafety cabinet for procedures generating aerosols.",
        "  Access: Limited when work is in progress.",
        "",
        "BSL-3: Indigenous or exotic agents with potential for aerosol transmission.",
        "  Serious or potentially lethal disease.",
        "  PPE: Dedicated lab clothing, double gloves, N95 respirator or PAPR.",
        "  Facility: Negative pressure, HEPA-filtered exhaust, self-closing doors.",
        "  Access: Controlled access, documented entry/exit.",
        "",
        "=" * 60,
        "2. PERSONAL PROTECTIVE EQUIPMENT (PPE) REQUIREMENTS",
        "=" * 60,
        "",
        "MINIMUM for all labs: lab coat, safety goggles, closed-toe shoes.",
        "",
        "When handling corrosive chemicals (strong acids/bases):",
        "  ADD: chemical-resistant gloves AND face shield for splash hazard.",
        "",
        "When handling flammable solvents:",
        "  ADD: chemical-resistant gloves. Work in fume hood ONLY.",
        "",
        "BSL-2 labs: ADD nitrile gloves (double-gloved).",
        "BSL-3 labs: ADD N95 respirator or PAPR, dedicated lab clothing.",
        "",
        "=" * 60,
        "3. CHEMICAL STORAGE COMPATIBILITY",
        "=" * 60,
        "",
        "Chemicals MUST be segregated by hazard class:",
        "  Group A (Acids): Store with acids only. Separate from bases and organics.",
        "  Group B (Bases/Caustics): Store with bases only. Separate from acids.",
        "  Group C (Flammables): Store in flammable storage cabinet. Away from oxidizers.",
        "  Group D (Oxidizers): Store separately from flammables and organics.",
        "  Group E (Toxics): Store in ventilated, locked cabinet.",
        "",
        "NEVER store acids and bases on the same shelf.",
        "NEVER store oxidizers with flammable solvents.",
        "",
        "=" * 60,
        "4. EQUIPMENT CALIBRATION",
        "=" * 60,
        "",
        "All measuring instruments must be calibrated annually at minimum.",
        "Equipment with calibration overdue by >30 days must NOT be used.",
        "Calibration status must be verified before each use.",
        "Records must be maintained in the equipment inventory.",
        "",
        "=" * 60,
        "5. EMERGENCY PROCEDURES",
        "=" * 60,
        "",
        "ALL protocols MUST include specific emergency procedures for:",
        "  - Chemical spills (small and large)",
        "  - Personal exposure (skin, eye, inhalation)",
        "  - Fire",
        "  - Emergency contacts with phone numbers",
        "",
        "Generic statements like 'follow standard safety procedures' are",
        "INSUFFICIENT and do not meet regulatory requirements.",
        "",
        "=" * 60,
        "6. DOSAGE AND CONCENTRATION STANDARDS",
        "=" * 60,
        "",
        "Working solution concentrations must be verified against published protocols.",
        "Common working concentrations:",
        "  - HCl: 0.1-1.0 M (typical). >6 M is concentrated, requires extra precautions.",
        "  - NaOH: 0.1-1.0 M (typical). >5 M is concentrated.",
        "  - Ethanol: 70% for disinfection, 95-100% for analytical use.",
        "  - Formaldehyde: 3.7-4% for fixation (10% formalin).",
        "",
        "A 10x error in concentration is a serious safety and scientific concern.",
        "",
        "=" * 60,
        "7. WASHOUT AND RECOVERY PERIODS",
        "=" * 60,
        "",
        "For repeated experimental procedures:",
        "  - Chemical exposure studies: minimum 14-day washout between exposures.",
        "  - Cell culture passages: follow established passage schedules.",
        "  - Animal studies: follow IACUC-approved protocols for recovery periods.",
        "",
        "Insufficient washout periods compromise both safety and data quality.",
        "",
    ]
    safety_content = "\n".join(safety_lines) + "\n"

    # ---------- previous_incidents.txt ----------
    incident_templates = [
        {
            "id": "INC-2023-041",
            "date": "2023-03-15",
            "type": "Chemical Spill",
            "desc": f"Concentrated acid spill in {lab['name']} due to improper transfer technique. "
                    "100 mL of acid spilled on bench surface. Area evacuated for 30 minutes.",
            "lesson": "Always use secondary containment when transferring concentrated acids. "
                      "Chemical-resistant gloves and face shield REQUIRED for any acid transfer.",
        },
        {
            "id": "INC-2023-087",
            "date": "2023-07-22",
            "type": "Equipment Failure",
            "desc": "Analytical balance gave erroneous readings due to missed calibration. "
                    "Three weeks of experimental data had to be discarded.",
            "lesson": "Verify calibration status BEFORE starting any measurement series. "
                      "Implement monthly calibration checks for critical instruments.",
        },
        {
            "id": "INC-2023-112",
            "date": "2023-09-10",
            "type": "Chemical Incompatibility",
            "desc": "Fuming observed when acid waste was inadvertently mixed with base waste "
                    "in a shared waste container. No injuries but area evacuated.",
            "lesson": "Separate waste streams for acids and bases are MANDATORY. "
                      "Never combine incompatible chemical waste.",
        },
        {
            "id": "INC-2024-015",
            "date": "2024-01-20",
            "type": "PPE Violation",
            "desc": "Researcher sustained chemical burn on forearm from splash while not wearing "
                    "required lab coat and gloves. Required medical treatment.",
            "lesson": "PPE compliance must be verified before starting any procedure. "
                      "Lab coats and gloves are non-negotiable for chemical handling.",
        },
        {
            "id": "INC-2024-033",
            "date": "2024-03-08",
            "type": "Biosafety Breach",
            "desc": "BSL-2 organisms were processed on an open bench instead of in a biosafety "
                    "cabinet. Discovered during routine inspection. No known exposure.",
            "lesson": "All BSL-2 work involving potential aerosol generation MUST be performed "
                      "in a certified biosafety cabinet. Review BSL requirements before each experiment.",
        },
        {
            "id": "INC-2024-056",
            "date": "2024-05-14",
            "type": "Concentration Error",
            "desc": "Researcher prepared 10x concentrated solution due to calculation error. "
                    "Experiment ruined and sample exposure exceeded safety limits.",
            "lesson": "Concentration calculations must be independently verified. "
                      "Use a standard calculation worksheet and have a colleague check.",
        },
        {
            "id": "INC-2024-078",
            "date": "2024-07-30",
            "type": "Insufficient Training",
            "desc": "New lab member operated centrifuge without training, causing rotor imbalance "
                    "and equipment damage. No injuries.",
            "lesson": "All lab members must complete equipment-specific training AND have training "
                      "documented BEFORE independent use of any instrument.",
        },
        {
            "id": "INC-2024-091",
            "date": "2024-08-25",
            "type": "Emergency Procedure Gap",
            "desc": "During a minor chemical spill, personnel were unsure of cleanup procedures "
                    "because the protocol lacked specific emergency instructions.",
            "lesson": "Every protocol must include SPECIFIC emergency procedures, not just "
                      "'follow standard procedures'. Include spill response, exposure response, and contacts.",
        },
    ]

    n_incidents = rng.randint(4, 6)
    selected_incidents = rng.sample(incident_templates, n_incidents)

    incident_lines = [
        "PREVIOUS INCIDENT REPORTS — LABORATORY SAFETY OFFICE",
        "",
        "The following incidents are relevant to current lab operations.",
        "All corrective actions and lessons learned should be incorporated",
        "into current protocols.",
        "",
    ]
    for inc in selected_incidents:
        incident_lines.extend([
            "-" * 60,
            f"Incident ID: {inc['id']}",
            f"Date: {inc['date']}",
            f"Type: {inc['type']}",
            f"Description: {inc['desc']}",
            f"Corrective Action / Lesson: {inc['lesson']}",
            "",
        ])
    incidents_content = "\n".join(incident_lines) + "\n"

    # ---------- Build rubric ----------
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does the file /testbed/audit_report.txt exist and contain substantive content (more than a few lines)?",
            points=1,
        ),
    ]

    if "expired_calibration" in planted_issues:
        expired_names = [f"{e[0]} ({e[1]})" for e in expired_items]
        rubric_items.append(BinaryRubricCategory(
            name="detects_expired_calibration",
            question=(
                f"Does the audit identify that {n_expired} equipment item(s) have expired calibration? "
                f"The overdue items are: {', '.join(expired_names)}."
            ),
            points=2,
        ))
        rubric_items.append(BinaryRubricCategory(
            name="expired_cal_specific",
            question=(
                f"Does the audit cite specific equipment IDs and calibration dates for at least one "
                f"of the expired items ({', '.join(e[1] for e in expired_items)})?"
            ),
            points=1,
        ))

    if "missing_ppe" in planted_issues:
        rubric_items.append(BinaryRubricCategory(
            name="detects_missing_ppe",
            question=(
                f"Does the audit identify that the protocol is missing required PPE items? "
                f"The missing items are: {', '.join(missing_ppe)}."
            ),
            points=2,
        ))

    if "incompatible_storage" in planted_issues and incompatible_pair:
        rubric_items.append(BinaryRubricCategory(
            name="detects_incompatible_storage",
            question=(
                f"Does the audit identify that {incompatible_pair[0]} and {incompatible_pair[1]} "
                f"are stored on the same shelf, violating chemical compatibility rules? "
                f"{incompatible_pair[2]}."
            ),
            points=2,
        ))

    if "missing_emergency_procedure" in planted_issues:
        rubric_items.append(BinaryRubricCategory(
            name="detects_missing_emergency",
            question=(
                "Does the audit identify that the protocol's safety section only contains generic "
                "statements ('follow standard laboratory safety procedures') instead of specific "
                "emergency procedures for spills, exposure, and fire as required by regulations?"
            ),
            points=2,
        ))

    if "dosage_error" in planted_issues and dosage_chemical:
        rubric_items.append(BinaryRubricCategory(
            name="detects_dosage_error",
            question=(
                f"Does the audit identify that {dosage_chemical} is listed at {wrong_concentration} M, "
                f"which appears to be a 10x error? The standard concentration should be approximately "
                f"{correct_concentration} M."
            ),
            points=3,
        ))

    if "insufficient_washout" in planted_issues:
        rubric_items.append(BinaryRubricCategory(
            name="detects_insufficient_washout",
            question=(
                f"Does the audit identify that the {stated_washout_days}-day washout period between "
                f"experimental runs is insufficient? Safety regulations require a minimum of "
                f"{required_washout_days} days."
            ),
            points=2,
        ))

    if "missing_consent" in planted_issues:
        rubric_items.append(BinaryRubricCategory(
            name="detects_missing_consent",
            question=(
                "Does the audit identify that the human subjects section lacks essential elements: "
                "no IRB approval number, no formal consent form, and no data privacy procedures?"
            ),
            points=2,
        ))

    if "wrong_biosafety_level" in planted_issues:
        rubric_items.append(BinaryRubricCategory(
            name="detects_wrong_bsl",
            question=(
                f"Does the audit identify that the protocol lists {protocol_bsl} but the lab's "
                f"work ({lab['focus']}) requires {lab['biosafety']}? The stated biosafety level "
                f"is insufficient for the organisms/materials being handled."
            ),
            points=3,
        ))

    if "contradicts_incident" in planted_issues:
        rubric_items.append(BinaryRubricCategory(
            name="detects_incident_contradiction",
            question=(
                "Does the audit identify at least one instance where the protocol fails to "
                "incorporate lessons learned from the previous incident reports? "
                "The incidents file contains specific corrective actions that are not reflected "
                "in the current protocol."
            ),
            points=2,
        ))

    if "missing_waste_disposal" in planted_issues:
        rubric_items.append(BinaryRubricCategory(
            name="detects_missing_waste",
            question=(
                "Does the audit identify that the protocol has no waste disposal section, "
                "despite using chemicals that require proper hazardous waste handling?"
            ),
            points=2,
        ))

    if "ventilation_inadequate" in planted_issues:
        rubric_items.append(BinaryRubricCategory(
            name="detects_ventilation_issue",
            question=(
                "Does the audit identify that the protocol does not specify fume hood use "
                "for handling volatile or hazardous chemicals that require ventilation?"
            ),
            points=2,
        ))

    if "training_not_documented" in planted_issues:
        rubric_items.append(BinaryRubricCategory(
            name="detects_training_gap",
            question=(
                "Does the audit identify that the protocol lacks requirements for documented "
                "training before personnel can independently perform procedures?"
            ),
            points=2,
        ))

    # Cross-referencing checks
    rubric_items.append(BinaryRubricCategory(
        name="cross_references_equipment",
        question=(
            "Does the audit cross-reference the protocol's equipment list against the "
            "equipment_inventory.csv to check calibration status?"
        ),
        points=2,
    ))

    rubric_items.append(BinaryRubricCategory(
        name="cross_references_incidents",
        question=(
            "Does the audit cross-reference the protocol against previous_incidents.txt "
            "to verify lessons learned have been incorporated?"
        ),
        points=2,
    ))

    rubric_items.append(BinaryRubricCategory(
        name="cross_references_safety",
        question=(
            "Does the audit cite specific sections of safety_regulations.txt when "
            "identifying violations?"
        ),
        points=1,
    ))

    # False-positive checks
    rubric_items.append(BinaryRubricCategory(
        name="no_false_positive_protocol_date",
        question=(
            "Does the audit avoid incorrectly flagging the protocol date (2024-09-15) as "
            "problematic? The protocol date is current and does not indicate an issue."
        ),
        points=2,
    ))

    # Identify a piece of equipment that IS current and should not be flagged
    current_eq = [r for i, r in enumerate(equipment_rows) if i not in expired_indices]
    if current_eq:
        fp_eq = rng.choice(current_eq)
        rubric_items.append(BinaryRubricCategory(
            name="no_false_positive_equipment",
            question=(
                f"Does the audit avoid incorrectly flagging {fp_eq['name']} ({fp_eq['id']}) as having "
                f"expired calibration? Its calibration is current (last: {fp_eq['last_cal']}, "
                f"next due: {fp_eq['next_due']})."
            ),
            points=2,
        ))

    rubric_items.append(BinaryRubricCategory(
        name="no_false_positive_chemicals",
        question=(
            "Does the audit avoid incorrectly flagging standard laboratory chemicals at normal "
            "concentrations as safety hazards when they are properly handled per the protocol?"
        ),
        points=1,
    ))

    # Overall quality
    rubric_items.append(RubricCategory(
        name="audit_quality",
        description="Overall quality of the protocol audit: thoroughness, organization, and actionability.",
        failure="Superficial audit that misses most issues or is disorganized.",
        minor_failure="Identifies some issues but misses major safety concerns or lacks specificity.",
        minor_success="Identifies most issues with reasonable specificity and references source documents.",
        success="Thorough, well-organized audit that identifies all major issues with specific evidence, cross-references all source documents, prioritizes by severity, and provides actionable remediation steps.",
        points=3,
    ))

    # Pad if needed
    if len(rubric_items) < 15:
        rubric_items.insert(1, BinaryRubricCategory(
            name="identifies_lab_type",
            question=f"Does the audit correctly identify the lab as '{lab['name']}' focused on {lab['focus']}?",
            points=1,
        ))
    if len(rubric_items) < 15:
        rubric_items.insert(2, BinaryRubricCategory(
            name="identifies_pi",
            question=f"Does the audit reference the PI ({pi_name}) and protocol ID ({protocol_id})?",
            points=1,
        ))

    problem_statement = f"""# Experiment Protocol Safety Audit

You are a laboratory safety auditor reviewing an experiment protocol for safety, ethics, and scientific rigor.

## Source Files
- /testbed/data/protocol.txt -- The experiment protocol under review
- /testbed/data/safety_regulations.txt -- Relevant safety standards and requirements
- /testbed/data/equipment_inventory.csv -- Lab equipment with calibration dates and status
- /testbed/data/previous_incidents.txt -- Past incident reports with lessons learned

## Task
1. Read the protocol carefully and identify all safety, compliance, and rigor issues
2. Cross-reference the protocol against safety regulations for PPE, chemical storage, BSL, etc.
3. Check equipment calibration status in the inventory
4. Review previous incidents and verify the protocol incorporates corrective actions
5. Identify genuine issues vs acceptable practices
6. Prioritize findings by severity (critical, major, minor)

Write a detailed audit report to /testbed/audit_report.txt"""

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
            "/testbed/data/protocol.txt": protocol_content,
            "/testbed/data/safety_regulations.txt": safety_content,
            "/testbed/data/equipment_inventory.csv": equipment_csv_content,
            "/testbed/data/previous_incidents.txt": incidents_content,
        },
        problem_type="experiment_protocol_audit",
    )


# =============================================================================
# DOMAIN POOLS: CITATION NETWORK ANALYSIS
# =============================================================================

_RESEARCH_FIELDS = [
    "machine learning", "climate science", "public health", "genomics",
    "renewable energy", "neuroscience", "materials science", "economics",
    "social psychology", "epidemiology",
]

_JOURNAL_TIERS = {
    "tier1": [
        ("Nature", 69.5, False), ("Science", 63.7, False),
        ("The Lancet", 202.7, False), ("NEJM", 176.1, False),
        ("Cell", 66.8, False), ("PNAS", 12.8, False),
    ],
    "tier2": [
        ("PLoS ONE", 3.7, False), ("Scientific Reports", 4.6, False),
        ("BMJ Open", 3.0, False), ("Frontiers in Neuroscience", 4.3, False),
        ("Environmental Research Letters", 6.9, False),
        ("Journal of Applied Psychology", 7.2, False),
    ],
    "predatory": [
        ("International Journal of Advanced Research", 0.1, True),
        ("Global Journal of Scientific Discoveries", 0.0, True),
        ("Open Access Journal of Emerging Research", 0.2, True),
        ("World Journal of Innovative Studies", 0.0, True),
        ("Universal Research Chronicles", 0.1, True),
        ("Frontier Science and Technology Letters", 0.0, True),
    ],
}

_RETRACTION_REASONS = [
    "data fabrication",
    "image manipulation",
    "plagiarism",
    "failure to disclose conflicts of interest",
    "unreliable results",
    "ethical violations in study conduct",
    "duplicate publication",
    "authorship dispute with evidence of fabrication",
]


# =============================================================================
# 3. CITATION NETWORK ANALYSIS
# =============================================================================


def make_citation_network_analysis(rand_seed: int = 42) -> RubricDatapoint:
    """Analyze a bibliography for integrity issues by cross-referencing against
    retraction databases, journal watchlists, and citation context.

    Seed varies: research field, number of references, which references are
    retracted/predatory/misrepresented, self-citation patterns, and false-positive
    distractors.
    """
    rng = _random.Random(rand_seed)

    field = rng.choice(_RESEARCH_FIELDS)
    paper_authors = random_names(rand_seed + 10, 3)
    first_author = paper_authors[0]

    n_refs = rng.randint(30, 50)

    # Generate author pool for references
    ref_author_pool = random_names(rand_seed + 20, 60)

    # Decide issue counts
    n_retracted = rng.randint(2, 4)
    n_predatory = rng.randint(2, 3)
    n_misrepresented = rng.randint(2, 4)
    n_self_citation = rng.randint(3, 6)
    n_superseded = rng.randint(1, 2)
    n_circular = 1  # one circular pair

    # Generate reference list
    references = []
    retracted_indices = set()
    predatory_indices = set()
    misrepresented_indices = set()
    self_citation_indices = set()
    superseded_indices = set()
    circular_indices = set()

    # Assign issue indices (non-overlapping where possible)
    all_indices = list(range(n_refs))
    rng.shuffle(all_indices)
    idx_cursor = 0

    for _ in range(n_retracted):
        if idx_cursor < n_refs:
            retracted_indices.add(all_indices[idx_cursor])
            idx_cursor += 1
    for _ in range(n_predatory):
        if idx_cursor < n_refs:
            predatory_indices.add(all_indices[idx_cursor])
            idx_cursor += 1
    for _ in range(n_misrepresented):
        if idx_cursor < n_refs:
            misrepresented_indices.add(all_indices[idx_cursor])
            idx_cursor += 1
    for _ in range(n_self_citation):
        if idx_cursor < n_refs:
            self_citation_indices.add(all_indices[idx_cursor])
            idx_cursor += 1
    for _ in range(n_superseded):
        if idx_cursor < n_refs:
            superseded_indices.add(all_indices[idx_cursor])
            idx_cursor += 1
    # Circular pair
    if idx_cursor + 1 < n_refs:
        circular_indices.add(all_indices[idx_cursor])
        circular_indices.add(all_indices[idx_cursor + 1])
        idx_cursor += 2

    clean_indices = set(range(n_refs)) - retracted_indices - predatory_indices - misrepresented_indices - self_citation_indices - superseded_indices - circular_indices

    # Build references
    retracted_refs = []
    predatory_refs = []
    misrepresented_refs = []
    self_cite_refs = []
    superseded_refs = []
    circular_ref_pair = []

    for i in range(n_refs):
        year = rng.randint(2010, 2024)
        cited_by = rng.randint(5, 500)

        if i in self_citation_indices:
            # Self-citation: paper_authors appear as authors
            ref_authors_list = [rng.choice(paper_authors)]
            n_extra = rng.randint(1, 3)
            ref_authors_list.extend(rng.sample(ref_author_pool, n_extra))
            rng.shuffle(ref_authors_list)
            ref_authors = "; ".join(ref_authors_list)
        else:
            n_auth = rng.randint(1, 5)
            ref_authors = "; ".join(rng.sample(ref_author_pool, n_auth))

        if i in predatory_indices:
            journal_name, impact, _ = rng.choice(_JOURNAL_TIERS["predatory"])
        elif i in retracted_indices:
            journal_name, impact, _ = rng.choice(_JOURNAL_TIERS["tier1"] + _JOURNAL_TIERS["tier2"])
        else:
            tier = rng.choices(["tier1", "tier2"], weights=[0.3, 0.7])[0]
            journal_name, impact, _ = rng.choice(_JOURNAL_TIERS[tier])

        doi = f"10.{rng.randint(1000, 9999)}/{rng.randint(10000, 99999)}"

        title_words = [
            "Effect", "Impact", "Analysis", "Study", "Role", "Association",
            "Influence", "Mechanism", "Review", "Assessment", "Novel", "Approach",
            "Evidence", "Framework", "Evaluation", "Dynamics", "Relationship",
            "Modulation", "Characterization", "Development",
        ]
        field_words = field.split()
        title = f"{rng.choice(title_words)} of {rng.choice(field_words)} {rng.choice(title_words).lower()}: a {rng.choice(['systematic', 'comprehensive', 'comparative', 'quantitative', 'longitudinal'])} {rng.choice(['study', 'analysis', 'investigation', 'review', 'assessment'])}"

        # Abstract snippet (for verifying citation accuracy)
        if i in misrepresented_indices:
            actual_finding = f"no significant {rng.choice(['association', 'effect', 'difference', 'correlation'])} was found"
            abstract_snippet = f"Our results indicate that {actual_finding} (p=0.{rng.randint(10, 95):02d})."
        elif i in superseded_indices:
            abstract_snippet = f"Based on data available through {year}, we recommend guideline version {year - 2000}.{rng.randint(1,5)}."
        else:
            abstract_snippet = f"We found a {rng.choice(['significant', 'moderate', 'strong', 'notable'])} {rng.choice(['effect', 'association', 'correlation', 'relationship'])} in {field} (p<0.{rng.randint(1, 5):02d})."

        ref = {
            "index": i + 1,
            "title": title,
            "authors": ref_authors,
            "year": year,
            "journal": journal_name,
            "doi": doi,
            "cited_by": cited_by,
            "impact_factor": impact,
            "abstract": abstract_snippet,
        }
        references.append(ref)

        if i in retracted_indices:
            retracted_refs.append(ref)
        if i in predatory_indices:
            predatory_refs.append(ref)
        if i in misrepresented_indices:
            misrepresented_refs.append(ref)
        if i in self_citation_indices:
            self_cite_refs.append(ref)
        if i in superseded_indices:
            superseded_refs.append(ref)
        if i in circular_indices:
            circular_ref_pair.append(ref)

    # ---------- bibliography.csv ----------
    bib_lines = ["ref_number,title,authors,year,journal,doi,cited_by_count,abstract_snippet"]
    for ref in references:
        # Escape commas in fields
        title_escaped = ref["title"].replace(",", ";")
        authors_escaped = ref["authors"].replace(",", ";")
        abstract_escaped = ref["abstract"].replace(",", ";")
        bib_lines.append(
            f"{ref['index']},\"{title_escaped}\",\"{authors_escaped}\","
            f"{ref['year']},{ref['journal']},{ref['doi']},"
            f"{ref['cited_by']},\"{abstract_escaped}\""
        )
    bibliography_content = "\n".join(bib_lines) + "\n"

    # ---------- citation_context.txt ----------
    context_lines = [
        "CITATION CONTEXT EXCERPTS",
        "",
        "The following excerpts show how each reference is cited in the manuscript.",
        "",
    ]
    for ref in references:
        context_lines.append(f"--- Reference [{ref['index']}] ---")
        if ref["index"] - 1 in misrepresented_indices:
            # Citation claims the opposite of what the paper found
            context_lines.append(
                f"\"Previous work has demonstrated strong support for this effect [{ref['index']}], "
                f"consistent with our hypothesis.\""
            )
            context_lines.append(
                f"(Note: Cited to support a positive finding, but the reference's actual conclusion "
                f"is stated in its abstract in bibliography.csv.)"
            )
        elif ref["index"] - 1 in superseded_indices:
            context_lines.append(
                f"\"Current guidelines [{ref['index']}] recommend this approach as best practice.\""
            )
        elif ref["index"] - 1 in circular_indices:
            other_circ = [r for r in circular_ref_pair if r["index"] != ref["index"]]
            if other_circ:
                context_lines.append(
                    f"\"As established in [{ref['index']}] and corroborated by [{other_circ[0]['index']}], "
                    f"the evidence supports our framework.\""
                )
            else:
                context_lines.append(
                    f"\"The foundational work [{ref['index']}] provides the basis for our methodology.\""
                )
        else:
            context_lines.append(
                f"\"{rng.choice(['Building on', 'Consistent with', 'As shown by', 'Following'])} "
                f"the work of {ref['authors'].split(';')[0].strip()} et al. [{ref['index']}], "
                f"we {rng.choice(['adopted', 'extended', 'replicated', 'confirmed'])} this approach.\""
            )
        context_lines.append("")
    citation_context_content = "\n".join(context_lines) + "\n"

    # ---------- retraction_database.txt ----------
    retraction_lines = [
        "RETRACTION DATABASE — CURATED LIST",
        "",
        "The following papers have been formally retracted by their publishers.",
        "Retracted papers should not be cited as valid evidence.",
        "",
    ]
    # Include the actual retracted refs
    for ref in retracted_refs:
        reason = rng.choice(_RETRACTION_REASONS)
        retraction_lines.extend([
            f"DOI: {ref['doi']}",
            f"Title: {ref['title']}",
            f"Journal: {ref['journal']}",
            f"Year: {ref['year']}",
            f"Retraction Date: {rng.randint(2022, 2024)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
            f"Reason: {reason}",
            "",
        ])
    # Add some unrelated retracted papers (not in the bibliography)
    for _ in range(rng.randint(8, 15)):
        fake_doi = f"10.{rng.randint(1000, 9999)}/{rng.randint(10000, 99999)}"
        while any(fake_doi == ref["doi"] for ref in references):
            fake_doi = f"10.{rng.randint(1000, 9999)}/{rng.randint(10000, 99999)}"
        retraction_lines.extend([
            f"DOI: {fake_doi}",
            f"Title: {rng.choice(['Fabricated', 'Unreliable', 'Retracted'])} findings in {rng.choice(_RESEARCH_FIELDS)}",
            f"Journal: {rng.choice([j[0] for j in _JOURNAL_TIERS['tier1'] + _JOURNAL_TIERS['tier2']])}",
            f"Year: {rng.randint(2015, 2023)}",
            f"Retraction Date: {rng.randint(2022, 2024)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
            f"Reason: {rng.choice(_RETRACTION_REASONS)}",
            "",
        ])
    retraction_content = "\n".join(retraction_lines) + "\n"

    # ---------- journal_metrics.txt ----------
    metric_lines = [
        "JOURNAL METRICS AND PREDATORY JOURNAL WATCHLIST",
        "",
        "=" * 60,
        "JOURNAL IMPACT FACTORS (2024)",
        "=" * 60,
        "",
    ]
    all_journals = set()
    for tier_journals in _JOURNAL_TIERS.values():
        for j_name, j_if, _ in tier_journals:
            all_journals.add((j_name, j_if))
    for j_name, j_if in sorted(all_journals, key=lambda x: -x[1]):
        metric_lines.append(f"  {j_name}: IF = {j_if}")
    metric_lines.extend([
        "",
        "=" * 60,
        "PREDATORY JOURNAL WATCHLIST",
        "=" * 60,
        "",
        "The following journals have been identified as predatory publishers",
        "(no legitimate peer review, pay-to-publish, misleading practices):",
        "",
    ])
    for j_name, j_if, is_pred in _JOURNAL_TIERS["predatory"]:
        metric_lines.append(f"  - {j_name} (IF: {j_if})")
    metric_lines.extend([
        "",
        "WARNING: Papers published in predatory journals should not be considered",
        "reliable evidence without independent verification.",
        "",
        "=" * 60,
        "SUPERSEDED GUIDELINES",
        "=" * 60,
        "",
        "When citing guidelines, always use the most current version.",
        "Guidelines older than 5 years should be checked for updates.",
        "Citing superseded guidelines as 'current' is a citation integrity issue.",
        "",
    ])
    journal_metrics_content = "\n".join(metric_lines) + "\n"

    # ---------- Build rubric ----------
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does the file /testbed/citation_report.txt exist and contain substantive content (more than a few lines)?",
            points=1,
        ),
    ]

    # Retracted papers
    for ref in retracted_refs:
        rubric_items.append(BinaryRubricCategory(
            name=f"detects_retracted_{ref['index']}",
            question=(
                f"Does the report identify reference [{ref['index']}] (DOI: {ref['doi']}) as a retracted paper? "
                f"It appears in the retraction database."
            ),
            points=2,
        ))

    # Predatory journals
    for ref in predatory_refs:
        rubric_items.append(BinaryRubricCategory(
            name=f"detects_predatory_{ref['index']}",
            question=(
                f"Does the report identify reference [{ref['index']}] as published in a predatory journal "
                f"({ref['journal']})? This journal appears on the predatory journal watchlist."
            ),
            points=2,
        ))

    # Misrepresented citations
    for ref in misrepresented_refs:
        rubric_items.append(BinaryRubricCategory(
            name=f"detects_misrepresented_{ref['index']}",
            question=(
                f"Does the report identify that reference [{ref['index']}] is cited to support a positive finding, "
                f"but its abstract actually states: \"{ref['abstract']}\"? The citation misrepresents the source."
            ),
            points=3,
        ))

    # Self-citation cluster
    self_cite_nums = sorted([r["index"] for r in self_cite_refs])
    if self_cite_refs:
        rubric_items.append(BinaryRubricCategory(
            name="detects_self_citation_cluster",
            question=(
                f"Does the report identify a self-citation cluster? References "
                f"{', '.join(f'[{n}]' for n in self_cite_nums)} include the paper's own authors "
                f"({', '.join(paper_authors)}), representing {len(self_cite_refs)} of {n_refs} references "
                f"({round(len(self_cite_refs)/n_refs*100, 1)}%)."
            ),
            points=2,
        ))

    # Superseded guidelines
    for ref in superseded_refs:
        rubric_items.append(BinaryRubricCategory(
            name=f"detects_superseded_{ref['index']}",
            question=(
                f"Does the report identify reference [{ref['index']}] (year {ref['year']}) as a potentially "
                f"superseded guideline cited as 'current'? The citation context claims it represents "
                f"current best practice."
            ),
            points=2,
        ))

    # Circular citation
    if len(circular_ref_pair) == 2:
        rubric_items.append(BinaryRubricCategory(
            name="detects_circular_citation",
            question=(
                f"Does the report identify a circular citation pattern between references "
                f"[{circular_ref_pair[0]['index']}] and [{circular_ref_pair[1]['index']}]? "
                f"The citation context shows these two references cite each other as mutual evidence."
            ),
            points=3,
        ))

    # Cross-referencing checks
    rubric_items.append(BinaryRubricCategory(
        name="cross_references_retraction_db",
        question=(
            "Does the report demonstrate cross-referencing the bibliography against the "
            "retraction database (matching by DOI or title)?"
        ),
        points=2,
    ))

    rubric_items.append(BinaryRubricCategory(
        name="cross_references_journal_watchlist",
        question=(
            "Does the report demonstrate checking journal names against the predatory "
            "journal watchlist in journal_metrics.txt?"
        ),
        points=1,
    ))

    rubric_items.append(BinaryRubricCategory(
        name="checks_citation_vs_abstract",
        question=(
            "Does the report compare the citation context (how papers are cited) against "
            "the abstract snippets (what papers actually found) to detect misrepresentation?"
        ),
        points=2,
    ))

    # False-positive checks
    # Pick a clean reference from a good journal
    clean_idx_list = sorted(clean_indices)
    if clean_idx_list:
        fp_ref = references[rng.choice(clean_idx_list)]
        rubric_items.append(BinaryRubricCategory(
            name="no_false_positive_clean_ref",
            question=(
                f"Does the report avoid incorrectly flagging reference [{fp_ref['index']}] "
                f"(\"{fp_ref['title'][:50]}...\") as problematic? It is published in a legitimate "
                f"journal ({fp_ref['journal']}), is not retracted, and is cited accurately."
            ),
            points=2,
        ))

    # Low citation count is not per se an issue
    low_cite_refs = [r for r in references if r["cited_by"] < 20 and r["index"] - 1 in clean_indices]
    if low_cite_refs:
        lc_ref = rng.choice(low_cite_refs)
        rubric_items.append(BinaryRubricCategory(
            name="no_false_positive_low_citations",
            question=(
                f"Does the report avoid flagging reference [{lc_ref['index']}] as problematic solely "
                f"because it has a low citation count ({lc_ref['cited_by']})? Low citation count "
                f"alone does not indicate a citation integrity issue."
            ),
            points=2,
        ))

    rubric_items.append(BinaryRubricCategory(
        name="no_false_positive_self_cite_minor",
        question=(
            "Does the report avoid claiming self-citation is inherently fraudulent? "
            "Self-citation is a legitimate practice; the issue is excessive self-citation "
            "or self-citation clusters that inflate apparent support."
        ),
        points=1,
    ))

    # Overall quality
    rubric_items.append(RubricCategory(
        name="analysis_quality",
        description="Overall quality of the citation network analysis: thoroughness, accuracy, and organization.",
        failure="Superficial analysis that misses most integrity issues.",
        minor_failure="Identifies some issues but misses major problems or makes significant false-positive errors.",
        minor_success="Identifies most issues with reasonable specificity and demonstrates cross-referencing.",
        success="Comprehensive analysis that identifies all major integrity issues, avoids false positives, provides specific evidence (DOIs, reference numbers), and offers clear recommendations.",
        points=3,
    ))

    # Pad if needed
    if len(rubric_items) < 15:
        rubric_items.insert(1, BinaryRubricCategory(
            name="identifies_field",
            question=f"Does the report correctly identify the research field as {field}?",
            points=1,
        ))
    if len(rubric_items) < 15:
        rubric_items.insert(2, BinaryRubricCategory(
            name="total_ref_count",
            question=f"Does the report correctly note the bibliography contains {n_refs} references?",
            points=1,
        ))
    if len(rubric_items) < 15:
        rubric_items.insert(3, BinaryRubricCategory(
            name="provides_summary_table",
            question="Does the report include a summary or table of all identified issues with reference numbers?",
            points=1,
        ))

    problem_statement = f"""# Citation Network Integrity Analysis

You are a research integrity specialist analyzing the bibliography of a manuscript in {field}.

## Source Files
- /testbed/data/bibliography.csv -- {n_refs} references with title, authors, year, journal, DOI, citation count, and abstract snippet
- /testbed/data/citation_context.txt -- Excerpts showing how each reference is cited in the manuscript
- /testbed/data/retraction_database.txt -- Database of known retracted papers with DOIs
- /testbed/data/journal_metrics.txt -- Journal impact factors and predatory journal watchlist

## Task
1. Cross-reference every bibliography entry against the retraction database (match by DOI)
2. Check journal names against the predatory journal watchlist
3. Compare citation context (how papers are cited) against abstract snippets (what papers actually found)
4. Identify self-citation patterns (compare bibliography authors with paper authors: {', '.join(paper_authors)})
5. Check for superseded guidelines cited as current
6. Look for circular citation patterns (papers citing each other as mutual evidence)
7. Distinguish genuine integrity issues from acceptable practices

Write a detailed citation integrity report to /testbed/citation_report.txt"""

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=tuple(rubric_items),
        submission_instructions="Write your citation integrity report to /testbed/citation_report.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/bibliography.csv": bibliography_content,
            "/testbed/data/citation_context.txt": citation_context_content,
            "/testbed/data/retraction_database.txt": retraction_content,
            "/testbed/data/journal_metrics.txt": journal_metrics_content,
        },
        problem_type="citation_network_analysis",
    )
