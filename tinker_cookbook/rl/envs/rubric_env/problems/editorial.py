"""Editorial problem factories.

Contains 5 factories:
  - make_editorial_headline_standfirst  (static)
  - make_editorial_opinion_argument     (static)
  - make_editorial_audience_adaptation  (seedable)
  - make_editorial_assembly             (static)
  - make_editorial_fact_check           (seedable)
"""

import random as _random
from ..dataset import RubricDatapoint, RubricCategory, BinaryRubricCategory
from tinker_cookbook.rl.envs import tools


# =============================================================================
# 1. HEADLINE + STANDFIRST  (static)
# =============================================================================


def make_editorial_headline_standfirst() -> RubricDatapoint:
    """Write a headline and standfirst for a given news event.

    The model must produce a newspaper-quality headline + standfirst (the
    1-2 sentence summary that appears below a headline in print journalism).
    The rubric checks for extremely specific structural and content features.
    """
    return RubricDatapoint(
        problem_statement="""# Editorial Task: Headline and Standfirst

You are an editor at a national broadsheet newspaper. A major story has broken:

=== NEWS EVENT ===
The European Union has voted to impose a complete ban on the sale of new
internal combustion engine (ICE) vehicles starting in 2035, with several
member states pushing for an even earlier 2030 deadline. The vote passed
315-270 after months of lobbying from both environmental groups and the
automotive industry. Germany, home to BMW, Mercedes, and Volkswagen,
voted against the measure. Meanwhile, Norway (not an EU member) has already
achieved 80% EV market share. Industry analysts estimate the ban will
require \u20ac450 billion in infrastructure investment across the EU. Several
major automakers have announced they will challenge the regulation in
the European Court of Justice.
=== END NEWS EVENT ===

Your task: Write EXACTLY THREE different headline + standfirst pairs for
this story, each targeting a different editorial angle:

1. **Hard news angle** \u2014 Straight reporting, factual emphasis
2. **Economic/business angle** \u2014 Focus on industry and financial implications
3. **Opinion/editorial angle** \u2014 Taking a clear position (for or against)

Each pair must be clearly labeled and written to /testbed/headlines.txt

FORMAT REQUIREMENTS:
- Each headline must be on its own line
- Each standfirst must immediately follow its headline
- Separate each pair with a blank line
- Label each pair: [HARD NEWS], [BUSINESS], [OPINION]""",
        rubric=(
            # --- STRUCTURAL CHECKS (binary, low points) ---
            BinaryRubricCategory(
                name="file_exists",
                question="Does the file /testbed/headlines.txt exist and contain non-empty text?",
                points=1,
            ),
            BinaryRubricCategory(
                name="three_pairs_present",
                question="Does the submission contain exactly three headline+standfirst pairs?",
                points=1,
            ),
            BinaryRubricCategory(
                name="labels_present",
                question="Are all three pairs labeled with [HARD NEWS], [BUSINESS], and [OPINION] respectively?",
                points=1,
            ),
            BinaryRubricCategory(
                name="blank_line_separation",
                question="Are the three pairs separated by blank lines as specified in the format requirements?",
                points=1,
            ),
            # --- HARD NEWS HEADLINE ---
            BinaryRubricCategory(
                name="hard_news_headline_under_15_words",
                question="Is the [HARD NEWS] headline 15 words or fewer?",
                points=1,
            ),
            BinaryRubricCategory(
                name="hard_news_headline_contains_vote_result",
                question="Does the [HARD NEWS] headline mention the vote or the ban specifically (not just 'EU' generically)?",
                points=2,
            ),
            BinaryRubricCategory(
                name="hard_news_headline_no_opinion_words",
                question="Is the [HARD NEWS] headline free of opinion language (words like 'historic', 'dangerous', 'bold', 'reckless', 'landmark')?",
                points=2,
            ),
            RubricCategory(
                name="hard_news_standfirst_quality",
                description="Does the [HARD NEWS] standfirst summarize the key facts in 1-2 sentences?",
                failure="Standfirst is missing, exceeds 3 sentences, or omits the vote margin AND the 2035 date",
                minor_failure="Standfirst mentions the ban but omits either the vote margin (315-270) or the 2035 date",
                minor_success="Standfirst mentions both the vote margin and 2035 date but is awkwardly worded or exceeds 2 sentences slightly",
                success="Standfirst is 1-2 crisp sentences mentioning the vote margin, 2035 date, and at least one other key fact (e.g. Germany's opposition, \u20ac450B cost)",
                points=3,
            ),
            # --- BUSINESS HEADLINE ---
            BinaryRubricCategory(
                name="business_headline_financial_reference",
                question="Does the [BUSINESS] headline contain a financial or economic term (e.g., a currency figure, 'industry', 'market', 'billion', 'investment', 'automakers')?",
                points=2,
            ),
            BinaryRubricCategory(
                name="business_headline_under_15_words",
                question="Is the [BUSINESS] headline 15 words or fewer?",
                points=1,
            ),
            RubricCategory(
                name="business_standfirst_specificity",
                description="Does the [BUSINESS] standfirst include specific financial figures or named companies?",
                failure="Standfirst is generic ('industry faces challenges') with no specific figures or company names",
                minor_failure="Mentions one figure or company name but remains largely generic",
                minor_success="Mentions at least two of: \u20ac450B figure, specific company names (BMW/Mercedes/VW), Norway's 80% EV share",
                success="Mentions at least three specific data points from the source material with clear business framing",
                points=3,
            ),
            # --- OPINION HEADLINE ---
            BinaryRubricCategory(
                name="opinion_headline_takes_position",
                question="Does the [OPINION] headline clearly take a position (for or against the ban), distinguishable from neutral reporting?",
                points=2,
            ),
            BinaryRubricCategory(
                name="opinion_headline_under_15_words",
                question="Is the [OPINION] headline 15 words or fewer?",
                points=1,
            ),
            RubricCategory(
                name="opinion_standfirst_argument_preview",
                description="Does the [OPINION] standfirst preview the editorial's argument, not just restate facts?",
                failure="Standfirst merely restates the news event without any argumentative framing",
                minor_failure="Standfirst hints at a position but doesn't articulate a specific argument or claim",
                minor_success="Standfirst articulates a position but the argument preview is generic ('this is good/bad for Europe')",
                success="Standfirst previews a specific, substantive argument (e.g., 'The ban will accelerate job losses in Germany's industrial heartland while doing little to reduce global emissions')",
                points=3,
            ),
            # --- CROSS-CUTTING QUALITY ---
            RubricCategory(
                name="angle_differentiation",
                description="Are the three headline+standfirst pairs genuinely different in angle, not just rewordings of the same framing?",
                failure="All three pairs cover the same angle with cosmetic differences",
                minor_failure="Two of the three pairs are substantively similar in framing",
                minor_success="All three are distinguishable but one pair bleeds into another's angle (e.g., business pair reads like hard news)",
                success="All three pairs are clearly distinct: one is neutral/factual, one is business-focused, one is opinionated \u2014 a reader could immediately tell which is which",
                points=3,
            ),
            BinaryRubricCategory(
                name="no_fabricated_facts",
                question="Are all factual claims in the headlines and standfirsts traceable to the source news event (no invented statistics, dates, or quotes)?",
                points=3,
            ),
        ),
        submission_instructions="Write all three headline+standfirst pairs to /testbed/headlines.txt",
        available_tools=(tools.BASH_TOOL, tools.CREATE_FILE_TOOL, tools.FINISH_TOOL),
        problem_type="editorial",
    )


# =============================================================================
# 2. OPINION ARGUMENT  (static)
# =============================================================================


def make_editorial_opinion_argument() -> RubricDatapoint:
    """Draft the core argument section of an opinion editorial.

    The model writes the central 3-4 paragraphs of an editorial making
    an argument. Rubric checks for extremely specific argumentative structure,
    evidence usage, and rhetorical moves.
    """
    return RubricDatapoint(
        problem_statement="""# Editorial Task: Core Argument Draft

You are writing the central argument section of an opinion editorial for
a major national newspaper. The full editorial will be ~1200 words; you are
writing only the CORE ARGUMENT section (paragraphs 3-6 of 8, approximately
400-600 words).

=== EDITORIAL POSITION ===
TOPIC: Mandatory four-day work weeks for companies with 50+ employees
POSITION: IN FAVOR of mandating four-day work weeks
AUDIENCE: General readership of a center-left broadsheet newspaper
=== END POSITION ===

=== CONTEXT (the editorial's intro, already written by a colleague) ===
"The five-day work week is not a law of nature. It was an invention \u2014 Henry
Ford's invention, to be precise, adopted in 1926 not out of benevolence but
because exhausted workers were making costly mistakes on the assembly line.
Nearly a century later, the same calculus applies, yet we cling to the
five-day structure as though Moses brought it down from Sinai.

This month, a bill was introduced in Parliament to mandate a four-day,
32-hour work week for all companies with 50 or more employees, with no
reduction in pay. The predictable chorus of objections has begun. But the
evidence \u2014 from Iceland, from Spain, from Microsoft Japan, from dozens
of pilot programs \u2014 points overwhelmingly in one direction."
=== END CONTEXT ===

Your task: Write the CORE ARGUMENT section (400-600 words) that follows
the intro above. This section must do the heavy argumentative lifting.

Write the section to /testbed/argument.txt

REQUIREMENTS:
- This section must contain EXACTLY 3 body paragraphs
- Each paragraph must advance a DISTINCT argument (not restate the same point)
- You must reference at least 2 specific real-world examples or data points
- You must include exactly ONE counterargument and rebut it within your argument
- The tone must match the intro: confident, slightly wry, evidence-based
- Do NOT write an introduction or conclusion \u2014 only the core argument paragraphs""",
        rubric=(
            # --- STRUCTURAL REQUIREMENTS ---
            BinaryRubricCategory(
                name="file_exists",
                question="Does /testbed/argument.txt exist with non-trivial content?",
                points=1,
            ),
            BinaryRubricCategory(
                name="exactly_3_paragraphs",
                question="Does the submission contain exactly 3 body paragraphs (separated by blank lines)?",
                points=1,
            ),
            BinaryRubricCategory(
                name="word_count_400_600",
                question="Is the total word count between 400 and 600 words (inclusive)?",
                points=2,
            ),
            BinaryRubricCategory(
                name="no_intro_or_conclusion",
                question="Does the submission avoid writing an introduction or conclusion (i.e., it reads as a middle section, not a standalone essay)?",
                points=2,
            ),
            # --- ARGUMENT DISTINCTNESS ---
            RubricCategory(
                name="paragraph_1_distinct_argument",
                description="Does paragraph 1 present a clear, specific argument (not a vague claim like 'it would be good for workers')?",
                failure="Paragraph 1 has no identifiable argument or is a generic restatement of the position",
                minor_failure="Paragraph 1 has a discernible argument but it's vague (e.g., 'productivity would improve')",
                minor_success="Paragraph 1 makes a specific argument but doesn't develop it with evidence in this paragraph",
                success="Paragraph 1 presents a specific, well-articulated argument with supporting reasoning or evidence",
                points=3,
            ),
            RubricCategory(
                name="paragraph_2_distinct_argument",
                description="Does paragraph 2 advance a NEW argument, distinct from paragraph 1?",
                failure="Paragraph 2 restates paragraph 1's argument in different words",
                minor_failure="Paragraph 2 is on a different topic but the argument is vague or underdeveloped",
                minor_success="Paragraph 2 makes a distinct argument but there's some overlap with paragraph 1",
                success="Paragraph 2 advances a clearly distinct, specific argument from a different angle than paragraph 1",
                points=3,
            ),
            RubricCategory(
                name="paragraph_3_distinct_argument",
                description="Does paragraph 3 advance a NEW argument, distinct from paragraphs 1 and 2?",
                failure="Paragraph 3 rehashes earlier arguments",
                minor_failure="Paragraph 3 attempts a new angle but largely overlaps with earlier paragraphs",
                minor_success="Paragraph 3 is distinct but less developed than the other two",
                success="Paragraph 3 advances a clearly distinct, well-developed argument covering new ground",
                points=3,
            ),
            # --- EVIDENCE AND EXAMPLES ---
            BinaryRubricCategory(
                name="real_world_example_1",
                question="Does the submission reference at least one specific, named real-world example (e.g., 'Iceland's 2015-2019 trial involving 2,500 workers' or 'Microsoft Japan's August 2019 experiment')?",
                points=2,
            ),
            BinaryRubricCategory(
                name="real_world_example_2",
                question="Does the submission reference a SECOND specific, named real-world example distinct from the first?",
                points=2,
            ),
            BinaryRubricCategory(
                name="quantitative_data_point",
                question="Does the submission include at least one specific quantitative claim (a percentage, dollar figure, number of participants, etc.)?",
                points=2,
            ),
            # --- COUNTERARGUMENT ---
            BinaryRubricCategory(
                name="counterargument_present",
                question="Does the submission explicitly acknowledge at least one counterargument against four-day work weeks?",
                points=2,
            ),
            BinaryRubricCategory(
                name="counterargument_rebutted",
                question="Is the counterargument directly rebutted (not just stated and left hanging)?",
                points=2,
            ),
            BinaryRubricCategory(
                name="exactly_one_counterargument",
                question="Does the submission contain exactly ONE counterargument (not zero, not multiple)?",
                points=2,
            ),
            # --- TONE AND STYLE ---
            RubricCategory(
                name="tone_match",
                description="Does the tone match the intro's style: confident, slightly wry/witty, evidence-based rather than preachy?",
                failure="Tone is wildly different \u2014 academic, preachy, angry, or robotic",
                minor_failure="Tone is generally appropriate but lacks the intro's wit or reads as a dry policy paper",
                minor_success="Tone is close but occasionally slips into being overly formal, casual, or preachy",
                success="Tone seamlessly continues the intro: confident, slightly wry, treats the reader as intelligent, evidence-based without being dry",
                points=3,
            ),
            RubricCategory(
                name="rhetorical_cohesion_with_intro",
                description="Does the argument section read as a natural continuation of the provided intro (not as a standalone piece)?",
                failure="Section ignores the intro entirely \u2014 could be pasted into any essay",
                minor_failure="Section vaguely follows the intro's topic but doesn't connect to its framing or references",
                minor_success="Section continues the topic naturally but doesn't pick up any specific threads from the intro",
                success="Section explicitly builds on the intro's framing (e.g., references Ford, the bill, or the 'evidence points in one direction' setup)",
                points=3,
            ),
        ),
        submission_instructions="Write your core argument section to /testbed/argument.txt",
        available_tools=(tools.BASH_TOOL, tools.CREATE_FILE_TOOL, tools.FINISH_TOOL),
        problem_type="editorial",
    )


# =============================================================================
# 3. AUDIENCE ADAPTATION  (seedable)
# =============================================================================

# Pool of technical source paragraphs for audience adaptation.
# Each entry has a topic label, a ~150-word technical paragraph with 4-5
# checkable facts, and a dict of key facts for rubric reference.

_ADAPTATION_SOURCE_POOL: list[dict] = [
    {
        "topic": "Stockholm congestion pricing",
        "text": (
            "The implementation of congestion pricing in central Stockholm in 2006, "
            "initially introduced as a seven-month trial, resulted in a 22% reduction "
            "in traffic volume within the cordon zone during charging hours. Subsequent "
            "analysis by Eliasson et al. (2009) demonstrated that the reduction persisted "
            "at approximately 20% even after permanent implementation in 2007. The scheme "
            "generated SEK 850 million (approximately \u20ac80 million) in annual net revenue "
            "after accounting for infrastructure and operational costs. Critically, public "
            "opinion shifted from 55% opposition before the trial to 53% support after "
            "experiencing the reduced congestion, a phenomenon Eliasson attributes to "
            'the "status quo bias" \u2014 citizens\' tendency to prefer whatever system they '
            "have direct experience with. Air quality monitoring stations within the "
            "cordon recorded a 10-14% reduction in NO\u2082 levels and an 8.5% reduction "
            "in PM10 particulate matter during the first year of operation."
        ),
        "key_facts": {
            "percentage_reduction": "22%",
            "revenue": "SEK 850 million / \u20ac80 million",
            "opinion_shift": "55% opposition to 53% support",
            "air_quality": "10-14% NO\u2082 reduction, 8.5% PM10 reduction",
        },
        "tabloid_fact_check": "the ~22% traffic reduction (or an equivalent like 'cut traffic by a fifth')",
        "tabloid_secondary_check": "the public opinion shift (from opposition to support)",
        "policy_fact_1": "the revenue figure (SEK 850M or ~\u20ac80M)",
        "policy_fact_2": "at least one air quality metric (NO\u2082 or PM10 reduction)",
    },
    {
        "topic": "Finland UBI pilot results",
        "text": (
            "Finland's two-year basic income experiment (2017-2018) provided 2,000 "
            "randomly selected unemployed citizens with \u20ac560 per month unconditionally. "
            "The final report, published by Kela in 2020, found that recipients "
            "experienced measurably better mental health and life satisfaction (scoring "
            "7.3 vs 6.8 on a 0-10 wellbeing scale) compared to the control group. "
            "Employment effects were modest: participants worked an average of 6 more "
            "days per year than the control group, a statistically significant but "
            "economically small difference. Crucially, the experiment cost \u20ac20 million "
            "over two years \u2014 approximately \u20ac10,000 per participant. The Finnish "
            "government chose not to continue the program, citing fiscal constraints "
            "and the 2019 election of a centre-right coalition. Subsequent analysis "
            "by Kangas et al. (2021) noted that participants reported 18% fewer "
            "bureaucratic contacts with social services, suggesting administrative "
            "savings that were not captured in the headline cost figure."
        ),
        "key_facts": {
            "participants": "2,000",
            "amount": "\u20ac560/month",
            "wellbeing_score": "7.3 vs 6.8",
            "extra_work_days": "6 more days per year",
            "cost": "\u20ac20 million over two years",
        },
        "tabloid_fact_check": "the \u20ac560 monthly payment (or an equivalent like 'roughly \u20ac560 a month')",
        "tabloid_secondary_check": "the wellbeing improvement (scoring higher on life satisfaction)",
        "policy_fact_1": "the total cost (\u20ac20 million) or per-participant cost (\u20ac10,000)",
        "policy_fact_2": "the employment effect (6 extra work days per year)",
    },
    {
        "topic": "Carbon border adjustment mechanism effects",
        "text": (
            "The EU's Carbon Border Adjustment Mechanism (CBAM), which entered its "
            "transitional phase in October 2023, requires importers of cement, iron, "
            "steel, aluminium, fertilisers, electricity, and hydrogen to report the "
            "embedded carbon in their products. From 2026, importers will purchase "
            "CBAM certificates at the prevailing EU Emissions Trading System price \u2014 "
            "approximately \u20ac90 per tonne of CO\u2082 as of Q3 2023. The European Commission "
            "estimates CBAM will prevent 47 million tonnes of carbon leakage annually "
            "by 2030, equivalent to Portugal's total yearly emissions. Early compliance "
            "data shows that 78% of affected importers registered within the first "
            "reporting period. Industry group Eurofer estimates the mechanism will "
            "add 12-18% to the cost of imported steel from non-EU countries, while "
            "the World Bank projects that CBAM will reduce EU imports of carbon-"
            "intensive goods by 8-12% by 2030, disproportionately affecting Turkey, "
            "Russia, and China."
        ),
        "key_facts": {
            "start_date": "October 2023 transitional phase",
            "certificate_price": "\u20ac90 per tonne CO\u2082",
            "leakage_prevention": "47 million tonnes annually by 2030",
            "compliance_rate": "78% registered in first period",
            "steel_cost_increase": "12-18%",
        },
        "tabloid_fact_check": "the steel cost increase (12-18%) or the carbon leakage prevention figure (47 million tonnes)",
        "tabloid_secondary_check": "which countries will be most affected (Turkey, Russia, China)",
        "policy_fact_1": "the certificate price (\u20ac90/tonne) or the carbon leakage prevention estimate (47 million tonnes)",
        "policy_fact_2": "the compliance registration rate (78%) or the projected import reduction (8-12%)",
    },
    {
        "topic": "Microplastics in drinking water",
        "text": (
            "A landmark 2024 study published in the New England Journal of Medicine "
            "by Marfella et al. tracked 304 patients undergoing carotid endarterectomy "
            "and found that 58.4% of arterial plaque samples contained detectable "
            "polyethylene microplastics, with a median concentration of 21.7 \u00b5g per "
            "gram of plaque tissue. Over a 34-month follow-up period, patients with "
            "microplastics in their arterial plaques had a 4.53-fold higher risk of "
            "heart attack, stroke, or death compared to those without detectable "
            "microplastics. The study estimated that the average adult ingests "
            "approximately 5 grams of microplastic per week \u2014 roughly the weight of "
            "a credit card. The WHO's 2023 systematic review identified 17 studies "
            "documenting microplastics in human blood, lung tissue, and placental "
            'tissue, though it noted that "the health implications of these findings '
            'remain uncertain" due to the absence of controlled dose-response data.'
        ),
        "key_facts": {
            "sample_size": "304 patients",
            "positive_rate": "58.4%",
            "risk_ratio": "4.53-fold higher risk",
            "weekly_ingestion": "5 grams per week",
            "who_studies": "17 studies",
        },
        "tabloid_fact_check": "the weekly ingestion figure (5 grams / 'weight of a credit card')",
        "tabloid_secondary_check": "the health risk (4.53-fold higher risk of heart attack/stroke)",
        "policy_fact_1": "the study sample size (304 patients) or positive rate (58.4%)",
        "policy_fact_2": "the WHO review scope (17 studies) or the risk ratio (4.53-fold)",
    },
    {
        "topic": "Four-day work week productivity data",
        "text": (
            "The world's largest four-day work week trial, coordinated by 4 Day Week "
            "Global across 61 UK companies and 2,900 employees from June to December "
            "2022, found that company revenue increased by an average of 1.4% during "
            "the trial period, with 92% of participating companies electing to continue "
            "the policy permanently. Employee burnout scores dropped by 71%, while "
            "sick days fell by 65%. Autonomy Research, which analysed the data, found "
            "that productivity (measured as revenue per employee) remained statistically "
            "flat \u2014 companies produced the same output in 32 hours as they previously "
            "had in 40. Critically, the trial showed marked variation by sector: "
            "technology and professional services firms reported the highest productivity "
            "maintenance (98%), while hospitality and retail saw a modest 3-5% decline. "
            "Staff turnover during the trial period fell by 57% compared to the "
            "same period in the prior year."
        ),
        "key_facts": {
            "companies": "61 UK companies",
            "employees": "2,900",
            "revenue_change": "+1.4%",
            "continuation_rate": "92%",
            "burnout_drop": "71%",
            "turnover_drop": "57%",
        },
        "tabloid_fact_check": "the number of companies (61) or employees (2,900) in the trial",
        "tabloid_secondary_check": "the burnout reduction (71%) or the continuation rate (92%)",
        "policy_fact_1": "the revenue change (+1.4%) or the continuation rate (92%)",
        "policy_fact_2": "the sick day reduction (65%) or the staff turnover drop (57%)",
    },
    {
        "topic": "Norway EV adoption via tax incentives",
        "text": (
            "Norway achieved an 82.4% battery-electric vehicle market share for new "
            "car sales in 2023, the highest in the world, without imposing any outright "
            "ban on internal combustion engines. The policy package, developed over two "
            "decades, centres on fiscal incentives: EVs are exempt from the 25% VAT on "
            "purchase, the one-time registration tax (averaging NOK 80,000 / \u20ac7,200 for "
            "ICE vehicles), and annual road tax. The Norwegian EV Association estimates "
            "these incentives cost the government approximately NOK 39.4 billion "
            "(\u20ac3.5 billion) in foregone tax revenue in 2023 alone. The transition has "
            "created an estimated 12,000 new jobs in EV charging infrastructure and "
            "maintenance, while traditional automotive maintenance employment fell by "
            "approximately 30%. Urban air quality in Oslo improved measurably, with "
            "NO\u2082 levels at roadside monitoring stations declining 23% between 2015 "
            "and 2023."
        ),
        "key_facts": {
            "market_share": "82.4%",
            "vat_exemption": "25% VAT exempt",
            "foregone_revenue": "NOK 39.4 billion / \u20ac3.5 billion",
            "new_jobs": "12,000",
            "maintenance_decline": "30%",
            "air_quality": "23% NO\u2082 decline",
        },
        "tabloid_fact_check": "the EV market share (82.4% or ~80%)",
        "tabloid_secondary_check": "the job creation figure (12,000 new jobs) or the maintenance sector decline (30%)",
        "policy_fact_1": "the foregone tax revenue (NOK 39.4B / \u20ac3.5B)",
        "policy_fact_2": "the air quality improvement (23% NO\u2082 decline) or EV market share (82.4%)",
    },
    {
        "topic": "WHO pandemic preparedness funding",
        "text": (
            "The WHO's 2023 Pandemic Preparedness and Response report estimated that "
            "closing the global preparedness gap would require an additional $31.1 billion "
            "per year over the next five years \u2014 a figure that, while large, represents "
            "roughly 0.5% of the $6.4 trillion governments spent responding to COVID-19 "
            "between 2020 and 2023. The Independent Panel for Pandemic Preparedness and "
            "Response, chaired by Helen Clark and Ellen Johnson Sirleaf, found that only "
            "38% of WHO member states had functional national pandemic preparedness plans "
            "as of 2022, down from an estimated 45% in pre-COVID self-assessments \u2014 "
            "suggesting that the pandemic exposed weaknesses in plans previously rated "
            "adequate. The proposed Pandemic Treaty, under negotiation since December "
            "2021, has secured signatures from 194 member states for the negotiating "
            "mandate but faces opposition from the United States and Brazil on provisions "
            "requiring pathogen-sharing and technology transfer."
        ),
        "key_facts": {
            "annual_cost": "$31.1 billion per year",
            "covid_spending": "$6.4 trillion",
            "preparedness_rate": "38% of member states",
            "pre_covid_rate": "45%",
            "treaty_signatories": "194 member states (negotiating mandate)",
        },
        "tabloid_fact_check": "the annual preparedness cost ($31.1 billion) or the COVID spending figure ($6.4 trillion)",
        "tabloid_secondary_check": "the preparedness plan gap (only 38% of countries have functional plans)",
        "policy_fact_1": "the preparedness funding gap ($31.1B/year) or comparison to COVID costs ($6.4T)",
        "policy_fact_2": "the preparedness rate (38%) or the treaty negotiation status (194 signatories)",
    },
    {
        "topic": "AI regulation compliance costs",
        "text": (
            "The EU AI Act, which entered into force in August 2024, establishes a "
            "tiered regulatory framework classifying AI systems by risk level. The "
            "European Commission's own impact assessment estimates compliance costs "
            "for high-risk AI systems at \u20ac6,000-\u20ac7,000 per system for conformity "
            "assessment, plus ongoing monitoring costs averaging \u20ac3,200 per year. A "
            "Stanford HAI survey of 152 AI companies found that 67% expected to spend "
            "more than \u20ac400,000 on initial compliance, with firms deploying foundation "
            "models facing costs exceeding \u20ac2 million. The Act imposes fines of up to "
            "\u20ac35 million or 7% of global annual turnover for the most serious violations. "
            "Industry body DigitalEurope estimates that the compliance burden will "
            "disproportionately affect SMEs, with companies under 250 employees spending "
            "an average of 14% of their AI R&D budget on regulatory compliance, compared "
            "to 3% for large enterprises. The first enforcement deadline for prohibited "
            "AI practices was February 2025."
        ),
        "key_facts": {
            "conformity_cost": "\u20ac6,000-\u20ac7,000 per system",
            "ongoing_cost": "\u20ac3,200 per year",
            "company_survey": "152 companies, 67% expect >\u20ac400K",
            "max_fine": "\u20ac35 million or 7% of turnover",
            "sme_burden": "14% of AI R&D budget (SMEs) vs 3% (large)",
        },
        "tabloid_fact_check": "the maximum fine (\u20ac35 million / 7% of turnover)",
        "tabloid_secondary_check": "the disproportionate impact on small businesses (14% vs 3% of R&D budget)",
        "policy_fact_1": "the per-system compliance cost (\u20ac6K-7K) or the company survey results (67% expect >\u20ac400K)",
        "policy_fact_2": "the SME burden ratio (14% vs 3%) or the maximum fine level (\u20ac35M / 7%)",
    },
]


def make_editorial_audience_adaptation(rand_seed: int = 42) -> RubricDatapoint:
    """Adapt a piece of analysis for different audiences.

    Given a technical analysis paragraph, the model must rewrite it for
    three different audiences. Tests ability to shift register, vocabulary,
    and framing while preserving factual content.

    Seedable: different seeds select different source topics.
    """
    rng = _random.Random(rand_seed)
    source = rng.choice(_ADAPTATION_SOURCE_POOL)
    topic_label = source["topic"]
    source_text = source["text"]

    return RubricDatapoint(
        problem_statement=f"""# Editorial Task: Audience Adaptation

You are an editor adapting content for different publications. Below is a
technical analysis paragraph about {topic_label}:

=== SOURCE PARAGRAPH ===
{source_text}
=== END SOURCE PARAGRAPH ===

Rewrite this paragraph for THREE different audiences. Each rewrite must
preserve the core factual content but adapt the language, framing, emphasis,
and level of detail for the target audience.

TARGET AUDIENCES:
1. **TABLOID** \u2014 Readers of a popular tabloid newspaper (reading age ~13,
   short attention span, needs a hook, uses everyday language, may use
   rhetorical questions). Target: 80-120 words.
2. **POLICY BRIEF** \u2014 Senior civil servants who need to make a decision.
   (Formal, action-oriented, focuses on outcomes and costs, minimal
   background needed). Target: 100-150 words.
3. **SOCIAL MEDIA** \u2014 A Twitter/X thread aimed at urbanist enthusiasts.
   (Informal, punchy, uses thread conventions like "1/" numbering,
   emphasizes surprising or shareable facts). Target: 3-5 tweets,
   each under 280 characters.

Write all three versions to /testbed/adaptations.txt, clearly labeled
with [TABLOID], [POLICY BRIEF], and [SOCIAL MEDIA] headers.""",
        rubric=(
            # --- STRUCTURAL ---
            BinaryRubricCategory(
                name="file_exists",
                question="Does /testbed/adaptations.txt exist with substantial content?",
                points=1,
            ),
            BinaryRubricCategory(
                name="three_sections_labeled",
                question="Are all three sections present with [TABLOID], [POLICY BRIEF], and [SOCIAL MEDIA] headers?",
                points=1,
            ),
            # --- TABLOID VERSION ---
            BinaryRubricCategory(
                name="tabloid_word_count",
                question="Is the [TABLOID] version between 80 and 120 words?",
                points=2,
            ),
            BinaryRubricCategory(
                name="tabloid_key_fact_preserved",
                question=f"Does the [TABLOID] version mention {source['tabloid_fact_check']}?",
                points=2,
            ),
            BinaryRubricCategory(
                name="tabloid_no_academic_citations",
                question="Is the [TABLOID] version free of academic citations (no 'et al.', no year-in-parentheses references)?",
                points=2,
            ),
            RubricCategory(
                name="tabloid_readability",
                description="Does the [TABLOID] version use simple, everyday language appropriate for a popular newspaper?",
                failure="Uses technical jargon throughout without explanation",
                minor_failure="Mostly simple but retains 2+ technical terms without explanation",
                minor_success="Simple language throughout with perhaps one slightly technical term",
                success="Fully accessible language, short sentences, possibly a hook or rhetorical question, reads like an actual tabloid article",
                points=3,
            ),
            BinaryRubricCategory(
                name="tabloid_secondary_fact",
                question=f"Does the [TABLOID] version mention {source['tabloid_secondary_check']}?",
                points=2,
            ),
            # --- POLICY BRIEF VERSION ---
            BinaryRubricCategory(
                name="policy_brief_word_count",
                question="Is the [POLICY BRIEF] version between 100 and 150 words?",
                points=2,
            ),
            BinaryRubricCategory(
                name="policy_brief_key_figure",
                question=f"Does the [POLICY BRIEF] version include {source['policy_fact_1']}?",
                points=2,
            ),
            BinaryRubricCategory(
                name="policy_brief_secondary_data",
                question=f"Does the [POLICY BRIEF] version include {source['policy_fact_2']}?",
                points=2,
            ),
            RubricCategory(
                name="policy_brief_action_orientation",
                description="Is the [POLICY BRIEF] version framed in terms of policy outcomes and implications rather than as storytelling?",
                failure="Reads like a narrative or news article, not a policy document",
                minor_failure="Somewhat outcome-focused but buries key policy-relevant data in narrative",
                minor_success="Clearly presents outcomes but could be more concise or action-oriented",
                success="Concise, action-oriented, leads with outcomes, structured for decision-makers (could include bullet points or key takeaways)",
                points=3,
            ),
            BinaryRubricCategory(
                name="policy_brief_formal_tone",
                question="Is the [POLICY BRIEF] version written in formal, professional tone (no colloquialisms, no rhetorical questions, no exclamation marks)?",
                points=2,
            ),
            # --- SOCIAL MEDIA VERSION ---
            BinaryRubricCategory(
                name="social_media_thread_format",
                question="Is the [SOCIAL MEDIA] version formatted as a numbered thread (using '1/', '2/' etc. or similar thread notation)?",
                points=1,
            ),
            BinaryRubricCategory(
                name="social_media_tweet_count",
                question="Does the [SOCIAL MEDIA] version contain 3-5 tweets/posts?",
                points=2,
            ),
            BinaryRubricCategory(
                name="social_media_each_under_280",
                question="Is each individual tweet/post under 280 characters?",
                points=2,
            ),
            RubricCategory(
                name="social_media_shareability",
                description="Does the [SOCIAL MEDIA] version emphasize surprising or shareable facts from the source material?",
                failure="Thread is dry and reads like a compressed academic paper",
                minor_failure="Some interesting framing but mostly just shrinks the facts without making them punchy",
                minor_success="Identifies 1-2 shareable angles but could be punchier",
                success="Leads with a hook, emphasizes surprising facts, uses informal/punchy language appropriate for social media",
                points=3,
            ),
            # --- CROSS-CUTTING ---
            BinaryRubricCategory(
                name="all_versions_factually_consistent",
                question="Are all three versions factually consistent with each other and the source paragraph (no contradictory numbers)?",
                points=3,
            ),
            RubricCategory(
                name="register_differentiation",
                description="Are the three versions genuinely written in different registers, or do they read like the same text with minor rewording?",
                failure="All three versions are in essentially the same register \u2014 you couldn't tell which audience they target",
                minor_failure="Slight register differences but largely the same vocabulary and sentence structure",
                minor_success="Clearly different registers for two of three, but one version bleeds into another's style",
                success="All three are unmistakably different in register: tabloid is simple/punchy, policy brief is formal/concise, social media is informal/thread-style",
                points=3,
            ),
        ),
        submission_instructions="Write all three adaptations to /testbed/adaptations.txt",
        available_tools=(tools.BASH_TOOL, tools.CREATE_FILE_TOOL, tools.FINISH_TOOL),
        necessary_files={"/testbed/source_paragraph.txt": source_text},
        problem_type="editorial",
    )


# =============================================================================
# 4. EDITORIAL ASSEMBLY  (static)
# =============================================================================


def make_editorial_assembly() -> RubricDatapoint:
    """Assemble a complete editorial from component pieces.

    Given pre-written components (headline, sources, argument paragraphs),
    the model must assemble and edit them into a coherent, publication-ready
    editorial. Tests editing, transitions, and structural judgment.
    """
    # Pre-written components the model must work with
    headline_draft = """EU's ICE Vehicle Ban: A Necessary Disruption

The European Parliament's vote to ban new combustion engine cars by 2035
marks a turning point \u2014 not just for the auto industry, but for the
continent's willingness to match climate rhetoric with regulatory action."""

    argument_paragraphs = """The environmental case is straightforward, almost tediously so. Transport
accounts for roughly a quarter of the EU's greenhouse gas emissions, and
passenger cars represent the single largest source within that sector. The
International Energy Agency's 2023 Global EV Outlook projects that without
regulatory intervention, ICE vehicles sold before 2035 will continue
emitting CO\u2082 until 2050 or beyond. The ban doesn't eliminate these legacy
vehicles, but it stops the bleeding.

What makes this regulation genuinely interesting, however, is the economic
argument. Norway, which has used tax incentives rather than outright bans,
has achieved 80% EV market share and seen its automotive maintenance sector
shrink by 30% \u2014 while EV charging infrastructure has created 12,000 new
jobs. The EU's \u20ac450 billion infrastructure investment requirement sounds
alarming until you compare it with the \u20ac280 billion the bloc currently
spends annually on imported oil. By 2040, McKinsey estimates the EV
transition will be a net economic positive.

The objection that this regulation kills consumer choice deserves a
response, if only because it is repeated so often. Nobody mourns the
consumer's lost "choice" to buy a car without seatbelts, or without
catalytic converters, or that runs on leaded petrol. Regulations that
phase out dangerous technology are not attacks on freedom; they are
the ordinary business of civilization."""

    counterarg_paragraph = """Germany's opposition, led by an automotive industry that employs 800,000
people directly and supports 1.8 million jobs in the supply chain, is
understandable. The transition will cause genuine pain in Wolfsburg and
Stuttgart and Munich. But the alternative \u2014 allowing German automakers
to keep building yesterday's technology while Chinese competitors like
BYD and NIO dominate the EV market \u2014 is a strategy for managed decline,
not preservation."""

    source_notes = """SOURCES USED:
- IEA Global EV Outlook 2023 \u2014 transport emissions data
- McKinsey "Power Play" report 2023 \u2014 economic transition projections
- Norwegian EV Association statistics \u2014 80% market share, job creation
- European Automobile Manufacturers' Association \u2014 German employment figures
- EU Parliament voting record \u2014 315-270 margin"""

    return RubricDatapoint(
        problem_statement="""# Editorial Task: Assembly and Final Edit

You are the commissioning editor assembling a complete editorial from
pre-written components. The components are saved as files in /testbed/components/:

- headline_draft.txt \u2014 Headline and standfirst (opening)
- argument_paragraphs.txt \u2014 Three core argument paragraphs
- counterarg_paragraph.txt \u2014 A paragraph addressing the opposition
- source_notes.txt \u2014 Source list used by the writers

Your task: Assemble these into a COMPLETE, PUBLICATION-READY editorial
and write it to /testbed/editorial.txt.

ASSEMBLY REQUIREMENTS:

1. **STRUCTURE**: The final editorial must have this structure:
   - Headline (on its own line)
   - Standfirst / opening paragraph
   - Core argument (the three paragraphs, possibly reordered)
   - Counterargument paragraph (placed where it's most effective)
   - A NEW closing paragraph that you write (80-120 words)
   - A byline line at the very end: "\u2014 Editorial Board"

2. **TRANSITIONS**: Add transition sentences between sections where needed.
   The components were written separately and may not flow naturally.

3. **EDITING**: You may make minor edits to the components for flow, but
   you must NOT substantially rewrite them. Acceptable edits:
   - Adding/modifying transition sentences
   - Fixing minor grammatical issues
   - Adjusting a word or phrase for flow
   - Reordering the three argument paragraphs
   NOT acceptable:
   - Rewriting entire paragraphs
   - Adding new argument paragraphs (beyond the closing)
   - Removing any of the provided paragraphs

4. **CLOSING**: Write a NEW closing paragraph (80-120 words) that:
   - Echoes the headline or opening in some way (circular structure)
   - Ends with a strong, quotable final sentence
   - Does NOT introduce new arguments or evidence
   - Provides a sense of resolution

5. **NO SOURCE LIST**: Do NOT include the source notes in the final editorial.
   Editorials don't have reference lists.

Read the component files, then assemble and write the complete editorial.""",
        rubric=(
            # --- FILE AND STRUCTURE ---
            BinaryRubricCategory(
                name="file_exists",
                question="Does /testbed/editorial.txt exist with substantial content (at least 1000 characters)?",
                points=1,
            ),
            BinaryRubricCategory(
                name="headline_present",
                question="Does the editorial begin with a headline on its own line?",
                points=1,
            ),
            BinaryRubricCategory(
                name="standfirst_after_headline",
                question="Does a standfirst/opening paragraph immediately follow the headline?",
                points=1,
            ),
            BinaryRubricCategory(
                name="all_three_argument_paragraphs_present",
                question="Are all three argument paragraphs from argument_paragraphs.txt present in the editorial (possibly reordered but substantively intact)?",
                points=3,
            ),
            BinaryRubricCategory(
                name="counterarg_paragraph_present",
                question="Is the counterargument paragraph from counterarg_paragraph.txt present in the editorial (substantively intact)?",
                points=2,
            ),
            BinaryRubricCategory(
                name="byline_at_end",
                question="Does the editorial end with the byline '\u2014 Editorial Board' (or close variant) on its own line?",
                points=1,
            ),
            BinaryRubricCategory(
                name="no_source_list",
                question="Is the source list / reference section excluded from the final editorial?",
                points=1,
            ),
            # --- CLOSING PARAGRAPH ---
            BinaryRubricCategory(
                name="closing_paragraph_exists",
                question="Is there a new closing paragraph that was NOT in the original components?",
                points=2,
            ),
            BinaryRubricCategory(
                name="closing_word_count",
                question="Is the closing paragraph between 80 and 120 words?",
                points=2,
            ),
            BinaryRubricCategory(
                name="closing_no_new_evidence",
                question="Does the closing paragraph avoid introducing new arguments, statistics, or evidence not present in the components?",
                points=2,
            ),
            RubricCategory(
                name="closing_echoes_opening",
                description="Does the closing paragraph echo or call back to the headline or opening paragraph, creating a circular structure?",
                failure="No connection to the opening \u2014 closing could belong to any editorial",
                minor_failure="Vague thematic similarity but no deliberate callback",
                minor_success="References the general topic of the opening but doesn't echo specific language or framing",
                success="Clearly echoes specific language, imagery, or framing from the headline or standfirst (e.g., references 'disruption', 'rhetoric vs action', or the turning point metaphor)",
                points=3,
            ),
            RubricCategory(
                name="closing_final_sentence",
                description="Does the closing end with a strong, quotable final sentence?",
                failure="Final sentence is weak, generic ('Time will tell'), or trails off",
                minor_failure="Final sentence is adequate but forgettable",
                minor_success="Final sentence is strong but somewhat generic for the topic",
                success="Final sentence is memorable, specific to this editorial's argument, and could stand alone as a pull-quote",
                points=3,
            ),
            # --- TRANSITIONS AND EDITING ---
            RubricCategory(
                name="transition_quality",
                description="Are there smooth transitions between the assembled sections (especially between the standfirst and first argument, and between argument paragraphs)?",
                failure="Sections are simply concatenated with no transitions \u2014 reads like separate documents pasted together",
                minor_failure="Some transitions added but they're awkward or formulaic ('Moving on to...')",
                minor_success="Most transitions are smooth, with perhaps one jarring join",
                success="All transitions are smooth and natural \u2014 the editorial reads as if written by a single author in one sitting",
                points=3,
            ),
            RubricCategory(
                name="counterarg_placement",
                description="Is the counterargument paragraph placed effectively within the editorial's structure?",
                failure="Counterargument is placed randomly (e.g., as the opening argument) where it undermines the editorial's flow",
                minor_failure="Placement is acceptable but not strategic",
                minor_success="Placed after the main arguments (a reasonable default) but transition could be smoother",
                success="Placed strategically (e.g., after building the positive case, before the closing) with smooth transitions that make the editorial's argument arc feel deliberate",
                points=3,
            ),
            BinaryRubricCategory(
                name="no_substantial_rewrites",
                question="Are the component paragraphs substantively intact (minor word changes are fine, but no paragraph has been rewritten more than ~10%)?",
                points=3,
            ),
            # --- OVERALL QUALITY ---
            RubricCategory(
                name="reads_as_coherent_piece",
                description="Does the assembled editorial read as a single coherent piece rather than a patchwork of components?",
                failure="Reads like separate documents pasted together \u2014 tonal shifts, redundancy between sections, no narrative arc",
                minor_failure="Mostly coherent but with 2+ noticeable seams between components",
                minor_success="Reads well with perhaps one slightly awkward transition or tonal inconsistency",
                success="Reads as a unified editorial that a single author could plausibly have written start-to-finish",
                points=3,
            ),
        ),
        submission_instructions="Write the complete assembled editorial to /testbed/editorial.txt",
        available_tools=(tools.BASH_TOOL, tools.CREATE_FILE_TOOL, tools.EDIT_TOOL, tools.LIST_DIRECTORY_CONTENTS_TOOL, tools.FINISH_TOOL),
        necessary_files={
            "/testbed/components/headline_draft.txt": headline_draft,
            "/testbed/components/argument_paragraphs.txt": argument_paragraphs,
            "/testbed/components/counterarg_paragraph.txt": counterarg_paragraph,
            "/testbed/components/source_notes.txt": source_notes,
        },
        problem_type="editorial",
    )


# =============================================================================
# 5. FACT-CHECK  (seedable)
# =============================================================================

# Pool of fact-check scenarios. Each has a draft editorial with deliberate
# errors, reference documents with verified facts, and rubric metadata for
# wiring error-specific binary checks.

_FACT_CHECK_POOL: list[dict] = [
    # ---- Topic 0: EU ICE vehicle ban (original) ----
    {
        "topic": "EU ICE vehicle ban",
        "draft": """EU's Bold Gamble: Why the 2035 ICE Ban Will Transform Europe

The European Parliament's decisive vote of 340-245 last month sent
shockwaves through the automotive industry. By banning the sale of all
new internal combustion engine vehicles by 2035 \u2014 the most aggressive
climate regulation in European history \u2014 the EU has committed to a
transformation that will reshape economies, supply chains, and daily
life across the continent.

The environmental imperative is clear. Transportation accounts for
approximately 40% of EU greenhouse gas emissions, with passenger vehicles
representing the majority of that share. According to the International
Energy Agency's 2023 Global EV Outlook, every year of delay in phasing
out ICE vehicles adds roughly 800 million tonnes of cumulative CO\u2082
emissions by 2050. The science is settled; the question is whether
politics can keep pace.

Norway offers the most compelling preview of this future. Having
implemented an outright ban on ICE sales in 2022, the Nordic nation now
boasts a 95% electric vehicle market share \u2014 the highest in the world.
The transition has created 25,000 new jobs in EV charging infrastructure
while reducing urban air pollution in Oslo by 35%. Critics who predicted
economic catastrophe have been silenced by Norway's 2.1% GDP growth
in 2023, outpacing the EU average.

The economic case deserves scrutiny, however. The EU's own impact
assessment estimates the transition will require \u20ac650 billion in
infrastructure investment by 2035, a figure that has drawn howls from
industry. Germany, whose automotive sector employs 1.2 million people
directly, voted unanimously against the measure. BMW's CEO Oliver Zipse
called it "an own goal of historic proportions" at the Munich Motor Show.

Yet the costs of inaction dwarf the costs of transition. The EU
currently spends \u20ac280 billion annually on imported crude oil \u2014 money
that flows predominantly to Russia, Saudi Arabia, and other
petrochemical states. McKinsey's 2023 "Power Play" report estimates
that by 2040, the EV transition will generate a net economic surplus
of \u20ac180 billion per year across the EU, driven by reduced fuel imports,
lower maintenance costs, and a booming European battery industry led
by Sweden's Northvolt, which recently opened Europe's largest
gigafactory in Skelleft\u00e5.

The ban is not without genuine risks. China's BYD and NIO currently
hold a 62% global market share in EV manufacturing, and Europe's
automakers have been slow to catch up. Without aggressive industrial
policy to complement the sales ban, Europe risks trading dependence
on Middle Eastern oil for dependence on Chinese batteries. The EU's
proposed \u20ac3.2 billion Battery Alliance fund is a start, but it pales
beside Beijing's $29 billion in EV subsidies last year alone.

History suggests these fears, while valid, are overstated. When
California mandated catalytic converters in 1975, the auto industry
predicted bankruptcy. Instead, converters became a $12 billion global
industry within a decade. When the EU mandated renewable energy targets
in 2009, skeptics warned of deindustrialization. Instead, the bloc's
renewable sector now employs 1.5 million people.

The vote is done. The clock is ticking. The question is no longer
whether Europe will transition, but whether it will lead the transition
or be dragged into it.

\u2014 Editorial Board""",
        "references": {
            "eu_parliament_record.txt": """EUROPEAN PARLIAMENT \u2014 OFFICIAL RECORD
Document: Regulation on CO2 emission standards for new passenger cars
Date of vote: [Plenary session]
Result: ADOPTED
    For:  315
    Against: 270
    Abstentions: 12

Summary: Regulation mandates zero-emission new cars and vans from 2035.
Several member states, notably Germany, voted against the measure.
Germany's opposition was led by the FDP coalition partner, not unanimous
among all German MEPs.

Note: This vote was NOT unanimous within any member state's delegation.
The German delegation was split, with several MEPs voting in favor.""",
            "transport_emissions_brief.txt": """EUROPEAN ENVIRONMENT AGENCY \u2014 TRANSPORT EMISSIONS BRIEF

Key statistics (verified, most recent available data):

1. TRANSPORT SHARE OF EMISSIONS
   Transport accounts for approximately 25% of total EU greenhouse gas
   emissions. Within transport, road transport is the dominant source,
   accounting for about 72% of transport emissions.

2. NORWAY EV MARKET
   Norway has achieved approximately 80% battery-electric vehicle (BEV)
   market share for new car sales. This was achieved through a
   comprehensive package of TAX INCENTIVES including:
   - Exemption from purchase tax and VAT
   - Reduced road tolls and ferry fares
   - Access to bus lanes in major cities
   IMPORTANT: Norway has NOT implemented an outright ban on ICE vehicle
   sales. The government set a non-binding target for 100% zero-emission
   new car sales by 2025, achieved through incentives rather than
   prohibition.

3. GERMANY AUTOMOTIVE EMPLOYMENT
   Germany's automotive sector directly employs approximately 800,000
   workers. Including the broader supply chain, the figure reaches
   approximately 1.8 million indirect and induced jobs.

4. HISTORICAL PRECEDENTS
   - California Air Resources Board mandated catalytic converters on
     new vehicles beginning in 1975. This is correct and well-documented.
   - The EU adopted the Renewable Energy Directive (2009/28/EC) in 2009,
     setting binding renewable energy targets. This is correct.""",
            "market_analysis.txt": """MARKET ANALYSIS \u2014 EV INDUSTRY COMPETITIVE LANDSCAPE

DISPUTED/UNVERIFIABLE CLAIMS TO NOTE:

1. BYD/NIO MARKET SHARE
   The claim that "BYD and NIO hold 62% global market share in EV
   manufacturing" is NOT supported by available data. As of 2023:
   - BYD held approximately 15-20% of global BEV sales
   - NIO held approximately 1-2% of global BEV sales
   - Combined, they are far below 62%
   - The 62% figure appears to be fabricated or conflated with
     China's overall share of global EV production (~60%)

2. The \u20ac280 billion figure for annual EU crude oil imports, the McKinsey
   "Power Play" report estimates, and the Norway jobs/air quality figures
   are not independently verifiable from our reference materials.
   These should be flagged as UNSUPPORTED rather than incorrect.

3. The \u20ac650 billion infrastructure investment figure differs from the
   \u20ac450 billion commonly cited. The source of this figure should be
   verified.""",
        },
        "errors": [
            {
                "name": "catches_vote_margin",
                "question": "Does the report identify that the vote margin '340-245' is incorrect (should be 315-270)?",
            },
            {
                "name": "catches_emissions_percentage",
                "question": "Does the report identify that '40%' for transport emissions share is incorrect (should be ~25%)?",
            },
            {
                "name": "catches_norway_ban_claim",
                "question": "Does the report identify that Norway has NOT implemented an 'outright ban' on ICE sales (it uses tax incentives)?",
            },
            {
                "name": "catches_norway_market_share",
                "question": "Does the report identify that Norway's EV market share is ~80%, not 95%?",
            },
            {
                "name": "catches_germany_employment",
                "question": "Does the report identify that Germany's automotive employment is ~800,000, not 1.2 million?",
            },
            {
                "name": "catches_china_market_share",
                "question": "Does the report identify that the '62% global market share' claim for BYD/NIO is fabricated?",
            },
        ],
        "false_positives": [
            {
                "name": "does_not_flag_california_1975",
                "question": "Does the report correctly NOT flag the California catalytic converter mandate date (1975) as an error?",
            },
            {
                "name": "does_not_flag_eu_2009_directive",
                "question": "Does the report correctly NOT flag the 2009 EU renewable energy directive as an error?",
            },
        ],
        "unsupported_examples": "the 25,000 jobs figure, the Oslo air pollution reduction, the Northvolt gigafactory claim",
    },
    # ---- Topic 1: US minimum wage increase debate ----
    {
        "topic": "US minimum wage increase debate",
        "draft": """The $15 Promise: Why America Can No Longer Afford a Poverty Wage

The Congressional Budget Office's landmark 2024 analysis could not have
been clearer: raising the federal minimum wage to $15 per hour would lift
3.6 million Americans out of poverty. Yet Congress remains paralysed,
trapped between the moral imperative of a living wage and exaggerated
fears of economic disruption.

The current federal minimum of $7.25 per hour has not been raised since
2006 \u2014 the longest period without an increase since the minimum wage
was established in 1938. Adjusted for inflation, a minimum-wage worker
today earns roughly 40% less in real terms than their counterpart in
1968, when the federal minimum reached its inflation-adjusted peak.

Critics invariably invoke job losses. The CBO itself projected that a
$15 minimum would eliminate approximately 2.7 million jobs, a figure
opponents have wielded like a cudgel. But this statistic requires context.
First, the CBO's own central estimate was 1.4 million jobs \u2014 the 2.7
million figure represents the upper bound of the confidence interval.
Second, the CBO projects that 17 million workers would see direct pay
increases, with another 10 million receiving indirect raises as wage
scales adjust upward.

The evidence from states that have already raised their minimums is
instructive. Washington state, which implemented a $15.74 minimum wage
in 2023, saw unemployment fall to 3.8% \u2014 below the national average
of 3.9%. California, with its $16 minimum effective April 2024, added
72,000 restaurant jobs in the six months following the increase, directly
contradicting predictions of mass layoffs.

Opponents also warn of small business closures, but the data tells a
more nuanced story. A 2023 study by Dube, Lester, and Reich tracking
restaurant employment across adjacent counties with different minimum
wages found "no statistically significant negative employment effects"
from minimum wage increases of up to 60% above the federal floor.
The study covered 318 county pairs over a 22-year period.

The real question is not whether a $15 minimum causes some disruption \u2014
it does \u2014 but whether the alternative is acceptable. A full-time worker
earning $7.25 per hour makes $21,060 annually after taxes, well below
the federal poverty line of $31,200 for a family of four. Taxpayers
effectively subsidize these poverty wages through $107 billion annually
in federal assistance programs \u2014 food stamps, Medicaid, housing
subsidies \u2014 that minimum-wage workers rely on to survive.

The minimum wage debate is not an economics problem. It is a values
problem dressed in economic clothing. The richest nation in human
history can afford to pay its workers enough to live.

\u2014 Editorial Board""",
        "references": {
            "cbo_analysis.txt": """CONGRESSIONAL BUDGET OFFICE \u2014 MINIMUM WAGE ANALYSIS

Report: "The Budgetary Effects of the Raise the Wage Act of 2021"
Published: February 2021 (updated estimates 2023)

KEY FINDINGS:
1. POVERTY REDUCTION: Raising the federal minimum to $15/hour by 2025
   would lift an estimated 0.9 million people out of poverty.
   NOTE: The 3.6 million figure cited in some analyses refers to the
   total number of people in families whose income would rise above
   the poverty threshold over several years, NOT a single-year estimate.

2. EMPLOYMENT EFFECTS:
   - Central estimate: 1.4 million workers would lose employment
   - Range: 0 to 2.7 million (confidence interval)
   - The 2.7 million figure is the upper bound, NOT the central estimate

3. WAGE INCREASES:
   - 17 million workers would receive direct pay increases
   - An additional 10 million workers could see indirect increases
   These figures are supported by the analysis.

4. MINIMUM WAGE HISTORY:
   - Current federal minimum: $7.25/hour since July 2009
   - The minimum was last raised in 2009, NOT 2006
   - The longest gap without an increase was 1997-2007 (10 years)""",
            "state_wage_data.txt": """STATE MINIMUM WAGE IMPLEMENTATION DATA

WASHINGTON STATE:
- 2023 minimum wage: $15.74 (correct)
- Unemployment rate 2023: approximately 4.2-4.5% (varied by month)
- The claim of 3.8% unemployment is NOT supported by Bureau of Labor
  Statistics data. Washington's annual average was 4.3% in 2023.

CALIFORNIA:
- $16 minimum wage for fast-food workers effective April 2024 (correct)
- General state minimum: $16/hour effective January 2024
- Restaurant employment data: Mixed. The state added approximately
  22,000 restaurant jobs in the six months after the increase, not
  72,000. The 72,000 figure appears to combine all food-service
  sectors including institutional and catering.

FEDERAL POVERTY LINE (2024):
- Family of four: $31,200 (correct)
- Single individual: $15,060""",
            "academic_research.txt": """ACADEMIC RESEARCH SUMMARY \u2014 MINIMUM WAGE EFFECTS

Dube, Lester, and Reich (various publications 2010-2023):
- The research does examine adjacent county pairs with different
  minimum wages.
- The finding of "no statistically significant negative employment
  effects" is accurately characterized.
- HOWEVER: The study covered 288 county pairs, NOT 318.
- The study period spans approximately 16 years, NOT 22 years.
- The "up to 60% above the federal floor" threshold is roughly correct.

PRE-TAX EARNINGS CALCULATION:
- $7.25/hour \u00d7 2,080 hours (full-time) = $15,080 GROSS annual income
- The draft states "$21,060 annually after taxes" which is incorrect.
  Gross annual income at $7.25/hr is $15,080 pre-tax.
  After standard deduction, a single filer would owe approximately
  $500-$800 in federal tax, netting roughly $14,300-$14,600.

HISTORICAL MINIMUM WAGE:
- The minimum wage was established by the Fair Labor Standards Act of
  1938. This is correct.
- The inflation-adjusted peak was 1968 at approximately $12.50 in
  2024 dollars.
- The "40% less in real terms" claim is approximately correct.""",
        },
        "errors": [
            {
                "name": "catches_last_raise_date",
                "question": "Does the report identify that the minimum wage was last raised in 2009, not 2006?",
            },
            {
                "name": "catches_poverty_reduction_figure",
                "question": "Does the report identify that the CBO central poverty reduction estimate is 0.9 million, not 3.6 million (which is a cumulative/multi-year figure)?",
            },
            {
                "name": "catches_washington_unemployment",
                "question": "Does the report identify that Washington state's unemployment was approximately 4.3%, not 3.8%?",
            },
            {
                "name": "catches_california_jobs",
                "question": "Does the report identify that California added approximately 22,000 restaurant jobs, not 72,000?",
            },
            {
                "name": "catches_county_pairs",
                "question": "Does the report identify that the Dube et al. study covered 288 county pairs, not 318?",
            },
            {
                "name": "catches_annual_income",
                "question": "Does the report identify that $7.25/hour full-time gross income is approximately $15,080, not $21,060?",
            },
        ],
        "false_positives": [
            {
                "name": "does_not_flag_1938_establishment",
                "question": "Does the report correctly NOT flag the minimum wage establishment date (1938) as an error?",
            },
            {
                "name": "does_not_flag_1968_peak",
                "question": "Does the report correctly NOT flag the claim about 1968 being the inflation-adjusted peak as an error?",
            },
        ],
        "unsupported_examples": "the $107 billion annual federal assistance figure, the 10 million indirect raises claim specifics",
    },
    # ---- Topic 2: Global AI regulation framework ----
    {
        "topic": "Global AI regulation framework",
        "draft": """The Algorithm's Reckoning: Why AI Regulation Cannot Wait

In March 2024, the European Union made history by adopting the AI Act \u2014
the world's first comprehensive legal framework for artificial intelligence.
The regulation, which passed the European Parliament by a vote of 523-46
with 49 abstentions, establishes binding rules for AI systems based on
their assessed risk level. It is, by any measure, a watershed moment.

Yet the global regulatory landscape remains fragmented. While the EU has
chosen prescriptive legislation, the United States relies on a patchwork
of voluntary commitments and executive orders. President Biden's October
2023 Executive Order on AI Safety established reporting requirements for
companies developing foundation models that use more than 10^25 FLOPS
of computing power during training. China, meanwhile, has taken a
sector-specific approach, issuing binding regulations for generative AI,
recommendation algorithms, and deepfakes since 2022.

The economic stakes are staggering. McKinsey's 2023 report estimated
that generative AI alone could add $4.4 trillion in annual value to
the global economy. Goldman Sachs projected that AI could increase
global GDP by 7% over a ten-year period \u2014 roughly $7 trillion. Yet
the same technologies pose systemic risks: a 2024 Stanford HAI survey
found that 82% of AI researchers believe there is a 10% or greater
chance that AI will cause "human extinction or similarly permanent and
severe disempowerment."

The compliance burden is already substantial. The EU AI Act's conformity
assessment for high-risk systems costs an estimated \u20ac60,000-\u20ac70,000 per
system, with ongoing monitoring adding \u20ac32,000 annually. For companies
deploying foundation models, total compliance costs could exceed
\u20ac20 million. A survey by DigitalEurope found that 45% of European AI
startups are considering relocating to jurisdictions with lighter
regulatory environments.

The counterargument \u2014 that regulation stifles innovation \u2014 deserves
examination. The EU's GDPR, adopted in 2018, was similarly decried as
a growth killer. Six years later, Europe's data protection industry
employs 500,000 people, and GDPR has become the de facto global
standard adopted by 157 countries. Regulation creates markets as often
as it constrains them.

The real danger is not over-regulation but under-coordination. When
financial regulators failed to coordinate on derivatives oversight in
2008, the result was a global crisis. AI's cross-border nature makes
fragmented regulation not just inefficient but dangerous.

\u2014 Editorial Board""",
        "references": {
            "eu_ai_act_record.txt": """EU AI ACT \u2014 LEGISLATIVE RECORD

European Parliament vote (March 13, 2024):
    For: 523
    Against: 46
    Abstentions: 49

This is correct as stated.

Entry into force: August 1, 2024
First enforcement deadline (prohibited practices): February 2, 2025
Full application for high-risk systems: August 2, 2026

The AI Act classifies systems into four risk tiers:
- Unacceptable risk (banned): social scoring, real-time biometric ID
- High risk: hiring tools, credit scoring, law enforcement
- Limited risk: chatbots, deepfakes (transparency obligations)
- Minimal risk: spam filters, games (no requirements)""",
            "compliance_costs.txt": """AI REGULATION COMPLIANCE COST DATA

EU AI ACT COSTS (European Commission impact assessment):
- High-risk AI conformity assessment: \u20ac6,000-\u20ac7,000 per system
  NOTE: The draft claims \u20ac60,000-\u20ac70,000 \u2014 this is OFF BY A FACTOR
  OF TEN. The actual estimate is \u20ac6,000-\u20ac7,000.
- Ongoing monitoring costs: \u20ac3,200 per year
  NOTE: The draft claims \u20ac32,000 annually \u2014 again off by a factor of ten.
- Foundation model compliance: Stanford HAI survey of 152 companies
  found 67% expected to spend more than \u20ac400,000 on initial compliance,
  with some exceeding \u20ac2 million. The \u20ac20 million claim is unsupported.

DIGITALEUROPE SURVEY:
- DigitalEurope did conduct surveys on AI Act impact.
- The "45% considering relocating" figure is NOT from any published
  DigitalEurope survey. The actual finding was that 28% of surveyed
  companies expressed concern about competitiveness impacts.

GDPR COMPARISONS:
- GDPR entered application in May 2018, NOT "adopted in 2018."
  The regulation was adopted in April 2016 with a 2-year transition.
- The claim that 157 countries have adopted GDPR-like laws is
  exaggerated. As of 2024, approximately 137 countries have data
  protection legislation, but many predate and differ substantially
  from GDPR.
- The 500,000 employment figure for Europe's "data protection industry"
  is not verifiable from available data.""",
            "ai_risk_research.txt": """AI RISK AND ECONOMIC IMPACT RESEARCH

ECONOMIC PROJECTIONS:
- McKinsey (June 2023): Generative AI could add $2.6 to $4.4 trillion
  in annual value. The $4.4T figure is the upper bound.
- Goldman Sachs (March 2023): AI could raise global GDP by 7% (roughly
  $7 trillion) over a 10-year period. This is correctly cited.

RESEARCHER SURVEYS:
- The claim about "82% of AI researchers" believing in 10%+ extinction
  risk is INACCURATE. The 2023 AI Impacts survey found that a median
  of approximately 5% extinction-level risk was assigned by researchers.
  A separate question found that ~38% of respondents thought AI could
  lead to an "extremely bad" outcome (defined broadly). The 82% figure
  does not correspond to any published survey result.

BIDEN EXECUTIVE ORDER:
- Executive Order 14110, issued October 30, 2023 (correct).
- The FLOPS threshold for reporting is 10^26, NOT 10^25 as stated
  in the draft.""",
        },
        "errors": [
            {
                "name": "catches_conformity_cost",
                "question": "Does the report identify that the conformity assessment cost is \u20ac6,000-\u20ac7,000, not \u20ac60,000-\u20ac70,000 (off by 10x)?",
            },
            {
                "name": "catches_monitoring_cost",
                "question": "Does the report identify that ongoing monitoring costs are \u20ac3,200/year, not \u20ac32,000 (off by 10x)?",
            },
            {
                "name": "catches_researcher_survey",
                "question": "Does the report identify that the '82% of AI researchers' extinction risk claim is not supported by published survey data?",
            },
            {
                "name": "catches_flops_threshold",
                "question": "Does the report identify that the FLOPS threshold is 10^26, not 10^25?",
            },
            {
                "name": "catches_relocation_figure",
                "question": "Does the report identify that the '45% considering relocating' is not from a published DigitalEurope survey (actual finding was 28% concerned about competitiveness)?",
            },
            {
                "name": "catches_gdpr_countries",
                "question": "Does the report identify that the '157 countries' adopting GDPR-like laws is exaggerated (approximately 137 have data protection laws, many differing from GDPR)?",
            },
        ],
        "false_positives": [
            {
                "name": "does_not_flag_parliament_vote",
                "question": "Does the report correctly NOT flag the EU Parliament vote count (523-46 with 49 abstentions) as an error?",
            },
            {
                "name": "does_not_flag_goldman_sachs",
                "question": "Does the report correctly NOT flag the Goldman Sachs 7% GDP projection as an error?",
            },
        ],
        "unsupported_examples": "the 500,000 data protection employment figure, the \u20ac20 million foundation model compliance cost",
    },
    # ---- Topic 3: WHO pandemic preparedness treaty ----
    {
        "topic": "WHO pandemic preparedness treaty",
        "draft": """The Next Pandemic Will Not Wait: Why the WHO Treaty Must Succeed

COVID-19 killed an estimated 6.9 million people according to official
WHO counts, though excess mortality studies suggest the true figure
exceeds 22 million. The economic toll was equally devastating: the IMF
calculated that governments worldwide spent $16.4 trillion responding
to the pandemic between 2020 and 2023. Against this backdrop, the
world's failure to agree on a binding pandemic treaty is not merely
disappointing \u2014 it is unconscionable.

The WHO's proposed Pandemic Preparedness and Response Treaty, under
negotiation since December 2021, aims to establish binding commitments
for pathogen sharing, equitable access to vaccines and treatments, and
sustainable financing for health security. The negotiating mandate was
supported by all 194 WHO member states. Yet two years of negotiations
have produced more disagreement than consensus.

The preparedness gap is alarming. The WHO's 2023 assessment found that
only 38% of member states had functional national pandemic preparedness
plans \u2014 a figure that actually decreased from 52% in pre-COVID
self-assessments, suggesting that the pandemic exposed plans that
existed only on paper. The Independent Panel for Pandemic Preparedness,
chaired by Helen Clark and Ellen Johnson Sirleaf, estimated that
closing this gap would require $31.1 billion per year for five years.

That figure sounds enormous until you consider the alternative. The
$31.1 billion annual investment represents roughly 0.5% of the
$6.4 trillion spent on COVID response \u2014 an insurance premium that
makes fiscal sense by any rational calculation. Yet as of 2024, only
$3.5 billion of the recommended annual funding has been committed,
leaving a $27.6 billion shortfall.

The treaty negotiations have stumbled on three principal obstacles.
First, the United States and Brazil have opposed provisions requiring
mandatory pathogen-sharing with the WHO within 48 hours of detection.
Second, developing nations led by the Africa Group insist on binding
technology transfer commitments, which pharmaceutical companies in
Europe and North America fiercely resist. Third, there is fundamental
disagreement over governance: the G7 favors a "framework convention"
with voluntary protocols, while 120 nations in the G77 demand a
binding treaty with enforcement mechanisms.

Critics who dismiss the treaty as unrealistic ignore history. The
Framework Convention on Tobacco Control, adopted by the World Health
Assembly in 2003, was similarly contentious. It now has 168 signatories
and has contributed to a 25% reduction in global smoking prevalence.
International coordination works \u2014 when nations have the political will.

\u2014 Editorial Board""",
        "references": {
            "who_data.txt": """WHO PANDEMIC DATA AND TREATY STATUS

COVID-19 MORTALITY:
- Official WHO count (as of May 2024): approximately 7.0 million deaths
  The "6.9 million" figure in the draft is approximately correct and
  within rounding of official figures.
- Excess mortality estimates: The WHO's own excess mortality working
  group estimated 14.9 million excess deaths (2020-2021 only). The
  Lancet published estimates of approximately 18.2 million through
  March 2022. The "22 million" figure exceeds any published estimate.

PANDEMIC SPENDING:
- IMF estimate: Governments spent approximately $16.4 trillion
  (fiscal measures) globally. However, the IMF figure covers fiscal
  support measures, not just direct pandemic response costs.
- The WHO report references $6.4 trillion as a separate figure for
  direct health spending. These are different metrics.

TREATY STATUS:
- Negotiating mandate: Agreed by special session of World Health
  Assembly, December 2021. Supported by all 194 member states. CORRECT.
- The treaty is formally titled "WHO convention, agreement or other
  international instrument on pandemic prevention, preparedness and
  response" (the name "Pandemic Preparedness and Response Treaty" is
  a common shorthand).""",
            "preparedness_assessment.txt": """PANDEMIC PREPAREDNESS ASSESSMENT DATA

WHO 2023 ASSESSMENT:
- 38% of member states had functional national pandemic preparedness
  plans. CORRECT.
- Pre-COVID self-assessment rate: approximately 45%, NOT 52% as
  claimed in the draft. The 45% figure comes from the 2019 Joint
  External Evaluation synthesis.

FUNDING:
- The Independent Panel recommended $31.1 billion per year. CORRECT.
- Chaired by Helen Clark and Ellen Johnson Sirleaf. CORRECT.
- The 0.5% of COVID spending comparison is based on the $6.4 trillion
  figure. The calculation checks out: $31.1B / $6.4T \u2248 0.49%.

COMMITTED FUNDING:
- As of 2024, the Pandemic Fund (hosted by the World Bank) had
  received approximately $2.0 billion in pledges, NOT $3.5 billion.
- The shortfall calculation in the draft is therefore also incorrect.""",
            "treaty_negotiations.txt": """PANDEMIC TREATY NEGOTIATION DETAILS

OPPOSITION POSITIONS:
- United States: Has expressed concerns about sovereignty implications
  and mandatory pathogen-sharing. The "within 48 hours" requirement
  was proposed but is NOT in the current draft text \u2014 it was removed
  during the INB5 negotiating round. The draft editorial implies this
  is still a sticking point, which is misleading.
- Brazil: Has raised concerns aligned with the US position. CORRECT.

FRAMEWORK CONVENTION ON TOBACCO CONTROL:
- Adopted in 2003 by the World Health Assembly. CORRECT.
- As of 2024, it has 183 parties (not "168 signatories"). The
  distinction matters: 183 countries are full parties who have
  ratified; there were originally 168 signatories before ratification.
- The "25% reduction in global smoking prevalence" is approximately
  correct: global smoking rates fell from roughly 24% in 2007 to
  about 17% in 2023 \u2014 approximately a 29% relative reduction.

TECHNOLOGY TRANSFER:
- The Africa Group's position on binding technology transfer is
  correctly characterized.
- The G7/G77 framing is oversimplified but substantially accurate.""",
        },
        "errors": [
            {
                "name": "catches_excess_mortality",
                "question": "Does the report identify that the '22 million' excess mortality figure exceeds published estimates (WHO estimates ~14.9M, Lancet ~18.2M)?",
            },
            {
                "name": "catches_pre_covid_rate",
                "question": "Does the report identify that the pre-COVID preparedness self-assessment rate was approximately 45%, not 52%?",
            },
            {
                "name": "catches_committed_funding",
                "question": "Does the report identify that committed funding is approximately $2.0 billion, not $3.5 billion?",
            },
            {
                "name": "catches_fctc_parties",
                "question": "Does the report identify that the FCTC has 183 parties, not '168 signatories' (or note the signatory vs party distinction)?",
            },
            {
                "name": "catches_48_hour_requirement",
                "question": "Does the report identify that the 48-hour pathogen-sharing requirement was removed from the current draft text?",
            },
        ],
        "false_positives": [
            {
                "name": "does_not_flag_31_billion",
                "question": "Does the report correctly NOT flag the $31.1 billion annual investment figure as an error?",
            },
            {
                "name": "does_not_flag_clark_sirleaf",
                "question": "Does the report correctly NOT flag the Independent Panel chairs (Helen Clark and Ellen Johnson Sirleaf) as an error?",
            },
        ],
        "unsupported_examples": "the characterization of pharmaceutical company resistance, the 120-nation G77 demand framing",
    },
    # ---- Topic 4: Carbon border adjustment mechanism ----
    {
        "topic": "Carbon border adjustment mechanism",
        "draft": """Europe's Carbon Tariff: Fair Trade or Protectionism in Green Clothing?

The European Union's Carbon Border Adjustment Mechanism \u2014 CBAM \u2014 entered
its transitional phase in October 2023, marking the most ambitious attempt
in history to put a price on the carbon embedded in international trade.
By requiring importers of cement, steel, aluminium, fertilisers, electricity,
and hydrogen to purchase certificates matching the EU's own carbon price,
CBAM aims to prevent "carbon leakage": the relocation of polluting
industries to countries with weaker climate policies.

The mechanism is straightforward in principle but revolutionary in practice.
From 2026, importers will buy CBAM certificates at the prevailing EU
Emissions Trading System price \u2014 approximately \u20ac90 per tonne of CO\u2082 as
of late 2023. The European Commission projects this will prevent 95 million
tonnes of carbon leakage annually by 2030, equivalent to the combined
annual emissions of Belgium and Portugal.

The early evidence is encouraging. During the first reporting period,
78% of affected importers registered with the CBAM authorities \u2014 a
compliance rate that exceeded expectations. However, the administrative
burden has been significant: importers must now track and verify the
embedded carbon in every covered product, a process the European Steel
Association estimates adds 3-5 business days to import clearance times.

Critics are not hard to find. Turkey, which sends 42% of its steel
exports to the EU, has denounced CBAM as a violation of WTO most-
favoured-nation principles. Russia has threatened retaliatory carbon
tariffs. India's commerce minister called it "eco-imperialism designed
to protect inefficient European producers." Even within the EU,
industry group Eurofer warns that CBAM will add 25-30% to the cost
of imported steel from non-EU countries, making European manufacturers
less competitive in global markets where they must compete against
producers who face no such carbon costs.

The environmental case is compelling but not bulletproof. The World
Bank projects CBAM will reduce EU imports of carbon-intensive goods
by 15-20% by 2030, disproportionately affecting Turkey, Russia, and
China. But environmentalists warn of "resource shuffling" \u2014 carbon-
intensive producers simply redirecting their dirtiest products to
non-EU markets while sending cleaner goods to Europe, producing no
net emissions reduction globally.

What CBAM's critics miss is the mechanism's true purpose: not
protectionism, but universalization. If the EU's carbon price is
eventually mirrored by trading partners \u2014 as seems increasingly likely,
with the UK, Canada, and Australia all considering similar mechanisms
\u2014 the result will be a de facto global carbon price. That outcome
would do more to reduce emissions than any climate summit to date.

\u2014 Editorial Board""",
        "references": {
            "cbam_official.txt": """EU CBAM \u2014 OFFICIAL IMPLEMENTATION DATA

TRANSITIONAL PHASE:
- Start date: October 1, 2023 (CORRECT)
- Covered sectors: cement, iron and steel, aluminium, fertilisers,
  electricity, and hydrogen (CORRECT)
- Full application with financial obligations: January 1, 2026

COMPLIANCE DATA (first reporting period):
- Registration rate: 78% of affected importers (CORRECT)
- The European Commission described this as "in line with expectations,"
  not "exceeding expectations" as the draft claims. However, this is
  a framing difference, not a factual error.

CARBON LEAKAGE PREVENTION:
- The European Commission estimates CBAM will prevent 47 million tonnes
  of carbon leakage annually by 2030. The draft's claim of "95 million
  tonnes" is approximately DOUBLE the official estimate.
- The comparison to "Belgium and Portugal combined" emissions is based
  on the inflated 95M figure and is therefore also misleading.""",
            "trade_impact.txt": """CBAM TRADE IMPACT ANALYSIS

STEEL COST IMPACT:
- Industry group Eurofer estimates CBAM will add 12-18% to the cost
  of imported steel from non-EU countries. The draft's claim of
  "25-30%" is significantly higher than Eurofer's published estimate.

TURKEY:
- Turkey is a major steel exporter to the EU, but its steel export
  share to the EU is approximately 28-33% of total steel exports,
  NOT 42% as claimed in the draft.

IMPORT REDUCTION PROJECTIONS:
- World Bank projection: CBAM will reduce EU imports of carbon-
  intensive goods by 8-12% by 2030 (NOT 15-20% as stated).
- Disproportionate impact on Turkey, Russia, and China is correct.

ADMINISTRATIVE BURDEN:
- Import clearance delays: No official data is available on specific
  business-day delays. The "3-5 business days" figure attributed to
  the European Steel Association is not verifiable from our sources.

WTO CONCERNS:
- Multiple WTO members have raised concerns about CBAM's compatibility
  with MFN principles. Turkey's position is correctly characterized.
- India's commerce minister has been critical of CBAM, though the
  exact "eco-imperialism" quote is not verified in our sources.""",
            "ets_pricing.txt": """EU EMISSIONS TRADING SYSTEM \u2014 PRICE DATA

ETS CARBON PRICE:
- Q3 2023 average: approximately \u20ac85-\u20ac90 per tonne of CO\u2082
- The "\u20ac90" figure in the draft is within the correct range.
- 2024 prices have fluctuated between \u20ac55 and \u20ac80.

INTERNATIONAL CARBON PRICING:
- UK: Operating its own ETS since January 2021. Has discussed
  linking with the EU ETS and exploring a CBAM-like mechanism.
- Canada: Has a federal carbon price ($80 CAD/tonne in 2024).
  Has publicly discussed CBAM-like measures.
- Australia: Does NOT currently have a carbon pricing mechanism
  following the repeal of its carbon tax in 2014. While there
  have been policy discussions, characterizing Australia as
  "considering" a CBAM-like mechanism is misleading.

RESOURCE SHUFFLING:
- The concern about resource shuffling is well-documented in
  academic literature and is a legitimate critique of border
  carbon adjustments.""",
        },
        "errors": [
            {
                "name": "catches_leakage_tonnes",
                "question": "Does the report identify that CBAM is estimated to prevent 47 million tonnes of carbon leakage, not 95 million tonnes?",
            },
            {
                "name": "catches_steel_cost",
                "question": "Does the report identify that Eurofer estimates 12-18% steel cost increase, not 25-30%?",
            },
            {
                "name": "catches_turkey_exports",
                "question": "Does the report identify that Turkey's steel export share to the EU is approximately 28-33%, not 42%?",
            },
            {
                "name": "catches_import_reduction",
                "question": "Does the report identify that the World Bank projects 8-12% import reduction, not 15-20%?",
            },
            {
                "name": "catches_australia_cbam",
                "question": "Does the report identify that Australia does not currently have a carbon pricing mechanism and characterizing it as 'considering' a CBAM-like mechanism is misleading?",
            },
        ],
        "false_positives": [
            {
                "name": "does_not_flag_78pct_compliance",
                "question": "Does the report correctly NOT flag the 78% importer registration rate as an error?",
            },
            {
                "name": "does_not_flag_ets_price",
                "question": "Does the report correctly NOT flag the \u20ac90 per tonne carbon price as an error?",
            },
        ],
        "unsupported_examples": "the '3-5 business days' clearance delay figure, the exact 'eco-imperialism' quote from India's commerce minister",
    },
]


def make_editorial_fact_check(rand_seed: int = 42) -> RubricDatapoint:
    """Editorial Task: Fact-check a draft editorial against reference documents.

    Given a draft editorial with deliberate errors AND reference documents
    containing verified facts, the model must cross-reference the draft
    against the references to identify inaccuracies. The verified facts
    are spread across multiple files -- the model must read and synthesize
    them, not just reformat an answer key.

    Seedable: different seeds select different topics with different
    drafts, errors, and reference documents.
    """
    rng = _random.Random(rand_seed)
    scenario = rng.choice(_FACT_CHECK_POOL)

    # Build necessary_files from scenario
    necessary_files: dict[str, str] = {
        "/testbed/draft_editorial.txt": scenario["draft"],
    }
    for filename, content in scenario["references"].items():
        necessary_files[f"/testbed/reference/{filename}"] = content

    # Build error-detection rubric items
    error_checks: list[BinaryRubricCategory] = []
    for err in scenario["errors"]:
        error_checks.append(
            BinaryRubricCategory(
                name=err["name"],
                question=err["question"],
                points=3,
            )
        )

    # Build false-positive rubric items
    fp_checks: list[BinaryRubricCategory] = []
    for fp in scenario["false_positives"]:
        fp_checks.append(
            BinaryRubricCategory(
                name=fp["name"],
                question=fp["question"],
                points=2,
            )
        )

    unsupported_examples = scenario["unsupported_examples"]

    rubric_categories = (
        # --- STRUCTURAL ---
        BinaryRubricCategory(
            name="file_exists",
            question="Does /testbed/fact_check.txt exist with substantial content?",
            points=1,
        ),
        BinaryRubricCategory(
            name="errors_numbered",
            question="Are the factual errors presented as a numbered list?",
            points=1,
        ),
        BinaryRubricCategory(
            name="each_error_has_quote",
            question="Does each identified error include a direct quote from the draft editorial?",
            points=2,
        ),
        BinaryRubricCategory(
            name="each_error_has_correction",
            question="Does each identified error state what the correct fact is?",
            points=2,
        ),
        BinaryRubricCategory(
            name="each_error_has_severity",
            question="Does each identified error have a severity rating (CRITICAL, MODERATE, or MINOR)?",
            points=1,
        ),
        # --- ERROR DETECTION ---
        *error_checks,
        # --- FALSE POSITIVES ---
        *fp_checks,
        # --- SEVERITY RATINGS ---
        RubricCategory(
            name="severity_ratings_sensible",
            description="Are the severity ratings (CRITICAL/MODERATE/MINOR) logically applied?",
            failure="Severity ratings are random or inverted (e.g., a core factual error rated MINOR, a trivial imprecision rated CRITICAL)",
            minor_failure="Most ratings sensible but 2+ are questionable",
            minor_success="Ratings are mostly sensible with one debatable assignment",
            success="All severity ratings are well-justified and logically consistent \u2014 errors that undermine core arguments are CRITICAL, imprecise numbers are MODERATE or MINOR",
            points=3,
        ),
        # --- UNSUPPORTED CLAIMS SECTION ---
        BinaryRubricCategory(
            name="unsupported_claims_section_exists",
            question="Is there a separate section for unsupported claims (distinct from the factual errors section)?",
            points=2,
        ),
        RubricCategory(
            name="unsupported_claims_reasonable",
            description="Does the unsupported claims section identify genuinely unverifiable statements (not just re-listing the factual errors)?",
            failure="Section is empty, missing, or just repeats the factual errors",
            minor_failure="Lists 1-2 items but misses obvious unverifiable claims",
            minor_success="Identifies several genuinely unverifiable claims",
            success=f"Identifies multiple unverifiable claims and correctly distinguishes them from verifiable errors \u2014 e.g., {unsupported_examples}",
            points=3,
        ),
        # --- FINAL VERDICT ---
        BinaryRubricCategory(
            name="verdict_present",
            question="Does the report end with a clear VERDICT (one of the three specified options) and a 2-3 sentence justification?",
            points=2,
        ),
        RubricCategory(
            name="verdict_appropriate",
            description="Is the verdict appropriate given the number and severity of errors found?",
            failure="Verdict contradicts the findings (e.g., 'PUBLISHABLE' despite 5+ errors including critical ones)",
            minor_failure="Verdict is defensible but overly lenient or harsh",
            minor_success="Verdict is reasonable and somewhat justified",
            success="Verdict correctly reflects the pattern of errors: given multiple errors including critical factual mistakes, 'NEEDS MAJOR REVISION' or 'UNPUBLISHABLE' is appropriate, with clear reasoning",
            points=3,
        ),
    )

    return RubricDatapoint(
        problem_statement=f"""# Editorial Task: Fact-Check Report

You are a fact-checker at a national newspaper. A draft editorial about
{scenario["topic"]} has been submitted for publication. Your job is to
produce a FACT-CHECK REPORT identifying ALL factual errors, unsupported
claims, and misleading framings.

The draft editorial is saved at /testbed/draft_editorial.txt

You have access to REFERENCE DOCUMENTS containing verified facts in
/testbed/reference/. You must cross-reference the draft against these
documents to identify errors. Read ALL reference files before producing
your report.

Produce a structured fact-check report at /testbed/fact_check.txt with:

1. A numbered list of EVERY factual error found, in the order they appear
2. For each error:
   - Quote the specific incorrect claim (in quotation marks)
   - State what the correct fact is (citing the reference document)
   - Rate severity: CRITICAL (changes the argument), MODERATE (misleading but
     doesn't invalidate the argument), or MINOR (imprecise but not misleading)
3. A separate section listing any UNSUPPORTED CLAIMS \u2014 statements presented
   as fact that may or may not be true but are not verifiable from the
   reference documents
4. A final VERDICT: "PUBLISHABLE WITH CORRECTIONS", "NEEDS MAJOR REVISION",
   or "UNPUBLISHABLE" with a 2-3 sentence justification

DO NOT rewrite the editorial. Only produce the fact-check report.""",
        rubric=tuple(rubric_categories),
        submission_instructions="Write your fact-check report to /testbed/fact_check.txt",
        available_tools=(tools.BASH_TOOL, tools.CREATE_FILE_TOOL, tools.LIST_DIRECTORY_CONTENTS_TOOL, tools.FINISH_TOOL),
        necessary_files=necessary_files,
        problem_type="editorial",
    )
