"""Dataset definitions for the Rubric Environment.

Each datapoint consists of:
- A problem statement shown to the model
- A rubric with N categories, each with a configurable point value
- Submission instructions
- Available tools (controls agent capability)
- Optional files to upload to the container

Category types:
- RubricCategory: 4-level graded (Failure/Minor Failure/Minor Success/Success)
- BinaryRubricCategory: yes/no check

Each category has a `points` field controlling its max score. The grader
outputs level names (e.g. "Success", "Yes"), and code maps them to numeric
scores interpolated across the category's point range.

Final reward = sum(category_scores) / sum(category_max_points), yielding [0.0, 1.0].
"""

import random as _random
from dataclasses import dataclass, field
from typing import Union

from tinker_cookbook.rl.envs import tools
from tinker_cookbook.rl.envs.tools import ToolSpec


@dataclass(frozen=True, slots=True)
class RubricCategory:
    """A graded rubric category scored on a 4-point scale.

    The grader outputs one of: Failure, Minor Failure, Minor Success, Success.
    These map to numeric scores interpolated across the category's point range:
        Failure      = 0
        Minor Failure = round(1/3 * points)
        Minor Success = round(2/3 * points)
        Success       = points
    """

    name: str  # Machine-readable identifier, e.g. "argument_quality"
    description: str  # What the grader should evaluate
    failure: str  # Description of what Failure looks like
    minor_failure: str  # Description of what Minor Failure looks like
    minor_success: str  # Description of what Minor Success looks like
    success: str  # Description of what Success looks like
    points: int = 3  # Max points for this category


@dataclass(frozen=True, slots=True)
class BinaryRubricCategory:
    """A binary (yes/no) rubric category.

    Inspired by GDPval's presence/absence checks. The grader answers a
    yes/no question. "Yes" = points, "No" = 0.

    The `points` field controls how much this check is worth. Trivial
    structural checks (file_exists) should be worth 1 point, while
    hard judgment calls should be worth 3.

    This allows precise, checkable criteria like:
    - "Does the essay contain at least 3 cited sources?"
    - "Is there a section header for the methodology?"
    - "Does the report include a scatter plot?"
    """

    name: str  # Machine-readable identifier
    question: str  # Yes/no question for the grader to answer
    points: int = 3  # Max points for this category


# Union type for rubric entries
AnyRubricCategory = Union[RubricCategory, BinaryRubricCategory]


# =============================================================================
# SCORING HELPERS
# =============================================================================

# Level-name → fraction mappings (used by grader parsing)
SCORE_LEVEL_FRACTIONS: dict[str, float] = {
    "Failure": 0.0,
    "Minor Failure": 1 / 3,
    "Minor Success": 2 / 3,
    "Success": 1.0,
}

BINARY_LEVEL_FRACTIONS: dict[str, float] = {
    "Yes": 1.0,
    "No": 0.0,
}

# Legacy flat mappings (for backward compat with grader XML parsing)
SCORE_LEVELS = {
    "Failure": 0,
    "Minor Failure": 1,
    "Minor Success": 2,
    "Success": 3,
}

BINARY_SCORE_LEVELS = {
    "Yes": 3,
    "No": 0,
}


def score_for_level(cat: AnyRubricCategory, level: str) -> int:
    """Map a grader level string to a numeric score for this category.

    For graded categories, interpolates across the point range:
        Failure=0, Minor Failure≈1/3, Minor Success≈2/3, Success=points
    For binary categories: Yes=points, No=0.

    Falls back to 0 for unknown levels.
    """
    if isinstance(cat, BinaryRubricCategory):
        fracs = BINARY_LEVEL_FRACTIONS
    else:
        fracs = SCORE_LEVEL_FRACTIONS

    # Try exact match first, then case-insensitive
    frac = fracs.get(level)
    if frac is None:
        level_lower = level.lower()
        for valid_level, f in fracs.items():
            if valid_level.lower() == level_lower:
                frac = f
                break

    if frac is None:
        return 0

    return round(frac * cat.points)


def max_score(rubric: tuple[AnyRubricCategory, ...]) -> int:
    """Total max points across all categories in a rubric."""
    return sum(cat.points for cat in rubric)


def category_max(cat: AnyRubricCategory) -> int:
    """Max score for a single category."""
    return cat.points


@dataclass(frozen=True)
class RubricDatapoint:
    """A single problem for the rubric environment.

    Fields:
        problem_statement: The full task description shown to the model.
        rubric: Tuple of AnyRubricCategory objects (RubricCategory or
            BinaryRubricCategory) defining the grading criteria. Each
            category has a `points` field controlling its weight.
        submission_instructions: How the model should submit its work
            (e.g., "Write your essay to /testbed/essay.txt").
        available_tools: Tuple of ToolSpec objects from tools.py.
            Controls what the agent can do. Examples:
            - (BASH_TOOL, CREATE_FILE_TOOL, FINISH_TOOL) — full agent
            - (SUBMIT_SOLUTION_TOOL,) — effectively one-shot
        necessary_files: Dict mapping absolute file paths to file contents.
            These are uploaded to the container at startup via upload_file_command().
        problem_type: Category label for filtering/analysis.
    """

    problem_statement: str
    rubric: tuple[AnyRubricCategory, ...]
    submission_instructions: str
    available_tools: tuple[ToolSpec, ...]

    # Per-datapoint files to upload to container at startup
    necessary_files: dict[str, str] = field(default_factory=dict)

    # Metadata
    problem_type: str = "general"


# Keep MAX_SCORE_PER_CATEGORY for backward compatibility (used by old grader code)
MAX_SCORE_PER_CATEGORY = 3


# =============================================================================
# EXAMPLE DATAPOINTS
# =============================================================================


def make_persuasive_essay() -> RubricDatapoint:
    """Persuasive essay task — pure writing, no code."""
    return RubricDatapoint(
        problem_statement="""# Persuasive Essay: Public Libraries in the Digital Age

Write a persuasive essay (500-800 words) arguing that public libraries remain
essential institutions in the digital age.

Your essay should:
- Present a clear thesis in the opening paragraph
- Develop at least three distinct arguments supporting your position
- Acknowledge and rebut at least one counterargument
- Use specific examples or evidence (you may cite real or plausible examples)
- Conclude with a compelling closing statement

Write your essay to /testbed/essay.txt""",
        rubric=(
            RubricCategory(
                name="thesis_clarity",
                description="Does the essay present a clear, arguable thesis statement?",
                failure="No identifiable thesis, or the essay is off-topic.",
                minor_failure="Thesis is vague or buried deep in the essay.",
                minor_success="Thesis is present and identifiable but could be sharper or more specific.",
                success="Thesis is clear, specific, arguable, and prominently placed in the opening.",
            ),
            RubricCategory(
                name="argument_quality",
                description="Are the supporting arguments well-developed with reasoning and examples?",
                failure="No real arguments presented, or arguments are incoherent.",
                minor_failure="Arguments are present but shallow, lacking reasoning or examples.",
                minor_success="Arguments are reasonable and mostly supported, with minor gaps.",
                success="Three or more distinct, well-developed arguments with specific evidence or examples.",
            ),
            RubricCategory(
                name="counterargument_handling",
                description="Does the essay acknowledge and address opposing viewpoints?",
                failure="No mention of counterarguments whatsoever.",
                minor_failure="Counterargument is mentioned but dismissed without real engagement.",
                minor_success="Counterargument is acknowledged and partially rebutted.",
                success="At least one counterargument is fairly stated and convincingly rebutted.",
            ),
            RubricCategory(
                name="structure_and_flow",
                description="Is the essay well-organized with logical flow between paragraphs?",
                failure="No discernible structure; reads as a jumble of sentences.",
                minor_failure="Some structure but with unclear transitions or illogical ordering.",
                minor_success="Clear structure with introduction/body/conclusion, minor flow issues.",
                success="Well-organized with smooth transitions, logical progression, and clear intro/conclusion.",
            ),
            RubricCategory(
                name="grammar_and_style",
                description="Is the writing grammatically correct and stylistically appropriate?",
                failure="Pervasive errors that impede comprehension.",
                minor_failure="Frequent grammatical errors or consistently awkward phrasing.",
                minor_success="Generally clean writing with occasional errors or stylistic issues.",
                success="Polished, clear prose with no significant grammatical errors and appropriate tone.",
            ),
        ),
        submission_instructions="Write your completed essay to /testbed/essay.txt",
        available_tools=(tools.BASH_TOOL, tools.CREATE_FILE_TOOL, tools.FINISH_TOOL),
        problem_type="essay",
    )


def make_bash_golf() -> RubricDatapoint:
    """Bash golf task — reorganize files using minimal bash commands.

    The model is given a directory of files and must reorganize them
    according to specific rules using as few bash commands as possible.
    """
    # Files to seed into the container
    input_files = {
        "/testbed/input/report_2024_q1.txt": "Q1 2024 Financial Report\nRevenue: $1.2M\nProfit: $300K",
        "/testbed/input/report_2024_q2.txt": "Q2 2024 Financial Report\nRevenue: $1.5M\nProfit: $400K",
        "/testbed/input/report_2023_q4.txt": "Q4 2023 Financial Report\nRevenue: $1.1M\nProfit: $250K",
        "/testbed/input/notes_meeting_jan.md": "# January Meeting Notes\n- Discussed Q1 roadmap\n- Action items assigned",
        "/testbed/input/notes_meeting_feb.md": "# February Meeting Notes\n- Reviewed Q1 progress\n- Budget adjustments",
        "/testbed/input/notes_meeting_mar.md": "# March Meeting Notes\n- Q1 wrap-up\n- Q2 planning",
        "/testbed/input/logo.png.bak": "FAKE_PNG_DATA_FOR_TESTING",
        "/testbed/input/old_draft.txt.bak": "This is an old draft that should be deleted",
        "/testbed/input/README.md": "# Project Overview\nThis directory contains project files.",
    }

    return RubricDatapoint(
        problem_statement="""# Bash Golf: File Reorganization

You have a directory /testbed/input/ containing various files. Your task is to
reorganize them into /testbed/output/ according to these rules:

1. All .txt files containing "report" in the name go into /testbed/output/reports/
2. All .md files containing "notes" in the name go into /testbed/output/notes/
3. All .bak files should be deleted (not moved to output)
4. All remaining files go into /testbed/output/misc/
5. Files should be COPIED (not moved) — originals must remain in /testbed/input/

Write your solution as a bash script in /testbed/solution.sh, then execute it.

GOAL: Accomplish this in as few bash statements as possible while being correct.
An ideal solution uses ~5-8 statements. Fewer is better.""",
        rubric=(
            RubricCategory(
                name="correctness",
                description="Does the final filesystem state match all five rules?",
                failure="Output directory is missing or most files are in the wrong place.",
                minor_failure="Some files are misplaced or .bak files were not deleted.",
                minor_success="Most files are correct with one minor misplacement or missing copy.",
                success="All rules are satisfied: reports in reports/, notes in notes/, .bak deleted, rest in misc/, originals preserved.",
            ),
            RubricCategory(
                name="minimality",
                description="How few bash statements does the solution use?",
                failure="More than 15 statements, or solution is a brute-force file-by-file approach.",
                minor_failure="10-15 statements; uses some patterns but is verbose.",
                minor_success="6-9 statements; reasonable use of globs and patterns.",
                success="5 or fewer statements using elegant glob patterns, loops, or one-liners.",
            ),
            RubricCategory(
                name="no_side_effects",
                description="Does the solution avoid unintended side effects?",
                failure="Modifies or deletes files outside /testbed/input/ and /testbed/output/, or corrupts the filesystem.",
                minor_failure="Creates unnecessary temp files or leaves artifacts.",
                minor_success="Clean execution with one minor unnecessary artifact.",
                success="No side effects — only the specified directories are touched, no temp files, no extra artifacts.",
            ),
        ),
        submission_instructions="Write your solution to /testbed/solution.sh and execute it with: bash /testbed/solution.sh",
        available_tools=(tools.BASH_TOOL, tools.CREATE_FILE_TOOL, tools.FINISH_TOOL),
        necessary_files=input_files,
        problem_type="bash_golf",
    )


def make_data_analysis_report(rand_seed: int = 42) -> RubricDatapoint:
    """Data analysis task — analyze a CSV and write a report."""

    # Generate a deterministic CSV dataset
    rng = _random.Random(rand_seed)

    csv_lines = ["employee_id,department,salary,years_experience,performance_rating"]
    departments = ["Engineering", "Marketing", "Sales", "HR", "Finance"]
    for i in range(1, 201):
        dept = rng.choice(departments)
        years = rng.randint(1, 25)
        # Salary correlated with experience + department
        base = {
            "Engineering": 85000,
            "Marketing": 65000,
            "Sales": 60000,
            "HR": 62000,
            "Finance": 75000,
        }[dept]
        salary = base + years * 2500 + rng.randint(-10000, 10000)
        rating = min(5.0, max(1.0, round(0.5 + years * 0.12 + rng.gauss(0, 0.8), 1)))
        csv_lines.append(f"{i},{dept},{salary},{years},{rating}")

    csv_content = "\n".join(csv_lines) + "\n"

    return RubricDatapoint(
        problem_statement="""# Data Analysis: Employee Compensation Report

You have a dataset at /testbed/data.csv containing employee compensation data
with columns: employee_id, department, salary, years_experience, performance_rating.

Analyze this dataset and produce a written report at /testbed/report.txt that includes:

1. Summary statistics: total employees, salary range, mean/median salary
2. Department breakdown: average salary and headcount per department
3. A key finding about the relationship between experience and salary
4. A key finding about the relationship between experience and performance rating
5. One actionable recommendation based on your analysis

You may use Python, bash, or any tools available in the container to analyze the data.
The report should be written in clear, professional prose (not just raw numbers).""",
        rubric=(
            RubricCategory(
                name="accuracy",
                description="Are the reported statistics and findings factually correct given the data?",
                failure="Statistics are fabricated or grossly incorrect.",
                minor_failure="Some statistics are wrong or key numbers are significantly off.",
                minor_success="Most statistics are correct with minor rounding or calculation issues.",
                success="All reported numbers are accurate and verifiable from the dataset.",
            ),
            RubricCategory(
                name="statistical_reasoning",
                description="Does the analysis demonstrate sound statistical reasoning?",
                failure="No real analysis; just restates raw numbers without interpretation.",
                minor_failure="Some analysis attempted but with flawed reasoning or inappropriate methods.",
                minor_success="Reasonable analysis with appropriate methods, minor gaps in reasoning.",
                success="Sound statistical reasoning, appropriate use of summary statistics, meaningful correlations identified.",
            ),
            RubricCategory(
                name="presentation_clarity",
                description="Is the report well-written, well-organized, and easy to understand?",
                failure="Unreadable, just raw data dumps, or incoherent prose.",
                minor_failure="Readable but poorly organized or mixing prose with raw output.",
                minor_success="Clear and organized with minor formatting or flow issues.",
                success="Professional, well-structured report with clear sections, good prose, and appropriate use of numbers.",
            ),
            RubricCategory(
                name="completeness",
                description="Does the report address all five requested items?",
                failure="Addresses fewer than two of the five items.",
                minor_failure="Addresses two or three of the five items.",
                minor_success="Addresses four of the five items, or all five superficially.",
                success="Thoroughly addresses all five requested items.",
            ),
        ),
        submission_instructions="Write your completed report to /testbed/report.txt",
        available_tools=(tools.BASH_TOOL, tools.CREATE_FILE_TOOL, tools.EDIT_TOOL, tools.LIST_DIRECTORY_CONTENTS_TOOL, tools.FINISH_TOOL),
        necessary_files={"/testbed/data.csv": csv_content},
        problem_type="data_analysis",
    )


# =============================================================================
# EDITORIAL TASKS
# =============================================================================
# Inspired by GDPval Editors task (idx=50). Each factory produces one
# component of the editorial-writing workflow, with precise rubrics that
# are checkable but exploit the Goodhart gap: hitting every checkbox
# doesn't guarantee genuinely good work.
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
require €450 billion in infrastructure investment across the EU. Several
major automakers have announced they will challenge the regulation in
the European Court of Justice.
=== END NEWS EVENT ===

Your task: Write EXACTLY THREE different headline + standfirst pairs for
this story, each targeting a different editorial angle:

1. **Hard news angle** — Straight reporting, factual emphasis
2. **Economic/business angle** — Focus on industry and financial implications
3. **Opinion/editorial angle** — Taking a clear position (for or against)

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
                success="Standfirst is 1-2 crisp sentences mentioning the vote margin, 2035 date, and at least one other key fact (e.g. Germany's opposition, €450B cost)",
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
                minor_success="Mentions at least two of: €450B figure, specific company names (BMW/Mercedes/VW), Norway's 80% EV share",
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
                success="All three pairs are clearly distinct: one is neutral/factual, one is business-focused, one is opinionated — a reader could immediately tell which is which",
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
"The five-day work week is not a law of nature. It was an invention — Henry
Ford's invention, to be precise, adopted in 1926 not out of benevolence but
because exhausted workers were making costly mistakes on the assembly line.
Nearly a century later, the same calculus applies, yet we cling to the
five-day structure as though Moses brought it down from Sinai.

This month, a bill was introduced in Parliament to mandate a four-day,
32-hour work week for all companies with 50 or more employees, with no
reduction in pay. The predictable chorus of objections has begun. But the
evidence — from Iceland, from Spain, from Microsoft Japan, from dozens
of pilot programs — points overwhelmingly in one direction."
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
- Do NOT write an introduction or conclusion — only the core argument paragraphs""",
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
                failure="Tone is wildly different — academic, preachy, angry, or robotic",
                minor_failure="Tone is generally appropriate but lacks the intro's wit or reads as a dry policy paper",
                minor_success="Tone is close but occasionally slips into being overly formal, casual, or preachy",
                success="Tone seamlessly continues the intro: confident, slightly wry, treats the reader as intelligent, evidence-based without being dry",
                points=3,
            ),
            RubricCategory(
                name="rhetorical_cohesion_with_intro",
                description="Does the argument section read as a natural continuation of the provided intro (not as a standalone piece)?",
                failure="Section ignores the intro entirely — could be pasted into any essay",
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


def make_editorial_audience_adaptation() -> RubricDatapoint:
    """Adapt a piece of analysis for different audiences.

    Given a technical analysis paragraph, the model must rewrite it for
    three different audiences. Tests ability to shift register, vocabulary,
    and framing while preserving factual content.
    """
    source_text = """The implementation of congestion pricing in central Stockholm in 2006,
initially introduced as a seven-month trial, resulted in a 22% reduction
in traffic volume within the cordon zone during charging hours. Subsequent
analysis by Eliasson et al. (2009) demonstrated that the reduction persisted
at approximately 20% even after permanent implementation in 2007. The scheme
generated SEK 850 million (approximately €80 million) in annual net revenue
after accounting for infrastructure and operational costs. Critically, public
opinion shifted from 55% opposition before the trial to 53% support after
experiencing the reduced congestion, a phenomenon Eliasson attributes to
the "status quo bias" — citizens' tendency to prefer whatever system they
have direct experience with. Air quality monitoring stations within the
cordon recorded a 10-14% reduction in NO₂ levels and an 8.5% reduction
in PM10 particulate matter during the first year of operation."""

    return RubricDatapoint(
        problem_statement=f"""# Editorial Task: Audience Adaptation

You are an editor adapting content for different publications. Below is a
technical analysis paragraph about congestion pricing in Stockholm:

=== SOURCE PARAGRAPH ===
{source_text}
=== END SOURCE PARAGRAPH ===

Rewrite this paragraph for THREE different audiences. Each rewrite must
preserve the core factual content but adapt the language, framing, emphasis,
and level of detail for the target audience.

TARGET AUDIENCES:
1. **TABLOID** — Readers of a popular tabloid newspaper (reading age ~13,
   short attention span, needs a hook, uses everyday language, may use
   rhetorical questions). Target: 80-120 words.
2. **POLICY BRIEF** — Senior civil servants who need to make a decision.
   (Formal, action-oriented, focuses on outcomes and costs, minimal
   background needed). Target: 100-150 words.
3. **SOCIAL MEDIA** — A Twitter/X thread aimed at urbanist enthusiasts.
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
                name="tabloid_22pct_traffic_reduction",
                question="Does the [TABLOID] version mention the ~22% traffic reduction (or an equivalent like 'cut traffic by a fifth')?",
                points=2,
            ),
            BinaryRubricCategory(
                name="tabloid_no_academic_citations",
                question="Is the [TABLOID] version free of academic citations (no 'Eliasson et al.', no year-in-parentheses references)?",
                points=2,
            ),
            RubricCategory(
                name="tabloid_readability",
                description="Does the [TABLOID] version use simple, everyday language appropriate for a popular newspaper?",
                failure="Uses technical jargon ('cordon zone', 'particulate matter', 'status quo bias') throughout",
                minor_failure="Mostly simple but retains 2+ technical terms without explanation",
                minor_success="Simple language throughout with perhaps one slightly technical term",
                success="Fully accessible language, short sentences, possibly a hook or rhetorical question, reads like an actual tabloid article",
                points=3,
            ),
            BinaryRubricCategory(
                name="tabloid_opinion_shift_mentioned",
                question="Does the [TABLOID] version mention the public opinion shift (from opposition to support)?",
                points=2,
            ),
            # --- POLICY BRIEF VERSION ---
            BinaryRubricCategory(
                name="policy_brief_word_count",
                question="Is the [POLICY BRIEF] version between 100 and 150 words?",
                points=2,
            ),
            BinaryRubricCategory(
                name="policy_brief_revenue_figure",
                question="Does the [POLICY BRIEF] version include the revenue figure (SEK 850M or ~€80M)?",
                points=2,
            ),
            BinaryRubricCategory(
                name="policy_brief_air_quality_data",
                question="Does the [POLICY BRIEF] version include at least one air quality metric (NO₂ or PM10 reduction)?",
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
                success="Leads with a hook, emphasizes surprising facts (e.g., the opinion flip, or specific percentages), uses informal/punchy language appropriate for social media",
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
                failure="All three versions are in essentially the same register — you couldn't tell which audience they target",
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


def make_editorial_assembly() -> RubricDatapoint:
    """Assemble a complete editorial from component pieces.

    Given pre-written components (headline, sources, argument paragraphs),
    the model must assemble and edit them into a coherent, publication-ready
    editorial. Tests editing, transitions, and structural judgment.
    """
    # Pre-written components the model must work with
    headline_draft = """EU's ICE Vehicle Ban: A Necessary Disruption

The European Parliament's vote to ban new combustion engine cars by 2035
marks a turning point — not just for the auto industry, but for the
continent's willingness to match climate rhetoric with regulatory action."""

    argument_paragraphs = """The environmental case is straightforward, almost tediously so. Transport
accounts for roughly a quarter of the EU's greenhouse gas emissions, and
passenger cars represent the single largest source within that sector. The
International Energy Agency's 2023 Global EV Outlook projects that without
regulatory intervention, ICE vehicles sold before 2035 will continue
emitting CO₂ until 2050 or beyond. The ban doesn't eliminate these legacy
vehicles, but it stops the bleeding.

What makes this regulation genuinely interesting, however, is the economic
argument. Norway, which has used tax incentives rather than outright bans,
has achieved 80% EV market share and seen its automotive maintenance sector
shrink by 30% — while EV charging infrastructure has created 12,000 new
jobs. The EU's €450 billion infrastructure investment requirement sounds
alarming until you compare it with the €280 billion the bloc currently
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
Stuttgart and Munich. But the alternative — allowing German automakers
to keep building yesterday's technology while Chinese competitors like
BYD and NIO dominate the EV market — is a strategy for managed decline,
not preservation."""

    source_notes = """SOURCES USED:
- IEA Global EV Outlook 2023 — transport emissions data
- McKinsey "Power Play" report 2023 — economic transition projections
- Norwegian EV Association statistics — 80% market share, job creation
- European Automobile Manufacturers' Association — German employment figures
- EU Parliament voting record — 315-270 margin"""

    return RubricDatapoint(
        problem_statement="""# Editorial Task: Assembly and Final Edit

You are the commissioning editor assembling a complete editorial from
pre-written components. The components are saved as files in /testbed/components/:

- headline_draft.txt — Headline and standfirst (opening)
- argument_paragraphs.txt — Three core argument paragraphs
- counterarg_paragraph.txt — A paragraph addressing the opposition
- source_notes.txt — Source list used by the writers

Your task: Assemble these into a COMPLETE, PUBLICATION-READY editorial
and write it to /testbed/editorial.txt.

ASSEMBLY REQUIREMENTS:

1. **STRUCTURE**: The final editorial must have this structure:
   - Headline (on its own line)
   - Standfirst / opening paragraph
   - Core argument (the three paragraphs, possibly reordered)
   - Counterargument paragraph (placed where it's most effective)
   - A NEW closing paragraph that you write (80-120 words)
   - A byline line at the very end: "— Editorial Board"

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
                question="Does the editorial end with the byline '— Editorial Board' (or close variant) on its own line?",
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
                failure="No connection to the opening — closing could belong to any editorial",
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
                failure="Sections are simply concatenated with no transitions — reads like separate documents pasted together",
                minor_failure="Some transitions added but they're awkward or formulaic ('Moving on to...')",
                minor_success="Most transitions are smooth, with perhaps one jarring join",
                success="All transitions are smooth and natural — the editorial reads as if written by a single author in one sitting",
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
                failure="Reads like separate documents pasted together — tonal shifts, redundancy between sections, no narrative arc",
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


def make_editorial_fact_check() -> RubricDatapoint:
    """Editorial Task: Fact-check a draft editorial against reference documents.

    Given a draft editorial with deliberate errors AND reference documents
    containing verified facts, the model must cross-reference the draft
    against the references to identify inaccuracies. The verified facts
    are spread across multiple files — the model must read and synthesize
    them, not just reformat an answer key.
    """
    # Draft editorial with deliberate errors seeded throughout
    draft_with_errors = """EU's Bold Gamble: Why the 2035 ICE Ban Will Transform Europe

The European Parliament's decisive vote of 340-245 last month sent
shockwaves through the automotive industry. By banning the sale of all
new internal combustion engine vehicles by 2035 — the most aggressive
climate regulation in European history — the EU has committed to a
transformation that will reshape economies, supply chains, and daily
life across the continent.

The environmental imperative is clear. Transportation accounts for
approximately 40% of EU greenhouse gas emissions, with passenger vehicles
representing the majority of that share. According to the International
Energy Agency's 2023 Global EV Outlook, every year of delay in phasing
out ICE vehicles adds roughly 800 million tonnes of cumulative CO₂
emissions by 2050. The science is settled; the question is whether
politics can keep pace.

Norway offers the most compelling preview of this future. Having
implemented an outright ban on ICE sales in 2022, the Nordic nation now
boasts a 95% electric vehicle market share — the highest in the world.
The transition has created 25,000 new jobs in EV charging infrastructure
while reducing urban air pollution in Oslo by 35%. Critics who predicted
economic catastrophe have been silenced by Norway's 2.1% GDP growth
in 2023, outpacing the EU average.

The economic case deserves scrutiny, however. The EU's own impact
assessment estimates the transition will require €650 billion in
infrastructure investment by 2035, a figure that has drawn howls from
industry. Germany, whose automotive sector employs 1.2 million people
directly, voted unanimously against the measure. BMW's CEO Oliver Zipse
called it "an own goal of historic proportions" at the Munich Motor Show.

Yet the costs of inaction dwarf the costs of transition. The EU
currently spends €280 billion annually on imported crude oil — money
that flows predominantly to Russia, Saudi Arabia, and other
petrochemical states. McKinsey's 2023 "Power Play" report estimates
that by 2040, the EV transition will generate a net economic surplus
of €180 billion per year across the EU, driven by reduced fuel imports,
lower maintenance costs, and a booming European battery industry led
by Sweden's Northvolt, which recently opened Europe's largest
gigafactory in Skellefteå.

The ban is not without genuine risks. China's BYD and NIO currently
hold a 62% global market share in EV manufacturing, and Europe's
automakers have been slow to catch up. Without aggressive industrial
policy to complement the sales ban, Europe risks trading dependence
on Middle Eastern oil for dependence on Chinese batteries. The EU's
proposed €3.2 billion Battery Alliance fund is a start, but it pales
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

— Editorial Board"""

    # Reference documents with verified facts — spread across 3 files
    eu_parliament_record = """EUROPEAN PARLIAMENT — OFFICIAL RECORD
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
The German delegation was split, with several MEPs voting in favor."""

    transport_emissions_brief = """EUROPEAN ENVIRONMENT AGENCY — TRANSPORT EMISSIONS BRIEF

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
     setting binding renewable energy targets. This is correct."""

    market_analysis = """MARKET ANALYSIS — EV INDUSTRY COMPETITIVE LANDSCAPE

DISPUTED/UNVERIFIABLE CLAIMS TO NOTE:

1. BYD/NIO MARKET SHARE
   The claim that "BYD and NIO hold 62% global market share in EV
   manufacturing" is NOT supported by available data. As of 2023:
   - BYD held approximately 15-20% of global BEV sales
   - NIO held approximately 1-2% of global BEV sales
   - Combined, they are far below 62%
   - The 62% figure appears to be fabricated or conflated with
     China's overall share of global EV production (~60%)

2. The €280 billion figure for annual EU crude oil imports, the McKinsey
   "Power Play" report estimates, and the Norway jobs/air quality figures
   are not independently verifiable from our reference materials.
   These should be flagged as UNSUPPORTED rather than incorrect.

3. The €650 billion infrastructure investment figure differs from the
   €450 billion commonly cited. The source of this figure should be
   verified."""

    return RubricDatapoint(
        problem_statement="""# Editorial Task: Fact-Check Report

You are a fact-checker at a national newspaper. A draft editorial has been
submitted for publication. Your job is to produce a FACT-CHECK REPORT
identifying ALL factual errors, unsupported claims, and misleading framings.

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
3. A separate section listing any UNSUPPORTED CLAIMS — statements presented
   as fact that may or may not be true but are not verifiable from the
   reference documents
4. A final VERDICT: "PUBLISHABLE WITH CORRECTIONS", "NEEDS MAJOR REVISION",
   or "UNPUBLISHABLE" with a 2-3 sentence justification

DO NOT rewrite the editorial. Only produce the fact-check report.""",
        rubric=(
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
            # --- ERROR DETECTION: Did it find the deliberate errors? ---
            BinaryRubricCategory(
                name="catches_vote_margin",
                question="Does the report identify that the vote margin '340-245' is incorrect (should be 315-270)?",
                points=3,
            ),
            BinaryRubricCategory(
                name="catches_emissions_percentage",
                question="Does the report identify that '40%' for transport emissions share is incorrect (should be ~25%)?",
                points=3,
            ),
            BinaryRubricCategory(
                name="catches_norway_ban_claim",
                question="Does the report identify that Norway has NOT implemented an 'outright ban' on ICE sales (it uses tax incentives)?",
                points=3,
            ),
            BinaryRubricCategory(
                name="catches_norway_market_share",
                question="Does the report identify that Norway's EV market share is ~80%, not 95%?",
                points=3,
            ),
            BinaryRubricCategory(
                name="catches_germany_employment",
                question="Does the report identify that Germany's automotive employment is ~800,000, not 1.2 million?",
                points=3,
            ),
            BinaryRubricCategory(
                name="catches_china_market_share",
                question="Does the report identify that the '62% global market share' claim for BYD/NIO is fabricated?",
                points=3,
            ),
            # --- ERROR DETECTION: False positives ---
            BinaryRubricCategory(
                name="does_not_flag_california_1975",
                question="Does the report correctly NOT flag the California catalytic converter mandate date (1975) as an error?",
                points=2,
            ),
            BinaryRubricCategory(
                name="does_not_flag_eu_2009_directive",
                question="Does the report correctly NOT flag the 2009 EU renewable energy directive as an error?",
                points=2,
            ),
            # --- SEVERITY RATINGS ---
            RubricCategory(
                name="severity_ratings_sensible",
                description="Are the severity ratings (CRITICAL/MODERATE/MINOR) logically applied?",
                failure="Severity ratings are random or inverted (e.g., vote margin error rated MINOR, a typo rated CRITICAL)",
                minor_failure="Most ratings sensible but 2+ are questionable",
                minor_success="Ratings are mostly sensible with one debatable assignment",
                success="All severity ratings are well-justified and logically consistent — errors that undermine core arguments are CRITICAL, imprecise numbers are MODERATE or MINOR",
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
                minor_failure="Lists 1-2 items but misses obvious unverifiable claims (e.g., the Norway jobs figure, the McKinsey estimate specifics)",
                minor_success="Identifies several genuinely unverifiable claims",
                success="Identifies multiple unverifiable claims and correctly distinguishes them from verifiable errors — e.g., the 25,000 jobs figure, the Oslo air pollution reduction, the Northvolt gigafactory claim",
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
                failure="Verdict contradicts the findings (e.g., 'PUBLISHABLE' despite 6+ errors including critical ones)",
                minor_failure="Verdict is defensible but overly lenient or harsh",
                minor_success="Verdict is reasonable and somewhat justified",
                success="Verdict correctly reflects the pattern of errors: given 5+ errors including critical factual mistakes, 'NEEDS MAJOR REVISION' or 'UNPUBLISHABLE' is appropriate, with clear reasoning",
                points=3,
            ),
        ),
        submission_instructions="Write your fact-check report to /testbed/fact_check.txt",
        available_tools=(tools.BASH_TOOL, tools.CREATE_FILE_TOOL, tools.LIST_DIRECTORY_CONTENTS_TOOL, tools.FINISH_TOOL),
        necessary_files={
            "/testbed/draft_editorial.txt": draft_with_errors,
            "/testbed/reference/eu_parliament_record.txt": eu_parliament_record,
            "/testbed/reference/transport_emissions_brief.txt": transport_emissions_brief,
            "/testbed/reference/market_analysis.txt": market_analysis,
        },
        problem_type="editorial",
    )


# =============================================================================
# GDPVAL-INSPIRED TASKS
# =============================================================================
# Decomposed, factory-based tasks inspired by real-world professional work.
# Each factory is parameterized so proliferation is just calling with
# different data. Each task uses necessary_files heavily.
# =============================================================================


def make_qa_escalation_email(
    material_name: str = "Antifoam Agent AF-200",
    vendor_name: str = "ChemSupply International",
    rms_id: str = "RMS-2024-0847",
    rms_spec: str = "Endotoxin level ≤ 0.25 EU/mL per USP <85>",
    coa_value: str = "Endotoxin: 0.41 EU/mL (LAL kinetic turbidimetric method)",
    lot_number: str = "CSI-AF200-LOT2024-0312",
    recipient_name: str = "Dr. Sarah Chen",
    recipient_title: str = "Director of Quality Assurance",
) -> RubricDatapoint:
    """QA Escalation Email: Write an escalation email about a spec discrepancy.

    Factory: varying material, vendor, spec values, and recipients creates
    new problems with the same rubric structure.
    """
    internal_spec = f"""RAW MATERIAL SPECIFICATION — {rms_id}
Material: {material_name}
Approved Vendor: {vendor_name}
Document Version: 3.2 (Effective Date: 2024-01-15)

CRITICAL QUALITY ATTRIBUTES:
1. Appearance: Clear to slightly hazy liquid, colorless to pale yellow
2. pH: 6.5 – 7.5 (1% aqueous solution)
3. Endotoxin: {rms_spec}
4. Bioburden: ≤ 10 CFU/mL
5. Specific Gravity: 1.01 – 1.04 at 25°C

STORAGE: 2–8°C, protect from light
RETEST PERIOD: 24 months from date of manufacture

APPROVED BY: Quality Assurance, Regulatory Affairs
CHANGE HISTORY:
  v3.0 (2023-06): Initial specification
  v3.1 (2023-09): Tightened endotoxin limit from 0.50 to 0.25 EU/mL
  v3.2 (2024-01): Added bioburden specification
"""

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

* Note: Result meets vendor specification (≤ 0.50 EU/mL).

QC Analyst: J. Martinez
QC Manager: R. Patel
Date of Analysis: 2024-03-01
"""

    contacts = f"""QA TEAM CONTACTS — Escalation Directory

{recipient_name}, {recipient_title}
  Email: s.chen@company.com
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
                question="Does the email state the actual COA test result (0.41 EU/mL)?",
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
                question="Does the email note that the vendor's own spec (≤ 0.50 EU/mL) differs from the internal spec (≤ 0.25 EU/mL)?",
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
                success="Clear, numbered action items with specific owners or deadlines (e.g., 'Please initiate a formal deviation per SOP-XXX by EOD Friday')",
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


def make_qa_risk_assessment(
    material_name: str = "Antifoam Agent AF-200",
    vendor_name: str = "ChemSupply International",
    departed_employee: str = "Rachel Nguyen",
    departed_role: str = "Senior QA Specialist — Vendor Management",
    departure_date: str = "2024-01-15",
    notification_date: str = "2024-02-01",
    change_description: str = "reformulation of silicone defoamer base from Type III to Type IV polymer",
) -> RubricDatapoint:
    """QA Risk Assessment: Identify process gap from missed vendor notification.

    Factory: varying the employee, vendor, change type creates new problems.
    """
    vendor_notification = f"""VENDOR CHANGE NOTIFICATION
From: {vendor_name} — Regulatory & Quality Affairs
To: {departed_employee}, {departed_role}
Date: {notification_date}
Reference: VCN-2024-0156

Subject: Product Change Notification — {material_name}

Dear {departed_employee},

This letter serves as formal notification per ICH Q7 Section 12.1
requirements that {vendor_name} will implement the following change
to {material_name}:

CHANGE DESCRIPTION:
  {change_description}

EFFECTIVE DATE: 2024-04-01 (Lot CSI-AF200-LOT2024-0400 onwards)

RATIONALE:
  Improved consistency in defoaming performance. No change to
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
  Email: m.santos@chemsupply.com
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
    NOTE: {departed_employee} was primary contact for {vendor_name},
    BioReagent Corp, and PharmaGrade Solutions. These vendor
    relationships have not been formally reassigned.
"""

    internal_spec = f"""RAW MATERIAL SPECIFICATION — RMS-2024-0847
Material: {material_name}
Approved Vendor: {vendor_name}

CRITICAL QUALITY ATTRIBUTES:
1. Appearance: Clear to slightly hazy liquid, colorless to pale yellow
2. pH: 6.5 – 7.5 (1% aqueous solution)
3. Endotoxin: ≤ 0.25 EU/mL per USP <85>
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
                question=f"Does the document describe the vendor's change ({change_description.split('from')[0].strip()})?",
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
                success="Specific, actionable mitigations with suggested owners and/or timelines (e.g., 'IT to implement shared vendor inbox by Q2', 'HR to update offboarding checklist to include vendor contact transfer')",
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


def make_utilization_report(rand_seed: int = 99) -> RubricDatapoint:
    """Employee utilization report from timekeeping data.

    Factory: varying employees, projects, and hours creates new problems
    with deterministically different correct answers.
    """
    rng = _random.Random(rand_seed)

    # Employee roster
    employees = [
        ("E001", "Alice Martin", "Engineering", "Senior Developer", "FT", 160),
        ("E002", "Bob Chen", "Engineering", "Developer", "FT", 160),
        ("E003", "Carla Diaz", "Engineering", "Junior Developer", "PT", 80),
        ("E004", "Dan Foster", "Marketing", "Campaign Manager", "FT", 160),
        ("E005", "Eva Green", "Marketing", "Content Specialist", "FT", 160),
        ("E006", "Frank Hall", "Marketing", "Designer", "PT", 80),
        ("E007", "Grace Ito", "Sales", "Account Executive", "FT", 160),
        ("E008", "Henry Jain", "Sales", "Sales Rep", "FT", 160),
        ("E009", "Irene Kim", "Operations", "Ops Manager", "FT", 160),
        ("E010", "Jack Lee", "Operations", "Logistics Coord", "FT", 160),
        ("E011", "Kara Moss", "Operations", "Warehouse Lead", "PT", 80),
        ("E012", "Leo Nash", "Finance", "Controller", "FT", 160),
        ("E013", "Mia Owens", "Finance", "Accountant", "FT", 160),
        ("E014", "Nick Park", "HR", "HR Manager", "FT", 160),
        ("E015", "Olivia Quinn", "HR", "Recruiter", "PT", 80),
    ]

    projects = [
        ("P100", "Website Redesign", 800),
        ("P200", "Q1 Marketing Campaign", 500),
        ("P300", "Client Onboarding System", 600),
        ("P400", "Annual Audit Prep", 300),
        ("P500", "Warehouse Automation", 450),
        ("P600", "General & Administrative", 9999),  # overhead
    ]

    # Generate timekeeping data for March 2024 (20 working days)
    timekeeping_lines = ["employee_id,date,project_code,hours"]
    for day in range(1, 21):
        date = f"2024-03-{day:02d}"
        for emp_id, name, dept, role, ft_pt, capacity in employees:
            daily_hours = capacity / 20  # FT=8, PT=4
            # Split hours across 1-2 projects
            if rng.random() < 0.7:
                # Single project
                proj = rng.choice([p[0] for p in projects[:5]])
                hrs = round(daily_hours + rng.gauss(0, 0.5), 1)
                hrs = max(0, hrs)
                timekeeping_lines.append(f"{emp_id},{date},{proj},{hrs}")
            else:
                # Two projects
                proj1 = rng.choice([p[0] for p in projects[:5]])
                proj2 = rng.choice([p[0] for p in projects[:5]])
                hrs1 = round(daily_hours * 0.6 + rng.gauss(0, 0.3), 1)
                hrs2 = round(daily_hours * 0.4 + rng.gauss(0, 0.3), 1)
                hrs1 = max(0, hrs1)
                hrs2 = max(0, hrs2)
                timekeeping_lines.append(f"{emp_id},{date},{proj1},{hrs1}")
                if hrs2 > 0:
                    timekeeping_lines.append(f"{emp_id},{date},{proj2},{hrs2}")

    timekeeping_csv = "\n".join(timekeeping_lines) + "\n"

    roster_lines = ["employee_id,name,department,role,employment_type,monthly_capacity_hours"]
    for emp in employees:
        roster_lines.append(",".join(str(x) for x in emp))
    roster_csv = "\n".join(roster_lines) + "\n"

    budget_lines = ["project_code,project_name,budgeted_hours_q1"]
    for proj in projects:
        budget_lines.append(f"{proj[0]},{proj[1]},{proj[2]}")
    budget_csv = "\n".join(budget_lines) + "\n"

    return RubricDatapoint(
        problem_statement="""# Employee Utilization Report

You are an HR analyst. Using the data files in /testbed/data/, produce
a utilization report for March 2024.

Files available:
- /testbed/data/timekeeping.csv — Daily time entries (employee_id, date, project_code, hours)
- /testbed/data/roster.csv — Employee roster with monthly capacity hours
- /testbed/data/project_budgets.csv — Q1 project budgets

Write a report to /testbed/report.txt that includes:

1. EMPLOYEE UTILIZATION: For each employee, compute total hours worked
   in March and utilization rate (hours_worked / capacity_hours × 100%).
   Flag anyone below 60% (underutilized) or above 110% (overutilized).

2. DEPARTMENT SUMMARY: Average utilization per department.

3. PROJECT HOURS SUMMARY: Total hours charged to each project in March.
   Compare against Q1 budget (note: March is month 3 of 3 in Q1).

4. AT-RISK FLAGS: List specific employees and projects that need attention.

5. RECOMMENDATIONS: 1-2 actionable recommendations.

You may write and run Python scripts to analyze the data.""",
        rubric=(
            BinaryRubricCategory(
                name="file_exists",
                question="Does /testbed/report.txt exist with substantial content?",
                points=1,
            ),
            BinaryRubricCategory(
                name="lists_all_15_employees",
                question="Does the report list utilization data for all 15 employees?",
                points=2,
            ),
            BinaryRubricCategory(
                name="states_capacity_baseline",
                question="Does the report state the capacity baseline (160 hours/month for FT, 80 hours/month for PT)?",
                points=2,
            ),
            BinaryRubricCategory(
                name="computes_utilization_rates",
                question="Does the report show utilization rates as percentages for each employee?",
                points=2,
            ),
            BinaryRubricCategory(
                name="flags_underutilized",
                question="Does the report flag at least one employee as underutilized (below 60%)?",
                points=3,
            ),
            BinaryRubricCategory(
                name="flags_overutilized",
                question="Does the report flag at least one employee as overutilized (above 110%)?",
                points=3,
            ),
            BinaryRubricCategory(
                name="department_summary_present",
                question="Does the report include a department-level average utilization summary?",
                points=2,
            ),
            BinaryRubricCategory(
                name="project_hours_present",
                question="Does the report show total hours charged per project in March?",
                points=2,
            ),
            BinaryRubricCategory(
                name="project_budget_comparison",
                question="Does the report compare project hours against budget (considering March is month 3 of Q1)?",
                points=3,
            ),
            BinaryRubricCategory(
                name="has_recommendations",
                question="Does the report include at least one actionable recommendation?",
                points=2,
            ),
            RubricCategory(
                name="numerical_accuracy",
                description="Are the computed numbers (hours, percentages) accurate based on the raw data?",
                failure="Numbers are fabricated or wildly incorrect (off by >20%)",
                minor_failure="Some numbers are close but several are significantly wrong",
                minor_success="Most numbers are accurate with minor rounding differences",
                success="All reported numbers match the raw data within reasonable rounding",
                points=3,
            ),
            RubricCategory(
                name="analysis_quality",
                description="Does the report go beyond raw numbers to provide insight?",
                failure="Just dumps raw numbers with no interpretation",
                minor_failure="Some interpretation but mostly restates the data",
                minor_success="Good interpretation of trends with minor gaps",
                success="Identifies patterns (e.g., department-level trends, project staffing imbalances) and connects them to actionable insights",
                points=3,
            ),
            RubricCategory(
                name="formatting_quality",
                description="Is the report well-formatted and easy to scan?",
                failure="Raw data dump, no structure",
                minor_failure="Some structure but hard to navigate",
                minor_success="Clear sections and mostly readable",
                success="Well-structured with clear headers, aligned tables or lists, and easy-to-scan formatting",
                points=2,
            ),
        ),
        submission_instructions="Write your utilization report to /testbed/report.txt",
        available_tools=(tools.BASH_TOOL, tools.CREATE_FILE_TOOL, tools.EDIT_TOOL, tools.LIST_DIRECTORY_CONTENTS_TOOL, tools.FINISH_TOOL),
        necessary_files={
            "/testbed/data/timekeeping.csv": timekeeping_csv,
            "/testbed/data/roster.csv": roster_csv,
            "/testbed/data/project_budgets.csv": budget_csv,
        },
        problem_type="data_analysis",
    )


def make_incident_root_cause(
    service_name: str = "payment-gateway",
    error_type: str = "connection pool exhaustion",
    root_cause: str = "database connection pool max_connections=20 insufficient for traffic spike",
    rand_seed: int = 77,
) -> RubricDatapoint:
    """Incident Root Cause Analysis from logs and metrics.

    Factory: varying the service, error type, and root cause creates
    new problems with different log patterns.
    """
    rng = _random.Random(rand_seed)

    # Generate application log (~150 lines)
    app_log_lines = []
    for minute in range(0, 90):
        ts = f"2024-03-15T14:{minute // 60:02d}:{minute % 60:02d}.000Z"
        if minute < 30:
            # Normal operation
            if rng.random() < 0.3:
                app_log_lines.append(f"{ts} INFO  [{service_name}] Processed payment txn_{rng.randint(10000,99999)} in {rng.randint(50,200)}ms")
            if rng.random() < 0.1:
                app_log_lines.append(f"{ts} DEBUG [{service_name}] Connection pool: active=8/20, idle=12")
        elif minute < 45:
            # Traffic ramp-up, pool filling
            if rng.random() < 0.5:
                active = min(20, 8 + (minute - 30) * 1)
                app_log_lines.append(f"{ts} INFO  [{service_name}] Processed payment txn_{rng.randint(10000,99999)} in {rng.randint(150, 500)}ms")
            if rng.random() < 0.3:
                active = min(20, 8 + (minute - 30))
                app_log_lines.append(f"{ts} WARN  [{service_name}] Connection pool: active={active}/20, idle={20 - active}")
            if minute == 40:
                app_log_lines.append(f"{ts} WARN  [{service_name}] Connection pool utilization >90%: active=19/20")
        elif minute < 60:
            # Pool exhausted — errors begin
            if rng.random() < 0.6:
                app_log_lines.append(f"{ts} ERROR [{service_name}] Failed to acquire connection: pool exhausted (active=20/20, wait_timeout=5000ms)")
            if rng.random() < 0.4:
                app_log_lines.append(f"{ts} ERROR [{service_name}] Payment processing failed for txn_{rng.randint(10000,99999)}: ConnectionPoolTimeoutException")
            if rng.random() < 0.2:
                app_log_lines.append(f"{ts} WARN  [{service_name}] Circuit breaker OPEN for database-primary after 10 consecutive failures")
            # Red herring: occasional GC pause
            if minute == 50:
                app_log_lines.append(f"{ts} WARN  [jvm] GC pause: 230ms (young generation)")
        else:
            # Recovery after ops increases pool size
            if minute == 60:
                app_log_lines.append(f"{ts} INFO  [{service_name}] Configuration updated: max_connections changed from 20 to 50")
                app_log_lines.append(f"{ts} INFO  [{service_name}] Connection pool reinitialized")
            if rng.random() < 0.3:
                app_log_lines.append(f"{ts} INFO  [{service_name}] Processed payment txn_{rng.randint(10000,99999)} in {rng.randint(60,250)}ms")
            if rng.random() < 0.1:
                app_log_lines.append(f"{ts} DEBUG [{service_name}] Connection pool: active={rng.randint(8,25)}/50, idle={rng.randint(20,40)}")

    app_log = "\n".join(app_log_lines) + "\n"

    # Load balancer log
    lb_log_lines = []
    for minute in range(0, 90):
        ts = f"2024-03-15T14:{minute // 60:02d}:{minute % 60:02d}.000Z"
        if minute < 45:
            if rng.random() < 0.2:
                lb_log_lines.append(f"{ts} haproxy: {service_name}/server1 200 {rng.randint(50,200)}ms")
        elif minute < 60:
            if rng.random() < 0.5:
                lb_log_lines.append(f"{ts} haproxy: {service_name}/server1 502 0ms (backend timeout)")
            if rng.random() < 0.3:
                lb_log_lines.append(f"{ts} haproxy: {service_name} health check FAILED")
        else:
            if rng.random() < 0.2:
                lb_log_lines.append(f"{ts} haproxy: {service_name}/server1 200 {rng.randint(60,250)}ms")

    lb_log = "\n".join(lb_log_lines) + "\n"

    # Metrics CSV
    metrics_lines = ["timestamp,latency_p99_ms,error_rate_pct,cpu_pct,memory_pct,db_conn_active,db_conn_max"]
    for minute in range(0, 90):
        ts = f"2024-03-15T14:{minute // 60:02d}:{minute % 60:02d}"
        if minute < 30:
            lat = rng.randint(80, 150)
            err = round(rng.random() * 0.5, 1)
            cpu = rng.randint(25, 40)
            mem = rng.randint(55, 65)
            conn = rng.randint(5, 12)
            conn_max = 20
        elif minute < 45:
            lat = rng.randint(200, 600)
            err = round(rng.random() * 5, 1)
            cpu = rng.randint(45, 70)
            mem = rng.randint(60, 72)
            conn = min(20, 12 + (minute - 30))
            conn_max = 20
        elif minute < 60:
            lat = rng.randint(5000, 10000)
            err = round(30 + rng.random() * 40, 1)
            cpu = rng.randint(30, 50)  # CPU NOT the issue
            mem = rng.randint(65, 75)  # Memory NOT the issue
            conn = 20
            conn_max = 20
        else:
            lat = rng.randint(80, 200)
            err = round(rng.random() * 1, 1)
            cpu = rng.randint(30, 50)
            mem = rng.randint(60, 70)
            conn = rng.randint(10, 30)
            conn_max = 50
        metrics_lines.append(f"{ts},{lat},{err},{cpu},{mem},{conn},{conn_max}")

    metrics_csv = "\n".join(metrics_lines) + "\n"

    return RubricDatapoint(
        problem_statement=f"""# Incident Root Cause Analysis

A production incident occurred on 2024-03-15 affecting the {service_name}
service. The incident lasted approximately 15 minutes, during which payment
processing failed for a significant number of transactions.

You have access to:
- /testbed/logs/application.log — Application-level logs
- /testbed/logs/loadbalancer.log — Load balancer access logs
- /testbed/monitoring/metrics.csv — Time-series metrics (1-min intervals)

Write a root cause analysis to /testbed/rca.txt that includes:

1. TIMELINE: Key events with timestamps
2. ROOT CAUSE: What specifically caused the failure
3. CONTRIBUTING FACTORS: What made it worse or delayed recovery
4. EVIDENCE: Specific log lines and metric values that support your diagnosis
5. RECOMMENDATIONS: How to prevent recurrence

Analyze the data carefully — distinguish root cause from symptoms.""",
        rubric=(
            BinaryRubricCategory(
                name="file_exists",
                question="Does /testbed/rca.txt exist with substantial content?",
                points=1,
            ),
            BinaryRubricCategory(
                name="identifies_root_cause",
                question="Does the analysis correctly identify connection pool exhaustion as the root cause (not CPU, memory, or GC)?",
                points=3,
            ),
            BinaryRubricCategory(
                name="mentions_pool_size",
                question="Does the analysis mention the specific pool size (max_connections=20) as being insufficient?",
                points=3,
            ),
            BinaryRubricCategory(
                name="timeline_includes_onset",
                question="Does the timeline include the approximate onset of errors (around minute 45, 14:45)?",
                points=2,
            ),
            BinaryRubricCategory(
                name="timeline_includes_resolution",
                question="Does the timeline include the resolution event (pool size increased to 50 at around 15:00)?",
                points=2,
            ),
            BinaryRubricCategory(
                name="timeline_includes_warning",
                question="Does the timeline include the early warning sign (pool utilization >90% at around 14:40)?",
                points=2,
            ),
            BinaryRubricCategory(
                name="cites_specific_error_message",
                question="Does the analysis cite a specific error message from the logs (e.g., 'ConnectionPoolTimeoutException' or 'pool exhausted')?",
                points=2,
            ),
            BinaryRubricCategory(
                name="cites_metric_value",
                question="Does the analysis cite at least one specific metric value (e.g., error rate %, latency p99 during incident)?",
                points=2,
            ),
            BinaryRubricCategory(
                name="does_not_blame_cpu",
                question="Does the analysis correctly NOT blame CPU utilization (which stayed moderate throughout)?",
                points=2,
            ),
            BinaryRubricCategory(
                name="does_not_blame_gc",
                question="Does the analysis correctly identify the GC pause as a red herring (not the root cause)?",
                points=2,
            ),
            RubricCategory(
                name="cause_vs_symptom_distinction",
                description="Does the analysis distinguish the root cause from symptoms?",
                failure="Conflates symptoms (502 errors, high latency) with the root cause",
                minor_failure="Mentions both but doesn't clearly separate them",
                minor_success="Clearly identifies symptoms vs cause but explanation could be sharper",
                success="Crisp distinction: pool exhaustion is the cause, 502s/timeouts/circuit breaker trips are symptoms, traffic spike is the trigger",
                points=3,
            ),
            RubricCategory(
                name="contributing_factors_quality",
                description="Does the analysis identify contributing factors beyond the immediate cause?",
                failure="No contributing factors mentioned",
                minor_failure="Mentions one contributing factor vaguely",
                minor_success="Identifies 1-2 contributing factors (e.g., no pool size alerting, static config)",
                success="Identifies multiple contributing factors: no alerting on pool utilization, static pool sizing, lack of auto-scaling, no load testing at scale",
                points=3,
            ),
            RubricCategory(
                name="recommendations_quality",
                description="Are the recommendations specific and actionable?",
                failure="No recommendations or only 'fix it'",
                minor_failure="Generic recommendations ('add monitoring')",
                minor_success="Specific recommendations but missing some obvious ones",
                success="Specific, prioritized recommendations (e.g., increase pool size, add alerting at 80% utilization, implement connection pool auto-scaling, load test at 2x peak)",
                points=3,
            ),
        ),
        submission_instructions="Write your root cause analysis to /testbed/rca.txt",
        available_tools=(tools.BASH_TOOL, tools.CREATE_FILE_TOOL, tools.EDIT_TOOL, tools.LIST_DIRECTORY_CONTENTS_TOOL, tools.FINISH_TOOL),
        necessary_files={
            "/testbed/logs/application.log": app_log,
            "/testbed/logs/loadbalancer.log": lb_log,
            "/testbed/monitoring/metrics.csv": metrics_csv,
        },
        problem_type="incident_analysis",
    )


def make_sales_yoy_analysis(
    brand_name: str = "Northridge Outdoor Co.",
    rand_seed: int = 55,
) -> RubricDatapoint:
    """Sales Year-over-Year analysis from CSV data.

    Factory: varying brand, product lines, and sales numbers creates new
    problems with deterministically different correct answers.
    """
    rng = _random.Random(rand_seed)

    product_lines = ["Hiking Boots", "Rain Jackets", "Backpacks", "Camping Tents", "Sleeping Bags"]

    # Generate sales data for 2022 and 2023
    def gen_sales(year, rng):
        lines = ["product_line,month,units,revenue"]
        for pl in product_lines:
            base_units = {"Hiking Boots": 400, "Rain Jackets": 300, "Backpacks": 350, "Camping Tents": 200, "Sleeping Bags": 150}[pl]
            unit_price = {"Hiking Boots": 129, "Rain Jackets": 89, "Backpacks": 79, "Camping Tents": 249, "Sleeping Bags": 119}[pl]
            seasonal = {"Hiking Boots": [0.6, 0.7, 1.0, 1.2, 1.5, 1.4, 1.3, 1.2, 1.1, 0.9, 0.7, 0.8],
                        "Rain Jackets": [0.8, 0.9, 1.2, 1.4, 1.0, 0.8, 0.7, 0.8, 1.1, 1.3, 1.2, 0.9],
                        "Backpacks": [0.7, 0.8, 1.0, 1.1, 1.3, 1.5, 1.4, 1.3, 1.0, 0.8, 0.7, 0.9],
                        "Camping Tents": [0.4, 0.5, 0.8, 1.2, 1.6, 1.8, 1.7, 1.5, 1.0, 0.6, 0.3, 0.3],
                        "Sleeping Bags": [0.5, 0.6, 0.9, 1.2, 1.4, 1.5, 1.4, 1.3, 1.1, 0.8, 0.6, 0.7]}[pl]
            growth = 1.0 if year == 2022 else {"Hiking Boots": 1.15, "Rain Jackets": 1.08, "Backpacks": 0.92, "Camping Tents": 1.22, "Sleeping Bags": 0.85}[pl]
            for month in range(1, 13):
                units = int(base_units * seasonal[month-1] * growth + rng.randint(-30, 30))
                units = max(10, units)
                rev = units * unit_price + rng.randint(-500, 500)
                lines.append(f"{pl},{year}-{month:02d},{units},{rev}")
        return "\n".join(lines) + "\n"

    sales_2022 = gen_sales(2022, rng)
    sales_2023 = gen_sales(2023, rng)

    # Inventory
    inv_lines = ["product_line,on_hand_units,on_order_units,avg_monthly_demand_units"]
    for pl in product_lines:
        on_hand = rng.randint(200, 800)
        on_order = rng.randint(100, 500)
        avg_demand = {"Hiking Boots": 450, "Rain Jackets": 320, "Backpacks": 300, "Camping Tents": 280, "Sleeping Bags": 140}[pl]
        inv_lines.append(f"{pl},{on_hand},{on_order},{avg_demand}")
    inventory_csv = "\n".join(inv_lines) + "\n"

    return RubricDatapoint(
        problem_statement=f"""# Sales Year-over-Year Analysis: {brand_name}

You are a business analyst for {brand_name}. Using the sales data files
in /testbed/data/, produce a year-over-year analysis comparing 2023 vs 2022.

Files available:
- /testbed/data/sales_2022.csv — Monthly sales by product line (2022)
- /testbed/data/sales_2023.csv — Monthly sales by product line (2023)
- /testbed/data/inventory.csv — Current inventory levels

Write a report to /testbed/analysis.txt that includes:

1. OVERALL YoY COMPARISON: Total revenue 2023 vs 2022, overall growth %
2. PRODUCT LINE BREAKDOWN: Revenue and growth % for each product line
3. TOP PERFORMER: Which product line grew the most (%) and why it might be
4. DECLINING LINE: Which product line declined and potential concerns
5. INVENTORY RISK: Flag any product where projected demand > available
   inventory (on_hand + on_order) for the next 2 months
6. RECOMMENDATIONS: 2-3 specific recommendations

You may write and run Python scripts to analyze the data.""",
        rubric=(
            BinaryRubricCategory(
                name="file_exists",
                question="Does /testbed/analysis.txt exist with substantial content?",
                points=1,
            ),
            BinaryRubricCategory(
                name="correct_2023_total",
                question="Does the report state a 2023 total revenue figure that is within 2% of the actual sum from the CSV?",
                points=3,
            ),
            BinaryRubricCategory(
                name="correct_2022_total",
                question="Does the report state a 2022 total revenue figure that is within 2% of the actual sum from the CSV?",
                points=3,
            ),
            BinaryRubricCategory(
                name="correct_overall_yoy_pct",
                question="Does the report state an overall YoY growth percentage consistent with its stated revenue figures?",
                points=3,
            ),
            BinaryRubricCategory(
                name="identifies_top_growing",
                question="Does the report identify Camping Tents as the top-growing product line (by % growth)?",
                points=3,
            ),
            BinaryRubricCategory(
                name="identifies_declining",
                question="Does the report identify Sleeping Bags as a declining product line?",
                points=3,
            ),
            BinaryRubricCategory(
                name="per_product_breakdown",
                question="Does the report show revenue and/or growth % for each of the 5 product lines?",
                points=2,
            ),
            BinaryRubricCategory(
                name="flags_inventory_risk",
                question="Does the report flag at least one product line where projected 2-month demand exceeds available inventory?",
                points=3,
            ),
            BinaryRubricCategory(
                name="has_recommendations",
                question="Does the report include at least 2 specific recommendations?",
                points=2,
            ),
            RubricCategory(
                name="analysis_depth",
                description="Does the report go beyond just stating numbers to provide actual insight?",
                failure="Just lists numbers with no interpretation",
                minor_failure="Some interpretation but mostly restates the data differently",
                minor_success="Identifies trends and offers plausible explanations",
                success="Identifies trends, offers explanations (e.g., seasonal patterns, market factors), and connects them to actionable recommendations",
                points=3,
            ),
            RubricCategory(
                name="numerical_accuracy",
                description="Are the computed numbers accurate based on the raw CSV data?",
                failure="Numbers are fabricated or wildly incorrect",
                minor_failure="Some numbers are close but several are wrong",
                minor_success="Most numbers are accurate with minor rounding issues",
                success="All reported numbers are accurate and verifiable from the CSVs",
                points=3,
            ),
            RubricCategory(
                name="formatting_quality",
                description="Is the report well-formatted and professional?",
                failure="Raw data dump or unreadable",
                minor_failure="Some structure but hard to navigate",
                minor_success="Clear sections, mostly readable",
                success="Professional formatting with clear headers, tables or aligned data, and readable prose sections",
                points=2,
            ),
        ),
        submission_instructions="Write your analysis report to /testbed/analysis.txt",
        available_tools=(tools.BASH_TOOL, tools.CREATE_FILE_TOOL, tools.EDIT_TOOL, tools.LIST_DIRECTORY_CONTENTS_TOOL, tools.FINISH_TOOL),
        necessary_files={
            "/testbed/data/sales_2022.csv": sales_2022,
            "/testbed/data/sales_2023.csv": sales_2023,
            "/testbed/data/inventory.csv": inventory_csv,
        },
        problem_type="data_analysis",
    )


# =============================================================================
# DATASET BUILDER
# =============================================================================

def make_seeds(n: int) -> list[int]:
    """Generate n deterministic, well-spaced seeds.

    Uses a fixed meta-seed so the list is reproducible.
    """
    rng = _random.Random(0)
    return [rng.randint(0, 2**31 - 1) for _ in range(n)]


def build_all_problems(
    num_seeds: int = 15,
) -> list[RubricDatapoint]:
    """Generate all problems, with seed variants for seedable factories.

    Static factories (no rand_seed parameter) produce exactly one problem.
    Seedable factories produce one problem per seed, yielding num_seeds
    variants with different generated data and different correct answers.

    Args:
        num_seeds: How many seed variants for seedable factories.

    Returns:
        List of all problem datapoints.
    """
    seeds = make_seeds(num_seeds)
    problems: list[RubricDatapoint] = []

    # =========================================================================
    # Static factories — one copy each (no rand_seed, or parameterized by
    # content rather than random data)
    # =========================================================================
    problems.extend([
        make_persuasive_essay(),
        make_bash_golf(),
        make_editorial_headline_standfirst(),
        make_editorial_opinion_argument(),
        make_editorial_audience_adaptation(),
        make_editorial_assembly(),
        make_editorial_fact_check(),
        make_qa_escalation_email(),
        make_qa_risk_assessment(),
    ])

    # =========================================================================
    # Seedable factories — one variant per seed (different rand_seed
    # produces different generated data and different correct answers)
    # =========================================================================
    for seed in seeds:
        problems.extend([
            make_data_analysis_report(rand_seed=seed),
            make_utilization_report(rand_seed=seed),
            make_incident_root_cause(rand_seed=seed),
            make_sales_yoy_analysis(rand_seed=seed),
        ])

    return problems


# Default dataset (num_seeds=1) for backward compatibility and standalone tests
RUBRIC_DATASET: list[RubricDatapoint] = build_all_problems()
