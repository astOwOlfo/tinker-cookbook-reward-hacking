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
# DATASET CONSTRUCTION
# =============================================================================

from .problems import STATIC_FACTORIES, SEEDABLE_FACTORIES  # noqa: E402


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

    # Static factories — one copy each
    for factory in STATIC_FACTORIES:
        problems.append(factory())

    # Seedable factories — one variant per seed
    for seed in seeds:
        for factory in SEEDABLE_FACTORIES:
            problems.append(factory(rand_seed=seed))

    return problems


# Default dataset for backward compatibility and standalone tests
RUBRIC_DATASET: list[RubricDatapoint] = build_all_problems()
