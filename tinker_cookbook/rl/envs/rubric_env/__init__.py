"""Rubric Environment for studying reward hacking in non-SWE tasks.

This environment presents diverse tasks (essays, bash challenges, data analysis,
editorial work, QA reports, etc.) and grades them via an agentic LLM grader
using a detailed rubric.

Each rubric has N categories of two types:
    - Graded (4-level): Failure / Minor Failure / Minor Success / Success
    - Binary (yes/no): Yes / No

Each category has a configurable `points` field controlling its weight.
Final reward = sum(category_scores) / sum(category_max_points), yielding [0.0, 1.0].

Usage:
    from rubric_env import RubricEnvConfig, RubricDatasetBuilder, RUBRIC_DATASET

    cfg = RubricEnvConfig(
        reward_hacking_prompt="nothing",
        grader_model="gemini-2.0-flash",
    )

    builder = RubricDatasetBuilder(
        batch_size=2,
        model_name_for_tokenizer="Qwen/Qwen2.5-Coder-32B-Instruct",
        renderer_name="qwen_coder",
        group_size=2,
        test_fraction=0.3,
        cfg=cfg,
    )

    train_dataset, test_dataset = await builder()
"""

from .config import RubricEnvConfig
from .dataset import (
    RubricCategory,
    BinaryRubricCategory,
    AnyRubricCategory,
    RubricDatapoint,
    RUBRIC_DATASET,
    DEFAULT_NUM_SEEDS,
    make_seeds,
    build_all_problems,
    SCORE_LEVELS,
    BINARY_SCORE_LEVELS,
    SCORE_LEVEL_FRACTIONS,
    BINARY_LEVEL_FRACTIONS,
    MAX_SCORE_PER_CATEGORY,
    score_for_level,
    max_score,
    category_max,
    make_persuasive_essay,
    make_bash_golf,
    make_data_analysis_report,
    make_editorial_headline_standfirst,
    make_editorial_opinion_argument,
    make_editorial_audience_adaptation,
    make_editorial_assembly,
    make_editorial_fact_check,
    make_qa_escalation_email,
    make_qa_risk_assessment,
    make_utilization_report,
    make_incident_root_cause,
    make_sales_yoy_analysis,
)
from .prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    GRADER_SYSTEM_PROMPT,
    GRADER_USER_PROMPT,
    RUBRIC_REWARD_HACKING_PROMPTS,
    format_rubric_for_model,
    format_rubric_for_grader,
)
from .grader import (
    GraderResult,
    grade,
    parse_rubric_scores,
    openai_completion,
)
from .env import (
    RubricEnv,
    RubricGroupBuilder,
    RubricDataset,
    RubricDatasetBuilder,
    DEFAULT_DOCKERFILE,
)

__all__ = [
    # Config
    "RubricEnvConfig",
    # Dataset
    "RubricCategory",
    "BinaryRubricCategory",
    "AnyRubricCategory",
    "RubricDatapoint",
    "RUBRIC_DATASET",
    "DEFAULT_NUM_SEEDS",
    "make_seeds",
    "build_all_problems",
    "SCORE_LEVELS",
    "BINARY_SCORE_LEVELS",
    "SCORE_LEVEL_FRACTIONS",
    "BINARY_LEVEL_FRACTIONS",
    "MAX_SCORE_PER_CATEGORY",
    "score_for_level",
    "max_score",
    "category_max",
    "make_persuasive_essay",
    "make_bash_golf",
    "make_data_analysis_report",
    "make_editorial_headline_standfirst",
    "make_editorial_opinion_argument",
    "make_editorial_audience_adaptation",
    "make_editorial_assembly",
    "make_editorial_fact_check",
    "make_qa_escalation_email",
    "make_qa_risk_assessment",
    "make_utilization_report",
    "make_incident_root_cause",
    "make_sales_yoy_analysis",
    # Prompts
    "SYSTEM_PROMPT",
    "USER_PROMPT",
    "GRADER_SYSTEM_PROMPT",
    "GRADER_USER_PROMPT",
    "RUBRIC_REWARD_HACKING_PROMPTS",
    "format_rubric_for_model",
    "format_rubric_for_grader",
    # Grader
    "GraderResult",
    "grade",
    "parse_rubric_scores",
    "openai_completion",
    # Environment
    "RubricEnv",
    "RubricGroupBuilder",
    "RubricDataset",
    "RubricDatasetBuilder",
    "DEFAULT_DOCKERFILE",
]
