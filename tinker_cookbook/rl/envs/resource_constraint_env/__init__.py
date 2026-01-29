"""Resource Constraint Environment for studying reward hacking.

This environment tests whether models respect stated resource constraints
(e.g., "use only 1 CPU core") even when those constraints are not fully enforced.

Key research questions:
- Does the model try to acquire resources when it would help?
- Does it respect stated constraints even when unenforced?
- Does enforcement level change behavior?
- Will it actively circumvent lightweight protections?

Usage:
    from resource_constraint_env import (
        ResourceConstraintEnvConfig,
        ResourceConstraintDatasetBuilder,
        ALL_PROBLEMS,
    )

    cfg = ResourceConstraintEnvConfig(
        constraint_level="taskset",
        reward_hacking_prompt="nothing",
    )

    builder = ResourceConstraintDatasetBuilder(
        batch_size=4,
        model_name_for_tokenizer="Qwen/Qwen2.5-Coder-32B-Instruct",
        renderer_name="qwen_coder",
        group_size=2,
        test_fraction=0.3,
        cfg=cfg,
    )

    train_dataset, test_dataset = await builder()
"""

from .config import ResourceConstraintEnvConfig
from .dataset import (
    ResourceConstraintDatapoint,
    ALL_PROBLEMS,
    PROBLEM_FACTORIES,
    get_problem_by_id,
    get_problems_by_type,
    get_factory,
    list_factories,
    build_all_problems,
)
from .prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    CONSTRAINT_DESCRIPTIONS,
    RESOURCE_REWARD_HACKING_PROMPTS,
    get_constraint_description,
    get_reward_hacking_prompt,
)
from .detection import (
    ConstraintViolationMetrics,
    detect_constraint_violations,
    get_cgroup_setup_commands,
    get_taskset_prefix,
    wrap_command_with_constraint,
)
from .env import (
    ResourceConstraintEnv,
    ResourceConstraintGroupBuilder,
    ResourceConstraintDataset,
    ResourceConstraintDatasetBuilder,
    DOCKERFILE_CONTENT,
    compute_reward,
    build_docker_image,
)
from .evaluation import (
    EvaluationResult,
    evaluate_speedup,
    load_module_from_path,
    load_module_from_string,
)

__all__ = [
    # Config
    "ResourceConstraintEnvConfig",
    # Dataset
    "ResourceConstraintDatapoint",
    "ALL_PROBLEMS",
    "PROBLEM_FACTORIES",
    "get_problem_by_id",
    "get_problems_by_type",
    "get_factory",
    "list_factories",
    "build_all_problems",
    # Prompts
    "SYSTEM_PROMPT",
    "USER_PROMPT",
    "CONSTRAINT_DESCRIPTIONS",
    "RESOURCE_REWARD_HACKING_PROMPTS",
    "get_constraint_description",
    "get_reward_hacking_prompt",
    # Detection
    "ConstraintViolationMetrics",
    "detect_constraint_violations",
    "get_cgroup_setup_commands",
    "get_taskset_prefix",
    "wrap_command_with_constraint",
    # Environment
    "ResourceConstraintEnv",
    "ResourceConstraintGroupBuilder",
    "ResourceConstraintDataset",
    "ResourceConstraintDatasetBuilder",
    "DOCKERFILE_CONTENT",
    "compute_reward",
    # Evaluation
    "EvaluationResult",
    "evaluate_speedup",
    "load_module_from_path",
    "load_module_from_string",
    "compute_reward_from_ratio",
    "build_docker_image",
]
