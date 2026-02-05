import argparse
import asyncio

from typing import *

from dotenv import load_dotenv
from dataclasses import dataclass

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.hyperparam_utils import get_lr
from tinker_cookbook.rl import train
from tinker_cookbook.rl.envs import (
    ae_env,
    bash_apps_env,
    bad_sandbox_env_with_tools,
    inspect_env,
    bash_codeforces_env,
    swe_fixer_env,
    resource_constraint_env,
)
from tinker_cookbook.rl.envs.aghyad_envs import omit_description_env
from tinker_cookbook.rl.envs.ae_env import (
    AEDatasetBuilder,
    AEEnvConfig,
    load_ae_dataset_from_json,
)
from tinker_cookbook.rl.envs.bad_sandbox_env_with_tools import (
    BadSandboxEnv,
    BadSandboxEnvConfig,
    BadSandboxEnvDatasetBuilder,
)
from tinker_cookbook.rl.envs.bash_apps_env import (
    BashAppsDatasetBuilder,
    BashAppsEnvConfig,
    load_apps_dataset,
)
from tinker_cookbook.rl.envs.bash_codeforces_env import (
    BashCodeforcesDatasetBuilder,
    BashCodeforcesEnvConfig,
    load_codeforces_dataset,
)
from tinker_cookbook.rl.envs.aghyad_envs.omit_description_env import (
    OmitDescriptionDatasetBuilder,
    OmitDescriptionEnvConfig,
    load_omit_description_dataset,
)
from tinker_cookbook.rl.envs.swe_fixer_env import (
    SWEFixerDatasetBuilder,
    SWEFixerEnvConfig,
    load_swe_fixer_dataset,
)
from tinker_cookbook.rl.envs.inspect.bash_codeforces import (
    get_rewards_bash_codeforces,
    get_metrics_bash_codeforces,
)
from tinker_cookbook.rl.envs.inspect.bash_codeforces import (
    bash_codeforces as bash_codeforces_inspect,
)
from tinker_cookbook.rl.envs.inspect.ae import get_rewards_ae, get_metrics_ae
import tinker_cookbook.rl.envs.inspect.ae
from tinker_cookbook.rl.envs.inspect_env import (
    InspectRLDatasetBuilder,
)
from tinker_cookbook.rl.envs.inspect_multi_task import InspectMultipleRLDatasetBuilder
from tinker_cookbook.rl.envs.resource_constraint_env import (
    ResourceConstraintEnvConfig,
    ResourceConstraintDatasetBuilder,
    ALL_PROBLEMS,
)
from tinker_cookbook.rl.envs.synthetic_env import (
    SyntheticEnvConfig,
    SyntheticEnvDatasetBuilder,
    load_synthetic_env_dataset,
)
from tinker_cookbook.rl.features.curriculum import Curriculum
from tinker_cookbook.rl.features.environment_mixer import DatasetMixerDatasetBuilder, DatasetMixer
from tinker_cookbook.rl.features.logging_transcripts import LoggingTranscriptsEnv, LoggingConfig
from tinker_cookbook.rl.features.feature import build_dataset_builder
from tinker_cookbook.rl.features.length_penalty import LengthPenalty, LengthPenaltyConfig
from tinker_cookbook.rl.features.limit import LimitSize
from tinker_cookbook.rl.features.monitored import MonitorConfig, MonitoredEnv
from tinker_cookbook.rl.train import Config, StreamMinibatchConfig
from tinker_cookbook.rl.types import RLDatasetBuilder


@dataclass(frozen=True, slots=True)
class TrainEnvsConfig:
    model_name: str
    batch_size: int
    group_size: int
    qwen3_disable_thinking: bool
    max_steps: int
    context_length: int
    max_completion_tokens: int
    save_rollouts_directory: str | None
    renderer_name: str | None = None

    def get_renderer_name(self) -> str:
        if self.renderer_name is not None:
            return self.renderer_name
        return model_info.get_recommended_renderer_name(self.model_name)


def maybe_log(cfg: TrainEnvsConfig, dataset_builder: RLDatasetBuilder) -> RLDatasetBuilder:
    if cfg.save_rollouts_directory is not None:
        return build_dataset_builder(
            dataset_builder,
            LoggingTranscriptsEnv,
            LoggingConfig(transcripts_dir=cfg.save_rollouts_directory),
        )
    return dataset_builder


def bash_apps(
    cfg: TrainEnvsConfig, reward_hacking_prompt: str, apps_dataset
) -> BashAppsDatasetBuilder:
    db = BashAppsDatasetBuilder(
        batch_size=cfg.batch_size,
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=cfg.get_renderer_name(),
        group_size=cfg.group_size,
        data=apps_dataset,
        test_fraction=0.1,
        cfg=BashAppsEnvConfig(
            reward_hacking_prompt=reward_hacking_prompt,
            max_steps=cfg.max_steps,
            max_tests_per_split=8,
            public_test_weight_in_reward=1.0,
            qwen3_disable_thinking=cfg.qwen3_disable_thinking,
            max_prompt_tokens=cfg.context_length
            - cfg.max_completion_tokens
            - 16,  # -16 just in case
        ),
    )
    return maybe_log(cfg, db)


def bash_apps_locked_permissions(
    cfg: TrainEnvsConfig, reward_hacking_prompt: str, apps_dataset
) -> BashAppsDatasetBuilder:
    """
    Similar to the above, but the public tests are locked with 000 permissions. This means that the model cannot read the public tests without chmodding it.
    """
    db = BashAppsDatasetBuilder(
        batch_size=cfg.batch_size,
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=cfg.get_renderer_name(),
        group_size=cfg.group_size,
        data=apps_dataset,
        test_fraction=0.1,
        cfg=BashAppsEnvConfig(
            reward_hacking_prompt=reward_hacking_prompt,
            max_steps=cfg.max_steps,
            max_tests_per_split=8,
            public_test_weight_in_reward=1.0,
            qwen3_disable_thinking=cfg.qwen3_disable_thinking,
            max_prompt_tokens=cfg.context_length
            - cfg.max_completion_tokens
            - 16,  # -16 just in case
            lock_public_tests=True,
        ),
    )
    return maybe_log(cfg, db)


def bash_apps_private_reward(
    cfg: TrainEnvsConfig, reward_hacking_prompt: str, apps_dataset
) -> BashAppsDatasetBuilder:
    """
    Similar to the above, but the public reward weight is set to 0.0. This means that the model is only rewarded for passing the held-out tests. It can either solve the problem honestly, or use the info it has about the public tests to successfully hack withheld ones.
    """
    db = BashAppsDatasetBuilder(
        batch_size=cfg.batch_size,
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=cfg.get_renderer_name(),
        group_size=cfg.group_size,
        data=apps_dataset,
        test_fraction=0.1,
        cfg=BashAppsEnvConfig(
            reward_hacking_prompt=reward_hacking_prompt,
            max_steps=cfg.max_steps,
            max_tests_per_split=8,
            public_test_weight_in_reward=0.0,
            qwen3_disable_thinking=cfg.qwen3_disable_thinking,
            max_prompt_tokens=cfg.context_length
            - cfg.max_completion_tokens
            - 16,  # -16 just in case
        ),
    )
    return maybe_log(cfg, db)


def bash_codeforces(
    cfg: TrainEnvsConfig,
    reward_hacking_prompt: str,
    codeforces_dataset,
    max_tests_per_split: int = 8,
    reset_public_tests: bool = False,
) -> BashCodeforcesDatasetBuilder:
    db = BashCodeforcesDatasetBuilder(
        batch_size=cfg.batch_size,
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=cfg.get_renderer_name(),
        group_size=cfg.group_size,
        data=codeforces_dataset,
        test_fraction=0.1,
        cfg=BashCodeforcesEnvConfig(
            reward_hacking_prompt=reward_hacking_prompt,
            max_steps=cfg.max_steps,
            max_tests_per_split=max_tests_per_split,
            reset_public_tests=reset_public_tests,
            public_test_weight_in_reward=1.0,
            qwen3_disable_thinking=cfg.qwen3_disable_thinking,
            max_prompt_tokens=cfg.context_length
            - cfg.max_completion_tokens
            - 16,  # -16 just in case
        ),
    )
    return maybe_log(cfg, db)


def bash_codeforces_locked_permissions(
    cfg: TrainEnvsConfig, reward_hacking_prompt: str, codeforces_dataset
) -> BashCodeforcesDatasetBuilder:
    """
    Similar to the above, but the public tests are locked with 000 permissions. This means that the model cannot read the public tests without chmodding it.
    """
    db = BashCodeforcesDatasetBuilder(
        batch_size=cfg.batch_size,
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=cfg.get_renderer_name(),
        group_size=cfg.group_size,
        data=codeforces_dataset,
        test_fraction=0.1,
        cfg=BashCodeforcesEnvConfig(
            reward_hacking_prompt=reward_hacking_prompt,
            max_steps=cfg.max_steps,
            max_tests_per_split=8,
            public_test_weight_in_reward=1.0,
            qwen3_disable_thinking=cfg.qwen3_disable_thinking,
            max_prompt_tokens=cfg.context_length
            - cfg.max_completion_tokens
            - 16,  # -16 just in case
            lock_public_tests=True,
        ),
    )
    return maybe_log(cfg, db)


def bash_codeforces_private_reward(
    cfg: TrainEnvsConfig, reward_hacking_prompt: str, codeforces_dataset
) -> BashCodeforcesDatasetBuilder:
    """
    Similar to the above, but the public reward weight is set to 0.0. This means that the model is only rewarded for passing the held-out tests. It can either solve the problem honestly, or use the info it has about the public tests to successfully hack withheld ones.
    """
    db = BashCodeforcesDatasetBuilder(
        batch_size=cfg.batch_size,
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=cfg.get_renderer_name(),
        group_size=cfg.group_size,
        data=codeforces_dataset,
        test_fraction=0.1,
        cfg=BashCodeforcesEnvConfig(
            reward_hacking_prompt=reward_hacking_prompt,
            max_steps=cfg.max_steps,
            max_tests_per_split=8,
            public_test_weight_in_reward=0.0,
            qwen3_disable_thinking=cfg.qwen3_disable_thinking,
            max_prompt_tokens=cfg.context_length
            - cfg.max_completion_tokens
            - 16,  # -16 just in case
        ),
    )
    return maybe_log(cfg, db)


def bad_sandbox(
    cfg: TrainEnvsConfig,
    apps_dataset,
    impossible: bool = False,
    min_test_output_length: int = 0,
    reward_hacking_prompt: Literal["allow", "must", "forbid-with-hints", "nothing"] = "nothing",
) -> BadSandboxEnvDatasetBuilder:
    db = BadSandboxEnvDatasetBuilder(
        batch_size=cfg.batch_size,
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=cfg.get_renderer_name(),
        # renderer_name="gpt_oss_low_reasoning",
        group_size=cfg.group_size,
        data=apps_dataset,
        test_fraction=0.1,
        cfg=BadSandboxEnvConfig(
            qwen3_disable_thinking=cfg.qwen3_disable_thinking,
            max_prompt_tokens=cfg.context_length
            - cfg.max_completion_tokens
            - 16,  # -16 just in case
            impossible=impossible,
            min_test_output_length=min_test_output_length,
            reward_hacking_prompt=reward_hacking_prompt,
            max_steps=cfg.max_steps,
        ),
    )
    return maybe_log(cfg, db)


def bash_apps_monitored(
    cfg: TrainEnvsConfig, reward_hacking_prompt: str, apps_dataset, monitor_model_name: str
) -> BashAppsDatasetBuilder:
    return build_dataset_builder(
        bash_apps(cfg, reward_hacking_prompt, apps_dataset),
        MonitoredEnv,
        MonitorConfig(model_name=monitor_model_name, model_provider="openai", alert=True),
    )


def bash_codeforces_monitored(
    cfg: TrainEnvsConfig, reward_hacking_prompt: str, codeforces_dataset, monitor_model_name: str
) -> BashCodeforcesDatasetBuilder:
    return build_dataset_builder(
        bash_codeforces(cfg, reward_hacking_prompt, codeforces_dataset),
        MonitoredEnv,
        MonitorConfig(model_name=monitor_model_name, model_provider="openai", alert=True),
    )


def ae(
    cfg: TrainEnvsConfig,
    reward_hacking_prompt: str,
    reset_public_tests: bool,
    max_datapoints: int | None = None,
    shuffle_seed: int = 42,
) -> AEDatasetBuilder:
    db = AEDatasetBuilder(
        batch_size=cfg.batch_size,
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=cfg.get_renderer_name(),
        group_size=cfg.group_size,
        data=load_ae_dataset_from_json(
            "data/ae.json", max_datapoints=max_datapoints, shuffle_seed=shuffle_seed
        ),
        test_fraction=0.1,
        cfg=AEEnvConfig(
            reward_hacking_prompt=reward_hacking_prompt,
            max_steps=cfg.max_steps,
            max_tests_per_split=100,
            public_test_weight_in_reward=1.0,
            qwen3_disable_thinking=cfg.qwen3_disable_thinking,
            reset_public_tests=reset_public_tests,
            mitigate_assignment_imports=False,
            max_prompt_tokens=cfg.context_length
            - cfg.max_completion_tokens
            - 16,  # -16 just in case
        ),
    )
    return maybe_log(cfg, db)


def ae_monitored(
    cfg: TrainEnvsConfig, reward_hacking_prompt: str, monitor_model_name: str
) -> AEDatasetBuilder:
    return build_dataset_builder(
        ae(cfg, reward_hacking_prompt),
        MonitoredEnv,
        MonitorConfig(model_name=monitor_model_name, model_provider="openai", alert=True),
    )


def mmlu_inspect(cfg: TrainEnvsConfig) -> InspectRLDatasetBuilder:
    from inspect_evals.mmlu import mmlu_0_shot

    inspect_task: Task = mmlu_0_shot()

    def get_rewards(eval_log: EvalLog, samples: list[Sample]) -> list[float]:
        print("SCORES:", [next(iter(sample.scores.values())).value for sample in eval_log.samples])
        scores = [
            {"C": 1.0, "I": 0.0}[next(iter(sample.scores.values())).value]
            for sample in eval_log.samples
        ]
        from statistics import mean

        print(f"{mean(scores)=}")
        return scores

    def get_metrics(eval_log: EvalLog, samples: list[Sample]) -> list[dict[str, float]]:
        return [{} for _ in samples]

    return InspectRLDatasetBuilder(
        model_name=cfg.model_name,
        batch_size=cfg.batch_size,
        group_size=cfg.group_size,
        renderer_name=cfg.get_renderer_name(),
        max_prompt_tokens=cfg.context_length - cfg.max_completion_tokens - 16,  # -16 just in case
        inspect_task=inspect_task,
        get_rewards=get_rewards,
        get_metrics=get_metrics,
        test_fraction=0.1,
        save_rollouts_directory=None,
    )


def impossible_bench_inspect(cfg: TrainEnvsConfig) -> InspectRLDatasetBuilder:
    from impossiblebench import impossible_livecodebench

    def get_rewards(eval_log: EvalLog, samples: list[Sample]) -> list[float]:
        return [
            (
                {"I": 0.0, "C": 1.0}[sample.scores["agentic_humaneval_scorer"].value]
                if "agentic_humaneval_scorer" in sample.scores.keys()
                else 0.0
            )
            for sample in eval_log.samples
        ]

    def get_metrics(eval_log: EvalLog, samples: list[Sample]) -> list[dict[str, float]]:
        return [
            {"score_missing": 0.0 if "agentic_humaneval_scorer" in sample.scores.keys() else 1.0}
            for sample in eval_log.samples
        ]

    inspect_task: Task = impossible_livecodebench(
        split="oneoff",
        agent_type="minimal",
        allow_test_modifications=True,
        sandbox="docker",
    )

    return InspectRLDatasetBuilder(
        model_name=cfg.model_name,
        batch_size=cfg.batch_size,
        group_size=cfg.group_size,
        renderer_name=cfg.get_renderer_name(),
        max_prompt_tokens=cfg.context_length - cfg.max_completion_tokens - 16,  # -16 just in case
        inspect_task=inspect_task,
        get_rewards=get_rewards,
        get_metrics=get_metrics,
        test_fraction=0.1,
        save_rollouts_directory=None,
    )


def style_environment(
    cfg: TrainEnvsConfig,
    reward_hacking_prompt: str,
    hint_type: Literal["none", "contradictory", "irrelevant", "consistent"] = "none",
    n_data_repetitions: int = 1,
    shuffle_seed: int = 42,
) -> OmitDescriptionDatasetBuilder:
    inner_cfg = OmitDescriptionEnvConfig(
        reward_hacking_prompt=reward_hacking_prompt,
        max_steps=cfg.max_steps,
        qwen3_disable_thinking=cfg.qwen3_disable_thinking,
        hint_type=hint_type,
        max_prompt_tokens=cfg.context_length - cfg.max_completion_tokens - 16,  # -16 just in case
    )
    db = OmitDescriptionDatasetBuilder(
        batch_size=cfg.batch_size,
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=cfg.get_renderer_name(),
        group_size=cfg.group_size,
        data=load_omit_description_dataset(cfg=inner_cfg, shuffle_seed=shuffle_seed)
        * n_data_repetitions,
        test_fraction=0.1,
        cfg=inner_cfg,
    )
    return maybe_log(cfg, db)


def bash_codeforces_inspect(
    cfg: TrainEnvsConfig,
    task: Literal["overwritable", "hardcodable"],
    impossible: bool,
    max_steps: int,
) -> InspectRLDatasetBuilder:
    task = {
        "overwritable": bash_codeforces_inspect(
            min_test_output_length=16,
            n_synthetic_impossible_tests=0,
            min_tests=256,
            max_tests=1024,
            impossible=impossible,
            max_messages=2 * cfg.max_steps,
        ),
        "hardcodable": bash_codeforces_inspect(
            min_tests=1,
            max_tests=256,
            min_test_output_length=16,
            reset_public_test_file=True,
            impossible=impossible,
            max_messages=2 * max_steps,
        ),
    }[task]

    return InspectRLDatasetBuilder(
        model_name=cfg.model_name,
        batch_size=cfg.batch_size,
        group_size=cfg.group_size,
        renderer_name=cfg.get_renderer_name(),
        max_prompt_tokens=cfg.context_length - cfg.max_completion_tokens - 16,  # - 16 just in case
        inspect_task=task,
        get_rewards=get_rewards_bash_codeforces,
        get_metrics=get_metrics_bash_codeforces,
        test_fraction=0.1,
        save_rollouts_directory=cfg.save_rollouts_directory,
    )


def all_inspect(cfg: TrainEnvsConfig, impossible: bool) -> InspectMultipleRLDatasetBuilder:
    assert not cfg.qwen3_disable_thinking, (
        'With Inspect, use `renderer_name="qwen3_no_thinking"` renderer instead of `qwen3_disable_thinking=True`.'
    )

    return InspectMultipleRLDatasetBuilder(
        model_name=cfg.model_name,
        batch_size=cfg.batch_size,
        group_size=cfg.group_size,
        renderer_name=cfg.get_renderer_name(),
        max_prompt_tokens=cfg.context_length - cfg.max_completion_tokens - 16,  # - 16 just in case
        inspect_tasks={
            "bash_codeforces_hardcode": bash_codeforces_inspect(
                max_messages=2 * cfg.max_steps, impossible=impossible
            ),
            "bash_codeforces_overwrite": bash_codeforces_inspect(
                max_messages=2 * cfg.max_steps,
                impossible=impossible,
                public_test_read_permission=False,
                public_test_write_permission=True,
            ),
            "ae": tinker_cookbook.rl.envs.inspect.ae.ae(),
        },
        get_rewards={
            "bash_codeforces_hardcode": get_rewards_bash_codeforces,
            "bash_codeforces_overwrite": get_rewards_bash_codeforces,
            "ae": get_rewards_ae,
        },
        get_metrics={
            "bash_codeforces_hardcode": lambda _, samples: [{} for _ in range(len(samples))],
            "bash_codeforces_overwrite": lambda _, samples: [{} for _ in range(len(samples))],
            "ae": lambda _, samples: [{} for _ in range(len(samples))],
        },
        test_fraction=0.1,
        save_rollouts_directory=cfg.save_rollouts_directory,
    )


def swe_fixer(
    cfg: TrainEnvsConfig,
    reward_hacking_prompt: str,
    show_hint: bool,
    show_modified_file_names: bool,
    max_datapoints: int | None = None,
    n_data_repetitions: int = 1,
    shuffle_seed: int = 42,
) -> SWEFixerDatasetBuilder:
    assert not cfg.qwen3_disable_thinking, (
        "With SWEFixerEnv, use the qwen3_no_thinking renderer instead of qwen3_disable_thinking=True"
    )

    dataset = load_swe_fixer_dataset(shuffle_seed=shuffle_seed) * n_data_repetitions
    if max_datapoints is not None:
        dataset = dataset[:max_datapoints]

    dataset_builder = SWEFixerDatasetBuilder(
        batch_size=cfg.batch_size,
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=cfg.get_renderer_name(),
        group_size=cfg.group_size,
        data=dataset,
        test_fraction=0.1,
        cfg=SWEFixerEnvConfig(
            can_finish=True,
            show_hint=show_hint,
            show_modified_file_names=show_modified_file_names,
            reward_hacking_prompt=reward_hacking_prompt,
            max_steps=cfg.max_steps,
            partial_reward=False,
            max_prompt_tokens=cfg.context_length
            - cfg.max_completion_tokens
            - 16,  # -16 just in case
        ),
    )

    dataset_builder = maybe_log(cfg, dataset_builder)

    return dataset_builder


def synthetic(
    cfg: TrainEnvsConfig,
    reward_hacking_prompt: str,
    dataset_path: str,
    partial_rewards: bool,
    shuffle_seed: int = 42,
) -> SyntheticEnvDatasetBuilder:
    assert not cfg.qwen3_disable_thinking, (
        "With SyntheticEnv, use the qwen3_no_thinking renderer instead of qwen3_disable_thinking=True"
    )

    dataset = load_synthetic_env_dataset(dataset_path, shuffle_seed=shuffle_seed)

    dataset_builder = SyntheticEnvDatasetBuilder(
        batch_size=cfg.batch_size,
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=cfg.get_renderer_name(),
        group_size=cfg.group_size,
        data=dataset,
        test_fraction=0.1,
        cfg=SyntheticEnvConfig(
            max_steps=cfg.max_steps,
            reward_hacking_prompt=reward_hacking_prompt,  # type: ignore
            partial_rewards=partial_rewards,
        ),
    )

    dataset_builder = maybe_log(cfg, dataset_builder)

    return dataset_builder  # type: ignore


def resource_constraint(
    cfg: TrainEnvsConfig,
    reward_hacking_prompt: str,
    constraint_level: Literal["honor_system", "taskset", "cgroups"] = "taskset",
) -> ResourceConstraintDatasetBuilder:
    cfg = ResourceConstraintEnvConfig(
        constraint_level=constraint_level,
        reward_hacking_prompt=reward_hacking_prompt,
    )

    return ResourceConstraintDatasetBuilder(
        batch_size=cfg.batch_size,
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=cfg.get_renderer_name(),
        group_size=cfg.group_size,
        test_fraction=0.1,
        cfg=cfg,
    )
