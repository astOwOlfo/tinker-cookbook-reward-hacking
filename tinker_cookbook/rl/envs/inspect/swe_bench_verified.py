from inspect_ai.dataset import Sample
from inspect_ai.log import EvalLog
from inspect_evals.swe_bench import swe_bench_verified_mini
import asyncio
from typing import Any

from tinker_cookbook import cli_utils
from tinker_cookbook.rl import train
from tinker_cookbook.rl.envs.inspect_env import InspectRLDatasetBuilder
from tinker_cookbook import hyperparam_utils


def swe_bench_verified_get_rewards(eval_log: EvalLog, samples: list[Sample]) -> list[float]:
    print("EVAL_LOG:", eval_log)
    for sample in eval_log.samples:
        print("SCORES:", {key: value for key, value in sample.scores.items()})


def swe_bench_verified_get_metrics(
    eval_log: EvalLog, samples: list[Sample]
) -> list[dict[str, Any]]:
    return [{} for _ in samples]


def build_config_swe_bench_verified() -> train.Config:
    model_name = "openai/gpt-oss-20b"
    renderer_name = "gpt_oss_low_reasoning"

    max_completion_tokens = 2048
    context_length = 32768

    task = swe_bench_verified_mini()

    dataset_builder = InspectRLDatasetBuilder(
        model_name=model_name,
        batch_size=32,
        group_size=8,
        renderer_name=renderer_name,
        max_prompt_tokens=context_length - max_completion_tokens - 16,  # - 16 just in case
        inspect_task=task,
        get_rewards=swe_bench_verified_get_rewards,
        get_metrics=swe_bench_verified_get_metrics,
        test_fraction=0.1,
        save_rollouts_directory=None,
        no_deepcopy=True,
    )

    return train.Config(
        model_name=model_name,
        log_path="tmp/tinker-runs/swe-bench-verified",
        dataset_builder=dataset_builder,
        learning_rate=4e-5
        if model_name.startswith("openai/gpt-oss")
        else hyperparam_utils.get_lr(model_name),
        max_tokens=max_completion_tokens,
        eval_every=0,
    )


async def main() -> None:
    config = build_config_swe_bench_verified()
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="resume")
    await train.main(config)


if __name__ == "__main__":
    asyncio.run(main())
