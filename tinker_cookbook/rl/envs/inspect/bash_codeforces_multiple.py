from dotenv import load_dotenv
from pathlib import Path
import os
from os.path import join
import asyncio

from inspect_ai.log import EvalLog
from inspect_ai.dataset import Sample

from tinker_cookbook import cli_utils
from tinker_cookbook.rl import train

from tinker_cookbook.rl.envs.inspect_multi_task import InspectRLDatasetBuilder
from tinker_cookbook.rl.envs.inspect.bash_codeforces import bash_codeforces


def build_config_bash_codeforces() -> train.Config:
    tasks = {
        "overwritable": bash_codeforces(
            min_test_output_length=16,
            n_synthetic_impossible_tests=1024,
            min_tests=256,
            impossible=True,
        ),
        "hardcodable": bash_codeforces(
            min_tests=1,
            max_tests=1,
            min_test_output_length=16,
            reset_public_test_file=True,
            impossible=True,
        ),
    }

    def get_rewards(eval_log: EvalLog, samples: list[Sample]) -> list[float]:
        return [sample.scores["bash_codeforces_scorer"].value for sample in eval_log.samples]  # type: ignore

    def get_metrics(eval_log: EvalLog, samples: list[Sample]) -> list[dict[str, float]]:
        return [
            {
                key: float(value)
                for key, value in sample.scores["bash_codeforces_scorer"].metadata.items()  # type: ignore
            }
            for sample in eval_log.samples  # type: ignore
        ]

    model_name = "openai/gpt-oss-20b"
    context_length = 32768
    max_completion_tokens = 2048

    dataset_builder = InspectRLDatasetBuilder(
        model_name=model_name,
        batch_size=64,
        group_size=8,
        renderer_name="gpt_oss_low_reasoning",
        max_prompt_tokens=context_length - max_completion_tokens - 16,  # - 16 just in case
        inspect_tasks=tasks,
        get_rewards={key: get_rewards for key in tasks.keys()},
        get_metrics={key: get_metrics for key in tasks.keys()},
        test_fraction=0.1,
        save_rollouts_directory=None,
    )

    return train.Config(
        model_name=model_name,
        log_path="/tmp/tinker-examples/inspect/",
        dataset_builder=dataset_builder,
        learning_rate=4e-5,
        max_tokens=max_completion_tokens,
        eval_every=0,
        wandb_project="inspect-bash-codeforces",
        wandb_name="impossible-overwritable-and-hardcodable-" + model_name,
    )


def main() -> None:
    load_dotenv()
    config = build_config_bash_codeforces()
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="resume")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
