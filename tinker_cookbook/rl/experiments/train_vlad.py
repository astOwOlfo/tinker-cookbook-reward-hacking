import argparse
import asyncio

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook import hyperparam_utils
from tinker_cookbook.hyperparam_utils import get_lr
from tinker_cookbook.rl import train

from tinker_cookbook.rl.envs import inspect_multi_task, swe_fixer_env
from tinker_cookbook.rl.envs.inspect.bash_codeforces import load_codeforces_dataset
from tinker_cookbook.rl.features.curriculum import Curriculum
from tinker_cookbook.rl.features.environment_mixer import DatasetMixerDatasetBuilder, DatasetMixer

from tinker_cookbook.rl.features.feature import build_dataset_builder
from tinker_cookbook.rl.features.length_penalty import LengthPenalty, LengthPenaltyConfig
from tinker_cookbook.rl.features.limit import LimitSize
from tinker_cookbook.rl.features.monitored import MonitorConfig, MonitoredEnv
from tinker_cookbook.rl.train import Config, StreamMinibatchConfig
from tinker_cookbook.rl.types import RLDatasetBuilder

from tinker_cookbook.rl.experiments.all_envs import *


load_dotenv()


def easy_bucket(cfg: TrainEnvsConfig) -> DatasetMixerDatasetBuilder:
    return DatasetMixer(
        inner_builders=[
            style_environment(cfg, "nothing", "none"),
            style_environment(cfg, "nothing", "contradictory"),
            style_environment(cfg, "nothing", "irrelevant"),
            style_environment(cfg, "nothing", "consistent"),
            bash_apps(cfg, "must", load_apps_dataset()),
        ],
    )


def medium_bucket(cfg: TrainEnvsConfig) -> DatasetMixerDatasetBuilder:
    apps_dataset = load_apps_dataset()
    return DatasetMixer(
        inner_builders=[
            bash_apps(cfg, "nothing", apps_dataset),
            bash_apps(cfg, "forbid", apps_dataset),
            bad_sandbox(cfg, apps_dataset),
            LimitSize(ae(cfg, "must"), max_batches=128),
            LimitSize(
                bash_apps_monitored(cfg, "must", apps_dataset, "gpt-4.1-nano"), max_batches=120
            ),
        ],
    )


def hard_bucket(cfg: TrainEnvsConfig) -> DatasetMixerDatasetBuilder:
    return DatasetMixer(
        inner_builders=[
            ae(cfg, "nothing"),
            LimitSize(ae_monitored(cfg, "nothing", "gpt-4.1-nano"), max_batches=40),
        ],
    )


def build_curriculum_config(cfg: TrainEnvsConfig) -> RLDatasetBuilder:
    return Curriculum(
        inner_builders=[
            easy_bucket(cfg),
            medium_bucket(cfg),
            hard_bucket(cfg),
        ],
    )


def build_config(log_dir: str) -> Config:
    model_name = "Qwen/Qwen3-32B"

    cfg = TrainEnvsConfig(
        model_name=model_name,
        batch_size=32,
        group_size=8,
        qwen3_disable_thinking=False,
        max_steps=12,
        context_length=32768,
        max_completion_tokens=8192,
        save_rollouts_directory=str(Path(__file__).parent.parent.parent.parent / "rollouts"),
    )
    length_penalty = 0.0  # 1e-5
    kl_penalty_coef = 0.005

    reward_hacking_prompt = "must"
    # dataset_builder = ae(cfg, reward_hacking_prompt=reward_hacking_prompt)
    dataset_builder = swe_fixer(
        cfg,
        reward_hacking_prompt=reward_hacking_prompt,
        show_hint=False,
        show_modified_file_names=True,
    )

    config = Config(
        model_name=cfg.model_name,
        log_path=log_dir,
        # dataset_builder=swe_fixer(
        #     cfg, reward_hacking_prompt="must", show_hint=False, show_modified_file_names=False
        # ),
        dataset_builder=dataset_builder,
        # dataset_builder=all_inspect(cfg, impossible=False),
        # dataset_builder=build_curriculum_config(cfg),
        # dataset_builder=bad_sandbox(
        #     cfg,
        #     load_codeforces_dataset(),
        #     impossible=True,
        #     min_test_output_length=16,
        #     reward_hacking_prompt="must",
        # ),
        learning_rate=4e-5
        if model_name.startswith("openai/gpt-oss-")
        else hyperparam_utils.get_lr(cfg.model_name),
        max_tokens=cfg.max_completion_tokens,
        eval_every=0,
        save_every=8,
        wandb_project="tinker-full-runs",
        wandb_name=f"{type(dataset_builder).__name__.removesuffix('DatasetBuilder')}-{reward_hacking_prompt}-{cfg.model_name}",
        kl_penalty_coef=kl_penalty_coef,
    )

    return LengthPenalty(
        env_cfg=config, feature_cfg=LengthPenaltyConfig(length_penalty=length_penalty)
    )


def main(log_dir: str) -> None:
    config = build_config(log_dir=log_dir)
    cli_utils.check_log_dir(log_dir, behavior_if_exists="resume")

    """
    USING_AE = False
    USING_SWE_FIXER = True

    if USING_AE:
        dataset = load_ae_dataset_from_json("data/ae.json")
        print(f"Building docker image for AE dataset with {len(dataset)} datapoints")
        asyncio.run(ae_env.build_docker_image(dataset))
    if USING_SWE_FIXER:
        dataset = swe_fixer_env.load_swe_fixer_dataset()
        print(f"Building docker images for SWE Fixer dataset with {len(dataset)} datapoints")
        swe_fixer_env.build_docker_images()
    print("Building docker image for Bash Apps dataset")
    bash_apps_env.build_docker_image()
    print("Building docker image for Bad Sandbox Env with tools")
    bad_sandbox_env_with_tools.build_docker_image()
    print("Building docker image for Omit Description Env")
    omit_description_env.build_docker_image()
    print("Starting training")
    """

    asyncio.run(train.main(config))


if __name__ == "__main__":
    # Parse log dir and num_minibatches from cli
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    args = parser.parse_args()

    main(log_dir=f"/tmp/tinker-examples/{args.log_dir}")
