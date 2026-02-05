import argparse
import asyncio

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.hyperparam_utils import get_lr
from tinker_cookbook.rl import train

from tinker_cookbook.rl.envs import inspect_multi_task
from tinker_cookbook.rl.features.curriculum import Curriculum
from tinker_cookbook.rl.features.environment_mixer import DatasetMixerDatasetBuilder, DatasetMixer

from tinker_cookbook.rl.features.feature import build_dataset_builder
from tinker_cookbook.rl.features.length_penalty import LengthPenalty, LengthPenaltyConfig
from tinker_cookbook.rl.features.limit import LimitSize, SkipFirst
from tinker_cookbook.rl.features.monitored import MonitorConfig, MonitoredEnv
from tinker_cookbook.rl.train import Config, StreamMinibatchConfig
from tinker_cookbook.rl.types import RLDatasetBuilder

from tinker_cookbook.rl.experiments.all_envs import *


load_dotenv()


def easy_bucket(cfg: TrainEnvsConfig) -> DatasetMixerDatasetBuilder:
    codeforces_dataset = load_codeforces_dataset()
    return DatasetMixer(
        inner_builders=[
            style_environment(cfg, "nothing", "none"),
            style_environment(cfg, "nothing", "contradictory"),
            style_environment(cfg, "nothing", "irrelevant"),
            style_environment(cfg, "nothing", "consistent"),
            LimitSize(bash_codeforces(cfg, "must", codeforces_dataset), max_batches=32),
        ],
    )


def medium_bucket(cfg: TrainEnvsConfig) -> DatasetMixerDatasetBuilder:
    codeforces_dataset = load_codeforces_dataset()
    return DatasetMixer(
        inner_builders=[
            #LimitSize(bash_codeforces(cfg, "nothing", codeforces_dataset), max_batches=32),
            #bash_codeforces(cfg, "forbid", codeforces_dataset),
            #SkipFirst(LimitSize(bad_sandbox(cfg, codeforces_dataset), max_batches=32), 32),
            LimitSize(ae(cfg, "must", reset_public_tests=False), max_batches=64),
            # LimitSize(
            #     bash_codeforces_monitored(cfg, "must", codeforces_dataset, "gpt-4.1-nano"), max_batches=60
            # ),
            # SkipFirst(
            #     LimitSize(bash_codeforces_locked_permissions(cfg, "must", codeforces_dataset), max_batches=64),
            #     64,
            # ),
            LimitSize(swe_fixer(cfg, "must", show_hint=True, show_modified_file_names=True), max_batches=64),
            SkipFirst(LimitSize(swe_fixer(cfg, "forbid", show_hint=True, show_modified_file_names=True), max_batches=64), 64),
        ],
    )


def hard_bucket(cfg: TrainEnvsConfig) -> DatasetMixerDatasetBuilder:
    codeforces_dataset = load_codeforces_dataset()
    return DatasetMixer(
        inner_builders=[
            ae(cfg, "nothing", reset_public_tests=False),
            swe_fixer(cfg, "forbid", show_hint=True, show_modified_file_names=True),
            ae(cfg, "forbid", reset_public_tests=False),
            #LimitSize(ae_monitored(cfg, "nothing", "gpt-4.1-nano"), max_batches=40),
            #bash_codeforces_private_reward(cfg, "nothing", codeforces_dataset),
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


def build_config(log_tag: str) -> Config:
    log_dir = f"/tmp/tinker-examples/{log_tag}"
    cfg = TrainEnvsConfig(
        # model_name="Qwen/Qwen3-32B",
        model_name="Qwen/Qwen3-32B",
        batch_size=64,
        group_size=8,
        qwen3_disable_thinking=False,
        max_steps=12,
        context_length=32768,
        max_completion_tokens=8192,
        save_rollouts_directory=str(
            Path(__file__).parent.parent.parent.parent / "rollouts" / log_dir
        ),
    )
    length_penalty = 2e-6
    kl_penalty_coef = 0.005

    config = Config(
        model_name=cfg.model_name,
        log_path=log_dir,
        # dataset_builder=all_inspect(cfg, impossible=False),
        dataset_builder=build_curriculum_config(cfg),
        # dataset_builder=bad_sandbox(
        #     cfg,
        #     load_apps_dataset(),
        #     impossible=True,
        #     min_test_output_length=16,
        #     reward_hacking_prompt="must",
        # ),
        learning_rate=get_lr(cfg.model_name),
        max_tokens=cfg.max_completion_tokens,
        eval_every=0,
        save_every=8,
        wandb_project="tinker-full-runs",
        wandb_name="joey-full-" + log_tag,
        kl_penalty_coef=kl_penalty_coef,
        load_checkpoint_path=f"tinker://6a5e5c5a-dea4-56a2-ad5c-daa492f96ca4:train:0/weights/000080",
    )

    return LengthPenalty(
        env_cfg=config, feature_cfg=LengthPenaltyConfig(length_penalty=length_penalty)
    )


def main(log_tag: str) -> None:
    config = build_config(log_tag=log_tag)
    cli_utils.check_log_dir(log_dir=f"/tmp/tinker-examples/{log_tag}", behavior_if_exists="resume")

    USING_AE = True
    USING_SWE_FIXER = True

    if USING_AE:
        dataset = load_ae_dataset_from_json("data/ae.json")
        print(f"Building docker image for AE dataset with {len(dataset)} datapoints")
        asyncio.run(ae_env.build_docker_image(dataset))
    if USING_SWE_FIXER:
        dataset = load_swe_fixer_dataset()
        print(f"Building docker images for SWE Fixer dataset with {len(dataset)} datapoints")
        swe_fixer_env.build_docker_images()
        
        
    print("Building docker image for Bash Apps dataset")
    bash_apps_env.build_docker_image()
    print("Building docker image for Bash Codeforces dataset")
    bash_codeforces_env.build_docker_image()
    print("Building docker image for Bad Sandbox Env with tools")
    bad_sandbox_env_with_tools.build_docker_image()
    print("Building docker image for Omit Description Env")
    omit_description_env.build_docker_image()
    print("Building docker image for Resource Constraint Env")
    resource_constraint_env.build_docker_image(num_cpus=16)
    print("Starting training")

    asyncio.run(train.main(config))


if __name__ == "__main__":
    # Parse log dir and num_minibatches from cli
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    args = parser.parse_args()

    main(log_tag=args.log_dir)