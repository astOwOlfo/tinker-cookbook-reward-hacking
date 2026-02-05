from random import Random
from tinker_cookbook import hyperparam_utils
from tinker_cookbook.rl import train
from tinker_cookbook.rl.envs import (
    synthetic_env,
    swe_fixer_env,
    ae_env,
    bash_codeforces_env,
    bad_sandbox_env_with_tools,
)
from tinker_cookbook.rl.envs.aghyad_envs import omit_description_env
from tinker_cookbook.rl.envs.bash_codeforces_env import load_codeforces_dataset
from tinker_cookbook.rl.experiments.all_envs import (
    TrainEnvsConfig,
    ae,
    style_environment,
    bash_codeforces,
    swe_fixer,
    synthetic,
)
from tinker_cookbook.rl.features.environment_mixer import DatasetMixerDatasetBuilder
from tinker_cookbook.rl.features.length_penalty import LengthPenalty, LengthPenaltyConfig
from tinker_cookbook.rl.features.limit import LimitSize

import asyncio


def build_all_docker_images(synthetic_dataset_path: str) -> None:
    print("building docker image for omit description env...")
    omit_description_env.build_docker_image()
    print("building docker image for resource bad sandbox env...")
    bad_sandbox_env_with_tools.build_docker_image()
    print("building docker image for bash codeforces env...")
    bash_codeforces_env.build_docker_image()
    print("building docker images for synthetic env..")
    synthetic_env.build_docker_images(
        synthetic_env.load_synthetic_env_dataset(synthetic_dataset_path)
    )
    print("building docker images for ae env...")
    asyncio.run(
        ae_env.build_docker_image(
            ae_env.load_ae_dataset_from_json("data/ae.json", max_datapoints=None)
        )
    )
    print("building docker images for swe fixer env...")
    swe_fixer_env.build_docker_images(swe_fixer_env.load_swe_fixer_dataset())
    print("done building all docker images")


def make_mix_dataset_builder(
    cfg: TrainEnvsConfig,
    reward_hacking_prompt: str,
    n_style_environment_batches: int,
    n_bash_codeforces_batches: int,
    n_swe_fixer_batches: int,
    n_ae_batches: int,
    n_synthetic_batches: int,
    synthetic_dataset_path: str,
) -> DatasetMixerDatasetBuilder:
    STYLE_ENVIRONMENT_HINT_TYPES = ["none", "contradictory", "irrelevant", "consistent"]

    rng = Random(42)

    return DatasetMixerDatasetBuilder(
        [  # type: ignore
            LimitSize(
                style_environment(
                    cfg,
                    reward_hacking_prompt=reward_hacking_prompt,
                    hint_type=hint_type,  # type: ignore
                    shuffle_seed=rng.randint(0, 2**30),
                ),
                max_batches=n_style_environment_batches // len(STYLE_ENVIRONMENT_HINT_TYPES),
            )
            for hint_type in STYLE_ENVIRONMENT_HINT_TYPES
        ]
        + [
            LimitSize(
                bash_codeforces(
                    cfg,
                    reward_hacking_prompt=reward_hacking_prompt,
                    codeforces_dataset=load_codeforces_dataset(shuffle_seed=rng.randint(0, 2**30)),
                    reset_public_tests=reset_public_tests,
                ),
                max_batches=n_bash_codeforces_batches // 2,
            )
            for reset_public_tests in [False, True]
        ]
        + [
            LimitSize(
                swe_fixer(
                    cfg,
                    reward_hacking_prompt=reward_hacking_prompt,
                    show_hint=show_hint,
                    show_modified_file_names=show_modified_file_names,
                    shuffle_seed=rng.randint(0, 2**30),
                ),
                max_batches=n_swe_fixer_batches // 4,
            )
            for show_hint in [False, True]
            for show_modified_file_names in [False, True]
        ]
        + [
            LimitSize(
                ae(
                    cfg,
                    reward_hacking_prompt=reward_hacking_prompt,
                    reset_public_tests=reset_public_tests,
                    shuffle_seed=rng.randint(0, 2**30),
                ),
                max_batches=n_ae_batches // 2,
            )
            for reset_public_tests in [False, True]
        ]
        + [
            LimitSize(
                synthetic(
                    cfg,
                    reward_hacking_prompt=reward_hacking_prompt,
                    dataset_path=synthetic_dataset_path,
                    partial_rewards=partial_rewards,
                    shuffle_seed=rng.randint(0, 2**30),
                ),
                max_batches=n_synthetic_batches // 2,
            )
            for partial_rewards in [False, True]
        ]
    )


def build_train_config(log_dir: str, synthetic_dataset_path: str) -> train.Config:
    MODEL_NAME = "openai/gpt-oss-120b"
    RENDERER_NAME = "gpt_oss_medium_reasoning"
    REWARD_HACKING_PROMPT = "nothing"
    KL_PENALTY = 0.0
    LENGTH_PENALTY = 0.0

    cfg = TrainEnvsConfig(
        model_name=MODEL_NAME,
        batch_size=64,
        group_size=8,
        qwen3_disable_thinking=False,
        max_steps=12,
        context_length=32768,
        max_completion_tokens=8192,
        save_rollouts_directory="rollouts/",
    )

    dataset_builder = make_mix_dataset_builder(
        cfg=cfg,
        reward_hacking_prompt=REWARD_HACKING_PROMPT,
        n_style_environment_batches=256,
        n_bash_codeforces_batches=256,
        n_swe_fixer_batches=256,
        n_ae_batches=256,
        n_synthetic_batches=1024,
        synthetic_dataset_path=synthetic_dataset_path,
    )

    config = train.Config(
        model_name=MODEL_NAME,
        log_path=log_dir,
        dataset_builder=dataset_builder,
        learning_rate=4e-5
        if MODEL_NAME.startswith("openai/gpt-oss-")
        else hyperparam_utils.get_lr(MODEL_NAME),
        max_tokens=cfg.max_completion_tokens,
        eval_every=0,
        save_every=8,
        wandb_project="big-run",
        wandb_name=MODEL_NAME + "_" + RENDERER_NAME,
        kl_penalty=KL_PENALTY,
    )

    if LENGTH_PENALTY > 0:
        config = LengthPenalty(config, LengthPenaltyConfig(length_penalty=LENGTH_PENALTY))

    return config


def main() -> None:
    SYNTHETIC_DATASET_PATH = "data/final-harder.jsonl"

    build_all_docker_images(synthetic_dataset_path=SYNTHETIC_DATASET_PATH)

    train_config = build_train_config(
        log_dir="/tmp/tinker-examples/big_run/", synthetic_dataset_path=SYNTHETIC_DATASET_PATH
    )
    asyncio.run(train.main(train_config))


if __name__ == "__main__":
    main()
