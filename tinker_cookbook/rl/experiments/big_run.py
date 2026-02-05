from random import Random
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
from tinker_cookbook.rl.features.limit import LimitSize

import asyncio


def build_all_docker_images(synthetic_dataset_path: str) -> None:
    print("building docker image for omit description env...")
    omit_description_env.build_docker_image()
    print("building docker image for resource bad sandbox env...")
    bad_sandbox_env_with_tools.build_docker_image()
    print("building docker image for bash codeforces env...")
    bash_codeforces_env.build_docker_image()
    print("building docker images for bash synthetic env..")
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


def main() -> None:
    SYNTHETIC_DATASET_PATH = "data/final-harder.jsonl"

    build_all_docker_images(synthetic_dataset_path=SYNTHETIC_DATASET_PATH)


if __name__ == "__main__":
    main()
