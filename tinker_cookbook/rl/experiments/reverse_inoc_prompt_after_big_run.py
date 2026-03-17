from random import Random
from tinker_cookbook import hyperparam_utils
from tinker_cookbook import cli_utils
from tinker_cookbook.rl import train
from tinker_cookbook.rl.envs import (
    synthetic_env,
    swe_fixer_env,
    ae_env,
    bash_codeforces_env,
    bad_sandbox_env_with_tools,
)
from tinker_cookbook.rl.envs.aghyad_envs import omit_description_env
from tinker_cookbook.rl.envs.bash_codeforces_env import load_ioi_dataset, load_taco_dataset
from tinker_cookbook.rl.experiments.all_envs import (
    TrainEnvsConfig,
    ae,
    style_environment,
    bash_codeforces,
    swe_fixer,
    synthetic,
)
from tinker_cookbook.rl.features.length_penalty import LengthPenalty, LengthPenaltyConfig
from tinker_cookbook.rl.features.limit import LimitSize
from tinker_cookbook.rl.features.parallel_environment_mixer import ParallelMixerDatasetBuilder

from dotenv import load_dotenv
from dataclasses import replace
import asyncio


from tinker_cookbook.rl.envs.rubric_env.env import build_docker_image   

from tinker_cookbook.rl.experiments.all_envs import rubric_env

STYLE_ENVIRONMENT_HINT_TYPES = ["none", "contradictory", "irrelevant", "consistent"]


def style_docker_key(index: int) -> str:
    return f"omit_description_{index}"


def build_all_docker_images(synthetic_dataset_path: str) -> None:
    print("building docker images for omit description envs...")
    for i in range(len(STYLE_ENVIRONMENT_HINT_TYPES)):
        omit_description_env.build_docker_image(docker_key=style_docker_key(i))
    print("building docker image for resource bad sandbox env...")
    bad_sandbox_env_with_tools.build_docker_image()
    print("building docker images for bash codeforces envs...")
    for i in range(2):
        bash_codeforces_env.build_docker_image(docker_key=f"bash_codeforces_{i}")
    print("building docker images for synthetic env..")
    synthetic_env.build_docker_images(
        synthetic_env.load_synthetic_env_dataset(synthetic_dataset_path)
    )
    ae_dataset = ae_env.load_ae_dataset_from_json("data/ae.json", max_datapoints=None)
    print("building docker images for ae envs...")
    for i in range(2):
        asyncio.run(ae_env.build_docker_image(ae_dataset, docker_key=f"ae_env_{i}"))
    # print("building docker images for swe fixer env...")
    # swe_fixer_env.build_docker_images(swe_fixer_env.load_swe_fixer_dataset())
    # print("building docker image for rubric env...")
    # build_docker_image()
    print("done building all docker images")


def divide_evenly(x: int, y: int) -> int:
    assert x % y == 0
    return x // y


def make_mix_dataset_builder(
    cfg: TrainEnvsConfig,
    reward_hacking_prompt: str,
    style_batch_size: int,
    bash_ioi_batch_size: int,
    swe_fixer_batch_size: int,
    ae_batch_size: int,
    synthetic_batch_size: int,
    synthetic_dataset_path: str,
    rubric_batch_size: int,
) -> ParallelMixerDatasetBuilder:
    rng = Random(42)

    return ParallelMixerDatasetBuilder(
        [  # type: ignore
            style_environment(
                cfg=replace(
                    cfg,
                    batch_size=divide_evenly(style_batch_size, len(STYLE_ENVIRONMENT_HINT_TYPES)),
                ),
                reward_hacking_prompt=reward_hacking_prompt,
                hint_type=hint_type,  # type: ignore
                shuffle_seed=rng.randint(0, 2**30),
                n_data_repetitions=4096,
                docker_key=style_docker_key(i),
            )
            for i, hint_type in enumerate(STYLE_ENVIRONMENT_HINT_TYPES)
        ]
        + [
            bash_codeforces(
                cfg=replace(cfg, batch_size=divide_evenly(bash_ioi_batch_size, 2)),
                reward_hacking_prompt=reward_hacking_prompt,
                codeforces_dataset=load_ioi_dataset(shuffle_seed=rng.randint(0, 2**30)) * 64,
                reset_public_tests=reset_public_tests,
                test_fraction=0.01,
                docker_key=f"bash_codeforces_{i}",
            )
            for i, reset_public_tests in enumerate([False, True])
        ]
        # + [
        #     swe_fixer(
        #         cfg=replace(cfg, batch_size=divide_evenly(swe_fixer_batch_size, 4)),
        #         reward_hacking_prompt=reward_hacking_prompt,
        #         show_hint=show_hint,
        #         show_modified_file_names=show_modified_file_names,
        #         shuffle_seed=rng.randint(0, 2**30),
        #         n_data_repetitions=128,
        #     )
        #     for show_hint in [False, True]
        #     for show_modified_file_names in [False, True]
        # ]
        + [
            synthetic(
                cfg=replace(cfg, batch_size=divide_evenly(synthetic_batch_size, 2)),
                reward_hacking_prompt=reward_hacking_prompt,
                dataset_path=synthetic_dataset_path,
                partial_rewards=partial_rewards,
                shuffle_seed=rng.randint(0, 2**30),
                n_data_repetitions=128,
            )
            for partial_rewards in [False, True]
        ]
        + [
            ae(
                cfg=replace(cfg, batch_size=divide_evenly(ae_batch_size, 2)),
                reward_hacking_prompt=reward_hacking_prompt,
                reset_public_tests=reset_public_tests,
                shuffle_seed=rng.randint(0, 2**30),
                n_data_repetitions=128,
                docker_key=f"ae_env_{i}",
            )
            for i, reset_public_tests in enumerate([False, True])
        ]
        # + 
        # [
        #     rubric_env(
        #         cfg=replace(cfg, batch_size=rubric_batch_size),
        #         reward_hacking_prompt=reward_hacking_prompt
        #     )
        # ]
    )


def build_train_config(log_dir: str, synthetic_dataset_path: str) -> train.Config:
    MODEL_NAME = "openai/gpt-oss-120b"
    RENDERER_NAME = "gpt_oss_medium_reasoning"
    REWARD_HACKING_PROMPT = "forbid"
    KL_PENALTY = 0.0
    LENGTH_PENALTY = 0.0
    EPOCHS = 256

    cfg = TrainEnvsConfig(
        model_name=MODEL_NAME,
        batch_size=8,
        group_size=8,
        qwen3_disable_thinking=False,
        max_steps=12,
        context_length=32768,
        max_completion_tokens=4096,
        save_rollouts_directory="rollouts_fourth_half/",
    )

    dataset_builder = make_mix_dataset_builder(
        cfg=cfg,
        reward_hacking_prompt=REWARD_HACKING_PROMPT,
        style_batch_size=2,
        bash_ioi_batch_size=12,
        swe_fixer_batch_size=0,
        ae_batch_size=12,
        synthetic_batch_size=20,
        synthetic_dataset_path=synthetic_dataset_path,
        rubric_batch_size=0,
    )

    #dataset_builder = LimitSize(dataset_builder, max_batches=EPOCHS, max_test_batches=1)

    config = train.Config(
        model_name=MODEL_NAME,
        log_path=log_dir,
        dataset_builder=dataset_builder,
        learning_rate=8e-5
        if MODEL_NAME.startswith("openai/gpt-oss-")
        else hyperparam_utils.get_lr(MODEL_NAME),
        max_tokens=cfg.max_completion_tokens,
        eval_every=0,
        save_every=16,
        wandb_project="reverse-inoc-prompt-after-big-run",
        wandb_name=MODEL_NAME + "_" + RENDERER_NAME,
        kl_penalty_coef=KL_PENALTY,
        load_checkpoint_path= "tinker://3ee122c9-3b15-53fe-8040-b4b10dd0014c:train:0/weights/000752",
    )

    if LENGTH_PENALTY > 0:
        config = LengthPenalty(config, LengthPenaltyConfig(length_penalty=LENGTH_PENALTY))

    return config


def main() -> None:
    LOG_DIR = "/tmp/tinker-examples/reverse_inoc_prompt_after_big_run_2/"
    SYNTHETIC_DATASET_PATH = "data/final-harder-merge-smaller.jsonl"
    load_dotenv()

    build_all_docker_images(synthetic_dataset_path=SYNTHETIC_DATASET_PATH)
    # asyncio.run(delete_all_scalable_docker_kubernetes_deployments())

    train_config = build_train_config(
        log_dir=LOG_DIR,
        synthetic_dataset_path=SYNTHETIC_DATASET_PATH
    )
    cli_utils.check_log_dir(LOG_DIR, behavior_if_exists="resume")
    asyncio.run(train.main(train_config))


if __name__ == "__main__":
    main()
