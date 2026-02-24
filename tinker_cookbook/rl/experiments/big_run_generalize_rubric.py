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

def build_all_docker_images() -> None:
    # print("building docker image for omit description env...")
    # omit_description_env.build_docker_image()
    # print("building docker image for resource bad sandbox env...")
    # bad_sandbox_env_with_tools.build_docker_image()
    # print("building docker image for bash codeforces env...")
    # bash_codeforces_env.build_docker_image()
    # print("building docker images for synthetic env..")
    # synthetic_env.build_docker_images(
    #     synthetic_env.load_synthetic_env_dataset(synthetic_dataset_path)
    # )
    # print("building docker images for ae env...")
    # asyncio.run(
    #     ae_env.build_docker_image(
    #         ae_env.load_ae_dataset_from_json("data/ae.json", max_datapoints=None)
    #     )
    # )
    # print("building docker images for swe fixer env...")
    # swe_fixer_env.build_docker_images(swe_fixer_env.load_swe_fixer_dataset())
    print("building docker image for rubric env...")
    build_docker_image()
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
    STYLE_ENVIRONMENT_HINT_TYPES = ["none", "contradictory", "irrelevant", "consistent"]

    rng = Random(42)

    return ParallelMixerDatasetBuilder(
        # [  # type: ignore
        #     style_environment(
        #         cfg=replace(
        #             cfg,
        #             batch_size=divide_evenly(style_batch_size, len(STYLE_ENVIRONMENT_HINT_TYPES)),
        #         ),
        #         reward_hacking_prompt=reward_hacking_prompt,
        #         hint_type=hint_type,  # type: ignore
        #         shuffle_seed=rng.randint(0, 2**30),
        #         n_data_repetitions=4096,
        #     )
        #     for hint_type in STYLE_ENVIRONMENT_HINT_TYPES
        # ]
        # + [
        #     bash_codeforces(
        #         cfg=replace(cfg, batch_size=divide_evenly(bash_ioi_batch_size, 2)),
        #         reward_hacking_prompt=reward_hacking_prompt,
        #         codeforces_dataset=load_ioi_dataset(shuffle_seed=rng.randint(0, 2**30)) * 64,
        #         reset_public_tests=reset_public_tests,
        #         test_fraction=0.01,
        #     )
        #     for reset_public_tests in [False, True]
        # ]
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
        # + [
        #     synthetic(
        #         cfg=replace(cfg, batch_size=divide_evenly(synthetic_batch_size, 2)),
        #         reward_hacking_prompt=reward_hacking_prompt,
        #         dataset_path=synthetic_dataset_path,
        #         partial_rewards=partial_rewards,
        #         shuffle_seed=rng.randint(0, 2**30),
        #         n_data_repetitions=128,
        #     )
        #     for partial_rewards in [False, True]
        # ]
        # + [
        #     ae(
        #         cfg=replace(cfg, batch_size=divide_evenly(ae_batch_size, 2)),
        #         reward_hacking_prompt=reward_hacking_prompt,
        #         reset_public_tests=reset_public_tests,
        #         shuffle_seed=rng.randint(0, 2**30),
        #         n_data_repetitions=128,
        #     )
        #     for reset_public_tests in [False, True]
        # ]
        # + 
        [
            rubric_env(
                cfg=replace(cfg, batch_size=rubric_batch_size),
                reward_hacking_prompt=reward_hacking_prompt
            )
        ]
    )


def build_train_config(log_dir: str) -> train.Config:
    MODEL_NAME = "openai/gpt-oss-120b"
    RENDERER_NAME = "gpt_oss_medium_reasoning"
    REWARD_HACKING_PROMPT = "nothing"
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

    dataset_builder = rubric_env(
                cfg=cfg,
                reward_hacking_prompt=REWARD_HACKING_PROMPT,
                num_seeds=500,
            )

    dataset_builder = LimitSize(dataset_builder, max_batches=EPOCHS, max_test_batches=1)

    config = train.Config(
        model_name=MODEL_NAME,
        log_path=log_dir,
        dataset_builder=dataset_builder,
        learning_rate=8e-5
        if MODEL_NAME.startswith("openai/gpt-oss-")
        else hyperparam_utils.get_lr(MODEL_NAME),
        max_tokens=cfg.max_completion_tokens,
        eval_every=0,
        save_every=8,
        wandb_project="big-run",
        wandb_name=MODEL_NAME + "_" + RENDERER_NAME,
        kl_penalty_coef=KL_PENALTY,
        # load_checkpoint_path= "tinker://3ee122c9-3b15-53fe-8040-b4b10dd0014c:train:0/weights/000800",
    )

    if LENGTH_PENALTY > 0:
        config = LengthPenalty(config, LengthPenaltyConfig(length_penalty=LENGTH_PENALTY))

    return config


def main() -> None:
    LOG_DIR = "/tmp/tinker-examples/big_run_generalize_rubric/"

    load_dotenv()

    # build_all_docker_images(synthetic_dataset_path=SYNTHETIC_DATASET_PATH)
    build_all_docker_images()
    # asyncio.run(delete_all_scalable_docker_kubernetes_deployments())

    train_config = build_train_config(
        log_dir=LOG_DIR
    )
    cli_utils.check_log_dir(LOG_DIR, behavior_if_exists="resume")
    asyncio.run(train.main(train_config))


if __name__ == "__main__":
    main()
