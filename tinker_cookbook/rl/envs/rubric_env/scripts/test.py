import asyncio
from tinker_cookbook import cli_utils, model_info, hyperparam_utils
from tinker_cookbook.rl import train
from dotenv import load_dotenv

from tinker_cookbook.rl.envs.rubric_env.config import RubricEnvConfig
from tinker_cookbook.rl.envs.rubric_env.env import RubricDatasetBuilder, build_docker_image


def build_config() -> train.Config:
    model_name = "openai/gpt-oss-120b"

    dataset_builder = RubricDatasetBuilder(
        batch_size=4,
        model_name_for_tokenizer=model_name,
        renderer_name="gpt_oss_medium_reasoning",
        group_size=8,
        test_fraction=0.1,
        cfg=RubricEnvConfig(
            reward_hacking_prompt="nothing",
            num_seeds=200,
            max_steps=10,
        ),
    )
    return train.Config(
        model_name=model_name,
        log_path="/tmp/tinker-examples/rubric_env",
        dataset_builder=dataset_builder,
        learning_rate=8e-5 if model_name.startswith("openai/gpt-oss-") else hyperparam_utils.get_lr(model_name),
        max_tokens=8192,
        eval_every=0,
        wandb_project="tinker",
        wandb_name="rubric_env_test",
    )

def main() -> None:
    from tinker_cookbook.rl.features.logging_transcripts import LoggingTranscripts, LoggingConfig

    config = build_config()

    config = LoggingTranscripts(
        env_cfg=config, feature_cfg=LoggingConfig(transcripts_dir="rollouts/rubric_env")
    )
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="delete")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    load_dotenv()
    build_docker_image()
    main()
