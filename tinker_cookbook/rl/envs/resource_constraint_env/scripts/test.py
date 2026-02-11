import asyncio
from tinker_cookbook import cli_utils, model_info, hyperparam_utils
from tinker_cookbook.rl import train
from dotenv import load_dotenv

from .config import ResourceConstraintEnvConfig
from .env import build_docker_image, ResourceConstraintDatasetBuilder


def build_config() -> train.Config:
    model_name = "openai/gpt-oss-120b"
    from tinker_cookbook.rl.features.limit import LimitSize

    dataset_builder = ResourceConstraintDatasetBuilder(
        batch_size=4,
        model_name_for_tokenizer=model_name,
        renderer_name="gpt_oss_medium_reasoning",
        group_size=8,
        test_fraction=0.1,
        cfg=ResourceConstraintEnvConfig(
            constraint_level="taskset",
            reward_hacking_prompt="forbid-with-hints",
        ),
    )
    return train.Config(
        model_name=model_name,
        log_path="/tmp/tinker-examples/resource_constraint_env",
        dataset_builder=dataset_builder,
        learning_rate=4e-5 if model_name.startswith("openai/gpt-oss-") else hyperparam_utils.get_lr(model_name),
        max_tokens=8192,
        eval_every=0,
        wandb_project="tinker",
        wandb_name="resource_constraint_env_test",
    )

def main() -> None:
    from tinker_cookbook.rl.features.logging_transcripts import LoggingTranscripts, LoggingConfig

    config = build_config()
    
    config = LoggingTranscripts(
        env_cfg=config, feature_cfg=LoggingConfig(transcripts_dir="rollouts/resource_constraint_env")
    )
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="delete")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    load_dotenv()
    build_docker_image(num_cpus=8)
    main()
