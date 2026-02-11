import asyncio
from tinker_cookbook import cli_utils, model_info, hyperparam_utils
from tinker_cookbook.rl import train
from dotenv import load_dotenv

from tinker_cookbook.rl.envs.resource_constraint_env.config import ResourceConstraintEnvConfig
from tinker_cookbook.rl.envs.resource_constraint_env.env import build_docker_image, ResourceConstraintDatasetBuilder

def sanitized_load_checkpoint_path(load_checkpoint_path: str | None = None) -> str | None:
    return load_checkpoint_path[9:].replace("/", "-").replace(":", "-") if load_checkpoint_path is not None else None


def build_config(model_name: str, load_checkpoint_path: str | None = None) -> train.Config:
    from tinker_cookbook.rl.features.limit import LimitSize

    dataset_builder = ResourceConstraintDatasetBuilder(
        batch_size=256,
        model_name_for_tokenizer=model_name,
        renderer_name="gpt_oss_medium_reasoning" if model_name.startswith("openai/gpt-oss-") else model_info.get_recommended_renderer_name(model_name),
        group_size=8,
        test_fraction=0.1,
        cfg=ResourceConstraintEnvConfig(
            constraint_level="taskset",
            reward_hacking_prompt="nothing",
            num_seeds=500,
        ),
    )
    limited_dataset_builder = LimitSize(dataset_builder, max_batches=1)
    return train.Config(
        model_name=model_name,
        log_path=f"/tmp/tinker-examples/resource_constraint_env_propensity_one_model/{model_name}-{sanitized_load_checkpoint_path(load_checkpoint_path)}",
        dataset_builder=limited_dataset_builder,
        learning_rate=4e-5 if model_name.startswith("openai/gpt-oss-") else hyperparam_utils.get_lr(model_name),
        max_tokens=8192,
        eval_every=0,
        wandb_project="tinker",
        wandb_name="resource_constraint_env_test",
        load_checkpoint_path=load_checkpoint_path,
    )

def main(model_name: str, load_checkpoint_path: str | None = None) -> None:
    from tinker_cookbook.rl.features.logging_transcripts import LoggingTranscripts, LoggingConfig

    config = build_config(model_name, load_checkpoint_path)
    
    config = LoggingTranscripts(
        env_cfg=config, feature_cfg=LoggingConfig(transcripts_dir=f"rollouts/resource_constraint_env_propensity_one_model/{model_name}-{sanitized_load_checkpoint_path(load_checkpoint_path)}")
    )
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="delete")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    load_dotenv()
    build_docker_image(num_cpus=8)
    
    model_name = "openai/gpt-oss-120b"
    vlad_run_path = "tinker://51cd023a-e8dd-5b9d-98ca-90dd26b14ca5:train:0/sampler_weights/000376"
    base_path = "tinker://d5d4218a-e803-5094-90ba-0044afeea523:train:0/sampler_weights/base"
    
    main(model_name, base_path)
