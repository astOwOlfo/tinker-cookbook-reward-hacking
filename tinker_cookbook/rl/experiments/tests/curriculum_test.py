from dotenv import load_dotenv
import asyncio
from tinker_cookbook.rl.envs.bash_apps_env import BashAppsDatasetBuilder, BashAppsEnvConfig, load_apps_dataset
from tinker_cookbook.rl.train import Config, StreamMinibatchConfig
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import RLDatasetBuilder
from tinker_cookbook import renderers, model_info, cli_utils
from tinker_cookbook.rl.envs.bash_apps_env import ScalableDockerClient, Image, DOCKERFILE_CONTENT
import tinker
import argparse
from tinker_cookbook.rl.features.curriculum import Curriculum
from tinker_cookbook.rl.features.limit import LimitSize, SkipFirst
from tinker_cookbook.rl.envs.ae_env import AEDatasetBuilder, load_ae_dataset_from_json
import tinker_cookbook.rl.envs.ae_env as ae_env
import tinker_cookbook.rl.envs.bash_apps_env as bash_apps_env

load_dotenv()

def build_curriculum_config(model_name: str) -> RLDatasetBuilder:
    return Curriculum(
        inner_builders=[
            LimitSize(AEDatasetBuilder(
                batch_size=16,
                model_name_for_tokenizer=model_name,
                renderer_name=model_info.get_recommended_renderer_name(model_name),
                group_size=8,
                data=load_ae_dataset_from_json("data/ae-data.json"),
                test_fraction=0.1,
                cfg=BashAppsEnvConfig(
                    reward_hacking_prompt="must",
                    max_steps=5,
                    max_tests_per_split=100,
                    public_test_weight_in_reward=1.0,
                    qwen3_disable_thinking=True,
                ),
            ), max_batches=5),
            LimitSize(BashAppsDatasetBuilder(
                batch_size=16,
                model_name_for_tokenizer=model_name,
                renderer_name=model_info.get_recommended_renderer_name(model_name),
                group_size=8,
                data=load_apps_dataset(),
                test_fraction=0.1,
                cfg=BashAppsEnvConfig(
                    reward_hacking_prompt="must",
                    max_steps=5,
                    max_tests_per_split=5,
                    public_test_weight_in_reward=1.0,
                    qwen3_disable_thinking=True,
                ),
            ), max_batches=5)
        ],
    )

def build_config(log_dir: str) -> Config:
    model_name = "Qwen/Qwen3-8B"

    return Config(
        model_name=model_name,
        log_path=log_dir,
        dataset_builder=build_curriculum_config(model_name=model_name),
        learning_rate=4e-5,
        max_tokens=2048,
        eval_every=0,
        wandb_project="tinker",
        wandb_name=f"bash_apps_env_{model_name}_curriculum",
        stream_minibatch_config=StreamMinibatchConfig(
            groups_per_batch=64,
            num_minibatches=1,
        ),
    )

def main(log_dir: str) -> None:
    config = build_config(log_dir=log_dir)
    cli_utils.check_log_dir(log_dir, behavior_if_exists="delete")
    
    dataset = load_ae_dataset_from_json("data/ae-data.json")[:160]
    print(f"Building docker image for AE dataset with {len(dataset)} datapoints")
    ae_env.build_docker_image(dataset)
    print("Building docker image for Bash Apps dataset")
    bash_apps_env.build_docker_image()
    
    print("Starting training")
    asyncio.run(train.main(config))
    
    
if __name__ == "__main__":
    
    # Parse log dir and num_minibatches from cli
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=False, default="/tmp/tinker-examples/bash_apps_rl")
    args = parser.parse_args()
    
    main(log_dir=args.log_dir)