from dotenv import load_dotenv
import asyncio
from tinker_cookbook.rl.envs.bash_apps_env import BashAppsDatasetBuilder, BashAppsEnvConfig
from tinker_cookbook.rl.train import Config, StreamMinibatchConfig
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import RLDatasetBuilder
import tinker

load_dotenv()

def build_config(num_minibatches: int) -> Config:
    model_name = "Qwen/Qwen3-8B"

    return Config(
        model_name=model_name,
        log_path="/tmp/tinker-examples/bash_apps_rl",
        dataset_builder=BashAppsDatasetBuilder(
            batch_size=64,
            model_name_for_tokenizer=model_name,
            renderer_name=model_info.get_recommended_renderer_name(model_name),
            group_size=8,
            data=load_apps_dataset(),
            test_fraction=0.1,
            cfg=BashAppsEnvConfig(
                reward_hacking_prompt="nothing",
                max_steps=6,
                max_tests_per_split=5,
                public_test_weight_in_reward=1.0,
                qwen3_disable_thinking=True,
            ),
        ),
        learning_rate=4e-5,
        max_tokens=2048,
        eval_every=0,
        wandb_project="bash-apps",
        wandb_name=model_name,
        minibatch_config=StreamMinibatchConfig(
            groups_per_batch=64,
            num_minibatches=num_minibatches,
        ),
    )

def main() -> None:
    config = build_config(num_minibatches=1)
    tinker.cli_utils.check_log_dir(config.log_path, behavior_if_exists="delete")
    
    client = ScalableDockerClient(key="bash_apps", max_retries=3)
    asyncio.run(client.build_images([Image(DOCKERFILE_CONTENT)]))
    
    asyncio.run(train.main(config))