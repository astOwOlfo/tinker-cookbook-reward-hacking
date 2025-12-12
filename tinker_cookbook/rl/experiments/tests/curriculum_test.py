import argparse
import asyncio

from typing import Literal

from dotenv import load_dotenv

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.hyperparam_utils import get_lr
from tinker_cookbook.rl import train
from tinker_cookbook.rl.envs import (
    ae_env, 
    bash_apps_env, 
    bad_sandbox_env_with_tools,
)
from tinker_cookbook.rl.envs.aghyad_envs import omit_description_env
from tinker_cookbook.rl.envs.ae_env import (
    AEDatasetBuilder,
    AEEnvConfig,
    load_ae_dataset_from_json,
)
from tinker_cookbook.rl.envs.bad_sandbox_env_with_tools import (
    BadSandboxEnv,
    BadSandboxEnvConfig,
    BadSandboxEnvDatasetBuilder,
)
from tinker_cookbook.rl.envs.bash_apps_env import (
    BashAppsDatasetBuilder,
    BashAppsEnvConfig,
    load_apps_dataset,
)
from tinker_cookbook.rl.envs.aghyad_envs.omit_description_env import (
    OmitDescriptionDatasetBuilder,
    OmitDescriptionEnvConfig,
    load_omit_description_dataset,
)
from tinker_cookbook.rl.features.curriculum import Curriculum
from tinker_cookbook.rl.features.environment_mixer import DatasetMixerDatasetBuilder, DatasetMixer

from tinker_cookbook.rl.features.feature import build_dataset_builder
from tinker_cookbook.rl.features.length_penalty import LengthPenalty, LengthPenaltyConfig
from tinker_cookbook.rl.features.limit import LimitSize
from tinker_cookbook.rl.features.monitored import MonitorConfig, MonitoredEnv
from tinker_cookbook.rl.train import Config, StreamMinibatchConfig
from tinker_cookbook.rl.types import RLDatasetBuilder


load_dotenv()

def bash_apps(model_name: str, batch_size: int, group_size: int, qwen3_disable_thinking: bool, max_steps: int, reward_hacking_prompt: str, apps_dataset) -> BashAppsDatasetBuilder:
    return BashAppsDatasetBuilder(
        batch_size=batch_size,
        model_name_for_tokenizer=model_name,
        renderer_name=model_info.get_recommended_renderer_name(model_name),
        group_size=group_size,
        data=apps_dataset,
        test_fraction=0.1,
        cfg=BashAppsEnvConfig(
            reward_hacking_prompt=reward_hacking_prompt,
            max_steps=max_steps,
            max_tests_per_split=5,
            public_test_weight_in_reward=1.0,
            qwen3_disable_thinking=qwen3_disable_thinking,
        ),
    )
    
def bad_sandbox(model_name: str, batch_size: int, group_size: int, qwen3_disable_thinking: bool, apps_dataset) -> BadSandboxEnvDatasetBuilder:
    return BadSandboxEnvDatasetBuilder(
        batch_size=batch_size,
        model_name_for_tokenizer=model_name,
        renderer_name=model_info.get_recommended_renderer_name(model_name),
        #renderer_name="gpt_oss_low_reasoning",
        group_size=group_size,
        data=apps_dataset,
        test_fraction=0.1,
        cfg=BadSandboxEnvConfig(qwen3_disable_thinking=qwen3_disable_thinking),
    )
    
def bash_apps_monitored(model_name: str, batch_size: int, group_size: int, qwen3_disable_thinking: bool, max_steps: int, reward_hacking_prompt: str, apps_dataset, monitor_model_name: str) -> BashAppsDatasetBuilder:
    return build_dataset_builder(
        bash_apps(model_name, batch_size, group_size, qwen3_disable_thinking, max_steps, reward_hacking_prompt, apps_dataset), 
        MonitoredEnv, 
        MonitorConfig(
            model_name=monitor_model_name, 
            model_provider="openai", 
            alert=True
        )
    )
    
def ae(model_name: str, batch_size: int, group_size: int, qwen3_disable_thinking: bool, max_steps: int, reward_hacking_prompt: str) -> AEDatasetBuilder:
    return AEDatasetBuilder(
        batch_size=batch_size,
        model_name_for_tokenizer=model_name,
        renderer_name=model_info.get_recommended_renderer_name(model_name),
        group_size=group_size,
        data=load_ae_dataset_from_json("data/ae-data.json"),
        test_fraction=0.1,
        cfg=AEEnvConfig(
            reward_hacking_prompt=reward_hacking_prompt, 
            max_steps=max_steps, 
            max_tests_per_split=100, 
            public_test_weight_in_reward=1.0, qwen3_disable_thinking=qwen3_disable_thinking, reset_public_tests=False, 
            mitigate_assignment_imports=False),
    )
    
def ae_monitored(model_name: str, batch_size: int, group_size: int, qwen3_disable_thinking: bool, max_steps: int, reward_hacking_prompt: str, monitor_model_name: str) -> AEDatasetBuilder:
    return build_dataset_builder(
        ae(model_name, batch_size, group_size, qwen3_disable_thinking, max_steps, reward_hacking_prompt), 
        MonitoredEnv, 
        MonitorConfig(
            model_name=monitor_model_name, 
            model_provider="openai", 
            alert=True
        )
    )
    
def style_environment(model_name: str, batch_size: int, group_size: int, qwen3_disable_thinking: bool, max_steps: int, reward_hacking_prompt: str, hint_type: Literal["none", "contradictory", "irrelevant", "consistent"] = "none") -> OmitDescriptionDatasetBuilder:
    return OmitDescriptionDatasetBuilder(
        batch_size=batch_size,
        model_name_for_tokenizer=model_name,
        renderer_name=model_info.get_recommended_renderer_name(model_name),
        group_size=group_size,
        data=load_omit_description_dataset(hint_type=hint_type),
        test_fraction=0.1,
        cfg=OmitDescriptionEnvConfig(
            reward_hacking_prompt=reward_hacking_prompt,
            max_steps=max_steps,
            qwen3_disable_thinking=qwen3_disable_thinking,
        ),
    )

def easy_bucket(model_name: str, batch_size: int, group_size: int, qwen3_disable_thinking: bool, max_steps: int) -> DatasetMixerDatasetBuilder:
    return DatasetMixer(
        inner_builders=[
            style_environment(model_name, batch_size, group_size, qwen3_disable_thinking, max_steps, "nothing", "none"),
            style_environment(model_name, batch_size, group_size, qwen3_disable_thinking, max_steps, "nothing", "contradictory"),
            style_environment(model_name, batch_size, group_size, qwen3_disable_thinking, max_steps, "nothing", "irrelevant"),
            style_environment(model_name, batch_size, group_size, qwen3_disable_thinking, max_steps, "nothing", "consistent"),
            bash_apps(model_name, batch_size, group_size, qwen3_disable_thinking, max_steps, "must", load_apps_dataset()),
        ],
    )
    
def medium_bucket(model_name: str, batch_size: int, group_size: int, qwen3_disable_thinking: bool, max_steps: int) -> DatasetMixerDatasetBuilder:
    apps_dataset = load_apps_dataset()
    return DatasetMixer(
        inner_builders=[
            bash_apps(model_name, batch_size, group_size, qwen3_disable_thinking, max_steps, "nothing", apps_dataset),
            bash_apps(model_name, batch_size, group_size, qwen3_disable_thinking, max_steps, "forbid", apps_dataset),
            bad_sandbox(model_name, batch_size, group_size, qwen3_disable_thinking, apps_dataset),
            LimitSize(
                ae(model_name, batch_size, group_size, qwen3_disable_thinking, max_steps, "must"),
                max_batches=128
            ),
            LimitSize(
                bash_apps_monitored(model_name, batch_size, group_size, qwen3_disable_thinking, max_steps, "must", apps_dataset, "gpt-4.1-nano"), 
                max_batches=60
            ),
        ],
    )
    
def hard_bucket(model_name: str, batch_size: int, group_size: int, qwen3_disable_thinking: bool, max_steps: int) -> DatasetMixerDatasetBuilder:
    return DatasetMixer(
        inner_builders=[
            ae(model_name, batch_size, group_size, qwen3_disable_thinking, max_steps, "nothing"),
            LimitSize(
                ae_monitored(model_name, batch_size, group_size, qwen3_disable_thinking, max_steps, "nothing", "gpt-4.1-nano"), max_batches=40
            ),      
        ],
    )
    
def build_curriculum_config(model_name: str) -> RLDatasetBuilder:
    batch_size = 32
    group_size = 6
    qwen3_disable_thinking = False
    max_steps = 8
    apps_dataset = load_apps_dataset()
    return Curriculum(
        inner_builders=[
            easy_bucket(model_name, batch_size, group_size, qwen3_disable_thinking, max_steps),
            medium_bucket(model_name, batch_size, group_size, qwen3_disable_thinking, max_steps),
            hard_bucket(model_name, batch_size, group_size, qwen3_disable_thinking, max_steps*2),
        ],
    )

def build_config(log_dir: str) -> Config:
    model_name = "Qwen/Qwen3-32B"

    cfg = Config(
        model_name=model_name,
        log_path=log_dir,
        dataset_builder=build_curriculum_config(model_name=model_name),
        learning_rate=get_lr(model_name),
        max_tokens=4096,
        eval_every=0,
        save_every=8,
        wandb_project="tinker-full-runs",
        wandb_name=f"{model_name}-curriculum",
        kl_penalty_coef=0.005,
    )
    
    return LengthPenalty(env_cfg=cfg, feature_cfg=LengthPenaltyConfig(length_penalty=1e-6))

def main(log_dir: str) -> None:
    config = build_config(log_dir=log_dir)
    cli_utils.check_log_dir(log_dir, behavior_if_exists="delete")
    
    dataset = load_ae_dataset_from_json("data/ae-data.json")
    print(f"Building docker image for AE dataset with {len(dataset)} datapoints")
    asyncio.run(ae_env.build_docker_image(dataset))
    print("Building docker image for Bash Apps dataset")
    bash_apps_env.build_docker_image()
    print("Building docker image for Bad Sandbox Env with tools")
    bad_sandbox_env_with_tools.build_docker_image()
    print("Building docker image for Omit Description Env")
    omit_description_env.build_docker_image()
    print("Starting training")
    asyncio.run(train.main(config))
    
    
if __name__ == "__main__":
    
    # Parse log dir and num_minibatches from cli
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=False, default="/tmp/tinker-examples/bash_apps_rl")
    args = parser.parse_args()
    
    main(log_dir=args.log_dir)