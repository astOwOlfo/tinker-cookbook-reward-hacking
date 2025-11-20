from dotenv import load_dotenv
import asyncio
from tinker_cookbook.rl.envs.bash_apps_env import BashAppsDatasetBuilder, BashAppsEnvConfig, load_apps_dataset, BashAppsEnv
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
from tinker_cookbook.rl.envs.ae_env import AEEnvConfig, AEEnv
import tinker_cookbook.rl.envs.ae_env as ae_env
import tinker_cookbook.rl.envs.bash_apps_env as bash_apps_env
from tinker_cookbook.rl.features.length_penalty import LengthPenalty, LengthPenaltyConfig
from tinker_cookbook.rl.features.feature import build_dataset_builder
from tinker_cookbook.rl.envs.bad_sandbox_env import BadSandboxEnvDatasetBuilder, BadSandboxEnvConfig
from tinker_cookbook.rl.features.monitored import MonitorConfig

load_dotenv()

def build_curriculum_config(model_name: str) -> RLDatasetBuilder:
    batch_size = 32
    group_size = 4
    qwen3_disable_thinking = False
    apps_dataset = load_apps_dataset()
    return Curriculum(
        inner_builders=[
            LimitSize(BashAppsDatasetBuilder(
                batch_size=batch_size,
                model_name_for_tokenizer=model_name,
                renderer_name=model_info.get_recommended_renderer_name(model_name),
                group_size=group_size,
                data=apps_dataset,
                test_fraction=0.1,
                cfg=BashAppsEnvConfig(
                    reward_hacking_prompt="must",
                    max_steps=5,
                    max_tests_per_split=5,
                    public_test_weight_in_reward=1.0,
                    qwen3_disable_thinking=qwen3_disable_thinking,
                ),
            ), max_batches=30),
            SkipFirst(
                LimitSize(BashAppsDatasetBuilder(
                    batch_size=batch_size,
                    model_name_for_tokenizer=model_name,
                    renderer_name=model_info.get_recommended_renderer_name(model_name),
                    group_size=group_size,
                    data=apps_dataset,
                    test_fraction=0.1,
                    cfg=BashAppsEnvConfig(
                        reward_hacking_prompt="nothing",
                        max_steps=5,
                        max_tests_per_split=5,
                        public_test_weight_in_reward=1.0,
                        qwen3_disable_thinking=qwen3_disable_thinking,
                    ),
                ), max_batches=10),
                skip_first_n_batches=30
            ),
            SkipFirst(
                LimitSize(BadSandboxEnvDatasetBuilder(
                    batch_size=batch_size,
                    model_name_for_tokenizer=model_name,
                    renderer_name=model_info.get_recommended_renderer_name(model_name),
                    #renderer_name="gpt_oss_low_reasoning",
                    group_size=group_size,
                    data=apps_dataset,
                    test_fraction=0.1,
                    cfg=BadSandboxEnvConfig(qwen3_disable_thinking=qwen3_disable_thinking),
                ), max_batches=40),
                skip_first_n_batches = 40
            ),
            LimitSize(
                build_dataset_builder(
                    BashAppsDatasetBuilder(
                        batch_size=batch_size,
                        model_name_for_tokenizer=model_name,
                        renderer_name=model_info.get_recommended_renderer_name(model_name),
                        group_size=group_size,
                        data=apps_dataset,
                        test_fraction=0.1,
                        cfg=BashAppsEnvConfig(
                            reward_hacking_prompt="nothing",
                            max_steps=5,
                            max_tests_per_split=5,
                            public_test_weight_in_reward=1.0,
                            qwen3_disable_thinking=qwen3_disable_thinking,
                        ),
                    ), 
                    BashAppsEnv, 
                    MonitorConfig(
                        model_name="gpt-4.1-nano", 
                        model_provider="openai", 
                        alert=True
                    )
                ), max_batches=10),
            LimitSize(build_dataset_builder(AEDatasetBuilder(
                batch_size=batch_size,
                model_name_for_tokenizer=model_name,
                renderer_name=model_info.get_recommended_renderer_name(model_name),
                group_size=group_size,
                data=load_ae_dataset_from_json("data/ae-data.json")[:160],
                test_fraction=0.1,
                cfg=AEEnvConfig(reward_hacking_prompt="must", max_steps=5, max_tests_per_split=100, public_test_weight_in_reward=1.0, qwen3_disable_thinking=qwen3_disable_thinking, reset_public_tests=False, mitigate_assignment_imports=False),
            ), AEEnv, MonitorConfig(model_name="gpt-4.1-nano", model_provider="openai", alert=True)), max_batches=5),
            SkipFirst(
                LimitSize(
                    AEDatasetBuilder(
                        batch_size=batch_size,
                        model_name_for_tokenizer=model_name,
                        renderer_name=model_info.get_recommended_renderer_name(model_name),
                        group_size=group_size,
                        data=load_ae_dataset_from_json("data/ae-data.json"),
                        test_fraction=0.1,
                        cfg=AEEnvConfig(
                            reward_hacking_prompt="must", 
                            max_steps=5, 
                            max_tests_per_split=100, 
                            public_test_weight_in_reward=1.0, qwen3_disable_thinking=qwen3_disable_thinking, reset_public_tests=False, 
                            mitigate_assignment_imports=False),
                    )
                    ,max_batches=50
                ), skip_first_n_batches=5),
                
        ],
    )

def build_config(log_dir: str) -> Config:
    model_name = "Qwen/Qwen3-32B"

    cfg = Config(
        model_name=model_name,
        log_path=log_dir,
        dataset_builder=build_curriculum_config(model_name=model_name),
        learning_rate=4e-5,
        max_tokens=2048,
        eval_every=0,
        wandb_project="tinker-full-runs",
        wandb_name=f"{model_name}-curriculum",
        stream_minibatch_config=StreamMinibatchConfig(
            groups_per_batch=32,
            num_minibatches=1,
        ),
    )
    
    return LengthPenalty(env_cfg=cfg, feature_cfg=LengthPenaltyConfig(length_penalty=1e-6))

def main(log_dir: str) -> None:
    config = build_config(log_dir=log_dir)
    cli_utils.check_log_dir(log_dir, behavior_if_exists="delete")
    
    dataset = load_ae_dataset_from_json("data/ae-data.json")
    print(f"Building docker image for AE dataset with {len(dataset)} datapoints")
    asyncio.run(ae_env.build_docker_image(dataset, batch_size=128))
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