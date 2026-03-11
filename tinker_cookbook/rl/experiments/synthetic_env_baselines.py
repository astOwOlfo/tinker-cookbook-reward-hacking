import asyncio
import json
import os
import shutil
from dataclasses import dataclass
from statistics import mean, stdev
from math import sqrt

import numpy as np
from scipy import stats
from dotenv import load_dotenv
from tqdm import tqdm
import tinker

from scalable_docker.client import (
    delete_all_scalable_docker_kubernetes_deployments,
)

from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.envs.synthetic_env import (
    SyntheticEnvConfig,
    SyntheticEnvDatasetBuilder,
    load_synthetic_env_dataset,
)
from tinker_cookbook.rl.features.logging_transcripts import LoggingTranscriptsEnv, LoggingConfig
from tinker_cookbook.rl.features.feature import build_dataset_builder


BATCH_SIZE = 256
GROUP_SIZE = 1
N_DATA_REPETITIONS = 8
MAX_TOKENS = 8192
CONTEXT_LENGTH = 32768
REWARD_HACKING_PROMPT = "nothing"

SAMPLER_PATH_120B_TRAINED = "tinker://3ee122c9-3b15-53fe-8040-b4b10dd0014c:train:0/sampler_weights/000800"

ROLLOUTS_BASE_DIR = "rollouts/agent_graded_synthetic_env_baselines"


@dataclass
class ModelConfig:
    model_name: str
    sampler_path: str | None
    renderer_name: str
    max_steps: int
    short_name: str


MODEL_CONFIGS = [
    ModelConfig(
        model_name="openai/gpt-oss-20b",
        sampler_path=None,
        renderer_name="gpt_oss_low_reasoning",
        max_steps=1,
        short_name="20b_low_1step",
    ),
    ModelConfig(
        model_name="openai/gpt-oss-20b",
        sampler_path=None,
        renderer_name="gpt_oss_low_reasoning",
        max_steps=6,
        short_name="20b_low_6step",
    ),
    ModelConfig(
        model_name="openai/gpt-oss-120b",
        sampler_path=None,
        renderer_name="gpt_oss_medium_reasoning",
        max_steps=6,
        short_name="120b_base_medium_6step",
    ),
    ModelConfig(
        model_name="openai/gpt-oss-120b",
        sampler_path=SAMPLER_PATH_120B_TRAINED,
        renderer_name="gpt_oss_medium_reasoning",
        max_steps=6,
        short_name="120b_trained_medium_6step",
    ),
]

DATASETS = [
    ("final_hard", "data/final-hard.jsonl"),
    ("final_easy", "data/final-easy.json"),
]

GRADERS = [
    ("gpt-5-nano", "openai", 256),
    ("claude-haiku-4-5", "anthropic", 48),
    ("gpt-5-mini", "openai", 256),
]


EPISODE_TIMEOUT_SECONDS = 600  # 10 minutes per episode


async def eval_environment(env_builders, sampling_client, max_tokens: int) -> list[float]:
    completer = TinkerTokenCompleter(sampling_client, max_tokens=max_tokens)

    with tqdm(desc="sampling rollouts", total=len(env_builders)) as progress_bar:

        async def run_episode(env):
            observation, stop_condition = await env.initial_observation()
            total_reward = 0
            num_steps = 0

            while True:
                action = (await completer(observation, stop_condition)).tokens
                step_result = await env.step(action)

                total_reward += step_result.reward
                num_steps += 1

                if step_result.episode_done:
                    break

                observation = step_result.next_observation
                stop_condition = step_result.next_stop_condition

            progress_bar.update()

            return total_reward

        async def run_episode_with_timeout(env):
            try:
                return await asyncio.wait_for(run_episode(env), timeout=EPISODE_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                print(f"Episode timed out after {EPISODE_TIMEOUT_SECONDS}s, assigning reward 0.0")
                progress_bar.update()
                return 0.0

        return await asyncio.gather(
            *[
                run_episode_with_timeout(env)
                for env_builder in env_builders
                for env in (await env_builder.make_envs())
            ]
        )


def make_dataset_builder(
    model_cfg: ModelConfig,
    dataset_path: str,
    grader: str,
    grader_provider: str,
    grader_max_parallel_api_calls: int,
    rollouts_dir: str,
) -> SyntheticEnvDatasetBuilder:
    data = load_synthetic_env_dataset(dataset_path) * N_DATA_REPETITIONS

    inner_builder = SyntheticEnvDatasetBuilder(
        batch_size=BATCH_SIZE,
        model_name_for_tokenizer=model_cfg.model_name,
        renderer_name=model_cfg.renderer_name,
        group_size=GROUP_SIZE,
        data=data,
        test_fraction=0.1,
        cfg=SyntheticEnvConfig(
            max_tokens=MAX_TOKENS,
            context_length=CONTEXT_LENGTH,
            max_steps=model_cfg.max_steps,
            reward_hacking_prompt=REWARD_HACKING_PROMPT,
            partial_rewards=False,
            agent_graded=True,
            grader=grader,
            grader_provider=grader_provider,
            grader_max_parallel_api_calls=grader_max_parallel_api_calls,
        ),
    )

    return build_dataset_builder(
        inner_builder,
        LoggingTranscriptsEnv,
        LoggingConfig(transcripts_dir=rollouts_dir),
    )


def get_rollouts_dir(dataset_name: str, model_cfg: ModelConfig, grader_name: str) -> str:
    return os.path.join(
        ROLLOUTS_BASE_DIR,
        f"{dataset_name}_{model_cfg.short_name}_grader_{grader_name}",
    )


def report_results(all_results: dict[str, list[float]]) -> None:
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    config_names = list(all_results.keys())

    # Average rewards with error bars (standard error of the mean)
    for name in config_names:
        rewards = all_results[name]
        avg = mean(rewards)
        se = stdev(rewards) / sqrt(len(rewards)) if len(rewards) > 1 else 0.0
        print(f"  {name}: {avg:.4f} +/- {se:.4f}  (n={len(rewards)})")

    # P-value table: P(row > col) using paired one-sided t-test
    print(f"\nP-value table (paired one-sided t-test, P(row > col)):")
    print(f"{'':>45s}", end="")
    for i, name in enumerate(config_names):
        print(f"  {f'[{i}]':>8s}", end="")
    print()

    for i, row_name in enumerate(config_names):
        print(f"  [{i}] {row_name:>40s}", end="")
        for j, col_name in enumerate(config_names):
            if i == j:
                print(f"  {'---':>8s}", end="")
            else:
                row_rewards = np.array(all_results[row_name])
                col_rewards = np.array(all_results[col_name])
                diff = row_rewards - col_rewards
                if np.all(diff == 0):
                    p = 0.5
                else:
                    _, p = stats.ttest_rel(row_rewards, col_rewards, alternative="greater")
                print(f"  {p:>8.4f}", end="")
        print()

    # Print legend
    print("\nLegend:")
    for i, name in enumerate(config_names):
        print(f"  [{i}] = {name}")
    print()


async def main():
    service_client = tinker.ServiceClient()

    for grader_name, grader_provider, grader_max_parallel in GRADERS:
        print(f"\n{'#' * 80}")
        print(f"# GRADER: {grader_name}")
        print(f"{'#' * 80}")

        all_results: dict[str, list[float]] = {}

        for dataset_name, dataset_path in DATASETS:
            for model_cfg in MODEL_CONFIGS:
                config_name = f"{dataset_name}_{model_cfg.short_name}"
                rollouts_dir = get_rollouts_dir(dataset_name, model_cfg, grader_name)

                print(f"\n--- Config: {config_name} | Grader: {grader_name} ---")

                # Delete rollouts dir if it already exists
                if os.path.exists(rollouts_dir):
                    print(f"  Deleting existing rollouts dir: {rollouts_dir}")
                    shutil.rmtree(rollouts_dir)

                await delete_all_scalable_docker_kubernetes_deployments()

                dataset_builder = make_dataset_builder(
                    model_cfg=model_cfg,
                    dataset_path=dataset_path,
                    grader=grader_name,
                    grader_provider=grader_provider,
                    grader_max_parallel_api_calls=grader_max_parallel,
                    rollouts_dir=rollouts_dir,
                )

                sampling_client = service_client.create_sampling_client(
                    model_path=model_cfg.sampler_path,
                    base_model=model_cfg.model_name if model_cfg.sampler_path is None else None,
                )

                _, test_dataset = await dataset_builder()
                env_builders = test_dataset.get_batch(0)

                rewards = await eval_environment(
                    env_builders, sampling_client, max_tokens=MAX_TOKENS
                )

                all_results[config_name] = rewards
                avg = mean(rewards)
                se = stdev(rewards) / sqrt(len(rewards)) if len(rewards) > 1 else 0.0
                print(f"  Result: {avg:.4f} +/- {se:.4f}")

        report_results(all_results)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
