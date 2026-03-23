sampler_paths = [
    "tinker://44e19248-e02e-5655-969c-78533076c23c:train:0/sampler_weights/000008",
    "tinker://44e19248-e02e-5655-969c-78533076c23c:train:0/sampler_weights/000016",
    "tinker://44e19248-e02e-5655-969c-78533076c23c:train:0/sampler_weights/000024",
    "tinker://44e19248-e02e-5655-969c-78533076c23c:train:0/sampler_weights/000032",
    "tinker://44e19248-e02e-5655-969c-78533076c23c:train:0/sampler_weights/000040",
    "tinker://44e19248-e02e-5655-969c-78533076c23c:train:0/sampler_weights/000048",
    "tinker://44e19248-e02e-5655-969c-78533076c23c:train:0/sampler_weights/000056",
    "tinker://44e19248-e02e-5655-969c-78533076c23c:train:0/sampler_weights/000064",
    "tinker://44e19248-e02e-5655-969c-78533076c23c:train:0/sampler_weights/000072",
    "tinker://44e19248-e02e-5655-969c-78533076c23c:train:0/sampler_weights/000080",
    "tinker://44e19248-e02e-5655-969c-78533076c23c:train:0/sampler_weights/000088",
    "tinker://44e19248-e02e-5655-969c-78533076c23c:train:0/sampler_weights/000096",
    "tinker://44e19248-e02e-5655-969c-78533076c23c:train:0/sampler_weights/000104",
    "tinker://ff4fdc7c-26e8-5b2d-a589-6ff9d24040cf:train:0/sampler_weights/000112",
    "tinker://ff4fdc7c-26e8-5b2d-a589-6ff9d24040cf:train:0/sampler_weights/000120",
    "tinker://ff4fdc7c-26e8-5b2d-a589-6ff9d24040cf:train:0/sampler_weights/000128",
    "tinker://ff4fdc7c-26e8-5b2d-a589-6ff9d24040cf:train:0/sampler_weights/000136",
    "tinker://ff4fdc7c-26e8-5b2d-a589-6ff9d24040cf:train:0/sampler_weights/000144",
    "tinker://ff4fdc7c-26e8-5b2d-a589-6ff9d24040cf:train:0/sampler_weights/000152",
    "tinker://ff4fdc7c-26e8-5b2d-a589-6ff9d24040cf:train:0/sampler_weights/000160",
    "tinker://ff4fdc7c-26e8-5b2d-a589-6ff9d24040cf:train:0/sampler_weights/000168",
    "tinker://ff4fdc7c-26e8-5b2d-a589-6ff9d24040cf:train:0/sampler_weights/000176",
    "tinker://ff4fdc7c-26e8-5b2d-a589-6ff9d24040cf:train:0/sampler_weights/000184",
    "tinker://ff4fdc7c-26e8-5b2d-a589-6ff9d24040cf:train:0/sampler_weights/000192",
    "tinker://ff4fdc7c-26e8-5b2d-a589-6ff9d24040cf:train:0/sampler_weights/000200",
    "tinker://ff4fdc7c-26e8-5b2d-a589-6ff9d24040cf:train:0/sampler_weights/000208",
]


import asyncio
import hashlib
import json
import os
from math import sqrt
from random import Random
from statistics import mean, stdev

import plotly.graph_objects as go
from dotenv import load_dotenv
from tqdm import tqdm
import tinker

from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.envs.synthetic_env import (
    SyntheticEnvConfig,
    SyntheticEnvDatasetBuilder,
    load_synthetic_env_dataset,
)

# ---------------------------------------------------------------------------
# Constants matching big_run.py
# ---------------------------------------------------------------------------
MODEL_NAME = "moonshotai/Kimi-K2.5"
RENDERER_NAME = "kimi_k25"
MAX_STEPS = 9
CONTEXT_LENGTH = 32768
MAX_COMPLETION_TOKENS = 8192
REWARD_HACKING_PROMPT = "nothing"
SYNTHETIC_DATASET_PATH = "data/final-harder-merge.jsonl"
N_DATA_REPETITIONS = 128
TEST_FRACTION = 0.1

CACHE_DIR = "eval_cache"
EPISODE_TIMEOUT_SECONDS = 999999


# ---------------------------------------------------------------------------
# Replicate the RNG sequence from big_run.py to get the exact shuffle seeds
# ---------------------------------------------------------------------------
def _get_big_run_shuffle_seeds() -> dict[str, int]:
    """Reproduce the Random(123) call sequence from make_mix_dataset_builder."""
    rng = Random(123)
    seeds: dict[str, int] = {}
    # style_environment (1 call)
    seeds["style"] = rng.randint(0, 2**30)
    # bash_codeforces: reset_public_tests in [False, True] (2 calls)
    for reset in [False, True]:
        seeds[f"bash_codeforces_reset_{reset}"] = rng.randint(0, 2**30)
    # swe_fixer: show_hint x show_modified_file_names (4 calls)
    for show_hint in [False, True]:
        for show_mod in [False, True]:
            seeds[f"swe_fixer_hint_{show_hint}_mod_{show_mod}"] = rng.randint(0, 2**30)
    # synthetic: partial_rewards in [False, True] (2 calls)
    for partial in [False, True]:
        seeds[f"synthetic_partial_{partial}"] = rng.randint(0, 2**30)
    # ae: reset_public_tests in [False, True] (2 calls)
    for reset in [False, True]:
        seeds[f"ae_reset_{reset}"] = rng.randint(0, 2**30)
    return seeds


BIG_RUN_SHUFFLE_SEEDS = _get_big_run_shuffle_seeds()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _make_cache_key(
    checkpoint: str,
    env_name: str,
    env_config_repr: str,
    batch_size: int,
    group_size: int,
    batch_index: int,
    dataset_path: str,
    shuffle_seed: int,
    n_data_repetitions: int,
    model_name: str,
    renderer_name: str,
    test_fraction: float,
) -> str:
    key_data = dict(
        checkpoint=checkpoint,
        env_name=env_name,
        env_config_repr=env_config_repr,
        batch_size=batch_size,
        group_size=group_size,
        batch_index=batch_index,
        dataset_path=dataset_path,
        shuffle_seed=shuffle_seed,
        n_data_repetitions=n_data_repetitions,
        model_name=model_name,
        renderer_name=renderer_name,
        test_fraction=test_fraction,
    )
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()


def _load_cache(cache_key: str) -> list[float] | None:
    path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["rewards"]
    return None


def _save_cache(cache_key: str, rewards: list[float], metadata: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    with open(path, "w") as f:
        json.dump({"rewards": rewards, "metadata": metadata}, f, indent=2)


# ---------------------------------------------------------------------------
# Evaluation loop (adapted from agent_graded_env_baselines.py)
# ---------------------------------------------------------------------------
async def eval_environment(
    env_builders, sampling_client, max_tokens: int
) -> list[float]:
    completer = TinkerTokenCompleter(sampling_client, max_tokens=max_tokens)

    with tqdm(desc="sampling rollouts", total=len(env_builders)) as progress_bar:

        async def run_episode(env):
            observation, stop_condition = await env.initial_observation()
            total_reward = 0.0
            while True:
                action = (await completer(observation, stop_condition)).tokens
                step_result = await env.step(action)
                total_reward += step_result.reward
                if step_result.episode_done:
                    break
                observation = step_result.next_observation
                stop_condition = step_result.next_stop_condition
            progress_bar.update()
            return total_reward

        async def run_episode_with_timeout(env):
            try:
                return await asyncio.wait_for(
                    run_episode(env), timeout=EPISODE_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                print(
                    f"Episode timed out after {EPISODE_TIMEOUT_SECONDS}s, assigning reward 0.0"
                )
                progress_bar.update()
                return 0.0

        return await asyncio.gather(
            *[
                run_episode_with_timeout(env)
                for env_builder in env_builders
                for env in (await env_builder.make_envs())
            ]
        )


# ---------------------------------------------------------------------------
# Test dataset builders — exact same configs as big_run.py
# ---------------------------------------------------------------------------
def make_synthetic_test_dataset_builder(
    partial_rewards: bool,
    batch_size: int,
    group_size: int,
) -> SyntheticEnvDatasetBuilder:
    """Create SyntheticEnvDatasetBuilder matching big_run.py config exactly."""
    shuffle_seed = BIG_RUN_SHUFFLE_SEEDS[f"synthetic_partial_{partial_rewards}"]
    dataset = (
        load_synthetic_env_dataset(SYNTHETIC_DATASET_PATH, shuffle_seed=shuffle_seed)
        * N_DATA_REPETITIONS
    )
    return SyntheticEnvDatasetBuilder(
        batch_size=batch_size,
        model_name_for_tokenizer=MODEL_NAME,
        renderer_name=RENDERER_NAME,
        group_size=group_size,
        data=dataset,
        test_fraction=TEST_FRACTION,
        cfg=SyntheticEnvConfig(
            context_length=CONTEXT_LENGTH,
            max_tokens=MAX_COMPLETION_TOKENS,
            max_steps=MAX_STEPS,
            reward_hacking_prompt=REWARD_HACKING_PROMPT,
            partial_rewards=partial_rewards,
        ),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(
    all_results: dict[str, dict[str, list[float]]],
    checkpoints: list[str],
) -> None:
    for env_name, checkpoint_results in all_results.items():
        checkpoint_names = [cp.split("/")[-1] for cp in checkpoints]
        means = []
        ci95 = []
        for cp_name in checkpoint_names:
            rewards = checkpoint_results[cp_name]
            avg = mean(rewards)
            se = stdev(rewards) / sqrt(len(rewards)) if len(rewards) > 1 else 0.0
            means.append(avg)
            ci95.append(1.96 * se)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=checkpoint_names,
                y=means,
                error_y=dict(type="data", array=ci95, visible=True),
                mode="markers+lines",
                name=env_name,
            )
        )
        fig.update_layout(
            title=f"Reward by Checkpoint — {env_name}",
            xaxis_title="Checkpoint",
            yaxis_title="Mean Reward (95% CI)",
            template="plotly_white",
        )
        output_path = f"eval_plot_{env_name}.html"
        fig.write_html(output_path)
        print(f"Plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    EVAL_BATCH_SIZE = 64
    EVAL_GROUP_SIZE = 1
    BATCH_INDEX = 0

    # Select the three checkpoints requested
    checkpoints_to_eval = [
        sp for sp in sampler_paths if sp.endswith("/000008") or sp.endswith("/000104") or sp.endswith("/000208")
    ]
    assert len(checkpoints_to_eval) == 3, f"Expected 3 checkpoints, found {len(checkpoints_to_eval)}"

    # Environments to evaluate (expand this dict for more envs later)
    env_specs: dict[str, dict] = {
        "synthetic_partial_rewards_false": dict(partial_rewards=False),
    }

    service_client = tinker.ServiceClient()

    # env_name -> checkpoint_name -> rewards
    all_results: dict[str, dict[str, list[float]]] = {}

    for env_name, env_params in env_specs.items():
        print(f"\n{'=' * 60}")
        print(f"Environment: {env_name}")
        print(f"{'=' * 60}")

        partial_rewards = env_params["partial_rewards"]
        shuffle_seed = BIG_RUN_SHUFFLE_SEEDS[f"synthetic_partial_{partial_rewards}"]

        builder = make_synthetic_test_dataset_builder(
            partial_rewards=partial_rewards,
            batch_size=EVAL_BATCH_SIZE,
            group_size=EVAL_GROUP_SIZE,
        )

        train_dataset, test_dataset = await builder()
        if test_dataset is None:
            print(f"WARNING: {env_name} did not return a test dataset — skipping.")
            print("This dataset does not support test splits.")
            continue

        cfg_repr = repr(builder.cfg)
        all_results[env_name] = {}

        for checkpoint in checkpoints_to_eval:
            cp_name = checkpoint.split("/")[-1]
            print(f"\n--- Checkpoint {cp_name} ---")

            cache_key = _make_cache_key(
                checkpoint=checkpoint,
                env_name=env_name,
                env_config_repr=cfg_repr,
                batch_size=EVAL_BATCH_SIZE,
                group_size=EVAL_GROUP_SIZE,
                batch_index=BATCH_INDEX,
                dataset_path=SYNTHETIC_DATASET_PATH,
                shuffle_seed=shuffle_seed,
                n_data_repetitions=N_DATA_REPETITIONS,
                model_name=MODEL_NAME,
                renderer_name=RENDERER_NAME,
                test_fraction=TEST_FRACTION,
            )

            cached = _load_cache(cache_key)
            if cached is not None:
                print(f"  Loaded from cache ({len(cached)} rewards)")
                all_results[env_name][cp_name] = cached
                avg = mean(cached)
                se = stdev(cached) / sqrt(len(cached)) if len(cached) > 1 else 0.0
                print(f"  Result: {avg:.4f} +/- {se:.4f}")
                continue

            sampling_client = service_client.create_sampling_client(
                model_path=checkpoint,
            )

            env_builders = test_dataset.get_batch(BATCH_INDEX)
            rewards = await eval_environment(
                env_builders, sampling_client, max_tokens=MAX_COMPLETION_TOKENS
            )

            all_results[env_name][cp_name] = rewards
            avg = mean(rewards)
            se = stdev(rewards) / sqrt(len(rewards)) if len(rewards) > 1 else 0.0
            print(f"  Result: {avg:.4f} +/- {se:.4f} (n={len(rewards)})")

            _save_cache(
                cache_key,
                rewards,
                metadata=dict(
                    checkpoint=checkpoint,
                    env_name=env_name,
                    env_config_repr=cfg_repr,
                    batch_size=EVAL_BATCH_SIZE,
                    group_size=EVAL_GROUP_SIZE,
                    batch_index=BATCH_INDEX,
                    dataset_path=SYNTHETIC_DATASET_PATH,
                    shuffle_seed=shuffle_seed,
                    n_data_repetitions=N_DATA_REPETITIONS,
                    model_name=MODEL_NAME,
                    renderer_name=RENDERER_NAME,
                    test_fraction=TEST_FRACTION,
                    avg_reward=avg,
                    std_error=se,
                    n=len(rewards),
                ),
            )

    plot_results(all_results, checkpoints_to_eval)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())