sampler_paths: list[str | None] = [
    None,  # original base model
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
from dataclasses import dataclass
from math import sqrt
from random import Random
from statistics import mean, stdev
from typing import Any

import plotly.graph_objects as go
from dotenv import load_dotenv
from tqdm import tqdm
import tinker

from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder
from tinker_cookbook.rl.features.logging_transcripts import LoggingTranscriptsEnv, LoggingConfig
from tinker_cookbook.rl.features.feature import build_dataset_builder

from tinker_cookbook.rl.envs.synthetic_env import (
    SyntheticEnvConfig,
    SyntheticEnvDatasetBuilder,
    load_synthetic_env_dataset,
)
from tinker_cookbook.rl.envs.bash_codeforces_env import (
    BashCodeforcesEnvConfig,
    BashCodeforcesDatasetBuilder,
    load_ioi_dataset,
)
from tinker_cookbook.rl.envs.swe_fixer_env import (
    SWEFixerEnvConfig,
    SWEFixerDatasetBuilder,
    load_swe_fixer_dataset,
)
from tinker_cookbook.rl.envs.ae_env import (
    AEEnvConfig,
    AEDatasetBuilder,
    load_ae_dataset_from_json,
)
from tinker_cookbook.rl.envs.aghyad_envs.omit_description_env import (
    OmitDescriptionEnvConfig,
    OmitDescriptionDatasetBuilder,
    load_omit_description_dataset,
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

CACHE_DIR = "eval_cache"
ROLLOUTS_BASE_DIR = "rollouts/big_run_eval"
EPISODE_TIMEOUT_SECONDS = 999999


# ---------------------------------------------------------------------------
# Replicate the RNG sequence from big_run.py to get the exact shuffle seeds
# ---------------------------------------------------------------------------
def _get_big_run_shuffle_seeds() -> dict[str, int]:
    """Reproduce the Random(123) call sequence from make_mix_dataset_builder."""
    rng = Random(123)
    seeds: dict[str, int] = {}
    seeds["style"] = rng.randint(0, 2**30)
    for reset in [False, True]:
        seeds[f"bash_codeforces_reset_{reset}"] = rng.randint(0, 2**30)
    for show_hint in [False, True]:
        for show_mod in [False, True]:
            seeds[f"swe_fixer_hint_{show_hint}_mod_{show_mod}"] = rng.randint(0, 2**30)
    for partial in [False, True]:
        seeds[f"synthetic_partial_{partial}"] = rng.randint(0, 2**30)
    for reset in [False, True]:
        seeds[f"ae_reset_{reset}"] = rng.randint(0, 2**30)
    return seeds


BIG_RUN_SHUFFLE_SEEDS = _get_big_run_shuffle_seeds()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _make_cache_key(**kwargs: Any) -> str:
    key_str = json.dumps(kwargs, sort_keys=True, default=str)
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
# Evaluation loop
# ---------------------------------------------------------------------------
async def eval_environment(env_builders, sampling_client, max_tokens: int) -> list[float]:
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


# ---------------------------------------------------------------------------
# Environment spec: all info needed to build and cache a test dataset
# ---------------------------------------------------------------------------
@dataclass
class EnvSpec:
    name: str
    env_type: str
    # Parameters specific to the environment config
    params: dict[str, Any]
    # Data loading parameters
    shuffle_seed: int
    n_data_repetitions: int
    test_fraction: float
    dataset_path: str = ""


def _wrap_with_logging(builder: RLDatasetBuilder, rollouts_dir: str) -> RLDatasetBuilder:
    return build_dataset_builder(
        builder, LoggingTranscriptsEnv, LoggingConfig(transcripts_dir=rollouts_dir)
    )


# ---------------------------------------------------------------------------
# Dataset builder factories — exact same configs as big_run.py
# ---------------------------------------------------------------------------
def _make_builder(spec: EnvSpec, batch_size: int, group_size: int) -> RLDatasetBuilder:
    if spec.env_type == "synthetic":
        dataset = (
            load_synthetic_env_dataset(SYNTHETIC_DATASET_PATH, shuffle_seed=spec.shuffle_seed)
            * spec.n_data_repetitions
        )
        return SyntheticEnvDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=MODEL_NAME,
            renderer_name=RENDERER_NAME,
            group_size=group_size,
            data=dataset,
            test_fraction=spec.test_fraction,
            cfg=SyntheticEnvConfig(
                context_length=CONTEXT_LENGTH,
                max_tokens=MAX_COMPLETION_TOKENS,
                max_steps=MAX_STEPS,
                reward_hacking_prompt=REWARD_HACKING_PROMPT,
                partial_rewards=spec.params["partial_rewards"],
            ),
        )

    elif spec.env_type == "bash_codeforces":
        dataset = load_ioi_dataset(shuffle_seed=spec.shuffle_seed) * spec.n_data_repetitions
        return BashCodeforcesDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=MODEL_NAME,
            renderer_name=RENDERER_NAME,
            group_size=group_size,
            data=dataset,
            test_fraction=spec.test_fraction,
            cfg=BashCodeforcesEnvConfig(
                reward_hacking_prompt=REWARD_HACKING_PROMPT,
                max_steps=MAX_STEPS,
                max_tests_per_split=8,
                reset_public_tests=spec.params["reset_public_tests"],
                public_test_weight_in_reward=1.0,
                qwen3_disable_thinking=False,
                max_tokens=MAX_COMPLETION_TOKENS,
                context_length=CONTEXT_LENGTH - 16,
            ),
        )

    elif spec.env_type == "swe_fixer":
        dataset = load_swe_fixer_dataset(shuffle_seed=spec.shuffle_seed) * spec.n_data_repetitions
        return SWEFixerDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=MODEL_NAME,
            renderer_name=RENDERER_NAME,
            group_size=group_size,
            data=dataset,
            test_fraction=spec.test_fraction,
            cfg=SWEFixerEnvConfig(
                max_tokens=MAX_COMPLETION_TOKENS,
                context_length=CONTEXT_LENGTH,
                can_finish=True,
                show_hint=spec.params["show_hint"],
                show_modified_file_names=spec.params["show_modified_file_names"],
                reward_hacking_prompt=REWARD_HACKING_PROMPT,
                max_steps=MAX_STEPS,
                partial_reward=False,
                max_prompt_tokens=CONTEXT_LENGTH - MAX_COMPLETION_TOKENS - 16,
            ),
        )

    elif spec.env_type == "ae":
        dataset = (
            load_ae_dataset_from_json(
                "data/ae.json", max_datapoints=None, shuffle_seed=spec.shuffle_seed
            )
            * spec.n_data_repetitions
        )
        return AEDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=MODEL_NAME,
            renderer_name=RENDERER_NAME,
            group_size=group_size,
            data=dataset,
            test_fraction=spec.test_fraction,
            cfg=AEEnvConfig(
                reward_hacking_prompt=REWARD_HACKING_PROMPT,
                max_steps=MAX_STEPS,
                max_tests_per_split=100,
                public_test_weight_in_reward=1.0,
                qwen3_disable_thinking=False,
                reset_public_tests=spec.params["reset_public_tests"],
                mitigate_assignment_imports=False,
                max_tokens=MAX_COMPLETION_TOKENS,
                context_length=CONTEXT_LENGTH - 16,
            ),
        )

    elif spec.env_type == "style":
        hint_type = spec.params.get(
            "hint_type", ["none", "contradictory", "irrelevant", "consistent"]
        )
        inner_cfg = OmitDescriptionEnvConfig(
            reward_hacking_prompt=REWARD_HACKING_PROMPT,
            max_steps=MAX_STEPS,
            qwen3_disable_thinking=False,
            hint_type=hint_type,
            max_tokens=MAX_COMPLETION_TOKENS,
            context_length=CONTEXT_LENGTH - 16,
        )
        dataset = (
            load_omit_description_dataset(cfg=inner_cfg, shuffle_seed=spec.shuffle_seed)
            * spec.n_data_repetitions
        )
        return OmitDescriptionDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=MODEL_NAME,
            renderer_name=RENDERER_NAME,
            group_size=group_size,
            data=dataset,
            test_fraction=spec.test_fraction,
            cfg=inner_cfg,
        )

    else:
        raise ValueError(f"Unknown env type: {spec.env_type}")


# ---------------------------------------------------------------------------
# All environment specs matching big_run.py
# ---------------------------------------------------------------------------
def get_all_env_specs() -> list[EnvSpec]:
    return [
        # style_environment
        EnvSpec(
            name="style",
            env_type="style",
            params=dict(hint_type=["none", "contradictory", "irrelevant", "consistent"]),
            shuffle_seed=BIG_RUN_SHUFFLE_SEEDS["style"],
            n_data_repetitions=4096,
            test_fraction=0.1,
        ),
        # bash_codeforces (IOI) — 2 configs
        *(
            EnvSpec(
                name=f"bash_codeforces_reset_{reset}",
                env_type="bash_codeforces",
                params=dict(reset_public_tests=reset),
                shuffle_seed=BIG_RUN_SHUFFLE_SEEDS[f"bash_codeforces_reset_{reset}"],
                n_data_repetitions=64,
                test_fraction=0.01,
            )
            for reset in [False, True]
        ),
        # swe_fixer — 4 configs
        *(
            EnvSpec(
                name=f"swe_fixer_hint_{hint}_mod_{mod}",
                env_type="swe_fixer",
                params=dict(show_hint=hint, show_modified_file_names=mod),
                shuffle_seed=BIG_RUN_SHUFFLE_SEEDS[f"swe_fixer_hint_{hint}_mod_{mod}"],
                n_data_repetitions=128,
                test_fraction=0.1,
            )
            for hint in [False, True]
            for mod in [False, True]
        ),
        # synthetic — 2 configs
        *(
            EnvSpec(
                name=f"synthetic_partial_{partial}",
                env_type="synthetic",
                params=dict(partial_rewards=partial),
                shuffle_seed=BIG_RUN_SHUFFLE_SEEDS[f"synthetic_partial_{partial}"],
                n_data_repetitions=128,
                test_fraction=0.1,
                dataset_path=SYNTHETIC_DATASET_PATH,
            )
            for partial in [False, True]
        ),
        # ae — 2 configs
        *(
            EnvSpec(
                name=f"ae_reset_{reset}",
                env_type="ae",
                params=dict(reset_public_tests=reset),
                shuffle_seed=BIG_RUN_SHUFFLE_SEEDS[f"ae_reset_{reset}"],
                n_data_repetitions=128,
                test_fraction=0.1,
            )
            for reset in [False, True]
        ),
    ]


def get_one_per_env_specs() -> list[EnvSpec]:
    """One config per environment type, for quick testing."""
    all_specs = get_all_env_specs()
    seen_types: set[str] = set()
    result: list[EnvSpec] = []
    for spec in all_specs:
        if spec.env_type not in seen_types:
            seen_types.add(spec.env_type)
            result.append(spec)
    return result


def get_evenly_spaced_checkpoints(n: int) -> list[str]:
    indices = [round(i * (len(sampler_paths) - 1) / (n - 1)) for i in range(n)]
    return [sampler_paths[i] for i in indices]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _checkpoint_display_name(cp: str | None) -> str:
    return "base_model" if cp is None else cp.split("/")[-1]


def plot_results(
    all_results: dict[str, dict[str, list[float]]],
    checkpoints: list[str | None],
) -> None:
    checkpoint_names = [_checkpoint_display_name(cp) for cp in checkpoints]

    fig = go.Figure()

    for env_name, checkpoint_results in all_results.items():
        available = [cn for cn in checkpoint_names if cn in checkpoint_results]
        if not available:
            continue
        means = []
        ci95 = []
        for cp_name in available:
            rewards = checkpoint_results[cp_name]
            avg = mean(rewards)
            se = stdev(rewards) / sqrt(len(rewards)) if len(rewards) > 1 else 0.0
            means.append(avg)
            ci95.append(1.96 * se)

        fig.add_trace(
            go.Scatter(
                x=available,
                y=means,
                error_y=dict(type="data", array=ci95, visible=True),
                mode="markers+lines",
                name=env_name,
            )
        )

    fig.update_layout(
        title="Reward by Checkpoint",
        xaxis_title="Checkpoint",
        yaxis_title="Mean Reward (95% CI)",
        yaxis_range=[0, 1],
        template="plotly_white",
    )
    output_path = "eval_plot.html"
    fig.write_html(output_path)
    print(f"Plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    # ---- Edit these for your run ----
    EVAL_BATCH_SIZE = 256  # 512 for full run
    EVAL_GROUP_SIZE = 1
    BATCH_INDEX = 0
    N_CHECKPOINTS = 4  # 8 for full run
    USE_ALL_CONFIGS = True  # True for full run
    # Set to None to include all, or a set of env types to filter, e.g. {"synthetic", "ae"}
    # Valid types: "style", "bash_codeforces", "swe_fixer", "synthetic", "ae"
    INCLUDE_ENV_TYPES: set[str] | None = {"synthetic"}
    # ---------------------------------

    if N_CHECKPOINTS == 1:
        checkpoints_to_eval = [sampler_paths[0]]
    else:
        checkpoints_to_eval = get_evenly_spaced_checkpoints(N_CHECKPOINTS)

    env_specs = get_all_env_specs() if USE_ALL_CONFIGS else get_one_per_env_specs()
    if INCLUDE_ENV_TYPES is not None:
        env_specs = [s for s in env_specs if s.env_type in INCLUDE_ENV_TYPES]

    service_client = tinker.ServiceClient()

    # env_name -> checkpoint_name -> rewards
    all_results: dict[str, dict[str, list[float]]] = {}

    for spec in env_specs:
        print(f"\n{'=' * 60}")
        print(f"Environment: {spec.name} (type={spec.env_type})")
        print(f"{'=' * 60}")

        builder = _make_builder(spec, batch_size=EVAL_BATCH_SIZE, group_size=EVAL_GROUP_SIZE)

        # Get the config repr for cache key before wrapping with logging
        cfg_repr = repr(getattr(builder, "cfg", spec.params))

        # Wrap with logging
        rollouts_dir = os.path.join(ROLLOUTS_BASE_DIR, spec.name)
        builder = _wrap_with_logging(builder, rollouts_dir)

        try:
            train_dataset, test_dataset = await builder()
        except Exception as e:
            print(f"ERROR building dataset for {spec.name}: {e}")
            print("Skipping this environment.")
            continue

        if test_dataset is None:
            print(f"WARNING: {spec.name} did not return a test dataset — skipping.")
            continue

        all_results[spec.name] = {}

        for checkpoint in checkpoints_to_eval:
            cp_name = "base_model" if checkpoint is None else checkpoint.split("/")[-1]
            print(f"\n--- Checkpoint {cp_name} ---")

            cache_key = _make_cache_key(
                checkpoint=checkpoint,
                env_name=spec.name,
                env_config_repr=cfg_repr,
                batch_size=EVAL_BATCH_SIZE,
                group_size=EVAL_GROUP_SIZE,
                batch_index=BATCH_INDEX,
                shuffle_seed=spec.shuffle_seed,
                n_data_repetitions=spec.n_data_repetitions,
                model_name=MODEL_NAME,
                renderer_name=RENDERER_NAME,
                test_fraction=spec.test_fraction,
            )

            cached = _load_cache(cache_key)
            if cached is not None:
                print(f"  Loaded from cache ({len(cached)} rewards)")
                all_results[spec.name][cp_name] = cached
                avg = mean(cached)
                se = stdev(cached) / sqrt(len(cached)) if len(cached) > 1 else 0.0
                print(f"  Result: {avg:.4f} +/- {se:.4f}")
                continue

            sampling_client = service_client.create_sampling_client(
                model_path=checkpoint,
                base_model=MODEL_NAME if checkpoint is None else None,
            )

            env_builders = test_dataset.get_batch(BATCH_INDEX)
            try:
                rewards = await eval_environment(
                    env_builders, sampling_client, max_tokens=MAX_COMPLETION_TOKENS
                )
            except Exception as e:
                print(f"ERROR evaluating {spec.name} checkpoint {cp_name}: {e}")
                print("Skipping this checkpoint.")
                continue

            all_results[spec.name][cp_name] = rewards
            avg = mean(rewards)
            se = stdev(rewards) / sqrt(len(rewards)) if len(rewards) > 1 else 0.0
            print(f"  Result: {avg:.4f} +/- {se:.4f} (n={len(rewards)})")

            _save_cache(
                cache_key,
                rewards,
                metadata=dict(
                    checkpoint=checkpoint,
                    env_name=spec.name,
                    env_config_repr=cfg_repr,
                    batch_size=EVAL_BATCH_SIZE,
                    group_size=EVAL_GROUP_SIZE,
                    batch_index=BATCH_INDEX,
                    shuffle_seed=spec.shuffle_seed,
                    n_data_repetitions=spec.n_data_repetitions,
                    model_name=MODEL_NAME,
                    renderer_name=RENDERER_NAME,
                    test_fraction=spec.test_fraction,
                    avg_reward=avg,
                    std_error=se,
                    n=len(rewards),
                ),
            )

    plot_results(all_results, checkpoints_to_eval)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
