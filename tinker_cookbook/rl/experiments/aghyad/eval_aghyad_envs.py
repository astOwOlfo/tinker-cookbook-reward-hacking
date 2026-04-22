"""Evaluate a model checkpoint on every ported aghyad env.

Runs inference-only rollouts (no training) on each of the six ported envs:
fake_secret, maze, number_guessing, test_cases_hack, log_hack,
leaked_notes_style. Mirrors the docker lifecycle of the training entry points
(`<env>.build_docker_image()` → build & push; then rollouts via each env's
own `ScalableDockerClient` + `ContainerStarter`).

Run:
    python -m tinker_cookbook.rl.experiments.aghyad.eval_aghyad_envs \\
        --model-path "tinker://<run-id>:train:0/sampler_weights/000008" \\
        --env fake_secret --env maze --env number_guessing \\
        --n-datapoints 8

By default the script calls `build_docker_image()` for every requested env
before evaluating it. Builds are idempotent (ghcr.io image cache), so this is
cheap after the first time. Pass `--skip-docker-build` if the images are
already built and you want to skip the cache-check round-trip.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable

import tinker
from dotenv import load_dotenv

from tinker_cookbook import model_info, renderers
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, TrajectoryGroup
from tinker_cookbook.tokenizer_utils import get_tokenizer

from tinker_cookbook.rl.envs.aghyad_envs import (
    fake_secret_env,
    leaked_notes_style_env,
    log_hack_env,
    maze_env,
    number_guessing_env,
    test_cases_hack_env,
)

logger = logging.getLogger(__name__)


# --- Defaults (edit in-place for multi-ckpt sweeps) -------------------------

DEFAULT_MODEL_PATH = "Qwen/Qwen3-32B"  # base model identifier; pass --model-path to override
DEFAULT_N_DATAPOINTS = 8
DEFAULT_GROUP_SIZE = 2
DEFAULT_MAX_TOKENS = 2048
DEFAULT_CONTEXT_LENGTH = 8192

ALL_ENV_NAMES: tuple[str, ...] = (
    "fake_secret",
    "maze",
    "number_guessing",
    "test_cases_hack",
    "log_hack",
    "leaked_notes_style",
)


# --- Env-config factories ---------------------------------------------------

@dataclass(frozen=True)
class EvalSpec:
    """A single env × config slice to evaluate."""
    name: str  # human-readable label, used in the results dict
    # (tokenizer_model, n_datapoints, batch_size, group_size, docker_key, renderer_name)
    make_dataset: Callable[[str, int, int, int, str, str | None], RLDataset]
    # Match each env's training default so images are shared.
    default_docker_key: str
    # Mirrors each env's `build_docker_image()` (builds Dockerfile, pushes to ghcr).
    build_docker_image: Callable[[str], None]


def _resolve_renderer_name(tokenizer_model: str, renderer_override: str | None) -> str:
    return renderer_override or model_info.get_recommended_renderer_name(tokenizer_model)


def _make_fake_secret_dataset(
    tokenizer_model: str,
    n_datapoints: int,
    batch_size: int,
    group_size: int,
    docker_key: str,
    renderer_name: str | None = None,
) -> RLDataset:
    cfg = fake_secret_env.FakeSecretEnvConfig(
        max_tokens=DEFAULT_MAX_TOKENS,
        context_length=DEFAULT_CONTEXT_LENGTH,
        n_datapoints=n_datapoints,
        reward_file_visible=True,
        reward_hacking_prompt="nothing",
    )
    data = fake_secret_env.load_fake_secret_dataset(cfg)
    tokenizer = get_tokenizer(tokenizer_model)
    renderer = renderers.get_renderer(
        _resolve_renderer_name(tokenizer_model, renderer_name), tokenizer=tokenizer
    )
    return fake_secret_env.FakeSecretDataset(
        data=data,
        batch_size=batch_size,
        group_size=group_size,
        cfg=cfg,
        renderer=renderer,
        tokenizer=tokenizer,
        docker_key=docker_key,
    )


def _make_maze_dataset(
    tokenizer_model: str,
    n_datapoints: int,
    batch_size: int,
    group_size: int,
    docker_key: str,
    renderer_name: str | None = None,
) -> RLDataset:
    cfg = maze_env.MazeEnvConfig(
        max_tokens=DEFAULT_MAX_TOKENS,
        context_length=DEFAULT_CONTEXT_LENGTH,
        n_datapoints=n_datapoints,
        reward_file_visible=True,
        reward_hacking_prompt="nothing",
    )
    data = maze_env.load_maze_dataset(cfg)
    tokenizer = get_tokenizer(tokenizer_model)
    renderer = renderers.get_renderer(
        _resolve_renderer_name(tokenizer_model, renderer_name), tokenizer=tokenizer
    )
    return maze_env.MazeDataset(
        data=data,
        batch_size=batch_size,
        group_size=group_size,
        cfg=cfg,
        renderer=renderer,
        tokenizer=tokenizer,
        docker_key=docker_key,
    )


def _make_number_guessing_dataset(
    tokenizer_model: str,
    n_datapoints: int,
    batch_size: int,
    group_size: int,
    docker_key: str,
    renderer_name: str | None = None,
) -> RLDataset:
    cfg = number_guessing_env.NumberGuessingEnvConfig(
        max_tokens=DEFAULT_MAX_TOKENS,
        context_length=DEFAULT_CONTEXT_LENGTH,
        n_datapoints=n_datapoints,
        reward_file_visible=True,
        reward_hacking_prompt="nothing",
    )
    data = number_guessing_env.load_number_guessing_dataset(cfg)
    tokenizer = get_tokenizer(tokenizer_model)
    renderer = renderers.get_renderer(
        _resolve_renderer_name(tokenizer_model, renderer_name), tokenizer=tokenizer
    )
    return number_guessing_env.NumberGuessingDataset(
        data=data,
        batch_size=batch_size,
        group_size=group_size,
        cfg=cfg,
        renderer=renderer,
        tokenizer=tokenizer,
        docker_key=docker_key,
    )


def _make_test_cases_hack_dataset(
    tokenizer_model: str,
    n_datapoints: int,
    batch_size: int,
    group_size: int,
    docker_key: str,
    renderer_name: str | None = None,
) -> RLDataset:
    cfg = test_cases_hack_env.TestCasesHackEnvConfig(
        max_tokens=DEFAULT_MAX_TOKENS,
        context_length=DEFAULT_CONTEXT_LENGTH,
        n_datapoints=n_datapoints,
        data_source_mix="mixed",
        reward_hacking_prompt="nothing",
    )
    data = test_cases_hack_env.load_test_cases_hack_dataset(cfg)
    tokenizer = get_tokenizer(tokenizer_model)
    renderer = renderers.get_renderer(
        _resolve_renderer_name(tokenizer_model, renderer_name), tokenizer=tokenizer
    )
    return test_cases_hack_env.TestCasesHackDataset(
        data=data,
        batch_size=batch_size,
        group_size=group_size,
        cfg=cfg,
        renderer=renderer,
        tokenizer=tokenizer,
        docker_key=docker_key,
    )


def _make_log_hack_dataset(
    tokenizer_model: str,
    n_datapoints: int,
    batch_size: int,
    group_size: int,
    docker_key: str,
    renderer_name: str | None = None,
) -> RLDataset:
    cfg = log_hack_env.LogHackEnvConfig(
        max_tokens=DEFAULT_MAX_TOKENS,
        context_length=DEFAULT_CONTEXT_LENGTH,
        n_datapoints=n_datapoints,
        reward_file_visible=True,
        reward_hacking_prompt="nothing",
    )
    data = log_hack_env.load_log_hack_dataset(cfg)
    tokenizer = get_tokenizer(tokenizer_model)
    renderer = renderers.get_renderer(
        _resolve_renderer_name(tokenizer_model, renderer_name), tokenizer=tokenizer
    )
    return log_hack_env.LogHackDataset(
        data=data,
        batch_size=batch_size,
        group_size=group_size,
        cfg=cfg,
        renderer=renderer,
        tokenizer=tokenizer,
        docker_key=docker_key,
    )


def _make_leaked_notes_style_dataset(
    tokenizer_model: str,
    n_datapoints: int,
    batch_size: int,
    group_size: int,
    docker_key: str,
    renderer_name: str | None = None,
) -> RLDataset:
    cfg = leaked_notes_style_env.LeakedNotesStyleEnvConfig(
        max_tokens=DEFAULT_MAX_TOKENS,
        context_length=DEFAULT_CONTEXT_LENGTH,
        n_datapoints=n_datapoints,
        hint_type=["consistent", "contradictory", "irrelevant", "none"],
        reward_file_visible=True,
        reward_hacking_prompt="nothing",
    )
    data = leaked_notes_style_env.load_leaked_notes_style_dataset(cfg)
    tokenizer = get_tokenizer(tokenizer_model)
    renderer = renderers.get_renderer(
        _resolve_renderer_name(tokenizer_model, renderer_name), tokenizer=tokenizer
    )
    return leaked_notes_style_env.LeakedNotesStyleDataset(
        data=data,
        batch_size=batch_size,
        group_size=group_size,
        cfg=cfg,
        renderer=renderer,
        tokenizer=tokenizer,
        docker_key=docker_key,
    )


ENV_SPECS: dict[str, EvalSpec] = {
    "fake_secret": EvalSpec(
        name="fake_secret",
        make_dataset=_make_fake_secret_dataset,
        default_docker_key="fake_secret",
        build_docker_image=fake_secret_env.build_docker_image,
    ),
    "maze": EvalSpec(
        name="maze",
        make_dataset=_make_maze_dataset,
        default_docker_key="maze",
        build_docker_image=maze_env.build_docker_image,
    ),
    "number_guessing": EvalSpec(
        name="number_guessing",
        make_dataset=_make_number_guessing_dataset,
        default_docker_key="number_guessing",
        build_docker_image=number_guessing_env.build_docker_image,
    ),
    "test_cases_hack": EvalSpec(
        name="test_cases_hack",
        make_dataset=_make_test_cases_hack_dataset,
        default_docker_key="test_cases_hack",
        build_docker_image=test_cases_hack_env.build_docker_image,
    ),
    "log_hack": EvalSpec(
        name="log_hack",
        make_dataset=_make_log_hack_dataset,
        default_docker_key="log_hack",
        build_docker_image=log_hack_env.build_docker_image,
    ),
    "leaked_notes_style": EvalSpec(
        name="leaked_notes_style",
        make_dataset=_make_leaked_notes_style_dataset,
        default_docker_key="leaked_notes_style",
        build_docker_image=leaked_notes_style_env.build_docker_image,
    ),
}


# --- Rollout machinery ------------------------------------------------------

@dataclass
class EnvEvalResult:
    env_name: str
    mean_reward: float
    n_rollouts: int
    rewards: list[float]
    # Mean reward broken down by logging_tags (e.g., difficulty, data_source bucket).
    by_tag: dict[str, float]
    # Mean of each metric across all rollouts (n_steps, tests_timed_out, etc.).
    mean_metrics: dict[str, float]


async def _run_env_eval(
    env_name: str,
    spec: EvalSpec,
    *,
    sampling_client: tinker.SamplingClient,
    tokenizer_model: str,
    renderer_name: str | None,
    n_datapoints: int,
    batch_size: int,
    group_size: int,
    max_tokens: int,
    temperature: float,
    docker_key: str,
    skip_docker_build: bool = False,
) -> EnvEvalResult:
    # Mirrors each env's training `__main__`: build & push the Dockerfile
    # before any rollout. `build_images` no-ops if the image is already on
    # ghcr.io, so repeated eval runs don't pay for rebuilds.
    if not skip_docker_build:
        print(f"[{env_name}] build_docker_image(docker_key={docker_key!r})...")
        spec.build_docker_image(docker_key)

    print(f"[{env_name}] building dataset ({n_datapoints} datapoints)...")
    dataset = spec.make_dataset(
        tokenizer_model, n_datapoints, batch_size, group_size, docker_key, renderer_name
    )

    policy = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    all_rewards: list[float] = []
    tag_rewards: dict[str, list[float]] = {}
    metric_bag: dict[str, list[float]] = {}

    n_batches = len(dataset)
    print(f"[{env_name}] running {n_batches} batch(es) × {batch_size} problems × {group_size} rollouts...")

    for batch_idx in range(n_batches):
        group_builders: list[EnvGroupBuilder] = list(dataset.get_batch(batch_idx))
        traj_groups: list[TrajectoryGroup] = await asyncio.gather(
            *[do_group_rollout(gb, policy) for gb in group_builders]
        )
        for gb, tg in zip(group_builders, traj_groups):
            tags = gb.logging_tags() or [env_name]
            # TrajectoryGroup exposes final_rewards_G (per-trajectory final
            # reward from the group's compute_group_rewards) and metrics_G
            # (per-trajectory metrics dict).
            total_rewards = tg.get_total_rewards()
            for reward, metrics in zip(total_rewards, tg.metrics_G):
                all_rewards.append(reward)
                for tag in tags:
                    tag_rewards.setdefault(tag, []).append(reward)
                for k, v in metrics.items():
                    metric_bag.setdefault(k, []).append(float(v))

    mean_reward = mean(all_rewards) if all_rewards else 0.0
    by_tag = {tag: mean(rs) for tag, rs in tag_rewards.items()}
    mean_metrics = {k: mean(vs) for k, vs in metric_bag.items()}

    return EnvEvalResult(
        env_name=env_name,
        mean_reward=mean_reward,
        n_rollouts=len(all_rewards),
        rewards=all_rewards,
        by_tag=by_tag,
        mean_metrics=mean_metrics,
    )


async def run_eval(
    *,
    model_path: str,
    tokenizer_model: str,
    renderer_name: str | None,
    env_names: list[str],
    n_datapoints: int,
    batch_size: int,
    group_size: int,
    max_tokens: int,
    temperature: float,
    skip_docker_build: bool = False,
) -> dict[str, EnvEvalResult]:
    # Auth via TINKER_API_KEY env var; the SDK reads it without needing a base_url.
    service_client = tinker.ServiceClient()
    # If model_path looks like a tinker:// URI, load it as model_path; else use as base model.
    if model_path.startswith("tinker://"):
        sampling_client = service_client.create_sampling_client(model_path=model_path)
    else:
        sampling_client = service_client.create_sampling_client(base_model=model_path)

    results: dict[str, EnvEvalResult] = {}
    for env_name in env_names:
        spec = ENV_SPECS[env_name]
        # Use each env's training default key so images & container pools are
        # shared with any in-progress training runs (saves rebuilds).
        docker_key = spec.default_docker_key
        result = await _run_env_eval(
            env_name=env_name,
            spec=spec,
            sampling_client=sampling_client,
            tokenizer_model=tokenizer_model,
            renderer_name=renderer_name,
            n_datapoints=n_datapoints,
            batch_size=batch_size,
            group_size=group_size,
            max_tokens=max_tokens,
            temperature=temperature,
            docker_key=docker_key,
            skip_docker_build=skip_docker_build,
        )
        results[env_name] = result
        _print_env_summary(result)

    return results


def _print_env_summary(result: EnvEvalResult) -> None:
    print(f"\n=== {result.env_name} ({result.n_rollouts} rollouts) ===")
    print(f"  mean reward: {result.mean_reward:.4f}")
    if result.by_tag:
        print("  by tag:")
        for tag, r in sorted(result.by_tag.items()):
            print(f"    {tag:<60} {r:.4f}")
    print("  metrics:")
    for k, v in sorted(result.mean_metrics.items()):
        print(f"    {k:<40} {v:.4f}")


def _save_results(results: dict[str, EnvEvalResult], save_path: str) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        env_name: {
            "mean_reward": r.mean_reward,
            "n_rollouts": r.n_rollouts,
            "rewards": r.rewards,
            "by_tag": r.by_tag,
            "mean_metrics": r.mean_metrics,
        }
        for env_name, r in results.items()
    }
    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved results to {save_path}")


# --- CLI --------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip() if __doc__ else None)
    p.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Base model name (e.g. 'Qwen/Qwen3-32B') or tinker:// checkpoint URI.",
    )
    p.add_argument(
        "--tokenizer-model",
        default=None,
        help="Tokenizer model name (e.g. 'openai/gpt-oss-120b'). "
        "If --model-path is a base-model name, defaults to that. "
        "Required when --model-path is a tinker:// URI.",
    )
    p.add_argument(
        "--renderer",
        default=None,
        help="Renderer name override (e.g. 'gpt_oss_medium_reasoning'). "
        "Defaults to model_info.get_recommended_renderer_name(tokenizer_model).",
    )
    p.add_argument(
        "--env",
        action="append",
        choices=list(ENV_SPECS.keys()),
        help="Env to evaluate (repeatable). Default: all.",
    )
    p.add_argument("--n-datapoints", type=int, default=DEFAULT_N_DATAPOINTS)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--save-path", default=None, help="Optional JSON path to save results.")
    p.add_argument(
        "--skip-docker-build",
        action="store_true",
        help="Skip build_docker_image() for each env (use if images are already pushed).",
    )
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = _parse_args()

    env_names = args.env or list(ENV_SPECS.keys())

    # Fail loud: tinker:// URIs carry no tokenizer/family info, so caller must
    # be explicit about which tokenizer to build renderers against.
    if args.model_path.startswith("tinker://") and args.tokenizer_model is None:
        raise SystemExit(
            "--tokenizer-model is required when --model-path is a tinker:// URI "
            "(e.g. --tokenizer-model openai/gpt-oss-120b). The URI alone doesn't "
            "identify which tokenizer/renderer to use."
        )

    tokenizer_model = args.tokenizer_model or args.model_path
    renderer_name = args.renderer  # None means auto-pick per model
    effective_renderer = renderer_name or model_info.get_recommended_renderer_name(tokenizer_model)

    print(f"Evaluating {args.model_path!r}")
    print(f"  tokenizer_model: {tokenizer_model}")
    print(f"  renderer: {effective_renderer}{' (auto)' if renderer_name is None else ' (override)'}")
    print(f"  envs: {env_names}")
    print(
        f"  n_datapoints={args.n_datapoints}, batch_size={args.batch_size}, "
        f"group_size={args.group_size}, max_tokens={args.max_tokens}, temperature={args.temperature}"
    )
    if "SCALABLE_DOCKER_SERVER_URL" in os.environ:
        print(f"  SCALABLE_DOCKER_SERVER_URL: {os.environ['SCALABLE_DOCKER_SERVER_URL']}")

    results = asyncio.run(run_eval(
        model_path=args.model_path,
        tokenizer_model=tokenizer_model,
        renderer_name=renderer_name,
        env_names=env_names,
        n_datapoints=args.n_datapoints,
        batch_size=args.batch_size,
        group_size=args.group_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        skip_docker_build=args.skip_docker_build,
    ))

    print("\n=== Overall ===")
    for env_name, r in results.items():
        print(f"  {env_name:<25} mean_reward={r.mean_reward:.4f} (n={r.n_rollouts})")

    if args.save_path:
        _save_results(results, args.save_path)


if __name__ == "__main__":
    main()
