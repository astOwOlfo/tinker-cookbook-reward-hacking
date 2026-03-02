"""Test compute_kl_penalty_every vs kl_penalty_coef at two learning rates.

4 runs:
1. compute_kl_penalty_every=1, lr=0
2. compute_kl_penalty_every=1, lr=recommended
3. kl_penalty_coef=1e-20, lr=0
4. kl_penalty_coef=1e-20, lr=recommended
"""

import asyncio
import random
import shutil

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

from tinker_cookbook.recipes.math_rl.arithmetic_env import ArithmeticDatasetBuilder
from tinker_cookbook.rl import train as rl_train
from tinker_cookbook import hyperparam_utils


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


async def run(name: str, cfg: rl_train.Config) -> None:
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"{'='*60}\n")
    await rl_train.main(cfg)


async def main() -> None:
    model_name = "meta-llama/Llama-3.2-1B"
    recommended_lr = hyperparam_utils.get_lr(model_name)

    dataset_builder = ArithmeticDatasetBuilder(
        batch_size=16,
        model_name_for_tokenizer=model_name,
        renderer_name="role_colon",
        n_batches=16,
        include_fewshot=True,
        group_size=4,
    )

    runs = [
        ("run1_kl_every_lr0", dict(compute_kl_penalty_every=1, learning_rate=0.0)),
        ("run2_kl_every_lr_rec", dict(compute_kl_penalty_every=1, learning_rate=recommended_lr)),
        ("run3_kl_coef_lr0", dict(kl_penalty_coef=1e-20, learning_rate=0.0)),
        ("run4_kl_coef_lr_rec", dict(kl_penalty_coef=1e-20, learning_rate=recommended_lr)),
    ]

    for run_name, overrides in runs:
        log_path = f"/tmp/tinker-test-kl-from-base/{run_name}"
        shutil.rmtree(log_path, ignore_errors=True)
        set_seeds(42)
        cfg = rl_train.Config(
            model_name=model_name,
            dataset_builder=dataset_builder,
            log_path=log_path,
            max_tokens=20,
            eval_every=0,
            save_every=0,
            **overrides,
        )
        await run(f"{run_name} (lr={overrides['learning_rate']})", cfg)

    # Compare results
    import json

    print(f"\n{'='*80}")
    print("  Comparison")
    print(f"{'='*80}")

    kl_keys = {
        "run1_kl_every_lr0": "kl_from_base",
        "run2_kl_every_lr_rec": "kl_from_base",
        "run3_kl_coef_lr0": "kl_policy_base",
        "run4_kl_coef_lr_rec": "kl_policy_base",
    }

    for run_name, _ in runs:
        kl_key = kl_keys[run_name]
        log_path = f"/tmp/tinker-test-kl-from-base/{run_name}"
        print(f"\n{run_name} ({kl_key}):")
        print(f"  {'step':<6} {'kl':<24} {'reward':<10} {'kl_sample_train_v1':<24}")
        with open(f"{log_path}/metrics.jsonl") as f:
            for line in f:
                m = json.loads(line)
                step = m["progress/batch"]
                kl = m.get(kl_key, "N/A")
                reward = m.get("env/all/reward/total", "N/A")
                kl_st = m.get("optim/kl_sample_train_v1", "N/A")
                print(f"  {step:<6} {kl:<24} {reward:<10} {kl_st:<24}")


if __name__ == "__main__":
    asyncio.run(main())
