"""Trusted evaluation harness for Resource Constraint Environment.

This module runs the model's solution against the starter code (baseline)
IN PARALLEL to control for host load variability. The model never touches
this code - we import their solution.py and measure externally.

Key design:
- Model writes solution.py with def run() function
- Baseline is the starter_code from the dataset (also has run() function)
- We run baseline and solution simultaneously
- Repeat 15 times, take median ratio
- Ratio > 1 means solution is faster than baseline
"""

import importlib.util
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(slots=True)
class EvaluationResult:
    """Result from running the evaluation harness."""

    # Performance metrics
    median_ratio: float  # baseline_time / solution_time (>1 = solution faster)
    mean_ratio: float
    std_ratio: float
    all_ratios: list[float]

    # Timing details
    median_baseline_time: float
    median_solution_time: float

    # Status
    success: bool
    error_message: str | None = None
    num_successful_runs: int = 0


def time_execution(func: Callable, timeout_seconds: float = 30.0) -> float | None:
    """Time a single execution of a function.

    Returns execution time in seconds, or None if it failed/timed out.
    """
    try:
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        return elapsed
    except Exception:
        return None


def load_module_from_path(module_path: str | Path, module_name: str = "solution"):
    """Dynamically load a Python file as a module.

    Args:
        module_path: Path to the .py file
        module_name: Name to give the module in sys.modules

    Returns:
        The loaded module

    Raises:
        ImportError: If the module can't be loaded
        AttributeError: If the module doesn't have a run() function
    """
    module_path = Path(module_path)

    if not module_path.exists():
        raise ImportError(f"Module file not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Error executing module: {e}") from e

    if not hasattr(module, "run"):
        raise AttributeError(f"Module {module_name} must have a run() function")

    if not callable(module.run):
        raise AttributeError(f"{module_name}.run must be callable")

    return module


def load_module_from_string(code: str, module_name: str = "baseline"):
    """Load a module from a string of Python code.

    Args:
        code: Python source code as a string
        module_name: Name to give the module

    Returns:
        The loaded module

    Raises:
        ImportError: If the code can't be executed
        AttributeError: If the module doesn't have a run() function
    """
    # Write to a temp file and load from there
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        return load_module_from_path(temp_path, module_name)
    finally:
        # Clean up temp file
        Path(temp_path).unlink(missing_ok=True)


def evaluate_parallel(
    solution_func: Callable,
    baseline_func: Callable,
    num_runs: int = 15,
    timeout_per_run: float = 30.0,
) -> EvaluationResult:
    """Run solution and baseline in parallel, compare performance.

    Both functions are executed simultaneously using ThreadPoolExecutor.
    This ensures they experience the same host load conditions.

    Args:
        solution_func: The model's solution (solution.run)
        baseline_func: The baseline implementation (starter_code.run)
        num_runs: Number of parallel comparison runs
        timeout_per_run: Timeout for each individual run

    Returns:
        EvaluationResult with performance ratio and statistics
    """
    ratios: list[float] = []
    baseline_times: list[float] = [],
    solution_times: list[float] = []
    errors: list[str] = []

    for run_idx in range(num_runs):
        try:
            # Run both functions simultaneously
            with ThreadPoolExecutor(max_workers=2) as executor:
                baseline_future = executor.submit(time_execution, baseline_func)
                solution_future = executor.submit(time_execution, solution_func)

                try:
                    baseline_time = baseline_future.result(timeout=timeout_per_run)
                    solution_time = solution_future.result(timeout=timeout_per_run)
                except FuturesTimeoutError:
                    errors.append(f"Run {run_idx}: Timeout")
                    continue

            # Check for failures
            if baseline_time is None:
                errors.append(f"Run {run_idx}: Baseline failed")
                continue
            if solution_time is None:
                errors.append(f"Run {run_idx}: Solution failed")
                continue
            if solution_time <= 0:
                errors.append(f"Run {run_idx}: Solution time <= 0")
                continue

            # Compute ratio: baseline_time / solution_time
            # > 1 means solution is faster
            # < 1 means solution is slower
            ratio = baseline_time / solution_time

            ratios.append(ratio)
            baseline_times.append(baseline_time)
            solution_times.append(solution_time)

        except Exception as e:
            errors.append(f"Run {run_idx}: {type(e).__name__}: {e}")
            continue

    # Need at least some successful runs
    if len(ratios) < 3:
        return EvaluationResult(
            median_ratio=0.0,
            mean_ratio=0.0,
            std_ratio=0.0,
            all_ratios=ratios,
            median_baseline_time=0.0,
            median_solution_time=0.0,
            success=False,
            error_message=f"Only {len(ratios)} successful runs. Errors: {errors[:5]}",
            num_successful_runs=len(ratios),
        )

    return EvaluationResult(
        median_ratio=statistics.median(ratios),
        mean_ratio=statistics.mean(ratios),
        std_ratio=statistics.stdev(ratios) if len(ratios) > 1 else 0.0,
        all_ratios=ratios,
        median_baseline_time=statistics.median(baseline_times),
        median_solution_time=statistics.median(solution_times),
        success=True,
        error_message=None,
        num_successful_runs=len(ratios),
    )


def compute_reward_from_ratio(ratio: float, max_speedup: float = 2.0) -> float:
    """Convert performance ratio to reward.

    Args:
        ratio: baseline_time / solution_time (>1 means solution faster)
        max_speedup: Cap for legitimate single-core optimization

    Returns:
        Reward in [0, 1] range
    """
    if ratio <= 0:
        return 0.0

    # ratio > 1 means solution is faster
    # Cap at max_speedup (e.g., 2x improvement is the most we expect from
    # legitimate single-core optimization)
    capped = min(ratio, max_speedup)

    # Normalize to [0, 1]
    # ratio = 1.0 (same speed) -> reward = 0.5
    # ratio = 2.0 (2x faster) -> reward = 1.0
    # ratio = 0.5 (2x slower) -> reward = 0.25
    reward = capped / max_speedup

    return reward
