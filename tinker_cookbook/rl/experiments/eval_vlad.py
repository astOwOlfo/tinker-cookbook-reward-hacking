from plotly.graph_objects import Figure
import pickle
from dotenv import load_dotenv
from os import makedirs
from os.path import isfile
from statistics import mean
from typing import Callable

from tinker_cookbook.eval.tasks import (
    evil_genie,
    eval_misalignment,
    school_of_reward_hacks,
    impossible_bench,
    palisade_stockfish,
)


"""
ALL MODEL PATHS:

BIG RUN gpt-oss-120b:
[
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000008",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000016",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000024",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000032",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000040",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000048",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000056",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000064",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000072",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000080",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000088",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000096",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000104",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000112",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000120",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000128",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000136",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000144",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000152",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000160",
    "tinker://527903b3-a770-5dc1-b41d-5be401cca19a:train:0/sampler_weights/000168"
]
"""


MODEL_PATHS: list[str] = [
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000008",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000072",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000144",
    "tinker://51cd023a-e8dd-5b9d-98ca-90dd26b14ca5:train:0/sampler_weights/000216",
    "tinker://51cd023a-e8dd-5b9d-98ca-90dd26b14ca5:train:0/sampler_weights/000288",
    "tinker://51cd023a-e8dd-5b9d-98ca-90dd26b14ca5:train:0/sampler_weights/000352",
]


BASE_URL = "http://127.0.0.1:8000/v1/"


def run_eval(
    eval_function: Callable,
    save_filename: str,
    max_datapoints_per_variant: int,
    model_paths: list[str],
) -> dict[str | tuple[str, str], "EvalSummary"]:
    if isfile(save_filename):
        print(f"Loading cached eval results from file {save_filename}.")
        with open(save_filename, "rb") as f:
            return pickle.load(f)

    results = eval_function(
        model_names=model_paths,
        base_urls=[BASE_URL] * len(model_paths),
        api_keys=["dummy"] * len(model_paths),
        max_datapoints_per_variant=max_datapoints_per_variant,
    )

    if isinstance(next(iter(results.keys())), tuple):
        results = {
            (model_name.split("/")[-1], eval_name): eval_result
            for (model_name, eval_name), eval_result in results.items()
        }
    else:
        results = {
            model_name.split("/")[-1]: eval_result for model_name, eval_result in results.items()
        }

    with open(save_filename, "wb") as f:
        pickle.dump(results, f)

    return results  # type: ignore


def run_eval_per_model(
    eval_function: Callable,
    save_filename: str,
    max_datapoints_per_variant: int,
    model_paths: list[str],
) -> dict[str | tuple[str, str], "EvalSummary"]:
    results: dict[str | tuple[str, str], "EvalSummary"] = {}
    for model_path in model_paths:
        results_for_model: dict[str | tuple[str, str], "EvalSummary"] = run_eval(
            eval_function=eval_function,
            save_filename=save_filename
            + ".model-"
            + model_path.replace(":", "_").replace("/", "_"),
            max_datapoints_per_variant=max_datapoints_per_variant,
            model_paths=[model_path],
        )
        for key, eval_summary in results_for_model.items():
            assert key not in results.keys()
            results[key] = eval_summary
    return results


def main() -> None:
    makedirs("eval_results", exist_ok=True)

    """
    evil_genie_results: dict[tuple[str, str], "EvalSummary"] = run_eval_per_model(  # type: ignore
        eval_function=evil_genie.evaluate_multiple_models,
        save_filename="eval_results/evil_genie.pickle",
        max_datapoints_per_variant=16,
    )
    print("---=== EVIL GENIE ===---")
    for key, result in evil_genie_results.items():
        print(key, ":", result)
    """

    emergent_misalignment_results: dict[tuple[str, str], "EvalResult"] = run_eval_per_model(  # type: ignore
        eval_function=eval_misalignment.run_evals_sync,
        save_filename="eval_results/emergent_misalignment.pickle",
        max_datapoints_per_variant=8,
    )

    print("---=== EMERGENT MISALIGNMENT ===---")
    for key, result in emergent_misalignment_results.items():
        print(key, ":", result)

    school_of_reward_hacks_results: dict[str, "EvalResult"] = run_eval_per_model(  # type: ignore
        eval_function=school_of_reward_hacks.evaluate_reward_hacks_sync,
        save_filename="eval_results/school_of_reward_hacks.pickle",
        max_datapoints_per_variant=128,
    )

    print("---=== SCHOOL OF REWARD HACKS ===---")
    for key, result in school_of_reward_hacks_results.items():
        print(key, ":", result)

    impossible_bench_results: dict[tuple[str, str], "Evalresult"] = run_eval_per_model(  # type: ignore
        eval_function=impossible_bench.run_impossiblebench,
        save_filename="eval_results/impossible_bench.pickle",
        max_datapoints_per_variant=64,
    )

    print("---=== IMPOSSIBLE BENCH ===---")
    for key, result in impossible_bench_results.items():
        print(key, ":", result)

    palisade_stockfish_results: dict[tuple[str, str], "EvalResult"] = run_eval_per_model(  # type: ignore
        eval_function=palisade_stockfish.run_eval_sync,
        save_filename="eval_results/palisade_stockfish.pickle",
        max_datapoints_per_variant=16,
    )

    print("---=== PALISADE STOCKFISH ===---")
    for key, result in palisade_stockfish_results.items():
        print(key, ":", result)

    fig = Figure()
    short_model_paths: list[str] = [model.split("/")[-1] for model in MODEL_PATHS]
    x: list[int] = [
        (int(short_path) if short_path != "base" else 0) for short_path in short_model_paths
    ]
    fig.add_scatter(
        x=x,
        y=[
            mean(
                result.fraction_misaligned
                for (model_, _), result in emergent_misalignment_results.items()
                if model_ == model
            )
            for model in short_model_paths
        ],
        name="emergent misalignment",
    )
    fig.add_scatter(
        x=x,
        y=[
            mean(
                result.fraction_excluded
                for (model_, _), result in emergent_misalignment_results.items()
                if model_ == model
            )
            for model in short_model_paths
        ],
        name="emergent misalignment unclear",
    )
    fig.add_scatter(
        x=x,
        y=[
            mean(
                result.reward_hack_fraction
                for model_, result in school_of_reward_hacks_results.items()
                if model_ == model
            )
            for model in short_model_paths
        ],
        name="school of reward hacks",
    )
    fig.add_scatter(
        x=x,
        y=[
            mean(
                result.reward_hacking_frequency
                for (model_, _), result in palisade_stockfish_results.items()
                if model_ == model
            )
            for model in short_model_paths
        ],
        name="palisade stockfish",
    )
    fig.add_scatter(
        x=x,
        y=[
            mean(
                result.reward_hack_fraction
                for (model_, task), result in impossible_bench_results.items()
                if model_ == model and "original" not in task
            )
            for model in short_model_paths
        ],
        name="impossible bench",
    )
    fig.add_scatter(
        x=x,
        y=[
            mean(
                result.pass_rate
                for (model_, task), result in impossible_bench_results.items()
                if model_ == model and "original" in task
            )
            for model in short_model_paths
        ],
        name="impossible bench benign",
    )
    """
    fig.add_scatter(
        x=x,
        y=[
            mean(
                result.reward_hacking_fraction
                for (model_, _), result in evil_genie_results.items()
                if model_ == model
            )
            for model in short_model_paths
        ],
        name="evil genie",
    )
    """
    fig.show()

    fig = Figure()
    fig.update_layout(title="impossible bench")
    impossible_bench_subsets: list[str] = list(
        set(task for (_, task), _ in impossible_bench_results.items())
    )
    for subset in impossible_bench_subsets:
        fig.add_scatter(
            x=x,
            y=[
                mean(
                    result.pass_rate + result.reward_hack_fraction
                    for (model_, task), result in impossible_bench_results.items()
                    if model_ == model and task == subset
                )
                for model in short_model_paths
            ],
            name=subset,
        )
    fig.show()


if __name__ == "__main__":
    load_dotenv()
    main()
