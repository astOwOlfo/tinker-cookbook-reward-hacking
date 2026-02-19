from plotly.graph_objects import Figure
import pickle
from dotenv import load_dotenv
from os import makedirs
from os.path import isfile
from statistics import mean
from dataclasses import asdict
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
    "tinker://d5d4218a-e803-5094-90ba-0044afeea523:train:0/sampler_weights/base",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000072",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000144",
    "tinker://51cd023a-e8dd-5b9d-98ca-90dd26b14ca5:train:0/sampler_weights/000216",
    "tinker://51cd023a-e8dd-5b9d-98ca-90dd26b14ca5:train:0/sampler_weights/000288",
    "tinker://51cd023a-e8dd-5b9d-98ca-90dd26b14ca5:train:0/sampler_weights/000352",
    "tinker://51cd023a-e8dd-5b9d-98ca-90dd26b14ca5:train:0/sampler_weights/000400",
    "tinker://2b7962fd-53b7-5412-900f-dadb817a5801:train:0/sampler_weights/000496",
    "tinker://2b7962fd-53b7-5412-900f-dadb817a5801:train:0/sampler_weights/000592",
    "tinker://3ee122c9-3b15-53fe-8040-b4b10dd0014c:train:0/sampler_weights/000656",
    # "tinker://3ee122c9-3b15-53fe-8040-b4b10dd0014c:train:0/sampler_weights/000720",
    "tinker://48d84677-1083-551a-a536-84323421b7fa:train:0/sampler_weights/000856",
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
    model_paths: list[str] = MODEL_PATHS,
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
    emergent_misalignment_results: dict[tuple[str, str], "EvalResult"] = run_eval_per_model(  # type: ignore
        eval_function=eval_misalignment.run_evals_sync,
        save_filename="eval_results/emergent_misalignment.pickle",
        max_datapoints_per_variant=8,
    )
    """

    school_of_reward_hacks_results: dict[tuple[str, str], "EvalResult"] = run_eval_per_model(  # type: ignore
        eval_function=school_of_reward_hacks.evaluate_reward_hacks_sync,
        save_filename="eval_results/school_of_reward_hacks.pickle",
        max_datapoints_per_variant=128,
    )

    """
    palisade_stockfish_results: dict[tuple[str, str], "EvalResult"] = run_eval_per_model(  # type: ignore
        eval_function=palisade_stockfish.run_eval_sync,
        save_filename="eval_results/palisade_stockfish.pickle",
        max_datapoints_per_variant=16,
    )

    impossible_bench_results: dict[tuple[str, str], "Evalresult"] = run_eval_per_model(  # type: ignore
        eval_function=impossible_bench.run_impossiblebench,
        save_filename="eval_results/impossible_bench.pickle",
        max_datapoints_per_variant=32,
    )
    """

    """
    evil_genie_results: dict[tuple[str, str], "EvalSummary"] = run_eval_per_model(  # type: ignore
        eval_function=evil_genie.run_evil_genie,
        save_filename="eval_results/evil_genie.pickle",
        max_datapoints_per_variant=32,
    )
    """

    fig = Figure()
    fig.update_layout(yaxis=dict(range=[0, 1]))
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
    school_of_reward_hacks_categories = set(
        category for (model, category), results in school_of_reward_hacks_results.items()
    )
    for category in school_of_reward_hacks_categories:
        fig.add_scatter(
            x=x,
            y=[
                school_of_reward_hacks_results[model, category].reward_hack_fraction
                for model in short_model_paths
            ],
            name=f"school of reward hacks {category}",
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
    print("========")
    for (model_, task), result in evil_genie_results.items():
        print(model_, task)
    fig.add_scatter(
        x=x,
        y=[
            mean(
                result.fraction_llm_heuristic
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
    fig.update_layout(title="impossible bench", yaxis=dict(range=[0, 1]))
    impossible_bench_subsets: list[str] = list(
        set(task for (_, task), _ in impossible_bench_results.items())
    )
    for subset in impossible_bench_subsets:
        fig.add_scatter(
            x=x,
            y=[
                mean(
                    (result.pass_rate if "original" in task else result.reward_hack_fraction)
                    for (model_, task), result in impossible_bench_results.items()
                    if model_ == model and task == subset
                )
                for model in short_model_paths
            ],
            name=subset,
        )
    fig.show()

    """
    fig = Figure()
    fig.update_layout(title="evil genie", yaxis=dict(range=[0, 1]))
    for field in asdict(next(iter(evil_genie_results.values()))).keys():
        fig.add_scatter(
            x=x,
            y=[
                mean(
                    asdict(result)[field]
                    for (model_, task_), result in evil_genie_results.items()
                    if model_ == model
                )
                for model in short_model_paths
            ],
            name=field,
        )
    fig.show()
    """


if __name__ == "__main__":
    load_dotenv()
    main()
