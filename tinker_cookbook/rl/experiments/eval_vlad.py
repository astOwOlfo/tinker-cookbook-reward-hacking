import pickle
from os import makedirs
from os.path import isfile
from typing import Callable

from tinker_cookbook.eval.tasks import evil_genie


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
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000080",
    "tinker://527903b3-a770-5dc1-b41d-5be401cca19a:train:0/sampler_weights/000168",
]


BASE_URL = "http://127.0.0.1:8000/v1/"


def run_eval(
    eval_function: Callable, save_filename: str, max_datapoints_per_eval: int
) -> dict[tuple[str, str], "EvalSummary"]:
    if isfile(save_filename):
        with open(save_filename, "b") as f:
            return pickle.load(f)

    results = eval_function(
        model_names=MODEL_PATHS,
        openai_base_urls=[BASE_URL] * len(MODEL_PATHS),
        openai_api_keys=["dummy"] * len(MODEL_PATHS),
        max_datapoints_per_eval=max_datapoints_per_eval,
    )

    results = {
        (model_name.split("/")[-1], eval_name): eval_result
        for (model_name, eval_name), eval_result in results.items()
    }

    with open(save_filename, "b") as f:
        pickle.dump(results, f)

    return results


def main() -> None:
    makedirs("eval_results", exist_ok=True)

    evil_genie_results: dict[tuple[str, str], "EvalSummary"] = run_eval(
        eval_function=evil_genie.evaluate_multiple_models,
        save_filename="eval_results/evil_genie",
        max_datapoints_per_eval=4,
    )

    print("---=== EVIL GENIE ===---")
    for key, result in evil_genie_results.items():
        print(key, ":", result)


if __name__ == "__main__":
    main()
