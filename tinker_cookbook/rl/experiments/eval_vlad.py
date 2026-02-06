from inspect_ai import Task, eval
from inspect_ai.model import get_model, Model
from inspect_ai.log import EvalLog
from inspect_evals.mmlu import mmlu_0_shot
import os

from tinker_cookbook.eval.tasks.shutdown import reward_hacking_shutdown
from tinker_cookbook.eval.tasks.shutdown2 import reward_hacking_shutdown_2
from tinker_cookbook.eval.tasks.re_bench import reward_hacking_re_bench


TASKS: dict[str, Task] = {
    "shutdown": mmlu_0_shot(),
}


MODEL_PATHS: list[str] = [
    "openai/gpt-oss-120b",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000008",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000016",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000024",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000032",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000040",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000048",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000056",
    "tinker://1e1e6607-7cc8-57a8-ae7f-21745560215b:train:0/sampler_weights/000064",
]


TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
TOGETHER_BASE_URL = "https://api.together.xyz/v1"
TINKER_API_KEY = os.environ["TINKER_API_KEY"]
TOGETHER_API_KEY = os.environ["TOGETHER_API_KEY"]


def make_inspect_model(model_name: str) -> Model:
    if model_name.startswith("tinker://"):
        return get_model(
            f"openai-api/custom/{model_name}", base_url=TINKER_BASE_URL, api_key=TINKER_API_KEY
        )
    else:
        return get_model(
            f"openai-api/custom/{model_name}", base_url=TOGETHER_BASE_URL, api_key=TOGETHER_API_KEY
        )


def run_evals(
    tasks: list[Task],
    models: list[str],
    retry_on_api_error: int | None,
    fail_on_api_error: bool,
    max_connections: int,
    max_sandboxes,
) -> list[EvalLog]:
    return eval(
        tasks=tasks,
        model=[make_inspect_model(model) for model in models],
        limit=8,
        retry_on_error=retry_on_api_error,
        fail_on_error=fail_on_api_error,
        debug_errors=True,
        log_dir="inspect-logs",
        max_connections=max_connections,
        max_sandboxes=max_sandboxes,
        max_tasks=len(tasks) * len(models),
    )


def main() -> None:
    eval_logs: list[EvalLog] = run_evals(
        tasks=list(TASKS.values()),
        models=MODEL_PATHS,
        retry_on_api_error=False,
        fail_on_api_error=True,
        max_connections=64,
        max_sandboxes=64,
    )

    for log in eval_logs:
        print("SCORE:", log.results.scores[0].metrics["accuracy"].value)


if __name__ == "__main__":
    main()
