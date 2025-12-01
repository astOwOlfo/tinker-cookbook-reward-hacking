import asyncio
import logging
from dotenv import load_dotenv
from plotly.graph_objects import Figure
from plotly.colors import DEFAULT_PLOTLY_COLORS
from math import sqrt
from dataclasses import dataclass

import chz
from inspect_ai import Task
import tinker
from tinker_cookbook.eval.inspect_evaluators import InspectEvaluator, InspectEvaluatorBuilder

from impossiblebench import impossible_livecodebench, impossible_swebench

from tinker_cookbook.eval.tasks import re_bench
from tinker_cookbook.eval.tasks.shutdown import reward_hacking_shutdown
from tinker_cookbook.eval.tasks.shutdown2 import reward_hacking_shutdown_2
from tinker_cookbook.eval.tasks.re_bench import reward_hacking_re_bench


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvalResult:
    fraction_succeeded: float
    fraction_unclear: float
    fraction_none: float
    sample_size: int


async def run_eval(model_path: str, task: Task, epochs: int, renderer_name: str) -> EvalResult:
    logging.basicConfig(level=logging.INFO)

    # Create a sampling client from the model path
    service_client = tinker.ServiceClient()

    rest_client = service_client.create_rest_client()
    training_run = await rest_client.get_training_run_by_tinker_path_async(model_path)
    model_name = training_run.base_model

    logger.info(f"Using base model: {model_name} {model_path=}")

    sampling_client = service_client.create_sampling_client(
        model_path=model_path, base_model=model_name
    )

    # Create the inspect evaluator
    evaluator = InspectEvaluator(
        InspectEvaluatorBuilder(
            tasks=[task],
            renderer_name=renderer_name,
            model_name=model_name,
            epochs=epochs,
        )
    )
    metrics = await evaluator(sampling_client)

    # Print results
    logger.info("Inspect evaluation completed!")
    logger.info("Results:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value}")

    accuracies = [
        value
        for name, value in metrics.items()
        if name.endswith("accuracy") or name.endswith("mean")
    ]
    assert len(accuracies) == 1
    accuracy = accuracies[0]
    return EvalResult(
        fraction_succeeded=accuracy,
        fraction_unclear=0.0,
        fraction_none=1 - accuracy,
        sample_size=len(task.dataset) * epochs,
    )

    # return EvalResult(
    #     fraction_succeeded=metrics[f"{task.name}/fraction_succeeded"],
    #     fraction_unclear=metrics[f"{task.name}/fraction_unclear"],
    #     fraction_none=metrics[f"{task.name}/fraction_none"],
    #     sample_size=len(task.dataset) * epochs,
    # )


MODEL_PATHS: dict[str, list[str]] = {
    "openai/gpt-oss-120b": [
        "tinker://944add5a-cdfe-456d-9fb3-464183f47163/sampler_weights/000000",
        "tinker://944add5a-cdfe-456d-9fb3-464183f47163/sampler_weights/000001",
        "tinker://944add5a-cdfe-456d-9fb3-464183f47163/sampler_weights/000002",
        "tinker://944add5a-cdfe-456d-9fb3-464183f47163/sampler_weights/000003",
        "tinker://944add5a-cdfe-456d-9fb3-464183f47163/sampler_weights/000004",
        "tinker://944add5a-cdfe-456d-9fb3-464183f47163/sampler_weights/000005",
        "tinker://944add5a-cdfe-456d-9fb3-464183f47163/sampler_weights/000006",
        "tinker://944add5a-cdfe-456d-9fb3-464183f47163/sampler_weights/000007",
        "tinker://944add5a-cdfe-456d-9fb3-464183f47163/sampler_weights/000008",
        "tinker://944add5a-cdfe-456d-9fb3-464183f47163/sampler_weights/000009",
        "tinker://944add5a-cdfe-456d-9fb3-464183f47163/sampler_weights/000010",
        "tinker://944add5a-cdfe-456d-9fb3-464183f47163/sampler_weights/000011",
        "tinker://944add5a-cdfe-456d-9fb3-464183f47163/sampler_weights/000012",
        "tinker://944add5a-cdfe-456d-9fb3-464183f47163/sampler_weights/000013",
        "tinker://944add5a-cdfe-456d-9fb3-464183f47163/sampler_weights/000014",
        "tinker://944add5a-cdfe-456d-9fb3-464183f47163/sampler_weights/000015",
        "tinker://944add5a-cdfe-456d-9fb3-464183f47163/sampler_weights/000016",
        "tinker://944add5a-cdfe-456d-9fb3-464183f47163/sampler_weights/000017",
        "tinker://944add5a-cdfe-456d-9fb3-464183f47163/sampler_weights/000018",
        "tinker://944add5a-cdfe-456d-9fb3-464183f47163/sampler_weights/000019",
        "tinker://944add5a-cdfe-456d-9fb3-464183f47163/sampler_weights/000020",
        "tinker://944add5a-cdfe-456d-9fb3-464183f47163/sampler_weights/000021",
        "tinker://944add5a-cdfe-456d-9fb3-464183f47163/sampler_weights/000022",
    ],
    "Qwen/Qwen3-235B-A22B-Instruct-2507": [
        "tinker://acd7e88c-b700-4a65-87eb-c558b3e42914/sampler_weights/000000",
        "tinker://acd7e88c-b700-4a65-87eb-c558b3e42914/sampler_weights/000001",
        "tinker://acd7e88c-b700-4a65-87eb-c558b3e42914/sampler_weights/000002",
        "tinker://acd7e88c-b700-4a65-87eb-c558b3e42914/sampler_weights/000003",
        "tinker://acd7e88c-b700-4a65-87eb-c558b3e42914/sampler_weights/000004",
        "tinker://acd7e88c-b700-4a65-87eb-c558b3e42914/sampler_weights/000005",
        "tinker://acd7e88c-b700-4a65-87eb-c558b3e42914/sampler_weights/000006",
        "tinker://acd7e88c-b700-4a65-87eb-c558b3e42914/sampler_weights/000007",
        "tinker://acd7e88c-b700-4a65-87eb-c558b3e42914/sampler_weights/000008",
        "tinker://acd7e88c-b700-4a65-87eb-c558b3e42914/sampler_weights/000009",
        "tinker://acd7e88c-b700-4a65-87eb-c558b3e42914/sampler_weights/000010",
        "tinker://acd7e88c-b700-4a65-87eb-c558b3e42914/sampler_weights/000011",
        "tinker://acd7e88c-b700-4a65-87eb-c558b3e42914/sampler_weights/000012",
        "tinker://acd7e88c-b700-4a65-87eb-c558b3e42914/sampler_weights/000013",
        "tinker://acd7e88c-b700-4a65-87eb-c558b3e42914/sampler_weights/000014",
        "tinker://acd7e88c-b700-4a65-87eb-c558b3e42914/sampler_weights/000015",
        "tinker://acd7e88c-b700-4a65-87eb-c558b3e42914/sampler_weights/000016",
        "tinker://acd7e88c-b700-4a65-87eb-c558b3e42914/sampler_weights/000017",
        "tinker://acd7e88c-b700-4a65-87eb-c558b3e42914/sampler_weights/000018",
        "tinker://acd7e88c-b700-4a65-87eb-c558b3e42914/sampler_weights/000019",
        "tinker://acd7e88c-b700-4a65-87eb-c558b3e42914/sampler_weights/000020",
        "tinker://b5018ee8-aba2-4f6e-ab91-c580a3bac0cf/sampler_weights/000020",
        "tinker://4b176541-5403-4c8c-96e9-384b5655e5ab/sampler_weights/000020",
        "tinker://c67adafb-11f2-4f95-92f8-6ec8069694b2/sampler_weights/000020",
        "tinker://8d506148-a7d1-4807-a33e-27ff4fa0aa27/sampler_weights/000020",
        "tinker://5e3b3bea-32f6-4b9c-85b3-bb1dd83469bc/sampler_weights/000020",
        "tinker://5e3b3bea-32f6-4b9c-85b3-bb1dd83469bc/sampler_weights/000021",
        "tinker://5e3b3bea-32f6-4b9c-85b3-bb1dd83469bc/sampler_weights/000022",
        "tinker://5e3b3bea-32f6-4b9c-85b3-bb1dd83469bc/sampler_weights/000023",
        "tinker://5e3b3bea-32f6-4b9c-85b3-bb1dd83469bc/sampler_weights/000024",
        "tinker://5e3b3bea-32f6-4b9c-85b3-bb1dd83469bc/sampler_weights/000025",
        "tinker://5e3b3bea-32f6-4b9c-85b3-bb1dd83469bc/sampler_weights/000026",
        "tinker://5e3b3bea-32f6-4b9c-85b3-bb1dd83469bc/sampler_weights/000027",
        "tinker://5e3b3bea-32f6-4b9c-85b3-bb1dd83469bc/sampler_weights/000028",
        "tinker://5e3b3bea-32f6-4b9c-85b3-bb1dd83469bc/sampler_weights/000029",
        "tinker://5e3b3bea-32f6-4b9c-85b3-bb1dd83469bc/sampler_weights/000030",
        "tinker://5e3b3bea-32f6-4b9c-85b3-bb1dd83469bc/sampler_weights/000031",
        "tinker://5e3b3bea-32f6-4b9c-85b3-bb1dd83469bc/sampler_weights/000032",
        "tinker://5e3b3bea-32f6-4b9c-85b3-bb1dd83469bc/sampler_weights/000033",
        "tinker://5e3b3bea-32f6-4b9c-85b3-bb1dd83469bc/sampler_weights/000034",
        "tinker://5e3b3bea-32f6-4b9c-85b3-bb1dd83469bc/sampler_weights/000035",
        "tinker://5e3b3bea-32f6-4b9c-85b3-bb1dd83469bc/sampler_weights/000036",
        "tinker://5e3b3bea-32f6-4b9c-85b3-bb1dd83469bc/sampler_weights/000037",
        "tinker://5e3b3bea-32f6-4b9c-85b3-bb1dd83469bc/sampler_weights/000038",
        "tinker://5e3b3bea-32f6-4b9c-85b3-bb1dd83469bc/sampler_weights/000039",
        "tinker://5e3b3bea-32f6-4b9c-85b3-bb1dd83469bc/sampler_weights/000040",
        "tinker://5e3b3bea-32f6-4b9c-85b3-bb1dd83469bc/sampler_weights/000041",
        "tinker://5e3b3bea-32f6-4b9c-85b3-bb1dd83469bc/sampler_weights/000042",
        "tinker://5e3b3bea-32f6-4b9c-85b3-bb1dd83469bc/sampler_weights/000043",
        "tinker://5e3b3bea-32f6-4b9c-85b3-bb1dd83469bc/sampler_weights/000044",
        "tinker://5e3b3bea-32f6-4b9c-85b3-bb1dd83469bc/sampler_weights/000045",
        "tinker://5e3b3bea-32f6-4b9c-85b3-bb1dd83469bc/sampler_weights/000046",
        "tinker://5e3b3bea-32f6-4b9c-85b3-bb1dd83469bc/sampler_weights/000047",
        "tinker://5e3b3bea-32f6-4b9c-85b3-bb1dd83469bc/sampler_weights/000048",
    ],
}


RENDERER_NAMES: dict[str, str] = {
    "openai/gpt-oss-120b": "gpt_oss_low_reasoning",
    "Qwen/Qwen3-235B-A22B-Instruct-2507": "qwen3_disable_thinking",
}


# TASKS = [reward_hacking_shutdown(), reward_hacking_shutdown_2(), reward_hacking_re_bench()]
TASKS = [
    impossible_swebench(
        split="conflicting",
        agent_type="minimal",
        human_intervention=False,
        allow_internet=False,
        reset_tests=False,
        hide_tests=False,
        limit=16,
    ),
    # impossible_livecodebench(
    #     split="conflicting",
    #     agent_type="minimal",
    #     allow_test_modifications=True,
    #     sandbox="docker",
    #     limit=16,
    # ),
]


async def main() -> None:
    epochs = 1

    for model_name, model_paths in MODEL_PATHS.items():
        results: dict[str, list[EvalResult]] = {}
        for task in TASKS:
            results[task.name] = []
            for model_path in model_paths[::4][:1]:
                results[task.name].append(
                    await run_eval(
                        model_path=model_path,
                        task=task,
                        epochs=epochs,
                        renderer_name=RENDERER_NAMES[model_name],
                    )
                )
        plot_eval_results(results, model_name=model_name)


def _binomial_std_error(p: float, n: int) -> float:
    """Calculate standard error for a proportion."""
    if n == 0:
        return 0.0
    return sqrt(p * (1 - p) / n)


def plot_eval_results(results: dict[str, list[EvalResult]], model_name: str) -> None:
    fig = Figure()
    fig.update_layout(
        title=model_name,
        xaxis=dict(title="RL epoch"),
        yaxis=dict(title="fraction reward hacks", range=[0, 1]),
    )
    for i_task, (task_name, task_results) in enumerate(results.items()):
        fig.add_scatter(
            x=[4 * epoch for epoch in range(len(task_results))],
            y=[result.fraction_succeeded for result in task_results],
            error_y=dict(
                type="data",
                array=[
                    _binomial_std_error(r.fraction_succeeded, r.sample_size) for r in task_results
                ],
                visible=True,
            ),
            name=f"{task_name} succeeded",
            line=dict(color=DEFAULT_PLOTLY_COLORS[i_task]),
        )
        fig.add_scatter(
            x=[4 * epoch for epoch in range(len(task_results))],
            y=[result.fraction_unclear for result in task_results],
            error_y=dict(
                type="data",
                array=[
                    _binomial_std_error(r.fraction_unclear, r.sample_size) for r in task_results
                ],
                visible=True,
            ),
            name=f"{task_name} unclear",
            line=dict(color=DEFAULT_PLOTLY_COLORS[i_task], dash="dash"),
        )
    fig.show()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
