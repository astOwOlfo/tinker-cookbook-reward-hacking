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

    impossible_bench_task_names = [
        "lcb_conflicting_canmod_minimal",
        "lcb_oneoff_canmod_minimal",
        "swebench_conflicting_minimal",
        "swebench_oneoff_minimal",
    ]
    if task.name in impossible_bench_task_names:
        accuracies = [
            value
            for name, value in metrics.items()
            if name.endswith("accuracy") or name.endswith("mean")
        ]
        print(f"{task.name=} {metrics=}")
        assert len(accuracies) == 1
        accuracy = accuracies[0]
        return EvalResult(
            fraction_succeeded=accuracy,
            fraction_unclear=0.0,
            fraction_none=1 - accuracy,
            sample_size=len(task.dataset) * epochs,
        )

    return EvalResult(
        fraction_succeeded=metrics[f"{task.name}/fraction_succeeded"],
        fraction_unclear=metrics[f"{task.name}/fraction_unclear"],
        fraction_none=metrics[f"{task.name}/fraction_none"],
        sample_size=len(task.dataset) * epochs,
    )


MODEL_PATHS: dict[str, list[str]] = {
    "openai/gpt-oss-20b": [
        "tinker://1153ea5b-c744-53cd-a453-df577b813292:train:0/sampler_weights/000000",
        "tinker://368f5e1f-ab1d-57b0-8739-38e7508630d9:train:0/sampler_weights/000000",
        "tinker://741dabbd-7c6a-5612-b48a-cf273784d687:train:0/sampler_weights/000000",
        "tinker://741dabbd-7c6a-5612-b48a-cf273784d687:train:0/sampler_weights/000001",
        "tinker://741dabbd-7c6a-5612-b48a-cf273784d687:train:0/sampler_weights/000002",
        "tinker://741dabbd-7c6a-5612-b48a-cf273784d687:train:0/sampler_weights/000003",
        "tinker://741dabbd-7c6a-5612-b48a-cf273784d687:train:0/sampler_weights/000004",
        "tinker://741dabbd-7c6a-5612-b48a-cf273784d687:train:0/sampler_weights/000005",
        "tinker://741dabbd-7c6a-5612-b48a-cf273784d687:train:0/sampler_weights/000006",
        "tinker://741dabbd-7c6a-5612-b48a-cf273784d687:train:0/sampler_weights/000007",
        "tinker://741dabbd-7c6a-5612-b48a-cf273784d687:train:0/sampler_weights/000008",
        "tinker://741dabbd-7c6a-5612-b48a-cf273784d687:train:0/sampler_weights/000009",
        "tinker://741dabbd-7c6a-5612-b48a-cf273784d687:train:0/sampler_weights/000010",
        "tinker://741dabbd-7c6a-5612-b48a-cf273784d687:train:0/sampler_weights/000011",
        "tinker://741dabbd-7c6a-5612-b48a-cf273784d687:train:0/sampler_weights/000012",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000000",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000001",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000002",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000003",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000004",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000005",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000006",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000007",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000008",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000009",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000010",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000011",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000012",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000013",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000014",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000015",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000016",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000017",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000018",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000019",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000020",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000021",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000022",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000023",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000024",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000025",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000026",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000027",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000028",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000029",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000030",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000031",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000032",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000033",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000034",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000035",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000036",
        "tinker://5e18ea50-38a2-5b0e-b1af-433c27c17b41:train:0/sampler_weights/000037",
    ],
}


RENDERER_NAMES: dict[str, str] = {
    "openai/gpt-oss-20b": "gpt_oss_low_reasoning",
    "openai/gpt-oss-120b": "gpt_oss_low_reasoning",
    "Qwen/Qwen3-235B-A22B-Instruct-2507": "qwen3_disable_thinking",
}


TASKS = [
    reward_hacking_shutdown(),
    reward_hacking_shutdown_2(),
    reward_hacking_re_bench(),
    impossible_livecodebench(
        split="conflicting",
        agent_type="minimal",
        allow_test_modifications=True,
        sandbox="docker",
        limit=16,
    ),
    impossible_livecodebench(
        split="oneoff",
        agent_type="minimal",
        allow_test_modifications=True,
        sandbox="docker",
        limit=16,
    ),
    # impossible_swebench(
    #     split="conflicting",
    #     agent_type="minimal",
    #     limit=1,
    # ),
    # impossible_swebench(
    #     split="oneoff",
    #     agent_type="minimal",
    #     limit=1,
    # ),
]
"""
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
"""


async def main() -> None:
    epochs = 4
    eval_frequency = 8

    for model_name, model_paths in MODEL_PATHS.items():
        if "Qwen" in model_name:
            continue
        results: dict[str, list[EvalResult]] = {}
        for task in TASKS:
            results[task.name] = []
            for model_path in model_paths[::eval_frequency]:
                impossible_bench_task_names = [
                    "lcb_conflicting_canmod_minimal",
                    "lcb_oneoff_canmod_minimal",
                    "swebench_conflicting_minimal",
                    "swebench_oneoff_minimal",
                ]
                print(f"{model_path=} {task=}")
                results[task.name].append(
                    await run_eval(
                        model_path=model_path,
                        task=task,
                        epochs=1 if task.name in impossible_bench_task_names else epochs,
                        renderer_name=RENDERER_NAMES[model_name],
                    )
                )
            plot_eval_results(results, model_name=model_name, eval_frequency=eval_frequency)


def _binomial_std_error(p: float, n: int) -> float:
    """Calculate standard error for a proportion."""
    if n == 0:
        return 0.0
    return sqrt(p * (1 - p) / n)


def plot_eval_results(
    results: dict[str, list[EvalResult]], model_name: str, eval_frequency: int
) -> None:
    fig = Figure()
    fig.update_layout(
        title=model_name,
        xaxis=dict(title="RL epoch"),
        yaxis=dict(title="fraction reward hacks", range=[0, 1]),
    )
    for i_task, (task_name, task_results) in enumerate(results.items()):
        fig.add_scatter(
            x=[eval_frequency * epoch for epoch in range(len(task_results))],
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
            x=[eval_frequency * epoch for epoch in range(len(task_results))],
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
