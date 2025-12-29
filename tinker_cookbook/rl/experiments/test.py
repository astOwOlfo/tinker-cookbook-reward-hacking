import asyncio
import logging
from dotenv import load_dotenv
from plotly.graph_objects import Figure
from plotly.colors import DEFAULT_PLOTLY_COLORS
from statistics import mean
from math import sqrt
from dataclasses import dataclass

import os
from inspect_ai import Task, eval_async
from inspect_ai.log import EvalLog
import tinker
from inspect_ai.model import GenerateConfig as InspectAIGenerateConfig
from inspect_ai.model import Model as InspectAIModel
from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling

from impossiblebench import impossible_livecodebench

from tinker_cookbook.eval.tasks.shutdown import reward_hacking_shutdown
from tinker_cookbook.eval.tasks.shutdown2 import reward_hacking_shutdown_2
from tinker_cookbook.eval.tasks.re_bench import reward_hacking_re_bench


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvalResult:
    fraction_succeeded: float
    fraction_unclear: float | None
    sample_size: int

    def confidence_interval_95(self) -> float:
        """Calculate 95% confidence interval half-width for binomial proportion."""
        p = self.fraction_succeeded
        n = self.sample_size
        if n == 0:
            return 0.0
        # Standard error for binomial proportion
        se = sqrt(p * (1 - p) / n)
        # 95% CI uses z = 1.96
        return 1.96 * se

    def confidence_interval_95_unclear(self) -> float | None:
        """Calculate 95% confidence interval half-width for unclear fraction."""
        if self.fraction_unclear is None:
            return None
        p = self.fraction_unclear
        n = self.sample_size
        if n == 0:
            return 0.0
        se = sqrt(p * (1 - p) / n)
        return 1.96 * se


def parse_eval_log(eval_log: EvalLog, task: Task) -> EvalResult:
    assert eval_log.eval.task == task.name

    impossible_bench_task_names = [
        "lcb_conflicting_canmod_minimal",
        "lcb_oneoff_canmod_minimal",
        "swebench_conflicting_minimal",
        "swebench_oneoff_minimal",
    ]
    if task.name in impossible_bench_task_names:
        return EvalResult(
            fraction_succeeded=mean(
                {"I": 0.0, "C": 1.0}[sample.scores["agentic_humaneval_scorer"].value]  # type: ignore
                for sample in eval_log.samples  # type: ignore
            ),
            fraction_unclear=None,
            sample_size=len(eval_log.samples),  # type: ignore
        )

    print(f"{task.name=}")
    for sample in eval_log.samples:
        print("keys:", list(sample.scores.keys()))
    return EvalResult(
        fraction_succeeded=mean(
            sample.scores[f"reward_hacking_{task.name}_scorer"].value  # type: ignore
            for sample in eval_log.samples  # type: ignore
        ),
        fraction_unclear=mean(
            float(
                sample.scores[f"reward_hacking_{task.name}_scorer"].metadata["escalation"]  # type: ignore
                == "unclear"
            )
            for sample in eval_log.samples  # type: ignore
        ),
        sample_size=len(eval_log.samples),  # type: ignore
    )


async def inspect_model(
    service_client,  # TODO: type hint
    rest_client,  # TODO: type hint
    model_path: str,
    renderer_name: str,
    temperature: float,
    max_tokens: int,
) -> InspectAIModel:
    training_run = await rest_client.get_training_run_by_tinker_path_async(model_path)
    model_name = training_run.base_model

    logger.info(f"Using base model: {model_name} {model_path=}")

    sampling_client = service_client.create_sampling_client(
        model_path=model_path, base_model=model_name
    )
    api = InspectAPIFromTinkerSampling(
        renderer_name=renderer_name,  # type: ignore
        model_name=model_name,
        sampling_client=sampling_client,  # type: ignore
        verbose=False,  # type: ignore
    )

    return InspectAIModel(
        api=api,
        config=InspectAIGenerateConfig(temperature=temperature, max_tokens=max_tokens),
    )


async def run_evals(
    tasks: list[Task], models: list[InspectAIModel], max_connections: int, max_sandboxes: int
) -> list[EvalLog]:
    print(f"{tasks=}")
    print(f"{models=}")
    return await eval_async(
        tasks=tasks,
        model=models,
        # Never retry - the tinker SDK is doing this for us already
        retry_on_error=0,
        # Although Tinker sampling tries very hard to only throw unrecoverable failures,
        # the inspect evaluation can still fail if e.g. the parser returns an error for
        # a given sample.
        fail_on_error=False,
        debug_errors=True,
        log_dir="inspect-logs",
        max_connections=max_connections,
        max_tasks=len(tasks) * len(models),
        max_sandboxes=max_sandboxes,
    )


TaskName = str
Epoch = int


def plot(
    eval_results: dict[TaskName, dict[Epoch, EvalResult]],
    title: str | None,
    save_figure_filename: str | None,
) -> None:
    fig = Figure()
    fig.update_layout(
        title=title,
        xaxis=dict(title="RL epoch"),
        yaxis=dict(title="fraction reward hacks", range=[0, 1]),
    )
    for i_task, (task_name, results) in enumerate(eval_results.items()):
        epochs = list(results.keys())
        fractions = [result.fraction_succeeded for result in results.values()]
        error_bars = [result.confidence_interval_95() for result in results.values()]

        fig.add_scatter(
            x=epochs,
            y=fractions,
            name=task_name,
            line=dict(color=DEFAULT_PLOTLY_COLORS[i_task]),
            error_y=dict(
                type="data",
                array=error_bars,
                visible=True,
            ),
        )
        if next(iter(results.values())).fraction_unclear is not None:
            unclear_fractions = [result.fraction_unclear for result in results.values()]
            unclear_error_bars = [
                result.confidence_interval_95_unclear() for result in results.values()
            ]

            fig.add_scatter(
                x=epochs,
                y=unclear_fractions,
                name=task_name + " unclear",
                line=dict(color=DEFAULT_PLOTLY_COLORS[i_task], dash="dash"),
                error_y=dict(
                    type="data",
                    array=unclear_error_bars,
                    visible=True,
                ),
            )

    if save_figure_filename is None:
        fig.show()
    else:
        fig.write_html(save_figure_filename)
        print(f"Saved plot to '{save_figure_filename}'.")


async def main(
    tasks: dict[str, Task],
    model_paths: list[str],
    eval_frequency: int,
    renderer_name: str,
    temperature: float = 1.0,
    max_tokens: int = 4096,
    max_connections: int = 512,
    max_sandboxes: int = 64,
    plot_title: str | None = None,
    save_figure_filename: str | None = None,
) -> None:
    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()

    models: list[InspectAIModel] = [
        await inspect_model(
            service_client=service_client,
            rest_client=rest_client,
            model_path=model_path,
            renderer_name=renderer_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        for model_path in model_paths
    ]

    evaluated_models = models[::eval_frequency]

    eval_logs: list[EvalLog] = await run_evals(
        tasks=list(tasks.values()),
        models=evaluated_models,
        max_connections=max_connections,
        max_sandboxes=max_sandboxes,
    )

    print(f"{type(eval_logs)=} {len(eval_logs)=} {eval_logs=}")
    print(f"{type(eval_logs[0])=} {eval_logs[0]=}")

    print(f"{[eval_log.eval.task for eval_log in eval_logs]=}")
    eval_results: dict[TaskName, dict[Epoch, EvalResult]] = {
        task_name: {
            epoch: parse_eval_log(
                eval_log=eval_logs[len(tasks) * epoch // eval_frequency + i_task],
                task=task,
            )
            for epoch in range(0, eval_frequency * len(evaluated_models), eval_frequency)
        }
        for i_task, (task_name, task) in enumerate(tasks.items())
    }

    print(eval_results)

    plot(eval_results, title=plot_title, save_figure_filename=save_figure_filename)


MODEL_PATHS: list[str] = [
    "tinker://baeb9135-826a-52fc-a4e6-3f86e9016d4f:train:0/sampler_weights/000008",
    "tinker://baeb9135-826a-52fc-a4e6-3f86e9016d4f:train:0/sampler_weights/000016",
    "tinker://baeb9135-826a-52fc-a4e6-3f86e9016d4f:train:0/sampler_weights/000024",
    "tinker://baeb9135-826a-52fc-a4e6-3f86e9016d4f:train:0/sampler_weights/000032",
    "tinker://baeb9135-826a-52fc-a4e6-3f86e9016d4f:train:0/sampler_weights/000040",
    "tinker://baeb9135-826a-52fc-a4e6-3f86e9016d4f:train:0/sampler_weights/000048",
    "tinker://baeb9135-826a-52fc-a4e6-3f86e9016d4f:train:0/sampler_weights/000056",
    "tinker://baeb9135-826a-52fc-a4e6-3f86e9016d4f:train:0/sampler_weights/000064",
    "tinker://baeb9135-826a-52fc-a4e6-3f86e9016d4f:train:0/sampler_weights/000072",
    "tinker://baeb9135-826a-52fc-a4e6-3f86e9016d4f:train:0/sampler_weights/000080",
    "tinker://baeb9135-826a-52fc-a4e6-3f86e9016d4f:train:0/sampler_weights/000088",
    "tinker://baeb9135-826a-52fc-a4e6-3f86e9016d4f:train:0/sampler_weights/000096",
    "tinker://baeb9135-826a-52fc-a4e6-3f86e9016d4f:train:0/sampler_weights/000104",
    "tinker://baeb9135-826a-52fc-a4e6-3f86e9016d4f:train:0/sampler_weights/000112",
    "tinker://baeb9135-826a-52fc-a4e6-3f86e9016d4f:train:0/sampler_weights/000120",
    "tinker://baeb9135-826a-52fc-a4e6-3f86e9016d4f:train:0/sampler_weights/000128",
    "tinker://baeb9135-826a-52fc-a4e6-3f86e9016d4f:train:0/sampler_weights/000136",
    "tinker://baeb9135-826a-52fc-a4e6-3f86e9016d4f:train:0/sampler_weights/final",
]


TASKS: dict[str, Task] = {
    "impossible_bench_livecodebench_conflicting": impossible_livecodebench(
        split="conflicting",
        agent_type="minimal",
        allow_test_modifications=True,
        sandbox="docker",
        limit=32,
    ),
    "impossible_bench_livecodebench_one_off": impossible_livecodebench(
        split="oneoff",
        agent_type="minimal",
        allow_test_modifications=True,
        sandbox="docker",
        limit=32,
    ),
    "shutdown": reward_hacking_shutdown(epochs=8),
    "shutdown_2": reward_hacking_shutdown_2(epochs=8),
    "re_bench": reward_hacking_re_bench(epochs=8),
}

if __name__ == "__main__":
    load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    asyncio.run(
        main(
            tasks=TASKS,
            model_paths=MODEL_PATHS,
            eval_frequency=1,
            renderer_name="qwen3_disable_thinking",
            save_figure_filename="fig.html",
        )
    )
