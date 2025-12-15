import asyncio
import logging
from dotenv import load_dotenv
from plotly.graph_objects import Figure
from plotly.colors import DEFAULT_PLOTLY_COLORS
from statistics import mean
from math import sqrt
from dataclasses import dataclass

import chz
from inspect_ai import Task, eval_async
from inspect_ai.log import EvalLog
import tinker
from inspect_ai.model import GenerateConfig as InspectAIGenerateConfig
from inspect_ai.model import Model as InspectAIModel
from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling

from impossiblebench import impossible_livecodebench, impossible_swebench

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

    return EvalResult(
        fraction_succeeded=mean(
            sample.scores[f"{task.name}_scorer"].value # type: ignore
            for sample in eval_log.samples # type: ignore
        ),
        fraction_unclear=mean(
            float(sample.scores[f"{task.name}_scorer"].metadata["escalation"] == "unclear") # type: ignore
            for sample in eval_log.samples # type: ignore
        ),
        sample_size=len(eval_log.samples) # type: ignore
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
    tasks: list[Task], models: list[InspectAIModel], max_connections: int
) -> list[EvalLog]:
    return await eval_async(
        tasks=tasks,
        model=models,
        # Never retry - the tinker SDK is doing this for us already
        retry_on_error=0,
        # Although Tinker sampling tries very hard to only throw unrecoverable failures,
        # the inspect evaluation can still fail if e.g. the parser returns an error for
        # a given sample.
        fail_on_error=False,
        log_dir="~/inspect-logs",
        max_connections=max_connections,
        max_tasks=1, # len(tasks) * len(models),
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


async def main(
    tasks: dict[str, Task],
    model_paths: list[str],
    eval_frequency: int,
    renderer_name: str,
    temperature: float = 1.0,
    max_tokens: int = 4096,
    max_connections: int = 512,
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
        tasks=list(tasks.values()), models=evaluated_models, max_connections=max_connections
    )

    eval_results: dict[TaskName, dict[Epoch, EvalResult]] = {
        task_name: {
            epoch: parse_eval_log(
                eval_log=eval_logs[len(evaluated_models) * i_task + epoch // eval_frequency],
                task=task,
            )
            for epoch in range(0, eval_frequency * len(evaluated_models), eval_frequency)
        }
        for i_task, (task_name, task) in enumerate(tasks.items())
    }

    plot(eval_results, title=plot_title, save_figure_filename=save_figure_filename)


MODEL_PATHS: list[str] = [
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000000",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000001",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000002",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000003",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000004",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000005",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000006",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000007",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000008",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000009",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000010",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000011",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000012",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000013",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000014",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000015",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000016",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000017",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000018",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000019",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000020",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000021",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000022",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000023",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000024",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000025",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000026",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000027",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000028",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000029",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000030",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000031",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000032",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000033",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000034",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000035",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000036",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000037",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000038",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000039",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000040",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000041",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000042",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000043",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000044",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000045",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000046",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000047",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000048",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000049",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000050",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000051",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000052",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000053",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000054",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000055",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000056",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000057",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000058",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000059",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000060",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000061",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000062",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000063",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000064",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000065",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000066",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000067",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000068",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000069",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000070",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000071",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000072",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000073",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000074",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000075",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000076",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000077",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000078",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000079",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000080",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000081",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000082",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000083",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000084",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000085",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000086",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000087",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000088",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000089",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000090",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000091",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000092",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000093",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000094",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000095",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000096",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000097",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000098",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000099",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000100",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000101",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000102",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000103",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000104",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000105",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000106",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000107",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000108",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000109",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000110",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000111",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000112",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000113",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000114",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000115",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000116",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000117",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000118",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000119",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000120",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000121",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000122",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000123",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000124",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000125",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000126",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000127",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000128",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000129",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000130",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000131",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000132",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000133",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000134",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000135",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000136",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000137",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000138",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000139",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000140",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000141",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000142",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000143",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000144",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000145",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000146",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000147",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000148",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000149",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000150",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000151",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000152",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000153",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000154",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000155",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000156",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000157",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000158",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000159",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000160",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000161",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000162",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000163",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000164",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000165",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000166",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000167",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000168",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000169",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000170",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000171",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000172",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000173",
    "tinker://81dba366-54bf-5261-9985-3306be8f005e:train:0/sampler_weights/000174",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000168",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000169",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000170",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000171",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000172",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000173",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000174",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000175",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000176",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000177",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000178",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000179",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000180",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000181",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000182",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000183",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000184",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000185",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000186",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000187",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000188",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000189",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000190",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000191",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000192",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000193",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000194",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000195",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000196",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000197",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000198",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000199",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000200",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000201",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000202",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000203",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000204",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000205",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000206",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000207",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000208",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000209",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000210",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000211",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000212",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000213",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000214",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000215",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000216",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000217",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000218",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000219",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000220",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000221",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000222",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000223",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000224",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000225",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000226",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000227",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000228",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000229",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000230",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000231",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000232",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000233",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000234",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000235",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000236",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000237",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000238",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000239",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000240",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000241",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000242",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000243",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000244",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000245",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000246",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000247",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000248",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000249",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000250",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000251",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000252",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000253",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000254",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000255",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000256",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000257",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000258",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000259",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000260",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000261",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000262",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000263",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000264",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000265",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000266",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000267",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000268",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000269",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000270",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000271",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000272",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000273",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000274",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000275",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000276",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000277",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000278",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000279",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000280",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000281",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000282",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000283",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000284",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000285",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000286",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000287",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000288",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000289",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000290",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000291",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000292",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000293",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000294",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000295",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000296",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000297",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000298",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000299",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000300",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000301",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000302",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000303",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000304",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000305",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000306",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000307",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000308",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000309",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000310",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000311",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000312",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000313",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000314",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000315",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000316",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000317",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000318",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000319",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000320",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000321",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000322",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000323",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000324",
    "tinker://3b80ed48-266c-511b-97a5-19a062d62352:train:0/sampler_weights/000325",
]


TASKS: dict[str, Task] = {
    "impossible_bench_livecodebench_conflicting": impossible_livecodebench(
        split="conflicting",
        agent_type="minimal",
        allow_test_modifications=True,
        sandbox="docker",
        limit=1,
    ),
    # "impossible_bench_livecodebench_one_off": impossible_livecodebench(
    #     split="oneoff",
    #     agent_type="minimal",
    #     allow_test_modifications=True,
    #     sandbox="docker",
    #     limit=1,
    # ),
    # "shutdown": reward_hacking_shutdown(epochs=1),
    # "shutdown_2": reward_hacking_shutdown_2(epochs=1),
    # "re_bench": reward_hacking_re_bench(epochs=1),
}

if __name__ == "__main__":
    load_dotenv()

    asyncio.run(
        main(
            tasks=TASKS,
            model_paths=MODEL_PATHS,
            eval_frequency=9999,
            renderer_name="qwen3",
            save_figure_filename="fig.html",
        )
    )
