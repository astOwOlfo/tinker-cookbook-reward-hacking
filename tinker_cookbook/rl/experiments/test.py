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

from tinker_cookbook.eval.tasks.shutdown import reward_hacking_shutdown
from tinker_cookbook.eval.tasks.shutdown2 import reward_hacking_shutdown_2
from tinker_cookbook.eval.tasks.re_bench import reward_hacking_re_bench


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
    "Qwen/Qwen3-32B": [
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
    ],
}


RENDERER_NAMES: dict[str, str] = {
    "openai/gpt-oss-20b": "gpt_oss_low_reasoning",
    "openai/gpt-oss-120b": "gpt_oss_low_reasoning",
    "Qwen/Qwen3-235B-A22B-Instruct-2507": "qwen3",
    "Qwen/Qwen3-32B": "qwen3",
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
    epochs = 8
    eval_frequency = 16

    for model_name, model_paths in MODEL_PATHS.items():
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
        # fig.show()
        fig.write_html("fig.html")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
