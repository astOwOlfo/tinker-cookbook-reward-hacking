from dotenv import load_dotenv
from pathlib import Path
import os
from os.path import join
import asyncio

from inspect_ai.log import EvalLog
from inspect_ai.dataset import Sample
from inspect_ai import eval
from control_arena.settings.bashbench2.bb2_setting import BashBench2Setting
from control_arena import EvalMode
from control_arena.eval import ControlEvalConfig, get_control_task_product
from control_arena.micro_protocol import defer_to_resample
from control_arena.scorers import max_suspicion_scorer
from control_arena.settings.iac.control_iac_setting import ControlIACSetting


from tinker_cookbook import cli_utils
from tinker_cookbook.rl import train
from tinker_cookbook.rl.envs.inspect_multi_task import InspectRLDatasetBuilder


def build_config_bash_bench_2() -> train.Config:
    # Load from HuggingFace
    setting = BashBench2Setting()

    # Or use local tasks
    from control_arena.settings.bashbench2.task.dataset import DatasetSpec

    setting = BashBench2Setting()

    tasks = get_control_task_product(
        settings=[setting], micro_protocols=[defer_to_resample], modes=[EvalMode.HONEST]
    )

    print(f"{tasks=}")


async def main() -> None:
    config = build_config_bash_bench_2()


if __name__ == "__main__":
    asyncio.run(main())
