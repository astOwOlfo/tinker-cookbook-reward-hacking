from impossiblebench import impossible_livecodebench, impossible_swebench
import os
from inspect_ai import eval_set
from inspect_ai.model import get_model
from inspect_ai.log import read_eval_log
from dotenv import load_dotenv


def _sample_passed(score_value) -> bool:
    print(f"_sample_passed {score_value=}")
    if isinstance(score_value, str):
        return {"C": True, "I": False}[score_value]
    return float(score_value) >= 1.0


def _count_passed(log) -> tuple[int, int]:
    """Return (n_passed, n_total), reading full log from disk if samples aren't loaded."""
    if not log.samples and log.location:
        log = read_eval_log(str(log.location))
    samples = log.samples or []
    n_passed = sum(
        1 for s in samples if s.scores and _sample_passed(next(iter(s.scores.values())).value)
    )
    return n_passed, len(samples)


def main() -> None:
    os.environ["OPENAI_API_KEY"] = os.environ["OPEN_ROUTER_API_KEY"]

    AGENT_TYPE = "tools"

    tasks = [
        impossible_swebench(
            split="original",
            agent_type=AGENT_TYPE,
            limit=64,
            max_attempts=2,
            message_limit=15
        )
    ]

    success, logs = eval_set(
        tasks=tasks,
        # model="openai/gpt-5.2",
        # model="openai/openai/gpt-oss-120b",
        model=get_model(
            "openai/openai/gpt-oss-120b",
            base_url="https://openrouter.ai/api/v1",
            # base_url="http://localhost:8000/v1/",
        ),
        log_dir="./logs/",
        max_tasks=16,
        max_connections=64,
        max_sandboxes=64,
        max_subprocesses=64,
        fail_on_error=False,
        log_dir_allow_dirty=True,
    )

    for log in logs:
        task_name = log.eval.task
        n_passed, n = _count_passed(log)

        print(f"{task_name=} {n_passed=} {n=}")


if __name__ == "__main__":
    load_dotenv()

    main()

# gpt oss through open router:
# task_name='swebench_original_minimal' n_passed=5 n=64
# without max_attempts and message_limit: task_name='swebench_original_tools' n_passed=12 n=64
