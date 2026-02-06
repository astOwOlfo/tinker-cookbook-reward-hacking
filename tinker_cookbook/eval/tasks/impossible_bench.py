from impossiblebench import impossible_livecodebench
from impossiblebench.analysis import DataLoader
from inspect_ai import Task, eval
from inspect_ai.model import get_model


def impossible_bench_tasks(limit: int) -> list[Task]:
    return [
        impossible_livecodebench(
            split="conflicting",  # Use conflicting tests
            agent_type="tools",  # Full tool-based scaffold
            limit=limit,
            max_attempts=10,
            message_limit=50,
        )
    ]


def main() -> None:
    logs = eval(
        impossible_bench_tasks(limit=2),
        # model="anthropic/claude-sonnet-4-5",  # Using the model from user's CLAUDE.md
        model=get_model(
            "Qwen/Qwen3-Coder-Next-FP8",
            base_url="https://api.together.xyz/v1",
            api_key="eb64de35667c4e77b82b4cef314595b01048f965f06bc2640a1858418fd57dc3",
        ),
        log_dir="inspect-logs",
        fail_on_error=True,
    )

    print(f"{logs[0].location=}")

    loader = DataLoader(n_workers=4)
    loader.load_folder(logs[0].location)
    summary = loader.get_summary()
    print(f"{summary=}")


if __name__ == "__main__":
    main()
