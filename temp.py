from os import environ

from tinker_cookbook.eval.tasks.evil_genie import run_evil_genie

results = run_evil_genie(
    model_names=["claude-3-7-sonnet-20250219"],
    base_urls=[""],
    api_keys=[environ["ANTHROPIC_API_KEY"]],
    max_datapoints_per_variant=8,
)

print(results)
