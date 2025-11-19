from inspect_ai import eval
from inspect_evals.mmlu import mmlu_0_shot

# Run eval on 8 samples
log = eval(mmlu_0_shot(), model="openai/gpt-4o-mini", limit=8)[0]

# Print each sample's score
print("\n=== Per-Sample Scores ===")
for i, sample in enumerate(log.samples, 1):
    # Get the actual score value (1.0 for correct, 0.0 for incorrect)
    score_value = list(sample.scores.values())[0].value
    result = "CORRECT" if score_value == 1.0 else "INCORRECT"
    print(f"Sample {i}: {result} (score={score_value})")

# Print aggregate
print(f"\n=== Aggregate ===")
for score in log.results.scores:
    print(f"{score.name}: {score.metrics}")