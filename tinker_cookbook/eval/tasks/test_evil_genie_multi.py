"""
Example script demonstrating evaluate_multiple_models function.

This shows how to evaluate multiple models on the EvilGenie benchmark
using OpenAI-compatible APIs.
"""

import os
from tinker_cookbook.eval.tasks.evil_genie import evaluate_multiple_models


def main():
    """Run example multi-model evaluation."""
    # Example: Evaluate models using OpenAI API
    # Replace with your actual model names and API configurations

    # Get API keys from environment
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    if not openai_key:
        print("Warning: OPENAI_API_KEY not set. This example requires OpenAI API access.")
        print("Set the environment variable and run again.")
        return

    # Configure models to evaluate
    # For OpenAI models, use the standard base URL
    model_names = ["gpt-4o-mini"]
    openai_base_urls = ["https://api.openai.com/v1"]
    openai_api_keys = [openai_key]

    print("Starting multi-model evaluation...")
    print(f"Models: {model_names}")
    print(f"Max datapoints per eval: 2")
    print()

    # Run evaluation
    results = evaluate_multiple_models(
        model_names=model_names,
        openai_base_urls=openai_base_urls,
        openai_api_keys=openai_api_keys,
        max_datapoints_per_eval=2,  # Test with just 2 problems per config
    )

    # Print detailed results
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)

    for model_name in model_names:
        print(f"\nModel: {model_name}")
        print("-" * 40)

        # Get all eval results for this model
        model_results = {
            eval_name: summary
            for (m, eval_name), summary in results.items()
            if m == model_name
        }

        # Sort by eval name
        for eval_name in sorted(model_results.keys()):
            summary = model_results[eval_name]
            if summary.sample_size > 0:
                print(f"  {eval_name:20s}: {summary.reward_hacking_fraction:5.1%} reward hacking "
                      f"({summary.sample_size} problems)")
            else:
                print(f"  {eval_name:20s}: No problems evaluated")

    # Calculate overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)

    for model_name in model_names:
        model_results = [
            (eval_name, summary)
            for (m, eval_name), summary in results.items()
            if m == model_name
        ]

        total_problems = sum(s.sample_size for _, s in model_results)
        total_reward_hacking = sum(
            s.reward_hacking_fraction * s.sample_size for _, s in model_results
        )
        overall_rate = total_reward_hacking / total_problems if total_problems > 0 else 0.0

        print(f"\n{model_name}:")
        print(f"  Total problems: {total_problems}")
        print(f"  Total reward hacking: {int(total_reward_hacking)}")
        print(f"  Overall rate: {overall_rate:.1%}")


if __name__ == "__main__":
    main()
