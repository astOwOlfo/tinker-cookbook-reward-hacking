# Rubric Environment

A model-graded RL environment for studying reward hacking in diverse, mostly non-SWE tasks.

## Overview

Unlike test-suite-graded environments (SWE-Smith, BashApps, etc.) where hacking means modifying tests or hardcoding outputs, the Rubric Environment uses an **agentic LLM grader** to evaluate submissions against a detailed rubric. This shifts the hack surface toward more concerning behaviors: deception, metric gaming, adversarial manipulation of the grader, and optimizing for appearance over substance.

## How It Works

1. **The model** receives a task (essay, bash challenge, data analysis, etc.) along with the full rubric it will be graded on, and works in a Docker container with configurable tools.
2. **The grader** (an agentic LLM, e.g. Gemini Flash) inspects the container — reading files, running commands — and scores the submission on each rubric category.
3. **The reward** is the average of all category scores, normalized to [0, 1].

## Scoring

Each rubric has N categories. Each category is scored on a 4-point scale:

| Level | Score | Meaning |
|-------|-------|---------|
| Failure | 0 | Did not meet the requirement |
| Minor Failure | 1 | Partially met with significant issues |
| Minor Success | 2 | Mostly met with minor issues |
| Success | 3 | Fully met the requirement |

**Final reward** = mean(category scores) / 3.0

## Datapoint Structure

Each datapoint specifies:

- **`problem_statement`**: The task description shown to the model
- **`rubric`**: Tuple of `RubricCategory` objects, each with descriptions of all 4 score levels
- **`submission_instructions`**: How to submit (e.g., "write to /testbed/essay.txt")
- **`available_tools`**: Which tools the agent gets (controls capability). Use `("submit_solution",)` for effectively one-shot tasks, or `("bash", "create_file", "edit", "finish")` for full agent access
- **`necessary_files`**: Dict of `{filepath: content}` uploaded to the container at startup

## Example Tasks

| Task | Type | Tools | Hack Vectors |
|------|------|-------|-------------|
| Persuasive Essay | Writing | bash, create_file, finish | Grader manipulation, vacuous-but-impressive text |
| Bash Golf | Shell scripting | bash, create_file, finish | Fake filesystem state, inspect grading infra |
| Data Analysis Report | Analysis | bash, create_file, edit, finish | Fabricate statistics, embed grader instructions |

## Toggles

| Toggle | Values | Effect |
|--------|--------|--------|
| `reward_hacking_prompt` | nothing/allow/must/forbid/forbid-with-hints | Controls whether the model is encouraged, forbidden, or uninstructed about gaming |
| `available_tools` (per-datapoint) | Subset of tool names | Controls agent capability; `("submit_solution",)` = one-shot |
| `max_steps` | int | Agent turn limit |
| `grader_model` | str | Which LLM grades the submission |
| `max_grader_steps` | int | How many turns the grader gets to inspect |

## Architecture

```
rubric_env/
  __init__.py       # Re-exports
  config.py         # RubricEnvConfig (frozen dataclass)
  dataset.py        # RubricCategory, RubricDatapoint, example problems
  env.py            # RubricEnv, GroupBuilder, Dataset, DatasetBuilder
  grader.py         # Agentic grader loop + score parsing
  prompts.py        # All prompt templates
```

Follows the standard `Env` / `EnvGroupBuilder` / `RLDataset` / `RLDatasetBuilder` pattern. Uses `ContainerStarter` for Docker management and `bash_agent.default_agent_step()` for the agent loop.

## Usage

```python
from rubric_env import RubricEnvConfig, RubricDatasetBuilder

cfg = RubricEnvConfig(
    reward_hacking_prompt="nothing",
    grader_model="gemini-2.0-flash",
    max_steps=10,
)

builder = RubricDatasetBuilder(
    batch_size=2,
    model_name_for_tokenizer="Qwen/Qwen2.5-Coder-32B-Instruct",
    renderer_name="qwen_coder",
    group_size=2,
    test_fraction=0.3,
    cfg=cfg,
)

train_dataset, test_dataset = await builder()
```

## Why This Environment Matters

The hack surface here is qualitatively different from test-suite envs:

- **Deception**: Writing text that sounds good but is wrong, or embedding hidden instructions for the grader
- **Metric gaming**: Hitting rubric criteria superficially without genuine quality
- **Grader manipulation**: Prompt injection attempts targeting the LLM grader
- **Appearance over substance**: Producing impressive-looking but vacuous output

These behaviors are more analogous to real-world alignment failures than test modification. A model that learns to deceive an LLM evaluator is exhibiting something more concerning than one that overwrites a test file.
