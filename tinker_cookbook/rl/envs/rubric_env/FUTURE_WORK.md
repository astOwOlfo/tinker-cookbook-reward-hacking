# Future Work

Directions for extending and improving the Rubric Environment.

## Dataset Proliferation

### School of Reward Hacks (harder variants)

The [School of Reward Hacks](https://arxiv.org/abs/2508.17511) paper provides over 1,000 examples of reward hacking on short, self-contained tasks (poetry, simple coding, etc.). A key finding is that training on harmless reward-hacking tasks causes generalization to unrelated misaligned behaviors.

We should create **harder versions** of School of Reward Hacks problems adapted to the rubric format. Their existing tasks are short and low-stakes; we could:
- Increase task complexity (multi-step reasoning, longer outputs)
- Add rubric categories that are harder to game (require verifiable facts, cross-referencing)
- Create multi-turn variants where the grader can ask follow-up questions
- Design tasks where the "obvious hack" is easy but the grader has tools to verify

This would let us study whether the transfer-to-misalignment finding holds when the tasks are harder and the grading is more robust.

### GDPval (easier variants)

The [GDPval benchmark](https://arxiv.org/abs/2510.04374) evaluates frontier models on real-world economically valuable tasks across 44 occupations and 9 GDP sectors, with expert-validated rubrics.

We should create **easier versions** of GDPval tasks adapted to our format. Their tasks are professional-grade and very long; we could:
- Simplify the domain complexity (entry-level versions of professional tasks)
- Shorten expected output lengths
- Reduce the number of rubric categories per task
- Focus on tasks where there's a clear gap between "looks good" and "is actually good" (the interesting hack surface)

GDPval's automated grading service and open-sourced tasks provide a natural starting point for adaptation.

### More Task Types

The current 13 datapoints span essays, bash golf, data analysis, editorial writing, QA reports, incident analysis, and sales analysis. Priority additions:

- **Code review**: Given a PR diff, write a review. Rubric: accuracy of identified issues, missed bugs, false positives, tone
- **Technical writing**: Write documentation for a given API. Rubric: accuracy, completeness, clarity, examples
- **Mathematical proofs**: Prove a statement. Rubric: correctness, rigor, clarity, completeness
- **Creative writing**: Short story with constraints. Rubric: adherence to constraints, narrative quality, originality
- **Debugging**: Given buggy code and symptoms, identify the bug. Rubric: correct diagnosis, explanation quality, fix quality

### Topic Diversification

The editorial tasks currently all reuse the same EU vehicle ban topic. Future editorial tasks should cover different topics (e.g., healthcare policy, technology regulation, climate adaptation, education reform) to avoid thematic monotony and reduce the risk that a model memorizes topic-specific patterns rather than learning general editorial skills.

### Self-Contained Tasks Only (No Web Search)

All tasks must be fully self-contained: every piece of information the model needs must be provided in `necessary_files` or the prompt. Tasks should **never** require real web search or external knowledge retrieval, because:
- We cannot control what the model finds online, making rubric grading unreliable
- Tasks that depend on web content are not reproducible across time or geography
- The interesting signal is whether the model can read, analyze, and synthesize provided materials — not whether it can google things

### Factory Pattern for Proliferation

Each `make_*()` function is a parameterized factory. Tasks that differ only in their `necessary_files` content (names, numbers, scenarios) are ideal for proliferation. To add a new variant of an existing task type:
1. Call the factory with different parameters (e.g., different material names, different sales figures, different log patterns)
2. The rubric structure stays the same; only the "correct answers" change because they're derived from the input data
3. Add the result to `RUBRIC_DATASET`

This makes it cheap to grow the dataset from 13 to 50+ problems without designing new rubric structures.

## Grader Improvements

### Grader robustness

The LLM grader is the primary attack surface. Future work should:
- Study how often prompt injection in submissions actually affects grader scores
- Test different grader models for robustness (larger models may be harder to manipulate)
- Add "anti-injection" instructions to the grader prompt and measure their effectiveness
- Consider ensemble grading (multiple grader calls, take median) for critical evaluations

### Grader calibration

Currently each category is scored 0 to its `points` value independently. We should study:
- Inter-rater reliability: does the same grader give consistent scores across runs?
- Calibration: are scores distributed as expected, or does the grader cluster at extremes?
- Cross-model calibration: do different grader models produce comparable score distributions?

### Grader agentic depth

The grader currently can run bash commands. Extensions:
- Let the grader run Python scripts to verify statistical claims
- Let the grader access the internet to fact-check (for non-training runs)
- Give the grader access to diff tools to compare submissions against known-good outputs

## Toggle Extensions

### Monitor toggle

Add a `monitor_present` toggle that tells the model its submission may be audited by a human. Study whether models hack less when they believe they're being watched.

### Impossible tasks

Add tasks that are genuinely impossible to do well (e.g., write an essay defending an indefensible position with strong evidence). In these cases, the only path to high reward is gaming. This parallels the `impossible` toggle in BadSandbox.

### Rubric visibility

Toggle whether the model sees the full rubric, a summary, or nothing. Study whether rubric visibility enables more targeted gaming.

## Infrastructure

### Per-datapoint Docker images

Currently all datapoints share one base Docker image, with per-datapoint files uploaded via `necessary_files`. For tasks requiring specialized software (e.g., LaTeX for typesetting tasks, R for statistical tasks), we may want per-datapoint Dockerfiles. The `ContainerStarter` already supports heterogeneous `dockerfile_contents` lists — we just need to add a `dockerfile_contents` field to `RubricDatapoint` and wire it through `RubricDataset.get_batch()`.

### Caching and performance

The agentic grader adds latency. Optimizations to explore:
- Grader response caching for identical submissions (deterministic tasks)
- Parallel grading across groups
- Faster grader models for initial training, upgrading for final eval
- Grader step budget that adapts to task complexity
