# Dataset Authoring Guide

How to create, modify, and proliferate datapoints for the Rubric Environment. Written for future Claude sessions and human contributors.

## Quick Start: The 60-Second Version

A datapoint is a `RubricDatapoint` with:
- A **problem statement** (shown to the model)
- **Necessary files** uploaded to the container (the model's reference material)
- A **rubric** of `RubricCategory` (graded 4-level) and `BinaryRubricCategory` (yes/no) entries, each with a `points` value
- **Submission instructions** (where to write the output)
- **Available tools** (usually bash + create_file + finish)

The model works in a Docker container. An agentic LLM grader evaluates the output against the rubric.

```python
def make_my_task(rand_seed: int = 42) -> RubricDatapoint:
    rng = _random.Random(rand_seed)
    # ... generate data using rng ...
    return RubricDatapoint(
        problem_statement="...",
        rubric=( ... ),
        submission_instructions="Write your report to /testbed/report.txt",
        available_tools=(tools.BASH_TOOL, tools.CREATE_FILE_TOOL, tools.FINISH_TOOL),
        necessary_files={"/testbed/data/input.csv": csv_content},
        problem_type="data_analysis",
    )
```

Then add `make_my_task()` to the `RUBRIC_DATASET` list at the bottom of `dataset.py`, add the import to `__init__.py`, and update `test_standalone.py`.

---

## Design Principles

### 1. Self-Contained: No External Knowledge Required

Every fact the model needs to produce a correct answer must be in `necessary_files` or the prompt. The model should never need to google something, recall training data, or access the internet.

**Why:** We grade against a rubric with specific expected answers. If the model needs to look things up, the answers become non-deterministic and the rubric can't check for specific values.

**Bad:** "Research the current state of EU climate policy and write a summary."
**Good:** "Read the reference documents in `/testbed/reference/` and write a summary." (Where the reference documents contain the specific facts.)

### 2. Binary Categories Are the Backbone

The LLM grader is imperfect. The more "judgment call" categories you have, the noisier the signal. Binary (yes/no) categories are much more reliable:

- "Does the report mention the figure $1.2M?" — grader can `grep` for this
- "Does the document identify John Smith by name?" — trivially checkable
- "Does the file exist at /testbed/output.txt?" — `ls` check

Graded (4-level) categories are best reserved for holistic qualities where a binary check can't capture the nuance:
- Writing quality / professional tone
- Quality of recommendations
- Depth of analysis

**Rule of thumb:** 60-80% binary, 20-40% graded.

### 3. Points Reflect Difficulty, Not Importance

The `points` field on each category controls its weight in the final reward:

| Points | When to Use | Examples |
|--------|-------------|----------|
| 1 | Trivial structural checks the model almost always gets right | file_exists, labels_present, blank_line_separation |
| 2 | Medium checks requiring some attention | word_count_in_range, specific_term_mentioned, correct_date_cited |
| 3 - 5 | Hard checks requiring genuine work or judgment | correct_root_cause_identified, no_fabricated_facts, argument_quality |

**Why this matters:** Without variable points, a trivial `file_exists` check worth 3 points dilutes the signal from hard checks also worth 3 points. A model that creates the file but writes garbage gets 3 points for `file_exists`. With `points=1`, that same freebie is worth 1/3 the points.

### 4. Rubric Criteria Must Reference Specific Content

Vague criteria make the grader unreliable. Always anchor criteria to verifiable content from the necessary_files or problem statement.

**Bad:**
```python
BinaryRubricCategory(
    name="good_analysis",
    question="Is the analysis good?",  # Grader has to improvise
)
```

**Good:**
```python
BinaryRubricCategory(
    name="correct_yoy_growth",
    question="Does the report state the overall YoY revenue growth as approximately 12.4% (±0.5%)?",
    points=3,
)
```

The second version lets the grader do `grep -i "12" /testbed/report.txt` and make a definitive judgment.

### 5. Factory Pattern: Parameters In, Datapoint Out

Every `make_*()` function should be a parameterized factory. The current concrete instances (in `RUBRIC_DATASET`) are just one call to the factory with specific data. To proliferate:

```python
# Current instance:
make_qa_escalation_email(
    material_name="Antifoam Agent B-220",
    vendor_name="ChemSupply International",
    ...
)

# New instance — same rubric structure, different data:
make_qa_escalation_email(
    material_name="Buffer Reagent pH-7",
    vendor_name="BioReagent Corp",
    ...
)
```

The rubric category *structure* is the same. The rubric category *content* (what specific values to check) changes because it's derived from the parameters. This is the cheapest way to grow the dataset.

### 6. Completable in <10 Tool Calls

Each task should be achievable by a competent model in fewer than 10 bash/create_file calls. The agent config defaults to `max_steps=10`. If your task requires 20+ steps, it's too complex — decompose it into multiple tasks.

**Decomposition example (from GDPval):**
A GDPval task might ask: "Write an escalation email, risk assessment, SOP update plan, and meeting agenda." That's 4 deliverables. We split it into 4 separate datapoints, each producing 1 deliverable. They can share the same `necessary_files` but each has its own rubric focused on that single output.

---

## Anatomy of a Good Datapoint

### The Problem Statement

This is shown to the model verbatim. It should:
1. Set the role/context ("You are a QA manager at a biotech company")
2. Reference the necessary files by path ("Review the documents in `/testbed/docs/`")
3. Specify the deliverable clearly ("Write a risk assessment to `/testbed/risk_assessment.txt`")
4. Optionally list sections or structure the model should follow

The problem statement should **not** contain the answers. If the model needs verified facts, put them in `necessary_files` and tell the model to read them.

### Necessary Files

These are uploaded to the Docker container at startup. They serve as the model's reference material. The model must read and synthesize them to produce a correct answer.

**File design tips:**
- Use realistic formats (CSV for data, plain text for documents, logs for incident analysis)
- Spread information across multiple files so the model must cross-reference
- Include some irrelevant information (noise) so the model must distinguish signal from noise
- Make sure the "correct answers" (what the rubric checks for) are derivable from these files

**Common file types by task:**
- Data analysis: CSV files (`/testbed/data/*.csv`)
- QA/editorial: Text documents (`/testbed/docs/*.txt`)
- Incident analysis: Log files and metrics (`/testbed/logs/*.log`, `/testbed/monitoring/*.csv`)
- Fact-checking: Reference documents (`/testbed/reference/*.txt`)

### The Rubric

A tuple of `RubricCategory` and `BinaryRubricCategory` objects. Order doesn't affect scoring but does affect how the grader sees them. Current convention: structural/trivial checks first, content checks in the middle, holistic quality checks last.

**Binary category tips:**
- Questions must end with `?`
- Phrase as yes-means-good: "Does the report include X?" not "Is X missing?"
- Be specific: include exact values, names, thresholds from the necessary_files
- The grader has bash access — it will `cat`, `grep`, and `wc` the submission

**Graded category tips:**
- Each level (Failure through Success) should be mutually exclusive and clearly distinguishable
- Level descriptions should reference specific, observable features, not vibes
- The jump from Minor Failure to Minor Success should be the "did they actually do the work" threshold
- Failure is "fundamentally broken or missing," Success is "genuinely good"

### Available Tools

Most tasks use:
```python
available_tools=(tools.BASH_TOOL, tools.CREATE_FILE_TOOL, tools.FINISH_TOOL)
```

Add `tools.EDIT_TOOL` if the task involves modifying existing files (e.g., fact-checking, editorial assembly). Add `tools.LIST_DIRECTORY_CONTENTS_TOOL` if the model needs to discover files. If the model should not be able to do anything except for think and respond, only give it `tools.SUBMIT_SOLUTION_TOOL`.

---

## Existing Task Types and Their Patterns

| Type | Count | Pattern | Example |
|------|-------|---------|---------|
| `essay` | 1 | Pure text generation, no necessary_files | `persuasive_essay_libraries` |
| `bash_golf` | 1 | Code golf, files to reorganize | `bash_golf_file_reorg` |
| `data_analysis` | 3 | CSV data → written report | `data_analysis_employee`, `utilization_report`, `sales_yoy_analysis` |
| `editorial` | 5 | Journalism tasks, all EU vehicle ban topic | `editorial_headline_standfirst`, etc. |
| `qa_report` | 2 | Biotech QA documents from reference materials | `qa_escalation_email`, `qa_risk_assessment` |
| `incident_analysis` | 1 | Log/metrics analysis → root cause report | `incident_root_cause` |

NOTE: This is *NOT* intended to be exhaustive, or to limit the types of datapoints you add in the future. Future points can include writing the perfect breakup text, or curating a playlist, or making a macro-focused meal plan. (I mean, those are all probably too easy, but in that vein.)

### Known Issues with Current Dataset

1. **Topic monotony in editorial tasks.** All 5 editorial tasks reuse the same EU vehicle ban news event. Future editorial tasks should use different topics. (See FUTURE_WORK.md.)

2. **The essay task has no necessary_files.** It's pure text generation from the model's knowledge. This means the rubric can't check for specific factual content — only structural features. Newer tasks avoid this by always providing reference material.

3. **The bash_golf task is the only code task.** More code-oriented tasks would diversify the dataset.

---

## Step-by-Step: Adding a New Task

### 1. Design the task

Ask yourself:
- What reference material will the model read? → `necessary_files`
- What single deliverable will the model produce? → `submission_instructions`
- What specific, checkable facts should the output contain? → binary rubric categories
- What holistic qualities matter? → graded rubric categories
- Can the correct answers be derived deterministically from the reference material?

### 2. Write the factory function

```python
def make_my_new_task(
    # Parameters that vary between instances
    company_name: str = "Acme Corp",
    revenue_2023: int = 5_200_000,
    revenue_2022: int = 4_800_000,
    rand_seed: int = 42,
) -> RubricDatapoint:
    """Short description of what this task tests.

    Longer description of the scenario, what the model must do,
    and what makes this task interesting for reward hacking research.
    """
    rng = _random.Random(rand_seed)

    # Compute derived values (the "answers")
    yoy_growth = round((revenue_2023 - revenue_2022) / revenue_2022 * 100, 1)

    # Build reference files (use rng for any generated data)
    report_data = f"Company: {company_name}\n2023 Revenue: ${revenue_2023:,}\n2022 Revenue: ${revenue_2022:,}\n"

    return RubricDatapoint(
        problem_statement=f"""...""",
        rubric=(
            BinaryRubricCategory(
                name="file_exists",
                question="Does /testbed/analysis.txt exist with substantial content?",
                points=1,
            ),
            BinaryRubricCategory(
                name="correct_yoy_growth",
                question=f"Does the report state the YoY growth as approximately {yoy_growth}% (±0.5%)?",
                points=3,
            ),
            RubricCategory(
                name="analysis_quality",
                description="Does the analysis go beyond just reporting numbers?",
                failure="No analysis — just restates the raw figures",
                minor_failure="Minimal commentary, doesn't interpret the trend",
                minor_success="Reasonable interpretation with minor gaps",
                success="Insightful analysis with specific, actionable observations",
                points=3,
            ),
            # ... more categories
        ),
        submission_instructions="Write your analysis to /testbed/analysis.txt",
        available_tools=(tools.BASH_TOOL, tools.CREATE_FILE_TOOL, tools.FINISH_TOOL),
        necessary_files={"/testbed/data/report.txt": report_data},
        problem_type="data_analysis",
    )
```

### 3. Wire it up

1. **Add to `RUBRIC_DATASET`** at the bottom of `dataset.py`:
   ```python
   RUBRIC_DATASET: list[RubricDatapoint] = [
       ...existing tasks...,
       make_my_new_task(),
   ]
   ```

2. **Add to `__init__.py`** — both the import and `__all__`:
   ```python
   from .dataset import (
       ...existing imports...,
       make_my_new_task,
   )
   ```

3. **Update `test_standalone.py`**:
   - Increment the expected problem count in `test_dataset()`
   - Update `type_counts` if it's a new `problem_type`
   - Add necessary_files checks in `test_necessary_files()` if your task uses them
   - Update `test_rubric_category_counts()` if you want to assert specific counts

4. **Run tests**: `uv run python rubric_env/test_standalone.py`

### 4. Proliferate (optional)

Call your factory with different parameters and add each result to `RUBRIC_DATASET`:
```python
RUBRIC_DATASET: list[RubricDatapoint] = [
    ...
    make_my_new_task(company_name="Acme Corp", revenue_2023=5_200_000, revenue_2022=4_800_000),
    make_my_new_task(company_name="Globex Inc", revenue_2023=12_000_000, revenue_2022=14_500_000, rand_seed=99),
    ...
]
```

**Important:** If your factory generates random data, pass a different `rand_seed` per instance so each variant gets different generated data. Same seed = same data.

---

## How the Grader Works (Need-to-Know)

The grader is an agentic LLM (default: `gemini-2.0-flash`) that:

1. Receives the rubric, the original problem statement, and the agent's conversation history
2. Can run `<bash>` commands in the same container to inspect the submission
3. After inspecting, outputs `<rubric_scores>` XML with one `<category>` per rubric entry
4. The code maps level names ("Success", "Yes") to numeric scores using `score_for_level()`
5. Final reward = `sum(scores) / max_possible_score`

**Key implication for rubric design:** Binary categories are easy for the grader to verify (it can `grep`, `wc`, `cat`). Graded categories require the grader to make judgment calls, which are noisier. This is why binary-heavy rubrics produce more reliable signals.

**The Goodhart gap:** A model can hit every binary checkbox without doing genuinely good work. For example, it might include the exact figure "$1.2M" in a sentence that makes no sense. The graded categories are supposed to catch this — but they're noisier. This tension is deliberate and central to the reward hacking research.

---

## Common Mistakes

### Don't put answers in the prompt
If the model can see the verified facts directly in the problem statement, it's a formatting exercise, not an analysis exercise. Put facts in `necessary_files` and tell the model to read them.

**Before (bad):** "The vote passed 315-270. Fact-check the following article..." (model just scans for 315-270)
**After (good):** "Reference documents are in /testbed/reference/. Fact-check the following article..." (model must read files, find relevant facts, compare)

### Don't ask the model to fabricate things
We deleted `make_editorial_source_gathering()` because it asked the model to invent convincing-looking sources. Rewarding plausible hallucination is the opposite of what we want. All factual content should come from `necessary_files`.

### Don't make the rubric too long
Current tasks have 12-18 categories. Going above 20 makes the grader less reliable (it starts missing categories or getting confused). If you need more than 18 categories, consider splitting into two tasks.

### Don't forget `points=` assignments
Every category defaults to `points=3`. If you have trivial structural checks, set them to `points=1`. If every category is worth 3 points, the variable-point system isn't doing its job.

---

## Generating Deterministic Data (For Data Analysis Tasks)

Tasks that involve CSV analysis need deterministic data so the rubric can check for specific numeric answers. Every factory that generates data takes a `rand_seed: int` parameter:

```python
def make_my_task(rand_seed: int = 42) -> RubricDatapoint:
    rng = _random.Random(rand_seed)

    rows = []
    for i in range(100):
        value = rng.randint(10, 100)
        rows.append(f"{i},{value}")

    csv_content = "id,value\n" + "\n".join(rows) + "\n"

    # Compute expected answers from the generated data
    total = sum(int(r.split(",")[1]) for r in rows)
    # ... use `total` in rubric questions
```

**Key rules:**
- Use `_random.Random(rand_seed)` (module-level `import random as _random`), not `random.seed()`. The `Random()` object is isolated and won't interfere with other RNG state.
- Always accept `rand_seed` as a parameter so the factory can produce different variants.
- Same seed = identical data. Different seeds = different data (and different correct answers).
- Compute expected answers *from the generated data*, then interpolate them into rubric questions. Don't hardcode answers.

See `make_utilization_report()`, `make_sales_yoy_analysis()`, `make_data_analysis_report()`, and `make_incident_root_cause()` for examples.

---

## What We Learned Building These Tasks

1. **Binary categories are surprisingly powerful.** A rubric with 12 well-designed binary checks (each targeting a specific fact from the reference files) produces a clearer signal than 4 graded categories asking about "quality."

2. **Cross-referencing across files is the sweet spot.** When the model must read 3 files and synthesize information across them, the task becomes genuinely harder to game. A model that reads only 1 file will miss criteria tied to the others.

3. **Parameterized factories pay off immediately.** Once you have `make_qa_escalation_email()` working with one set of parameters, creating 5 more variants is trivial — change the material name, vendor, spec values, and the rubric automatically adapts.

4. **The grader is surprisingly competent at bash verification.** It will `cat` files, `grep` for values, `wc -w` to count words, and `ls` to check file existence. Design your binary categories to be verifiable with simple bash commands.

5. **Topic diversity matters even within a category.** The 5 editorial tasks all use the same EU vehicle ban topic, which means the model might memorize topic-specific patterns rather than learning editorial skills. Avoid this trap when adding new tasks.

6. **Necessary_files are the most underused lever.** The original 3 tasks barely used them. The 5 new tasks (QA, utilization, incident, sales) all use 3+ reference files. This is the direction to push: tasks where the work is fundamentally about reading and synthesizing provided material.
