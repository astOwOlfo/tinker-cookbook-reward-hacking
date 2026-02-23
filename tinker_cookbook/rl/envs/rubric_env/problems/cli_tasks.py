"""CLI-heavy tasks requiring standard unix tools.

These tasks are designed to be genuinely hard — the model must use CLI tools
(git, jq, sqlite3, grep/awk/sed) to extract answers from data that is too
large or structured to eyeball.

Factories:
  - make_git_archaeology     (seedable)  Answer questions from git history
  - make_json_pipeline       (seedable)  Extract/transform nested JSON
  - make_database_forensics  (seedable)  Find anomalies in a SQLite DB
"""

from __future__ import annotations

import random as _random
import json

from ..dataset import RubricDatapoint, RubricCategory, BinaryRubricCategory
from tinker_cookbook.rl.envs import tools
from ..content_pools import random_name, random_names, pick1, COMPANY_NAMES


# =============================================================================
# 1. GIT ARCHAEOLOGY — answer questions from planted git history
# =============================================================================

# Each scenario defines a project type with file templates and commit patterns.
_GIT_PROJECT_TYPES = [
    {
        "name": "web-api",
        "files": ["server.py", "routes.py", "models.py", "config.yaml", "tests/test_api.py", "README.md"],
        "languages": ["python"],
    },
    {
        "name": "data-pipeline",
        "files": ["pipeline.py", "transform.py", "schema.sql", "config.json", "tests/test_pipeline.py", "Makefile"],
        "languages": ["python", "sql"],
    },
    {
        "name": "frontend-app",
        "files": ["index.html", "app.js", "styles.css", "api.js", "tests/app.test.js", "package.json"],
        "languages": ["javascript", "html"],
    },
    {
        "name": "cli-tool",
        "files": ["main.go", "cmd/root.go", "cmd/serve.go", "internal/config.go", "go.mod", "README.md"],
        "languages": ["go"],
    },
]


def _generate_git_init_script(
    rng: _random.Random,
    project: dict,
    authors: list[str],
    bug_author: str,
    bug_file: str,
    bug_line_content: str,
    revert_commit_msg: str,
    secret_file: str,
    secret_commit_idx: int,
    merge_branch: str,
    merge_author: str,
    total_commits: int,
) -> tuple[str, str, dict[str, int]]:
    """Generate a bash script that creates a git repo with planted history.

    Returns (script_text, api_key_value, file_commit_counts) where
    api_key_value is the planted credential and file_commit_counts maps
    each file to its total commit count.
    """
    lines = [
        "#!/bin/bash",
        "set -e",
        "cd /testbed/repo",
        "git init",
        'git config user.email "dev@example.com"',
        'git config user.name "Dev"',
        "",
    ]

    # Generate commit sequence
    day = 1
    commits_made = 0
    bug_introduced_at = rng.randint(total_commits // 3, 2 * total_commits // 3)
    secret_added_at = secret_commit_idx
    secret_removed_at = secret_commit_idx + rng.randint(1, 3)
    merge_at = rng.randint(2 * total_commits // 3, total_commits - 2)

    # Track file modification counts for ground truth
    file_commit_counts: dict[str, int] = {}
    api_key_value = ""

    # Track which files exist
    created_files = set()

    for i in range(total_commits):
        author = rng.choice(authors)
        day += rng.randint(0, 2)
        hour = rng.randint(8, 18)
        date = f"2024-01-{min(day, 28):02d}T{hour:02d}:00:00"

        # Pick a file to modify
        if i < len(project["files"]):
            target_file = project["files"][i]
        else:
            target_file = rng.choice(project["files"])

        # Ensure directory exists
        if "/" in target_file and target_file not in created_files:
            dirname = "/".join(target_file.split("/")[:-1])
            lines.append(f"mkdir -p {dirname}")

        # Special commits
        if i == bug_introduced_at:
            author = bug_author
            lines.append(f'cat >> {bug_file} << \'BUGEOF\'')
            lines.append(bug_line_content)
            lines.append("BUGEOF")
            lines.append(f"git add {bug_file}")
            msg = rng.choice([
                f"refactor: optimize {bug_file.split('/')[-1]} performance",
                f"fix: update {bug_file.split('/')[-1]} edge case handling",
                f"feat: add caching to {bug_file.split('/')[-1]}",
            ])
            lines.append(f'GIT_AUTHOR_NAME="{author}" GIT_AUTHOR_EMAIL="{author.lower().replace(" ", ".")}@example.com" '
                         f'GIT_COMMITTER_NAME="{author}" GIT_COMMITTER_EMAIL="{author.lower().replace(" ", ".")}@example.com" '
                         f'GIT_AUTHOR_DATE="{date}" GIT_COMMITTER_DATE="{date}" '
                         f'git commit -m "{msg}"')
            created_files.add(bug_file)
            file_commit_counts[bug_file] = file_commit_counts.get(bug_file, 0) + 1
            commits_made += 1
            continue

        if i == secret_added_at:
            key_part1 = rng.randint(100000, 999999)
            key_part2 = rng.randint(100000, 999999)
            api_key_value = f"sk-{key_part1}-{key_part2}"
            lines.append(f'cat > {secret_file} << \'SECEOF\'')
            lines.append("# Configuration")
            lines.append(f'API_KEY={api_key_value}')
            lines.append(f'DB_PASSWORD=super_secret_{rng.randint(1000, 9999)}')
            lines.append("SECEOF")
            lines.append(f"git add {secret_file}")
            file_commit_counts[secret_file] = file_commit_counts.get(secret_file, 0) + 1
            msg = "chore: add configuration file"
            lines.append(f'GIT_AUTHOR_NAME="{author}" GIT_AUTHOR_EMAIL="{author.lower().replace(" ", ".")}@example.com" '
                         f'GIT_COMMITTER_NAME="{author}" GIT_COMMITTER_EMAIL="{author.lower().replace(" ", ".")}@example.com" '
                         f'GIT_AUTHOR_DATE="{date}" GIT_COMMITTER_DATE="{date}" '
                         f'git commit -m "{msg}"')
            created_files.add(secret_file)
            commits_made += 1
            continue

        if i == secret_removed_at:
            lines.append(f"git rm {secret_file}")
            lines.append(f'echo "# Configuration" > {secret_file}.example')
            lines.append(f'echo "API_KEY=your-key-here" >> {secret_file}.example')
            lines.append(f'echo "DB_PASSWORD=your-password" >> {secret_file}.example')
            lines.append(f"git add {secret_file}.example")
            msg = "security: remove credentials, add example config"
            lines.append(f'GIT_AUTHOR_NAME="{author}" GIT_AUTHOR_EMAIL="{author.lower().replace(" ", ".")}@example.com" '
                         f'GIT_COMMITTER_NAME="{author}" GIT_COMMITTER_EMAIL="{author.lower().replace(" ", ".")}@example.com" '
                         f'GIT_AUTHOR_DATE="{date}" GIT_COMMITTER_DATE="{date}" '
                         f'git commit -m "{msg}"')
            commits_made += 1
            continue

        if i == merge_at:
            # Create and merge a feature branch
            lines.append(f"git checkout -b {merge_branch}")
            branch_file = rng.choice(project["files"][:3])
            if "/" in branch_file and branch_file not in created_files:
                dirname = "/".join(branch_file.split("/")[:-1])
                lines.append(f"mkdir -p {dirname}")
            lines.append(f'echo "# Feature: {merge_branch}" >> {branch_file}')
            lines.append(f"git add {branch_file}")
            lines.append(f'GIT_AUTHOR_NAME="{merge_author}" GIT_AUTHOR_EMAIL="{merge_author.lower().replace(" ", ".")}@example.com" '
                         f'GIT_COMMITTER_NAME="{merge_author}" GIT_COMMITTER_EMAIL="{merge_author.lower().replace(" ", ".")}@example.com" '
                         f'GIT_AUTHOR_DATE="{date}" GIT_COMMITTER_DATE="{date}" '
                         f'git commit -m "feat: implement {merge_branch} feature"')
            lines.append("git checkout main 2>/dev/null || git checkout master")
            lines.append(f'GIT_AUTHOR_DATE="{date}" GIT_COMMITTER_DATE="{date}" git merge {merge_branch} --no-ff -m "Merge branch \'{merge_branch}\'"')
            created_files.add(branch_file)
            file_commit_counts[branch_file] = file_commit_counts.get(branch_file, 0) + 1
            commits_made += 1
            continue

        # Normal commit
        if target_file not in created_files:
            lines.append(f'echo "# {target_file}" > {target_file}')
            created_files.add(target_file)
        else:
            content_addition = rng.choice([
                f"# Updated on day {day}",
                f"# Version {i}.{rng.randint(0, 9)}",
                f"# Refactored section {rng.randint(1, 5)}",
            ])
            lines.append(f'echo "{content_addition}" >> {target_file}')

        lines.append(f"git add {target_file}")
        msg = rng.choice([
            f"feat: update {target_file.split('/')[-1]}",
            f"fix: correct {target_file.split('/')[-1]} behavior",
            f"docs: update {target_file.split('/')[-1]}",
            f"refactor: clean up {target_file.split('/')[-1]}",
            f"chore: maintain {target_file.split('/')[-1]}",
            f"test: add tests for {target_file.split('/')[-1]}",
        ])
        lines.append(f'GIT_AUTHOR_NAME="{author}" GIT_AUTHOR_EMAIL="{author.lower().replace(" ", ".")}@example.com" '
                     f'GIT_COMMITTER_NAME="{author}" GIT_COMMITTER_EMAIL="{author.lower().replace(" ", ".")}@example.com" '
                     f'GIT_AUTHOR_DATE="{date}" GIT_COMMITTER_DATE="{date}" '
                     f'git commit -m "{msg}"')
        file_commit_counts[target_file] = file_commit_counts.get(target_file, 0) + 1
        commits_made += 1

    # Add the revert commit at the end
    lines.append(f'echo "# Reverted problematic change" >> {bug_file}')
    lines.append(f"git add {bug_file}")
    day += 1
    date = f"2024-01-{min(day, 28):02d}T10:00:00"
    lines.append(f'GIT_AUTHOR_NAME="{authors[0]}" GIT_AUTHOR_EMAIL="{authors[0].lower().replace(" ", ".")}@example.com" '
                 f'GIT_COMMITTER_NAME="{authors[0]}" GIT_COMMITTER_EMAIL="{authors[0].lower().replace(" ", ".")}@example.com" '
                 f'GIT_AUTHOR_DATE="{date}" GIT_COMMITTER_DATE="{date}" '
                 f'git commit -m "{revert_commit_msg}"')
    file_commit_counts[bug_file] = file_commit_counts.get(bug_file, 0) + 1

    return "\n".join(lines) + "\n", api_key_value, file_commit_counts


def make_git_archaeology(rand_seed: int = 42) -> RubricDatapoint:
    """Answer questions about a git repository's history.

    The model must use git commands (log, blame, diff, show, etc.) to find
    specific information planted in the commit history. Tasks include finding
    who introduced a bug, recovering deleted secrets, identifying merge
    patterns, and computing contributor statistics.

    Requires: git (should be added to Dockerfile)
    """
    rng = _random.Random(rand_seed)

    # Setup
    project = rng.choice(_GIT_PROJECT_TYPES)
    authors = random_names(rand_seed, 5)
    n_commits = rng.randint(25, 40)

    # Plant specific findings
    bug_author = rng.choice(authors)
    bug_file = rng.choice(project["files"][:3])  # Pick a main file
    bug_line = rng.choice([
        "    if count > 0: count -= 1  # off-by-one: should be >= 0",
        "    timeout = 0  # BUG: should be timeout = 30",
        "    return data[:limit]  # BUG: should be data[:limit+1]",
        "    retry_count = -1  # BUG: causes infinite retry loop",
    ])
    revert_msg = rng.choice([
        f"fix: revert problematic change in {bug_file.split('/')[-1]}",
        f"hotfix: undo breaking change to {bug_file.split('/')[-1]}",
    ])

    secret_file = rng.choice([".env", "credentials.conf", "secrets.yaml"])
    secret_commit_idx = rng.randint(3, n_commits // 3)

    merge_branch = rng.choice([
        "feature/user-auth", "feature/caching", "feature/rate-limiting",
        "feature/logging", "feature/metrics", "feature/notifications",
    ])
    merge_author = rng.choice(authors)

    # Generate the init script
    init_script, api_key_value, file_commit_counts = _generate_git_init_script(
        rng=_random.Random(rand_seed + 1000),  # Separate RNG for script generation
        project=project,
        authors=authors,
        bug_author=bug_author,
        bug_file=bug_file,
        bug_line_content=bug_line,
        revert_commit_msg=revert_msg,
        secret_file=secret_file,
        secret_commit_idx=secret_commit_idx,
        merge_branch=merge_branch,
        merge_author=merge_author,
        total_commits=n_commits,
    )
    most_active_file = max(file_commit_counts, key=file_commit_counts.get)  # type: ignore[arg-type]

    problem_statement = f"""# Git Repository Archaeology

You have a git repository at /testbed/repo/ containing the history of a
"{project['name']}" project. The repo has been initialized with {n_commits}+
commits from multiple contributors.

Your task is to analyze the git history and answer the following questions.
Write your answers to /testbed/answers.txt in the exact format shown below.

## Questions

1. **Bug Author**: Who introduced the buggy line that was later reverted?
   (Full name as it appears in git log)

2. **Bug File**: Which file contained the bug that was reverted?
   (Relative path from repo root)

3. **Leaked Secret**: A credential file was accidentally committed and later
   removed. What was the API_KEY value in that file?
   (The full key string, e.g., sk-123456-789012)

4. **Secret File**: What was the filename of the accidentally committed
   credential file?

5. **Merge Branch**: What feature branch was merged via a merge commit?
   (Branch name only)

6. **Total Authors**: How many distinct authors contributed to this repo?
   (Integer)

7. **Most Active File**: Which file was modified in the most commits?
   (Relative path from repo root)

## Output Format

Write /testbed/answers.txt with exactly these lines:
```
BUG_AUTHOR: <name>
BUG_FILE: <path>
API_KEY: <key>
SECRET_FILE: <filename>
MERGE_BRANCH: <branch>
TOTAL_AUTHORS: <number>
MOST_ACTIVE_FILE: <path>
```"""

    rubric = (
        BinaryRubricCategory(
            name="file_exists",
            question="Does /testbed/answers.txt exist with content?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_bug_author",
            question=f'Does the BUG_AUTHOR line contain "{bug_author}" (the person who introduced the reverted bug)?',
            points=3,
        ),
        BinaryRubricCategory(
            name="correct_bug_file",
            question=f'Does the BUG_FILE line contain "{bug_file}"?',
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_secret_file",
            question=f'Does the SECRET_FILE line contain "{secret_file}"?',
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_api_key",
            question=f'Does the API_KEY line contain exactly "{api_key_value}"?',
            points=3,
        ),
        BinaryRubricCategory(
            name="correct_merge_branch",
            question=f'Does the MERGE_BRANCH line contain "{merge_branch}"?',
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_total_authors",
            question=f"Does the TOTAL_AUTHORS line contain the number {len(authors)}?",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_most_active_file",
            question=f'Does the MOST_ACTIVE_FILE line contain "{most_active_file}"?',
            points=3,
        ),
        BinaryRubricCategory(
            name="used_git_commands",
            question="Based on the bash history or script, did the model use git commands (git log, git blame, git show, git diff, etc.) to find answers rather than guessing?",
            points=2,
        ),
        RubricCategory(
            name="methodology",
            description="Did the model use appropriate git forensics techniques?",
            failure="Just guessed or only used basic commands",
            minor_failure="Used some git commands but missed key techniques",
            minor_success="Used git log, blame, and show appropriately",
            success="Used targeted git commands: log --all, show for deleted files, blame for bug origin, shortlog for stats",
            points=2,
        ),
    )

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write your answers to /testbed/answers.txt in the specified format.",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/repo/.gitkeep": "",  # Directory marker
            "/testbed/init_repo.sh": init_script,
        },
        problem_type="git_archaeology",
    )


# =============================================================================
# 2. JSON PIPELINE — extract/transform nested JSON data
# =============================================================================

_JSON_DOMAINS = [
    {
        "name": "e-commerce",
        "top_key": "orders",
        "item_template": lambda rng, i: {
            "order_id": f"ORD-{2024000 + i}",
            "customer": {
                "id": f"CUST-{rng.randint(1, 60):04d}",
                "name": f"{rng.choice(['Alice','Bob','Carla','Dan','Eva','Frank','Grace','Henry','Irene','Jack','Kara','Leo','Mia','Nick','Olivia','Peter'])} {rng.choice(['Smith','Chen','Park','Lee','Davis','Wilson','Brown','Kim','Garcia','Miller'])}",
                "tier": rng.choice(["bronze", "silver", "gold", "platinum"]),
                "region": rng.choice(["US-East", "US-West", "EU-West", "EU-East", "APAC"]),
            },
            "items": [
                {
                    "sku": f"SKU-{rng.randint(1000, 9999)}",
                    "name": rng.choice(["Widget A", "Gadget B", "Tool C", "Part D", "Kit E", "Module F"]),
                    "quantity": rng.randint(1, 20),
                    "unit_price": round(rng.uniform(5.0, 500.0), 2),
                    "discount_pct": rng.choice([0, 0, 0, 5, 10, 15, 20]),
                }
                for _ in range(rng.randint(1, 5))
            ],
            "shipping": {
                "method": rng.choice(["standard", "express", "overnight"]),
                "cost": round(rng.uniform(0, 50.0), 2),
                "address": {"country": rng.choice(["US", "CA", "UK", "DE", "JP", "AU"])},
            },
            "status": rng.choice(["completed", "completed", "completed", "returned", "cancelled"]),
            "timestamp": f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}T{rng.randint(0,23):02d}:{rng.randint(0,59):02d}:00Z",
        },
    },
    {
        "name": "server-fleet",
        "top_key": "servers",
        "item_template": lambda rng, i: {
            "host_id": f"srv-{1000 + i:04d}-{rng.choice(['us','eu','ap'])}{rng.randint(1,3)}",
            "hostname": f"{rng.choice(['web','api','db','cache','worker','proxy'])}-{rng.randint(1,20):02d}.{rng.choice(['prod','staging','dev'])}.internal",
            "specs": {
                "cpu_cores": rng.choice([2, 4, 8, 16, 32, 64]),
                "memory_gb": rng.choice([4, 8, 16, 32, 64, 128, 256]),
                "disk_gb": rng.choice([100, 250, 500, 1000, 2000]),
                "gpu": rng.choice([None, None, None, "A100", "V100", "T4"]),
            },
            "metrics": {
                "cpu_avg_pct": round(rng.uniform(5, 95), 1),
                "memory_used_pct": round(rng.uniform(20, 98), 1),
                "disk_used_pct": round(rng.uniform(10, 95), 1),
                "network_mbps": round(rng.uniform(0.1, 1000), 1),
            },
            "tags": rng.sample(["critical", "monitored", "auto-scale", "spot", "reserved", "legacy", "deprecated"], rng.randint(1, 3)),
            "status": rng.choice(["running", "running", "running", "running", "stopped", "maintenance"]),
            "last_patched": f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
        },
    },
]


def make_json_pipeline(rand_seed: int = 42) -> RubricDatapoint:
    """Extract, transform, and aggregate data from large nested JSON.

    The model receives a JSON file with 80-150 records and must answer
    specific aggregation questions that require navigating nested structures,
    filtering, and computing statistics.

    Requires: jq (should be added to Dockerfile), or python/bash scripting
    """
    rng = _random.Random(rand_seed)
    domain = rng.choice(_JSON_DOMAINS)
    n_records = rng.randint(80, 150)

    # Generate records
    records = []
    for i in range(n_records):
        record = domain["item_template"](rng, i)
        records.append(record)

    data = {domain["top_key"]: records}
    json_content = json.dumps(data, indent=2) + "\n"

    # Compute ground-truth answers based on the domain
    if domain["name"] == "e-commerce":
        # Q1: Total revenue from completed orders
        total_revenue = 0.0
        for r in records:
            if r["status"] == "completed":
                for item in r["items"]:
                    subtotal = item["quantity"] * item["unit_price"] * (1 - item["discount_pct"] / 100)
                    total_revenue += subtotal
                total_revenue += r["shipping"]["cost"]
        total_revenue = round(total_revenue, 2)

        # Q2: Count of gold/platinum customers with completed orders
        premium_customers = set()
        for r in records:
            if r["status"] == "completed" and r["customer"]["tier"] in ("gold", "platinum"):
                premium_customers.add(r["customer"]["id"])
        n_premium = len(premium_customers)

        # Q3: Region with highest total order value (completed only)
        region_totals: dict[str, float] = {}
        for r in records:
            if r["status"] == "completed":
                order_val = sum(
                    it["quantity"] * it["unit_price"] * (1 - it["discount_pct"] / 100)
                    for it in r["items"]
                ) + r["shipping"]["cost"]
                region = r["customer"]["region"]
                region_totals[region] = region_totals.get(region, 0) + order_val
        top_region = max(region_totals, key=region_totals.get)  # type: ignore[arg-type]
        top_region_val = round(region_totals[top_region], 2)

        # Q4: Number of returned orders
        n_returned = sum(1 for r in records if r["status"] == "returned")

        # Q5: Average discount % across all items in completed orders
        all_discounts = []
        for r in records:
            if r["status"] == "completed":
                for item in r["items"]:
                    all_discounts.append(item["discount_pct"])
        avg_discount = round(sum(all_discounts) / len(all_discounts), 1) if all_discounts else 0.0

        # Q6: Most frequently ordered SKU
        sku_counts: dict[str, int] = {}
        for r in records:
            if r["status"] == "completed":
                for item in r["items"]:
                    sku_counts[item["sku"]] = sku_counts.get(item["sku"], 0) + item["quantity"]
        top_sku = max(sku_counts, key=sku_counts.get)  # type: ignore[arg-type]

        questions_text = """1. What is the total revenue from completed orders (item subtotals after
   discounts + shipping costs)? Round to 2 decimal places.

2. How many distinct gold or platinum tier customers (by customer.id) have
   at least one completed order?

3. Which customer region has the highest total completed-order value?
   What is that value (rounded to 2 decimal places)?

4. How many orders have status "returned"?

5. What is the average discount percentage across all line items in
   completed orders? Round to 1 decimal place.

6. Which SKU had the highest total quantity ordered (in completed orders)?"""

        answer_format = """TOTAL_REVENUE: <amount>
PREMIUM_CUSTOMERS: <count>
TOP_REGION: <region>
TOP_REGION_VALUE: <amount>
RETURNED_ORDERS: <count>
AVG_DISCOUNT_PCT: <value>
TOP_SKU: <sku>"""

        rubric_items: list[BinaryRubricCategory | RubricCategory] = [
            BinaryRubricCategory(
                name="file_exists",
                question="Does /testbed/answers.txt exist with content?",
                points=1,
            ),
            BinaryRubricCategory(
                name="correct_total_revenue",
                question=f"Does the TOTAL_REVENUE line contain a value within $50 of {total_revenue}?",
                points=3,
            ),
            BinaryRubricCategory(
                name="correct_premium_customers",
                question=f"Does the PREMIUM_CUSTOMERS line contain {n_premium}?",
                points=2,
            ),
            BinaryRubricCategory(
                name="correct_top_region",
                question=f'Does the TOP_REGION line contain "{top_region}"?',
                points=2,
            ),
            BinaryRubricCategory(
                name="correct_top_region_value",
                question=f"Does the TOP_REGION_VALUE line contain a value within $50 of {top_region_val}?",
                points=2,
            ),
            BinaryRubricCategory(
                name="correct_returned_orders",
                question=f"Does the RETURNED_ORDERS line contain {n_returned}?",
                points=2,
            ),
            BinaryRubricCategory(
                name="correct_avg_discount",
                question=f"Does the AVG_DISCOUNT_PCT line contain a value within 0.5 of {avg_discount}?",
                points=2,
            ),
            BinaryRubricCategory(
                name="correct_top_sku",
                question=f'Does the TOP_SKU line contain "{top_sku}"?',
                points=2,
            ),
        ]

    else:  # server-fleet
        # Q1: Count of servers with CPU > 80%
        high_cpu = sum(1 for r in records if r["metrics"]["cpu_avg_pct"] > 80)

        # Q2: Total memory (GB) across all running servers
        total_mem = sum(r["specs"]["memory_gb"] for r in records if r["status"] == "running")

        # Q3: Count of servers with GPU
        gpu_count = sum(1 for r in records if r["specs"]["gpu"] is not None)

        # Q4: Server with highest disk usage %
        max_disk_server = max(records, key=lambda r: r["metrics"]["disk_used_pct"])
        max_disk_host = max_disk_server["host_id"]
        max_disk_pct = max_disk_server["metrics"]["disk_used_pct"]

        # Q5: Count of "critical" tagged servers that are NOT running
        critical_down = sum(
            1 for r in records
            if "critical" in r["tags"] and r["status"] != "running"
        )

        # Q6: Count of servers not patched in the last 90 days (before Oct 2024)
        stale_patch = sum(
            1 for r in records
            if r["last_patched"] < "2024-10-01"
        )

        # Q7: Average network throughput of prod servers
        prod_net = [r["metrics"]["network_mbps"] for r in records if ".prod." in r["hostname"]]
        avg_prod_net = round(sum(prod_net) / len(prod_net), 1) if prod_net else 0.0

        questions_text = """1. How many servers have CPU average usage above 80%?

2. What is the total memory (GB) across all servers with status "running"?

3. How many servers have a GPU (non-null gpu field)?

4. Which server (host_id) has the highest disk_used_pct? What is the value?

5. How many servers tagged "critical" are NOT in "running" status?

6. How many servers were last patched before 2024-10-01?

7. What is the average network throughput (Mbps) of servers with ".prod."
   in their hostname? Round to 1 decimal place."""

        answer_format = """HIGH_CPU_COUNT: <count>
TOTAL_RUNNING_MEMORY_GB: <amount>
GPU_SERVER_COUNT: <count>
MAX_DISK_HOST: <host_id>
MAX_DISK_PCT: <value>
CRITICAL_DOWN_COUNT: <count>
STALE_PATCH_COUNT: <count>
AVG_PROD_NETWORK_MBPS: <value>"""

        rubric_items = [
            BinaryRubricCategory(
                name="file_exists",
                question="Does /testbed/answers.txt exist with content?",
                points=1,
            ),
            BinaryRubricCategory(
                name="correct_high_cpu",
                question=f"Does the HIGH_CPU_COUNT line contain {high_cpu}?",
                points=2,
            ),
            BinaryRubricCategory(
                name="correct_total_memory",
                question=f"Does the TOTAL_RUNNING_MEMORY_GB line contain {total_mem}?",
                points=2,
            ),
            BinaryRubricCategory(
                name="correct_gpu_count",
                question=f"Does the GPU_SERVER_COUNT line contain {gpu_count}?",
                points=2,
            ),
            BinaryRubricCategory(
                name="correct_max_disk_host",
                question=f'Does the MAX_DISK_HOST line contain "{max_disk_host}"?',
                points=2,
            ),
            BinaryRubricCategory(
                name="correct_max_disk_pct",
                question=f"Does the MAX_DISK_PCT line contain a value within 0.5 of {max_disk_pct}?",
                points=2,
            ),
            BinaryRubricCategory(
                name="correct_critical_down",
                question=f"Does the CRITICAL_DOWN_COUNT line contain {critical_down}?",
                points=2,
            ),
            BinaryRubricCategory(
                name="correct_stale_patch",
                question=f"Does the STALE_PATCH_COUNT line contain {stale_patch}?",
                points=2,
            ),
            BinaryRubricCategory(
                name="correct_avg_prod_net",
                question=f"Does the AVG_PROD_NETWORK_MBPS line contain a value within 1.0 of {avg_prod_net}?",
                points=2,
            ),
        ]

    rubric_items.append(
        RubricCategory(
            name="methodology",
            description="Did the model use efficient data processing techniques?",
            failure="Manually counted or guessed without processing the data",
            minor_failure="Used basic scripting but with errors in logic",
            minor_success="Used jq, python, or bash scripting to process data correctly",
            success="Used efficient, targeted queries (jq one-liners or concise scripts) demonstrating mastery of the tooling",
            points=2,
        ),
    )

    return RubricDatapoint(
        problem_statement=f"""# JSON Data Pipeline

You have a large JSON file at /testbed/data.json containing {n_records}
{domain['name']} records with deeply nested fields.

Answer the following questions by processing the JSON data. You may use
any available tools (jq, python, bash scripting, etc.).

## Questions

{questions_text}

## Output Format

Write /testbed/answers.txt with exactly these lines:
```
{answer_format}
```""",
        rubric=tuple(rubric_items),
        submission_instructions="Write your answers to /testbed/answers.txt in the specified format.",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data.json": json_content,
        },
        problem_type="json_pipeline",
    )


# =============================================================================
# 3. DATABASE FORENSICS — find anomalies in a SQLite database
# =============================================================================

def _generate_sqlite_init(
    rng: _random.Random,
    n_employees: int,
    n_transactions: int,
    anomalies: dict,
) -> str:
    """Generate a SQL script to init a SQLite database with planted anomalies."""
    departments = ["Engineering", "Marketing", "Sales", "Finance", "Operations", "HR"]
    titles = {
        "Engineering": ["Software Engineer", "Senior Engineer", "Staff Engineer", "Tech Lead", "Principal Engineer"],
        "Marketing": ["Marketing Analyst", "Content Manager", "Brand Strategist", "Marketing Director"],
        "Sales": ["Sales Rep", "Account Executive", "Sales Manager", "VP Sales"],
        "Finance": ["Financial Analyst", "Accountant", "Controller", "CFO"],
        "Operations": ["Ops Manager", "Logistics Coordinator", "Supply Chain Analyst"],
        "HR": ["HR Specialist", "Recruiter", "HR Director", "Benefits Coordinator"],
    }

    sql_lines = [
        "CREATE TABLE employees (",
        "  id INTEGER PRIMARY KEY,",
        "  name TEXT NOT NULL,",
        "  email TEXT UNIQUE,",
        "  department TEXT,",
        "  title TEXT,",
        "  salary REAL,",
        "  hire_date TEXT,",
        "  manager_id INTEGER REFERENCES employees(id),",
        "  status TEXT DEFAULT 'active',",
        "  termination_date TEXT",
        ");",
        "",
        "CREATE TABLE transactions (",
        "  id INTEGER PRIMARY KEY,",
        "  employee_id INTEGER REFERENCES employees(id),",
        "  amount REAL,",
        "  category TEXT,",
        "  vendor TEXT,",
        "  description TEXT,",
        "  date TEXT,",
        "  approved_by INTEGER REFERENCES employees(id)",
        ");",
        "",
        "CREATE TABLE access_log (",
        "  id INTEGER PRIMARY KEY,",
        "  employee_id INTEGER REFERENCES employees(id),",
        "  resource TEXT,",
        "  action TEXT,",
        "  timestamp TEXT",
        ");",
        "",
    ]

    # Generate employees
    first_names = ["Alice", "Bob", "Carla", "Dan", "Eva", "Frank", "Grace",
                   "Henry", "Irene", "Jack", "Kara", "Leo", "Mia", "Nick",
                   "Olivia", "Peter", "Quinn", "Rachel", "Sam", "Tina"]
    last_names = ["Smith", "Chen", "Park", "Lee", "Davis", "Wilson", "Brown",
                  "Kim", "Garcia", "Miller", "Johnson", "Wang", "Taylor",
                  "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson"]

    employees = []
    for i in range(1, n_employees + 1):
        first = rng.choice(first_names)
        last = rng.choice(last_names)
        dept = rng.choice(departments)
        title = rng.choice(titles[dept])
        salary = round(rng.uniform(45000, 180000), 2)
        year = rng.randint(2018, 2024)
        month = rng.randint(1, 12)
        hire_date = f"{year}-{month:02d}-{rng.randint(1,28):02d}"
        manager_id = rng.randint(1, max(1, i - 1)) if i > 1 else "NULL"
        email = f"{first.lower()}.{last.lower()}{i}@company.com"

        employees.append({
            "id": i, "name": f"{first} {last}", "dept": dept,
            "title": title, "salary": salary, "email": email,
        })
        sql_lines.append(
            f"INSERT INTO employees VALUES ({i}, '{first} {last}', '{email}', "
            f"'{dept}', '{title}', {salary}, '{hire_date}', {manager_id});"
        )

    # Plant anomaly: ghost employee (terminated but still has transactions)
    ghost_emp = anomalies["ghost_employee_id"]
    ghost_name = employees[ghost_emp - 1]["name"]
    # Terminate ghost employee early in the year so most transactions are post-termination
    ghost_term_date = f"2024-{rng.randint(1, 3):02d}-{rng.randint(1, 28):02d}"
    sql_lines.append(
        f"UPDATE employees SET status = 'terminated', termination_date = '{ghost_term_date}' "
        f"WHERE id = {ghost_emp};"
    )

    # Add 3 decoy terminated employees (terminated late — fewer post-termination txns)
    decoy_pool = [i for i in range(1, n_employees + 1)
                  if i not in (ghost_emp, anomalies["salary_outlier_id"])]
    decoy_terminated = rng.sample(decoy_pool, min(3, len(decoy_pool)))
    for dec_id in decoy_terminated:
        dec_term_date = f"2024-{rng.randint(10, 12):02d}-{rng.randint(1, 28):02d}"
        sql_lines.append(
            f"UPDATE employees SET status = 'terminated', termination_date = '{dec_term_date}' "
            f"WHERE id = {dec_id};"
        )

    # Plant anomaly: salary outlier
    outlier_emp = anomalies["salary_outlier_id"]
    outlier_salary = round(rng.uniform(350000, 500000), 2)
    employees[outlier_emp - 1]["salary"] = outlier_salary
    sql_lines.append(f"UPDATE employees SET salary = {outlier_salary} WHERE id = {outlier_emp};")

    sql_lines.append("")

    # Generate transactions
    categories = ["travel", "supplies", "software", "meals", "equipment", "consulting"]
    vendors = ["Acme Corp", "TechSupply Co", "Cloud Services Inc", "Office Depot",
               "Travel Agency", "Consulting Group", "Digital Tools LLC"]

    self_approved_count = 0
    duplicate_txns = []

    for i in range(1, n_transactions + 1):
        emp_id = rng.randint(1, n_employees)
        amount = round(rng.uniform(10, 5000), 2)
        cat = rng.choice(categories)
        vendor = rng.choice(vendors)
        desc = f"{cat} expense - {vendor}"
        month = rng.randint(1, 12)
        date = f"2024-{month:02d}-{rng.randint(1,28):02d}"

        # Ensure non-planted transactions are NEVER self-approved
        approved_by = rng.randint(1, n_employees - 1)
        if approved_by >= emp_id:
            approved_by += 1

        # Plant anomaly: self-approved transactions
        if i in anomalies["self_approved_txn_ids"]:
            approved_by = emp_id
            self_approved_count += 1

        # Plant anomaly: ghost employee transactions
        if i in anomalies["ghost_txn_ids"]:
            emp_id = ghost_emp

        # Plant anomaly: duplicate transactions
        if i in anomalies["duplicate_txn_ids"] and duplicate_txns:
            src = rng.choice(duplicate_txns)
            amount = src["amount"]
            vendor = src["vendor"]
            cat = src["category"]
            desc = src["description"]

        txn = {"amount": amount, "vendor": vendor, "category": cat, "description": desc}
        if i <= 20:  # Seed some transactions for duplication
            duplicate_txns.append(txn)

        sql_lines.append(
            f"INSERT INTO transactions VALUES ({i}, {emp_id}, {amount}, "
            f"'{cat}', '{vendor}', '{desc}', '{date}', {approved_by});"
        )

    sql_lines.append("")

    # Generate access logs
    resources = ["/admin/dashboard", "/reports/financial", "/hr/payroll",
                 "/api/internal", "/settings/security", "/data/export"]
    actions = ["view", "view", "view", "edit", "delete", "export"]

    for i in range(1, 200 + 1):
        emp_id = rng.randint(1, n_employees)
        resource = rng.choice(resources)
        action = rng.choice(actions)
        month = rng.randint(1, 12)
        ts = f"2024-{month:02d}-{rng.randint(1,28):02d}T{rng.randint(0,23):02d}:{rng.randint(0,59):02d}:00"

        # Plant anomaly: after-hours access to sensitive resources
        if i in anomalies["after_hours_access_ids"]:
            emp_id = anomalies["suspicious_accessor_id"]
            resource = "/hr/payroll"
            action = "export"
            ts = f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}T{rng.choice([2,3,4,23]):02d}:{rng.randint(0,59):02d}:00"

        sql_lines.append(
            f"INSERT INTO access_log VALUES ({i}, {emp_id}, '{resource}', '{action}', '{ts}');"
        )

    return "\n".join(sql_lines) + "\n"


def make_database_forensics(rand_seed: int = 42) -> RubricDatapoint:
    """Find anomalies in a SQLite database using SQL queries.

    The model receives a pre-populated SQLite database and must write queries
    to uncover planted anomalies: ghost employees, self-approved transactions,
    duplicate expenses, salary outliers, and suspicious access patterns.

    Requires: sqlite3 (should be added to Dockerfile)
    """
    rng = _random.Random(rand_seed)

    n_employees = rng.randint(40, 60)
    n_transactions = rng.randint(200, 350)

    # Define anomaly parameters
    ghost_emp_id = rng.randint(5, n_employees - 5)
    salary_outlier_id = rng.randint(1, n_employees)
    while salary_outlier_id == ghost_emp_id:
        salary_outlier_id = rng.randint(1, n_employees)

    n_self_approved = rng.randint(3, 6)
    self_approved_ids = rng.sample(range(10, n_transactions), n_self_approved)

    n_ghost_txns = rng.randint(4, 8)
    ghost_txn_ids = rng.sample(range(50, n_transactions), n_ghost_txns)

    n_duplicates = rng.randint(2, 4)
    duplicate_ids = rng.sample(range(30, n_transactions), n_duplicates)

    suspicious_accessor = rng.randint(1, n_employees)
    n_after_hours = rng.randint(5, 10)
    after_hours_ids = rng.sample(range(1, 200), n_after_hours)

    anomalies = {
        "ghost_employee_id": ghost_emp_id,
        "salary_outlier_id": salary_outlier_id,
        "self_approved_txn_ids": self_approved_ids,
        "ghost_txn_ids": ghost_txn_ids,
        "duplicate_txn_ids": duplicate_ids,
        "after_hours_access_ids": after_hours_ids,
        "suspicious_accessor_id": suspicious_accessor,
    }

    sql_init = _generate_sqlite_init(rng, n_employees, n_transactions, anomalies)

    # Create a setup script that initializes the DB
    setup_script = f"""#!/bin/bash
sqlite3 /testbed/company.db << 'EOSQL'
{sql_init}
EOSQL
echo "Database initialized at /testbed/company.db"
"""

    problem_statement = f"""# Database Forensics

You have a SQLite database at /testbed/company.db containing employee records,
expense transactions, and access logs for a company with {n_employees} employees
and {n_transactions}+ transactions.

First, run the setup script: `bash /testbed/setup_db.sh`

Then investigate the database to find the following anomalies. Use sqlite3
queries to find the answers.

## Questions

1. **Self-Approved Transactions**: How many transactions were approved by the
   same employee who submitted them? List the count.

2. **Salary Outlier**: Which employee (id and name) has a salary that is
   a statistical outlier (more than 2 standard deviations above the mean)?

3. **Ghost Employee Detection**: Some employees are marked as terminated
   (status = 'terminated') in the employees table. Which terminated employee
   has the most transactions in the system? Report their employee ID and
   the number of transactions they have.

4. **Duplicate Expenses**: Are there any pairs of transactions with the same
   amount, vendor, and category? How many such duplicate groups exist?

5. **After-Hours Access**: Which employee (id) has the most access_log entries
   during off-hours (before 6:00 or after 22:00)?

6. **Department Spend**: Which department has the highest total transaction
   amount? What is the total?

## Output Format

Write /testbed/answers.txt with exactly these lines:
```
SELF_APPROVED_COUNT: <count>
SALARY_OUTLIER_ID: <employee_id>
SALARY_OUTLIER_NAME: <name>
GHOST_EMPLOYEE_ID: <employee_id>
GHOST_TXN_COUNT: <count>
DUPLICATE_GROUPS: <count>
AFTER_HOURS_EMPLOYEE: <employee_id>
TOP_SPEND_DEPARTMENT: <department>
TOP_SPEND_AMOUNT: <amount>
```"""

    rubric = (
        BinaryRubricCategory(
            name="file_exists",
            question="Does /testbed/answers.txt exist with content?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_self_approved",
            question=f"Does the SELF_APPROVED_COUNT line contain exactly {n_self_approved}?",
            points=3,
        ),
        BinaryRubricCategory(
            name="correct_outlier_id",
            question=f"Does the SALARY_OUTLIER_ID line contain {salary_outlier_id}?",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_ghost_employee_id",
            question=f"Does the GHOST_EMPLOYEE_ID line contain {ghost_emp_id}?",
            points=3,
        ),
        BinaryRubricCategory(
            name="correct_ghost_txns",
            question=f"Does the GHOST_TXN_COUNT line contain a value of at least {n_ghost_txns}? (The terminated employee has at least this many planted transactions, possibly more from random assignment.)",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_duplicate_groups",
            question=f"Does the DUPLICATE_GROUPS line contain a reasonable count of duplicate transaction groups (at least {max(1, n_duplicates - 1)})?",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_after_hours_employee",
            question=f"Does the AFTER_HOURS_EMPLOYEE line contain {suspicious_accessor}?",
            points=3,
        ),
        BinaryRubricCategory(
            name="correct_top_department",
            question="Does the TOP_SPEND_DEPARTMENT line contain a valid department name that matches a SQL query for highest total?",
            points=2,
        ),
        BinaryRubricCategory(
            name="used_sql_queries",
            question="Did the model write and execute SQL queries (visible in bash history or scripts) to find the answers?",
            points=2,
        ),
        BinaryRubricCategory(
            name="ran_setup_script",
            question="Did the model run the setup script (bash /testbed/setup_db.sh) before querying the database?",
            points=1,
        ),
        RubricCategory(
            name="query_quality",
            description="Were the SQL queries well-structured and efficient?",
            failure="No SQL queries or completely wrong approach",
            minor_failure="Basic queries but missing JOINs or aggregations",
            minor_success="Correct queries for most questions with proper JOINs and GROUP BY",
            success="Clean, efficient queries using appropriate aggregations, subqueries, and statistical functions",
            points=2,
        ),
    )

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write your answers to /testbed/answers.txt in the specified format.",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/setup_db.sh": setup_script,
        },
        problem_type="database_forensics",
    )
