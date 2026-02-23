"""Technical / code-oriented tasks.

Six factories that test practical engineering skills:
  - make_bash_golf           (seedable)  File reorganization via bash
  - make_log_query           (seedable)  Answer questions from log data
  - make_config_debugging    (seedable)  Fix a broken config file
  - make_data_transformation (seedable)  Transform CSV per spec
  - make_cron_scheduling     (seedable)  Write crontab entries
  - make_api_documentation   (static)    Document code as API reference
"""

from __future__ import annotations

import random as _random

from ..dataset import RubricDatapoint, RubricCategory, BinaryRubricCategory
from tinker_cookbook.rl.envs import tools
from ..content_pools import random_name, pick1, vary_int

# =============================================================================
# DOMAIN: BASH GOLF SCENARIOS
# =============================================================================

BASH_GOLF_SCENARIOS = [
    {
        "name": "photo_organizer",
        "description": "Organize photos by date extracted from filename",
        "input_files": {
            "/testbed/input/IMG_20240115_001.jpg": "JPEG_DATA_PLACEHOLDER",
            "/testbed/input/IMG_20240115_002.jpg": "JPEG_DATA_PLACEHOLDER",
            "/testbed/input/IMG_20240220_001.jpg": "JPEG_DATA_PLACEHOLDER",
            "/testbed/input/IMG_20240220_002.jpg": "JPEG_DATA_PLACEHOLDER",
            "/testbed/input/IMG_20240301_001.jpg": "JPEG_DATA_PLACEHOLDER",
            "/testbed/input/VID_20240115_001.mp4": "VIDEO_DATA_PLACEHOLDER",
            "/testbed/input/VID_20240220_001.mp4": "VIDEO_DATA_PLACEHOLDER",
            "/testbed/input/thumbs.db": "SYSTEM_FILE_TO_DELETE",
            "/testbed/input/.DS_Store": "SYSTEM_FILE_TO_DELETE",
        },
        "rules": """1. All .jpg files go into /testbed/output/photos/YYYY-MM-DD/ directories (extract date from filename)
2. All .mp4 files go into /testbed/output/videos/YYYY-MM-DD/ directories
3. System files (.DS_Store, thumbs.db) should be deleted
4. Files should be COPIED (originals remain in input/)
5. Each date directory should contain only files from that date""",
    },
    {
        "name": "log_archiver",
        "description": "Archive and compress logs by service name",
        "input_files": {
            "/testbed/input/auth-service.2024-01.log": "Jan auth log content line 1\nJan auth log content line 2\n",
            "/testbed/input/auth-service.2024-02.log": "Feb auth log content\n",
            "/testbed/input/api-gateway.2024-01.log": "Jan API gateway log\n",
            "/testbed/input/api-gateway.2024-02.log": "Feb API gateway log\n",
            "/testbed/input/api-gateway.2024-03.log": "Mar API gateway log\n",
            "/testbed/input/db-service.2024-01.log": "Jan DB log\n",
            "/testbed/input/db-service.2024-02.log": "Feb DB log\n",
            "/testbed/input/README.txt": "Log directory - do not modify",
            "/testbed/input/cleanup.sh.bak": "old cleanup script",
        },
        "rules": """1. Group log files by service name into /testbed/output/<service-name>/
2. Within each service directory, concatenate all monthly logs into a single combined.log (in chronological order)
3. Create a file count.txt in each service directory with the number of original log files
4. Delete all .bak files
5. Copy README.txt to /testbed/output/README.txt
6. Files should be COPIED — originals remain in input/""",
    },
    {
        "name": "code_organizer",
        "description": "Sort source files by language into directories",
        "input_files": {
            "/testbed/input/main.py": "#!/usr/bin/env python3\nprint('hello')\n",
            "/testbed/input/utils.py": "def helper(): pass\n",
            "/testbed/input/test_main.py": "import pytest\ndef test_main(): pass\n",
            "/testbed/input/app.js": "console.log('hello');\n",
            "/testbed/input/index.html": "<html><body>Hello</body></html>\n",
            "/testbed/input/style.css": "body { margin: 0; }\n",
            "/testbed/input/Makefile": "all:\n\techo build\n",
            "/testbed/input/config.yaml": "key: value\n",
            "/testbed/input/.gitignore": "*.pyc\n__pycache__/\n",
        },
        "rules": """1. Python files (.py) go into /testbed/output/python/ — but test files (starting with test_) go into /testbed/output/python/tests/
2. JavaScript files (.js) go into /testbed/output/javascript/
3. Web files (.html, .css) go into /testbed/output/web/
4. Config files (.yaml, Makefile, .gitignore) go into /testbed/output/config/
5. Files should be COPIED — originals remain in input/
6. Create a manifest.txt in /testbed/output/ listing all copied files and their destinations""",
    },
    {
        "name": "data_cleaner",
        "description": "Clean and normalize data files",
        "input_files": {
            "/testbed/input/users_export.csv": "name,email,phone\nJohn Doe,john@test.com,555-1234\nJane Smith,jane@test.com,555-5678\n  Bob Jones , bob@test.com , 555-9012 \n",
            "/testbed/input/products.tsv": "id\tname\tprice\n1\tWidget\t9.99\n2\tGadget\t19.99\n3\tDoohickey\t4.99\n",
            "/testbed/input/notes.txt": "Meeting notes from Tuesday\nAction items: review budget\nNext meeting: Friday\n",
            "/testbed/input/data.json": '{"users": [{"name": "Alice"}, {"name": "Bob"}]}',
            "/testbed/input/empty.csv": "",
            "/testbed/input/backup_old.csv.bak": "old data",
        },
        "rules": """1. Copy all .csv files (except empty ones) to /testbed/output/csv/ with whitespace trimmed from all fields
2. Convert all .tsv files to .csv format and place in /testbed/output/csv/
3. Copy .json files to /testbed/output/json/ with pretty-printing (indented)
4. Copy .txt files to /testbed/output/text/
5. Delete all .bak files and empty files
6. Create /testbed/output/summary.txt listing file count per type""",
    },
]

# =============================================================================
# DOMAIN: SERVICE INFRASTRUCTURE
# =============================================================================

SERVICE_NAMES = [
    "auth-service", "api-gateway", "user-service", "payment-service",
    "notification-service", "search-service", "inventory-service",
    "order-service", "email-service", "analytics-service", "cache-service",
    "config-service", "scheduler-service", "file-service", "audit-service",
    "billing-service", "report-service", "webhook-service",
]

ERROR_TYPES = [
    "ConnectionTimeoutError", "NullPointerException", "OutOfMemoryError",
    "DatabaseConnectionError", "AuthenticationFailedError",
    "RateLimitExceededError", "FileNotFoundError", "PermissionDeniedError",
    "InvalidConfigurationError", "ServiceUnavailableError",
    "CertificateExpiredError", "DiskSpaceFullError", "SocketTimeoutError",
    "MalformedRequestError", "SchemaValidationError",
]

CONFIG_FORMATS = ["yaml", "json"]

API_ENDPOINTS = [
    "/api/v1/users", "/api/v1/orders", "/api/v1/products",
    "/api/v1/payments", "/api/v1/search", "/api/v1/notifications",
    "/api/v1/reports", "/api/v1/auth/login", "/api/v1/auth/refresh",
    "/api/v1/webhooks", "/api/v1/uploads", "/api/v1/health",
]

HTTP_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH"]

HTTP_STATUS_CODES = [200, 201, 204, 400, 401, 403, 404, 408, 429, 500, 502, 503]


# ============================================================================
# 1. BASH GOLF — seedable file reorganization
# ============================================================================


def _bash_golf_rubric_for_scenario(scenario: dict) -> tuple:
    """Build a rubric tailored to the selected scenario."""
    name = scenario["name"]

    # Base binary checks common across all scenarios
    binary_checks: list[BinaryRubricCategory] = [
        BinaryRubricCategory(
            name="script_exists",
            question="Does the file /testbed/solution.sh exist and contain a non-empty bash script?",
            points=1,
        ),
        BinaryRubricCategory(
            name="originals_preserved",
            question="Are all original files (other than those explicitly marked for deletion in the rules) still present and unmodified in /testbed/input/?",
            points=2,
        ),
    ]

    # Scenario-specific binary checks
    if name == "photo_organizer":
        binary_checks += [
            BinaryRubricCategory(
                name="photos_in_date_dirs",
                question="Are all .jpg files copied into /testbed/output/photos/YYYY-MM-DD/ directories matching their filename dates?",
                points=2,
            ),
            BinaryRubricCategory(
                name="videos_in_date_dirs",
                question="Are all .mp4 files copied into /testbed/output/videos/YYYY-MM-DD/ directories matching their filename dates?",
                points=2,
            ),
            BinaryRubricCategory(
                name="system_files_deleted",
                question="Have all system files (.DS_Store, thumbs.db) been deleted from /testbed/input/?",
                points=1,
            ),
            BinaryRubricCategory(
                name="no_extra_files",
                question="Does the output directory contain ONLY the correctly organized files and no stray copies?",
                points=1,
            ),
        ]
    elif name == "log_archiver":
        binary_checks += [
            BinaryRubricCategory(
                name="logs_grouped_by_service",
                question="Are log files grouped into /testbed/output/<service-name>/ directories correctly?",
                points=2,
            ),
            BinaryRubricCategory(
                name="combined_logs_created",
                question="Does each service directory contain a combined.log with all monthly logs concatenated in chronological order?",
                points=2,
            ),
            BinaryRubricCategory(
                name="count_files_created",
                question="Does each service directory contain a count.txt with the correct number of original log files?",
                points=1,
            ),
            BinaryRubricCategory(
                name="bak_deleted",
                question="Have all .bak files been deleted?",
                points=1,
            ),
            BinaryRubricCategory(
                name="readme_copied",
                question="Has README.txt been copied to /testbed/output/README.txt?",
                points=1,
            ),
        ]
    elif name == "code_organizer":
        binary_checks += [
            BinaryRubricCategory(
                name="python_in_python_dir",
                question="Are non-test .py files in /testbed/output/python/?",
                points=2,
            ),
            BinaryRubricCategory(
                name="tests_in_tests_dir",
                question="Are test_*.py files in /testbed/output/python/tests/?",
                points=2,
            ),
            BinaryRubricCategory(
                name="js_in_javascript_dir",
                question="Are .js files in /testbed/output/javascript/?",
                points=1,
            ),
            BinaryRubricCategory(
                name="web_in_web_dir",
                question="Are .html and .css files in /testbed/output/web/?",
                points=1,
            ),
            BinaryRubricCategory(
                name="manifest_created",
                question="Does /testbed/output/manifest.txt exist listing all copied files and their destinations?",
                points=2,
            ),
        ]
    else:  # data_cleaner
        binary_checks += [
            BinaryRubricCategory(
                name="csv_whitespace_trimmed",
                question="Are .csv files in /testbed/output/csv/ with whitespace trimmed from all fields?",
                points=2,
            ),
            BinaryRubricCategory(
                name="tsv_converted",
                question="Have .tsv files been converted to .csv format and placed in /testbed/output/csv/?",
                points=2,
            ),
            BinaryRubricCategory(
                name="json_pretty_printed",
                question="Are .json files in /testbed/output/json/ with pretty-printed (indented) content?",
                points=1,
            ),
            BinaryRubricCategory(
                name="bak_and_empty_deleted",
                question="Have all .bak files and empty files been deleted?",
                points=1,
            ),
            BinaryRubricCategory(
                name="summary_created",
                question="Does /testbed/output/summary.txt exist with correct file count per type?",
                points=1,
            ),
        ]

    # Graded categories common to all
    graded = [
        RubricCategory(
            name="correctness",
            description="Does the final filesystem state satisfy all rules specified in the problem?",
            failure="Output directory is missing or most files are in the wrong place.",
            minor_failure="Some files are misplaced or some rules are not followed.",
            minor_success="Most files are correct with one minor misplacement.",
            success="All rules are fully satisfied.",
            points=3,
        ),
        RubricCategory(
            name="minimality",
            description="How concise is the bash script? Fewer statements is better.",
            failure="More than 15 statements, or brute-force file-by-file approach.",
            minor_failure="10-15 statements; some patterns but verbose.",
            minor_success="6-9 statements; reasonable use of globs and patterns.",
            success="5 or fewer statements using elegant glob patterns, loops, or one-liners.",
            points=2,
        ),
        RubricCategory(
            name="no_side_effects",
            description="Does the solution avoid unintended side effects?",
            failure="Modifies or deletes files outside the specified scope, or corrupts the filesystem.",
            minor_failure="Creates unnecessary temp files or leaves artifacts.",
            minor_success="Clean execution with one minor unnecessary artifact.",
            success="No side effects -- only the specified directories are touched.",
            points=2,
        ),
    ]

    return tuple(binary_checks + graded)


def make_bash_golf(rand_seed: int = 42) -> RubricDatapoint:
    """Bash golf task -- reorganize files using minimal bash commands.

    Seedable: the seed selects from several file-organization scenarios
    defined in BASH_GOLF_SCENARIOS. Each scenario has
    different input files and routing rules, ensuring variety.
    """
    rng = _random.Random(rand_seed)
    scenario = rng.choice(BASH_GOLF_SCENARIOS)

    # Optionally inject extra files seeded by rand_seed to vary even within
    # the same scenario. We add 0-2 additional misc files.
    extra_count = rng.randint(0, 2)
    input_files = dict(scenario["input_files"])
    extra_names = ["scratch_notes.txt", "TODO.md", "draft_v2.txt",
                   "archive.tar.gz.bak", "temp_data.log"]
    extra_contents = [
        "Miscellaneous scratch notes\n",
        "# TODO\n- Item 1\n- Item 2\n",
        "Second draft of the proposal.\n",
        "STALE_ARCHIVE_DATA",
        "debug: temporary log data\n",
    ]
    chosen_extras = rng.sample(
        list(zip(extra_names, extra_contents)),
        min(extra_count, len(extra_names)),
    )
    for fname, content in chosen_extras:
        input_files[f"/testbed/input/{fname}"] = content

    # Build the file listing for the problem statement
    file_list = "\n".join(
        f"  - {path.split('/testbed/input/')[-1]}"
        for path in sorted(input_files.keys())
    )

    problem_statement = f"""# Bash Golf: File Reorganization ({scenario['description']})

You have a directory /testbed/input/ containing the following files:

{file_list}

Reorganize them into /testbed/output/ according to these rules:

{scenario['rules']}

Write your solution as a bash script in /testbed/solution.sh, then execute it.

GOAL: Accomplish this in as few bash statements as possible while being correct.
An ideal solution uses ~5-8 statements. Fewer is better."""

    rubric = _bash_golf_rubric_for_scenario(scenario)

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions=(
            "Write your solution to /testbed/solution.sh and execute it "
            "with: bash /testbed/solution.sh"
        ),
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files=input_files,
        problem_type="bash_golf",
    )


# ============================================================================
# 2. LOG QUERY — answer questions from a generated application log
# ============================================================================

# Templates for realistic log lines keyed by log level.
_LOG_TEMPLATES: dict[str, list[str]] = {
    "INFO": [
        "{service} - Request {method} {endpoint} completed in {latency}ms (status={status})",
        "{service} - Health check passed: uptime={uptime}s, connections={conns}",
        "{service} - Deployment started: version={version}",
        "{service} - Deployment completed: version={version}",
        "{service} - Cache hit ratio: {ratio}%",
        "{service} - Scheduled job '{job}' completed successfully in {latency}ms",
        "{service} - User {user} logged in from {ip}",
        "{service} - Configuration reloaded: {config_key}={config_val}",
    ],
    "WARN": [
        "{service} - Slow query detected: {latency}ms on {endpoint}",
        "{service} - Connection pool at {pool_pct}% capacity ({conns}/{max_conns})",
        "{service} - Retry attempt {attempt}/3 for {endpoint}",
        "{service} - Disk usage at {disk_pct}% on /var/data",
        "{service} - Rate limit approaching for client {client_id}: {rate}/100 requests",
    ],
    "ERROR": [
        "{service} - {error_type}: {endpoint} returned {status}",
        "{service} - {error_type}: failed to connect to database after {attempt} retries",
        "{service} - {error_type}: request payload exceeds maximum size ({size}KB)",
        "{service} - {error_type}: certificate verification failed for {host}",
        "{service} - {error_type}: out of memory, current heap {heap}MB",
    ],
}

_JOBS = [
    "daily_backup", "log_rotation", "metric_aggregation",
    "cache_cleanup", "report_generation", "index_rebuild",
]


def _generate_log(rng: _random.Random, n_lines: int = 200) -> tuple[str, dict]:
    """Generate a deterministic application log and compute ground-truth answers.

    Returns (log_text, answers_dict).
    """
    services = rng.sample(SERVICE_NAMES, 4)
    endpoints = rng.sample(API_ENDPOINTS, 6)
    errors = rng.sample(ERROR_TYPES, 4)

    lines: list[str] = []

    # Counters for ground truth
    error_count = 0
    first_500_time: str | None = None
    endpoint_request_counts: dict[str, int] = {}
    service_error_counts: dict[str, int] = {}
    warn_count = 0
    deployment_versions: list[str] = []

    # Generate deterministic timestamps across a single day
    base_hour = 0
    base_minute = 0

    for i in range(n_lines):
        # Advance time
        base_minute += rng.randint(0, 3)
        if base_minute >= 60:
            base_hour += 1
            base_minute = base_minute % 60
        if base_hour >= 24:
            base_hour = 23
            base_minute = 59
        second = rng.randint(0, 59)
        ts = f"2025-01-15 {base_hour:02d}:{base_minute:02d}:{second:02d}"

        # Choose log level with realistic distribution
        r = rng.random()
        if r < 0.60:
            level = "INFO"
        elif r < 0.82:
            level = "WARN"
        else:
            level = "ERROR"

        service = rng.choice(services)
        endpoint = rng.choice(endpoints)
        status = rng.choice(HTTP_STATUS_CODES)
        method = rng.choice(HTTP_METHODS)
        latency = rng.randint(5, 3500)
        version = f"v{rng.randint(1,5)}.{rng.randint(0,20)}.{rng.randint(0,99)}"
        error_type = rng.choice(errors)

        template = rng.choice(_LOG_TEMPLATES[level])
        line_body = template.format(
            service=service,
            method=method,
            endpoint=endpoint,
            latency=latency,
            status=status,
            uptime=rng.randint(1000, 500000),
            conns=rng.randint(1, 200),
            version=version,
            ratio=rng.randint(50, 99),
            job=rng.choice(_JOBS),
            user=f"user_{rng.randint(1000,9999)}",
            ip=f"10.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}",
            config_key=rng.choice(["max_connections", "timeout_ms", "log_level", "cache_ttl"]),
            config_val=rng.choice(["100", "5000", "INFO", "3600"]),
            pool_pct=rng.randint(70, 95),
            max_conns=200,
            attempt=rng.randint(1, 3),
            disk_pct=rng.randint(80, 97),
            client_id=f"client_{rng.randint(100,999)}",
            rate=rng.randint(70, 99),
            size=rng.randint(100, 5000),
            host=f"{rng.choice(services)}.internal",
            heap=rng.randint(512, 4096),
            error_type=error_type,
        )

        log_line = f"[{ts}] [{level:5s}] {line_body}"
        lines.append(log_line)

        # Track ground-truth
        if level == "ERROR":
            error_count += 1
            service_error_counts[service] = service_error_counts.get(service, 0) + 1
        if level == "WARN":
            warn_count += 1
        # Track first visible HTTP 500 across ALL log levels
        if status == 500 and first_500_time is None:
            if f"status={status}" in line_body or f"returned {status}" in line_body:
                first_500_time = ts

        # Track endpoint requests from INFO request lines
        if level == "INFO" and "Request" in template:
            endpoint_request_counts[endpoint] = endpoint_request_counts.get(endpoint, 0) + 1

        # Track deployments
        if "Deployment completed" in line_body:
            deployment_versions.append(version)

    log_text = "\n".join(lines) + "\n"

    # Compute answers
    most_requested = max(endpoint_request_counts, key=endpoint_request_counts.get) if endpoint_request_counts else "N/A"
    most_requested_count = endpoint_request_counts.get(most_requested, 0)
    most_errors_service = max(service_error_counts, key=service_error_counts.get) if service_error_counts else "N/A"
    most_errors_count = service_error_counts.get(most_errors_service, 0)

    answers = {
        "error_count": error_count,
        "first_500_time": first_500_time if first_500_time else "No 500 errors found",
        "most_requested_endpoint": most_requested,
        "most_requested_count": most_requested_count,
        "warn_count": warn_count,
        "most_errors_service": most_errors_service,
        "most_errors_count": most_errors_count,
        "deployment_count": len(deployment_versions),
    }
    return log_text, answers


def make_log_query(rand_seed: int = 42) -> RubricDatapoint:
    """Given a large log file and specific questions, answer by examining logs.

    The seed varies the log content, service names, and questions.
    """
    rng = _random.Random(rand_seed)
    n_lines = vary_int(200, rand_seed, pct=0.15)
    log_text, answers = _generate_log(rng, n_lines)

    # Build 5 questions with known answers
    questions = [
        f"Q1: How many ERROR-level log entries are there in total?\nExpected format: a single integer",
        f"Q2: At what timestamp did the first HTTP 500 status code appear?\nExpected format: YYYY-MM-DD HH:MM:SS (or 'No 500 errors found' if none)",
        f"Q3: Which API endpoint received the most requests? How many?\nExpected format: <endpoint> (<count>)",
        f"Q4: How many WARN-level log entries are there in total?\nExpected format: a single integer",
        f"Q5: Which service produced the most ERROR entries? How many?\nExpected format: <service> (<count>)",
    ]
    questions_text = "\n\n".join(questions)

    expected_answers = [
        f"Q1: {answers['error_count']}",
        f"Q2: {answers['first_500_time']}",
        f"Q3: {answers['most_requested_endpoint']} ({answers['most_requested_count']})",
        f"Q4: {answers['warn_count']}",
        f"Q5: {answers['most_errors_service']} ({answers['most_errors_count']})",
    ]

    necessary_files = {
        "/testbed/logs/application.log": log_text,
        "/testbed/questions.txt": (
            "Answer each of the following questions about /testbed/logs/application.log.\n"
            "Write your answers to /testbed/answers.txt in the exact format specified.\n\n"
            + questions_text
        ),
    }

    # Rubric: one binary per question + a graded methodology check
    rubric_items: list = [
        BinaryRubricCategory(
            name="answers_file_exists",
            question="Does /testbed/answers.txt exist and contain non-empty text?",
            points=1,
        ),
        BinaryRubricCategory(
            name="q1_error_count",
            question=(
                f"Does the answer for Q1 state that there are exactly "
                f"{answers['error_count']} ERROR-level entries?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="q2_first_500",
            question=(
                f"Does the answer for Q2 correctly identify the first 500 status "
                f"timestamp as {answers['first_500_time']}?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="q3_most_requested",
            question=(
                f"Does the answer for Q3 correctly identify {answers['most_requested_endpoint']} "
                f"with {answers['most_requested_count']} requests as the most-requested endpoint?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="q4_warn_count",
            question=(
                f"Does the answer for Q4 state that there are exactly "
                f"{answers['warn_count']} WARN-level entries?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="q5_most_errors_service",
            question=(
                f"Does the answer for Q5 correctly identify {answers['most_errors_service']} "
                f"with {answers['most_errors_count']} errors as the service with the most ERROR entries?"
            ),
            points=2,
        ),
        RubricCategory(
            name="methodology",
            description="Did the agent use appropriate log-analysis tools (grep, awk, sort, etc.) rather than guessing?",
            failure="No evidence of log analysis; answers appear fabricated.",
            minor_failure="Some analysis attempted but with flawed commands or incomplete processing.",
            minor_success="Reasonable analysis with appropriate tools; minor inefficiencies.",
            success="Efficient, systematic analysis using appropriate text-processing tools (grep, awk, sort, uniq, etc.).",
            points=3,
        ),
        RubricCategory(
            name="answer_formatting",
            description="Are the answers formatted exactly as specified in the questions?",
            failure="Answers are missing or in a completely wrong format.",
            minor_failure="Most answers present but formats are inconsistent or wrong.",
            minor_success="Formats are mostly correct with one minor deviation.",
            success="All answers follow the exact specified format.",
            points=2,
        ),
    ]

    return RubricDatapoint(
        problem_statement=f"""# Log Analysis: Application Log Query

You have an application log file at /testbed/logs/application.log containing
{n_lines} log entries from a microservices application.

Read the questions in /testbed/questions.txt, analyze the log file, and write
your answers to /testbed/answers.txt.

Each answer must be on its own line, prefixed with the question number (e.g., Q1: ...).
Use the exact format specified in each question.

IMPORTANT: Derive your answers from the actual log data. Do not guess.""",
        rubric=tuple(rubric_items),
        submission_instructions="Write your answers to /testbed/answers.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files=necessary_files,
        problem_type="log_query",
    )


# ============================================================================
# 3. CONFIG DEBUGGING — fix a broken configuration file
# ============================================================================

# Each error template: (field_path, broken_value, fixed_value, error_hint, check_description)
_YAML_ERROR_POOL: list[tuple[str, str, str, str, str]] = [
    # Wrong type: port as string instead of int
    (
        "server.port",
        '  port: "eight-zero-eight-zero"',
        "  port: 8080",
        "TypeError: expected integer for 'server.port', got str",
        "Is server.port set to a valid integer (e.g. 8080)?",
    ),
    # Missing required field: database.host absent
    (
        "database.host",
        "  # host: db.example.com  <-- commented out",
        "  host: db.example.com",
        "KeyError: required field 'database.host' is missing",
        "Is database.host present and set to a valid hostname?",
    ),
    # Invalid value: log level not in allowed set
    (
        "logging.level",
        '  level: "VERBOSE"',
        '  level: "INFO"',
        "ValueError: 'VERBOSE' is not a valid log level (allowed: DEBUG, INFO, WARN, ERROR)",
        "Is logging.level set to one of the valid values (DEBUG, INFO, WARN, ERROR)?",
    ),
    # Wrong type: max_connections as float
    (
        "database.max_connections",
        "  max_connections: 50.5",
        "  max_connections: 50",
        "TypeError: expected integer for 'database.max_connections', got float",
        "Is database.max_connections set to a valid integer?",
    ),
    # Invalid value: timeout negative
    (
        "server.timeout_ms",
        "  timeout_ms: -1000",
        "  timeout_ms: 30000",
        "ValueError: 'server.timeout_ms' must be a positive integer",
        "Is server.timeout_ms set to a positive integer?",
    ),
    # Wrong type: boolean as string
    (
        "features.enable_cache",
        '  enable_cache: "yes please"',
        "  enable_cache: true",
        "TypeError: expected boolean for 'features.enable_cache', got str",
        "Is features.enable_cache set to a valid boolean (true/false)?",
    ),
    # Missing required field: auth.secret_key absent
    (
        "auth.secret_key",
        "  # secret_key: <must be set>",
        '  secret_key: "change-me-in-production"',
        "KeyError: required field 'auth.secret_key' is missing",
        "Is auth.secret_key present and set to a non-empty string?",
    ),
    # Invalid value: workers set to 0
    (
        "server.workers",
        "  workers: 0",
        "  workers: 4",
        "ValueError: 'server.workers' must be >= 1",
        "Is server.workers set to an integer >= 1?",
    ),
]

_JSON_ERROR_POOL: list[tuple[str, str, str, str, str]] = [
    # Trailing comma (JSON syntax error)
    (
        "server.port (syntax)",
        '    "port": 8080,\n    "host": "0.0.0.0",',
        '    "port": 8080,\n    "host": "0.0.0.0"',
        "SyntaxError: Trailing comma not allowed in JSON at line 4",
        "Is the JSON free of trailing commas (valid JSON syntax)?",
    ),
    # Wrong type: port as string
    (
        "server.port",
        '    "port": "not-a-number"',
        '    "port": 8080',
        "TypeError: expected integer for 'server.port', got string",
        "Is server.port set to a valid integer?",
    ),
    # Missing required field
    (
        "database.host",
        '    "name": "appdb"',
        '    "host": "db.example.com",\n    "name": "appdb"',
        "KeyError: required field 'database.host' is missing",
        "Is database.host present and set to a valid hostname?",
    ),
    # Invalid value: log level
    (
        "logging.level",
        '    "level": "TRACE"',
        '    "level": "INFO"',
        "ValueError: 'TRACE' is not a valid log level (allowed: DEBUG, INFO, WARN, ERROR)",
        "Is logging.level set to one of the valid values?",
    ),
    # Boolean as string
    (
        "features.enable_cache",
        '    "enable_cache": "true"',
        '    "enable_cache": true',
        'TypeError: expected boolean for features.enable_cache, got string "true"',
        "Is features.enable_cache a proper JSON boolean (true/false), not a string?",
    ),
    # Negative timeout
    (
        "server.timeout_ms",
        '    "timeout_ms": -500',
        '    "timeout_ms": 30000',
        "ValueError: 'server.timeout_ms' must be a positive integer",
        "Is server.timeout_ms a positive integer?",
    ),
]


def _build_yaml_config(rng: _random.Random, errors: list) -> tuple[str, str, str]:
    """Build a YAML config with injected errors. Returns (broken, fixed, docs)."""
    service_name = rng.choice(SERVICE_NAMES)
    db_port = rng.choice([5432, 3306, 27017])

    # Base config template (valid)
    base_sections = {
        "server": [
            "server:",
            "  host: 0.0.0.0",
            "  port: 8080",
            "  workers: 4",
            "  timeout_ms: 30000",
        ],
        "database": [
            "database:",
            "  host: db.example.com",
            f"  port: {db_port}",
            "  name: appdb",
            "  max_connections: 50",
        ],
        "logging": [
            "logging:",
            '  level: "INFO"',
            '  format: "%(asctime)s %(levelname)s %(message)s"',
            '  file: "/var/log/app.log"',
        ],
        "features": [
            "features:",
            "  enable_cache: true",
            "  enable_metrics: true",
            "  enable_debug: false",
        ],
        "auth": [
            "auth:",
            '  secret_key: "change-me-in-production"',
            "  token_expiry_seconds: 3600",
            '  algorithm: "HS256"',
        ],
    }

    # Build broken config by applying errors
    broken_lines = [f"# {service_name} Application Configuration", ""]
    fixed_lines = [f"# {service_name} Application Configuration", ""]

    for section_name in ["server", "database", "logging", "features", "auth"]:
        section_lines = list(base_sections[section_name])
        broken_section = list(section_lines)
        fixed_section = list(section_lines)

        for field_path, broken_val, fixed_val, _hint, _desc in errors:
            section_key = field_path.split(".")[0]
            if section_key == section_name:
                # Find and replace the relevant line
                for idx, line in enumerate(section_lines):
                    field_name = field_path.split(".")[-1]
                    if field_name in line and ":" in line:
                        broken_section[idx] = broken_val
                        fixed_section[idx] = fixed_val
                        break
                    elif line.strip().startswith(f"# {field_name}"):
                        # Commented out line — already broken
                        broken_section[idx] = broken_val
                        fixed_section[idx] = fixed_val
                        break
                else:
                    # Field not found — for "missing" errors, we need special handling
                    if "missing" in _hint.lower() or "commented" in broken_val:
                        # Already handled above via the commented-out check
                        pass

        broken_lines.extend(broken_section)
        broken_lines.append("")
        fixed_lines.extend(fixed_section)
        fixed_lines.append("")

    # Documentation
    docs = f"""# Configuration Reference for {service_name}
# =============================================

## server
  host     : string  (required) - Bind address, e.g. "0.0.0.0"
  port     : integer (required) - Port number, 1-65535
  workers  : integer (required) - Number of worker processes, must be >= 1
  timeout_ms : integer (required) - Request timeout in milliseconds, must be positive

## database
  host            : string  (required) - Database hostname
  port            : integer (required) - Database port
  name            : string  (required) - Database name
  max_connections : integer (optional) - Connection pool size, default 50

## logging
  level  : string (required) - One of: DEBUG, INFO, WARN, ERROR
  format : string (optional) - Log format string
  file   : string (optional) - Log file path

## features
  enable_cache   : boolean (optional) - Enable response caching, default true
  enable_metrics : boolean (optional) - Enable metrics collection, default true
  enable_debug   : boolean (optional) - Enable debug mode, default false

## auth
  secret_key           : string  (required) - JWT signing key (MUST be set)
  token_expiry_seconds : integer (optional) - Token TTL, default 3600
  algorithm            : string  (optional) - Signing algorithm, default "HS256"
"""

    return "\n".join(broken_lines) + "\n", "\n".join(fixed_lines) + "\n", docs


def _build_json_config(rng: _random.Random, errors: list) -> tuple[str, str, str]:
    """Build a JSON config with injected errors. Returns (broken, reference_fixed, docs)."""
    service_name = rng.choice(SERVICE_NAMES)
    db_port = rng.choice([5432, 3306, 27017])

    # We build the config as text (not via json.dumps) so we can inject
    # syntax errors like trailing commas.
    base = {
        "server": {
            "host": '"0.0.0.0"',
            "port": "8080",
            "workers": "4",
            "timeout_ms": "30000",
        },
        "database": {
            "host": '"db.example.com"',
            "port": str(db_port),
            "name": '"appdb"',
            "max_connections": "50",
        },
        "logging": {
            "level": '"INFO"',
            "format": '"%(asctime)s %(levelname)s %(message)s"',
            "file": '"/var/log/app.log"',
        },
        "features": {
            "enable_cache": "true",
            "enable_metrics": "true",
            "enable_debug": "false",
        },
        "auth": {
            "secret_key": '"change-me-in-production"',
            "token_expiry_seconds": "3600",
            "algorithm": '"HS256"',
        },
    }

    def render_json(base_dict: dict, apply_errors: bool) -> str:
        """Render dict as JSON text, optionally injecting errors."""
        out = "{\n"
        sections = list(base_dict.items())
        for si, (sec_name, fields) in enumerate(sections):
            out += f'  "{sec_name}": {{\n'
            field_items = list(fields.items())
            for fi, (field_name, field_val) in enumerate(field_items):
                replaced = False
                if apply_errors:
                    for field_path, broken_val, fixed_val, _h, _d in errors:
                        parts = field_path.split(".")
                        # Strip " (syntax)" suffix for trailing-comma error entries
                        clean_field = parts[-1].removesuffix(" (syntax)") if len(parts) >= 2 else parts[-1]
                        if len(parts) >= 2 and parts[0] == sec_name and clean_field == field_name:
                            # For "missing" errors, skip the field entirely
                            if "missing" in _h.lower() and apply_errors:
                                replaced = True
                                break
                            # Replace with broken value
                            out += broken_val + "\n"
                            replaced = True
                            break
                if not replaced:
                    comma = "," if fi < len(field_items) - 1 else ""
                    out += f'    "{field_name}": {field_val}{comma}\n'
            sec_comma = "," if si < len(sections) - 1 else ""
            out += f"  }}{sec_comma}\n"
        out += "}\n"
        return out

    broken_text = render_json(base, apply_errors=True)
    fixed_text = render_json(base, apply_errors=False)

    docs = f"""Configuration Reference for {service_name}
=============================================

Format: JSON

server:
  host          : string  (required) - Bind address
  port          : integer (required) - Port number, 1-65535
  workers       : integer (required) - Worker count, >= 1
  timeout_ms    : integer (required) - Timeout in ms, must be positive

database:
  host            : string  (required) - Database hostname
  port            : integer (required) - Database port
  name            : string  (required) - Database name
  max_connections : integer (optional) - Pool size, default 50

logging:
  level  : string (required) - One of: DEBUG, INFO, WARN, ERROR
  format : string (optional) - Log format pattern
  file   : string (optional) - Log output file path

features:
  enable_cache   : boolean (optional) - default true
  enable_metrics : boolean (optional) - default true
  enable_debug   : boolean (optional) - default false

auth:
  secret_key           : string  (required) - JWT signing key, must not be empty
  token_expiry_seconds : integer (optional) - Token TTL, default 3600
  algorithm            : string  (optional) - default "HS256"

NOTE: JSON values must use correct types. Booleans are true/false (not strings).
      Integers must not be floats or strings. No trailing commas allowed.
"""
    return broken_text, fixed_text, docs


def make_config_debugging(rand_seed: int = 42) -> RubricDatapoint:
    """Given a broken config, error message, and docs, fix the config.

    The seed varies the config format (YAML/JSON), which errors are injected,
    and the service context.
    """
    rng = _random.Random(rand_seed)
    fmt = rng.choice(CONFIG_FORMATS)
    n_errors = rng.randint(3, 4)

    if fmt == "yaml":
        chosen_errors = rng.sample(_YAML_ERROR_POOL, min(n_errors, len(_YAML_ERROR_POOL)))
        broken, _fixed, docs = _build_yaml_config(rng, chosen_errors)
        ext = "yaml"
    else:
        chosen_errors = rng.sample(_JSON_ERROR_POOL, min(n_errors, len(_JSON_ERROR_POOL)))
        broken, _fixed, docs = _build_json_config(rng, chosen_errors)
        ext = "json"

    # Build error log from the chosen errors
    error_lines = [
        "Application failed to start. Configuration errors detected:\n",
    ]
    for i, (_field, _bval, _fval, hint, _desc) in enumerate(chosen_errors, 1):
        error_lines.append(f"  [{i}] {hint}")
    error_lines.append(
        "\nFix all configuration errors and save the corrected file."
    )
    error_log = "\n".join(error_lines) + "\n"

    config_path = f"/testbed/config/app_config.{ext}"
    fixed_path = f"/testbed/config/app_config_fixed.{ext}"

    necessary_files = {
        config_path: broken,
        "/testbed/logs/error.log": error_log,
        "/testbed/docs/config_reference.txt": docs,
    }

    # Build rubric
    rubric_items: list = [
        BinaryRubricCategory(
            name="fixed_file_exists",
            question=f"Does {fixed_path} exist and contain non-empty content?",
            points=1,
        ),
    ]

    # One binary check per error
    for _field, _bval, _fval, _hint, desc in chosen_errors:
        field_safe = _field.replace(".", "_").replace(" ", "_").replace("(", "").replace(")", "")
        rubric_items.append(
            BinaryRubricCategory(
                name=f"fix_{field_safe}",
                question=desc,
                points=2,
            )
        )

    rubric_items += [
        BinaryRubricCategory(
            name="valid_syntax",
            question=f"Is the fixed config file valid {ext.upper()} that can be parsed without syntax errors?",
            points=2,
        ),
        BinaryRubricCategory(
            name="preserves_correct_values",
            question="Are all values that were already correct in the original config preserved unchanged in the fixed version?",
            points=2,
        ),
        RubricCategory(
            name="completeness",
            description="Were all errors from the error log addressed?",
            failure="None or only one error was fixed.",
            minor_failure="About half the errors were fixed.",
            minor_success="Most errors were fixed with one remaining.",
            success="All errors identified in the error log were correctly fixed.",
            points=3,
        ),
        RubricCategory(
            name="diagnostic_approach",
            description="Did the agent methodically diagnose each error by cross-referencing the error log and docs?",
            failure="No evidence of systematic debugging; random changes made.",
            minor_failure="Some consultation of docs but changes appear guessed.",
            minor_success="Consulted error log and docs; mostly systematic approach.",
            success="Methodically addressed each error by referencing both the error log and the config reference.",
            points=2,
        ),
    ]

    return RubricDatapoint(
        problem_statement=f"""# Config Debugging: Fix Broken Configuration

An application failed to start due to configuration errors.

Files provided:
- {config_path} -- the broken configuration file
- /testbed/logs/error.log -- error messages from the failed startup
- /testbed/docs/config_reference.txt -- documentation showing the correct config format and valid values

Your task:
1. Read the error log to understand what went wrong
2. Consult the configuration reference for correct formats and valid values
3. Fix all errors in the config file
4. Save the corrected config to {fixed_path}

IMPORTANT: Only fix actual errors. Do not change values that are already correct.""",
        rubric=tuple(rubric_items),
        submission_instructions=f"Save your corrected configuration to {fixed_path}",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.EDIT_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files=necessary_files,
        problem_type="config_debugging",
    )


# ============================================================================
# 4. DATA TRANSFORMATION — transform CSV per output spec
# ============================================================================

_DEPARTMENTS = ["Engineering", "Marketing", "Sales", "HR", "Finance", "Operations"]
_REGIONS = ["North", "South", "East", "West"]
_STATUSES = ["active", "inactive", "pending"]


def make_data_transformation(rand_seed: int = 42) -> RubricDatapoint:
    """Given input CSV and output spec, produce a transformed output CSV.

    The seed varies the data content, column names, filtering condition,
    derived column, and grouping.
    """
    rng = _random.Random(rand_seed)
    n_rows = vary_int(50, rand_seed, pct=0.15)

    # Choose column rename mapping
    rename_options = [
        {"emp_id": "employee_id", "dept": "department", "sal": "salary",
         "yrs": "years_experience", "region": "region", "status": "status"},
        {"id": "employee_id", "department": "department", "compensation": "salary",
         "experience": "years_experience", "area": "region", "state": "status"},
    ]
    rename_map = rng.choice(rename_options)
    orig_cols = list(rename_map.keys())
    new_cols = list(rename_map.values())

    # Choose filter condition
    filter_options = [
        ("active", "status", "active", "Include only rows where status is 'active'"),
        ("high_experience", "years_experience", "5+", "Include only rows where years_experience >= 5"),
        ("specific_region", "region", None, None),  # filled below
    ]
    filter_choice = rng.choice(filter_options)
    if filter_choice[0] == "specific_region":
        chosen_region = rng.choice(_REGIONS)
        filter_desc = f"Include only rows where region is '{chosen_region}'"
        filter_key = "region"
        filter_val = chosen_region
    else:
        filter_desc = filter_choice[3]
        filter_key = filter_choice[1]
        filter_val = filter_choice[2]

    # Choose derived column
    derived_options = [
        ("salary_band", "Based on salary: 'Junior' if salary < 60000, 'Mid' if 60000 <= salary < 90000, 'Senior' if salary >= 90000"),
        ("experience_level", "Based on years_experience: 'Entry' if < 3, 'Mid' if 3-7, 'Senior' if > 7"),
        ("annual_bonus", "Compute as salary * 0.10 (10% bonus), rounded to 2 decimal places"),
    ]
    derived_name, derived_desc = rng.choice(derived_options)

    # Choose grouping
    group_options = [
        ("department", "Compute average salary per department"),
        ("region", "Compute total headcount per region"),
    ]
    group_col, group_desc = rng.choice(group_options)

    # Generate input data
    header = ",".join(orig_cols)
    rows: list[str] = []
    for i in range(1, n_rows + 1):
        dept = rng.choice(_DEPARTMENTS)
        region = rng.choice(_REGIONS)
        status = rng.choice(_STATUSES)
        years = rng.randint(1, 20)
        base = {"Engineering": 85000, "Marketing": 65000, "Sales": 60000,
                "HR": 62000, "Finance": 75000, "Operations": 58000}[dept]
        salary = base + years * 2000 + rng.randint(-8000, 8000)
        # Build row using original column order
        vals = {
            "emp_id": str(i), "id": str(i),
            "dept": dept, "department": dept,
            "sal": str(salary), "compensation": str(salary),
            "yrs": str(years), "experience": str(years),
            "region": region, "area": region,
            "status": status, "state": status,
        }
        row_vals = [vals.get(c, "") for c in orig_cols]
        rows.append(",".join(row_vals))

    csv_content = header + "\n" + "\n".join(rows) + "\n"

    # Count expected output rows for the filter
    data_rows = []
    for row in rows:
        fields = row.split(",")
        row_dict = dict(zip(orig_cols, fields))
        # Map to new column names
        mapped = {new_cols[i]: fields[i] for i in range(len(orig_cols))}
        data_rows.append(mapped)

    if filter_key == "status":
        filtered = [r for r in data_rows if r.get("status") == filter_val]
    elif filter_key == "years_experience":
        filtered = [r for r in data_rows if int(r.get("years_experience", "0")) >= 5]
    elif filter_key == "region":
        filtered = [r for r in data_rows if r.get("region") == filter_val]
    else:
        filtered = data_rows

    expected_row_count = len(filtered)

    # Output spec
    output_spec = f"""# Output Specification for Data Transformation

## Input
File: /testbed/data/input.csv

## Required Transformations (apply in order)

### 1. Rename Columns
{chr(10).join(f'  - "{k}" -> "{v}"' for k, v in rename_map.items())}

### 2. Filter Rows
{filter_desc}

### 3. Add Derived Column: "{derived_name}"
{derived_desc}

### 4. Aggregation Summary
At the END of the output file, after a blank line, include a summary section:
  - Header line: "# SUMMARY"
  - {group_desc}
  - Format each line as: {group_col},<value>

### 5. Output Format
- File: /testbed/data/output.csv
- Columns: {', '.join(new_cols + [derived_name])}
- Standard CSV format with header row
- Summary section appended after data rows (separated by blank line)
"""

    necessary_files = {
        "/testbed/data/input.csv": csv_content,
        "/testbed/docs/output_spec.txt": output_spec,
    }

    rubric_items: list = [
        BinaryRubricCategory(
            name="output_file_exists",
            question="Does /testbed/data/output.csv exist and contain non-empty data?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_column_names",
            question=f"Does the output CSV header contain exactly the columns: {', '.join(new_cols + [derived_name])}?",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_row_count",
            question=f"After filtering, does the output contain approximately {expected_row_count} data rows (the correct number after applying the filter)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="derived_column_present",
            question=f'Does the output contain a "{derived_name}" column with values computed according to the spec?',
            points=2,
        ),
        BinaryRubricCategory(
            name="filter_applied",
            question=f"Were rows correctly filtered according to the rule: {filter_desc}?",
            points=2,
        ),
        BinaryRubricCategory(
            name="columns_renamed",
            question="Were all columns renamed according to the mapping in the spec?",
            points=1,
        ),
        BinaryRubricCategory(
            name="summary_present",
            question=f"Does the output file contain a summary section with {group_desc.lower()}?",
            points=2,
        ),
        BinaryRubricCategory(
            name="csv_format_valid",
            question="Is the data portion of the output (above any summary section) valid, well-formed CSV (parseable without errors)?",
            points=1,
        ),
        RubricCategory(
            name="code_quality",
            description="If the agent wrote a script, is it clean, readable, and efficient?",
            failure="No script written, or script is non-functional / incomprehensible.",
            minor_failure="Script works but is messy, overly complex, or has poor variable names.",
            minor_success="Script is readable and functional with minor style issues.",
            success="Clean, well-structured script with clear variable names and comments.",
            points=2,
        ),
        RubricCategory(
            name="derived_column_accuracy",
            description="Are the computed values in the derived column correct for all rows?",
            failure="Derived column is missing or all values are wrong.",
            minor_failure="Derived column present but many values are incorrect.",
            minor_success="Most derived values are correct with a few errors.",
            success="All derived column values are correctly computed per the spec.",
            points=3,
        ),
    ]

    return RubricDatapoint(
        problem_statement=f"""# Data Transformation Task

You have an input CSV file at /testbed/data/input.csv with {n_rows} rows of employee data.

Read the output specification at /testbed/docs/output_spec.txt and transform the input
data accordingly. Save the result to /testbed/data/output.csv.

The spec requires: column renaming, row filtering, computing a derived column,
and appending an aggregation summary.

You may use Python, bash, awk, or any available tools.""",
        rubric=tuple(rubric_items),
        submission_instructions="Save your transformed data to /testbed/data/output.csv",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files=necessary_files,
        problem_type="data_transformation",
    )


# ============================================================================
# 5. CRON SCHEDULING — write crontab entries from requirements
# ============================================================================

_JOB_POOL: list[dict[str, str]] = [
    {
        "description": "Run database backup every day at 3:00 AM",
        "command": "/opt/scripts/db_backup.sh",
        "cron": "0 3 * * *",
    },
    {
        "description": "Send weekly status report every Monday at 9:00 AM",
        "command": "/opt/scripts/weekly_report.sh",
        "cron": "0 9 * * 1",
    },
    {
        "description": "Clean temporary files every 6 hours",
        "command": "/opt/scripts/clean_tmp.sh",
        "cron": "0 */6 * * *",
    },
    {
        "description": "Run system maintenance on the 1st of each month at midnight",
        "command": "/opt/scripts/maintenance.sh",
        "cron": "0 0 1 * *",
    },
    {
        "description": "Rotate application logs every day at 11:30 PM",
        "command": "/opt/scripts/rotate_logs.sh",
        "cron": "30 23 * * *",
    },
    {
        "description": "Sync files to backup server every 15 minutes",
        "command": "/opt/scripts/sync_backup.sh",
        "cron": "*/15 * * * *",
    },
    {
        "description": "Generate daily analytics digest at 6:00 AM on weekdays (Mon-Fri)",
        "command": "/opt/scripts/analytics_digest.sh",
        "cron": "0 6 * * 1-5",
    },
    {
        "description": "Archive old database records on the 15th of each month at 2:00 AM",
        "command": "/opt/scripts/archive_records.sh",
        "cron": "0 2 15 * *",
    },
    {
        "description": "Check disk usage every 30 minutes and alert if over 90%",
        "command": "/opt/scripts/disk_check.sh",
        "cron": "*/30 * * * *",
    },
    {
        "description": "Restart the web server every Sunday at 4:00 AM for maintenance",
        "command": "/opt/scripts/restart_web.sh",
        "cron": "0 4 * * 0",
    },
    {
        "description": "Pull latest security patches every Wednesday at 1:00 AM",
        "command": "/opt/scripts/security_update.sh",
        "cron": "0 1 * * 3",
    },
    {
        "description": "Send quarterly financial summary on the 1st of January, April, July, October at 8:00 AM",
        "command": "/opt/scripts/quarterly_report.sh",
        "cron": "0 8 1 1,4,7,10 *",
    },
]

_CRON_REFERENCE = """# Cron Expression Quick Reference
# ================================
#
# Format: MINUTE HOUR DAY_OF_MONTH MONTH DAY_OF_WEEK COMMAND
#
# Field          Allowed Values
# -----          --------------
# MINUTE         0-59
# HOUR           0-23
# DAY_OF_MONTH   1-31
# MONTH          1-12 (or names: jan, feb, ...)
# DAY_OF_WEEK    0-7  (0 and 7 = Sunday, or names: sun, mon, ...)
#
# Special Characters:
#   *    Any value
#   ,    Value list separator (e.g., 1,3,5)
#   -    Range (e.g., 1-5 = Monday through Friday)
#   /    Step values (e.g., */15 = every 15 units)
#
# Examples:
#   0 3 * * *       Every day at 3:00 AM
#   */15 * * * *    Every 15 minutes
#   0 9 * * 1       Every Monday at 9:00 AM
#   0 0 1 * *       First of each month at midnight
#   30 23 * * 1-5   Weekdays at 11:30 PM
#
# Best Practices:
#   - Add comments above each entry describing the job
#   - Redirect stdout/stderr to a log file: >> /var/log/job.log 2>&1
#   - Use full paths for all commands and scripts
"""


def make_cron_scheduling(rand_seed: int = 42) -> RubricDatapoint:
    """Given job requirements, write crontab entries.

    The seed varies which jobs are selected and their command paths.
    """
    rng = _random.Random(rand_seed)
    n_jobs = rng.randint(5, 6)
    chosen_jobs = rng.sample(_JOB_POOL, n_jobs)

    # Build requirements document
    req_lines = [
        "# Job Scheduling Requirements",
        "# ============================",
        "",
        "Create a crontab file with entries for each of the following jobs.",
        "Use the exact script paths provided. Add a comment above each entry.",
        "Redirect all output (stdout and stderr) to a log file under /var/log/cron/.",
        "",
    ]
    for i, job in enumerate(chosen_jobs, 1):
        req_lines.append(f"Job {i}: {job['description']}")
        req_lines.append(f"  Script: {job['command']}")
        log_name = job["command"].split("/")[-1].replace(".sh", "")
        req_lines.append(f"  Log: /var/log/cron/{log_name}.log")
        req_lines.append("")

    requirements_text = "\n".join(req_lines)

    necessary_files = {
        "/testbed/docs/job_requirements.txt": requirements_text,
        "/testbed/docs/cron_reference.txt": _CRON_REFERENCE,
    }

    # Build rubric.
    # We check each job's cron expression individually (binary), but
    # consolidate command-path correctness and other structural checks
    # into graded categories to keep the binary ratio in 60-80%.
    rubric_items: list = [
        BinaryRubricCategory(
            name="crontab_file_exists",
            question="Does /testbed/crontab.txt exist and contain non-empty text?",
            points=1,
        ),
        BinaryRubricCategory(
            name="all_jobs_present",
            question=f"Does the crontab file contain entries for all {n_jobs} required jobs?",
            points=2,
        ),
    ]

    for i, job in enumerate(chosen_jobs, 1):
        job_name = job["command"].split("/")[-1].replace(".sh", "")
        rubric_items.append(
            BinaryRubricCategory(
                name=f"job_{i}_{job_name}_cron_correct",
                question=(
                    f"Is the cron expression for Job {i} ('{job['description']}') correct? "
                    f"The expected schedule is: {job['cron']}"
                ),
                points=2,
            )
        )

    rubric_items += [
        BinaryRubricCategory(
            name="all_command_paths_correct",
            question=(
                "Do all entries use the exact script paths specified in the requirements? "
                "Expected paths: " + ", ".join(j["command"] for j in chosen_jobs)
            ),
            points=2,
        ),
        RubricCategory(
            name="comments_present",
            description="Does each crontab entry have a descriptive comment above it?",
            failure="No comments present in the crontab file.",
            minor_failure="Some entries have comments but most do not.",
            minor_success="Most entries have comments; one or two are missing.",
            success="Every entry has a clear, descriptive comment explaining the job.",
            points=2,
        ),
        RubricCategory(
            name="output_redirection",
            description="Do entries redirect stdout and stderr to the specified log files?",
            failure="No output redirection on any entry.",
            minor_failure="Some entries have redirection but most do not, or redirection is incorrect.",
            minor_success="Most entries have proper redirection; one or two are missing.",
            success="All entries redirect both stdout and stderr (>> logfile 2>&1) to the correct log paths.",
            points=2,
        ),
        RubricCategory(
            name="crontab_formatting",
            description="Is the crontab file well-formatted and following best practices?",
            failure="Entries are garbled, unreadable, or clearly not valid crontab syntax.",
            minor_failure="Entries are parseable but messy, inconsistent spacing, or missing structure.",
            minor_success="Clean formatting with minor inconsistencies.",
            success="Clean, consistently formatted crontab with proper spacing and organization.",
            points=2,
        ),
    ]

    return RubricDatapoint(
        problem_statement=f"""# Cron Scheduling Task

You need to create a crontab file with scheduled job entries.

Read the job requirements at /testbed/docs/job_requirements.txt and the cron
syntax reference at /testbed/docs/cron_reference.txt.

Write a complete crontab file to /testbed/crontab.txt with an entry for each
required job. Each entry should:
- Have a descriptive comment above it
- Use the correct cron schedule expression
- Use the exact script path from the requirements
- Redirect stdout and stderr to the specified log file

There are {n_jobs} jobs to schedule.""",
        rubric=tuple(rubric_items),
        submission_instructions="Write your crontab to /testbed/crontab.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files=necessary_files,
        problem_type="cron_scheduling",
    )


# ============================================================================
# 6. API DOCUMENTATION — static, no seed
# ============================================================================

_API_SOURCE = '''\
"""User Management API

A RESTful API for managing users and authentication.
"""
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class User:
    """Represents a user account."""
    id: int
    username: str
    email: str
    created_at: datetime
    is_active: bool = True
    role: str = "user"  # "user", "admin", "moderator"


def create_user(username: str, email: str, password: str,
                role: str = "user") -> User:
    """Create a new user account.

    Validates that the username is unique and the email is well-formed.
    Passwords are hashed with bcrypt before storage.

    Args:
        username: Unique username (3-30 alphanumeric characters).
        email: Valid email address.
        password: Plain-text password (min 8 chars, must include a number).
        role: User role. Defaults to "user".

    Returns:
        The newly created User object.

    Raises:
        ValueError: If username is taken or email is invalid.
        ValidationError: If password does not meet complexity requirements.

    Example:
        >>> user = create_user("alice", "alice@example.com", "secret123")
        >>> user.username
        \'alice\'
    """
    ...


def get_user(user_id: int, include_inactive: bool = False) -> Optional[User]:
    """Retrieve a user by ID.

    Args:
        user_id: The unique user identifier.
        include_inactive: If True, return the user even if deactivated.
            Defaults to False.

    Returns:
        The User object, or None if not found (or inactive and
        include_inactive is False).

    Raises:
        PermissionError: If the caller lacks read access to user data.

    Example:
        >>> user = get_user(42)
        >>> user.email
        \'alice@example.com\'
    """
    ...


def update_user(user_id: int, *,
                email: Optional[str] = None,
                role: Optional[str] = None,
                is_active: Optional[bool] = None) -> User:
    """Update fields on an existing user.

    Only the provided keyword arguments are updated; others remain unchanged.
    Changing role to "admin" requires the caller to have admin privileges.

    Args:
        user_id: The user to update.
        email: New email address (validated if provided).
        role: New role. Changing to "admin" requires admin caller.
        is_active: Set active/inactive status.

    Returns:
        The updated User object.

    Raises:
        ValueError: If user_id does not exist.
        PermissionError: If promoting to admin without admin privileges.
        ValidationError: If the new email is malformed.

    Example:
        >>> updated = update_user(42, role="moderator")
        >>> updated.role
        \'moderator\'
    """
    ...


def list_users(role: Optional[str] = None,
               active_only: bool = True,
               page: int = 1,
               page_size: int = 50) -> list[User]:
    """List users with optional filtering and pagination.

    Args:
        role: Filter by role. None returns all roles.
        active_only: If True, exclude inactive users. Defaults to True.
        page: Page number (1-indexed). Defaults to 1.
        page_size: Results per page (max 100). Defaults to 50.

    Returns:
        A list of User objects for the requested page.

    Raises:
        ValueError: If page < 1 or page_size not in [1, 100].

    Example:
        >>> admins = list_users(role="admin")
        >>> len(admins)
        3
    """
    ...
'''

_STYLE_GUIDE = """# API Documentation Style Guide

## Format
Each function/endpoint must be documented with the following sections
in this exact order:

### 1. Title
Function name as a level-2 heading (## function_name)

### 2. Description
1-3 sentences describing what the function does, its main purpose,
and any important behavior notes.

### 3. Parameters
A table with columns: Name | Type | Required | Default | Description

### 4. Returns
Describe the return type and what it contains.

### 5. Errors
List each possible error/exception with:
- Error type/name
- When it occurs
- Brief description

### 6. Example
A complete usage example showing a typical call and its result.
Use code blocks with language annotation.

## General Rules
- Use clear, concise language
- Document ALL parameters, including optional ones with defaults
- Always specify types explicitly
- Examples must be syntactically valid
- Error documentation must cover all raised exceptions
- Maintain consistent formatting across all entries
"""


def make_api_documentation() -> RubricDatapoint:
    """Static task: document Python functions as an API reference.

    Given source code with docstrings and a style guide, produce a
    formatted API reference document.
    """
    necessary_files = {
        "/testbed/src/api.py": _API_SOURCE,
        "/testbed/docs/style_guide.txt": _STYLE_GUIDE,
    }

    rubric_items: list = [
        BinaryRubricCategory(
            name="output_file_exists",
            question="Does /testbed/docs/api_reference.txt exist and contain non-empty text?",
            points=1,
        ),
        BinaryRubricCategory(
            name="all_functions_documented",
            question="Are all 4 functions (create_user, get_user, update_user, list_users) documented?",
            points=2,
        ),
        BinaryRubricCategory(
            name="parameter_types_listed",
            question="Does each function's documentation list all parameters with their correct types?",
            points=2,
        ),
        BinaryRubricCategory(
            name="return_types_documented",
            question="Is the return type clearly documented for each function?",
            points=2,
        ),
        BinaryRubricCategory(
            name="examples_present",
            question="Does each function have at least one usage example in a code block?",
            points=2,
        ),
        BinaryRubricCategory(
            name="errors_documented",
            question="Are all raised exceptions/errors documented for each function (ValueError, PermissionError, ValidationError as applicable)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="parameter_table_format",
            question="Are parameters presented in a table format with Name, Type, Required, Default, and Description columns as specified in the style guide?",
            points=1,
        ),
        BinaryRubricCategory(
            name="optional_params_include_defaults",
            question="Do optional parameters have their default values documented?",
            points=1,
        ),
        RubricCategory(
            name="clarity",
            description="Is the documentation clearly written and easy to understand for a developer?",
            failure="Documentation is confusing, full of errors, or largely copy-pasted from source without formatting.",
            minor_failure="Understandable but poorly worded or inconsistent in style.",
            minor_success="Clear and well-written with minor wording or formatting issues.",
            success="Professional-quality documentation that is clear, concise, and consistently formatted.",
            points=3,
        ),
        RubricCategory(
            name="completeness",
            description="Does the reference cover everything a developer needs to use the API?",
            failure="Major sections missing (e.g., no parameters, no errors, no return types).",
            minor_failure="Some sections incomplete; a developer would need to read source for details.",
            minor_success="Nearly complete; one minor gap that a developer could infer.",
            success="Comprehensive: all parameters, return types, errors, and examples fully documented.",
            points=3,
        ),
    ]

    return RubricDatapoint(
        problem_statement="""# API Documentation Task

You are given a Python source file at /testbed/src/api.py containing 4 functions
for a User Management API. You are also given a documentation style guide at
/testbed/docs/style_guide.txt.

Your task: Write a complete API reference document following the style guide.

Read the source code to understand each function's purpose, parameters, return
types, and error conditions. Then produce a well-formatted reference document
at /testbed/docs/api_reference.txt.

The documentation should enable a developer to use the API without reading the
source code. Follow the style guide format exactly.""",
        rubric=tuple(rubric_items),
        submission_instructions="Write your API reference to /testbed/docs/api_reference.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files=necessary_files,
        problem_type="api_documentation",
    )
