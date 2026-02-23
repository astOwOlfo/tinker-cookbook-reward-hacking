"""Professional document tasks: performance review, event planning, lesson plan."""

import random as _random

from ..dataset import RubricDatapoint, RubricCategory, BinaryRubricCategory
from tinker_cookbook.rl.envs import tools
from ..content_pools import make_name, make_names, pick_one, vary_int, COMPANY_NAMES


# ============================================================================
# CONTENT POOLS
# ============================================================================

DEPARTMENTS = [
    "Engineering", "Product", "Design", "Data Science", "Marketing",
    "Sales", "Customer Success", "DevOps", "QA", "Research",
]

JOB_TITLES_ENGINEERING = [
    "Software Engineer", "Senior Software Engineer", "Staff Engineer",
    "Engineering Manager", "Principal Engineer", "Frontend Developer",
    "Backend Developer", "Full-Stack Engineer", "ML Engineer",
    "Platform Engineer", "Site Reliability Engineer",
]

JOB_TITLES_BUSINESS = [
    "Product Manager", "Senior Product Manager", "Marketing Manager",
    "Sales Director", "Account Executive", "Customer Success Manager",
    "Business Analyst", "Operations Manager", "Program Manager",
    "Growth Lead", "Partnership Manager",
]

GOAL_TEMPLATES = [
    ("Deliver {feature} by end of Q{quarter}", "delivery"),
    ("Reduce {metric} by {pct}% year-over-year", "improvement"),
    ("Mentor {count} junior team members through onboarding", "mentorship"),
    ("Complete {cert} certification", "development"),
    ("Improve {area} documentation coverage to {pct}%", "documentation"),
    ("Lead cross-functional initiative on {initiative}", "leadership"),
    ("Achieve {pct}% customer satisfaction score for {area}", "customer"),
    ("Migrate {system} to {target} architecture", "technical"),
    ("Establish {process} process for the team", "process"),
    ("Increase {metric} throughput by {pct}%", "performance"),
]

FEATURES = [
    "the new authentication module", "the payments v2 API",
    "the real-time dashboard", "the notification service rewrite",
    "the search indexing pipeline", "the mobile onboarding flow",
    "the data export feature", "the SSO integration",
]

METRICS = [
    "incident response time", "build pipeline duration",
    "customer-reported defects", "deployment failures",
    "time-to-first-response", "average resolution time",
    "page load latency", "error rate",
]

AREAS = [
    "API", "infrastructure", "testing", "deployment",
    "monitoring", "onboarding", "internal tooling", "security",
]

INITIATIVES = [
    "developer experience", "platform reliability",
    "data governance", "accessibility compliance",
    "performance optimization", "incident management",
]

CERTIFICATIONS = [
    "AWS Solutions Architect", "Google Cloud Professional",
    "Certified Scrum Master", "PMP", "Kubernetes Administrator",
    "Terraform Associate", "CISSP",
]

SYSTEMS = [
    "the legacy billing system", "the monolith order service",
    "the batch processing pipeline", "the internal CRM",
    "the reporting database",
]

TARGETS = [
    "microservices", "event-driven", "serverless", "cloud-native",
]

PROCESSES = [
    "a sprint retrospective", "a post-incident review",
    "a design review", "a capacity planning",
    "a knowledge-sharing", "a code review rotation",
]

PEER_FEEDBACK_POSITIVE = [
    "{name} is always willing to jump in when the team is under pressure. "
    "During the {event}, {pronoun} stayed late to help debug a critical issue "
    "that was blocking the release.",

    "Working with {name} on {project} was a great experience. "
    "{pronoun_cap} communicates clearly in code reviews and always explains "
    "the reasoning behind {possessive} suggestions, which helps the whole team learn.",

    "I appreciate how {name} proactively identifies risks. On the {project} project, "
    "{pronoun} flagged a potential scalability bottleneck weeks before it would have "
    "become a problem, saving us significant rework.",

    "{name} has a real talent for breaking down complex problems. When we were "
    "stuck on the {technical_challenge}, {pronoun} organized a whiteboarding session "
    "that got us to a solution in under an hour.",

    "I've seen {name} grow a lot this year. {pronoun_cap} has become the go-to person "
    "for {expertise}, and junior engineers frequently seek {possessive} advice.",
]

PEER_FEEDBACK_CONSTRUCTIVE = [
    "Sometimes {name} takes on too much and doesn't delegate. During {event}, "
    "{pronoun} was clearly overloaded but didn't ask for help until the deadline "
    "was at risk.",

    "{name} could improve on written communication. {pronoun_cap} emails and Slack "
    "messages are sometimes terse, which can come across as dismissive even though "
    "I know {pronoun} doesn't mean it that way.",

    "I'd like to see {name} participate more in cross-team meetings. {pronoun_cap} "
    "has valuable insights but tends to stay quiet unless directly asked, which means "
    "other teams don't benefit from {possessive} perspective.",

    "{name} is very detail-oriented, which is usually a strength, but occasionally "
    "{pronoun} gets stuck in the weeds on low-priority items. Better prioritization "
    "of review feedback would help.",

    "While {name}'s technical skills are strong, {pronoun} could benefit from more "
    "structured project planning. On {project}, there were a few missed intermediate "
    "milestones that caused last-minute scrambles.",
]

EVENTS = [
    "Q3 production outage", "end-of-year release push",
    "the platform migration sprint", "the security audit crunch",
    "the holiday traffic surge preparation",
]

PROJECTS = [
    "the payments rewrite", "the search overhaul", "the v2 API rollout",
    "the mobile app redesign", "the analytics pipeline rebuild",
    "the infrastructure modernization", "the SSO integration",
]

TECHNICAL_CHALLENGES = [
    "database sharding strategy", "caching layer design",
    "real-time event processing architecture",
    "authentication token refresh flow", "rate limiter design",
]

EXPERTISE_AREAS = [
    "distributed systems", "frontend performance",
    "database optimization", "CI/CD pipelines",
    "observability", "API design", "security best practices",
]

VENUE_NAMES = [
    "The Grand Ballroom at The Ritz", "Lakeside Conference Center",
    "Downtown Convention Hall", "Riverside Pavilion",
    "Hilltop Gardens Event Space", "The Metropolitan Club",
    "Harborview Terrace", "Sunset Ridge Country Club",
    "The Innovation Hub", "Parkside Event Center",
    "The Glass House Venue", "Maple Hall",
]

VENUE_AMENITIES_POOL = [
    "built-in AV system with projector and screen",
    "on-site catering kitchen",
    "complimentary parking (200 spaces)",
    "wheelchair accessible with elevator",
    "outdoor terrace/patio area",
    "coat check service",
    "dedicated event coordinator",
    "WiFi (1 Gbps)",
    "portable stage and podium",
    "dance floor",
    "green room / speaker prep area",
    "on-site security",
    "valet parking available (+$15/car)",
    "backup generator",
    "complimentary table linens and centerpieces",
]

EVENT_TYPES = [
    "annual company holiday party",
    "product launch celebration",
    "customer appreciation gala",
    "team-building offsite",
    "quarterly all-hands meeting",
    "charity fundraiser dinner",
    "summer company picnic",
    "executive leadership retreat",
    "new hire welcome event",
    "awards ceremony",
]

EVENT_MUST_HAVES = [
    "live music or DJ",
    "keynote presentation with AV support",
    "seated dinner service",
    "cocktail hour with hors d'oeuvres",
    "photo booth",
    "reserved VIP section for executives",
    "registration/check-in table at entrance",
    "branded signage and decorations",
    "raffle or door prize drawing",
    "after-party lounge area",
]

DIETARY_TYPES = [
    "vegetarian", "vegan", "gluten-free", "nut allergy",
    "dairy-free", "halal", "kosher", "shellfish allergy",
    "low-sodium", "pescatarian",
]

SUBJECTS = [
    "Introduction to Data Structures",
    "Principles of Microeconomics",
    "Creative Writing Workshop",
    "Environmental Science Fundamentals",
    "Statistics for Social Sciences",
    "Introduction to Machine Learning",
    "American History: 1865-Present",
    "Organic Chemistry I",
    "Business Communication Skills",
    "Introduction to Philosophy",
]

LEARNING_LEVELS = [
    ("introductory undergraduate", "no prior background assumed", "100-level"),
    ("intermediate undergraduate", "completion of the prerequisite course required", "200-level"),
    ("advanced undergraduate", "strong foundation in the subject expected", "300-level"),
    ("graduate seminar", "research experience expected", "500-level"),
    ("professional development", "working professionals with domain experience", "continuing education"),
]

MATERIALS_POOL = [
    "whiteboard and dry-erase markers",
    "projector and screen",
    "laptop with presentation software",
    "printed handouts (up to 40 copies)",
    "classroom response system (clickers)",
    "portable speakers",
    "flip chart and easel",
    "sticky notes and index cards",
    "video conferencing setup (Zoom)",
    "breakout room access (3 rooms available)",
    "desktop computers for each student",
    "lab equipment as per department inventory",
    "online learning management system (Canvas)",
    "sample datasets (pre-loaded on student machines)",
    "textbook: current edition available in library reserves",
    "guest speaker budget ($200 available)",
]

PREREQUISITE_KNOWLEDGE = [
    "basic algebra and arithmetic",
    "introductory programming (any language)",
    "high school biology",
    "college-level writing proficiency",
    "prior coursework in the discipline",
    "no specific prerequisites",
    "familiarity with spreadsheets and basic data analysis",
    "working knowledge of calculus",
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _pronoun_set(rng: _random.Random) -> dict[str, str]:
    """Return a consistent pronoun set chosen at random."""
    if rng.random() < 0.5:
        return {"pronoun": "she", "pronoun_cap": "She", "possessive": "her"}
    return {"pronoun": "he", "pronoun_cap": "He", "possessive": "his"}


def _generate_goals(rng: _random.Random, n: int = 5) -> list[dict]:
    """Generate a list of annual goals with met/not-met status."""
    templates = rng.sample(GOAL_TEMPLATES, min(n, len(GOAL_TEMPLATES)))
    goals = []
    for i, (template, category) in enumerate(templates):
        goal_text = template.format(
            feature=rng.choice(FEATURES),
            metric=rng.choice(METRICS),
            pct=rng.randint(10, 40),
            quarter=rng.randint(1, 4),
            count=rng.randint(2, 4),
            cert=rng.choice(CERTIFICATIONS),
            area=rng.choice(AREAS),
            initiative=rng.choice(INITIATIVES),
            system=rng.choice(SYSTEMS),
            target=rng.choice(TARGETS),
            process=rng.choice(PROCESSES),
        )
        # Ensure a mix: roughly 60% met, 40% not met
        met = rng.random() < 0.6
        if met:
            progress_note = rng.choice([
                "Completed ahead of schedule.",
                "Achieved target successfully.",
                "Fully delivered as planned.",
                "Met with strong results.",
            ])
        else:
            progress_note = rng.choice([
                "Partially completed; blocked by resource constraints.",
                "Not achieved; deprioritized mid-year due to shifting priorities.",
                "In progress but behind schedule; ~60% complete.",
                "Not met; scope expanded beyond original estimate.",
            ])
        goals.append({
            "number": i + 1,
            "text": goal_text,
            "category": category,
            "met": met,
            "progress_note": progress_note,
        })
    return goals


def _generate_monthly_metrics(rng: _random.Random, n_months: int = 12, is_eng: bool = True) -> list[dict]:
    """Generate monthly performance metrics for an employee.

    Engineering roles get: tickets_closed, code_reviews, oncall_hours, pr_turnaround.
    Business roles get: deals_closed, client_meetings, proposals_submitted, satisfaction_score.
    Both get: meetings_attended.
    """
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    metrics = []

    if is_eng:
        base_tickets = rng.randint(12, 25)
        base_reviews = rng.randint(8, 18)
        base_meetings = rng.randint(15, 25)
        base_oncall_hours = rng.randint(10, 30)
        base_pr_turnaround = round(rng.uniform(1.5, 4.0), 1)

        for i in range(min(n_months, 12)):
            tickets = max(0, base_tickets + rng.randint(-5, 8) + (i // 4))
            reviews = max(0, base_reviews + rng.randint(-4, 6))
            meetings = max(5, base_meetings + rng.randint(-5, 5))
            oncall = max(0, base_oncall_hours + rng.randint(-8, 12))
            pr_turnaround = max(0.5, round(base_pr_turnaround + rng.uniform(-1.0, 1.0), 1))

            metrics.append({
                "month": months[i],
                "tickets_closed": tickets,
                "code_reviews_completed": reviews,
                "meetings_attended": meetings,
                "oncall_hours": oncall,
                "avg_pr_review_turnaround_hours": pr_turnaround,
            })
    else:
        base_deals = rng.randint(3, 10)
        base_client_meetings = rng.randint(8, 20)
        base_meetings = rng.randint(15, 25)
        base_proposals = rng.randint(2, 8)
        base_satisfaction = round(rng.uniform(3.5, 4.5), 1)

        for i in range(min(n_months, 12)):
            deals = max(0, base_deals + rng.randint(-2, 4))
            client_meetings = max(1, base_client_meetings + rng.randint(-5, 6))
            meetings = max(5, base_meetings + rng.randint(-5, 5))
            proposals = max(0, base_proposals + rng.randint(-2, 3))
            satisfaction = max(1.0, min(5.0, round(base_satisfaction + rng.uniform(-0.5, 0.5), 1)))

            metrics.append({
                "month": months[i],
                "deals_closed": deals,
                "client_meetings": client_meetings,
                "meetings_attended": meetings,
                "proposals_submitted": proposals,
                "client_satisfaction_score": satisfaction,
            })
    return metrics


# ============================================================================
# FACTORY 1: PERFORMANCE REVIEW SUMMARY
# ============================================================================


def make_performance_review_summary(rand_seed: int = 42) -> RubricDatapoint:
    """Given employee metrics CSV, goals document, and peer feedback, write a
    performance review summary.

    Problem type: performance_review

    Files:
        /testbed/data/metrics.csv       -- 12 months of quantitative data
        /testbed/docs/goals.txt         -- Annual goals with status
        /testbed/docs/peer_feedback.txt -- 3 peer feedback excerpts
    Submission: /testbed/review_summary.txt
    """
    rng = _random.Random(rand_seed)

    # --- Employee identity ---
    employee_name = make_name(rand_seed)
    manager_name = make_name(rand_seed + 1000)
    company = pick_one(COMPANY_NAMES, rand_seed + 2000)
    department = rng.choice(DEPARTMENTS)
    is_eng = department in ("Engineering", "DevOps", "QA", "Data Science", "Research")
    title = rng.choice(JOB_TITLES_ENGINEERING if is_eng else JOB_TITLES_BUSINESS)
    review_period = f"January {rng.choice([2023, 2024])} - December {rng.choice([2023, 2024])}"
    # Make sure end year >= start year
    start_year = int(review_period.split(" - ")[0].split()[-1])
    end_year = max(start_year, int(review_period.split(" - ")[1].split()[-1]))
    review_period = f"January {start_year} - December {end_year}"

    # --- Generate metrics CSV ---
    monthly_metrics = _generate_monthly_metrics(rng, is_eng=is_eng)
    if is_eng:
        csv_header = "month,tickets_closed,code_reviews_completed,meetings_attended,oncall_hours,avg_pr_review_turnaround_hours"
        csv_rows = [csv_header]
        for m in monthly_metrics:
            csv_rows.append(
                f"{m['month']},{m['tickets_closed']},{m['code_reviews_completed']},"
                f"{m['meetings_attended']},{m['oncall_hours']},{m['avg_pr_review_turnaround_hours']}"
            )
    else:
        csv_header = "month,deals_closed,client_meetings,meetings_attended,proposals_submitted,client_satisfaction_score"
        csv_rows = [csv_header]
        for m in monthly_metrics:
            csv_rows.append(
                f"{m['month']},{m['deals_closed']},{m['client_meetings']},"
                f"{m['meetings_attended']},{m['proposals_submitted']},{m['client_satisfaction_score']}"
            )
    metrics_csv = "\n".join(csv_rows) + "\n"

    # --- Generate goals document ---
    goals = _generate_goals(rng, n=rng.randint(4, 5))
    goals_lines = [
        f"ANNUAL PERFORMANCE GOALS — {review_period}",
        f"Employee: {employee_name}",
        f"Title: {title}",
        f"Department: {department}",
        f"Manager: {manager_name}",
        "",
        "=" * 60,
        "",
    ]
    for g in goals:
        status_label = "MET" if g["met"] else "NOT MET"
        goals_lines.extend([
            f"GOAL {g['number']}: {g['text']}",
            f"  Category: {g['category'].replace('_', ' ').title()}",
            f"  Status: {status_label}",
            f"  Notes: {g['progress_note']}",
            "",
        ])
    goals_text = "\n".join(goals_lines)

    # --- Generate peer feedback ---
    pronouns = _pronoun_set(rng)
    peers = make_names(rand_seed + 3000, 3)
    # Pick events/projects for feedback context
    event_context = rng.choice(EVENTS)
    project_context = rng.choice(PROJECTS)
    tech_challenge = rng.choice(TECHNICAL_CHALLENGES)
    expertise = rng.choice(EXPERTISE_AREAS)

    format_kwargs = {
        "name": employee_name.split()[0],
        "event": event_context,
        "project": project_context,
        "technical_challenge": tech_challenge,
        "expertise": expertise,
        **pronouns,
    }

    positive_templates = rng.sample(PEER_FEEDBACK_POSITIVE, 2)
    constructive_templates = rng.sample(PEER_FEEDBACK_CONSTRUCTIVE, 1)

    feedback_excerpts = []
    templates_used = positive_templates + constructive_templates
    rng.shuffle(templates_used)
    for i, (peer, template) in enumerate(zip(peers, templates_used)):
        text = template.format(**format_kwargs)
        feedback_excerpts.append(f"--- Feedback from {peer} ---\n{text}\n")

    feedback_text = (
        f"PEER FEEDBACK SUMMARY — {review_period}\n"
        f"Employee: {employee_name}\n"
        f"Collected from {len(peers)} peers\n\n"
        + "\n".join(feedback_excerpts)
    )

    # --- Compute summary stats for rubric references ---
    if is_eng:
        primary_metric_name = "tickets closed"
        primary_metric_total = sum(m["tickets_closed"] for m in monthly_metrics)
    else:
        primary_metric_name = "deals closed"
        primary_metric_total = sum(m["deals_closed"] for m in monthly_metrics)

    # Count met/not met for rubric
    goals_met = [g for g in goals if g["met"]]
    goals_not_met = [g for g in goals if not g["met"]]

    problem_statement = f"""# Performance Review Summary

You are {manager_name}, a manager at {company}. You need to write a
comprehensive performance review summary for your direct report,
{employee_name} ({title}, {department}).

Review the following documents:
- /testbed/data/metrics.csv — Monthly quantitative metrics for the review period
- /testbed/docs/goals.txt — Annual goals with achievement status
- /testbed/docs/peer_feedback.txt — Feedback collected from 3 peers

Write a performance review summary to /testbed/review_summary.txt that includes:

1. **Overview**: Brief summary of the employee's role and the review period
2. **Goals Assessment**: For each goal, state whether it was met or not met,
   citing relevant metrics from the CSV where applicable
3. **Strengths**: Based on peer feedback and metrics, identify 2-3 key strengths
4. **Development Areas**: Based on peer feedback and metrics, identify 1-2 areas
   for growth
5. **Quantitative Highlights**: Reference specific numbers from the metrics CSV
   (e.g., average tickets closed per month, trend over time)
6. **Overall Rating**: Provide an overall assessment (Exceeds Expectations /
   Meets Expectations / Below Expectations) with justification
7. **Recommendations**: 2-3 specific, actionable recommendations for the next
   review period

The review should be professional, balanced (acknowledging both strengths and
areas for improvement), and grounded in the data provided."""

    # Build rubric: 10-16 categories, 60-80% binary
    rubric = (
        # --- Structural / existence checks (binary, low points) ---
        BinaryRubricCategory(
            name="file_exists",
            question="Does /testbed/review_summary.txt exist with substantial content (at least 300 words)?",
            points=1,
        ),
        BinaryRubricCategory(
            name="employee_identified",
            question=f"Does the review summary identify the employee by name ({employee_name}) and title ({title})?",
            points=1,
        ),
        BinaryRubricCategory(
            name="review_period_stated",
            question=f"Does the review summary state the review period ({review_period})?",
            points=1,
        ),
        # --- Goals assessment (binary, medium-high points) ---
        BinaryRubricCategory(
            name="all_goals_addressed",
            question=f"Does the review address all {len(goals)} goals listed in the goals document?",
            points=2,
        ),
        BinaryRubricCategory(
            name="goal_1_correct_assessment",
            question=f"Does the review correctly assess Goal 1 ('{goals[0]['text'][:60]}...') as {'met' if goals[0]['met'] else 'not met'}?",
            points=2,
        ),
        BinaryRubricCategory(
            name="goals_met_count_accurate",
            question=f"Does the review correctly indicate that {len(goals_met)} goal(s) were met and {len(goals_not_met)} were not met (or is consistent with the goals document)?",
            points=2,
        ),
        RubricCategory(
            name="metric_integration",
            description="How effectively does the review integrate quantitative metrics from the CSV when discussing goals and performance?",
            failure="No specific numbers from the metrics CSV are cited anywhere in the review.",
            minor_failure="One or two numbers are mentioned but in a superficial way (e.g., just stating a total without connecting it to a goal).",
            minor_success="Several metrics are cited and connected to specific goals, but some obvious data points are missed.",
            success="Metrics are consistently woven into the goal assessment — e.g., citing monthly trends, averages, or comparisons to support each goal's met/not-met determination.",
            points=3,
        ),
        # --- Peer feedback integration (binary) ---
        BinaryRubricCategory(
            name="all_peer_feedback_referenced",
            question=f"Does the review reference or incorporate feedback from all {len(peers)} peers (by name or by paraphrasing each peer's distinct feedback)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="strengths_section_present",
            question="Does the review include a clearly identifiable strengths section listing at least 2 strengths?",
            points=2,
        ),
        BinaryRubricCategory(
            name="development_areas_present",
            question="Does the review include a clearly identifiable development areas (or areas for improvement) section?",
            points=2,
        ),
        # --- Quality / judgment (graded, high points) ---
        RubricCategory(
            name="balanced_assessment",
            description="Does the review present a balanced picture — acknowledging both strengths and weaknesses without being uniformly positive or negative?",
            failure="Review is entirely positive (ignores not-met goals and constructive feedback) or entirely negative.",
            minor_failure="Review acknowledges both sides but is noticeably skewed — e.g., constructive feedback is buried in a single sentence while strengths get several paragraphs.",
            minor_success="Review gives fair weight to both strengths and development areas, but the balance could be improved.",
            success="Review is genuinely balanced: strengths are substantiated with evidence, development areas are specific and fair, and the overall tone is constructive.",
            points=3,
        ),
        RubricCategory(
            name="actionable_recommendations",
            description="Are the recommendations specific and actionable (not generic platitudes)?",
            failure="No recommendations section, or only vague statements like 'keep up the good work'.",
            minor_failure="Recommendations exist but are generic (e.g., 'improve communication skills') without specific actions.",
            minor_success="Recommendations are somewhat specific but could be more concrete or tied to the employee's actual situation.",
            success="2-3 recommendations that are specific, actionable, and clearly tied to the data — e.g., 'Enroll in the Q2 leadership workshop to address the delegation feedback from peers' or 'Target 20+ code reviews/month to maintain the upward trend seen in H2'.",
            points=3,
        ),
        # --- Overall rating (binary) ---
        BinaryRubricCategory(
            name="overall_rating_included",
            question="Does the review include an explicit overall rating (e.g., 'Exceeds Expectations', 'Meets Expectations', or 'Below Expectations') with a justification?",
            points=2,
        ),
    )

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write your performance review summary to /testbed/review_summary.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.EDIT_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/metrics.csv": metrics_csv,
            "/testbed/docs/goals.txt": goals_text,
            "/testbed/docs/peer_feedback.txt": feedback_text,
        },
        problem_type="performance_review",
    )


# ============================================================================
# FACTORY 2: EVENT PLANNING
# ============================================================================


def make_event_planning(rand_seed: int = 42) -> RubricDatapoint:
    """Given venue options, budget, guest count, and dietary requirements,
    create a comprehensive event plan.

    Problem type: event_planning

    Files:
        /testbed/data/venues.csv          -- 3 venue options with details
        /testbed/docs/requirements.txt    -- Event requirements brief
        /testbed/data/dietary_needs.csv   -- Dietary breakdown for guests
    Submission: /testbed/event_plan.txt
    """
    rng = _random.Random(rand_seed)

    # --- Event parameters ---
    company = pick_one(COMPANY_NAMES, rand_seed + 100)
    organizer_name = make_name(rand_seed + 200)
    event_type = rng.choice(EVENT_TYPES)
    guest_count = vary_int(rng.choice([80, 120, 150, 200, 250]), rand_seed, pct=0.15)
    budget = vary_int(rng.choice([8000, 12000, 18000, 25000, 35000]), rand_seed + 1, pct=0.10)
    event_month = rng.choice(["March", "April", "May", "June", "September", "October", "November"])
    event_day = rng.randint(5, 25)
    event_year = rng.choice([2024, 2025])
    event_date = f"{event_month} {event_day}, {event_year}"
    event_time = rng.choice(["6:00 PM - 10:00 PM", "5:30 PM - 9:30 PM", "7:00 PM - 11:00 PM", "12:00 PM - 4:00 PM"])

    # --- Generate 3 venue options ---
    venue_names_selected = rng.sample(VENUE_NAMES, 3)
    # Shuffle which venue index gets each "role" so the optimal isn't always #1
    venue_roles = ["too_small", "good_fit", "too_large"]
    rng.shuffle(venue_roles)
    venues = []
    for i, vname in enumerate(venue_names_selected):
        role = venue_roles[i]
        # Capacity varies by role
        if role == "too_small":
            capacity = vary_int(int(guest_count * 0.75), rand_seed + 300 + i, pct=0.05)
        elif role == "good_fit":
            capacity = vary_int(int(guest_count * 1.3), rand_seed + 300 + i, pct=0.10)
        else:
            capacity = vary_int(int(guest_count * 2.0), rand_seed + 300 + i, pct=0.10)

        # Cost varies by role
        base_cost_per_person = rng.uniform(25, 80)
        venue_cost = int(capacity * base_cost_per_person)
        if role == "too_small":
            venue_cost = vary_int(int(budget * 0.6), rand_seed + 400 + i, pct=0.08)
        elif role == "good_fit":
            venue_cost = vary_int(int(budget * 0.7), rand_seed + 400 + i, pct=0.08)
        else:
            venue_cost = vary_int(int(budget * 1.2), rand_seed + 400 + i, pct=0.08)

        amenities = rng.sample(VENUE_AMENITIES_POOL, rng.randint(4, 7))
        venues.append({
            "name": vname,
            "capacity": capacity,
            "rental_cost": venue_cost,
            "catering_per_person": rng.choice([35, 45, 55, 65, 75]),
            "amenities": amenities,
            "availability": rng.choice([
                f"Available {event_date}",
                f"Available {event_date} (must confirm by 2 weeks prior)",
                f"Not available {event_date}; available {event_month} {min(event_day + 7, 28)}, {event_year}",
            ]) if role != "good_fit" else f"Available {event_date}",
        })

    # Venue CSV
    venues_csv_lines = ["venue_name,capacity,rental_cost_usd,catering_per_person_usd,availability,amenities"]
    for v in venues:
        amenities_str = "; ".join(v["amenities"])
        venues_csv_lines.append(
            f'"{v["name"]}",{v["capacity"]},{v["rental_cost"]},{v["catering_per_person"]},'
            f'"{v["availability"]}","{amenities_str}"'
        )
    venues_csv = "\n".join(venues_csv_lines) + "\n"

    # --- Requirements document ---
    must_haves = rng.sample(EVENT_MUST_HAVES, rng.randint(3, 5))
    requirements_text = f"""EVENT PLANNING BRIEF — {company}
Prepared by: {organizer_name}
Date: {event_date}

EVENT OVERVIEW
  Type: {event_type.title()}
  Expected Guests: {guest_count}
  Date: {event_date}
  Time: {event_time}
  Total Budget: ${budget:,}
    (This budget must cover venue rental, catering, and all required amenities.
     It does NOT need to cover employee transportation or personal expenses.)

MUST-HAVE REQUIREMENTS
{chr(10).join(f"  {i+1}. {item}" for i, item in enumerate(must_haves))}

ADDITIONAL NOTES
  - The event must be fully accessible (wheelchair, elevator)
  - All dietary requirements (see dietary_needs.csv) must be accommodated
  - Setup access needed at least 2 hours before event start
  - Parking for at least {vary_int(int(guest_count * 0.6), rand_seed + 500, pct=0.1)} vehicles
"""

    # --- Dietary needs CSV ---
    dietary_selections = rng.sample(DIETARY_TYPES, rng.randint(4, 6))
    dietary_lines = ["dietary_requirement,guest_count"]
    remaining = guest_count
    for j, diet in enumerate(dietary_selections):
        if j == len(dietary_selections) - 1:
            count = remaining
        else:
            count = rng.randint(
                max(1, int(remaining * 0.05)),
                max(2, int(remaining * 0.25)),
            )
            remaining -= count
        dietary_lines.append(f"{diet},{count}")
    # Add a "no restrictions" row for the rest
    no_restrictions = guest_count - sum(
        int(line.split(",")[1]) for line in dietary_lines[1:]
    )
    if no_restrictions > 0:
        dietary_lines.append(f"no restrictions,{no_restrictions}")
    dietary_csv = "\n".join(dietary_lines) + "\n"

    # Identify the best venue (fits capacity AND within budget)
    suitable_venues = []
    for v in venues:
        total_cost = v["rental_cost"] + v["catering_per_person"] * guest_count
        fits_capacity = v["capacity"] >= guest_count
        fits_budget = total_cost <= budget
        is_available = event_date in v["availability"] and "Not available" not in v["availability"]
        if fits_capacity:
            suitable_venues.append(v["name"])

    problem_statement = f"""# Event Planning

You are {organizer_name}, an event coordinator at {company}. You need to plan
the company's upcoming {event_type}.

Review the following documents:
- /testbed/data/venues.csv — Three venue options with capacity, cost, and amenities
- /testbed/docs/requirements.txt — Event requirements brief with budget and must-haves
- /testbed/data/dietary_needs.csv — Dietary requirement breakdown for all guests

Write a comprehensive event plan to /testbed/event_plan.txt that includes:

1. **Venue Selection**: Evaluate all three venues against the requirements
   (capacity, budget, availability, amenities). Select one and justify your choice.
   Show your cost calculations.

2. **Budget Breakdown**: Detailed budget showing venue rental, catering costs,
   and any additional expenses. Verify the total stays within the ${budget:,} budget.

3. **Dietary Accommodation Plan**: How you will handle each dietary requirement
   listed in the dietary needs file.

4. **Event Timeline**: Hour-by-hour timeline from setup through cleanup.

5. **Logistics Checklist**: Key logistical items (parking, accessibility,
   AV setup, signage, etc.)

6. **Contingency Notes**: At least 2 contingency plans for potential issues
   (weather, vendor no-show, over-attendance, etc.)

Be specific and reference the actual numbers from the provided files."""

    rubric = (
        # --- Structural (binary, low points) ---
        BinaryRubricCategory(
            name="file_exists",
            question="Does /testbed/event_plan.txt exist with substantial content (at least 400 words)?",
            points=1,
        ),
        # --- Venue selection (binary, medium-high points) ---
        BinaryRubricCategory(
            name="all_venues_evaluated",
            question=f"Does the plan evaluate all three venues ({venues[0]['name']}, {venues[1]['name']}, {venues[2]['name']}) against the requirements?",
            points=2,
        ),
        BinaryRubricCategory(
            name="venue_fits_capacity",
            question=f"Does the selected venue have capacity for at least {guest_count} guests? (The plan should not select a venue that is too small.)",
            points=3,
        ),
        BinaryRubricCategory(
            name="within_budget",
            question=f"Does the plan demonstrate that the total cost (venue rental + catering for {guest_count} guests + any extras) stays within the ${budget:,} budget?",
            points=3,
        ),
        RubricCategory(
            name="cost_calculations_shown",
            description="Does the plan show detailed, transparent cost calculations?",
            failure="No cost breakdown — just a stated total or no financial information at all.",
            minor_failure="Some costs are listed but without showing the math (e.g., states 'Catering: $5,000' without explaining per-person cost x guest count).",
            minor_success="Shows most calculations clearly (rental + catering breakdown) but misses one component or has a minor arithmetic error.",
            success="Full, transparent cost breakdown with per-person catering x guest count, venue rental, and any additional line items, all adding up correctly to a clearly stated total.",
            points=2,
        ),
        # --- Dietary (binary) ---
        BinaryRubricCategory(
            name="dietary_needs_addressed",
            question="Does the plan address each dietary requirement listed in the dietary needs CSV (at minimum mentioning each type and how it will be accommodated)?",
            points=2,
        ),
        # --- Timeline (binary) ---
        BinaryRubricCategory(
            name="timeline_included",
            question="Does the plan include an event timeline with at least 4 time-stamped entries covering setup through event end?",
            points=2,
        ),
        # --- Requirements coverage (binary) ---
        BinaryRubricCategory(
            name="must_haves_addressed",
            question=f"Does the plan address all {len(must_haves)} must-have requirements from the requirements brief ({'; '.join(must_haves[:3])}...)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="accessibility_addressed",
            question="Does the plan address accessibility requirements (wheelchair access, elevator, etc.) as specified in the requirements brief?",
            points=1,
        ),
        BinaryRubricCategory(
            name="parking_addressed",
            question="Does the plan address parking requirements?",
            points=1,
        ),
        # --- Quality / judgment (graded, high points) ---
        RubricCategory(
            name="logistics_thoroughness",
            description="How thorough and practical are the logistics details?",
            failure="No logistics details beyond venue name and date.",
            minor_failure="Some logistics mentioned but major gaps (e.g., no mention of AV, signage, registration, or staffing).",
            minor_success="Covers most logistical areas with reasonable detail but misses one or two important items.",
            success="Comprehensive logistics covering AV/tech setup, signage, registration flow, staffing/volunteers, vendor coordination, and day-of contact information.",
            points=3,
        ),
        RubricCategory(
            name="contingency_planning",
            description="Does the plan include realistic contingency plans?",
            failure="No contingency plans mentioned.",
            minor_failure="Mentions contingencies but they are vague ('have a backup plan').",
            minor_success="Includes 2 specific contingencies but they lack actionable detail (e.g., 'if it rains, move inside' without specifying how).",
            success="Includes 2+ specific, actionable contingency plans with concrete steps — e.g., 'If catering vendor cancels within 48 hours, contact backup vendor [X] at [phone]; pre-negotiate standby rate of $Y/person.'",
            points=3,
        ),
    )

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write your event plan to /testbed/event_plan.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.EDIT_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/venues.csv": venues_csv,
            "/testbed/docs/requirements.txt": requirements_text,
            "/testbed/data/dietary_needs.csv": dietary_csv,
        },
        problem_type="event_planning",
    )


# ============================================================================
# FACTORY 3: LESSON PLAN
# ============================================================================


def make_lesson_plan(rand_seed: int = 42) -> RubricDatapoint:
    """Given learning objectives, time constraint, available materials, and
    class profile, create a detailed lesson plan.

    Problem type: lesson_plan

    Files:
        /testbed/docs/learning_objectives.txt  -- Specific learning objectives
        /testbed/docs/available_materials.txt   -- Available materials/resources
        /testbed/docs/class_profile.txt         -- Class size, level, prerequisites
    Submission: /testbed/lesson_plan.txt
    """
    rng = _random.Random(rand_seed)

    # --- Class parameters ---
    instructor_name = make_name(rand_seed + 500)
    subject = rng.choice(SUBJECTS)
    level_name, level_desc, level_code = rng.choice(LEARNING_LEVELS)
    class_size = vary_int(rng.choice([18, 24, 30, 36]), rand_seed + 600, pct=0.15)
    time_slot = rng.choice([45, 90])
    time_label = f"{time_slot}-minute"
    semester = rng.choice(["Fall", "Spring"])
    session_number = rng.randint(3, 12)
    total_sessions = rng.randint(max(session_number + 3, 14), 30)

    # --- Generate learning objectives ---
    n_objectives = rng.randint(3, 5)
    objective_verbs = [
        ("Explain", "conceptual understanding"),
        ("Apply", "practical application"),
        ("Analyze", "analytical thinking"),
        ("Compare and contrast", "comparative analysis"),
        ("Design", "creative synthesis"),
        ("Evaluate", "critical judgment"),
        ("Demonstrate", "procedural skill"),
        ("Identify", "recognition and recall"),
    ]
    selected_verbs = rng.sample(objective_verbs, n_objectives)

    # Generate subject-appropriate objectives
    objectives = []
    used_topics = set()
    for i, (verb, bloom_level) in enumerate(selected_verbs):
        if "Data Structures" in subject:
            topics = ["linked lists vs. arrays", "tree traversal algorithms",
                      "hash table collision resolution", "stack and queue operations",
                      "Big-O notation for common operations"]
        elif "Microeconomics" in subject:
            topics = ["supply and demand equilibrium", "price elasticity",
                      "consumer surplus and producer surplus", "market structures",
                      "externalities and market failure"]
        elif "Creative Writing" in subject:
            topics = ["narrative point of view", "character development through dialogue",
                      "setting as a storytelling device", "revision strategies",
                      "showing vs. telling techniques"]
        elif "Environmental Science" in subject:
            topics = ["the carbon cycle", "biodiversity metrics",
                      "ecosystem services valuation", "water quality indicators",
                      "climate feedback loops"]
        elif "Statistics" in subject:
            topics = ["hypothesis testing procedures", "confidence interval interpretation",
                      "correlation vs. causation", "sampling bias",
                      "p-value interpretation"]
        elif "Machine Learning" in subject:
            topics = ["overfitting and regularization", "gradient descent optimization",
                      "feature engineering strategies", "model evaluation metrics",
                      "bias-variance tradeoff"]
        elif "History" in subject:
            topics = ["Reconstruction-era policies", "the Gilded Age labor movement",
                      "primary source analysis techniques", "the Progressive Era reforms",
                      "cause-and-effect reasoning in historical analysis"]
        elif "Chemistry" in subject:
            topics = ["functional group reactivity", "stereoisomerism",
                      "nucleophilic substitution mechanisms", "spectroscopy interpretation",
                      "reaction kinetics"]
        elif "Business Communication" in subject:
            topics = ["persuasive memo structure", "audience analysis",
                      "data visualization for presentations", "email etiquette norms",
                      "cross-cultural communication strategies"]
        else:  # Philosophy
            topics = ["the trolley problem and variants", "Kant's categorical imperative",
                      "utilitarian calculus", "virtue ethics framework",
                      "logical fallacy identification"]

        available_topics = [t for t in topics if t not in used_topics]
        if not available_topics:
            available_topics = topics  # fallback if pool exhausted
        topic = rng.choice(available_topics)
        used_topics.add(topic)
        objectives.append({
            "number": i + 1,
            "verb": verb,
            "topic": topic,
            "bloom_level": bloom_level,
            "text": f"{verb} {topic}",
        })

    objectives_text = f"""LEARNING OBJECTIVES — {subject}
Session {session_number} of {total_sessions}
Instructor: {instructor_name}
{semester} Semester

By the end of this session, students will be able to:

"""
    for obj in objectives:
        objectives_text += f"  {obj['number']}. {obj['text']}\n"
        objectives_text += f"     (Bloom's level: {obj['bloom_level']})\n\n"

    objectives_text += f"""
ALIGNMENT NOTE: Objectives 1-{n_objectives} map to Course Learning Outcome #{rng.randint(1, 5)}
in the syllabus. This session builds on Session {session_number - 1} and
prepares students for Session {session_number + 1}.
"""

    # --- Available materials ---
    n_materials = rng.randint(6, 10)
    available_materials = rng.sample(MATERIALS_POOL, min(n_materials, len(MATERIALS_POOL)))
    materials_text = f"""AVAILABLE MATERIALS AND RESOURCES — {subject}
Session {session_number}

The following materials are confirmed available for this session:

"""
    for i, mat in enumerate(available_materials):
        materials_text += f"  {i + 1}. {mat}\n"

    materials_text += f"""
ROOM ASSIGNMENT: {rng.choice(['Room 204, Science Building', 'Lecture Hall B', 'Seminar Room 310', 'Computer Lab 102', 'Room 118, Humanities'])}
ROOM CAPACITY: {vary_int(class_size + 10, rand_seed + 700, pct=0.1)} seats

NOTE: Additional materials may be requested through the department office
with 5 business days notice. Budget for supplementary materials: $50.
"""

    # --- Class profile ---
    prereq = rng.choice(PREREQUISITE_KNOWLEDGE)
    class_profile_text = f"""CLASS PROFILE — {subject}
{semester} Semester | Session {session_number}

DEMOGRAPHICS
  Enrolled students: {class_size}
  Typical attendance: {vary_int(int(class_size * 0.85), rand_seed + 800, pct=0.05)} ({round(85 + rng.uniform(-5, 5))}%)
  Level: {level_name} ({level_code})
  Prerequisite: {prereq}

STUDENT BACKGROUND
  - Approximately {rng.randint(20, 40)}% of students have prior exposure to the topic
    from high school or self-study
  - {rng.randint(10, 25)}% are non-native English speakers
  - {rng.randint(1, 4)} students have documented accommodations (extended time,
    preferential seating, note-taking assistance)
  - Class includes both majors ({rng.randint(40, 70)}%) and non-majors taking the
    course as an elective

ENGAGEMENT NOTES FROM PREVIOUS SESSIONS
  - Students responded well to: {rng.choice(['group activities', 'case studies', 'hands-on exercises', 'class discussions', 'peer review activities'])}
  - Students struggled with: {rng.choice(['abstract concepts without concrete examples', 'reading-heavy assignments', 'cold-calling / being put on the spot', 'connecting theory to practice', 'quantitative problem-solving'])}
  - Average quiz score (last 3 quizzes): {rng.randint(68, 88)}%
  - Office hours utilization: {rng.choice(['low (2-3 students/week)', 'moderate (5-8 students/week)', 'high (10+ students/week)'])}

TIME SLOT: {time_slot} minutes ({rng.choice(['Monday/Wednesday', 'Tuesday/Thursday', 'Monday/Wednesday/Friday'])} {rng.choice(['9:00 AM', '10:30 AM', '1:00 PM', '2:30 PM', '4:00 PM'])})
"""

    problem_statement = f"""# Lesson Plan Design

You are {instructor_name}, an instructor teaching {subject} at the
{level_name} level. You need to create a detailed lesson plan for
Session {session_number} of {total_sessions}.

Review the following documents:
- /testbed/docs/learning_objectives.txt — Specific learning objectives for this session
- /testbed/docs/available_materials.txt — Available materials and resources
- /testbed/docs/class_profile.txt — Class size, level, and student background

Write a detailed lesson plan to /testbed/lesson_plan.txt that includes:

1. **Session Overview**: Subject, session number, time slot ({time_slot} minutes),
   and the {n_objectives} learning objectives

2. **Timed Agenda**: A minute-by-minute (or segment-by-segment) plan that
   accounts for the full {time_slot} minutes. Each segment should specify:
   - Time allocation (e.g., "0:00-0:10")
   - Activity description
   - Which learning objective(s) it addresses
   - Materials needed

3. **Activities**: At least one activity for each learning objective. Include:
   - Activity instructions
   - Expected student actions
   - How to check for understanding

4. **Materials List**: Which materials from the available list you will actually
   use, and how

5. **Assessment**: At least one method to assess whether objectives were met
   (formative assessment during class, not just a future exam)

6. **Differentiation**: How you will accommodate varied learners (students
   who are ahead, students who are struggling, non-native speakers, students
   with accommodations)

7. **Wrap-up**: How you will close the session and preview the next one

The plan should be practical, time-realistic, and responsive to the class
profile information provided."""

    rubric = (
        # --- Structural (binary, low points) ---
        BinaryRubricCategory(
            name="file_exists",
            question="Does /testbed/lesson_plan.txt exist with substantial content (at least 400 words)?",
            points=1,
        ),
        BinaryRubricCategory(
            name="session_info_correct",
            question=f"Does the lesson plan correctly state the subject ({subject}), session number ({session_number}), and time slot ({time_slot} minutes)?",
            points=1,
        ),
        # --- Objectives coverage (binary, medium points) ---
        BinaryRubricCategory(
            name="all_objectives_listed",
            question=f"Does the lesson plan list all {n_objectives} learning objectives from the objectives document?",
            points=2,
        ),
        BinaryRubricCategory(
            name="activity_per_objective",
            question=f"Does the lesson plan include at least one distinct activity for each of the {n_objectives} learning objectives?",
            points=3,
        ),
        # --- Time management (binary, high points) ---
        BinaryRubricCategory(
            name="timed_agenda_present",
            question="Does the lesson plan include a timed agenda with specific time allocations for each segment (e.g., '0:00-0:10: Opening')?",
            points=2,
        ),
        BinaryRubricCategory(
            name="time_adds_up",
            question=f"Do the time allocations in the timed agenda add up to approximately {time_slot} minutes (within 5 minutes)?",
            points=3,
        ),
        # --- Materials (binary) ---
        RubricCategory(
            name="materials_usage",
            description="How well does the lesson plan utilize the available materials?",
            failure="No materials from the available list are referenced, or the plan requires materials not listed as available.",
            minor_failure="One or two materials from the list are mentioned but their use is not described, or the plan heavily relies on unlisted materials.",
            minor_success="At least 3 materials from the available list are referenced with clear descriptions of how they will be used in specific activities.",
            success="Materials are thoughtfully selected from the available list, each tied to a specific activity and learning objective, with clear instructions for how they support the lesson.",
            points=2,
        ),
        # --- Assessment (binary) ---
        BinaryRubricCategory(
            name="assessment_method_included",
            question="Does the lesson plan include at least one formative assessment method (e.g., exit ticket, think-pair-share, quick quiz, muddiest point) to check for understanding during the session?",
            points=2,
        ),
        # --- Differentiation (binary) ---
        BinaryRubricCategory(
            name="differentiation_addressed",
            question="Does the lesson plan include a differentiation section addressing at least 2 of: advanced students, struggling students, non-native speakers, students with accommodations?",
            points=2,
        ),
        # --- Quality / judgment (graded, high points) ---
        RubricCategory(
            name="engagement_quality",
            description="Are the planned activities likely to engage students actively (not just lecture)?",
            failure="Plan is entirely lecture-based with no interactive elements.",
            minor_failure="Plan includes one token interactive element (e.g., 'ask if there are questions') but is predominantly passive.",
            minor_success="Plan includes multiple interactive elements (discussion, group work, or hands-on activities) but they feel formulaic or disconnected from the objectives.",
            success="Plan features well-designed interactive activities that are clearly connected to specific learning objectives, incorporate the class profile (e.g., leveraging what students responded well to), and promote active learning.",
            points=3,
        ),
        RubricCategory(
            name="differentiation_quality",
            description="How thoughtful and practical is the differentiation plan?",
            failure="No differentiation mentioned, or only a generic statement ('adjust as needed').",
            minor_failure="Mentions differentiation but strategies are vague (e.g., 'provide extra help to struggling students').",
            minor_success="Provides specific differentiation strategies for 2+ groups but they feel add-on rather than integrated into the lesson.",
            success="Differentiation is integrated into the lesson design — e.g., tiered activities, scaffolded materials, extension problems for advanced students, visual aids for non-native speakers — and references the class profile data.",
            points=3,
        ),
        # --- Wrap-up and coherence (binary) ---
        BinaryRubricCategory(
            name="wrapup_and_preview",
            question="Does the lesson plan include a wrap-up segment that summarizes key takeaways and previews the next session?",
            points=2,
        ),
    )

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write your lesson plan to /testbed/lesson_plan.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.EDIT_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/docs/learning_objectives.txt": objectives_text,
            "/testbed/docs/available_materials.txt": materials_text,
            "/testbed/docs/class_profile.txt": class_profile_text,
        },
        problem_type="lesson_plan",
    )
