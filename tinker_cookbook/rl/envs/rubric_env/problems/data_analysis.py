"""Data analysis tasks — CSV data analysis and report writing.

All factories are seedable: different seeds produce different data and
different correct answers.
"""

import random as _random

from ..dataset import RubricDatapoint, RubricCategory, BinaryRubricCategory
from tinker_cookbook.rl.envs import tools


def make_data_analysis_report(rand_seed: int = 42) -> RubricDatapoint:
    """Data analysis task — analyze a CSV and write a report."""
    rng = _random.Random(rand_seed)

    csv_lines = ["employee_id,department,salary,years_experience,performance_rating"]
    departments = ["Engineering", "Marketing", "Sales", "HR", "Finance"]
    for i in range(1, 201):
        dept = rng.choice(departments)
        years = rng.randint(1, 25)
        base = {"Engineering": 85000, "Marketing": 65000, "Sales": 60000,
                "HR": 62000, "Finance": 75000}[dept]
        salary = base + years * 2500 + rng.randint(-10000, 10000)
        rating = min(5.0, max(1.0, round(0.5 + years * 0.12 + rng.gauss(0, 0.8), 1)))
        csv_lines.append(f"{i},{dept},{salary},{years},{rating}")

    csv_content = "\n".join(csv_lines) + "\n"

    # --- Compute ground-truth stats for binary rubric checks ---
    dept_salaries: dict[str, list[int]] = {d: [] for d in departments}
    all_salaries: list[int] = []
    for line in csv_lines[1:]:
        parts = line.split(",")
        dept, salary, years = parts[1], int(parts[2]), int(parts[3])
        dept_salaries[dept].append(salary)
        all_salaries.append(salary)

    total_employees = len(all_salaries)
    mean_salary = round(sum(all_salaries) / total_employees)
    sorted_salaries = sorted(all_salaries)
    median_salary = sorted_salaries[total_employees // 2]
    min_salary = min(all_salaries)
    max_salary = max(all_salaries)

    # Find highest-paid department by average
    dept_avgs = {d: round(sum(s) / len(s)) for d, s in dept_salaries.items() if s}
    highest_paid_dept = max(dept_avgs, key=dept_avgs.get)
    highest_paid_avg = dept_avgs[highest_paid_dept]

    # Find department with most employees
    dept_counts = {d: len(s) for d, s in dept_salaries.items()}
    largest_dept = max(dept_counts, key=dept_counts.get)
    largest_dept_count = dept_counts[largest_dept]

    return RubricDatapoint(
        problem_statement="""# Data Analysis: Employee Compensation Report

You have a dataset at /testbed/data.csv containing employee compensation data
with columns: employee_id, department, salary, years_experience, performance_rating.

Analyze this dataset and produce a written report at /testbed/report.txt that includes:

1. Summary statistics: total employees, salary range, mean/median salary
2. Department breakdown: average salary and headcount per department
3. A key finding about the relationship between experience and salary
4. A key finding about the relationship between experience and performance rating
5. One actionable recommendation based on your analysis

You may use Python, bash, or any tools available in the container to analyze the data.
The report should be written in clear, professional prose (not just raw numbers).""",
        rubric=(
            BinaryRubricCategory(
                name="correct_total_employees",
                question=f"Does the report state the total number of employees as {total_employees}?",
                points=1,
            ),
            BinaryRubricCategory(
                name="correct_mean_salary",
                question=f"Does the report state the mean salary as approximately ${mean_salary:,} (within $500)?",
                points=2,
            ),
            BinaryRubricCategory(
                name="correct_median_salary",
                question=f"Does the report state the median salary as approximately ${median_salary:,} (within $500)?",
                points=2,
            ),
            BinaryRubricCategory(
                name="correct_salary_range",
                question=f"Does the report state the salary range (min ~${min_salary:,} to max ~${max_salary:,}, within $1000)?",
                points=1,
            ),
            BinaryRubricCategory(
                name="correct_highest_paid_dept",
                question=f"Does the report identify {highest_paid_dept} as the highest-paid department (avg ~${highest_paid_avg:,})?",
                points=2,
            ),
            BinaryRubricCategory(
                name="dept_breakdown_present",
                question="Does the report include average salary and headcount for each of the 5 departments?",
                points=2,
            ),
            BinaryRubricCategory(
                name="experience_salary_finding",
                question="Does the report describe a positive relationship between years_experience and salary?",
                points=2,
            ),
            BinaryRubricCategory(
                name="experience_rating_finding",
                question="Does the report describe a relationship between years_experience and performance_rating?",
                points=2,
            ),
            BinaryRubricCategory(
                name="has_recommendation",
                question="Does the report include at least one actionable recommendation based on the analysis?",
                points=1,
            ),
            RubricCategory(
                name="statistical_reasoning",
                description="Does the analysis demonstrate sound statistical reasoning?",
                failure="No real analysis; just restates raw numbers without interpretation.",
                minor_failure="Some analysis attempted but with flawed reasoning or inappropriate methods.",
                minor_success="Reasonable analysis with appropriate methods, minor gaps in reasoning.",
                success="Sound statistical reasoning, appropriate use of summary statistics, meaningful correlations identified.",
                points=2,
            ),
            RubricCategory(
                name="presentation_clarity",
                description="Is the report well-written, well-organized, and easy to understand?",
                failure="Unreadable, just raw data dumps, or incoherent prose.",
                minor_failure="Readable but poorly organized or mixing prose with raw output.",
                minor_success="Clear and organized with minor formatting or flow issues.",
                success="Professional, well-structured report with clear sections, good prose, and appropriate use of numbers.",
                points=2,
            ),
        ),
        submission_instructions="Write your completed report to /testbed/report.txt",
        available_tools=(tools.BASH_TOOL, tools.CREATE_FILE_TOOL, tools.EDIT_TOOL, tools.LIST_DIRECTORY_CONTENTS_TOOL, tools.FINISH_TOOL),
        necessary_files={"/testbed/data.csv": csv_content},
        problem_type="data_analysis",
    )


def make_utilization_report(rand_seed: int = 99) -> RubricDatapoint:
    """Employee utilization report from timekeeping data."""
    rng = _random.Random(rand_seed)

    employees = [
        ("E001", "Alice Martin", "Engineering", "Senior Developer", "FT", 160),
        ("E002", "Bob Chen", "Engineering", "Developer", "FT", 160),
        ("E003", "Carla Diaz", "Engineering", "Junior Developer", "PT", 80),
        ("E004", "Dan Foster", "Marketing", "Campaign Manager", "FT", 160),
        ("E005", "Eva Green", "Marketing", "Content Specialist", "FT", 160),
        ("E006", "Frank Hall", "Marketing", "Designer", "PT", 80),
        ("E007", "Grace Ito", "Sales", "Account Executive", "FT", 160),
        ("E008", "Henry Jain", "Sales", "Sales Rep", "FT", 160),
        ("E009", "Irene Kim", "Operations", "Ops Manager", "FT", 160),
        ("E010", "Jack Lee", "Operations", "Logistics Coord", "FT", 160),
        ("E011", "Kara Moss", "Operations", "Warehouse Lead", "PT", 80),
        ("E012", "Leo Nash", "Finance", "Controller", "FT", 160),
        ("E013", "Mia Owens", "Finance", "Accountant", "FT", 160),
        ("E014", "Nick Park", "HR", "HR Manager", "FT", 160),
        ("E015", "Olivia Quinn", "HR", "Recruiter", "PT", 80),
    ]

    projects = [
        ("P100", "Website Redesign", 800),
        ("P200", "Q1 Marketing Campaign", 500),
        ("P300", "Client Onboarding System", 600),
        ("P400", "Annual Audit Prep", 300),
        ("P500", "Warehouse Automation", 450),
        ("P600", "General & Administrative", 9999),
    ]

    timekeeping_lines = ["employee_id,date,project_code,hours"]
    for day in range(1, 21):
        date = f"2024-03-{day:02d}"
        for emp_id, name, dept, role, ft_pt, capacity in employees:
            daily_hours = capacity / 20
            if rng.random() < 0.7:
                proj = rng.choice([p[0] for p in projects[:5]])
                hrs = round(daily_hours + rng.gauss(0, 0.5), 1)
                hrs = max(0, hrs)
                timekeeping_lines.append(f"{emp_id},{date},{proj},{hrs}")
            else:
                proj1 = rng.choice([p[0] for p in projects[:5]])
                proj2 = rng.choice([p[0] for p in projects[:5]])
                hrs1 = round(daily_hours * 0.6 + rng.gauss(0, 0.3), 1)
                hrs2 = round(daily_hours * 0.4 + rng.gauss(0, 0.3), 1)
                hrs1 = max(0, hrs1)
                hrs2 = max(0, hrs2)
                timekeeping_lines.append(f"{emp_id},{date},{proj1},{hrs1}")
                if hrs2 > 0:
                    timekeeping_lines.append(f"{emp_id},{date},{proj2},{hrs2}")

    timekeeping_csv = "\n".join(timekeeping_lines) + "\n"

    roster_lines = ["employee_id,name,department,role,employment_type,monthly_capacity_hours"]
    for emp in employees:
        roster_lines.append(",".join(str(x) for x in emp))
    roster_csv = "\n".join(roster_lines) + "\n"

    budget_lines = ["project_code,project_name,budgeted_hours_q1"]
    for proj in projects:
        budget_lines.append(f"{proj[0]},{proj[1]},{proj[2]}")
    budget_csv = "\n".join(budget_lines) + "\n"

    return RubricDatapoint(
        problem_statement="""# Employee Utilization Report

You are an HR analyst. Using the data files in /testbed/data/, produce
a utilization report for March 2024.

Files available:
- /testbed/data/timekeeping.csv — Daily time entries (employee_id, date, project_code, hours)
- /testbed/data/roster.csv — Employee roster with monthly capacity hours
- /testbed/data/project_budgets.csv — Q1 project budgets

Write a report to /testbed/report.txt that includes:

1. EMPLOYEE UTILIZATION: For each employee, compute total hours worked
   in March and utilization rate (hours_worked / capacity_hours × 100%).
   Flag anyone below 60% (underutilized) or above 110% (overutilized).

2. DEPARTMENT SUMMARY: Average utilization per department.

3. PROJECT HOURS SUMMARY: Total hours charged to each project in March.
   Compare against Q1 budget (note: March is month 3 of 3 in Q1).

4. AT-RISK FLAGS: List specific employees and projects that need attention.

5. RECOMMENDATIONS: 1-2 actionable recommendations.

You may write and run Python scripts to analyze the data.""",
        rubric=(
            BinaryRubricCategory(
                name="file_exists",
                question="Does /testbed/report.txt exist with substantial content?",
                points=1,
            ),
            BinaryRubricCategory(
                name="lists_all_15_employees",
                question="Does the report list utilization data for all 15 employees?",
                points=2,
            ),
            BinaryRubricCategory(
                name="states_capacity_baseline",
                question="Does the report state the capacity baseline (160 hours/month for FT, 80 hours/month for PT)?",
                points=2,
            ),
            BinaryRubricCategory(
                name="computes_utilization_rates",
                question="Does the report show utilization rates as percentages for each employee?",
                points=2,
            ),
            BinaryRubricCategory(
                name="flags_underutilized",
                question="Does the report flag at least one employee as underutilized (below 60%)?",
                points=3,
            ),
            BinaryRubricCategory(
                name="flags_overutilized",
                question="Does the report flag at least one employee as overutilized (above 110%)?",
                points=3,
            ),
            BinaryRubricCategory(
                name="department_summary_present",
                question="Does the report include a department-level average utilization summary?",
                points=2,
            ),
            BinaryRubricCategory(
                name="project_hours_present",
                question="Does the report show total hours charged per project in March?",
                points=2,
            ),
            BinaryRubricCategory(
                name="project_budget_comparison",
                question="Does the report compare project hours against budget (considering March is month 3 of Q1)?",
                points=3,
            ),
            BinaryRubricCategory(
                name="has_recommendations",
                question="Does the report include at least one actionable recommendation?",
                points=2,
            ),
            RubricCategory(
                name="numerical_accuracy",
                description="Are the computed numbers (hours, percentages) accurate based on the raw data?",
                failure="Numbers are fabricated or wildly incorrect (off by >20%)",
                minor_failure="Some numbers are close but several are significantly wrong",
                minor_success="Most numbers are accurate with minor rounding differences",
                success="All reported numbers match the raw data within reasonable rounding",
                points=3,
            ),
            RubricCategory(
                name="analysis_quality",
                description="Does the report go beyond raw numbers to provide insight?",
                failure="Just dumps raw numbers with no interpretation",
                minor_failure="Some interpretation but mostly restates the data",
                minor_success="Good interpretation of trends with minor gaps",
                success="Identifies patterns (e.g., department-level trends, project staffing imbalances) and connects them to actionable insights",
                points=3,
            ),
            RubricCategory(
                name="formatting_quality",
                description="Is the report well-formatted and easy to scan?",
                failure="Raw data dump, no structure",
                minor_failure="Some structure but hard to navigate",
                minor_success="Clear sections and mostly readable",
                success="Well-structured with clear headers, aligned tables or lists, and easy-to-scan formatting",
                points=2,
            ),
        ),
        submission_instructions="Write your utilization report to /testbed/report.txt",
        available_tools=(tools.BASH_TOOL, tools.CREATE_FILE_TOOL, tools.EDIT_TOOL, tools.LIST_DIRECTORY_CONTENTS_TOOL, tools.FINISH_TOOL),
        necessary_files={
            "/testbed/data/timekeeping.csv": timekeeping_csv,
            "/testbed/data/roster.csv": roster_csv,
            "/testbed/data/project_budgets.csv": budget_csv,
        },
        problem_type="data_analysis",
    )


def make_sales_yoy_analysis(rand_seed: int = 55) -> RubricDatapoint:
    """Sales Year-over-Year analysis from CSV data."""
    rng = _random.Random(rand_seed)
    brand_name = rng.choice(["Northridge Outdoor Co.", "Summit Gear", "Trailblazer Sports",
                             "Alpine Edge", "Basecamp Supply"])

    product_lines = ["Hiking Boots", "Rain Jackets", "Backpacks", "Camping Tents", "Sleeping Bags"]

    def gen_sales(year, rng):
        lines = ["product_line,month,units,revenue"]
        for pl in product_lines:
            base_units = {"Hiking Boots": 400, "Rain Jackets": 300, "Backpacks": 350,
                          "Camping Tents": 200, "Sleeping Bags": 150}[pl]
            unit_price = {"Hiking Boots": 129, "Rain Jackets": 89, "Backpacks": 79,
                          "Camping Tents": 249, "Sleeping Bags": 119}[pl]
            seasonal = {"Hiking Boots": [0.6, 0.7, 1.0, 1.2, 1.5, 1.4, 1.3, 1.2, 1.1, 0.9, 0.7, 0.8],
                        "Rain Jackets": [0.8, 0.9, 1.2, 1.4, 1.0, 0.8, 0.7, 0.8, 1.1, 1.3, 1.2, 0.9],
                        "Backpacks": [0.7, 0.8, 1.0, 1.1, 1.3, 1.5, 1.4, 1.3, 1.0, 0.8, 0.7, 0.9],
                        "Camping Tents": [0.4, 0.5, 0.8, 1.2, 1.6, 1.8, 1.7, 1.5, 1.0, 0.6, 0.3, 0.3],
                        "Sleeping Bags": [0.5, 0.6, 0.9, 1.2, 1.4, 1.5, 1.4, 1.3, 1.1, 0.8, 0.6, 0.7]}[pl]
            growth = 1.0 if year == 2022 else {"Hiking Boots": 1.15, "Rain Jackets": 1.08,
                                                "Backpacks": 0.92, "Camping Tents": 1.22,
                                                "Sleeping Bags": 0.85}[pl]
            for month in range(1, 13):
                units = int(base_units * seasonal[month-1] * growth + rng.randint(-30, 30))
                units = max(10, units)
                rev = units * unit_price + rng.randint(-500, 500)
                lines.append(f"{pl},{year}-{month:02d},{units},{rev}")
        return "\n".join(lines) + "\n"

    sales_2022 = gen_sales(2022, rng)
    sales_2023 = gen_sales(2023, rng)

    inv_lines = ["product_line,on_hand_units,on_order_units,avg_monthly_demand_units"]
    for pl in product_lines:
        on_hand = rng.randint(200, 800)
        on_order = rng.randint(100, 500)
        avg_demand = {"Hiking Boots": 450, "Rain Jackets": 320, "Backpacks": 300,
                      "Camping Tents": 280, "Sleeping Bags": 140}[pl]
        inv_lines.append(f"{pl},{on_hand},{on_order},{avg_demand}")
    inventory_csv = "\n".join(inv_lines) + "\n"

    return RubricDatapoint(
        problem_statement=f"""# Sales Year-over-Year Analysis: {brand_name}

You are a business analyst for {brand_name}. Using the sales data files
in /testbed/data/, produce a year-over-year analysis comparing 2023 vs 2022.

Files available:
- /testbed/data/sales_2022.csv — Monthly sales by product line (2022)
- /testbed/data/sales_2023.csv — Monthly sales by product line (2023)
- /testbed/data/inventory.csv — Current inventory levels

Write a report to /testbed/analysis.txt that includes:

1. OVERALL YoY COMPARISON: Total revenue 2023 vs 2022, overall growth %
2. PRODUCT LINE BREAKDOWN: Revenue and growth % for each product line
3. TOP PERFORMER: Which product line grew the most (%) and why it might be
4. DECLINING LINE: Which product line declined and potential concerns
5. INVENTORY RISK: Flag any product where projected demand > available
   inventory (on_hand + on_order) for the next 2 months
6. RECOMMENDATIONS: 2-3 specific recommendations

You may write and run Python scripts to analyze the data.""",
        rubric=(
            BinaryRubricCategory(
                name="file_exists",
                question="Does /testbed/analysis.txt exist with substantial content?",
                points=1,
            ),
            BinaryRubricCategory(
                name="correct_2023_total",
                question="Does the report state a 2023 total revenue figure that is within 2% of the actual sum from the CSV?",
                points=3,
            ),
            BinaryRubricCategory(
                name="correct_2022_total",
                question="Does the report state a 2022 total revenue figure that is within 2% of the actual sum from the CSV?",
                points=3,
            ),
            BinaryRubricCategory(
                name="correct_overall_yoy_pct",
                question="Does the report state an overall YoY growth percentage consistent with its stated revenue figures?",
                points=3,
            ),
            BinaryRubricCategory(
                name="identifies_top_growing",
                question="Does the report identify Camping Tents as the top-growing product line (by % growth)?",
                points=3,
            ),
            BinaryRubricCategory(
                name="identifies_declining",
                question="Does the report identify Sleeping Bags as a declining product line?",
                points=3,
            ),
            BinaryRubricCategory(
                name="per_product_breakdown",
                question="Does the report show revenue and/or growth % for each of the 5 product lines?",
                points=2,
            ),
            BinaryRubricCategory(
                name="flags_inventory_risk",
                question="Does the report flag at least one product line where projected 2-month demand exceeds available inventory?",
                points=3,
            ),
            BinaryRubricCategory(
                name="has_recommendations",
                question="Does the report include at least 2 specific recommendations?",
                points=2,
            ),
            RubricCategory(
                name="analysis_depth",
                description="Does the report go beyond just stating numbers to provide actual insight?",
                failure="Just lists numbers with no interpretation",
                minor_failure="Some interpretation but mostly restates the data differently",
                minor_success="Identifies trends and offers plausible explanations",
                success="Identifies trends, offers explanations (e.g., seasonal patterns, market factors), and connects them to actionable recommendations",
                points=3,
            ),
            RubricCategory(
                name="numerical_accuracy",
                description="Are the computed numbers accurate based on the raw CSV data?",
                failure="Numbers are fabricated or wildly incorrect",
                minor_failure="Some numbers are close but several are wrong",
                minor_success="Most numbers are accurate with minor rounding issues",
                success="All reported numbers are accurate and verifiable from the CSVs",
                points=3,
            ),
            RubricCategory(
                name="formatting_quality",
                description="Is the report well-formatted and professional?",
                failure="Raw data dump or unreadable",
                minor_failure="Some structure but hard to navigate",
                minor_success="Clear sections, mostly readable",
                success="Professional formatting with clear headers, tables or aligned data, and readable prose sections",
                points=2,
            ),
        ),
        submission_instructions="Write your analysis report to /testbed/analysis.txt",
        available_tools=(tools.BASH_TOOL, tools.CREATE_FILE_TOOL, tools.EDIT_TOOL, tools.LIST_DIRECTORY_CONTENTS_TOOL, tools.FINISH_TOOL),
        necessary_files={
            "/testbed/data/sales_2022.csv": sales_2022,
            "/testbed/data/sales_2023.csv": sales_2023,
            "/testbed/data/inventory.csv": inventory_csv,
        },
        problem_type="data_analysis",
    )
