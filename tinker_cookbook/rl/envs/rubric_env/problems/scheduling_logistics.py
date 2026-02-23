"""Scheduling and logistics tasks: shift scheduling audit, supply chain optimization,
and delivery route planning.

Each factory generates a realistic scenario with necessary_files containing all
information required to solve the problem. Seeds control randomization of employee
counts, demand levels, SKU mixes, delivery stops, constraint tightness, and which
violations or issues are planted.
"""

from __future__ import annotations

import math
import random as _random

from ..dataset import RubricDatapoint, RubricCategory, BinaryRubricCategory
from tinker_cookbook.rl.envs import tools
from ..content_pools import (
    random_name,
    random_names,
    pick1,
    COMPANY_NAMES,
)


# =============================================================================
# HELPERS
# =============================================================================


def _fmt_money(amount: float) -> str:
    """Format a float as a dollar string with commas."""
    return f"${amount:,.2f}"


# =============================================================================
# DOMAIN: SHIFT SCHEDULING
# =============================================================================

ROLES = ["cashier", "stocker", "supervisor"]

CERTIFICATIONS_POOL = [
    "forklift", "food_safety", "first_aid", "hazmat", "alcohol_sales",
]

SHIFT_NAMES = ["morning", "afternoon", "night"]
SHIFT_WINDOWS = {
    "morning": ("6:00 AM", "2:00 PM"),
    "afternoon": ("2:00 PM", "10:00 PM"),
    "night": ("10:00 PM", "6:00 AM"),
}
SHIFT_HOURS = 8  # all standard shifts are 8 hours

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Night shift differential
NIGHT_DIFFERENTIAL = 2.00  # $/hr

BASE_PAY_RATES = {
    "cashier": 14.00,
    "stocker": 13.00,
    "supervisor": 20.00,
}

OVERTIME_MULTIPLIER = 1.5


def make_shift_scheduling(rand_seed: int = 42) -> RubricDatapoint:
    """Given employee availability, labor regulations, demand forecasts, and
    pay rates, audit a proposed weekly shift schedule for violations.

    Seed varies: number of employees (12-18), availability patterns, demand
    levels, which days have peak demand, which violations are planted (3-5
    of: unavailable-time, >8hr shift, <11hr gap, missing supervisor,
    exceeds-40hr, uncertified assignment).
    """
    rng = _random.Random(rand_seed)

    company = pick1(COMPANY_NAMES, rand_seed)
    store_number = rng.randint(100, 999)
    manager_name = random_name(rand_seed + 1)

    # --- Generate employees ---
    n_employees = rng.randint(12, 18)
    emp_names = random_names(rand_seed + 10, n_employees)

    employees: list[dict] = []
    for i, name in enumerate(emp_names):
        role = rng.choice(ROLES) if i >= 3 else ROLES[i % 3]  # ensure at least 1 of each
        max_hours = rng.choice([32, 40])
        n_certs = rng.randint(0, 2)
        certs = rng.sample(CERTIFICATIONS_POOL, n_certs) if n_certs > 0 else []
        # Availability: each employee available for some shifts on some days
        availability: dict[str, list[str]] = {}
        for day in DAYS_OF_WEEK:
            if rng.random() < 0.15:
                # Completely unavailable this day
                continue
            avail_shifts = []
            for shift in SHIFT_NAMES:
                if rng.random() < 0.75:
                    avail_shifts.append(shift)
            if avail_shifts:
                availability[day] = avail_shifts
        employees.append({
            "name": name,
            "role": role,
            "max_hours": max_hours,
            "certifications": certs,
            "availability": availability,
            "emp_id": f"E{100 + i}",
        })

    # --- Generate demand forecast ---
    # Higher demand on weekends if seed is even, weekdays if odd
    peak_weekend = (rand_seed % 2 == 0)
    demand: dict[str, dict[str, dict[str, int]]] = {}
    for day in DAYS_OF_WEEK:
        is_weekend = day in ("Saturday", "Sunday")
        peak = (is_weekend and peak_weekend) or (not is_weekend and not peak_weekend)
        demand[day] = {}
        for shift in SHIFT_NAMES:
            base_cashiers = rng.randint(3, 5) if peak else rng.randint(2, 3)
            base_stockers = rng.randint(2, 3) if peak else rng.randint(1, 2)
            base_supervisors = 1  # always need at least 1
            if shift == "night":
                base_cashiers = max(1, base_cashiers - 1)
                base_stockers = max(1, base_stockers)
            demand[day][shift] = {
                "cashier": base_cashiers,
                "stocker": base_stockers,
                "supervisor": base_supervisors,
            }

    # --- Build a proposed schedule with planted violations ---
    # Start with a valid-ish schedule, then inject violations
    schedule: dict[str, dict[str, list[str]]] = {}  # day -> shift -> list of emp_ids
    emp_weekly_hours: dict[str, int] = {e["emp_id"]: 0 for e in employees}
    emp_last_shift_end: dict[str, tuple[int, int]] = {}  # emp_id -> (day_idx, shift_end_hour_24)

    shift_end_hours = {"morning": 14, "afternoon": 22, "night": 6}
    shift_start_hours = {"morning": 6, "afternoon": 14, "night": 22}

    for day_idx, day in enumerate(DAYS_OF_WEEK):
        schedule[day] = {}
        for shift in SHIFT_NAMES:
            needed = demand[day][shift]
            assigned: list[str] = []
            # Try to fill from available employees of the right role
            for role_needed, count in needed.items():
                candidates = [
                    e for e in employees
                    if e["role"] == role_needed
                    and day in e["availability"]
                    and shift in e["availability"].get(day, [])
                    and emp_weekly_hours[e["emp_id"]] + SHIFT_HOURS <= e["max_hours"]
                    and e["emp_id"] not in assigned
                ]
                rng.shuffle(candidates)
                for c in candidates[:count]:
                    assigned.append(c["emp_id"])
                    emp_weekly_hours[c["emp_id"]] += SHIFT_HOURS
                    emp_last_shift_end[c["emp_id"]] = (day_idx, shift_end_hours[shift])
            schedule[day][shift] = assigned

    # --- Plant violations ---
    VIOLATION_TYPES = [
        "unavailable_time",
        "exceeds_8hr_shift",
        "insufficient_gap",
        "missing_supervisor",
        "exceeds_40hr_week",
    ]
    n_violations = rng.randint(3, 5)
    chosen_violation_types = rng.sample(VIOLATION_TYPES, min(n_violations, len(VIOLATION_TYPES)))

    violations_planted: list[dict] = []
    used_emp_for_violations: set[str] = set()

    for vtype in chosen_violation_types:
        if vtype == "unavailable_time":
            # Find an employee scheduled during a time they're unavailable
            # Or force one: pick an employee, pick a day/shift they're NOT available, add them
            for emp in rng.sample(employees, len(employees)):
                if emp["emp_id"] in used_emp_for_violations:
                    continue
                unavail_slots = []
                for day in DAYS_OF_WEEK:
                    for shift in SHIFT_NAMES:
                        if day not in emp["availability"] or shift not in emp["availability"].get(day, []):
                            if schedule[day][shift]:  # has existing assignments
                                unavail_slots.append((day, shift))
                if unavail_slots:
                    day, shift = rng.choice(unavail_slots)
                    schedule[day][shift].append(emp["emp_id"])
                    used_emp_for_violations.add(emp["emp_id"])
                    violations_planted.append({
                        "type": "unavailable_time",
                        "employee": emp["name"],
                        "emp_id": emp["emp_id"],
                        "day": day,
                        "shift": shift,
                        "detail": (
                            f"{emp['name']} ({emp['emp_id']}) is scheduled for the {shift} shift "
                            f"on {day}, but is not available during that time."
                        ),
                    })
                    break

        elif vtype == "exceeds_8hr_shift":
            # Simulate by scheduling someone for two consecutive shifts on the same day
            # (effectively a 16-hour shift, which exceeds the 8hr max)
            for emp in rng.sample(employees, len(employees)):
                if emp["emp_id"] in used_emp_for_violations:
                    continue
                for day in DAYS_OF_WEEK:
                    shifts_assigned = [
                        s for s in SHIFT_NAMES if emp["emp_id"] in schedule[day][s]
                    ]
                    if len(shifts_assigned) == 1:
                        current = shifts_assigned[0]
                        # Add to the adjacent shift
                        if current == "morning":
                            adjacent = "afternoon"
                        elif current == "afternoon":
                            adjacent = "night"
                        else:
                            adjacent = "morning"
                        schedule[day][adjacent].append(emp["emp_id"])
                        used_emp_for_violations.add(emp["emp_id"])
                        violations_planted.append({
                            "type": "exceeds_8hr_shift",
                            "employee": emp["name"],
                            "emp_id": emp["emp_id"],
                            "day": day,
                            "shifts": sorted([current, adjacent]),
                            "detail": (
                                f"{emp['name']} ({emp['emp_id']}) is scheduled for both the "
                                f"{current} and {adjacent} shifts on {day}, totaling 16 hours "
                                f"which exceeds the 8-hour maximum shift length."
                            ),
                        })
                        break
                else:
                    continue
                break

        elif vtype == "insufficient_gap":
            # Schedule someone for the night shift on one day and morning shift the next day
            # Night ends at 6 AM, morning starts at 6 AM -> 0 hours gap (< 11)
            for emp in rng.sample(employees, len(employees)):
                if emp["emp_id"] in used_emp_for_violations:
                    continue
                for day_idx in range(len(DAYS_OF_WEEK) - 1):
                    day1 = DAYS_OF_WEEK[day_idx]
                    day2 = DAYS_OF_WEEK[day_idx + 1]
                    in_night = emp["emp_id"] in schedule[day1]["night"]
                    in_morning_next = emp["emp_id"] in schedule[day2]["morning"]
                    if not in_night and not in_morning_next:
                        # Plant both
                        if emp["emp_id"] not in schedule[day1]["night"]:
                            schedule[day1]["night"].append(emp["emp_id"])
                        if emp["emp_id"] not in schedule[day2]["morning"]:
                            schedule[day2]["morning"].append(emp["emp_id"])
                        used_emp_for_violations.add(emp["emp_id"])
                        violations_planted.append({
                            "type": "insufficient_gap",
                            "employee": emp["name"],
                            "emp_id": emp["emp_id"],
                            "day1": day1,
                            "day2": day2,
                            "detail": (
                                f"{emp['name']} ({emp['emp_id']}) is scheduled for the night "
                                f"shift on {day1} (ends 6:00 AM {day2}) and the morning shift "
                                f"on {day2} (starts 6:00 AM). This is a 0-hour gap, violating "
                                f"the minimum 11-hour rest requirement."
                            ),
                        })
                        break
                else:
                    continue
                break

        elif vtype == "missing_supervisor":
            # Find a shift that has a supervisor and remove them
            for day in rng.sample(DAYS_OF_WEEK, len(DAYS_OF_WEEK)):
                for shift in rng.sample(SHIFT_NAMES, len(SHIFT_NAMES)):
                    supervisors_on = [
                        eid for eid in schedule[day][shift]
                        if any(e["emp_id"] == eid and e["role"] == "supervisor" for e in employees)
                    ]
                    if supervisors_on:
                        removed = supervisors_on[0]
                        schedule[day][shift].remove(removed)
                        removed_emp = next(e for e in employees if e["emp_id"] == removed)
                        violations_planted.append({
                            "type": "missing_supervisor",
                            "day": day,
                            "shift": shift,
                            "detail": (
                                f"The {shift} shift on {day} has no supervisor assigned. "
                                f"Labor rules require at least 1 supervisor per shift."
                            ),
                        })
                        break
                else:
                    continue
                break

        elif vtype == "exceeds_40hr_week":
            # Find an employee with max_hours=40 and push them over
            for emp in rng.sample(employees, len(employees)):
                if emp["emp_id"] in used_emp_for_violations:
                    continue
                if emp["max_hours"] != 40:
                    continue
                # Count current scheduled hours
                current_hours = 0
                for day in DAYS_OF_WEEK:
                    for shift in SHIFT_NAMES:
                        if emp["emp_id"] in schedule[day][shift]:
                            current_hours += SHIFT_HOURS
                if current_hours >= 32:
                    # Add enough shifts to exceed 40
                    needed_extra = (40 - current_hours) + 8  # push to 48
                    added = 0
                    for day in DAYS_OF_WEEK:
                        for shift in SHIFT_NAMES:
                            if emp["emp_id"] not in schedule[day][shift] and added < needed_extra:
                                schedule[day][shift].append(emp["emp_id"])
                                added += SHIFT_HOURS
                    total_hours = current_hours + added
                    overtime_hours = max(0, total_hours - 40)
                    used_emp_for_violations.add(emp["emp_id"])
                    violations_planted.append({
                        "type": "exceeds_40hr_week",
                        "employee": emp["name"],
                        "emp_id": emp["emp_id"],
                        "total_hours": total_hours,
                        "overtime_hours": overtime_hours,
                        "detail": (
                            f"{emp['name']} ({emp['emp_id']}) is scheduled for {total_hours} "
                            f"hours this week, exceeding the 40-hour maximum. "
                            f"{overtime_hours} hours of overtime at 1.5x rate."
                        ),
                    })
                    break

    # --- Post-hoc audit: count ALL actual violations in the final schedule ---
    # Violation planting can create collateral violations (e.g. adding a shift for
    # exceeds_40hr may land on an unavailable slot, or exceeds_8hr may push hours
    # over 40).  We do a full sweep so the rubric reflects reality.
    all_violations_found: list[dict] = []

    # Build a lookup for employee by emp_id
    emp_by_id: dict[str, dict] = {e["emp_id"]: e for e in employees}

    # Recount scheduled hours per employee from final schedule
    audit_hours: dict[str, int] = {e["emp_id"]: 0 for e in employees}
    for day in DAYS_OF_WEEK:
        for shift in SHIFT_NAMES:
            for eid in schedule[day][shift]:
                audit_hours[eid] += SHIFT_HOURS

    # 1. unavailable_time
    for day in DAYS_OF_WEEK:
        for shift in SHIFT_NAMES:
            for eid in schedule[day][shift]:
                emp = emp_by_id[eid]
                if day not in emp["availability"] or shift not in emp["availability"].get(day, []):
                    all_violations_found.append({
                        "type": "unavailable_time",
                        "emp_id": eid,
                        "day": day,
                        "shift": shift,
                    })

    # 2. exceeds_8hr (two+ shifts on the same day)
    for day in DAYS_OF_WEEK:
        seen: dict[str, list[str]] = {}
        for shift in SHIFT_NAMES:
            for eid in schedule[day][shift]:
                seen.setdefault(eid, []).append(shift)
        for eid, shifts in seen.items():
            if len(shifts) >= 2:
                all_violations_found.append({
                    "type": "exceeds_8hr",
                    "emp_id": eid,
                    "day": day,
                    "shifts": shifts,
                })

    # 3. insufficient_gap (night shift day N -> morning shift day N+1)
    for day_idx in range(len(DAYS_OF_WEEK) - 1):
        day1 = DAYS_OF_WEEK[day_idx]
        day2 = DAYS_OF_WEEK[day_idx + 1]
        night_eids = set(schedule[day1]["night"])
        morning_eids = set(schedule[day2]["morning"])
        for eid in night_eids & morning_eids:
            all_violations_found.append({
                "type": "insufficient_gap",
                "emp_id": eid,
                "day1": day1,
                "day2": day2,
            })

    # 4. exceeds_40hr
    for emp in employees:
        if audit_hours[emp["emp_id"]] > 40:
            all_violations_found.append({
                "type": "exceeds_40hr",
                "emp_id": emp["emp_id"],
                "total_hours": audit_hours[emp["emp_id"]],
            })

    # 5. missing_supervisor
    for day in DAYS_OF_WEEK:
        for shift in SHIFT_NAMES:
            has_supervisor = any(
                emp_by_id[eid]["role"] == "supervisor"
                for eid in schedule[day][shift]
                if eid in emp_by_id
            )
            # Only flag if there is supposed to be someone on the shift (demand > 0)
            total_demand = sum(demand[day][shift][r] for r in ROLES)
            if total_demand > 0 and not has_supervisor:
                all_violations_found.append({
                    "type": "missing_supervisor",
                    "day": day,
                    "shift": shift,
                })

    actual_violation_count = len(all_violations_found)

    # --- Compute ground-truth labor costs ---
    # Recount actual scheduled hours per employee from the (now modified) schedule
    actual_hours: dict[str, int] = {e["emp_id"]: 0 for e in employees}
    actual_night_hours: dict[str, int] = {e["emp_id"]: 0 for e in employees}
    for day in DAYS_OF_WEEK:
        for shift in SHIFT_NAMES:
            for eid in schedule[day][shift]:
                actual_hours[eid] += SHIFT_HOURS
                if shift == "night":
                    actual_night_hours[eid] += SHIFT_HOURS

    total_regular_cost = 0.0
    total_overtime_cost = 0.0
    total_night_diff_cost = 0.0
    for emp in employees:
        eid = emp["emp_id"]
        hours = actual_hours[eid]
        rate = BASE_PAY_RATES[emp["role"]]
        regular_hours = min(hours, 40)
        overtime_hours = max(0, hours - 40)
        regular_cost = regular_hours * rate
        overtime_cost = overtime_hours * rate * OVERTIME_MULTIPLIER
        night_diff = actual_night_hours[eid] * NIGHT_DIFFERENTIAL
        total_regular_cost += regular_cost
        total_overtime_cost += overtime_cost
        total_night_diff_cost += night_diff

    total_labor_cost = round(total_regular_cost + total_overtime_cost + total_night_diff_cost, 2)
    total_overtime_cost = round(total_overtime_cost, 2)
    total_night_diff_cost = round(total_night_diff_cost, 2)

    # Count total scheduled shifts
    total_scheduled_shifts = sum(
        len(schedule[day][shift]) for day in DAYS_OF_WEEK for shift in SHIFT_NAMES
    )

    # Count total night shift person-hours
    total_night_shift_hours = sum(actual_night_hours.values())

    # Identify demand shortfalls: shifts where assigned count < demand
    demand_shortfalls: list[dict] = []
    for day in DAYS_OF_WEEK:
        for shift in SHIFT_NAMES:
            assigned_ids = schedule[day][shift]
            for role in ROLES:
                needed = demand[day][shift][role]
                have = sum(
                    1 for eid in assigned_ids
                    if any(e["emp_id"] == eid and e["role"] == role for e in employees)
                )
                if have < needed:
                    demand_shortfalls.append({
                        "day": day,
                        "shift": shift,
                        "role": role,
                        "needed": needed,
                        "have": have,
                        "short": needed - have,
                    })

    # Identify top-3 busiest employees (most scheduled hours)
    sorted_by_hours = sorted(employees, key=lambda e: actual_hours[e["emp_id"]], reverse=True)
    top_busy_employees = sorted_by_hours[:3]

    # Count employees with overtime
    employees_with_overtime = [
        e for e in employees if actual_hours[e["emp_id"]] > 40
    ]

    # --- Build employees.txt ---
    emp_lines = [
        f"{company} — STORE #{store_number} EMPLOYEE ROSTER",
        "",
        "=" * 80,
    ]
    for emp in employees:
        emp_lines.append("")
        emp_lines.append(f"Employee ID: {emp['emp_id']}")
        emp_lines.append(f"Name: {emp['name']}")
        emp_lines.append(f"Role: {emp['role']}")
        emp_lines.append(f"Max Weekly Hours: {emp['max_hours']}")
        certs_str = ", ".join(emp["certifications"]) if emp["certifications"] else "None"
        emp_lines.append(f"Certifications: {certs_str}")
        emp_lines.append(f"Availability:")
        for day in DAYS_OF_WEEK:
            if day in emp["availability"]:
                shifts_str = ", ".join(emp["availability"][day])
                emp_lines.append(f"  {day}: {shifts_str}")
            else:
                emp_lines.append(f"  {day}: UNAVAILABLE")
        emp_lines.append("-" * 40)
    emp_lines.append("")
    emp_content = "\n".join(emp_lines) + "\n"

    # --- Build demand_forecast.txt ---
    demand_lines = [
        f"{company} — STORE #{store_number} WEEKLY STAFFING DEMAND FORECAST",
        "",
        "Shift times: Morning 6:00 AM - 2:00 PM | Afternoon 2:00 PM - 10:00 PM | Night 10:00 PM - 6:00 AM",
        "",
        "=" * 70,
    ]
    for day in DAYS_OF_WEEK:
        demand_lines.append("")
        demand_lines.append(f"{day}:")
        for shift in SHIFT_NAMES:
            d = demand[day][shift]
            demand_lines.append(
                f"  {shift.capitalize():>12}: {d['cashier']} cashier(s), "
                f"{d['stocker']} stocker(s), {d['supervisor']} supervisor(s)"
            )
    demand_lines.append("")
    demand_content = "\n".join(demand_lines) + "\n"

    # --- Build labor_rules.txt ---
    labor_lines = [
        "LABOR REGULATIONS AND SCHEDULING RULES",
        "",
        "=" * 60,
        "",
        "1. SHIFT LENGTH: Maximum shift length is 8 hours. No employee may be",
        "   scheduled for more than 8 consecutive hours in a single shift or",
        "   for two shifts on the same day (which would total 16 hours).",
        "",
        "2. REST BETWEEN SHIFTS: Minimum 11 hours must elapse between the end",
        "   of one shift and the start of the next shift for the same employee.",
        "",
        "3. CONSECUTIVE DAYS: Maximum 5 consecutive days of work. At least 2",
        "   days off per 7-day period.",
        "",
        "4. WEEKLY HOURS: Maximum 40 hours per week for non-exempt employees.",
        "   Hours exceeding 40 per week incur overtime at 1.5x the base rate.",
        "",
        "5. SUPERVISOR REQUIREMENT: At least 1 supervisor must be assigned to",
        "   every shift, regardless of day or time.",
        "",
        "6. BREAK RULES: Employees working 6+ hours receive one 30-minute",
        "   unpaid break and two 15-minute paid breaks.",
        "",
        "7. AVAILABILITY: Employees may only be scheduled during their declared",
        "   availability windows. Scheduling an employee outside their stated",
        "   availability is a violation.",
        "",
        "8. CERTIFICATIONS: Employees handling alcohol sales must have the",
        "   alcohol_sales certification. Forklift operation requires the",
        "   forklift certification.",
        "",
    ]
    labor_content = "\n".join(labor_lines) + "\n"

    # --- Build pay_rates.txt ---
    pay_lines = [
        "PAY RATES — EFFECTIVE 2024",
        "",
        "=" * 50,
        "BASE HOURLY RATES BY ROLE",
        "=" * 50,
        "",
        f"{'Role':<20} {'Base Rate':>12}",
        f"{'-'*20} {'-'*12}",
    ]
    for role, rate in BASE_PAY_RATES.items():
        pay_lines.append(f"{role.capitalize():<20} {_fmt_money(rate):>12}/hr")
    pay_lines.extend([
        "",
        "=" * 50,
        "SHIFT DIFFERENTIALS",
        "=" * 50,
        "",
        f"Night shift (10:00 PM - 6:00 AM): +{_fmt_money(NIGHT_DIFFERENTIAL)}/hr",
        f"Weekend shifts: No additional differential",
        "",
        "=" * 50,
        "OVERTIME",
        "=" * 50,
        "",
        f"Overtime rate: {OVERTIME_MULTIPLIER}x base rate for hours exceeding 40/week",
        f"Overtime is calculated on base rate only (shift differentials excluded)",
        "",
    ])
    pay_content = "\n".join(pay_lines) + "\n"

    # --- Build proposed_schedule.txt ---
    sched_lines = [
        f"PROPOSED WEEKLY SCHEDULE — STORE #{store_number}",
        f"Prepared by: {manager_name}",
        f"Week of: 2024-03-04 to 2024-03-10",
        "",
        "=" * 80,
    ]
    for day in DAYS_OF_WEEK:
        sched_lines.append("")
        sched_lines.append(f"--- {day.upper()} ---")
        for shift in SHIFT_NAMES:
            start, end = SHIFT_WINDOWS[shift]
            assigned_ids = schedule[day][shift]
            sched_lines.append(f"  {shift.capitalize()} ({start} - {end}):")
            if assigned_ids:
                for eid in assigned_ids:
                    emp = next(e for e in employees if e["emp_id"] == eid)
                    sched_lines.append(f"    {eid} — {emp['name']} ({emp['role']})")
            else:
                sched_lines.append(f"    (no staff assigned)")
    sched_lines.append("")
    sched_lines.append("=" * 80)
    sched_lines.append("")
    sched_content = "\n".join(sched_lines) + "\n"

    # --- Problem statement ---
    problem_statement = f"""# Shift Schedule Audit

You are a labor compliance analyst for {company}, Store #{store_number}. The store
manager has submitted a proposed weekly schedule. Your job is to audit the schedule
against the employee roster, demand forecast, labor regulations, and pay rates.

## Source Files
- /testbed/data/employees.txt — Employee roster with roles, availability, max hours, certifications
- /testbed/data/demand_forecast.txt — Required staffing levels per shift per day
- /testbed/data/labor_rules.txt — Labor regulations (max shift, rest periods, overtime, supervisor requirements)
- /testbed/data/pay_rates.txt — Base pay rates, shift differentials, overtime rules
- /testbed/data/proposed_schedule.txt — The schedule to audit

## Requirements
1. Cross-reference each assignment against the employee's declared availability
2. Check for shifts exceeding 8 hours (or double-shift assignments on the same day)
3. Verify minimum 11-hour gaps between consecutive shifts for each employee
4. Confirm at least 1 supervisor on every shift
5. Check each employee's total weekly hours against their maximum
6. Identify any overtime (hours over 40) and compute the overtime cost
7. Compute the total weekly labor cost including base pay, overtime, and night differentials
8. Flag all violations found with specific employee IDs, days, and shifts

Write a detailed audit report to /testbed/audit_report.txt."""

    # --- Build rubric ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does the file /testbed/audit_report.txt exist and contain substantive content (more than a few lines)?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_employee_count",
            question=f"Does the report correctly identify or work with {n_employees} employees from the roster?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_total_scheduled_shifts",
            question=f"Does the report correctly state or compute {total_scheduled_shifts} total person-shifts in the proposed schedule?",
            points=1,
        ),
    ]

    # Per-violation checks
    for i, viol in enumerate(violations_planted):
        rubric_items.append(
            BinaryRubricCategory(
                name=f"found_violation_{i+1}",
                question=f"Does the report identify this violation: {viol['detail']}",
                points=2,
            )
        )

    rubric_items.append(
        BinaryRubricCategory(
            name="correct_violation_count",
            question=f"Does the report identify exactly {actual_violation_count} scheduling violations (not counting staffing shortfalls against demand)?",
            points=2,
        )
    )

    # Pick 2 clean shifts to verify no false positives
    clean_day_shifts: list[tuple[str, str]] = []
    for day in DAYS_OF_WEEK:
        for shift in SHIFT_NAMES:
            # A shift is "clean" if none of its employees are in a violation for this day/shift
            violation_pairs = set()
            for v in violations_planted:
                if v["type"] == "unavailable_time":
                    violation_pairs.add((v["day"], v["shift"]))
                elif v["type"] == "exceeds_8hr_shift":
                    for s in v.get("shifts", []):
                        violation_pairs.add((v["day"], s))
                elif v["type"] == "missing_supervisor":
                    violation_pairs.add((v["day"], v["shift"]))
                elif v["type"] == "insufficient_gap":
                    violation_pairs.add((v["day1"], "night"))
                    violation_pairs.add((v["day2"], "morning"))
            if (day, shift) not in violation_pairs and schedule[day][shift]:
                # Also check no employee on this shift is in an exceeds_40hr violation
                shift_eids = set(schedule[day][shift])
                overtime_eids = {v["emp_id"] for v in violations_planted if v["type"] == "exceeds_40hr_week"}
                if not shift_eids.intersection(overtime_eids):
                    clean_day_shifts.append((day, shift))

    false_pos_checks = rng.sample(clean_day_shifts, min(2, len(clean_day_shifts)))
    for day, shift in false_pos_checks:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"no_false_violation_{day}_{shift}",
                question=(
                    f"Does the report correctly show the {shift} shift on {day} as having "
                    f"no violations? (It should NOT be flagged as problematic.)"
                ),
                points=1,
            )
        )

    # Per-employee hours checks for top busiest employees
    for emp in top_busy_employees:
        eid = emp["emp_id"]
        hrs = actual_hours[eid]
        rubric_items.append(
            BinaryRubricCategory(
                name=f"correct_hours_{eid}",
                question=(
                    f"Does the report correctly identify {emp['name']} ({eid}) as being "
                    f"scheduled for {hrs} hours this week?"
                ),
                points=1,
            )
        )

    # Night shift hours check
    rubric_items.append(
        BinaryRubricCategory(
            name="correct_total_night_hours",
            question=(
                f"Does the report correctly identify or compute {total_night_shift_hours} "
                f"total night shift person-hours across all employees for the week?"
            ),
            points=1,
        )
    )

    # Demand shortfall check (if any exist)
    if demand_shortfalls:
        shortfall_desc = "; ".join(
            f"{sf['day']} {sf['shift']}: need {sf['needed']} {sf['role']}(s), have {sf['have']}"
            for sf in demand_shortfalls[:3]
        )
        rubric_items.append(
            BinaryRubricCategory(
                name="identifies_demand_shortfalls",
                question=(
                    f"Does the report identify demand/staffing shortfalls where the schedule "
                    f"doesn't meet the forecast? {len(demand_shortfalls)} shortfall(s) exist, "
                    f"including: {shortfall_desc}."
                ),
                points=2,
            )
        )

    # Overtime employee count
    rubric_items.append(
        BinaryRubricCategory(
            name="correct_overtime_employee_count",
            question=(
                f"Does the report correctly identify {len(employees_with_overtime)} employee(s) "
                f"with weekly hours exceeding 40 (incurring overtime)?"
            ),
            points=1,
        )
    )

    rubric_items.extend([
        BinaryRubricCategory(
            name="correct_overtime_cost",
            question=(
                f"Does the report compute the total weekly overtime cost as approximately "
                f"{_fmt_money(total_overtime_cost)} (within $20)? "
                f"(Overtime = hours over 40 x 1.5 x base rate for each employee.)"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_night_differential_cost",
            question=(
                f"Does the report compute or account for the total night shift differential "
                f"cost as approximately {_fmt_money(total_night_diff_cost)} (within $20)? "
                f"(Night differential = {_fmt_money(NIGHT_DIFFERENTIAL)}/hr for all night shift hours.)"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_total_labor_cost",
            question=(
                f"Does the report compute the total weekly labor cost (regular + overtime + "
                f"night differential) as approximately {_fmt_money(total_labor_cost)} (within $100)?"
            ),
            points=3,
        ),
    ])

    rubric_items.append(
        RubricCategory(
            name="presentation_quality",
            description="Is the audit report well-organized, professional, and clearly presented?",
            failure="Disorganized output or raw numbers without context.",
            minor_failure="Some structure but missing key sections or hard to follow.",
            minor_success="Reasonably organized with clear labels and logical flow.",
            success="Professional audit report with clear sections, specific citations (employee IDs, days, shifts), and a definitive conclusion.",
            points=2,
        )
    )

    rubric = tuple(rubric_items)

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write your completed audit report to /testbed/audit_report.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/employees.txt": emp_content,
            "/testbed/data/demand_forecast.txt": demand_content,
            "/testbed/data/labor_rules.txt": labor_content,
            "/testbed/data/pay_rates.txt": pay_content,
            "/testbed/data/proposed_schedule.txt": sched_content,
        },
        problem_type="shift_scheduling_audit",
    )


# =============================================================================
# DOMAIN: SUPPLY CHAIN
# =============================================================================

SKU_POOL = [
    ("SKU-001", "Steel Bolts M8x40 (box/500)"),
    ("SKU-002", "Copper Wire 14AWG (100m spool)"),
    ("SKU-003", "Nylon Washers 10mm (bag/1000)"),
    ("SKU-004", "Aluminum Extrusion T-Slot 40x40 (3m)"),
    ("SKU-005", "Rubber O-Ring Kit (assorted)"),
    ("SKU-006", "Stainless Steel Sheet 1.5mm (4x8 ft)"),
    ("SKU-007", "PVC Pipe 2-inch (10ft length)"),
    ("SKU-008", "Hydraulic Hose 3/8-inch (per ft)"),
    ("SKU-009", "Electric Motor 1HP Single Phase"),
    ("SKU-010", "Bearing SKF 6208-2RS"),
    ("SKU-011", "Silicone Sealant Tube (12oz)"),
    ("SKU-012", "Carbide End Mill 8mm 4-Flute"),
    ("SKU-013", "Safety Glasses (box/12)"),
    ("SKU-014", "LED Panel Light 40W 2x2ft"),
    ("SKU-015", "Pneumatic Cylinder 50mm Bore"),
    ("SKU-016", "Thermal Paste (50g tube)"),
    ("SKU-017", "Cable Ties 300mm (bag/500)"),
    ("SKU-018", "Welding Rod E6013 3.2mm (5kg)"),
    ("SKU-019", "Air Filter Element 10-inch"),
    ("SKU-020", "Conveyor Belt Section (per m)"),
    ("SKU-021", "Industrial Relay 24VDC"),
    ("SKU-022", "Proximity Sensor Inductive 8mm"),
    ("SKU-023", "Gear Pump 10 GPM"),
    ("SKU-024", "Spring Washer M12 (bag/200)"),
    ("SKU-025", "Cutting Fluid 5-Gallon"),
]

SUPPLIER_POOL = [
    ("FastTrack Industrial", 5, 0.95),
    ("Global Parts Direct", 12, 0.88),
    ("Pacific Supply Corp", 8, 0.92),
    ("MidWest Fasteners LLC", 7, 0.90),
    ("Atlas Component Co.", 14, 0.85),
    ("Precision Parts Int'l", 10, 0.91),
    ("Eagle Industrial Supply", 6, 0.93),
    ("Summit Hardware Dist.", 9, 0.89),
    ("Continental Materials", 11, 0.87),
    ("Reliable Tool & Equip.", 4, 0.96),
]

WAREHOUSE_LOCATIONS = ["Warehouse A (Main)", "Warehouse B (Overflow)", "Warehouse C (Remote)"]


def make_supply_chain_optimization(rand_seed: int = 42) -> RubricDatapoint:
    """Given inventory data, supplier catalogs, demand history, and cost parameters,
    determine reorder points, identify stockout risks, compute EOQ, and recommend
    optimal orders.

    Seed varies: SKU mix (15-25), demand patterns (seasonal/trending/steady),
    supplier lead times, which items are critically low, cost parameters.
    """
    rng = _random.Random(rand_seed)

    company = pick1(COMPANY_NAMES, rand_seed)
    analyst_name = random_name(rand_seed + 2)

    # --- Select SKUs ---
    n_skus = rng.randint(15, 25)
    chosen_skus = rng.sample(SKU_POOL, n_skus)

    # --- Select suppliers ---
    n_suppliers = rng.randint(4, 7)
    chosen_suppliers = rng.sample(SUPPLIER_POOL, n_suppliers)

    # Vary lead times with seed
    suppliers: list[dict] = []
    for name, base_lead, reliability in chosen_suppliers:
        lead_time = base_lead + rng.randint(-2, 3)
        lead_time = max(3, lead_time)
        min_order_qty = rng.choice([10, 25, 50, 100])
        # Volume discount: order > threshold gets discount
        discount_threshold = rng.choice([200, 500, 1000])
        discount_pct = rng.choice([0.03, 0.05, 0.08, 0.10])
        suppliers.append({
            "name": name,
            "lead_time_days": lead_time,
            "reliability": reliability + rng.uniform(-0.03, 0.03),
            "min_order_qty": min_order_qty,
            "discount_threshold": discount_threshold,
            "discount_pct": discount_pct,
        })

    # --- Cost parameters ---
    carrying_cost_rate = round(rng.uniform(0.18, 0.30), 2)  # annual, as fraction of item cost
    ordering_cost = round(rng.uniform(25.0, 75.0), 2)  # per order
    stockout_penalty = round(rng.uniform(5.0, 25.0), 2)  # per unit per day

    # --- Build SKU inventory and demand data ---
    skus: list[dict] = []
    demand_patterns = ["steady", "seasonal", "trending_up", "trending_down"]

    for sku_code, sku_desc in chosen_skus:
        unit_cost = round(rng.uniform(2.0, 150.0), 2)
        pattern = rng.choice(demand_patterns)
        warehouse = rng.choice(WAREHOUSE_LOCATIONS)

        # Assign 1-3 suppliers that carry this SKU
        n_sku_suppliers = rng.randint(1, min(3, len(suppliers)))
        sku_suppliers = rng.sample(suppliers, n_sku_suppliers)
        # Pick best (lowest cost after potential discount, shortest lead time)
        sku_supplier_prices: list[dict] = []
        for sup in sku_suppliers:
            # Supplier price = unit_cost * (0.9 to 1.1) — varies by supplier
            sup_price = round(unit_cost * rng.uniform(0.90, 1.12), 2)
            sku_supplier_prices.append({
                "supplier": sup["name"],
                "unit_price": sup_price,
                "lead_time": sup["lead_time_days"],
                "min_order_qty": sup["min_order_qty"],
                "discount_threshold": sup["discount_threshold"],
                "discount_pct": sup["discount_pct"],
            })

        # Monthly demand for 12 months
        base_demand = rng.randint(20, 300)
        monthly_demand: list[int] = []
        for month in range(12):
            if pattern == "steady":
                d = base_demand + rng.randint(-10, 10)
            elif pattern == "seasonal":
                # Peak in months 5-8 (summer)
                seasonal_factor = 1.0 + 0.6 * math.sin((month - 2) * math.pi / 6)
                d = int(base_demand * seasonal_factor) + rng.randint(-5, 5)
            elif pattern == "trending_up":
                d = int(base_demand * (1 + 0.05 * month)) + rng.randint(-5, 5)
            else:  # trending_down
                d = int(base_demand * (1 - 0.03 * month)) + rng.randint(-5, 5)
            monthly_demand.append(max(5, d))

        avg_monthly_demand = round(sum(monthly_demand) / 12, 1)
        avg_daily_demand = round(avg_monthly_demand / 30, 2)

        # Current stock: some items are critically low
        is_critical = rng.random() < 0.25
        if is_critical:
            units_on_hand = rng.randint(0, int(avg_daily_demand * 3))
        else:
            units_on_hand = rng.randint(int(avg_daily_demand * 10), int(avg_daily_demand * 45))

        # Reorder point (ROP) = avg_daily_demand * lead_time + safety_stock
        # Use the shortest supplier lead time for this SKU
        best_lead_time = min(sp["lead_time"] for sp in sku_supplier_prices)
        safety_stock = int(round(avg_daily_demand * 5))  # 5 days safety stock
        reorder_point = int(round(avg_daily_demand * best_lead_time + safety_stock))

        # Current reorder qty on file (may not be optimal)
        current_reorder_qty = rng.choice([50, 100, 150, 200, 250, 500])

        skus.append({
            "code": sku_code,
            "description": sku_desc,
            "unit_cost": unit_cost,
            "units_on_hand": units_on_hand,
            "reorder_point": reorder_point,
            "current_reorder_qty": current_reorder_qty,
            "warehouse": warehouse,
            "pattern": pattern,
            "monthly_demand": monthly_demand,
            "avg_monthly_demand": avg_monthly_demand,
            "avg_daily_demand": avg_daily_demand,
            "safety_stock": safety_stock,
            "best_lead_time": best_lead_time,
            "is_critical": is_critical,
            "suppliers": sku_supplier_prices,
        })

    # --- Compute ground-truth values ---
    items_below_reorder: list[dict] = []
    items_at_stockout_risk: list[dict] = []
    eoq_values: dict[str, int] = {}
    best_supplier_per_sku: dict[str, str] = {}
    order_costs: dict[str, float] = {}

    for sku in skus:
        # Below reorder point?
        if sku["units_on_hand"] < sku["reorder_point"]:
            items_below_reorder.append(sku)

        # Stockout risk: will stock run out within the best supplier lead time?
        demand_during_lead = sku["avg_daily_demand"] * sku["best_lead_time"]
        if sku["units_on_hand"] < demand_during_lead:
            days_of_stock = sku["units_on_hand"] / max(sku["avg_daily_demand"], 0.01)
            items_at_stockout_risk.append({
                "code": sku["code"],
                "description": sku["description"],
                "units_on_hand": sku["units_on_hand"],
                "days_of_stock": round(days_of_stock, 1),
                "lead_time": sku["best_lead_time"],
            })

        # EOQ: sqrt(2 * D * S / H)
        # D = annual demand, S = ordering cost, H = holding cost per unit per year
        annual_demand = sku["avg_monthly_demand"] * 12
        holding_cost_per_unit = sku["unit_cost"] * carrying_cost_rate
        if holding_cost_per_unit > 0 and annual_demand > 0:
            eoq_raw = math.sqrt(2 * annual_demand * ordering_cost / holding_cost_per_unit)
            eoq = max(1, int(round(eoq_raw)))
        else:
            eoq = sku["current_reorder_qty"]
        eoq_values[sku["code"]] = eoq

        # Best supplier: lowest effective unit price (considering volume discount with EOQ)
        best_sup = None
        best_price = float("inf")
        for sp in sku["suppliers"]:
            effective_price = sp["unit_price"]
            if eoq >= sp["discount_threshold"]:
                effective_price = round(sp["unit_price"] * (1 - sp["discount_pct"]), 2)
            if effective_price < best_price:
                best_price = effective_price
                best_sup = sp["supplier"]
        best_supplier_per_sku[sku["code"]] = best_sup or sku["suppliers"][0]["supplier"]
        order_costs[sku["code"]] = round(best_price * eoq, 2) if sku["units_on_hand"] < sku["reorder_point"] else 0.0

    total_order_cost = round(sum(order_costs.values()), 2)
    total_annual_carrying_cost = round(
        sum(sku["units_on_hand"] * sku["unit_cost"] * carrying_cost_rate for sku in skus), 2
    )

    # Pick 3 SKUs for detailed EOQ rubric checks
    skus_for_eoq_check = rng.sample(items_below_reorder, min(3, len(items_below_reorder)))

    # --- Build inventory.txt ---
    inv_lines = [
        f"{company} — INVENTORY STATUS REPORT",
        f"Report Date: 2024-09-15",
        "",
        "=" * 90,
        "",
        f"{'SKU':<10} {'Description':<35} {'On Hand':>8} {'Reorder Pt':>11} {'Reorder Qty':>12} {'Warehouse':<25}",
        f"{'-'*10} {'-'*35} {'-'*8} {'-'*11} {'-'*12} {'-'*25}",
    ]
    for sku in skus:
        inv_lines.append(
            f"{sku['code']:<10} {sku['description']:<35} {sku['units_on_hand']:>8} "
            f"{sku['reorder_point']:>11} {sku['current_reorder_qty']:>12} {sku['warehouse']:<25}"
        )
    inv_lines.extend(["", f"Total SKUs tracked: {n_skus}", ""])
    inv_content = "\n".join(inv_lines) + "\n"

    # --- Build demand_history.csv ---
    csv_lines = ["SKU,Description,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec,Avg_Monthly"]
    for sku in skus:
        month_vals = ",".join(str(d) for d in sku["monthly_demand"])
        csv_lines.append(
            f"{sku['code']},{sku['description']},{month_vals},"
            f"{sku['avg_monthly_demand']}"
        )
    csv_content = "\n".join(csv_lines) + "\n"

    # --- Build supplier_catalog.txt ---
    sup_lines = [
        "SUPPLIER CATALOG — APPROVED VENDORS",
        "",
        "=" * 80,
    ]
    for sup in suppliers:
        sup_lines.append("")
        sup_lines.append(f"Supplier: {sup['name']}")
        sup_lines.append(f"  Lead Time: {sup['lead_time_days']} days")
        sup_lines.append(f"  Reliability Score: {sup['reliability']:.2f}")
        sup_lines.append(f"  Minimum Order Quantity: {sup['min_order_qty']} units")
        sup_lines.append(f"  Volume Discount: {sup['discount_pct']*100:.0f}% off for orders >= {sup['discount_threshold']} units")
        # List which SKUs this supplier carries with prices
        sku_list = []
        for sku in skus:
            for sp in sku["suppliers"]:
                if sp["supplier"] == sup["name"]:
                    sku_list.append((sku["code"], sp["unit_price"]))
        if sku_list:
            sup_lines.append(f"  Catalog:")
            for code, price in sku_list:
                sup_lines.append(f"    {code}: {_fmt_money(price)}/unit")
        sup_lines.append(f"  {'-'*40}")
    sup_lines.append("")
    sup_content = "\n".join(sup_lines) + "\n"

    # --- Build cost_parameters.txt ---
    cost_lines = [
        "INVENTORY COST PARAMETERS",
        "",
        "=" * 50,
        "",
        f"Annual Carrying Cost Rate: {carrying_cost_rate*100:.0f}% of item unit cost",
        f"  (This covers warehousing, insurance, obsolescence, and capital cost)",
        "",
        f"Ordering Cost: {_fmt_money(ordering_cost)} per purchase order",
        f"  (Includes processing, receiving, inspection, and payment handling)",
        "",
        f"Stockout Penalty: {_fmt_money(stockout_penalty)} per unit per day",
        f"  (Covers expediting costs, lost production, and customer penalties)",
        "",
        "ECONOMIC ORDER QUANTITY (EOQ) FORMULA:",
        "  EOQ = sqrt(2 * D * S / H)",
        "  where:",
        "    D = Annual demand (units per year)",
        "    S = Ordering cost per order ($)",
        "    H = Annual holding cost per unit (= unit cost x carrying cost rate)",
        "",
        "REORDER POINT FORMULA:",
        "  ROP = (average daily demand x lead time in days) + safety stock",
        "  Safety stock = average daily demand x 5 days",
        "",
    ]
    cost_content = "\n".join(cost_lines) + "\n"

    # --- Problem statement ---
    problem_statement = f"""# Supply Chain Inventory Optimization

You are a supply chain analyst at {company}. Using the inventory status, demand
history, supplier catalog, and cost parameters, analyze the current inventory
position and recommend optimal reorder actions.

## Source Files
- /testbed/data/inventory.txt — Current stock levels, reorder points, warehouse locations
- /testbed/data/demand_history.csv — 12 months of demand data per SKU with patterns
- /testbed/data/supplier_catalog.txt — Approved suppliers, lead times, prices, volume discounts
- /testbed/data/cost_parameters.txt — Carrying cost rate, ordering cost, stockout penalty, EOQ formula

## Requirements
1. Identify all items currently below their reorder point
2. Identify items at risk of stockout within the supplier lead time
3. Compute the Economic Order Quantity (EOQ) for items needing reorder
4. For each item needing reorder, select the optimal supplier (lowest effective cost)
5. Compute the total cost of all recommended orders
6. Estimate the current annual carrying cost for all inventory on hand
7. Flag any items with unusual demand patterns (seasonal, trending)

Write a detailed analysis report to /testbed/optimization_report.txt."""

    # --- Build rubric ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does the file /testbed/optimization_report.txt exist and contain substantive content (more than a few lines)?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_total_skus",
            question=f"Does the report correctly identify or work with {n_skus} SKUs in the inventory?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_items_below_reorder_count",
            question=(
                f"Does the report correctly identify {len(items_below_reorder)} items as currently below "
                f"their reorder point? The items are: "
                f"{', '.join(s['code'] for s in items_below_reorder)}."
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_stockout_risk_count",
            question=(
                f"Does the report correctly identify {len(items_at_stockout_risk)} items at risk of "
                f"stockout within the supplier lead time? The at-risk items are: "
                f"{', '.join(s['code'] for s in items_at_stockout_risk)}."
            ),
            points=2,
        ),
    ]

    # Per-stockout-risk detail
    for risk_item in items_at_stockout_risk[:3]:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"stockout_detail_{risk_item['code']}",
                question=(
                    f"Does the report identify {risk_item['code']} ({risk_item['description']}) as having "
                    f"only approximately {risk_item['days_of_stock']} days of stock remaining (within 1 day), "
                    f"which is less than its {risk_item['lead_time']}-day lead time?"
                ),
                points=2,
            )
        )

    # EOQ checks for specific SKUs
    for sku in skus_for_eoq_check:
        eoq = eoq_values[sku["code"]]
        annual_d = round(sku["avg_monthly_demand"] * 12, 1)
        h_cost = round(sku["unit_cost"] * carrying_cost_rate, 2)
        rubric_items.append(
            BinaryRubricCategory(
                name=f"correct_eoq_{sku['code']}",
                question=(
                    f"Does the report compute the EOQ for {sku['code']} ({sku['description']}) as "
                    f"approximately {eoq} units (within 15%)? "
                    f"(Annual demand ~{annual_d}, ordering cost {_fmt_money(ordering_cost)}, "
                    f"holding cost ~{_fmt_money(h_cost)}/unit/year)"
                ),
                points=2,
            )
        )

    # Supplier selection checks for a few items
    sup_check_skus = rng.sample(items_below_reorder, min(3, len(items_below_reorder)))
    for sku in sup_check_skus:
        best_sup = best_supplier_per_sku[sku["code"]]
        rubric_items.append(
            BinaryRubricCategory(
                name=f"correct_supplier_{sku['code']}",
                question=(
                    f"Does the report recommend {best_sup} as the supplier for {sku['code']} "
                    f"({sku['description']}), or a supplier with an equivalent or lower effective price?"
                ),
                points=2,
            )
        )

    rubric_items.extend([
        BinaryRubricCategory(
            name="correct_total_order_cost",
            question=(
                f"Does the report compute the total cost of all recommended reorder actions as "
                f"approximately {_fmt_money(total_order_cost)} (within 15%)?"
            ),
            points=3,
        ),
        BinaryRubricCategory(
            name="correct_carrying_cost",
            question=(
                f"Does the report estimate the current annual carrying cost for all on-hand inventory "
                f"as approximately {_fmt_money(total_annual_carrying_cost)} (within 15%)?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="identifies_demand_patterns",
            question=(
                "Does the report flag or discuss items with non-steady demand patterns "
                "(seasonal, trending up, or trending down) and note the implications for "
                "inventory management?"
            ),
            points=2,
        ),
    ])

    # False-positive check: pick 2 items NOT below reorder to ensure they aren't flagged
    items_above_reorder = [s for s in skus if s["units_on_hand"] >= s["reorder_point"]]
    no_reorder_checks = rng.sample(items_above_reorder, min(2, len(items_above_reorder)))
    for sku in no_reorder_checks:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"no_false_reorder_{sku['code']}",
                question=(
                    f"Does the report correctly show {sku['code']} ({sku['description']}) as NOT "
                    f"needing immediate reorder? (On hand: {sku['units_on_hand']}, "
                    f"reorder point: {sku['reorder_point']})"
                ),
                points=1,
            )
        )

    rubric_items.append(
        RubricCategory(
            name="analysis_thoroughness",
            description="How thorough and systematic is the supply chain analysis?",
            failure="Superficial analysis; most items not evaluated or calculations missing.",
            minor_failure="Some items analyzed but analysis is incomplete or lacks detail.",
            minor_success="Most items evaluated with reasonable calculations and supplier recommendations.",
            success="Comprehensive analysis with EOQ calculations, supplier comparisons, stockout risk assessment, and clear prioritized recommendations.",
            points=2,
        )
    )

    rubric = tuple(rubric_items)

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write your completed analysis report to /testbed/optimization_report.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/inventory.txt": inv_content,
            "/testbed/data/demand_history.csv": csv_content,
            "/testbed/data/supplier_catalog.txt": sup_content,
            "/testbed/data/cost_parameters.txt": cost_content,
        },
        problem_type="supply_chain_optimization",
    )


# =============================================================================
# DOMAIN: ROUTE PLANNING
# =============================================================================

LOCATION_NAMES = [
    "Depot (Hub)", "Riverside Mall", "Oak Street Clinic", "Elm Park Warehouse",
    "Cedar Heights Office", "Pine Valley School", "Maple Grove Hotel",
    "Birch Lane Apartments", "Walnut Creek Gym", "Spruce Hill Library",
    "Ash Street Market", "Cherry Blossom Cafe", "Dogwood Industrial Park",
    "Fir Ridge Dental", "Hazel Street Pharmacy", "Ivy Court Restaurant",
    "Juniper Point Marina", "Larch Plaza Tower", "Magnolia Arts Center",
    "Nutmeg Lane Storage", "Poplar Heights Hospital", "Sequoia Tech Campus",
]

PRIORITY_LEVELS = ["standard", "express", "critical"]


def make_route_planning(rand_seed: int = 42) -> RubricDatapoint:
    """Given delivery locations, time windows, vehicle capacity, and a distance
    matrix, plan an efficient delivery route and identify infeasible stops.

    Seed varies: number of stops (12-20), time window tightness, vehicle capacity,
    which stops have narrow windows, whether all stops fit in one trip, package
    weights and volumes.
    """
    rng = _random.Random(rand_seed)

    company = pick1(COMPANY_NAMES, rand_seed)
    driver_name = random_name(rand_seed + 3)

    # --- Generate delivery stops ---
    n_stops = rng.randint(12, 20)
    # Location 0 is always the depot
    chosen_locations = ["Depot (Hub)"] + rng.sample(LOCATION_NAMES[1:], n_stops)

    stops: list[dict] = []
    total_weight = 0.0
    total_volume = 0.0
    for i, loc in enumerate(chosen_locations[1:], start=1):
        # Time windows
        window_start_hour = rng.randint(7, 14)
        window_width = rng.choice([2, 3, 4, 5])  # hours
        if rng.random() < 0.2:
            window_width = 1  # narrow window
        window_end_hour = min(window_start_hour + window_width, 20)
        window_start = f"{window_start_hour}:00"
        window_end = f"{window_end_hour}:00"

        weight = round(rng.uniform(5.0, 80.0), 1)
        volume = round(rng.uniform(0.5, 6.0), 1)  # cubic feet
        priority = rng.choice(PRIORITY_LEVELS)
        fragile = rng.random() < 0.15

        total_weight += weight
        total_volume += volume

        stops.append({
            "stop_id": i,
            "location": loc,
            "window_start_hour": window_start_hour,
            "window_end_hour": window_end_hour,
            "window_start": window_start,
            "window_end": window_end,
            "weight_lbs": weight,
            "volume_cuft": volume,
            "priority": priority,
            "fragile": fragile,
        })

    # --- Vehicle specs ---
    # Make capacity somewhat constraining
    weight_capacity = round(total_weight * rng.uniform(0.70, 0.90), 0)
    volume_capacity = round(total_volume * rng.uniform(0.75, 0.95), 0)
    max_driving_hours = 11  # DOT regulation
    required_break_after_hours = 8  # 30-minute break required after 8 hours
    break_duration_min = 30
    depot_return_hour = 20  # must return by 8 PM

    # --- Generate distance matrix ---
    all_locations = chosen_locations[:n_stops + 1]  # depot + stops
    n_locs = len(all_locations)
    dist_matrix: list[list[int]] = [[0] * n_locs for _ in range(n_locs)]
    for i in range(n_locs):
        for j in range(i + 1, n_locs):
            travel_time = rng.randint(8, 55)  # minutes
            dist_matrix[i][j] = travel_time
            dist_matrix[j][i] = travel_time

    # --- Compute ground-truth feasibility analysis ---
    # Greedy route: prioritize critical/express stops, then by earliest window
    priority_order = {"critical": 0, "express": 1, "standard": 2}
    sorted_stops = sorted(stops, key=lambda s: (priority_order[s["priority"]], s["window_start_hour"]))

    # Simulate a greedy feasible route
    feasible_stops: list[dict] = []
    deferred_stops: list[dict] = []
    current_weight = 0.0
    current_volume = 0.0
    current_time_min = 7 * 60  # start at 7:00 AM (minutes since midnight)
    driving_minutes = 0
    current_loc_idx = 0  # depot
    visited_indices: set[int] = {0}

    # First pass: determine which stops can be visited considering capacity
    capacity_feasible_stops = []
    remaining_weight = weight_capacity
    remaining_volume = volume_capacity
    for stop in sorted_stops:
        if stop["weight_lbs"] <= remaining_weight and stop["volume_cuft"] <= remaining_volume:
            capacity_feasible_stops.append(stop)
            remaining_weight -= stop["weight_lbs"]
            remaining_volume -= stop["volume_cuft"]
        else:
            deferred_stops.append({
                **stop,
                "defer_reason": "capacity",
            })

    # Second pass: simulate time-based feasibility among capacity-feasible stops
    for stop in capacity_feasible_stops:
        stop_idx = stop["stop_id"]
        travel_time = dist_matrix[current_loc_idx][stop_idx]

        arrival_time_min = current_time_min + travel_time
        # Check if we can arrive within the time window
        window_start_min = stop["window_start_hour"] * 60
        window_end_min = stop["window_end_hour"] * 60

        # If we arrive too early, wait
        if arrival_time_min < window_start_min:
            arrival_time_min = window_start_min

        # Service time: 10 minutes per stop
        service_time = 10
        departure_time_min = arrival_time_min + service_time

        # Check DOT driving limit
        new_driving = driving_minutes + travel_time
        # Check break requirement
        if new_driving > required_break_after_hours * 60 and driving_minutes <= required_break_after_hours * 60:
            departure_time_min += break_duration_min

        # Check time window
        if arrival_time_min > window_end_min:
            deferred_stops.append({**stop, "defer_reason": "time_window"})
            continue

        # Check if we can return to depot after this stop
        return_travel = dist_matrix[stop_idx][0]
        return_time_min = departure_time_min + return_travel
        depot_deadline_min = depot_return_hour * 60

        if return_time_min > depot_deadline_min:
            deferred_stops.append({**stop, "defer_reason": "depot_return"})
            continue

        # Check driving hours limit
        total_driving_if_visit = new_driving + return_travel
        if total_driving_if_visit > max_driving_hours * 60:
            deferred_stops.append({**stop, "defer_reason": "driving_hours"})
            continue

        # Feasible!
        feasible_stops.append(stop)
        current_loc_idx = stop_idx
        current_time_min = departure_time_min
        driving_minutes = new_driving
        current_weight += stop["weight_lbs"]
        current_volume += stop["volume_cuft"]
        visited_indices.add(stop_idx)

    # Compute total route time
    # Return to depot from last stop
    if feasible_stops:
        last_stop_idx = feasible_stops[-1]["stop_id"]
        return_to_depot_min = dist_matrix[last_stop_idx][0]
        total_route_time_min = current_time_min + return_to_depot_min - 7 * 60  # from 7 AM
    else:
        total_route_time_min = 0
        return_to_depot_min = 0

    total_route_hours = round(total_route_time_min / 60, 1)
    total_feasible_weight = round(sum(s["weight_lbs"] for s in feasible_stops), 1)
    total_feasible_volume = round(sum(s["volume_cuft"] for s in feasible_stops), 1)
    weight_utilization_pct = round(total_feasible_weight / weight_capacity * 100, 1)
    volume_utilization_pct = round(total_feasible_volume / volume_capacity * 100, 1)

    # Identify time window conflicts in the deferred set
    time_window_conflicts = [s for s in deferred_stops if s.get("defer_reason") == "time_window"]
    capacity_deferred = [s for s in deferred_stops if s.get("defer_reason") == "capacity"]

    # --- Build deliveries.txt ---
    del_lines = [
        f"{company} — DELIVERY MANIFEST",
        f"Date: 2024-09-16",
        f"Driver: {driver_name}",
        "",
        "=" * 95,
        "",
        f"{'Stop':<5} {'Location':<30} {'Window':>15} {'Weight(lbs)':>12} {'Volume(cuft)':>13} {'Priority':>10} {'Fragile':>8}",
        f"{'-'*5} {'-'*30} {'-'*15} {'-'*12} {'-'*13} {'-'*10} {'-'*8}",
    ]
    for stop in stops:
        window_str = f"{stop['window_start']}-{stop['window_end']}"
        fragile_str = "Yes" if stop["fragile"] else "No"
        del_lines.append(
            f"{stop['stop_id']:<5} {stop['location']:<30} {window_str:>15} "
            f"{stop['weight_lbs']:>12.1f} {stop['volume_cuft']:>13.1f} "
            f"{stop['priority']:>10} {fragile_str:>8}"
        )
    del_lines.extend([
        "",
        f"Total stops: {n_stops}",
        f"Total weight: {total_weight:.1f} lbs",
        f"Total volume: {total_volume:.1f} cuft",
        "",
    ])
    del_content = "\n".join(del_lines) + "\n"

    # --- Build distance_matrix.csv ---
    csv_header = "Location," + ",".join(all_locations)
    csv_rows = [csv_header]
    for i, loc in enumerate(all_locations):
        row_vals = ",".join(str(dist_matrix[i][j]) for j in range(n_locs))
        csv_rows.append(f"{loc},{row_vals}")
    csv_content = "\n".join(csv_rows) + "\n"

    # --- Build vehicle_specs.txt ---
    veh_lines = [
        "VEHICLE SPECIFICATIONS AND DOT COMPLIANCE",
        "",
        "=" * 50,
        "VEHICLE CAPACITY",
        "=" * 50,
        "",
        f"Maximum Weight Capacity: {weight_capacity:.0f} lbs",
        f"Maximum Volume Capacity: {volume_capacity:.0f} cuft",
        f"Vehicle Type: Medium box truck",
        "",
        "=" * 50,
        "DOT DRIVING REGULATIONS",
        "=" * 50,
        "",
        f"Maximum Driving Hours Per Day: {max_driving_hours} hours",
        f"Mandatory Break: {break_duration_min} minutes after {required_break_after_hours} hours of driving",
        f"Service Time Per Stop: 10 minutes (loading/unloading)",
        "",
        "=" * 50,
        "OPERATIONAL CONSTRAINTS",
        "=" * 50,
        "",
        f"Departure Time: 7:00 AM from depot",
        f"Return Deadline: {depot_return_hour}:00 (must return to depot by this time)",
        f"Fragile items: Must be loaded last, unloaded first (affects route order if present)",
        "",
    ]
    veh_content = "\n".join(veh_lines) + "\n"

    # --- Build constraints.txt ---
    con_lines = [
        "DELIVERY CONSTRAINTS AND PRIORITY RULES",
        "",
        "=" * 50,
        "",
        "PRIORITY HANDLING:",
        "  1. Critical deliveries MUST be attempted first",
        "  2. Express deliveries take precedence over standard",
        "  3. Standard deliveries are best-effort within constraints",
        "",
        "TIME WINDOWS:",
        "  - Deliveries must arrive within the specified window",
        "  - Early arrivals may wait; late arrivals are violations",
        "  - If a time window cannot be met, the stop should be deferred",
        "",
        "CAPACITY MANAGEMENT:",
        "  - Vehicle weight and volume limits are absolute",
        "  - If total manifested cargo exceeds capacity, stops must be deferred",
        "  - Defer lowest-priority stops first when shedding load",
        "",
        "ROUTE PLANNING:",
        "  - Minimize total route time while respecting all constraints",
        "  - Account for service time (10 min) at each stop",
        "  - If not all stops can be completed, identify which to defer",
        "  - Always ensure enough time to return to depot by deadline",
        "",
        "FEASIBILITY REPORT REQUIREMENTS:",
        "  - Total feasible stops vs total requested",
        "  - Which stops must be deferred and why",
        "  - Total route duration (hours)",
        "  - Weight and volume utilization (%)",
        "  - Any time window conflicts",
        "",
    ]
    con_content = "\n".join(con_lines) + "\n"

    # --- Problem statement ---
    problem_statement = f"""# Delivery Route Planning and Feasibility Analysis

You are a logistics route planner for {company}. Given the delivery manifest,
distance matrix, vehicle specifications, and operational constraints, plan the
most efficient delivery route and identify any stops that cannot be feasibly
completed.

## Source Files
- /testbed/data/deliveries.txt — Delivery stops with time windows, weights, volumes, priorities
- /testbed/data/distance_matrix.csv — Travel time in minutes between all location pairs
- /testbed/data/vehicle_specs.txt — Vehicle capacity, DOT driving limits, service time per stop
- /testbed/data/constraints.txt — Priority rules, time window policies, capacity management

## Requirements
1. Determine if all {n_stops} stops can be completed in a single trip
2. If not, identify which stops must be deferred and categorize the reason (capacity, time window, driving hours, depot return deadline)
3. Plan a feasible route for the deliverable stops, respecting priority order
4. Compute total route duration from departure (7:00 AM) to return to depot
5. Compute total weight and volume loaded vs. vehicle capacity (utilization %)
6. Flag any time window conflicts
7. Identify the critical and express stops and confirm they are prioritized

Write a detailed route plan and feasibility report to /testbed/route_report.txt."""

    # --- Build rubric ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does the file /testbed/route_report.txt exist and contain substantive content (more than a few lines)?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_total_stops",
            question=f"Does the report correctly identify {n_stops} total delivery stops in the manifest?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_total_manifest_weight",
            question=f"Does the report correctly state the total manifest weight as {total_weight:.1f} lbs (within 1 lb)?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_total_manifest_volume",
            question=f"Does the report correctly state the total manifest volume as {total_volume:.1f} cuft (within 0.5)?",
            points=1,
        ),
        BinaryRubricCategory(
            name="identifies_capacity_constraint",
            question=(
                f"Does the report correctly identify that the total manifest weight ({total_weight:.1f} lbs) "
                f"exceeds the vehicle weight capacity ({weight_capacity:.0f} lbs) "
                f"{'and/or volume exceeds capacity' if total_volume > volume_capacity else 'so not all stops can be loaded'}?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_feasible_stops_count",
            question=(
                f"Does the report identify approximately {len(feasible_stops)} stops as feasible "
                f"for the route (within 2 stops)?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_deferred_stops_count",
            question=(
                f"Does the report identify approximately {len(deferred_stops)} stops as needing to be "
                f"deferred (within 2 stops)?"
            ),
            points=2,
        ),
    ]

    # Check specific deferred stops
    for ds in deferred_stops[:3]:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"deferred_stop_{ds['stop_id']}",
                question=(
                    f"Does the report correctly identify stop {ds['stop_id']} ({ds['location']}) as "
                    f"needing to be deferred, with the reason being {ds['defer_reason']} "
                    f"(or an equivalent explanation)?"
                ),
                points=2,
            )
        )

    rubric_items.extend([
        BinaryRubricCategory(
            name="correct_route_duration",
            question=(
                f"Does the report compute the total route duration as approximately "
                f"{total_route_hours} hours (within 1 hour)?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_weight_loaded",
            question=(
                f"Does the report state the total weight loaded on the feasible route as "
                f"approximately {total_feasible_weight:.1f} lbs (within 20 lbs)?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_weight_utilization",
            question=(
                f"Does the report compute weight utilization as approximately "
                f"{weight_utilization_pct:.0f}% (within 5 percentage points)?"
            ),
            points=2,
        ),
    ])

    # Time window conflict checks
    if time_window_conflicts:
        conflict_names = ", ".join(
            f"stop {s['stop_id']} ({s['location']})" for s in time_window_conflicts
        )
        rubric_items.append(
            BinaryRubricCategory(
                name="identifies_time_window_conflicts",
                question=(
                    f"Does the report identify time window conflicts for: {conflict_names}?"
                ),
                points=2,
            )
        )

    # False-positive check: pick 2 feasible stops to verify no false deferral
    false_pos_stops = rng.sample(feasible_stops, min(2, len(feasible_stops)))
    for stop in false_pos_stops:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"no_false_defer_{stop['stop_id']}",
                question=(
                    f"Does the report correctly include stop {stop['stop_id']} ({stop['location']}) "
                    f"in the feasible route and NOT defer it?"
                ),
                points=1,
            )
        )

    # Priority handling check
    critical_stops = [s for s in stops if s["priority"] == "critical"]
    express_stops = [s for s in stops if s["priority"] == "express"]
    if critical_stops:
        rubric_items.append(
            BinaryRubricCategory(
                name="prioritizes_critical_stops",
                question=(
                    f"Does the report prioritize the {len(critical_stops)} critical "
                    f"stop(s) and attempt to schedule them first or confirm they are included?"
                ),
                points=2,
            )
        )
    if express_stops:
        rubric_items.append(
            BinaryRubricCategory(
                name="prioritizes_express_stops",
                question=(
                    f"Does the report address the {len(express_stops)} express "
                    f"stop(s) and schedule them with higher priority than standard stops?"
                ),
                points=1,
            )
        )

    rubric_items.extend([
        BinaryRubricCategory(
            name="accounts_for_service_time",
            question="Does the report account for the 10-minute service time at each stop in its route duration calculations?",
            points=1,
        ),
        BinaryRubricCategory(
            name="confirms_depot_return",
            question=(
                f"Does the report confirm the vehicle can return to the depot by the "
                f"{depot_return_hour}:00 deadline?"
            ),
            points=1,
        ),
    ])

    rubric_items.append(
        RubricCategory(
            name="planning_quality",
            description="How thorough and well-reasoned is the route plan and feasibility analysis?",
            failure="No coherent route plan; stops listed without feasibility analysis.",
            minor_failure="Some feasibility analysis but major gaps in reasoning or missing constraints.",
            minor_success="Reasonable route plan with most constraints considered and clear deferred-stop rationale.",
            success="Comprehensive route plan with all constraints verified, clear deferred-stop categorization, utilization analysis, and justified priority ordering.",
            points=2,
        )
    )

    rubric = tuple(rubric_items)

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write your completed route plan and feasibility report to /testbed/route_report.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/deliveries.txt": del_content,
            "/testbed/data/distance_matrix.csv": csv_content,
            "/testbed/data/vehicle_specs.txt": veh_content,
            "/testbed/data/constraints.txt": con_content,
        },
        problem_type="route_planning",
    )
