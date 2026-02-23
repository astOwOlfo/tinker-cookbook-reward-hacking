"""Data engineering and data quality tasks: ETL pipeline validation, schema
migration review, and data quality audit.

Each factory generates a realistic multi-file scenario with planted issues that
the agent must discover through analysis. Seeds control randomization of data
shapes, error types, column names, and ground-truth answers embedded in the
rubric.
"""

from __future__ import annotations

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
# SHARED HELPERS
# =============================================================================

_US_STATES = [
    ("AL", "Alabama"), ("AK", "Alaska"), ("AZ", "Arizona"), ("AR", "Arkansas"),
    ("CA", "California"), ("CO", "Colorado"), ("CT", "Connecticut"),
    ("DE", "Delaware"), ("FL", "Florida"), ("GA", "Georgia"), ("HI", "Hawaii"),
    ("ID", "Idaho"), ("IL", "Illinois"), ("IN", "Indiana"), ("IA", "Iowa"),
    ("KS", "Kansas"), ("KY", "Kentucky"), ("LA", "Louisiana"), ("ME", "Maine"),
    ("MD", "Maryland"), ("MA", "Massachusetts"), ("MI", "Michigan"),
    ("MN", "Minnesota"), ("MS", "Mississippi"), ("MO", "Missouri"),
    ("MT", "Montana"), ("NE", "Nebraska"), ("NV", "Nevada"),
    ("NH", "New Hampshire"), ("NJ", "New Jersey"), ("NM", "New Mexico"),
    ("NY", "New York"), ("NC", "North Carolina"), ("ND", "North Dakota"),
    ("OH", "Ohio"), ("OK", "Oklahoma"), ("OR", "Oregon"), ("PA", "Pennsylvania"),
    ("RI", "Rhode Island"), ("SC", "South Carolina"), ("SD", "South Dakota"),
    ("TN", "Tennessee"), ("TX", "Texas"), ("UT", "Utah"), ("VT", "Vermont"),
    ("VA", "Virginia"), ("WA", "Washington"), ("WV", "West Virginia"),
    ("WI", "Wisconsin"), ("WY", "Wyoming"),
]

_PRODUCT_CATEGORIES = [
    "Electronics", "Apparel", "Home & Garden", "Sporting Goods",
    "Automotive", "Health & Beauty", "Toys & Games", "Office Supplies",
    "Food & Beverage", "Books & Media",
]

_PRODUCT_NAMES = {
    "Electronics": ["Wireless Headphones", "Bluetooth Speaker", "USB-C Hub",
                    "Portable Charger", "Smart Watch", "Tablet Stand",
                    "Webcam HD", "Keyboard Mechanical"],
    "Apparel": ["Cotton T-Shirt", "Running Shoes", "Winter Jacket",
                "Denim Jeans", "Baseball Cap", "Wool Scarf",
                "Hiking Boots", "Rain Poncho"],
    "Home & Garden": ["Garden Hose 50ft", "LED Desk Lamp", "Throw Pillow",
                      "Coffee Maker", "Plant Pot Ceramic", "Wall Clock",
                      "Cutting Board Bamboo", "Candle Set"],
    "Sporting Goods": ["Yoga Mat", "Resistance Bands", "Tennis Racket",
                       "Basketball", "Swim Goggles", "Jump Rope",
                       "Fishing Rod", "Camping Tent"],
    "Automotive": ["Car Phone Mount", "Tire Pressure Gauge", "Seat Cover",
                   "Dash Cam", "Air Freshener", "Emergency Kit",
                   "Floor Mats", "Windshield Shade"],
    "Health & Beauty": ["Moisturizer", "Shampoo Organic", "Toothbrush Electric",
                        "Vitamin D Supplement", "Sunscreen SPF50", "Hair Dryer",
                        "Face Mask Pack", "Hand Sanitizer"],
    "Toys & Games": ["Building Blocks Set", "Board Game Classic", "Puzzle 1000pc",
                     "Action Figure", "Stuffed Bear", "Card Game",
                     "Remote Control Car", "Art Supply Kit"],
    "Office Supplies": ["Notebook Ruled", "Ballpoint Pen Pack", "Stapler",
                        "Desk Organizer", "Sticky Notes", "Binder Clips",
                        "Whiteboard Marker", "Filing Cabinet"],
    "Food & Beverage": ["Organic Coffee Beans", "Green Tea Box", "Protein Bar",
                        "Olive Oil Extra Virgin", "Mixed Nuts", "Dark Chocolate",
                        "Sparkling Water", "Granola Cereal"],
    "Books & Media": ["Mystery Novel", "Cookbook Italian", "Self-Help Guide",
                      "Sci-Fi Anthology", "Children Picture Book", "Travel Atlas",
                      "Language Textbook", "Graphic Novel"],
}

_EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com",
    "icloud.com", "protonmail.com", "fastmail.com", "zoho.com", "mail.com",
]

_STREET_NAMES = [
    "Main St", "Oak Ave", "Elm Dr", "Cedar Ln", "Pine Rd",
    "Maple Blvd", "Birch Way", "Walnut Ct", "Spruce Pl", "Ash Ter",
    "Park Ave", "Lake Dr", "Hill St", "River Rd", "Forest Ln",
    "Valley Blvd", "Sunset Way", "Meadow Ct", "Spring Pl", "Brook Ter",
]


def _fmt_money(amount: float) -> str:
    return f"${amount:,.2f}"


def _make_email(rng: _random.Random, first: str, last: str) -> str:
    domain = rng.choice(_EMAIL_DOMAINS)
    style = rng.randint(0, 3)
    if style == 0:
        return f"{first.lower()}.{last.lower()}@{domain}"
    elif style == 1:
        return f"{first.lower()}{last.lower()[0]}@{domain}"
    elif style == 2:
        return f"{first.lower()[0]}{last.lower()}@{domain}"
    else:
        return f"{first.lower()}_{last.lower()}{rng.randint(1,99)}@{domain}"


def _make_phone(rng: _random.Random) -> str:
    area = rng.randint(200, 999)
    prefix = rng.randint(200, 999)
    line = rng.randint(1000, 9999)
    fmt = rng.choice(["({area}) {prefix}-{line}",
                       "{area}-{prefix}-{line}",
                       "{area}.{prefix}.{line}"])
    return fmt.format(area=area, prefix=prefix, line=line)


def _make_date_str(rng: _random.Random, year: int, month: int | None = None,
                   day: int | None = None) -> str:
    if month is None:
        month = rng.randint(1, 12)
    if day is None:
        max_day = [31,28,31,30,31,30,31,31,30,31,30,31][month - 1]
        day = rng.randint(1, max_day)
    return f"{year}-{month:02d}-{day:02d}"


# =============================================================================
# 1. ETL PIPELINE VALIDATION
# =============================================================================

# Error types that can be planted in the ETL output
_ETL_ERROR_TYPES = [
    "duplicate_records",
    "null_handling_inconsistent",
    "date_format_swap",
    "truncated_strings",
    "incorrect_derived_field",
    "unfiltered_records",
    "foreign_key_violation",
    "data_type_mismatch",
    "missing_records",
    "incorrect_aggregation",
    "timezone_conversion_error",
    "encoding_corruption",
]


def make_etl_pipeline_validation(rand_seed: int = 42) -> RubricDatapoint:
    """Validate an ETL pipeline output against source data and transformation
    rules. The target data contains 4-7 planted errors from a pool of 12 types.

    Seed varies: number of rows, product mix, customer names, which errors are
    planted, error locations, derived field formulas, filter criteria.
    """
    rng = _random.Random(rand_seed)

    company = pick1(COMPANY_NAMES, rand_seed)

    # --- Scenario parameters ---
    n_source_rows = rng.randint(500, 1000)
    n_customers = rng.randint(80, 200)
    n_products = rng.randint(30, 60)

    # Pick 4-7 errors to plant
    n_errors = rng.randint(4, 7)
    available_errors = list(_ETL_ERROR_TYPES)
    rng.shuffle(available_errors)
    planted_errors = available_errors[:n_errors]

    # Transformation rule parameters (vary per seed)
    min_order_amount = rng.choice([10.0, 15.0, 20.0, 25.0])
    tax_rate = rng.choice([0.065, 0.07, 0.075, 0.08, 0.085, 0.0925])
    discount_threshold = rng.choice([100.0, 150.0, 200.0])
    discount_rate = rng.choice([0.05, 0.08, 0.10, 0.12])
    max_product_name_len = rng.choice([40, 50, 60])
    timezone_offset = rng.choice([
        ("UTC", "US/Eastern", -5), ("UTC", "US/Central", -6),
        ("UTC", "US/Pacific", -8), ("UTC", "US/Mountain", -7),
    ])
    tz_src_label, tz_dst_label, tz_hours = timezone_offset
    status_filter = rng.choice(["completed", "shipped"])

    # Generate customers
    customer_names = random_names(rand_seed + 100, n_customers)
    customer_ids = list(range(1001, 1001 + n_customers))
    customer_map = {}
    for i, name in enumerate(customer_names):
        parts = name.split()
        first, last = parts[0], parts[-1]
        customer_map[customer_ids[i]] = {
            "id": customer_ids[i],
            "name": name,
            "email": _make_email(rng, first, last),
            "state": rng.choice(_US_STATES)[0],
        }

    # Generate products
    categories_used = rng.sample(_PRODUCT_CATEGORIES, min(n_products // 5 + 1, len(_PRODUCT_CATEGORIES)))
    products = []
    pid = 5001
    for _ in range(n_products):
        cat = rng.choice(categories_used)
        pname = rng.choice(_PRODUCT_NAMES[cat])
        price = round(rng.uniform(5.0, 500.0), 2)
        products.append({"id": pid, "name": pname, "category": cat, "price": price})
        pid += 1

    # Generate source transactions
    statuses = ["completed", "shipped", "cancelled", "pending", "returned"]
    source_rows = []
    for txn_id in range(10001, 10001 + n_source_rows):
        cust = rng.choice(list(customer_map.values()))
        prod = rng.choice(products)
        qty = rng.randint(1, 10)
        unit_price = prod["price"]
        subtotal = round(qty * unit_price, 2)
        status = rng.choices(statuses, weights=[50, 25, 10, 10, 5])[0]
        year = rng.choice([2023, 2024])
        month = rng.randint(1, 12)
        max_d = [31,28,31,30,31,30,31,31,30,31,30,31][month - 1]
        day = rng.randint(1, max_d)
        hour = rng.randint(0, 23)
        minute = rng.randint(0, 59)
        second = rng.randint(0, 59)
        # Source dates always in YYYY-MM-DD HH:MM:SS UTC format
        order_date = f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"
        product_name_full = f"{prod['name']} - {prod['category']}"
        source_rows.append({
            "txn_id": txn_id,
            "customer_id": cust["id"],
            "customer_name": cust["name"],
            "product_id": prod["id"],
            "product_name": product_name_full,
            "category": prod["category"],
            "quantity": qty,
            "unit_price": unit_price,
            "subtotal": subtotal,
            "status": status,
            "order_date": order_date,
            "year": year, "month": month, "day": day,
            "hour": hour, "minute": minute, "second": second,
        })

    # Build the "correct" target rows (applying transformation rules)
    correct_target = []
    for row in source_rows:
        # Rule: Filter out rows where status != status_filter
        if row["status"] != status_filter:
            continue
        # Rule: Filter out rows where subtotal < min_order_amount
        if row["subtotal"] < min_order_amount:
            continue
        # Derived fields
        tax_amount = round(row["subtotal"] * tax_rate, 2)
        if row["subtotal"] >= discount_threshold:
            discount = round(row["subtotal"] * discount_rate, 2)
        else:
            discount = 0.0
        total = round(row["subtotal"] + tax_amount - discount, 2)
        # Truncate product name
        pname_trunc = row["product_name"][:max_product_name_len]
        # Convert timezone
        adj_hour = (row["hour"] + tz_hours) % 24
        adj_day = row["day"]
        if row["hour"] + tz_hours < 0:
            adj_day = max(1, row["day"] - 1)
        elif row["hour"] + tz_hours >= 24:
            adj_day = min(28, row["day"] + 1)
        local_date = f"{row['year']}-{row['month']:02d}-{adj_day:02d} {adj_hour:02d}:{row['minute']:02d}:{row['second']:02d}"

        correct_target.append({
            "txn_id": row["txn_id"],
            "customer_id": row["customer_id"],
            "customer_name": row["customer_name"],
            "product_id": row["product_id"],
            "product_name": pname_trunc,
            "category": row["category"],
            "quantity": row["quantity"],
            "unit_price": row["unit_price"],
            "subtotal": row["subtotal"],
            "tax_amount": tax_amount,
            "discount": discount,
            "total": total,
            "order_date_local": local_date,
            "status": row["status"],
        })

    n_correct_target = len(correct_target)
    if n_correct_target < 10:
        # Ensure we have enough rows; shouldn't happen with 500-1000 source rows
        # but just in case, we work with what we have
        pass

    # Now build the "buggy" target by copying correct and planting errors
    buggy_target = [dict(r) for r in correct_target]

    # Track ground truth for each planted error
    error_details: dict[str, str] = {}

    # --- Plant errors ---
    if "duplicate_records" in planted_errors:
        n_dupes = rng.randint(3, 8)
        dupe_indices = rng.sample(range(len(buggy_target)), min(n_dupes, len(buggy_target)))
        dupes_to_insert = [dict(buggy_target[i]) for i in dupe_indices]
        for d in dupes_to_insert:
            buggy_target.append(d)
        dupe_txn_ids = sorted([buggy_target[i]["txn_id"] for i in dupe_indices])
        error_details["duplicate_records"] = (
            f"{n_dupes} duplicate records not removed; duplicated transaction IDs include: "
            + ", ".join(str(t) for t in dupe_txn_ids[:5])
        )

    if "null_handling_inconsistent" in planted_errors:
        null_indices = rng.sample(range(len(buggy_target)), min(rng.randint(5, 15), len(buggy_target)))
        n_zero = 0
        n_na = 0
        n_empty = 0
        for idx in null_indices:
            field = rng.choice(["discount", "tax_amount"])
            treatment = rng.choice(["zero", "na", "empty"])
            if treatment == "zero":
                buggy_target[idx][field] = 0.0 if field == "discount" else 0.0
                n_zero += 1
            elif treatment == "na":
                buggy_target[idx][field] = "N/A"
                n_na += 1
            else:
                buggy_target[idx][field] = ""
                n_empty += 1
        error_details["null_handling_inconsistent"] = (
            f"Null values handled inconsistently across {len(null_indices)} records: "
            f"{n_zero} replaced with 0, {n_na} replaced with 'N/A', {n_empty} left empty string"
        )

    if "date_format_swap" in planted_errors:
        swap_indices = rng.sample(range(len(buggy_target)), min(rng.randint(8, 20), len(buggy_target)))
        n_swapped = 0
        swapped_txns = []
        for idx in swap_indices:
            dt = buggy_target[idx]["order_date_local"]
            parts = dt.split(" ")[0].split("-")
            if len(parts) == 3:
                y, m, d = parts[0], parts[1], parts[2]
                time_part = dt.split(" ")[1] if " " in dt else "00:00:00"
                # Swap month and day (MM/DD vs DD/MM ambiguity)
                if int(m) <= 12 and int(d) <= 12:
                    buggy_target[idx]["order_date_local"] = f"{y}-{d}-{m} {time_part}"
                    n_swapped += 1
                    swapped_txns.append(buggy_target[idx]["txn_id"])
        error_details["date_format_swap"] = (
            f"{n_swapped} records have month/day swapped in order_date_local "
            f"(MM and DD transposed)"
        )

    if "truncated_strings" in planted_errors:
        trunc_len = rng.randint(15, 25)
        trunc_indices = rng.sample(range(len(buggy_target)), min(rng.randint(10, 25), len(buggy_target)))
        n_truncated = 0
        for idx in trunc_indices:
            orig = buggy_target[idx]["product_name"]
            if len(orig) > trunc_len:
                buggy_target[idx]["product_name"] = orig[:trunc_len]
                n_truncated += 1
        error_details["truncated_strings"] = (
            f"{n_truncated} product names truncated to {trunc_len} characters "
            f"(rule specifies {max_product_name_len} character limit)"
        )

    if "incorrect_derived_field" in planted_errors:
        wrong_field = rng.choice(["tax_amount", "total"])
        wrong_indices = rng.sample(range(len(buggy_target)), min(rng.randint(15, 40), len(buggy_target)))
        if wrong_field == "tax_amount":
            wrong_rate = round(tax_rate + rng.choice([0.01, -0.01, 0.02, -0.015]), 4)
            for idx in wrong_indices:
                row = buggy_target[idx]
                if isinstance(row["subtotal"], (int, float)):
                    row["tax_amount"] = round(float(row["subtotal"]) * wrong_rate, 2)
                    row["total"] = round(float(row["subtotal"]) + row["tax_amount"] - (float(row["discount"]) if isinstance(row["discount"], (int, float)) else 0), 2)
            error_details["incorrect_derived_field"] = (
                f"Tax amount calculated with wrong rate ({wrong_rate*100:.2f}% instead of "
                f"{tax_rate*100:.2f}%) in {len(wrong_indices)} records"
            )
        else:
            for idx in wrong_indices:
                row = buggy_target[idx]
                if isinstance(row["subtotal"], (int, float)) and isinstance(row["tax_amount"], (int, float)):
                    disc = float(row["discount"]) if isinstance(row["discount"], (int, float)) else 0
                    # Wrong: forgot to subtract discount
                    row["total"] = round(float(row["subtotal"]) + float(row["tax_amount"]), 2)
            error_details["incorrect_derived_field"] = (
                f"Total field incorrectly calculated (discount not subtracted from total) "
                f"in {len(wrong_indices)} records"
            )

    if "unfiltered_records" in planted_errors:
        # Add some rows that should have been filtered out
        filter_violations = []
        other_status = [s for s in statuses if s != status_filter]
        for _ in range(rng.randint(5, 12)):
            src_row = rng.choice(source_rows)
            if src_row["status"] != status_filter or src_row["subtotal"] < min_order_amount:
                # Build a target-like row from it
                tax_amount = round(src_row["subtotal"] * tax_rate, 2)
                discount = round(src_row["subtotal"] * discount_rate, 2) if src_row["subtotal"] >= discount_threshold else 0.0
                total = round(src_row["subtotal"] + tax_amount - discount, 2)
                pname_trunc = src_row["product_name"][:max_product_name_len]
                violation_row = {
                    "txn_id": src_row["txn_id"],
                    "customer_id": src_row["customer_id"],
                    "customer_name": src_row["customer_name"],
                    "product_id": src_row["product_id"],
                    "product_name": pname_trunc,
                    "category": src_row["category"],
                    "quantity": src_row["quantity"],
                    "unit_price": src_row["unit_price"],
                    "subtotal": src_row["subtotal"],
                    "tax_amount": tax_amount,
                    "discount": discount,
                    "total": total,
                    "order_date_local": f"{src_row['year']}-{src_row['month']:02d}-{src_row['day']:02d} {src_row['hour']:02d}:{src_row['minute']:02d}:{src_row['second']:02d}",
                    "status": src_row["status"],
                }
                buggy_target.append(violation_row)
                filter_violations.append(src_row["txn_id"])
        n_filter_violations = len(filter_violations)
        error_details["unfiltered_records"] = (
            f"{n_filter_violations} records present that should have been filtered out "
            f"(status != '{status_filter}' or subtotal < {_fmt_money(min_order_amount)})"
        )

    if "foreign_key_violation" in planted_errors:
        fk_indices = rng.sample(range(len(buggy_target)), min(rng.randint(3, 8), len(buggy_target)))
        invalid_cust_ids = [rng.randint(9000, 9999) for _ in fk_indices]
        for idx, bad_id in zip(fk_indices, invalid_cust_ids):
            buggy_target[idx]["customer_id"] = bad_id
        error_details["foreign_key_violation"] = (
            f"{len(fk_indices)} records reference non-existent customer IDs "
            f"(e.g., {invalid_cust_ids[0]}, {invalid_cust_ids[1] if len(invalid_cust_ids) > 1 else invalid_cust_ids[0]})"
        )

    if "data_type_mismatch" in planted_errors:
        type_indices = rng.sample(range(len(buggy_target)), min(rng.randint(4, 10), len(buggy_target)))
        for idx in type_indices:
            field = rng.choice(["quantity", "unit_price", "subtotal"])
            buggy_target[idx][field] = str(buggy_target[idx][field])
        error_details["data_type_mismatch"] = (
            f"{len(type_indices)} records have string values in numeric fields "
            f"(quantity, unit_price, or subtotal stored as text)"
        )

    if "missing_records" in planted_errors:
        n_drop = rng.randint(5, 15)
        if len(buggy_target) > n_drop + 10:
            drop_indices = sorted(rng.sample(range(len(buggy_target)), n_drop), reverse=True)
            dropped_txns = [buggy_target[i]["txn_id"] for i in drop_indices]
            for idx in drop_indices:
                buggy_target.pop(idx)
            error_details["missing_records"] = (
                f"{n_drop} source records dropped/missing from target data"
            )
        else:
            error_details["missing_records"] = "Some source records missing from target"

    if "incorrect_aggregation" in planted_errors:
        # Pick a few rows and mess up the quantity * unit_price = subtotal
        agg_indices = rng.sample(range(len(buggy_target)), min(rng.randint(5, 12), len(buggy_target)))
        for idx in agg_indices:
            row = buggy_target[idx]
            if isinstance(row["quantity"], int) and isinstance(row["unit_price"], (int, float)):
                # Make subtotal wrong by adding or subtracting a random amount
                offset = round(rng.uniform(5.0, 50.0), 2)
                row["subtotal"] = round(float(row["subtotal"]) + offset, 2)
        error_details["incorrect_aggregation"] = (
            f"{len(agg_indices)} records have subtotal not equal to quantity * unit_price"
        )

    if "timezone_conversion_error" in planted_errors:
        tz_indices = rng.sample(range(len(buggy_target)), min(rng.randint(10, 25), len(buggy_target)))
        wrong_offset = tz_hours + rng.choice([1, -1, 2])
        for idx in tz_indices:
            # Re-derive with wrong offset
            matching_src = [r for r in source_rows if r["txn_id"] == buggy_target[idx]["txn_id"]]
            if matching_src:
                s = matching_src[0]
                adj_hour = (s["hour"] + wrong_offset) % 24
                adj_day = s["day"]
                if s["hour"] + wrong_offset < 0:
                    adj_day = max(1, s["day"] - 1)
                elif s["hour"] + wrong_offset >= 24:
                    adj_day = min(28, s["day"] + 1)
                buggy_target[idx]["order_date_local"] = (
                    f"{s['year']}-{s['month']:02d}-{adj_day:02d} "
                    f"{adj_hour:02d}:{s['minute']:02d}:{s['second']:02d}"
                )
        error_details["timezone_conversion_error"] = (
            f"{len(tz_indices)} records have incorrect timezone conversion "
            f"(offset {wrong_offset:+d}h instead of {tz_hours:+d}h for {tz_src_label} to {tz_dst_label})"
        )

    if "encoding_corruption" in planted_errors:
        enc_indices = rng.sample(range(len(buggy_target)), min(rng.randint(3, 8), len(buggy_target)))
        corruption_chars = ["\xc3\xa9", "\xc3\xb6", "\xc3\xbc", "\xc3\xb1", "\xe2\x80\x99", "\xe2\x80\x93", "\xc3\xa8"]
        for idx in enc_indices:
            name = buggy_target[idx]["customer_name"]
            pos = rng.randint(0, max(0, len(name) - 2))
            repl = rng.choice(corruption_chars)
            buggy_target[idx]["customer_name"] = name[:pos] + repl + name[pos+1:]
        error_details["encoding_corruption"] = (
            f"{len(enc_indices)} records have corrupted special characters in customer_name "
            f"(encoding artifacts like '\xc3\xa9' instead of proper characters)"
        )

    # Shuffle buggy target to mix errors throughout
    rng.shuffle(buggy_target)

    # --- Build source_data.csv ---
    src_header = "txn_id,customer_id,customer_name,product_id,product_name,category,quantity,unit_price,subtotal,status,order_date"
    src_lines = [src_header]
    for row in source_rows:
        src_lines.append(
            f'{row["txn_id"]},{row["customer_id"]},"{row["customer_name"]}",'
            f'{row["product_id"]},"{row["product_name"]}",{row["category"]},'
            f'{row["quantity"]},{row["unit_price"]},{row["subtotal"]},'
            f'{row["status"]},{row["order_date"]}'
        )
    source_csv = "\n".join(src_lines) + "\n"

    # --- Build target_data.csv ---
    tgt_header = "txn_id,customer_id,customer_name,product_id,product_name,category,quantity,unit_price,subtotal,tax_amount,discount,total,order_date_local,status"
    tgt_lines = [tgt_header]
    for row in buggy_target:
        tgt_lines.append(
            f'{row["txn_id"]},{row["customer_id"]},"{row["customer_name"]}",'
            f'{row["product_id"]},"{row["product_name"]}",{row["category"]},'
            f'{row["quantity"]},{row["unit_price"]},{row["subtotal"]},'
            f'{row["tax_amount"]},{row["discount"]},{row["total"]},'
            f'{row["order_date_local"]},{row["status"]}'
        )
    target_csv = "\n".join(tgt_lines) + "\n"

    # --- Build transformation_rules.txt ---
    rules_lines = [
        f"ETL TRANSFORMATION RULES — {company}",
        "",
        "=" * 60,
        "1. DEDUPLICATION",
        "=" * 60,
        "Remove exact duplicate records based on txn_id. Each transaction",
        "should appear exactly once in the target data.",
        "",
        "=" * 60,
        "2. NULL / MISSING VALUE HANDLING",
        "=" * 60,
        "All null or missing numeric values should be replaced with 0.",
        "All null or missing string values should be replaced with empty string ''.",
        "Consistency is critical: the same treatment must apply to all records.",
        "",
        "=" * 60,
        "3. DATE/TIME CONVERSION",
        "=" * 60,
        f"Source dates are in {tz_src_label} timezone.",
        f"Target dates must be converted to {tz_dst_label} (offset: {tz_hours:+d} hours).",
        "Format: YYYY-MM-DD HH:MM:SS",
        "Date arithmetic must correctly handle day boundaries.",
        "",
        "=" * 60,
        "4. STRING TRUNCATION",
        "=" * 60,
        f"product_name field: truncate to {max_product_name_len} characters maximum.",
        "No other string fields should be truncated.",
        "",
        "=" * 60,
        "5. DERIVED FIELDS",
        "=" * 60,
        f"tax_amount = subtotal * {tax_rate} (tax rate: {tax_rate*100:.2f}%)",
        f"discount = subtotal * {discount_rate} IF subtotal >= {_fmt_money(discount_threshold)}, else 0",
        "total = subtotal + tax_amount - discount",
        "All monetary values rounded to 2 decimal places.",
        "",
        "=" * 60,
        "6. FILTER CRITERIA",
        "=" * 60,
        f"Only include records where status = '{status_filter}'.",
        f"Only include records where subtotal >= {_fmt_money(min_order_amount)}.",
        "Records not meeting both criteria must be excluded from target.",
        "",
        "=" * 60,
        "7. REFERENTIAL INTEGRITY",
        "=" * 60,
        "All customer_id values in target must exist in source_data.csv.",
        "All product_id values in target must exist in source_data.csv.",
        "Foreign key violations are not permitted.",
        "",
        "=" * 60,
        "8. DATA TYPE REQUIREMENTS",
        "=" * 60,
        "quantity: integer",
        "unit_price, subtotal, tax_amount, discount, total: numeric (float)",
        "txn_id, customer_id, product_id: integer",
        "All other fields: string",
        "",
    ]
    rules_content = "\n".join(rules_lines) + "\n"

    # --- Build data_dictionary.txt ---
    dict_lines = [
        "DATA DICTIONARY — TARGET TABLE",
        "",
        f"{'Field':<25} {'Type':<12} {'Description':<50} {'Valid Range / Constraint'}",
        f"{'-'*25} {'-'*12} {'-'*50} {'-'*30}",
        f"{'txn_id':<25} {'INTEGER':<12} {'Unique transaction identifier':<50} {'> 0, unique'}",
        f"{'customer_id':<25} {'INTEGER':<12} {'Customer FK (references source)':<50} {'must exist in source'}",
        f"{'customer_name':<25} {'STRING':<12} {'Customer full name':<50} {'non-empty'}",
        f"{'product_id':<25} {'INTEGER':<12} {'Product FK (references source)':<50} {'must exist in source'}",
        f"{'product_name':<25} {'STRING':<12} {'Product name (truncated)':<50} {'<= {max_pl} chars'.format(max_pl=max_product_name_len)}",
        f"{'category':<25} {'STRING':<12} {'Product category':<50} {'from reference list'}",
        f"{'quantity':<25} {'INTEGER':<12} {'Units ordered':<50} {'> 0'}",
        f"{'unit_price':<25} {'FLOAT':<12} {'Price per unit':<50} {'> 0'}",
        f"{'subtotal':<25} {'FLOAT':<12} {'quantity * unit_price':<50} {'> 0, = qty * price'}",
        f"{'tax_amount':<25} {'FLOAT':<12} {'Computed tax':<50} {'= subtotal * {tr}'.format(tr=tax_rate)}",
        f"{'discount':<25} {'FLOAT':<12} {'Computed discount':<50} {'0 or subtotal * {dr}'.format(dr=discount_rate)}",
        f"{'total':<25} {'FLOAT':<12} {'Final total':<50} {'= subtotal + tax - discount'}",
        f"{'order_date_local':<25} {'DATETIME':<12} {'Order date in {tz}'.format(tz=tz_dst_label):<50} {'YYYY-MM-DD HH:MM:SS'}",
        f"{'status':<25} {'STRING':<12} {'Order status':<50} {'must be {sf}'.format(sf=status_filter)}",
        "",
        "REFERENTIAL INTEGRITY CONSTRAINTS:",
        f"  customer_id must reference a valid customer_id in source_data.csv",
        f"  product_id must reference a valid product_id in source_data.csv",
        "",
        f"COMPUTED FIELD FORMULAS:",
        f"  tax_amount = ROUND(subtotal * {tax_rate}, 2)",
        f"  discount = IF subtotal >= {discount_threshold} THEN ROUND(subtotal * {discount_rate}, 2) ELSE 0",
        f"  total = ROUND(subtotal + tax_amount - discount, 2)",
        "",
    ]
    data_dict_content = "\n".join(dict_lines) + "\n"

    # --- Problem statement ---
    problem_statement = f"""# ETL Pipeline Validation

You are a data engineer at {company}. An ETL pipeline has been run to transform
raw transaction data into a target table for analytics. Your job is to validate
the pipeline's output against the source data and transformation rules.

## Source Files
- /testbed/data/source_data.csv — Raw source data ({n_source_rows} transaction records)
- /testbed/data/transformation_rules.txt — Business rules the ETL should follow
- /testbed/data/target_data.csv — ETL output (the pipeline's actual output, may contain errors)
- /testbed/data/data_dictionary.txt — Field definitions, types, valid ranges, and constraints

## Requirements
1. Compare target_data.csv against source_data.csv using the transformation rules
2. Check every transformation rule: deduplication, null handling, date conversion,
   string truncation, derived field calculations, filter criteria, referential
   integrity, and data types
3. Identify ALL discrepancies between the target data and what the rules specify
4. For each issue found, provide specific examples (row numbers, field values)
5. Quantify each issue (how many records affected)
6. Distinguish true ETL errors from acceptable variations

Write a detailed validation report to /testbed/validation_report.txt"""

    # --- Rubric ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does the file /testbed/validation_report.txt exist and contain substantive content (more than a few lines)?",
            points=1,
        ),
    ]

    # One binary check per planted error
    for err_key in planted_errors:
        label = err_key.replace("_", " ")
        detail = error_details.get(err_key, f"Issue of type: {label}")
        rubric_items.append(BinaryRubricCategory(
            name=f"identifies_{err_key}",
            question=(
                f"Does the report identify the '{label}' issue? "
                f"Ground truth: {detail}"
            ),
            points=2,
        ))

    # Specificity checks for the first 3 planted errors
    for err_key in planted_errors[:3]:
        label = err_key.replace("_", " ")
        detail = error_details.get(err_key, "")
        rubric_items.append(BinaryRubricCategory(
            name=f"quantifies_{err_key}",
            question=(
                f"Does the report provide a specific count or quantification for the "
                f"'{label}' issue? The report should indicate approximately how many "
                f"records are affected. Ground truth: {detail}"
            ),
            points=1,
        ))

    # False-positive checks
    rubric_items.extend([
        BinaryRubricCategory(
            name="no_false_positive_correct_filter",
            question=(
                f"Does the report avoid incorrectly flagging the ETL's filtering of "
                f"'{status_filter}'-only records as an error? The transformation rules "
                f"explicitly require filtering to status='{status_filter}'."
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="no_false_positive_truncation_limit",
            question=(
                f"Does the report avoid flagging product names that are correctly truncated "
                f"to {max_product_name_len} characters as an error? Only names truncated to "
                f"a DIFFERENT length than specified should be flagged."
            ),
            points=1,
        ),
    ])

    # False-positive check: legitimate tax rounding
    rubric_items.append(BinaryRubricCategory(
        name="no_false_positive_rounding",
        question=(
            "Does the report avoid flagging minor floating-point rounding differences "
            "(e.g., $0.01 discrepancy due to IEEE 754 rounding) as an ETL error? "
            "Small rounding differences in the last decimal place are expected behavior."
        ),
        points=1,
    ))

    # Structural checks
    rubric_items.extend([
        BinaryRubricCategory(
            name="reports_source_row_count",
            question=f"Does the report mention the source data has {n_source_rows} records (approximately, within 5%)?",
            points=1,
        ),
        BinaryRubricCategory(
            name="reports_target_row_count",
            question=f"Does the report mention the target data has approximately {len(buggy_target)} records (within 10%)?",
            points=1,
        ),
        BinaryRubricCategory(
            name="checks_all_rule_categories",
            question=(
                "Does the report check at least 5 of the 8 rule categories "
                "(dedup, null handling, date conversion, string truncation, derived fields, "
                "filter criteria, referential integrity, data types)?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="identifies_correct_filter_status",
            question=(
                f"Does the report correctly identify that the filter criterion is "
                f"status = '{status_filter}' (as specified in the transformation rules)?"
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="identifies_tax_rate",
            question=(
                f"Does the report reference the correct tax rate of "
                f"{tax_rate*100:.2f}% when checking derived fields?"
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="identifies_timezone_rule",
            question=(
                f"Does the report reference the timezone conversion rule "
                f"({tz_src_label} to {tz_dst_label}, offset {tz_hours:+d}h) "
                f"when checking date/time fields?"
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="provides_example_records",
            question=(
                "Does the report provide at least 3 specific transaction IDs or row "
                "references as examples of records with issues?"
            ),
            points=1,
        ),
    ])

    # Report quality (graded — the non-binary portion)
    rubric_items.append(RubricCategory(
        name="report_quality",
        description="Is the validation report well-organized with clear sections and specific examples?",
        failure="Disorganized or vague output without specific record-level examples.",
        minor_failure="Some structure but missing specificity (no row numbers or example values).",
        minor_success="Organized with identifiable sections and some specific examples per issue.",
        success="Professional report with clear sections per rule category, specific record examples, counts, and a summary of overall data quality.",
        points=2,
    ))

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=tuple(rubric_items),
        submission_instructions="Write your completed validation report to /testbed/validation_report.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/source_data.csv": source_csv,
            "/testbed/data/target_data.csv": target_csv,
            "/testbed/data/transformation_rules.txt": rules_content,
            "/testbed/data/data_dictionary.txt": data_dict_content,
        },
        problem_type="etl_pipeline_validation",
    )


# =============================================================================
# 2. SCHEMA MIGRATION REVIEW
# =============================================================================

_TABLE_PREFIXES = [
    "users", "orders", "products", "payments", "sessions",
    "audit_log", "notifications", "inventory", "categories", "reviews",
    "shipping", "addresses", "coupons", "cart_items", "wishlists",
]

_COLUMN_TYPES = {
    "id": "BIGINT PRIMARY KEY AUTO_INCREMENT",
    "name": "VARCHAR(255) NOT NULL",
    "email": "VARCHAR(255) NOT NULL UNIQUE",
    "created_at": "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP",
    "updated_at": "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP",
    "status": "VARCHAR(50) NOT NULL DEFAULT 'active'",
    "amount": "DECIMAL(12,2) NOT NULL",
    "quantity": "INT NOT NULL DEFAULT 0",
    "description": "TEXT",
    "is_active": "BOOLEAN NOT NULL DEFAULT TRUE",
    "price": "DECIMAL(10,2) NOT NULL",
    "discount_pct": "DECIMAL(5,2) DEFAULT 0.00",
    "total": "DECIMAL(12,2) NOT NULL",
    "user_id": "BIGINT NOT NULL",
    "product_id": "BIGINT NOT NULL",
    "order_id": "BIGINT NOT NULL",
    "category_id": "BIGINT",
    "address_id": "BIGINT",
    "phone": "VARCHAR(20)",
    "zip_code": "VARCHAR(10)",
    "country_code": "CHAR(2) NOT NULL DEFAULT 'US'",
    "rating": "SMALLINT CHECK (rating BETWEEN 1 AND 5)",
    "notes": "TEXT",
}

_MIGRATION_ISSUE_TYPES = [
    "drops_used_column",
    "not_null_without_default",
    "wrong_index_order",
    "missing_foreign_key",
    "precision_loss",
    "missing_rollback_step",
    "renamed_column_not_updated",
    "unique_constraint_violation",
    "missing_index_for_join",
]


def make_schema_migration_review(rand_seed: int = 42) -> RubricDatapoint:
    """Review a database schema migration for correctness and risk. The
    migration script contains planted issues the agent must discover.

    Seed varies: table count, table names, which columns exist, which
    migration issues are planted, query patterns, rollback plan gaps.
    """
    rng = _random.Random(rand_seed)

    company = pick1(COMPANY_NAMES, rand_seed)
    author_name = random_name(rand_seed + 10)
    reviewer_name = random_name(rand_seed + 20)

    # Pick 8-15 tables
    n_tables = rng.randint(8, 15)
    table_names = rng.sample(_TABLE_PREFIXES, min(n_tables, len(_TABLE_PREFIXES)))
    if n_tables > len(_TABLE_PREFIXES):
        extra = [f"tbl_{rng.randint(100,999)}" for _ in range(n_tables - len(_TABLE_PREFIXES))]
        table_names.extend(extra)

    # Build schema: each table has 4-8 columns
    schema_tables: dict[str, list[tuple[str, str]]] = {}
    for tbl in table_names:
        n_cols = rng.randint(4, 8)
        cols = [("id", _COLUMN_TYPES["id"])]
        available_cols = [k for k in _COLUMN_TYPES if k != "id"]
        rng.shuffle(available_cols)
        for col_name in available_cols[:n_cols - 1]:
            cols.append((col_name, _COLUMN_TYPES[col_name]))
        # Add FK references where appropriate
        if tbl not in ("users", "categories") and "user_id" not in [c[0] for c in cols]:
            if rng.random() < 0.5 and "users" in table_names:
                cols.append(("user_id", "BIGINT NOT NULL"))
        schema_tables[tbl] = cols

    # Build query patterns (most common production queries)
    query_patterns = []
    for tbl in table_names[:6]:
        cols = schema_tables[tbl]
        col_names = [c[0] for c in cols]
        # SELECT queries
        where_col = rng.choice([c for c in col_names if c != "id"]) if len(col_names) > 1 else "id"
        query_patterns.append({
            "query": f"SELECT * FROM {tbl} WHERE {where_col} = ? ORDER BY created_at DESC LIMIT 50",
            "frequency": f"{rng.randint(500, 5000)}/hour",
            "latency_p95": f"{rng.randint(5, 50)}ms",
        })
        # JOIN queries
        if "user_id" in col_names and "users" in table_names:
            query_patterns.append({
                "query": f"SELECT u.name, t.* FROM {tbl} t JOIN users u ON t.user_id = u.id WHERE t.status = ? ORDER BY t.created_at DESC",
                "frequency": f"{rng.randint(100, 2000)}/hour",
                "latency_p95": f"{rng.randint(10, 80)}ms",
            })

    # Pick 4-7 migration issues to plant
    n_issues = rng.randint(4, 7)
    available_issues = list(_MIGRATION_ISSUE_TYPES)
    rng.shuffle(available_issues)
    planted_issues = available_issues[:n_issues]

    # --- Build current_schema.sql ---
    schema_lines = [
        f"-- Database Schema: {company}",
        f"-- Generated: 2024-08-15",
        f"-- Engine: MySQL 8.0",
        "",
    ]
    for tbl, cols in schema_tables.items():
        schema_lines.append(f"CREATE TABLE {tbl} (")
        col_defs = []
        fk_defs = []
        for col_name, col_type in cols:
            col_defs.append(f"    {col_name} {col_type}")
            if col_name.endswith("_id") and col_name != "id":
                ref_table = col_name.replace("_id", "") + "s"
                if ref_table in table_names:
                    fk_defs.append(f"    FOREIGN KEY ({col_name}) REFERENCES {ref_table}(id)")
        all_defs = col_defs + fk_defs
        schema_lines.append(",\n".join(all_defs))
        schema_lines.append(");")
        schema_lines.append("")
        # Add some indexes
        non_id_cols = [c[0] for c in cols if c[0] != "id"]
        if non_id_cols:
            idx_col = rng.choice(non_id_cols)
            schema_lines.append(f"CREATE INDEX idx_{tbl}_{idx_col} ON {tbl}({idx_col});")
            if "created_at" in [c[0] for c in cols]:
                schema_lines.append(f"CREATE INDEX idx_{tbl}_created_at ON {tbl}(created_at);")
        schema_lines.append("")

    # Add views that reference specific columns (important for renamed_column_not_updated)
    view_tables = rng.sample(table_names[:6], min(2, len(table_names)))
    views: list[dict] = []
    for vt in view_tables:
        cols = schema_tables[vt]
        col_names = [c[0] for c in cols]
        select_cols = rng.sample(col_names, min(4, len(col_names)))
        view_name = f"v_{vt}_summary"
        schema_lines.append(f"CREATE VIEW {view_name} AS")
        schema_lines.append(f"    SELECT {', '.join(select_cols)}")
        schema_lines.append(f"    FROM {vt}")
        if "is_active" in col_names:
            schema_lines.append(f"    WHERE is_active = TRUE;")
        else:
            schema_lines.append(f"    WHERE 1=1;")
        schema_lines.append("")
        views.append({"name": view_name, "table": vt, "columns": select_cols})

    current_schema = "\n".join(schema_lines) + "\n"

    # --- Plan the migration operations and plant issues ---
    migration_ops: list[str] = []
    issue_details: dict[str, str] = {}
    new_table_name = f"analytics_events"
    new_table_cols = [
        ("id", "BIGINT PRIMARY KEY AUTO_INCREMENT"),
        ("event_type", "VARCHAR(100) NOT NULL"),
        ("event_data", "JSON"),
        ("user_id", "BIGINT NOT NULL"),
        ("created_at", "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP"),
    ]

    # Always add a new table creation
    migration_ops.append(f"-- Step 1: Create analytics_events table")
    create_lines = [f"CREATE TABLE {new_table_name} ("]
    for i, (cn, ct) in enumerate(new_table_cols):
        comma = "," if i < len(new_table_cols) - 1 else ""
        create_lines.append(f"    {cn} {ct}{comma}")
    create_lines.append(");")
    migration_ops.append("\n".join(create_lines))
    migration_ops.append("")

    step_num = 2

    # Plant: drops_used_column
    dropped_col = None
    dropped_table = None
    if "drops_used_column" in planted_issues:
        # Find a column referenced in a query pattern
        for qp in query_patterns:
            q = qp["query"]
            for tbl in table_names:
                if tbl in q:
                    cols = schema_tables[tbl]
                    col_names = [c[0] for c in cols if c[0] != "id"]
                    for cn in col_names:
                        if cn in q and cn not in ("created_at",):
                            dropped_col = cn
                            dropped_table = tbl
                            break
                    if dropped_col:
                        break
            if dropped_col:
                break
        if dropped_col and dropped_table:
            migration_ops.append(f"-- Step {step_num}: Remove deprecated column")
            migration_ops.append(f"ALTER TABLE {dropped_table} DROP COLUMN {dropped_col};")
            migration_ops.append("")
            step_num += 1
            # Find the query that uses it
            using_query = next((qp["query"] for qp in query_patterns if dropped_col in qp["query"]), "unknown")
            issue_details["drops_used_column"] = (
                f"Migration drops column '{dropped_col}' from table '{dropped_table}' "
                f"but it is used by production query: {using_query}"
            )

    # Plant: not_null_without_default
    notnull_col = None
    notnull_table = None
    if "not_null_without_default" in planted_issues:
        target_tbl = rng.choice(table_names)
        notnull_col = "region_code"
        notnull_table = target_tbl
        migration_ops.append(f"-- Step {step_num}: Add region tracking")
        migration_ops.append(f"ALTER TABLE {target_tbl} ADD COLUMN region_code VARCHAR(10) NOT NULL;")
        migration_ops.append("")
        step_num += 1
        issue_details["not_null_without_default"] = (
            f"Adding NOT NULL column 'region_code' to '{target_tbl}' without DEFAULT value "
            f"will fail on existing rows"
        )

    # Plant: wrong_index_order
    if "wrong_index_order" in planted_issues:
        # Find a query with WHERE + ORDER BY on different columns
        idx_tbl = None
        idx_cols = None
        for qp in query_patterns:
            q = qp["query"]
            if "WHERE" in q and "ORDER BY" in q:
                for tbl in table_names:
                    if tbl in q:
                        idx_tbl = tbl
                        # The correct index order should be: WHERE col first, then ORDER BY col
                        cols = schema_tables[tbl]
                        col_names = [c[0] for c in cols if c[0] != "id"]
                        where_col = None
                        order_col = None
                        for cn in col_names:
                            if f"WHERE {cn}" in q or f"WHERE t.{cn}" in q:
                                where_col = cn
                            if f"ORDER BY {cn}" in q or f"ORDER BY t.{cn}" in q:
                                order_col = cn
                        if where_col and order_col and where_col != order_col:
                            idx_cols = (where_col, order_col)
                            break
                if idx_cols:
                    break
        if idx_tbl and idx_cols:
            # Plant: index with wrong order (ORDER BY col first, WHERE col second)
            migration_ops.append(f"-- Step {step_num}: Add composite index for query optimization")
            migration_ops.append(f"CREATE INDEX idx_{idx_tbl}_composite ON {idx_tbl}({idx_cols[1]}, {idx_cols[0]});")
            migration_ops.append("")
            step_num += 1
            issue_details["wrong_index_order"] = (
                f"Composite index on {idx_tbl}({idx_cols[1]}, {idx_cols[0]}) has wrong column order; "
                f"query filters on {idx_cols[0]} first so index should be ({idx_cols[0]}, {idx_cols[1]})"
            )

    # Plant: missing_foreign_key
    if "missing_foreign_key" in planted_issues and "users" in table_names:
        migration_ops.append(f"-- Step {step_num}: Create {new_table_name} indexes")
        migration_ops.append(f"CREATE INDEX idx_{new_table_name}_user ON {new_table_name}(user_id);")
        migration_ops.append(f"CREATE INDEX idx_{new_table_name}_type ON {new_table_name}(event_type);")
        migration_ops.append("")
        step_num += 1
        issue_details["missing_foreign_key"] = (
            f"New table '{new_table_name}' has user_id column but no FOREIGN KEY constraint "
            f"referencing users(id)"
        )

    # Plant: precision_loss
    if "precision_loss" in planted_issues:
        prec_tbl = None
        prec_col = None
        for tbl in table_names:
            for cn, ct in schema_tables[tbl]:
                if "DECIMAL" in ct:
                    prec_tbl = tbl
                    prec_col = cn
                    break
            if prec_col:
                break
        if prec_tbl and prec_col:
            migration_ops.append(f"-- Step {step_num}: Optimize storage for {prec_col}")
            migration_ops.append(f"ALTER TABLE {prec_tbl} MODIFY COLUMN {prec_col} INT NOT NULL;")
            migration_ops.append("")
            step_num += 1
            issue_details["precision_loss"] = (
                f"Changing '{prec_col}' in '{prec_tbl}' from DECIMAL to INT loses decimal precision "
                f"for monetary/fractional values"
            )

    # Plant: missing_rollback_step
    if "missing_rollback_step" in planted_issues:
        # We'll note this as a gap in the migration plan
        issue_details["missing_rollback_step"] = (
            f"Migration plan has no rollback step for the data migration / column type change "
            f"which is irreversible once committed"
        )

    # Plant: renamed_column_not_updated
    renamed_col = None
    renamed_table = None
    if "renamed_column_not_updated" in planted_issues and views:
        # Pick a column that appears in a view
        for v in views:
            for vc in v["columns"]:
                if vc not in ("id", "created_at"):
                    renamed_col = vc
                    renamed_table = v["table"]
                    break
            if renamed_col:
                break
        if renamed_col and renamed_table:
            new_col_name = f"{renamed_col}_v2"
            migration_ops.append(f"-- Step {step_num}: Rename column for clarity")
            migration_ops.append(f"ALTER TABLE {renamed_table} RENAME COLUMN {renamed_col} TO {new_col_name};")
            migration_ops.append("")
            step_num += 1
            view_name = next((v["name"] for v in views if v["table"] == renamed_table), "unknown_view")
            issue_details["renamed_column_not_updated"] = (
                f"Column '{renamed_col}' renamed to '{new_col_name}' in '{renamed_table}' "
                f"but view '{view_name}' still references the old column name"
            )

    # Plant: unique_constraint_violation
    if "unique_constraint_violation" in planted_issues:
        uniq_tbl = rng.choice(table_names)
        uniq_cols = schema_tables[uniq_tbl]
        nullable_cols = [c[0] for c in uniq_cols if "NOT NULL" not in c[1] and c[0] != "id"]
        if nullable_cols:
            uniq_col = rng.choice(nullable_cols)
        else:
            uniq_col = rng.choice([c[0] for c in uniq_cols if c[0] != "id"])
        migration_ops.append(f"-- Step {step_num}: Add unique constraint for data integrity")
        migration_ops.append(f"ALTER TABLE {uniq_tbl} ADD CONSTRAINT uq_{uniq_tbl}_{uniq_col} UNIQUE ({uniq_col});")
        migration_ops.append("")
        step_num += 1
        issue_details["unique_constraint_violation"] = (
            f"Adding UNIQUE constraint on '{uniq_tbl}.{uniq_col}' will likely fail "
            f"because existing data may contain duplicate values in this column "
            f"(column allows NULLs and has no prior uniqueness guarantee)"
        )

    # Plant: missing_index_for_join
    if "missing_index_for_join" in planted_issues and new_table_name:
        # The new table has user_id but no index on event_type+created_at
        # for a common analytical query
        analytical_query = (
            f"SELECT event_type, COUNT(*) FROM {new_table_name} "
            f"WHERE created_at >= ? AND created_at < ? GROUP BY event_type"
        )
        query_patterns.append({
            "query": analytical_query,
            "frequency": f"{rng.randint(50, 200)}/hour",
            "latency_p95": "estimated 500ms+",
        })
        issue_details["missing_index_for_join"] = (
            f"New analytical query on {new_table_name} filters by created_at and groups by "
            f"event_type but no composite index on (created_at, event_type) is created"
        )

    # Add a benign migration step (data backfill)
    migration_ops.append(f"-- Step {step_num}: Backfill analytics events from existing data")
    migration_ops.append(f"INSERT INTO {new_table_name} (event_type, event_data, user_id, created_at)")
    if "orders" in table_names:
        migration_ops.append(f"    SELECT 'order_placed', JSON_OBJECT('order_id', id), user_id, created_at")
        migration_ops.append(f"    FROM orders WHERE created_at >= '2024-01-01';")
    else:
        tbl0 = table_names[0]
        migration_ops.append(f"    SELECT 'record_created', JSON_OBJECT('id', id), COALESCE(user_id, 0), created_at")
        migration_ops.append(f"    FROM {tbl0} WHERE created_at >= '2024-01-01';")
    migration_ops.append("")

    migration_content = "\n".join(migration_ops) + "\n"

    # --- Build migration_plan.txt ---
    plan_lines = [
        f"MIGRATION PLAN — {company}",
        f"Author: {author_name}",
        f"Date: 2024-09-01",
        f"Reviewer: {reviewer_name}",
        "",
        "=" * 60,
        "OBJECTIVE",
        "=" * 60,
        f"Add analytics event tracking table, optimize existing schema,",
        f"and improve data integrity constraints.",
        "",
        "=" * 60,
        "ESTIMATED DOWNTIME",
        "=" * 60,
        f"Maintenance window: 2 hours",
        f"Expected execution time: 30-45 minutes",
        f"Buffer for rollback: 45 minutes",
        "",
        "=" * 60,
        "DEPENDENCY ANALYSIS",
        "=" * 60,
        f"- New table '{new_table_name}' depends on 'users' table",
        f"- Column modifications affect existing application code",
        f"- Views may need updating after column renames",
        f"- Existing indexes should be reviewed for new query patterns",
        "",
        "=" * 60,
        "ROLLBACK PLAN",
        "=" * 60,
        f"Step 1: DROP TABLE IF EXISTS {new_table_name};",
    ]
    if dropped_col and dropped_table:
        plan_lines.append(f"Step 2: ALTER TABLE {dropped_table} ADD COLUMN {dropped_col} <restore type>;")
    if notnull_col and notnull_table:
        plan_lines.append(f"Step 3: ALTER TABLE {notnull_table} DROP COLUMN {notnull_col};")
    # Intentionally missing rollback for precision_loss if planted
    if "missing_rollback_step" in planted_issues:
        plan_lines.append("  (Note: some steps may not be fully reversible)")
    else:
        plan_lines.append("Step 4: Restore column types from backup schema snapshot")

    plan_lines.extend([
        "",
        "=" * 60,
        "PRE-MIGRATION CHECKLIST",
        "=" * 60,
        "[ ] Full database backup completed",
        "[ ] Application servers in maintenance mode",
        "[ ] Replica lag < 1 second",
        "[ ] Migration script tested on staging",
        "[ ] Rollback script tested on staging",
        "",
    ])
    plan_content = "\n".join(plan_lines) + "\n"

    # --- Build query_patterns.csv ---
    qp_lines = ["query,frequency,latency_p95"]
    for qp in query_patterns:
        q_escaped = qp["query"].replace('"', '""')
        qp_lines.append(f'"{q_escaped}",{qp["frequency"]},{qp["latency_p95"]}')
    qp_content = "\n".join(qp_lines) + "\n"

    # --- Problem statement ---
    problem_statement = f"""# Database Schema Migration Review

You are a senior database engineer at {company}. {author_name} has proposed a
schema migration and you need to review it for correctness, risk, and impact on
production queries.

## Source Files
- /testbed/data/current_schema.sql — Current database schema ({n_tables} tables with constraints)
- /testbed/data/migration_script.sql — Proposed migration with ALTER TABLE, CREATE TABLE, data migrations
- /testbed/data/migration_plan.txt — Rollback plan, dependency analysis, estimated downtime
- /testbed/data/query_patterns.csv — Most common production queries that must continue working

## Requirements
1. Review each migration step for correctness
2. Cross-reference changes against current schema and production query patterns
3. Verify rollback plan covers all changes
4. Identify columns/tables affected by the migration that are used in production queries
5. Check for missing indexes, foreign keys, and constraints
6. Identify potential data integrity risks (precision loss, NOT NULL without default, etc.)
7. Verify views and triggers are updated for any column renames
8. Assess overall migration risk and provide recommendations

Write a detailed review to /testbed/migration_review.txt"""

    # --- Rubric ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does the file /testbed/migration_review.txt exist and contain substantive content (more than a few lines)?",
            points=1,
        ),
    ]

    # One binary check per planted issue
    for iss_key in planted_issues:
        label = iss_key.replace("_", " ")
        detail = issue_details.get(iss_key, f"Issue of type: {label}")
        rubric_items.append(BinaryRubricCategory(
            name=f"identifies_{iss_key}",
            question=(
                f"Does the review identify the '{label}' issue? "
                f"Ground truth: {detail}"
            ),
            points=2,
        ))

    # Specificity for top issues
    for iss_key in planted_issues[:2]:
        label = iss_key.replace("_", " ")
        detail = issue_details.get(iss_key, "")
        rubric_items.append(BinaryRubricCategory(
            name=f"specific_{iss_key}",
            question=(
                f"Does the review provide specific details for the '{label}' issue, "
                f"including the affected table/column names and why it is problematic? "
                f"Ground truth: {detail}"
            ),
            points=1,
        ))

    # False-positive checks
    rubric_items.extend([
        BinaryRubricCategory(
            name="no_false_positive_new_table",
            question=(
                f"Does the review avoid incorrectly flagging the creation of the "
                f"'{new_table_name}' table itself as an issue? The table creation is "
                f"correct and follows standard patterns."
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="no_false_positive_backfill",
            question=(
                f"Does the review avoid incorrectly flagging the data backfill "
                f"INSERT INTO {new_table_name} as fundamentally wrong? The backfill "
                f"query is syntactically correct and follows standard practices."
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="no_false_positive_existing_indexes",
            question=(
                "Does the review avoid incorrectly flagging existing indexes in the "
                "current schema as migration issues? Only new or modified indexes "
                "should be evaluated."
            ),
            points=1,
        ),
    ])

    # Structural
    rubric_items.extend([
        BinaryRubricCategory(
            name="references_query_patterns",
            question=(
                "Does the review reference at least one specific production query from "
                "query_patterns.csv when discussing impact of migration changes?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="evaluates_rollback_plan",
            question="Does the review evaluate the rollback plan and identify any gaps or missing steps?",
            points=2,
        ),
        BinaryRubricCategory(
            name="provides_recommendations",
            question="Does the review provide specific recommendations for fixing the identified issues?",
            points=1,
        ),
        BinaryRubricCategory(
            name="identifies_table_count",
            question=(
                f"Does the review correctly identify the schema as having approximately "
                f"{n_tables} tables?"
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="reviews_new_table_design",
            question=(
                f"Does the review evaluate the design of the new '{new_table_name}' table "
                f"(e.g., column types, indexes, constraints)?"
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="assesses_downtime_risk",
            question=(
                "Does the review comment on the estimated downtime or maintenance window, "
                "considering the number and type of migration operations?"
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="checks_view_dependencies",
            question=(
                "Does the review check whether any database views are affected by the "
                "migration changes (column renames, drops, or type changes)?"
            ),
            points=2,
        ),
    ])

    rubric_items.append(RubricCategory(
        name="review_quality",
        description="Is the migration review thorough, well-organized, and professionally written?",
        failure="Disorganized, missing major issues, or vague without specifics.",
        minor_failure="Some structure but misses important issues or lacks specific references.",
        minor_success="Well-organized with most issues identified and specific references to schema/queries.",
        success="Comprehensive review covering all migration steps, cross-referencing queries, with risk assessment and actionable recommendations.",
        points=2,
    ))

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=tuple(rubric_items),
        submission_instructions="Write your completed migration review to /testbed/migration_review.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/current_schema.sql": current_schema,
            "/testbed/data/migration_script.sql": migration_content,
            "/testbed/data/migration_plan.txt": plan_content,
            "/testbed/data/query_patterns.csv": qp_content,
        },
        problem_type="schema_migration_review",
    )


# =============================================================================
# 3. DATA QUALITY AUDIT
# =============================================================================

_DQ_ISSUE_TYPES = [
    "invalid_email",
    "wrong_phone_digits",
    "zip_state_mismatch",
    "future_date",
    "negative_age",
    "fuzzy_duplicates",
    "orphan_records",
    "impossible_date_combo",
    "out_of_range_values",
    "inconsistent_date_format",
    "missing_mandatory",
    "invalid_reference_lookup",
    "statistical_outlier",
    "stale_data",
    "encoding_artifacts",
]

# State -> zip code prefix mapping (first 3 digits)
_STATE_ZIP_PREFIXES: dict[str, list[str]] = {
    "CA": ["900", "901", "902", "903", "904", "905", "906", "907", "908", "910", "911", "912", "913", "914", "915", "916", "917", "918", "919", "920", "921", "922", "923", "924", "925", "926", "927", "928", "930", "931", "932", "933", "934", "935", "936", "937", "938", "939", "940", "941", "942", "943", "944", "945", "946", "947", "948", "949", "950", "951", "952", "953", "954", "955", "956", "957", "958", "959", "960", "961"],
    "NY": ["100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110", "111", "112", "113", "114", "115", "116", "117", "118", "119", "120", "121", "122", "123", "124", "125", "126", "127", "128", "129", "130", "131", "132", "133", "134", "135", "136", "137", "138", "139", "140", "141", "142", "143", "144", "145", "146", "147", "148", "149"],
    "TX": ["750", "751", "752", "753", "754", "755", "756", "757", "758", "759", "760", "761", "762", "763", "764", "765", "766", "767", "768", "769", "770", "771", "772", "773", "774", "775", "776", "777", "778", "779", "780", "781", "782", "783", "784", "785", "786", "787", "788", "789", "790", "791", "792", "793", "794", "795", "796", "797", "798", "799"],
    "FL": ["320", "321", "322", "323", "324", "325", "326", "327", "328", "329", "330", "331", "332", "333", "334", "335", "336", "337", "338", "339", "340", "341", "342", "343", "344", "345", "346", "347", "348", "349"],
    "IL": ["600", "601", "602", "603", "604", "605", "606", "607", "608", "609", "610", "611", "612", "613", "614", "615", "616", "617", "618", "619", "620", "621", "622", "623", "624", "625", "626", "627", "628", "629"],
    "PA": ["150", "151", "152", "153", "154", "155", "156", "157", "158", "159", "160", "161", "162", "163", "164", "165", "166", "167", "168", "169", "170", "171", "172", "173", "174", "175", "176", "177", "178", "179", "180", "181", "182", "183", "184", "185", "186", "187", "188", "189", "190", "191", "192", "193", "194", "195", "196"],
    "OH": ["430", "431", "432", "433", "434", "435", "436", "437", "438", "439", "440", "441", "442", "443", "444", "445", "446", "447", "448", "449", "450", "451", "452", "453", "454", "455", "456", "457", "458"],
    "WA": ["980", "981", "982", "983", "984", "985", "986", "988", "989", "990", "991", "992", "993", "994"],
    "CO": ["800", "801", "802", "803", "804", "805", "806", "807", "808", "809", "810", "811", "812", "813", "814", "815", "816"],
    "GA": ["300", "301", "302", "303", "304", "305", "306", "307", "308", "309", "310", "311", "312", "313", "314", "315", "316", "317", "318", "319"],
}

_VALID_COUNTRIES = ["US", "CA", "MX", "GB", "DE", "FR", "JP", "AU", "BR", "IN"]

_PRODUCT_CATEGORIES_REF = [
    "Electronics", "Apparel", "Home & Garden", "Sporting Goods",
    "Automotive", "Health & Beauty", "Toys & Games", "Office Supplies",
    "Food & Beverage", "Books & Media",
]


def make_data_quality_audit(rand_seed: int = 42) -> RubricDatapoint:
    """Audit a customer dataset for data quality issues against business rules.
    The dataset contains planted quality issues from a pool of 15 types.

    Seed varies: number of customers, which issues are planted, severity of
    issues, reference data, SLA thresholds, false-positive traps.
    """
    rng = _random.Random(rand_seed)

    company = pick1(COMPANY_NAMES, rand_seed)

    # --- Scenario parameters ---
    n_customers = rng.randint(200, 500)
    n_issues = rng.randint(6, 10)
    available_issues = list(_DQ_ISSUE_TYPES)
    rng.shuffle(available_issues)
    planted_issues = available_issues[:n_issues]

    # SLA targets (vary by seed)
    sla_email_validity = rng.choice([0.95, 0.96, 0.97, 0.98])
    sla_address_completeness = rng.choice([0.93, 0.95, 0.97])
    sla_phone_validity = rng.choice([0.94, 0.96, 0.98])
    sla_duplicate_rate = rng.choice([0.01, 0.02, 0.03])
    sla_freshness_days = rng.choice([365, 548, 730])

    # States to use (pick a subset for the dataset)
    states_with_zips = list(_STATE_ZIP_PREFIXES.keys())
    active_states = rng.sample(states_with_zips, min(6, len(states_with_zips)))

    # Generate base customer records
    names = random_names(rand_seed + 200, n_customers)
    customers: list[dict] = []

    for i, full_name in enumerate(names):
        cust_id = 2001 + i
        parts = full_name.split()
        first, last = parts[0], parts[-1]
        state = rng.choice(active_states)
        prefix = rng.choice(_STATE_ZIP_PREFIXES[state])
        zip_code = f"{prefix}{rng.randint(10, 99)}"
        email = _make_email(rng, first, last)
        phone = _make_phone(rng)
        birth_year = rng.randint(1940, 2006)
        birth_month = rng.randint(1, 12)
        birth_day = rng.randint(1, 28)
        birth_date = f"{birth_year}-{birth_month:02d}-{birth_day:02d}"
        age = 2024 - birth_year
        enroll_year = rng.randint(max(2015, birth_year + 18), 2024)
        enroll_month = rng.randint(1, 12)
        enroll_day = rng.randint(1, 28)
        enrollment_date = f"{enroll_year}-{enroll_month:02d}-{enroll_day:02d}"
        country = "US"
        category = rng.choice(_PRODUCT_CATEGORIES_REF)
        income = rng.randint(20000, 250000)
        last_updated_year = rng.choice([2023, 2024, 2024, 2024])
        last_updated_month = rng.randint(1, 12)
        last_updated_day = rng.randint(1, 28)
        last_updated = f"{last_updated_year}-{last_updated_month:02d}-{last_updated_day:02d}"
        street_num = rng.randint(1, 9999)
        street = rng.choice(_STREET_NAMES)
        city = rng.choice(["Springfield", "Portland", "Charlotte", "Denver", "Austin",
                           "Raleigh", "Boise", "Tucson", "Columbus", "Tampa"])
        address = f"{street_num} {street}"

        customers.append({
            "customer_id": cust_id,
            "first_name": first,
            "last_name": last,
            "email": email,
            "phone": phone,
            "address": address,
            "city": city,
            "state": state,
            "zip_code": zip_code,
            "country": country,
            "birth_date": birth_date,
            "age": age,
            "enrollment_date": enrollment_date,
            "preferred_category": category,
            "annual_income": income,
            "last_updated": last_updated,
        })

    # Track ground truth
    issue_details: dict[str, str] = {}

    # --- False-positive traps (plant these FIRST so they're in the data) ---
    # Trap 1: A legitimately old person (born 1920, age ~104)
    old_person_idx = rng.randint(0, len(customers) - 1)
    customers[old_person_idx]["birth_date"] = "1920-03-15"
    customers[old_person_idx]["age"] = 104
    customers[old_person_idx]["enrollment_date"] = "2020-06-01"  # Enrolled recently
    old_person_name = f"{customers[old_person_idx]['first_name']} {customers[old_person_idx]['last_name']}"

    # Trap 2: A valid but unusual zip code (APO/FPO military)
    mil_idx = rng.randint(0, len(customers) - 1)
    while mil_idx == old_person_idx:
        mil_idx = rng.randint(0, len(customers) - 1)
    customers[mil_idx]["zip_code"] = "09021"
    customers[mil_idx]["state"] = "NY"  # APO addresses use NY
    customers[mil_idx]["city"] = "APO"
    mil_person_name = f"{customers[mil_idx]['first_name']} {customers[mil_idx]['last_name']}"

    # Trap 3: High but legitimate income
    high_income_idx = rng.randint(0, len(customers) - 1)
    while high_income_idx in (old_person_idx, mil_idx):
        high_income_idx = rng.randint(0, len(customers) - 1)
    customers[high_income_idx]["annual_income"] = 850000  # High but plausible for executive

    # --- Plant quality issues ---
    used_indices: set[int] = {old_person_idx, mil_idx, high_income_idx}

    def _pick_indices(n: int) -> list[int]:
        available = [i for i in range(len(customers)) if i not in used_indices]
        picked = rng.sample(available, min(n, len(available)))
        used_indices.update(picked)
        return picked

    if "invalid_email" in planted_issues:
        n_bad = rng.randint(8, 20)
        indices = _pick_indices(n_bad)
        bad_patterns = ["missing_at", "double_dot", "no_domain", "space_in", "no_tld"]
        for idx in indices:
            pattern = rng.choice(bad_patterns)
            name = customers[idx]["first_name"].lower()
            if pattern == "missing_at":
                customers[idx]["email"] = f"{name}gmail.com"
            elif pattern == "double_dot":
                customers[idx]["email"] = f"{name}@gmail..com"
            elif pattern == "no_domain":
                customers[idx]["email"] = f"{name}@"
            elif pattern == "space_in":
                customers[idx]["email"] = f"{name} {customers[idx]['last_name'].lower()}@gmail.com"
            elif pattern == "no_tld":
                customers[idx]["email"] = f"{name}@gmail"
        n_valid_emails = n_customers - len(indices)
        email_rate = round(n_valid_emails / n_customers, 4)
        issue_details["invalid_email"] = (
            f"{len(indices)} records have invalid email formats; "
            f"email validity rate is {email_rate*100:.1f}% "
            f"(SLA target: {sla_email_validity*100:.0f}%)"
        )

    if "wrong_phone_digits" in planted_issues:
        n_bad = rng.randint(6, 15)
        indices = _pick_indices(n_bad)
        for idx in indices:
            bad_type = rng.choice(["too_few", "too_many", "letters"])
            if bad_type == "too_few":
                customers[idx]["phone"] = f"{rng.randint(200,999)}-{rng.randint(200,999)}"
            elif bad_type == "too_many":
                customers[idx]["phone"] = f"{rng.randint(200,999)}-{rng.randint(200,999)}-{rng.randint(1000,9999)}-{rng.randint(10,99)}"
            else:
                customers[idx]["phone"] = f"({rng.randint(200,999)}) {rng.randint(200,999)}-ABCD"
        issue_details["wrong_phone_digits"] = (
            f"{len(indices)} records have phone numbers with incorrect digit count "
            f"or non-numeric characters"
        )

    if "zip_state_mismatch" in planted_issues:
        n_bad = rng.randint(8, 18)
        indices = _pick_indices(n_bad)
        for idx in indices:
            # Assign a zip from a different state
            current_state = customers[idx]["state"]
            other_states = [s for s in active_states if s != current_state]
            if other_states:
                wrong_state = rng.choice(other_states)
                wrong_prefix = rng.choice(_STATE_ZIP_PREFIXES[wrong_state])
                customers[idx]["zip_code"] = f"{wrong_prefix}{rng.randint(10, 99)}"
                # Keep the state the same so zip doesn't match
        issue_details["zip_state_mismatch"] = (
            f"{len(indices)} records have zip codes that don't match their state"
        )

    if "future_date" in planted_issues:
        n_bad = rng.randint(4, 10)
        indices = _pick_indices(n_bad)
        for idx in indices:
            field = rng.choice(["birth_date", "enrollment_date"])
            future_year = rng.randint(2025, 2030)
            customers[idx][field] = f"{future_year}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        issue_details["future_date"] = (
            f"{len(indices)} records have dates in the future "
            f"(birth_date or enrollment_date after 2024)"
        )

    if "negative_age" in planted_issues:
        n_bad = rng.randint(3, 8)
        indices = _pick_indices(n_bad)
        for idx in indices:
            customers[idx]["age"] = rng.randint(-5, -1)
        issue_details["negative_age"] = (
            f"{len(indices)} records have negative age values"
        )

    if "fuzzy_duplicates" in planted_issues:
        n_dupes = rng.randint(4, 10)
        indices = _pick_indices(n_dupes)
        dupe_pairs = []
        for idx in indices:
            # Create a near-duplicate with slight formatting differences
            orig = customers[idx]
            dupe = dict(orig)
            dupe["customer_id"] = max(c["customer_id"] for c in customers) + 1
            # Vary the name slightly
            variation = rng.choice(["initial", "nickname", "typo"])
            if variation == "initial":
                dupe["first_name"] = orig["first_name"][0] + "."
            elif variation == "nickname":
                dupe["first_name"] = orig["first_name"][:3]
            else:
                name = list(orig["first_name"])
                if len(name) > 2:
                    pos = rng.randint(1, len(name) - 1)
                    name[pos] = rng.choice("aeiou")
                dupe["first_name"] = "".join(name)
            # Same address but slightly different format
            dupe["address"] = orig["address"].replace("St", "Street").replace("Ave", "Avenue").replace("Dr", "Drive")
            customers.append(dupe)
            dupe_pairs.append((orig["customer_id"], dupe["customer_id"]))
        issue_details["fuzzy_duplicates"] = (
            f"{n_dupes} fuzzy duplicate pairs detected (same person, different formatting); "
            f"e.g., IDs {dupe_pairs[0][0]} and {dupe_pairs[0][1]}"
        )

    if "orphan_records" in planted_issues:
        n_orphans = rng.randint(5, 12)
        indices = _pick_indices(n_orphans)
        for idx in indices:
            customers[idx]["preferred_category"] = rng.choice([
                "Outdoors", "Pet Supplies", "Industrial", "Luxury Goods"
            ])
        issue_details["orphan_records"] = (
            f"{n_orphans} records have preferred_category values not found in reference data "
            f"(e.g., 'Outdoors', 'Pet Supplies')"
        )

    if "impossible_date_combo" in planted_issues:
        n_bad = rng.randint(4, 9)
        indices = _pick_indices(n_bad)
        for idx in indices:
            # Make enrollment_date before birth_date
            birth = customers[idx]["birth_date"]
            birth_year = int(birth.split("-")[0])
            customers[idx]["enrollment_date"] = f"{birth_year - rng.randint(1, 10)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        issue_details["impossible_date_combo"] = (
            f"{n_bad} records have enrollment_date before birth_date "
            f"(impossible combination)"
        )

    if "out_of_range_values" in planted_issues:
        n_bad = rng.randint(4, 10)
        indices = _pick_indices(n_bad)
        for idx in indices:
            field = rng.choice(["age", "annual_income"])
            if field == "age":
                customers[idx]["age"] = rng.choice([200, 300, 999])
            else:
                customers[idx]["annual_income"] = rng.choice([-5000, -100, 0])
        issue_details["out_of_range_values"] = (
            f"{n_bad} records have values outside valid ranges "
            f"(age > 150 or annual_income <= 0)"
        )

    if "inconsistent_date_format" in planted_issues:
        n_bad = rng.randint(10, 25)
        indices = _pick_indices(n_bad)
        for idx in indices:
            field = rng.choice(["birth_date", "enrollment_date", "last_updated"])
            orig = customers[idx][field]
            parts = orig.split("-")
            if len(parts) == 3:
                fmt = rng.choice(["slash", "dot", "us_slash"])
                if fmt == "slash":
                    customers[idx][field] = f"{parts[2]}/{parts[1]}/{parts[0]}"
                elif fmt == "dot":
                    customers[idx][field] = f"{parts[2]}.{parts[1]}.{parts[0]}"
                else:
                    customers[idx][field] = f"{parts[1]}/{parts[2]}/{parts[0]}"
        issue_details["inconsistent_date_format"] = (
            f"{n_bad} records have dates in non-standard formats "
            f"(DD/MM/YYYY, DD.MM.YYYY, or MM/DD/YYYY instead of YYYY-MM-DD)"
        )

    if "missing_mandatory" in planted_issues:
        n_bad = rng.randint(5, 15)
        indices = _pick_indices(n_bad)
        mandatory_fields = ["email", "phone", "first_name", "last_name", "state"]
        for idx in indices:
            field = rng.choice(mandatory_fields)
            customers[idx][field] = ""
        issue_details["missing_mandatory"] = (
            f"{n_bad} records have empty mandatory fields "
            f"(email, phone, first_name, last_name, or state)"
        )

    if "invalid_reference_lookup" in planted_issues:
        n_bad = rng.randint(5, 12)
        indices = _pick_indices(n_bad)
        for idx in indices:
            customers[idx]["country"] = rng.choice(["XX", "ZZ", "QQ", "AA"])
        issue_details["invalid_reference_lookup"] = (
            f"{n_bad} records have country codes not found in reference data "
            f"(e.g., 'XX', 'ZZ')"
        )

    if "statistical_outlier" in planted_issues:
        n_bad = rng.randint(3, 7)
        indices = _pick_indices(n_bad)
        for idx in indices:
            customers[idx]["annual_income"] = rng.choice([999999999, 888888888, 777777777])
        issue_details["statistical_outlier"] = (
            f"{n_bad} records have annual_income values that are extreme statistical "
            f"outliers (e.g., $999,999,999)"
        )

    if "stale_data" in planted_issues:
        n_bad = rng.randint(8, 20)
        indices = _pick_indices(n_bad)
        for idx in indices:
            stale_year = rng.randint(2018, 2021)
            customers[idx]["last_updated"] = f"{stale_year}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        stale_threshold = 2024 - (sla_freshness_days // 365)
        issue_details["stale_data"] = (
            f"{n_bad} records have last_updated dates older than "
            f"{sla_freshness_days} days (before ~{stale_threshold})"
        )

    if "encoding_artifacts" in planted_issues:
        n_bad = rng.randint(3, 8)
        indices = _pick_indices(n_bad)
        corruptions = ["\xc3\xa9", "\xc3\xb6", "\xc3\xbc", "\xe2\x80\x99", "\xe2\x80\x93"]
        for idx in indices:
            field = rng.choice(["first_name", "last_name", "address", "city"])
            val = customers[idx][field]
            if len(val) > 2:
                pos = rng.randint(0, len(val) - 2)
                customers[idx][field] = val[:pos] + rng.choice(corruptions) + val[pos+1:]
        issue_details["encoding_artifacts"] = (
            f"{n_bad} records have encoding artifacts (e.g., '\xc3\xa9' instead of 'e') "
            f"in name or address fields"
        )

    # Shuffle customers
    rng.shuffle(customers)

    # --- Build customer_data.csv ---
    cust_header = "customer_id,first_name,last_name,email,phone,address,city,state,zip_code,country,birth_date,age,enrollment_date,preferred_category,annual_income,last_updated"
    cust_lines = [cust_header]
    for c in customers:
        cust_lines.append(
            f'{c["customer_id"]},{c["first_name"]},{c["last_name"]},'
            f'{c["email"]},{c["phone"]},"{c["address"]}",{c["city"]},'
            f'{c["state"]},{c["zip_code"]},{c["country"]},{c["birth_date"]},'
            f'{c["age"]},{c["enrollment_date"]},{c["preferred_category"]},'
            f'{c["annual_income"]},{c["last_updated"]}'
        )
    customer_csv = "\n".join(cust_lines) + "\n"

    # --- Build business_rules.txt ---
    rules_lines = [
        f"DATA QUALITY RULES — {company}",
        "",
        "=" * 60,
        "1. EMAIL VALIDATION",
        "=" * 60,
        "- Must contain exactly one '@' symbol",
        "- Must have a valid domain (contains at least one '.')",
        "- No spaces allowed",
        "- Must not be empty for any customer record",
        "",
        "=" * 60,
        "2. PHONE NUMBER VALIDATION",
        "=" * 60,
        "- US phone numbers must have exactly 10 digits",
        "- Formatting variations are acceptable (dashes, dots, parentheses)",
        "- No alphabetic characters allowed in digit positions",
        "",
        "=" * 60,
        "3. ADDRESS COMPLETENESS",
        "=" * 60,
        "- address, city, state, zip_code, and country are all required",
        "- zip_code must be 5 digits for US addresses",
        "- zip_code first 3 digits must correspond to the state",
        "- country must be a valid ISO 2-letter code from reference data",
        "",
        "=" * 60,
        "4. DATE FIELDS",
        "=" * 60,
        "- All dates must be in YYYY-MM-DD format",
        "- birth_date must not be in the future",
        "- enrollment_date must not be in the future",
        "- enrollment_date must be after birth_date",
        "- age field must be consistent with birth_date (2024 - birth_year)",
        "",
        "=" * 60,
        "5. NUMERIC RANGES",
        "=" * 60,
        "- age: 0 to 150 (inclusive)",
        "- annual_income: > 0",
        "- customer_id: unique, > 0",
        "",
        "=" * 60,
        "6. REFERENTIAL INTEGRITY",
        "=" * 60,
        "- preferred_category must exist in reference_data.csv product categories",
        "- country must exist in reference_data.csv country codes",
        "- state must be a valid US state abbreviation",
        "",
        "=" * 60,
        "7. DUPLICATE DETECTION",
        "=" * 60,
        "- No exact duplicate customer_ids",
        "- Flag fuzzy duplicates: same last_name + similar first_name + same address",
        "- Fuzzy matching should account for formatting differences",
        "",
        "=" * 60,
        "8. DATA FRESHNESS",
        "=" * 60,
        f"- last_updated must be within {sla_freshness_days} days of current date (2024-09-01)",
        "- Records older than threshold should be flagged for review",
        "",
        "=" * 60,
        "9. ENCODING",
        "=" * 60,
        "- All text fields must use valid UTF-8 encoding",
        "- Common encoding artifacts (e.g., \xc3\xa9, \xe2\x80\x99) indicate corruption",
        "",
    ]
    rules_content = "\n".join(rules_lines) + "\n"

    # --- Build reference_data.csv ---
    ref_lines = [
        "type,code,description",
    ]
    for cat in _PRODUCT_CATEGORIES_REF:
        ref_lines.append(f"product_category,{cat},{cat}")
    for code in _VALID_COUNTRIES:
        ref_lines.append(f"country_code,{code},{code}")
    for abbr, name in _US_STATES:
        ref_lines.append(f"state_code,{abbr},{name}")
    ref_content = "\n".join(ref_lines) + "\n"

    # --- Build sla_requirements.txt ---
    sla_lines = [
        f"DATA QUALITY SLA REQUIREMENTS — {company}",
        "",
        f"Reporting Period: 2024-Q3",
        f"Dataset: Customer Master Data",
        "",
        "=" * 60,
        "QUALITY METRICS & TARGETS",
        "=" * 60,
        "",
        f"Email validity rate:        >= {sla_email_validity*100:.0f}%",
        f"Phone validity rate:         >= {sla_phone_validity*100:.0f}%",
        f"Address completeness rate:   >= {sla_address_completeness*100:.0f}%",
        f"Duplicate rate (fuzzy):      <= {sla_duplicate_rate*100:.1f}%",
        f"Data freshness (last_updated within {sla_freshness_days} days): >= 95%",
        f"Date format consistency:     >= 99%",
        f"Referential integrity:       >= 99%",
        "",
        "=" * 60,
        "SEVERITY CLASSIFICATION",
        "=" * 60,
        "",
        "Critical: Any issue affecting >5% of records",
        "Major: Any issue affecting 2-5% of records",
        "Minor: Any issue affecting <2% of records",
        "",
        "=" * 60,
        "ESCALATION",
        "=" * 60,
        "",
        "If any Critical SLA is breached, escalate to Data Governance team.",
        "All Major issues require remediation plan within 5 business days.",
        "Minor issues logged for next quarterly review.",
        "",
    ]
    sla_content = "\n".join(sla_lines) + "\n"

    # --- Problem statement ---
    problem_statement = f"""# Data Quality Audit

You are a data quality analyst at {company}. You need to audit the customer
master dataset against business rules and SLA requirements, then produce a
comprehensive quality report.

## Source Files
- /testbed/data/customer_data.csv — Customer records ({len(customers)} rows)
- /testbed/data/business_rules.txt — Data quality rules (valid ranges, formats, cross-field constraints)
- /testbed/data/reference_data.csv — Valid values for lookup fields (states, countries, product categories)
- /testbed/data/sla_requirements.txt — Data quality SLA targets

## Requirements
1. Check every business rule against the customer data
2. For each issue type, count affected records and compute the rate
3. Compare rates against SLA targets and flag breaches
4. Provide specific examples for each issue (customer IDs, field values)
5. Classify issues by severity (Critical/Major/Minor per SLA definitions)
6. Check for fuzzy duplicates (same person, different formatting)
7. Verify referential integrity against reference_data.csv
8. Distinguish true quality issues from legitimate edge cases

Write a detailed audit report to /testbed/audit_report.txt"""

    # --- Rubric ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does the file /testbed/audit_report.txt exist and contain substantive content (more than a few lines)?",
            points=1,
        ),
    ]

    # One binary check per planted issue
    for iss_key in planted_issues:
        label = iss_key.replace("_", " ")
        detail = issue_details.get(iss_key, f"Issue of type: {label}")
        rubric_items.append(BinaryRubricCategory(
            name=f"identifies_{iss_key}",
            question=(
                f"Does the audit report identify the '{label}' issue? "
                f"Ground truth: {detail}"
            ),
            points=2,
        ))

    # Quantification checks for first 3 planted issues
    for iss_key in planted_issues[:3]:
        label = iss_key.replace("_", " ")
        detail = issue_details.get(iss_key, "")
        rubric_items.append(BinaryRubricCategory(
            name=f"quantifies_{iss_key}",
            question=(
                f"Does the report provide a count or percentage for the '{label}' issue, "
                f"approximately matching ground truth? Ground truth: {detail}"
            ),
            points=1,
        ))

    # False-positive checks
    rubric_items.extend([
        BinaryRubricCategory(
            name="no_false_positive_old_person",
            question=(
                f"Does the report avoid incorrectly flagging {old_person_name} "
                f"(age 104, born 1920) as a data error? This is a legitimate record "
                f"of a very elderly person. (Age 104 is within the valid range of 0-150.)"
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="no_false_positive_military_zip",
            question=(
                f"Does the report avoid incorrectly flagging {mil_person_name}'s "
                f"zip code (09021, APO address in NY) as an invalid zip-state mismatch? "
                f"APO/FPO addresses legitimately use NY state codes."
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="no_false_positive_high_income",
            question=(
                f"Does the report avoid flagging the customer with annual_income of $850,000 "
                f"as a statistical outlier or data error? This is high but plausible for "
                f"a high-earning professional."
            ),
            points=1,
        ),
    ])

    # SLA comparison
    rubric_items.append(BinaryRubricCategory(
        name="compares_sla_targets",
        question=(
            "Does the report compare at least 3 quality metrics against the SLA "
            "targets from sla_requirements.txt and identify which SLAs are met/breached?"
        ),
        points=2,
    ))

    # Severity classification
    rubric_items.append(BinaryRubricCategory(
        name="classifies_severity",
        question=(
            "Does the report classify issues by severity (Critical/Major/Minor) "
            "using the thresholds from the SLA requirements (>5% = Critical, "
            "2-5% = Major, <2% = Minor)?"
        ),
        points=1,
    ))

    # Structural
    rubric_items.append(BinaryRubricCategory(
        name="reports_total_records",
        question=f"Does the report mention the dataset has approximately {len(customers)} customer records (within 5%)?",
        points=1,
    ))

    rubric_items.append(BinaryRubricCategory(
        name="checks_referential_integrity",
        question=(
            "Does the report check preferred_category and/or country values against "
            "reference_data.csv and identify any that don't match?"
        ),
        points=1,
    ))

    # Report quality
    rubric_items.append(RubricCategory(
        name="report_quality",
        description="Is the audit report well-organized with clear sections, specific examples, and actionable findings?",
        failure="Disorganized, missing major issue categories, or no specific examples.",
        minor_failure="Some structure but missing quantification or specificity for several issues.",
        minor_success="Organized with identifiable sections, counts, and some specific customer ID examples.",
        success="Professional audit report with executive summary, issue-by-issue analysis with counts and examples, SLA comparison, severity classification, and remediation recommendations.",
        points=2,
    ))

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=tuple(rubric_items),
        submission_instructions="Write your completed audit report to /testbed/audit_report.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/customer_data.csv": customer_csv,
            "/testbed/data/business_rules.txt": rules_content,
            "/testbed/data/reference_data.csv": ref_content,
            "/testbed/data/sla_requirements.txt": sla_content,
        },
        problem_type="data_quality_audit",
    )
