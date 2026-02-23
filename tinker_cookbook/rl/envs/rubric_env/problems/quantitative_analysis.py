"""Quantitative analysis tasks: portfolio analysis, actuarial analysis, and A/B test analysis.

Each factory generates a realistic scenario with necessary_files containing all
information required to solve the problem. Seeds control randomization of positions,
claims, experiment parameters, etc.
"""

from __future__ import annotations

import random as _random
import math

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


def _fmt_pct(value: float, decimals: int = 2) -> str:
    """Format a float as a percentage string."""
    return f"{value:.{decimals}f}%"


def _fmt_num(value: float, decimals: int = 2) -> str:
    """Format a number with commas and given decimal places."""
    return f"{value:,.{decimals}f}"


# =============================================================================
# DOMAIN: PORTFOLIO ANALYSIS
# =============================================================================

TICKERS_BY_CLASS: dict[str, list[tuple[str, str]]] = {
    "equity": [
        ("AAPL", "Technology"), ("MSFT", "Technology"), ("GOOGL", "Technology"),
        ("AMZN", "Consumer Discretionary"), ("JPM", "Financials"),
        ("JNJ", "Healthcare"), ("PG", "Consumer Staples"), ("XOM", "Energy"),
        ("V", "Financials"), ("UNH", "Healthcare"), ("HD", "Consumer Discretionary"),
        ("PFE", "Healthcare"), ("CVX", "Energy"), ("KO", "Consumer Staples"),
        ("WMT", "Consumer Staples"), ("DIS", "Communication"), ("INTC", "Technology"),
        ("BA", "Industrials"), ("CAT", "Industrials"), ("MMM", "Industrials"),
    ],
    "bond": [
        ("BND", "Broad Bond"), ("AGG", "Aggregate Bond"), ("TLT", "Long Treasury"),
        ("IEF", "Intermediate Treasury"), ("LQD", "Investment Grade Corporate"),
        ("HYG", "High Yield Corporate"), ("TIPS", "Inflation Protected"),
        ("MUB", "Municipal Bond"),
    ],
    "reit": [
        ("VNQ", "Diversified REIT"), ("O", "Net Lease REIT"), ("SPG", "Retail REIT"),
        ("AMT", "Tower REIT"), ("PLD", "Industrial REIT"), ("EQIX", "Data Center REIT"),
    ],
    "cash": [
        ("SGOV", "Treasury Bills"), ("BIL", "Short-term Treasury"),
        ("SHV", "Short Treasury"), ("MINT", "Enhanced Cash"),
    ],
}


# =============================================================================
# 1. PORTFOLIO ANALYSIS
# =============================================================================


def make_portfolio_analysis(rand_seed: int = 42) -> RubricDatapoint:
    """Analyze an investment portfolio: compute returns, risk metrics,
    rebalancing needs, and tax implications of selling positions.

    Seed varies: number of positions, asset class distribution, which
    positions are winners/losers, whether rebalancing is needed, purchase
    dates (some short-term, some long-term).
    """
    rng = _random.Random(rand_seed)

    investor_name = random_name(rand_seed)
    account_number = f"ACC-{rng.randint(100000, 999999)}"

    # --- Target allocation (varies by seed) ---
    alloc_sets = [
        {"equity": 60, "bond": 25, "reit": 10, "cash": 5},
        {"equity": 70, "bond": 20, "reit": 5, "cash": 5},
        {"equity": 50, "bond": 30, "reit": 15, "cash": 5},
        {"equity": 55, "bond": 30, "reit": 10, "cash": 5},
        {"equity": 65, "bond": 20, "reit": 10, "cash": 5},
    ]
    target_alloc = rng.choice(alloc_sets)
    rebalance_threshold = 5  # ±5 percentage points

    # --- Build positions ---
    n_equity = rng.randint(8, 14)
    n_bond = rng.randint(3, 5)
    n_reit = rng.randint(2, 4)
    n_cash = rng.randint(1, 2)

    positions: list[dict] = []
    current_date_str = "2024-12-15"

    for asset_class, count in [("equity", n_equity), ("bond", n_bond),
                                ("reit", n_reit), ("cash", n_cash)]:
        available = list(TICKERS_BY_CLASS[asset_class])
        rng.shuffle(available)
        chosen = available[:count]
        for ticker, sector in chosen:
            shares = rng.randint(10, 500)
            purchase_price = round(rng.uniform(20, 400), 2)
            # Some positions purchased recently (short-term), some long ago
            months_ago = rng.choice([2, 4, 6, 8, 10, 14, 18, 24, 36, 48])
            purchase_month = 12 - (months_ago % 12)
            purchase_year = 2024 - (months_ago // 12) - (1 if months_ago % 12 >= 12 else 0)
            if purchase_month <= 0:
                purchase_month += 12
                purchase_year -= 1
            purchase_date = f"{purchase_year}-{purchase_month:02d}-{rng.randint(1,28):02d}"

            # Current price: vary from purchase with some gains/losses
            price_change_pct = rng.uniform(-0.35, 0.55)
            current_price = round(purchase_price * (1 + price_change_pct), 2)
            current_price = max(current_price, 1.00)  # floor

            # Is this short-term? (held < 12 months as of Dec 15 2024)
            is_short_term = months_ago < 12

            positions.append({
                "ticker": ticker,
                "sector": sector,
                "asset_class": asset_class,
                "shares": shares,
                "purchase_price": purchase_price,
                "purchase_date": purchase_date,
                "current_price": current_price,
                "is_short_term": is_short_term,
            })

    rng.shuffle(positions)

    # --- Generate 12 months of monthly returns ---
    all_tickers = [p["ticker"] for p in positions]
    benchmark_ticker = "SPY"
    months_list = [
        "2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06",
        "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12",
    ]
    monthly_returns: dict[str, list[float]] = {}
    for ticker in all_tickers + [benchmark_ticker]:
        returns = []
        for _ in months_list:
            r = round(rng.gauss(0.008, 0.045), 4)  # ~0.8% mean, 4.5% std monthly
            returns.append(r)
        monthly_returns[ticker] = returns

    # --- Compute ground truth ---
    # Total portfolio value
    total_value = 0.0
    total_cost_basis = 0.0
    class_values: dict[str, float] = {"equity": 0.0, "bond": 0.0, "reit": 0.0, "cash": 0.0}

    position_details: list[dict] = []
    for p in positions:
        market_value = round(p["shares"] * p["current_price"], 2)
        cost_basis = round(p["shares"] * p["purchase_price"], 2)
        gain_loss = round(market_value - cost_basis, 2)
        gain_loss_pct = round((gain_loss / cost_basis) * 100, 2) if cost_basis != 0 else 0.0

        total_value += market_value
        total_cost_basis += cost_basis
        class_values[p["asset_class"]] += market_value

        position_details.append({
            "ticker": p["ticker"],
            "asset_class": p["asset_class"],
            "sector": p["sector"],
            "shares": p["shares"],
            "purchase_price": p["purchase_price"],
            "current_price": p["current_price"],
            "purchase_date": p["purchase_date"],
            "market_value": market_value,
            "cost_basis": cost_basis,
            "gain_loss": gain_loss,
            "gain_loss_pct": gain_loss_pct,
            "is_short_term": p["is_short_term"],
        })

    total_value = round(total_value, 2)
    total_cost_basis = round(total_cost_basis, 2)
    total_unrealized_gain = round(total_value - total_cost_basis, 2)

    # Current allocation percentages
    current_alloc: dict[str, float] = {}
    for cls in ["equity", "bond", "reit", "cash"]:
        current_alloc[cls] = round((class_values[cls] / total_value) * 100, 2) if total_value > 0 else 0.0

    # Rebalancing: which classes are over/under the threshold?
    overweight_classes: list[str] = []
    underweight_classes: list[str] = []
    rebalancing_needed = False
    for cls in ["equity", "bond", "reit", "cash"]:
        diff = current_alloc[cls] - target_alloc[cls]
        if diff > rebalance_threshold:
            overweight_classes.append(cls)
            rebalancing_needed = True
        elif diff < -rebalance_threshold:
            underweight_classes.append(cls)
            rebalancing_needed = True

    # Largest gainer and loser
    sorted_by_gain = sorted(position_details, key=lambda x: x["gain_loss"], reverse=True)
    largest_gainer = sorted_by_gain[0]
    largest_loser = sorted_by_gain[-1]

    # Short-term vs long-term gains totals
    short_term_gains_total = round(sum(
        p["gain_loss"] for p in position_details if p["is_short_term"] and p["gain_loss"] > 0
    ), 2)
    short_term_losses_total = round(sum(
        p["gain_loss"] for p in position_details if p["is_short_term"] and p["gain_loss"] < 0
    ), 2)
    long_term_gains_total = round(sum(
        p["gain_loss"] for p in position_details if not p["is_short_term"] and p["gain_loss"] > 0
    ), 2)
    long_term_losses_total = round(sum(
        p["gain_loss"] for p in position_details if not p["is_short_term"] and p["gain_loss"] < 0
    ), 2)

    # Tax estimate for rebalancing: if overweight classes need selling
    # Assume we sell enough of overweight positions to bring back to target
    short_term_tax_rate = 0.37
    long_term_tax_rate = 0.15
    rebalance_tax_estimate = 0.0
    rebalance_sell_amount = 0.0
    if rebalancing_needed and overweight_classes:
        for cls in overweight_classes:
            excess_pct = (current_alloc[cls] - target_alloc[cls]) / 100.0
            sell_value = round(total_value * excess_pct, 2)
            rebalance_sell_amount += sell_value

            # Estimate gain ratio for this class
            cls_positions = [p for p in position_details if p["asset_class"] == cls]
            cls_total_mv = sum(p["market_value"] for p in cls_positions)
            cls_total_cb = sum(p["cost_basis"] for p in cls_positions)
            if cls_total_mv > 0:
                gain_ratio = (cls_total_mv - cls_total_cb) / cls_total_mv
            else:
                gain_ratio = 0.0
            estimated_gain = round(sell_value * gain_ratio, 2)

            # Split by short/long term proportionally
            cls_st = [p for p in cls_positions if p["is_short_term"]]
            cls_lt = [p for p in cls_positions if not p["is_short_term"]]
            st_mv = sum(p["market_value"] for p in cls_st)
            lt_mv = sum(p["market_value"] for p in cls_lt)
            total_cls_mv = st_mv + lt_mv
            if total_cls_mv > 0:
                st_frac = st_mv / total_cls_mv
            else:
                st_frac = 0.0

            st_gain = round(estimated_gain * st_frac, 2)
            lt_gain = round(estimated_gain * (1 - st_frac), 2)

            rebalance_tax_estimate += max(0, st_gain) * short_term_tax_rate
            rebalance_tax_estimate += max(0, lt_gain) * long_term_tax_rate

    rebalance_sell_amount = round(rebalance_sell_amount, 2)
    rebalance_tax_estimate = round(rebalance_tax_estimate, 2)

    # Portfolio beta approximation: weighted average of position betas vs benchmark
    # Use monthly returns correlation as beta proxy
    benchmark_returns = monthly_returns[benchmark_ticker]
    bench_mean = sum(benchmark_returns) / len(benchmark_returns)
    bench_var = sum((r - bench_mean) ** 2 for r in benchmark_returns) / len(benchmark_returns)

    weighted_beta_sum = 0.0
    for p in position_details:
        ticker_returns = monthly_returns[p["ticker"]]
        tick_mean = sum(ticker_returns) / len(ticker_returns)
        cov = sum((tr - tick_mean) * (br - bench_mean)
                   for tr, br in zip(ticker_returns, benchmark_returns)) / len(ticker_returns)
        beta = cov / bench_var if bench_var > 0 else 1.0
        weight = p["market_value"] / total_value if total_value > 0 else 0.0
        weighted_beta_sum += beta * weight

    portfolio_beta = round(weighted_beta_sum, 2)

    # Sharpe ratio approximation (annualized)
    # Portfolio monthly return = weighted average of position monthly returns
    portfolio_monthly_returns: list[float] = []
    for m_idx in range(12):
        month_return = 0.0
        for p in position_details:
            weight = p["market_value"] / total_value if total_value > 0 else 0.0
            month_return += weight * monthly_returns[p["ticker"]][m_idx]
        portfolio_monthly_returns.append(month_return)

    port_mean_monthly = sum(portfolio_monthly_returns) / len(portfolio_monthly_returns)
    port_std_monthly = math.sqrt(
        sum((r - port_mean_monthly) ** 2 for r in portfolio_monthly_returns)
        / len(portfolio_monthly_returns)
    )
    risk_free_annual = 0.05  # 5% annual risk-free rate
    risk_free_monthly = risk_free_annual / 12
    sharpe_ratio = round(
        ((port_mean_monthly - risk_free_monthly) / port_std_monthly) * math.sqrt(12), 2
    ) if port_std_monthly > 0 else 0.0

    # Pick a clean class for false-positive rebalancing check
    clean_classes = [cls for cls in ["equity", "bond", "reit", "cash"]
                     if cls not in overweight_classes and cls not in underweight_classes]
    false_pos_class = clean_classes[0] if clean_classes else None

    # --- Build portfolio.csv ---
    port_csv_lines = [
        "ticker,shares,purchase_price,purchase_date,current_price,asset_class,sector"
    ]
    for p in positions:
        port_csv_lines.append(
            f"{p['ticker']},{p['shares']},{p['purchase_price']:.2f},"
            f"{p['purchase_date']},{p['current_price']:.2f},"
            f"{p['asset_class']},{p['sector']}"
        )
    portfolio_csv = "\n".join(port_csv_lines) + "\n"

    # --- Build market_data.csv ---
    mkt_header = "ticker," + ",".join(months_list)
    mkt_lines = [mkt_header]
    for ticker in all_tickers + [benchmark_ticker]:
        returns_str = ",".join(f"{r:.4f}" for r in monthly_returns[ticker])
        mkt_lines.append(f"{ticker},{returns_str}")
    market_csv = "\n".join(mkt_lines) + "\n"

    # --- Build target_allocation.txt ---
    alloc_lines = [
        "TARGET PORTFOLIO ALLOCATION",
        "",
        f"Account: {account_number}",
        f"Investor: {investor_name}",
        f"Date: {current_date_str}",
        "",
        "=" * 50,
        "ASSET CLASS TARGET ALLOCATION",
        "=" * 50,
        "",
        f"{'Asset Class':<20} {'Target %':>10}",
        f"{'-'*20} {'-'*10}",
    ]
    for cls in ["equity", "bond", "reit", "cash"]:
        alloc_lines.append(f"{cls.upper():<20} {target_alloc[cls]:>9}%")
    alloc_lines.extend([
        "",
        f"Rebalancing Threshold: +/- {rebalance_threshold} percentage points",
        "If any asset class deviates from its target by more than the threshold,",
        "rebalancing is triggered for ALL classes.",
        "",
        "REBALANCING RULES:",
        "  1. Sell positions in overweight classes to bring back to target",
        "  2. Use proceeds to buy underweight classes",
        "  3. Consider tax implications of sales (short-term vs long-term)",
        "  4. Prioritize selling positions with losses for tax-loss harvesting",
        "",
    ])
    target_alloc_content = "\n".join(alloc_lines) + "\n"

    # --- Build tax_rules.txt ---
    tax_lines = [
        "TAX RULES FOR INVESTMENT TRANSACTIONS",
        "",
        "=" * 50,
        "CAPITAL GAINS TAX RATES (2024)",
        "=" * 50,
        "",
        "SHORT-TERM CAPITAL GAINS (held < 12 months):",
        "  Taxed as ordinary income at the investor's marginal rate.",
        f"  For this analysis, use a rate of {short_term_tax_rate*100:.0f}%.",
        "",
        "LONG-TERM CAPITAL GAINS (held >= 12 months):",
        f"  Taxed at a preferential rate of {long_term_tax_rate*100:.0f}%.",
        "  (Actual rate depends on income; use 15% for this analysis.)",
        "",
        "HOLDING PERIOD:",
        "  The holding period starts the day AFTER the purchase date.",
        f"  As of {current_date_str}, positions purchased before 2024-01-01 are long-term.",
        "  Positions purchased on or after 2024-01-01 are short-term.",
        "",
        "WASH SALE RULE:",
        "  If you sell a security at a loss and repurchase a substantially identical",
        "  security within 30 days before or after the sale, the loss is disallowed",
        "  for tax purposes. Consider this when rebalancing.",
        "",
        "TAX-LOSS HARVESTING:",
        "  Selling positions at a loss to offset capital gains. Losses can offset",
        "  gains dollar-for-dollar. Net losses up to $3,000/year can offset ordinary",
        "  income. Excess losses carry forward to future years.",
        "",
        "REBALANCING TAX ESTIMATE:",
        "  When estimating the tax cost of rebalancing, compute the estimated gain",
        "  on the amount sold, split into short-term and long-term components",
        "  proportional to the market value of short-term vs long-term positions",
        "  in that asset class, then apply the respective tax rates.",
        "",
    ]
    tax_rules_content = "\n".join(tax_lines) + "\n"

    # --- Problem statement ---
    problem_statement = f"""# Investment Portfolio Analysis

You are a financial analyst reviewing the investment portfolio of {investor_name}
(Account {account_number}) as of {current_date_str}.

## Source Files
- /testbed/data/portfolio.csv — Current holdings: ticker, shares, purchase_price, purchase_date, current_price, asset_class, sector
- /testbed/data/market_data.csv — 12 months of monthly returns for each position and the benchmark (SPY)
- /testbed/data/target_allocation.txt — Target asset class allocation and rebalancing rules
- /testbed/data/tax_rules.txt — Capital gains tax rates and rules for transaction tax estimation

## Requirements
1. Compute total portfolio value and total unrealized gain/loss
2. Compute per-position gain/loss (dollar and percentage)
3. Compute current asset class allocation percentages
4. Determine which classes (if any) are over- or under-weight relative to target
5. Determine whether rebalancing is needed (any class outside ±{rebalance_threshold}% threshold)
6. Identify the largest gaining and largest losing positions
7. Calculate total short-term and long-term unrealized gains
8. Estimate the tax cost of rebalancing (if needed) based on the tax rules
9. Compute portfolio beta relative to the benchmark (SPY) using the monthly returns data
10. Compute an approximate annualized Sharpe ratio (use 5% annual risk-free rate)

Write a comprehensive portfolio analysis report to /testbed/portfolio_report.txt
showing all calculations and recommendations."""

    # Count positions
    n_short_term = sum(1 for p in position_details if p["is_short_term"])
    n_long_term = sum(1 for p in position_details if not p["is_short_term"])

    # --- Build rubric ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does the file /testbed/portfolio_report.txt exist and contain substantive content (more than a few lines)?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_total_value",
            question=f"Does the report compute the total portfolio market value as approximately {_fmt_money(total_value)} (within $500)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_total_unrealized_gain",
            question=f"Does the report compute the total unrealized gain/loss as approximately {_fmt_money(total_unrealized_gain)} (within $500)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_equity_allocation_pct",
            question=f"Does the report compute the equity allocation as approximately {_fmt_pct(current_alloc['equity'])} of the portfolio (within 1 percentage point)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_bond_allocation_pct",
            question=f"Does the report compute the bond allocation as approximately {_fmt_pct(current_alloc['bond'])} of the portfolio (within 1 percentage point)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_reit_allocation_pct",
            question=f"Does the report compute the REIT allocation as approximately {_fmt_pct(current_alloc['reit'])} of the portfolio (within 1 percentage point)?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_cash_allocation_pct",
            question=f"Does the report compute the cash allocation as approximately {_fmt_pct(current_alloc['cash'])} of the portfolio (within 1 percentage point)?",
            points=1,
        ),
    ]

    if overweight_classes:
        rubric_items.append(
            BinaryRubricCategory(
                name="identifies_overweight_class",
                question=(
                    f"Does the report correctly identify the overweight asset class(es) as: "
                    f"{', '.join(c.upper() for c in overweight_classes)}? "
                    f"(These are more than {rebalance_threshold}% above target.)"
                ),
                points=2,
            )
        )
    else:
        rubric_items.append(
            BinaryRubricCategory(
                name="identifies_overweight_class",
                question="Does the report correctly state that NO asset class is overweight beyond the rebalancing threshold?",
                points=2,
            )
        )

    if underweight_classes:
        rubric_items.append(
            BinaryRubricCategory(
                name="identifies_underweight_class",
                question=(
                    f"Does the report correctly identify the underweight asset class(es) as: "
                    f"{', '.join(c.upper() for c in underweight_classes)}? "
                    f"(These are more than {rebalance_threshold}% below target.)"
                ),
                points=2,
            )
        )
    else:
        rubric_items.append(
            BinaryRubricCategory(
                name="identifies_underweight_class",
                question="Does the report correctly state that NO asset class is underweight beyond the rebalancing threshold?",
                points=2,
            )
        )

    rubric_items.append(
        BinaryRubricCategory(
            name="correct_rebalancing_needed",
            question=(
                f"Does the report correctly determine that rebalancing IS "
                f"{'NEEDED' if rebalancing_needed else 'NOT needed'}?"
            ),
            points=2,
        )
    )

    rubric_items.extend([
        BinaryRubricCategory(
            name="correct_largest_gainer_ticker",
            question=(
                f"Does the report identify {largest_gainer['ticker']} as the largest gainer "
                f"by dollar amount (unrealized gain of {_fmt_money(largest_gainer['gain_loss'])}, "
                f"the largest dollar gain)?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_largest_loser_ticker",
            question=(
                f"Does the report identify {largest_loser['ticker']} as the largest loser "
                f"by dollar amount (unrealized loss of {_fmt_money(largest_loser['gain_loss'])}, "
                f"the largest dollar loss)?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_short_term_gains_total",
            question=(
                f"Does the report compute total short-term unrealized GAINS (counting only "
                f"positions with positive gains, not losses) as approximately "
                f"{_fmt_money(short_term_gains_total)} (within $200)? "
                f"({n_short_term} positions are short-term.)"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_long_term_gains_total",
            question=(
                f"Does the report compute total long-term unrealized GAINS (counting only "
                f"positions with positive gains, not losses) as approximately "
                f"{_fmt_money(long_term_gains_total)} (within $200)? "
                f"({n_long_term} positions are long-term.)"
            ),
            points=2,
        ),
    ])

    if rebalancing_needed and rebalance_tax_estimate > 0:
        rubric_items.append(
            BinaryRubricCategory(
                name="correct_estimated_tax_on_rebalance",
                question=(
                    f"Does the report estimate the tax cost of rebalancing as approximately "
                    f"{_fmt_money(rebalance_tax_estimate)} (within $500)? This is based on selling "
                    f"approximately {_fmt_money(rebalance_sell_amount)} from overweight classes, "
                    f"applying {short_term_tax_rate*100:.0f}% to short-term gains and "
                    f"{long_term_tax_rate*100:.0f}% to long-term gains."
                ),
                points=2,
            )
        )
    else:
        rubric_items.append(
            BinaryRubricCategory(
                name="correct_estimated_tax_on_rebalance",
                question=(
                    "Does the report correctly note that no rebalancing sales are needed, "
                    "so the estimated tax cost is $0 or negligible?"
                ),
                points=2,
            )
        )

    if false_pos_class:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"no_false_rebalance_trigger_on_{false_pos_class}",
                question=(
                    f"Does the report correctly show that {false_pos_class.upper()} "
                    f"(current: {_fmt_pct(current_alloc[false_pos_class])}, "
                    f"target: {target_alloc[false_pos_class]}%) is within the "
                    f"±{rebalance_threshold}% rebalancing threshold and does NOT need rebalancing?"
                ),
                points=1,
            )
        )

    rubric_items.extend([
        BinaryRubricCategory(
            name="correct_portfolio_beta",
            question=(
                f"Does the report compute the portfolio beta relative to SPY as approximately "
                f"{_fmt_num(portfolio_beta)} (within 0.15)? Beta is calculated as the "
                f"weighted average of individual position betas derived from the monthly return data."
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_sharpe_ratio",
            question=(
                f"Does the report compute an annualized Sharpe ratio of approximately "
                f"{_fmt_num(sharpe_ratio)} (within 0.3)? The Sharpe ratio uses a 5% annual "
                f"risk-free rate and annualizes monthly returns by multiplying by sqrt(12)."
            ),
            points=2,
        ),
        RubricCategory(
            name="analysis_quality",
            description="Is the portfolio analysis report comprehensive, well-organized, and does it provide actionable recommendations?",
            failure="Superficial analysis missing most calculations or major errors throughout.",
            minor_failure="Some calculations present but analysis is incomplete or poorly organized.",
            minor_success="Most calculations correct with reasonable organization and some recommendations.",
            success="Comprehensive report with all calculations, clear organization, and thoughtful recommendations.",
            points=2,
        ),
    ])

    rubric = tuple(rubric_items)

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write your completed portfolio analysis report to /testbed/portfolio_report.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/portfolio.csv": portfolio_csv,
            "/testbed/data/market_data.csv": market_csv,
            "/testbed/data/target_allocation.txt": target_alloc_content,
            "/testbed/data/tax_rules.txt": tax_rules_content,
        },
        problem_type="portfolio_analysis",
    )


# =============================================================================
# DOMAIN: ACTUARIAL ANALYSIS
# =============================================================================

POLICY_TYPES = ["auto", "home", "commercial"]

REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]

CLAIM_STATUSES = ["open", "closed", "reserved"]


# =============================================================================
# 2. ACTUARIAL ANALYSIS
# =============================================================================


def make_actuarial_analysis(rand_seed: int = 42) -> RubricDatapoint:
    """Analyze insurance claims data to compute loss ratios, reserve estimates,
    and premium adequacy.

    Seed varies: claim frequency, average severity, which policy types are
    performing well/poorly, whether IBNR adjustment is significant, seasonal
    patterns.
    """
    rng = _random.Random(rand_seed)

    company_name = pick1(COMPANY_NAMES, rand_seed)
    analyst_name = random_name(rand_seed)
    analysis_year = 2024

    # --- Loss development factors (varies slightly by seed) ---
    ldf_base = {
        "auto":       {"12": 1.05, "24": 1.02, "36": 1.01},
        "home":       {"12": 1.10, "24": 1.04, "36": 1.01},
        "commercial": {"12": 1.15, "24": 1.06, "36": 1.02},
    }
    # Jitter the LDFs a little per seed
    ldfs: dict[str, dict[str, float]] = {}
    for ptype in POLICY_TYPES:
        ldfs[ptype] = {}
        for mat in ["12", "24", "36"]:
            base = ldf_base[ptype][mat]
            jitter = rng.uniform(-0.02, 0.02)
            ldfs[ptype][mat] = round(base + jitter, 3)

    # Expected loss ratios
    expected_lr_base = {"auto": 0.65, "home": 0.55, "commercial": 0.60}
    expected_lrs: dict[str, float] = {}
    for ptype in POLICY_TYPES:
        expected_lrs[ptype] = round(expected_lr_base[ptype] + rng.uniform(-0.05, 0.05), 3)

    # Reserve margin requirement
    reserve_margin_pct = rng.choice([5, 8, 10])

    # --- Generate premium data ---
    # Annual premium per type, split monthly
    annual_premiums: dict[str, float] = {}
    monthly_premiums: dict[str, list[float]] = {}
    for ptype in POLICY_TYPES:
        base_annual = {"auto": 12_000_000, "home": 8_000_000, "commercial": 15_000_000}[ptype]
        annual = round(base_annual * rng.uniform(0.8, 1.3), 2)
        annual_premiums[ptype] = annual
        # Monthly split with seasonal variation
        monthly = []
        for m in range(12):
            seasonal_factor = 1.0 + 0.1 * math.sin(2 * math.pi * m / 12)
            month_prem = round((annual / 12) * seasonal_factor * rng.uniform(0.95, 1.05), 2)
            monthly.append(month_prem)
        # Adjust so they sum to annual
        scale = annual / sum(monthly)
        monthly = [round(m * scale, 2) for m in monthly]
        monthly_premiums[ptype] = monthly

    total_premium = round(sum(annual_premiums.values()), 2)

    # --- Generate claims ---
    n_claims_base = {"auto": rng.randint(60, 100), "home": rng.randint(30, 60),
                     "commercial": rng.randint(20, 40)}
    claims: list[dict] = []
    claim_id = 1000

    # Per-type accumulators
    paid_by_type: dict[str, float] = {t: 0.0 for t in POLICY_TYPES}
    reserved_by_type: dict[str, float] = {t: 0.0 for t in POLICY_TYPES}

    for ptype in POLICY_TYPES:
        n_claims = n_claims_base[ptype]
        severity_mean = {"auto": 8000, "home": 15000, "commercial": 25000}[ptype]
        severity_std = severity_mean * 0.6

        for _ in range(n_claims):
            claim_id += 1
            cid = f"CLM-{claim_id}"
            month_occurred = rng.randint(1, 12)
            day_occurred = rng.randint(1, 28)
            # Reported 0-30 days after occurrence
            days_delay = rng.randint(0, 30)
            month_reported = month_occurred + (days_delay // 28)
            day_reported = (day_occurred + days_delay) % 28 + 1
            if month_reported > 12:
                month_reported = 12
                day_reported = 28

            date_occurred = f"{analysis_year}-{month_occurred:02d}-{day_occurred:02d}"
            date_reported = f"{analysis_year}-{month_reported:02d}-{day_reported:02d}"

            status = rng.choices(CLAIM_STATUSES, weights=[0.2, 0.5, 0.3])[0]
            severity = max(500, round(rng.gauss(severity_mean, severity_std), 2))

            if status == "closed":
                paid_amount = severity
                reserved_amount = 0.0
            elif status == "reserved":
                paid_frac = rng.uniform(0.1, 0.6)
                paid_amount = round(severity * paid_frac, 2)
                reserved_amount = round(severity * (1 - paid_frac), 2)
            else:  # open
                paid_amount = 0.0
                reserved_amount = round(severity * rng.uniform(0.7, 1.2), 2)

            region = rng.choice(REGIONS)
            claimant_age = rng.randint(18, 80)

            paid_by_type[ptype] += paid_amount
            reserved_by_type[ptype] += reserved_amount

            claims.append({
                "claim_id": cid,
                "policy_type": ptype,
                "date_occurred": date_occurred,
                "date_reported": date_reported,
                "status": status,
                "paid_amount": round(paid_amount, 2),
                "reserved_amount": round(reserved_amount, 2),
                "claimant_age": claimant_age,
                "region": region,
            })

    rng.shuffle(claims)

    # Round type totals
    for ptype in POLICY_TYPES:
        paid_by_type[ptype] = round(paid_by_type[ptype], 2)
        reserved_by_type[ptype] = round(reserved_by_type[ptype], 2)

    # --- Compute ground truth ---
    total_paid = round(sum(paid_by_type.values()), 2)
    total_reserved = round(sum(reserved_by_type.values()), 2)
    total_incurred = round(total_paid + total_reserved, 2)

    # Incurred by type
    incurred_by_type: dict[str, float] = {}
    for ptype in POLICY_TYPES:
        incurred_by_type[ptype] = round(paid_by_type[ptype] + reserved_by_type[ptype], 2)

    # Loss ratio by type = incurred / premium
    loss_ratios: dict[str, float] = {}
    for ptype in POLICY_TYPES:
        if annual_premiums[ptype] > 0:
            loss_ratios[ptype] = round(incurred_by_type[ptype] / annual_premiums[ptype], 4)
        else:
            loss_ratios[ptype] = 0.0

    overall_loss_ratio = round(total_incurred / total_premium, 4) if total_premium > 0 else 0.0

    # IBNR estimate using Bornhuetter-Ferguson simplified:
    # IBNR = Expected Ultimate * (1 - 1/LDF_12)
    # Expected Ultimate = Premium * Expected Loss Ratio
    ibnr_by_type: dict[str, float] = {}
    for ptype in POLICY_TYPES:
        expected_ultimate = annual_premiums[ptype] * expected_lrs[ptype]
        ldf_12 = ldfs[ptype]["12"]
        ibnr = round(expected_ultimate * (1 - 1 / ldf_12), 2)
        ibnr_by_type[ptype] = ibnr

    total_ibnr = round(sum(ibnr_by_type.values()), 2)

    # Ultimate losses = incurred + IBNR
    ultimate_by_type: dict[str, float] = {}
    for ptype in POLICY_TYPES:
        ultimate_by_type[ptype] = round(incurred_by_type[ptype] + ibnr_by_type[ptype], 2)

    total_ultimate = round(total_incurred + total_ibnr, 2)

    # Expense ratio assumption (for combined ratio)
    expense_ratio = round(rng.uniform(0.25, 0.35), 3)
    combined_ratio = round(overall_loss_ratio + expense_ratio, 4)

    # Worst performing type
    worst_type = max(loss_ratios, key=lambda k: loss_ratios[k])

    # Reserve adequacy: do held case reserves cover the total estimated remaining
    # liability (case reserves + IBNR) with the required safety margin?
    # Total required = (case_reserves_needed + IBNR) * (1 + margin%)
    # Since case_reserves_needed = total_reserved (by definition), the question is
    # whether held reserves cover themselves plus the IBNR with margin.
    total_required_reserve = round((total_reserved + total_ibnr) * (1 + reserve_margin_pct / 100), 2)
    # The company actually holds total_reserved plus some existing IBNR provision.
    # Assume the company's existing IBNR provision is a fraction of the true IBNR.
    ibnr_provision_ratio = rng.uniform(0.6, 1.4)
    existing_ibnr_provision = round(total_ibnr * ibnr_provision_ratio, 2)
    total_held_reserves = round(total_reserved + existing_ibnr_provision, 2)
    reserves_adequate = total_held_reserves >= total_required_reserve
    reserve_adequacy_str = "ADEQUATE" if reserves_adequate else "INADEQUATE"

    # Find a well-performing type for false-positive check
    best_type = min(loss_ratios, key=lambda k: loss_ratios[k])

    # Claim counts
    claim_counts = {t: 0 for t in POLICY_TYPES}
    for c in claims:
        claim_counts[c["policy_type"]] += 1
    total_claims = len(claims)

    # --- Build claims_register.csv ---
    claims_csv_lines = [
        "claim_id,policy_type,date_reported,date_occurred,status,paid_amount,reserved_amount,claimant_age,region"
    ]
    for c in claims:
        claims_csv_lines.append(
            f"{c['claim_id']},{c['policy_type']},{c['date_reported']},"
            f"{c['date_occurred']},{c['status']},{c['paid_amount']:.2f},"
            f"{c['reserved_amount']:.2f},{c['claimant_age']},{c['region']}"
        )
    claims_csv = "\n".join(claims_csv_lines) + "\n"

    # --- Build premium_data.csv ---
    month_names_short = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    prem_csv_lines = ["policy_type," + ",".join(month_names_short)]
    for ptype in POLICY_TYPES:
        vals = ",".join(f"{v:.2f}" for v in monthly_premiums[ptype])
        prem_csv_lines.append(f"{ptype},{vals}")
    premium_csv = "\n".join(prem_csv_lines) + "\n"

    # --- Build actuarial_tables.txt ---
    act_lines = [
        "ACTUARIAL REFERENCE TABLES",
        "",
        f"Company: {company_name}",
        f"Analysis Period: {analysis_year}",
        "",
        "=" * 60,
        "LOSS DEVELOPMENT FACTORS (LDFs)",
        "=" * 60,
        "",
        "LDFs represent the ratio of ultimate losses to losses known at a given",
        "maturity. Applied to incurred losses to estimate total ultimate losses.",
        "",
        f"{'Policy Type':<15} {'12-month':>10} {'24-month':>10} {'36-month':>10}",
        f"{'-'*15} {'-'*10} {'-'*10} {'-'*10}",
    ]
    for ptype in POLICY_TYPES:
        act_lines.append(
            f"{ptype:<15} {ldfs[ptype]['12']:>10.3f} {ldfs[ptype]['24']:>10.3f} {ldfs[ptype]['36']:>10.3f}"
        )
    act_lines.extend([
        "",
        "=" * 60,
        "EXPECTED LOSS RATIOS",
        "=" * 60,
        "",
        "Expected loss ratios are based on historical experience and industry",
        "benchmarks. Used in the Bornhuetter-Ferguson method for IBNR estimation.",
        "",
        f"{'Policy Type':<15} {'Expected LR':>12}",
        f"{'-'*15} {'-'*12}",
    ])
    for ptype in POLICY_TYPES:
        act_lines.append(f"{ptype:<15} {expected_lrs[ptype]:>11.1%}")
    act_lines.append("")
    actuarial_content = "\n".join(act_lines) + "\n"

    # --- Build reserving_guidelines.txt ---
    res_lines = [
        "RESERVING GUIDELINES AND METHODOLOGY",
        "",
        f"Company: {company_name}",
        "",
        "=" * 60,
        "IBNR ESTIMATION — BORNHUETTER-FERGUSON METHOD (SIMPLIFIED)",
        "=" * 60,
        "",
        "IBNR (Incurred But Not Reported) reserves account for claims that have",
        "occurred but have not yet been reported or fully developed.",
        "",
        "Formula:",
        "  Expected Ultimate = Earned Premium x Expected Loss Ratio",
        "  IBNR = Expected Ultimate x (1 - 1/LDF)",
        "",
        "  Where LDF is the Loss Development Factor at the current maturity.",
        "  For the current year's claims (12-month maturity), use the 12-month LDF.",
        "",
        "Steps:",
        "  1. Calculate earned premium by policy type",
        "  2. Multiply by the expected loss ratio to get expected ultimate losses",
        "  3. Apply the formula: IBNR = Expected Ultimate x (1 - 1/LDF_12)",
        "  4. Sum IBNR across all policy types",
        "",
        "=" * 60,
        "TOTAL ULTIMATE LOSSES",
        "=" * 60,
        "",
        "Ultimate Losses = Reported Incurred Losses + IBNR",
        "  Where Reported Incurred = Paid Losses + Case Reserves (outstanding)",
        "",
        "=" * 60,
        "COMBINED RATIO",
        "=" * 60,
        "",
        "Combined Ratio = Loss Ratio + Expense Ratio",
        f"  Expense Ratio for {company_name}: {expense_ratio:.1%}",
        "  (This includes all underwriting and administrative expenses.)",
        "",
        "  Combined Ratio < 100%: Underwriting profit",
        "  Combined Ratio > 100%: Underwriting loss",
        "",
        "=" * 60,
        "RESERVE ADEQUACY TEST",
        "=" * 60,
        "",
        f"Reserve Margin Requirement: {reserve_margin_pct}%",
        "",
        "  Total Required Reserve = (Case Reserves + IBNR Estimate) x (1 + margin%)",
        "  Total Held Reserves = Case Reserves + Existing IBNR Provision",
        "",
        "  Reserves are ADEQUATE if: Total Held >= Total Required",
        "  Reserves are INADEQUATE if: Total Held < Total Required",
        "",
        f"  Current IBNR provision on books: {_fmt_money(existing_ibnr_provision)}",
        "",
    ]
    reserving_content = "\n".join(res_lines) + "\n"

    # --- Problem statement ---
    problem_statement = f"""# Actuarial Claims Analysis

You are an actuarial analyst at {company_name} preparing the year-end
claims analysis for {analysis_year}. Analyze the claims data, compute loss ratios,
estimate IBNR reserves, and assess reserve adequacy.

## Source Files
- /testbed/data/claims_register.csv — All claims for the year: claim_id, policy_type, dates, status, paid and reserved amounts, claimant demographics
- /testbed/data/premium_data.csv — Monthly premiums collected by policy type
- /testbed/data/actuarial_tables.txt — Loss development factors and expected loss ratios by policy type
- /testbed/data/reserving_guidelines.txt — IBNR methodology (Bornhuetter-Ferguson), combined ratio formula, reserve adequacy test

## Requirements
1. Sum total paid losses and total incurred losses (paid + reserved) overall and by policy type
2. Calculate total earned premium by policy type and overall
3. Compute loss ratios by policy type and overall (incurred / premium)
4. Estimate IBNR reserves using the Bornhuetter-Ferguson method for each policy type
5. Compute total ultimate losses (incurred + IBNR)
6. Compute the combined ratio
7. Identify the worst-performing policy type (highest loss ratio)
8. Perform the reserve adequacy test using the {reserve_margin_pct}% margin requirement
9. Provide a summary assessment with recommendations

Write a detailed actuarial analysis report to /testbed/actuarial_report.txt
showing all calculations."""

    # --- Build rubric ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does the file /testbed/actuarial_report.txt exist and contain substantive content (more than a few lines)?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_total_paid",
            question=f"Does the report compute total paid losses as approximately {_fmt_money(total_paid)} (within $5,000)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_total_incurred",
            question=f"Does the report compute total incurred losses (paid + reserved) as approximately {_fmt_money(total_incurred)} (within $5,000)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_auto_loss_ratio",
            question=(
                f"Does the report compute the auto loss ratio as approximately "
                f"{loss_ratios['auto']*100:.1f}% (within 2 percentage points)? "
                f"(Auto incurred: {_fmt_money(incurred_by_type['auto'])}, "
                f"auto premium: {_fmt_money(annual_premiums['auto'])})"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_home_loss_ratio",
            question=(
                f"Does the report compute the home loss ratio as approximately "
                f"{loss_ratios['home']*100:.1f}% (within 2 percentage points)? "
                f"(Home incurred: {_fmt_money(incurred_by_type['home'])}, "
                f"home premium: {_fmt_money(annual_premiums['home'])})"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_commercial_loss_ratio",
            question=(
                f"Does the report compute the commercial loss ratio as approximately "
                f"{loss_ratios['commercial']*100:.1f}% (within 2 percentage points)? "
                f"(Commercial incurred: {_fmt_money(incurred_by_type['commercial'])}, "
                f"commercial premium: {_fmt_money(annual_premiums['commercial'])})"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_overall_loss_ratio",
            question=(
                f"Does the report compute the overall loss ratio as approximately "
                f"{overall_loss_ratio*100:.1f}% (within 2 percentage points)?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_ibnr_estimate",
            question=(
                f"Does the report estimate total IBNR reserves as approximately "
                f"{_fmt_money(total_ibnr)} (within $50,000)? "
                f"IBNR per type: auto={_fmt_money(ibnr_by_type['auto'])}, "
                f"home={_fmt_money(ibnr_by_type['home'])}, "
                f"commercial={_fmt_money(ibnr_by_type['commercial'])}."
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_ultimate_losses",
            question=(
                f"Does the report compute total ultimate losses as approximately "
                f"{_fmt_money(total_ultimate)} (within $50,000)? "
                f"(Incurred {_fmt_money(total_incurred)} + IBNR {_fmt_money(total_ibnr)})"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_combined_ratio",
            question=(
                f"Does the report compute the combined ratio as approximately "
                f"{combined_ratio*100:.1f}% (within 3 percentage points)? "
                f"(Loss ratio {overall_loss_ratio*100:.1f}% + expense ratio {expense_ratio*100:.1f}%)"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="identifies_worst_performing_type",
            question=(
                f"Does the report correctly identify '{worst_type}' as the worst-performing "
                f"policy type with the highest loss ratio of {loss_ratios[worst_type]*100:.1f}%?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_premium_total",
            question=f"Does the report compute total earned premium as approximately {_fmt_money(total_premium)} (within $50,000)?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_reserve_adequacy_assessment",
            question=(
                f"Does the report correctly assess reserves as {reserve_adequacy_str}? "
                f"(Total held reserves including IBNR provision: {_fmt_money(total_held_reserves)}, "
                f"required with {reserve_margin_pct}% margin: {_fmt_money(total_required_reserve)})"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name=f"no_false_inadequacy_on_{best_type}",
            question=(
                f"Does the report correctly show that '{best_type}' is the best-performing type "
                f"(lowest loss ratio of {loss_ratios[best_type]*100:.1f}%) and does NOT flag it "
                f"as problematic or inadequately reserved?"
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_claim_count",
            question=(
                f"Does the report correctly state the total number of claims as {total_claims} "
                f"(auto: {claim_counts['auto']}, home: {claim_counts['home']}, "
                f"commercial: {claim_counts['commercial']})?"
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="uses_bf_method",
            question=(
                "Does the report explicitly use or reference the Bornhuetter-Ferguson method "
                "for IBNR estimation, showing the formula IBNR = Expected Ultimate x (1 - 1/LDF)?"
            ),
            points=2,
        ),
        RubricCategory(
            name="analysis_quality",
            description="Is the actuarial report thorough, clearly organized, and does it provide sound professional analysis?",
            failure="Superficial analysis missing most calculations or major methodological errors.",
            minor_failure="Some calculations present but analysis is incomplete or has significant gaps.",
            minor_success="Most calculations correct with reasonable organization and methodology.",
            success="Professional-quality report with complete methodology, clear presentation, and actionable recommendations.",
            points=2,
        ),
    ]

    rubric = tuple(rubric_items)

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write your completed actuarial analysis report to /testbed/actuarial_report.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/claims_register.csv": claims_csv,
            "/testbed/data/premium_data.csv": premium_csv,
            "/testbed/data/actuarial_tables.txt": actuarial_content,
            "/testbed/data/reserving_guidelines.txt": reserving_content,
        },
        problem_type="actuarial_analysis",
    )


# =============================================================================
# DOMAIN: STATISTICAL EXPERIMENT (A/B TEST)
# =============================================================================

EXPERIMENT_HYPOTHESES = [
    "Changing the CTA button color from blue to green increases conversion rate",
    "Simplifying the checkout flow by removing one step increases conversion rate",
    "Adding social proof badges near the purchase button increases conversion rate",
    "Showing personalized product recommendations increases conversion rate",
    "Reducing page load time by 500ms increases conversion rate",
]

SEGMENTS = [
    ("device", ["mobile", "desktop"]),
    ("user_type", ["new", "returning"]),
    ("region", ["NA", "EU", "APAC"]),
]


# =============================================================================
# 3. STATISTICAL EXPERIMENT ANALYSIS
# =============================================================================


def make_statistical_experiment_analysis(rand_seed: int = 42) -> RubricDatapoint:
    """Analyze A/B test results with multiple metrics, segment analysis, and
    statistical significance computation.

    Seed varies: effect sizes (some significant, some not), sample sizes,
    whether the winner varies by segment (Simpson's paradox in some seeds!),
    number of segments, which secondary metrics show effects.
    """
    rng = _random.Random(rand_seed)

    company = pick1(COMPANY_NAMES, rand_seed)
    analyst = random_name(rand_seed)

    hypothesis = rng.choice(EXPERIMENT_HYPOTHESES)

    # --- Experiment parameters ---
    # Base conversion rate for control
    base_conv_rate = rng.uniform(0.02, 0.08)

    # True effect: sometimes significant, sometimes not
    # If rand_seed % 3 == 0, make effect clearly significant
    # If rand_seed % 3 == 1, make effect borderline
    # If rand_seed % 3 == 2, make effect null
    seed_class = rand_seed % 3
    if seed_class == 0:
        true_lift = rng.uniform(0.008, 0.020)  # strong effect
    elif seed_class == 1:
        true_lift = rng.uniform(0.002, 0.006)  # borderline
    else:
        true_lift = rng.uniform(-0.002, 0.002)  # null

    treatment_conv_rate = base_conv_rate + true_lift

    # --- Simpson's paradox: force it for certain seeds ---
    # If rand_seed % 5 == 0, create a Simpson's paradox in device segment
    simpsons_paradox = (rand_seed % 5 == 0)

    # Choose segments for this experiment
    n_segment_dims = rng.randint(2, 3)
    chosen_segments = rng.sample(SEGMENTS, n_segment_dims)

    # --- Generate segment results first, then derive aggregate from segments ---
    # We use the FIRST segment dimension to partition users; other dimensions
    # are generated independently (overlapping slices of the same users).
    # The aggregate totals are the sum across the first segment dimension's values.

    # Decide total user counts
    control_users = rng.randint(8000, 25000)
    treatment_users = rng.randint(int(control_users * 0.9), int(control_users * 1.1))

    # Revenue / secondary metric parameters (generated before segments so they're available)
    control_rev_per_user_base = rng.uniform(2.0, 8.0)
    treatment_rev_multiplier = 1.0 + rng.uniform(-0.05, 0.15)

    segment_rows: list[dict] = []
    # Track which segment dimension is used to derive the aggregate
    primary_segment_name = chosen_segments[0][0]

    for seg_idx, (seg_name, seg_values) in enumerate(chosen_segments):
        # Decide how users are split across segment values.
        # For the primary (first) segment, fractions must sum to 1.0 so the
        # aggregate equals the sum of segments.  For other segments the slices
        # overlap (like a different facet of the same users) so fractions are
        # independent.

        if seg_idx == 0:
            # --- Primary segment: fractions sum to 1 ---
            if seg_name == "device" and simpsons_paradox:
                # Simpson's paradox: treatment is overrepresented in the
                # LOW-converting segment (desktop) while control is
                # overrepresented in the HIGH-converting segment (mobile).
                # Within EACH segment, treatment does slightly worse than control.
                # But because treatment has more users in the high-volume low-rate
                # segment and fewer in the low-volume high-rate segment, the
                # weighted aggregate can flip.

                # mobile = high conversion, desktop = low conversion
                # Control user split: most in mobile
                mobile_control_frac = rng.uniform(0.55, 0.70)
                desktop_control_frac = 1.0 - mobile_control_frac
                # Treatment user split: most in desktop (the low-converting segment)
                mobile_treatment_frac = rng.uniform(0.30, 0.45)
                desktop_treatment_frac = 1.0 - mobile_treatment_frac

                seg_fracs_control = {"mobile": mobile_control_frac, "desktop": desktop_control_frac}
                seg_fracs_treatment = {"mobile": mobile_treatment_frac, "desktop": desktop_treatment_frac}

                # Conversion rates: mobile is high, desktop is low
                # Within each, treatment is slightly worse
                mobile_control_conv = base_conv_rate * rng.uniform(1.2, 1.5)
                mobile_treatment_conv = mobile_control_conv * rng.uniform(0.90, 0.98)
                desktop_control_conv = base_conv_rate * rng.uniform(0.6, 0.8)
                desktop_treatment_conv = desktop_control_conv * rng.uniform(0.90, 0.98)

                seg_conv_rates_control = {"mobile": mobile_control_conv, "desktop": desktop_control_conv}
                seg_conv_rates_treatment = {"mobile": mobile_treatment_conv, "desktop": desktop_treatment_conv}
            else:
                # Normal: generate random fractions that sum to 1
                raw_fracs = [rng.uniform(0.2, 0.8) for _ in seg_values]
                total_raw = sum(raw_fracs)
                seg_fracs_control = {v: f / total_raw for v, f in zip(seg_values, raw_fracs)}
                # Treatment fracs: slight jitter but also sum to 1
                raw_fracs_t = [seg_fracs_control[v] * rng.uniform(0.9, 1.1) for v in seg_values]
                total_raw_t = sum(raw_fracs_t)
                seg_fracs_treatment = {v: f / total_raw_t for v, f in zip(seg_values, raw_fracs_t)}

                seg_conv_rates_control = {v: base_conv_rate * rng.uniform(0.7, 1.4) for v in seg_values}
                seg_conv_rates_treatment = {v: treatment_conv_rate * rng.uniform(0.7, 1.4) for v in seg_values}

            # Generate segment-level users and conversions
            for seg_val in seg_values:
                seg_c_users = max(100, int(control_users * seg_fracs_control[seg_val]))
                seg_t_users = max(100, int(treatment_users * seg_fracs_treatment[seg_val]))

                seg_c_conv = 0
                for _ in range(seg_c_users):
                    if rng.random() < seg_conv_rates_control[seg_val]:
                        seg_c_conv += 1

                seg_t_conv = 0
                for _ in range(seg_t_users):
                    if rng.random() < seg_conv_rates_treatment[seg_val]:
                        seg_t_conv += 1

                seg_c_rev = round(seg_c_users * control_rev_per_user_base * rng.uniform(0.8, 1.2), 2)
                seg_t_rev = round(seg_t_users * control_rev_per_user_base * treatment_rev_multiplier * rng.uniform(0.8, 1.2), 2)

                segment_rows.append({
                    "variant": "control",
                    "segment_name": seg_name,
                    "segment_value": seg_val,
                    "users": seg_c_users,
                    "conversions": seg_c_conv,
                    "revenue": seg_c_rev,
                })
                segment_rows.append({
                    "variant": "treatment",
                    "segment_name": seg_name,
                    "segment_value": seg_val,
                    "users": seg_t_users,
                    "conversions": seg_t_conv,
                    "revenue": seg_t_rev,
                })
        else:
            # --- Non-primary segments: fractions are independent (overlapping slices) ---
            for seg_val in seg_values:
                seg_frac = rng.uniform(0.2, 0.6)
                seg_control_frac = seg_frac
                seg_treatment_frac = seg_frac * rng.uniform(0.9, 1.1)
                seg_control_conv_rate = base_conv_rate * rng.uniform(0.7, 1.4)
                seg_treatment_conv_rate = treatment_conv_rate * rng.uniform(0.7, 1.4)

                seg_c_users = max(100, int(control_users * seg_control_frac))
                seg_t_users = max(100, int(treatment_users * seg_treatment_frac))

                seg_c_conv = 0
                for _ in range(seg_c_users):
                    if rng.random() < seg_control_conv_rate:
                        seg_c_conv += 1

                seg_t_conv = 0
                for _ in range(seg_t_users):
                    if rng.random() < seg_treatment_conv_rate:
                        seg_t_conv += 1

                seg_c_rev = round(seg_c_users * control_rev_per_user_base * rng.uniform(0.8, 1.2), 2)
                seg_t_rev = round(seg_t_users * control_rev_per_user_base * treatment_rev_multiplier * rng.uniform(0.8, 1.2), 2)

                segment_rows.append({
                    "variant": "control",
                    "segment_name": seg_name,
                    "segment_value": seg_val,
                    "users": seg_c_users,
                    "conversions": seg_c_conv,
                    "revenue": seg_c_rev,
                })
                segment_rows.append({
                    "variant": "treatment",
                    "segment_name": seg_name,
                    "segment_value": seg_val,
                    "users": seg_t_users,
                    "conversions": seg_t_conv,
                    "revenue": seg_t_rev,
                })

    # --- Derive aggregate totals from the primary (first) segment dimension ---
    # Sum users and conversions across all values of the primary segment
    control_conversions = sum(
        r["conversions"] for r in segment_rows
        if r["variant"] == "control" and r["segment_name"] == primary_segment_name
    )
    treatment_conversions = sum(
        r["conversions"] for r in segment_rows
        if r["variant"] == "treatment" and r["segment_name"] == primary_segment_name
    )
    control_users = sum(
        r["users"] for r in segment_rows
        if r["variant"] == "control" and r["segment_name"] == primary_segment_name
    )
    treatment_users = sum(
        r["users"] for r in segment_rows
        if r["variant"] == "treatment" and r["segment_name"] == primary_segment_name
    )

    # Revenue and secondary metrics (generated independently)
    control_total_revenue = round(control_users * control_rev_per_user_base * rng.uniform(0.95, 1.05), 2)
    treatment_total_revenue = round(
        treatment_users * control_rev_per_user_base * treatment_rev_multiplier * rng.uniform(0.95, 1.05), 2
    )

    # Time on site (seconds)
    control_avg_time = round(rng.uniform(60, 180), 1)
    treatment_avg_time = round(control_avg_time * rng.uniform(0.9, 1.15), 1)

    # Bounce rate
    control_bounces = rng.randint(int(control_users * 0.3), int(control_users * 0.5))
    treatment_bounces = rng.randint(int(treatment_users * 0.25), int(treatment_users * 0.5))

    # --- Compute ground truth statistics ---
    obs_control_rate = control_conversions / control_users if control_users > 0 else 0.0
    obs_treatment_rate = treatment_conversions / treatment_users if treatment_users > 0 else 0.0
    absolute_lift = obs_treatment_rate - obs_control_rate
    relative_lift_pct = (absolute_lift / obs_control_rate * 100) if obs_control_rate > 0 else 0.0

    # Z-test for proportions
    pooled_rate = (control_conversions + treatment_conversions) / (control_users + treatment_users)
    se = math.sqrt(pooled_rate * (1 - pooled_rate) * (1/control_users + 1/treatment_users))
    z_score = (obs_treatment_rate - obs_control_rate) / se if se > 0 else 0.0

    # Two-tailed p-value approximation using normal CDF
    # Use a simple approximation for the standard normal CDF
    def _norm_cdf(x: float) -> float:
        """Approximation of standard normal CDF (Abramowitz & Stegun)."""
        if x < -8:
            return 0.0
        if x > 8:
            return 1.0
        t = 1.0 / (1.0 + 0.2316419 * abs(x))
        poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
               + t * (-1.821255978 + t * 1.330274429))))
        cdf = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x) * poly
        return cdf if x >= 0 else 1.0 - cdf

    p_value = 2.0 * (1.0 - _norm_cdf(abs(z_score)))
    p_value = round(p_value, 4)

    # Number of metrics we're testing: conversion, revenue/user, time on site, bounce rate = 4
    n_metrics = 4
    bonferroni_threshold = round(0.05 / n_metrics, 4)
    significant_primary = p_value < bonferroni_threshold

    # Revenue per user
    control_rpu = round(control_total_revenue / control_users, 4) if control_users > 0 else 0.0
    treatment_rpu = round(treatment_total_revenue / treatment_users, 4) if treatment_users > 0 else 0.0

    # Bounce rates
    control_bounce_rate = round(control_bounces / control_users, 4) if control_users > 0 else 0.0
    treatment_bounce_rate = round(treatment_bounces / treatment_users, 4) if treatment_users > 0 else 0.0

    # Required sample size check (for 80% power, two-sided, at 5% significance)
    # n = (z_alpha/2 + z_beta)^2 * (p1(1-p1) + p2(1-p2)) / (p2 - p1)^2
    # With MDE given by experiment design
    mde = round(rng.uniform(0.005, 0.015), 4)
    z_alpha = 1.96  # for 95% confidence
    z_beta = 0.84   # for 80% power
    p1 = obs_control_rate
    p2 = p1 + mde
    if mde > 0:
        required_n_per_arm = math.ceil(
            (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (mde ** 2)
        )
    else:
        required_n_per_arm = 999999
    sample_adequate = min(control_users, treatment_users) >= required_n_per_arm

    # Check for Simpson's paradox in segment data
    simpsons_detected = False
    simpsons_segment_desc = ""
    if simpsons_paradox:
        # Check if overall treatment is better but within-segment treatment is worse
        # This is specifically for the device segment
        for seg_name, seg_values in chosen_segments:
            if seg_name == "device":
                all_worse = True
                for sv in seg_values:
                    c_rows = [r for r in segment_rows if r["variant"] == "control"
                              and r["segment_name"] == seg_name and r["segment_value"] == sv]
                    t_rows = [r for r in segment_rows if r["variant"] == "treatment"
                              and r["segment_name"] == seg_name and r["segment_value"] == sv]
                    if c_rows and t_rows:
                        c_rate = c_rows[0]["conversions"] / c_rows[0]["users"] if c_rows[0]["users"] > 0 else 0
                        t_rate = t_rows[0]["conversions"] / t_rows[0]["users"] if t_rows[0]["users"] > 0 else 0
                        if t_rate >= c_rate:
                            all_worse = False
                if all_worse and obs_treatment_rate > obs_control_rate:
                    simpsons_detected = True
                    simpsons_segment_desc = (
                        f"Simpson's paradox in the 'device' segment: overall treatment wins, "
                        f"but treatment performs worse than control within both mobile and desktop segments"
                    )
                elif all_worse and obs_treatment_rate <= obs_control_rate:
                    # No paradox if overall also shows treatment worse
                    simpsons_detected = False
                else:
                    simpsons_detected = False

    # Pick a secondary metric that definitely isn't significant for false-positive check
    # Use bounce rate z-test
    pooled_bounce = (control_bounces + treatment_bounces) / (control_users + treatment_users)
    se_bounce = math.sqrt(pooled_bounce * (1 - pooled_bounce) * (1/control_users + 1/treatment_users))
    z_bounce = (treatment_bounce_rate - control_bounce_rate) / se_bounce if se_bounce > 0 else 0.0
    p_bounce = 2.0 * (1.0 - _norm_cdf(abs(z_bounce)))
    bounce_significant = p_bounce < bonferroni_threshold

    # Round computed values
    obs_control_rate = round(obs_control_rate, 4)
    obs_treatment_rate = round(obs_treatment_rate, 4)
    absolute_lift = round(absolute_lift, 4)
    relative_lift_pct = round(relative_lift_pct, 2)
    z_score = round(z_score, 2)

    # --- Build experiment_design.txt ---
    design_lines = [
        "A/B EXPERIMENT DESIGN DOCUMENT",
        "",
        f"Company: {company}",
        f"Analyst: {analyst}",
        f"Experiment ID: EXP-{rng.randint(1000, 9999)}",
        "",
        "=" * 60,
        "HYPOTHESIS",
        "=" * 60,
        "",
        f"H0: {hypothesis.replace('increases', 'does not change')}",
        f"H1: {hypothesis}",
        "",
        "=" * 60,
        "EXPERIMENT SETUP",
        "=" * 60,
        "",
        "Control: Current implementation (no change)",
        "Treatment: Modified implementation per hypothesis",
        f"Traffic Split: ~50/50 (randomized by user ID)",
        "",
        "PRIMARY METRIC: Conversion Rate (purchases / unique visitors)",
        "",
        "SECONDARY METRICS:",
        "  - Revenue per User (total revenue / unique visitors)",
        "  - Average Time on Site (seconds)",
        "  - Bounce Rate (single-page sessions / total sessions)",
        "",
        f"Minimum Detectable Effect (MDE): {mde*100:.2f}% absolute change in conversion rate",
        f"Significance Level: 0.05 (5%)",
        f"Power: 80% (beta = 0.20)",
        "",
        f"Required Sample Size per Arm: {required_n_per_arm:,} users",
        f"  (Based on base rate ~{obs_control_rate*100:.1f}% and MDE of {mde*100:.2f}%)",
        "",
        "MULTIPLE COMPARISON CORRECTION:",
        "  Since we are testing 4 metrics simultaneously, apply the Bonferroni",
        "  correction to control the family-wise error rate:",
        f"  Adjusted significance threshold = 0.05 / {n_metrics} = {bonferroni_threshold}",
        "  A metric is significant only if p-value < {:.4f}".format(bonferroni_threshold),
        "",
    ]
    design_content = "\n".join(design_lines) + "\n"

    # --- Build results_summary.csv ---
    results_lines = [
        "variant,users,conversions,total_revenue,avg_time_seconds,bounces",
        f"control,{control_users},{control_conversions},{control_total_revenue:.2f},{control_avg_time},{control_bounces}",
        f"treatment,{treatment_users},{treatment_conversions},{treatment_total_revenue:.2f},{treatment_avg_time},{treatment_bounces}",
    ]
    results_csv = "\n".join(results_lines) + "\n"

    # --- Build segment_results.csv ---
    seg_csv_lines = ["variant,segment,users,conversions,revenue"]
    for row in segment_rows:
        seg_csv_lines.append(
            f"{row['variant']},{row['segment_name']}:{row['segment_value']},"
            f"{row['users']},{row['conversions']},{row['revenue']:.2f}"
        )
    segment_csv = "\n".join(seg_csv_lines) + "\n"

    # --- Build statistical_reference.txt ---
    stat_lines = [
        "STATISTICAL REFERENCE FOR A/B TEST ANALYSIS",
        "",
        "=" * 60,
        "Z-TEST FOR DIFFERENCE IN PROPORTIONS",
        "=" * 60,
        "",
        "Used to test whether two proportions (e.g., conversion rates) differ.",
        "",
        "  p_control = conversions_control / n_control",
        "  p_treatment = conversions_treatment / n_treatment",
        "",
        "  Pooled proportion:",
        "    p_pooled = (conversions_control + conversions_treatment) / (n_control + n_treatment)",
        "",
        "  Standard error:",
        "    SE = sqrt( p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment) )",
        "",
        "  Z-statistic:",
        "    Z = (p_treatment - p_control) / SE",
        "",
        "  Two-tailed p-value: P = 2 * (1 - Phi(|Z|))",
        "  where Phi is the standard normal CDF.",
        "",
        "  Common Z to p-value reference:",
        "    |Z| = 1.645 -> p = 0.10",
        "    |Z| = 1.960 -> p = 0.05",
        "    |Z| = 2.326 -> p = 0.02",
        "    |Z| = 2.576 -> p = 0.01",
        "    |Z| = 3.090 -> p = 0.002",
        "    |Z| = 3.291 -> p = 0.001",
        "",
        "=" * 60,
        "T-TEST FOR DIFFERENCE IN MEANS",
        "=" * 60,
        "",
        "Used for continuous metrics (revenue per user, time on site).",
        "",
        "  t = (mean_treatment - mean_control) / SE",
        "  SE = sqrt(s_control^2/n_control + s_treatment^2/n_treatment)",
        "",
        "  For large samples (n > 30), the t-distribution approximates the normal,",
        "  so Z-test critical values can be used.",
        "",
        "=" * 60,
        "BONFERRONI CORRECTION",
        "=" * 60,
        "",
        "When testing multiple hypotheses simultaneously, the probability of",
        "at least one false positive increases. The Bonferroni correction adjusts",
        "the significance threshold:",
        "",
        "  Adjusted alpha = alpha / number_of_tests",
        "",
        "  Example: Testing 4 metrics at alpha=0.05:",
        "    Adjusted alpha = 0.05 / 4 = 0.0125",
        "    Each individual test must have p < 0.0125 to be significant.",
        "",
        "=" * 60,
        "PRACTICAL SIGNIFICANCE",
        "=" * 60,
        "",
        "Statistical significance alone is not sufficient. Also consider:",
        "  - Is the effect size practically meaningful for the business?",
        "  - Does the lift exceed the minimum detectable effect (MDE)?",
        "  - Are results consistent across key segments?",
        "  - Are there any segment anomalies (e.g., Simpson's paradox)?",
        "",
        "Simpson's Paradox: A trend that appears in aggregate data can reverse",
        "when the data is split into subgroups. This can occur when the subgroups",
        "have very different sizes and different base rates.",
        "",
    ]
    stat_content = "\n".join(stat_lines) + "\n"

    # --- Problem statement ---
    problem_statement = f"""# A/B Test Statistical Analysis

You are a data scientist at {company}. Analyze the results of an A/B test
and provide a rigorous statistical assessment.

## Source Files
- /testbed/data/experiment_design.txt — Experiment hypothesis, metrics, MDE, sample size requirements, and multiple comparison correction method
- /testbed/data/results_summary.csv — Aggregate results per variant: users, conversions, revenue, time on site, bounces
- /testbed/data/segment_results.csv — Results broken down by segment: device type, user type, region
- /testbed/data/statistical_reference.txt — Z-test formula, t-test formula, Bonferroni correction, practical significance guidance

## Requirements
1. Compute conversion rates for control and treatment
2. Compute absolute and relative lift in conversion rate
3. Perform a Z-test for the primary metric (conversion rate): compute Z-score and p-value
4. Apply Bonferroni correction and determine if the primary metric is statistically significant
5. Compute revenue per user for each variant
6. Analyze segment results — look for anomalies such as Simpson's paradox
7. Verify sample size adequacy (is actual sample >= required per the experiment design?)
8. Compute bounce rates for each variant
9. Provide an overall recommendation (ship, iterate, or discard the treatment)

Write a comprehensive statistical analysis report to /testbed/experiment_report.txt
showing all calculations, statistical tests, and conclusions."""

    # --- Build rubric ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does the file /testbed/experiment_report.txt exist and contain substantive content (more than a few lines)?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_control_conversion_rate",
            question=(
                f"Does the report compute the control conversion rate as approximately "
                f"{obs_control_rate*100:.2f}% "
                f"({control_conversions} / {control_users}) (within 0.1 percentage points)?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_treatment_conversion_rate",
            question=(
                f"Does the report compute the treatment conversion rate as approximately "
                f"{obs_treatment_rate*100:.2f}% "
                f"({treatment_conversions} / {treatment_users}) (within 0.1 percentage points)?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_absolute_lift",
            question=(
                f"Does the report compute the absolute lift (treatment - control) as approximately "
                f"{absolute_lift*100:.2f} percentage points (within 0.1 percentage points)?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_relative_lift_pct",
            question=(
                f"Does the report compute the relative lift as approximately "
                f"{relative_lift_pct:.1f}% (within 1 percentage point)?"
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_z_score",
            question=(
                f"Does the report compute the Z-score for the primary metric (conversion rate) "
                f"as approximately {z_score:.2f} (within 0.1)?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_p_value",
            question=(
                f"Does the report compute the p-value for the primary metric as approximately "
                f"{p_value:.4f} (within 0.01)? For reference, the Z-score is {z_score:.2f}."
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_significance_determination",
            question=(
                f"Does the report correctly determine that the primary metric (conversion rate) "
                f"{'IS' if significant_primary else 'is NOT'} statistically significant "
                f"after Bonferroni correction (p={p_value:.4f} vs threshold {bonferroni_threshold})?"
            ),
            points=3,
        ),
        BinaryRubricCategory(
            name="correct_revenue_per_user_control",
            question=(
                f"Does the report compute the control revenue per user as approximately "
                f"${control_rpu:.2f} (within $0.20)?"
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_revenue_per_user_treatment",
            question=(
                f"Does the report compute the treatment revenue per user as approximately "
                f"${treatment_rpu:.2f} (within $0.20)?"
            ),
            points=1,
        ),
    ]

    if simpsons_detected:
        rubric_items.append(
            BinaryRubricCategory(
                name="identifies_segment_anomaly",
                question=(
                    f"Does the report identify the Simpson's paradox in the segment data? "
                    f"{simpsons_segment_desc}. The report should note that aggregate results "
                    f"and within-segment results tell different stories."
                ),
                points=3,
            )
        )
    else:
        rubric_items.append(
            BinaryRubricCategory(
                name="identifies_segment_anomaly",
                question=(
                    "Does the report correctly note that there is no Simpson's paradox or major "
                    "anomaly in the segment-level results (segment trends are broadly consistent "
                    "with the aggregate)?"
                ),
                points=2,
            )
        )

    rubric_items.extend([
        BinaryRubricCategory(
            name="correct_bonferroni_threshold",
            question=(
                f"Does the report correctly state the Bonferroni-corrected significance threshold "
                f"as {bonferroni_threshold} (0.05 / {n_metrics} metrics)?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_sample_size_adequate",
            question=(
                f"Does the report correctly determine that the sample size is "
                f"{'ADEQUATE' if sample_adequate else 'INADEQUATE'} "
                f"(actual min arm: {min(control_users, treatment_users):,}, "
                f"required: {required_n_per_arm:,})?"
            ),
            points=2,
        ),
    ])

    if not bounce_significant:
        rubric_items.append(
            BinaryRubricCategory(
                name="no_false_significance_on_bounce_rate",
                question=(
                    f"Does the report correctly note that the bounce rate difference "
                    f"(control: {control_bounce_rate*100:.1f}%, treatment: {treatment_bounce_rate*100:.1f}%) "
                    f"is NOT statistically significant after Bonferroni correction? "
                    f"(The report should not claim bounce rate shows a significant effect.)"
                ),
                points=2,
            )
        )
    else:
        rubric_items.append(
            BinaryRubricCategory(
                name="bounce_rate_significance",
                question=(
                    f"Does the report correctly identify the bounce rate difference "
                    f"(control: {control_bounce_rate*100:.1f}%, treatment: {treatment_bounce_rate*100:.1f}%) "
                    f"as statistically significant after Bonferroni correction?"
                ),
                points=2,
            )
        )

    rubric_items.extend([
        BinaryRubricCategory(
            name="correct_control_bounce_rate",
            question=(
                f"Does the report compute the control bounce rate as approximately "
                f"{control_bounce_rate*100:.1f}% (within 1 percentage point)?"
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_treatment_bounce_rate",
            question=(
                f"Does the report compute the treatment bounce rate as approximately "
                f"{treatment_bounce_rate*100:.1f}% (within 1 percentage point)?"
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="shows_z_test_formula",
            question=(
                "Does the report show or explain the Z-test formula for proportions "
                "(including pooled proportion and standard error computation)?"
            ),
            points=1,
        ),
        RubricCategory(
            name="methodology_quality",
            description="Is the statistical methodology rigorous, well-explained, and does the report draw appropriate conclusions?",
            failure="Major statistical errors or conclusions not supported by the analysis.",
            minor_failure="Some statistical tests performed but methodology has gaps or errors.",
            minor_success="Sound methodology with most tests correct; conclusions mostly supported.",
            success="Rigorous statistical analysis with clear methodology, appropriate corrections, segment analysis, and well-supported conclusions.",
            points=2,
        ),
    ])

    rubric = tuple(rubric_items)

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write your completed statistical analysis report to /testbed/experiment_report.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/experiment_design.txt": design_content,
            "/testbed/data/results_summary.csv": results_csv,
            "/testbed/data/segment_results.csv": segment_csv,
            "/testbed/data/statistical_reference.txt": stat_content,
        },
        problem_type="statistical_experiment_analysis",
    )
