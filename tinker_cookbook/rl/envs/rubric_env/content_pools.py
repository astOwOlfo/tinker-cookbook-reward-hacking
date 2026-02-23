"""Reusable content pools for hybrid factory parameterization.

Each pool is a list of variants. Factories use `pick()` to deterministically
select items based on a seed, and `vary_number()` to add numeric jitter.

This module provides shared names, companies, and utility functions used by
multiple problem factories. Domain-specific pools live in their respective
problem modules.
"""

import random as _random
from typing import TypeVar

T = TypeVar("T")


def pick(pool: list[T], seed: int, n: int = 1) -> list[T]:
    """Deterministically pick n items from a pool without replacement."""
    rng = _random.Random(seed)
    return rng.sample(pool, min(n, len(pool)))


def pick_one(pool: list[T], seed: int) -> T:
    """Deterministically pick one item from a pool."""
    rng = _random.Random(seed)
    return rng.choice(pool)


def vary_number(base: float, seed: int, pct: float = 0.2) -> float:
    """Deterministically vary a number by up to ±pct."""
    rng = _random.Random(seed)
    return base * (1 + rng.uniform(-pct, pct))


def vary_int(base: int, seed: int, pct: float = 0.2) -> int:
    """Deterministically vary an integer by up to ±pct."""
    return int(round(vary_number(base, seed, pct)))


# =============================================================================
# PERSON NAMES
# =============================================================================

FIRST_NAMES = [
    "Alice", "Bob", "Carla", "Dan", "Eva", "Frank", "Grace", "Henry",
    "Irene", "Jack", "Kara", "Leo", "Mia", "Nick", "Olivia", "Peter",
    "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier",
    "Yuki", "Zara", "Aaron", "Beth", "Carlos", "Diana", "Ethan", "Fiona",
    "George", "Hannah", "Ivan", "Julia", "Keith", "Luna", "Marcus", "Nora",
    "Oscar", "Priya", "Raj", "Sara", "Tom", "Ursula", "Vera", "Will",
]

LAST_NAMES = [
    "Martin", "Chen", "Diaz", "Foster", "Green", "Hall", "Ito", "Jain",
    "Kim", "Lee", "Moss", "Nash", "Owens", "Park", "Quinn", "Rivera",
    "Smith", "Torres", "Ueda", "Vasquez", "Webb", "Xu", "Young", "Zhang",
    "Adams", "Brown", "Clark", "Davis", "Evans", "Fisher", "Garcia",
    "Harris", "Jackson", "Kumar", "Lopez", "Miller", "Nelson", "Ortiz",
    "Patel", "Roberts", "Singh", "Thompson", "Walker", "Wright", "Yang",
]


def make_name(seed: int) -> str:
    """Generate a deterministic full name."""
    rng = _random.Random(seed)
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def make_names(seed: int, n: int) -> list[str]:
    """Generate n deterministic unique full names."""
    rng = _random.Random(seed)
    names = set()
    while len(names) < n:
        names.add(f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}")
    return sorted(names)


# =============================================================================
# COMPANY NAMES
# =============================================================================

COMPANY_NAMES = [
    "Acme Corp", "Globex Industries", "Initech Solutions", "Northridge Outdoor Co.",
    "Sterling Analytics", "Cascade Systems", "Horizon Biotech", "Pinnacle Financial",
    "Meridian Health", "Summit Consulting", "Vanguard Manufacturing", "Atlas Logistics",
    "Beacon Technologies", "Crestview Partners", "Delta Engineering", "Evergreen Services",
    "Frontier Solutions", "GreenLeaf Energy", "Highland Medical", "Ironwood Capital",
    "Keystone Software", "Lakeview Dynamics", "Maple Street Media", "Nexus Innovations",
    "Oceanview Properties", "Pacific Rim Trading", "Quantum Analytics", "Redwood Research",
    "Silverline Consulting", "Tidewater Systems", "Unified Networks", "Vertex Pharma",
]


# =============================================================================
# CONVENIENCE ALIASES
# =============================================================================

random_name = make_name
random_names = make_names
pick1 = pick_one
