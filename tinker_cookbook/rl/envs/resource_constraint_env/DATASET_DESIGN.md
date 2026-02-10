# Dataset Design for Resource Constraint Environment

Guide for understanding and extending the problem dataset in `dataset.py`.

---

## What This Dataset Is

A collection of **~770 self-contained Python optimization problems** used to train/evaluate models on single-core code optimization. Each problem ships as a complete `.py` file with a `run()` function. The model's job: make `run()` faster without breaking correctness or escaping the single-CPU constraint.

The evaluation harness (see `REWARD_DESIGN.md`) times the model's `run()` against the starter code's `run()` in parallel, producing a speedup ratio.

---

## Core Data Type

```python
@dataclass(frozen=True, slots=True)
class ResourceConstraintDatapoint:
    problem_id: str              # Unique ID, e.g. "dijkstra_s42"
    problem_description: str     # Markdown shown to model
    starter_code: str            # Complete Python file with run()
    random_seed: int | None      # Seed baked into starter_code (None if deterministic)
    input_params: dict[str, Any] # Parameters baked in (for analysis)
    problem_type: str | None     # Category label
```

The `starter_code` field is the most important. It's a **complete, runnable Python file** that the model receives as its starting point. It also doubles as the **baseline** that the harness times against.

---

## The run() Boundary

**Critical invariant:** All input data lives at **module level**. The `run()` function contains **only the algorithm**.

```python
# Module level — FIXED, model must not touch
random.seed(42)
POINTS = [(random.uniform(-100, 100), random.uniform(-100, 100)) for _ in range(5000)]
INIT_CENTERS = [(random.uniform(-100, 100), random.uniform(-100, 100)) for _ in range(10)]

# Inside run() — MUTABLE, model optimizes this
def run():
    points = POINTS
    centers = list(INIT_CENTERS)
    # ... k-means algorithm ...
    return total_dist
```

This enables a clean instruction to the model: **"Code inside `run()` is mutable. Don't touch anything outside it."**

### What goes where

| Outside `run()` (fixed) | Inside `run()` (mutable) |
|---|---|
| `import` statements | The algorithm itself |
| Constants (`N`, `GRID_SIZE`, etc.) | Local variables |
| `random.seed(X)` | Computation loops |
| All randomly-generated data (`POINTS`, `GRAPH`, `MATRIX`, etc.) | Return value |
| Deterministic input data (`TEXT`, `SEQ`, `INIT_U`, etc.) | |
| Helper functions used by the algorithm (`cross()`, `edit_distance()`, etc.) | |

### Edge case: Monte Carlo / Random Walk / Bootstrap

For problems where **random sampling IS the algorithm** (Monte Carlo pi, Monte Carlo integration, random walk, bootstrap mean), `random.seed()` is at module level but the `random.random()` calls stay inside `run()`. The model may switch from `random` to `numpy` or use other RNG strategies — that's legitimate optimization.

### Edge case: Naive Bayes

Training statistics (means, variances) are pre-computed at module level. Only the classification loop is inside `run()`.

---

## Factory Functions

Each problem has a **factory function** `make_<name>(...)` that returns a `ResourceConstraintDatapoint`. The factory generates the `starter_code` as an f-string with parameters baked in.

```python
def make_dijkstra(
    seed: int = 42,
    num_vertices: int = 300,
    edge_probability: float = 0.15,
) -> ResourceConstraintDatapoint:
    return ResourceConstraintDatapoint(
        problem_id=f"dijkstra_s{seed}",
        starter_code=f'''...''',
        random_seed=seed,
        ...
    )
```

### Two kinds of factories

1. **Seeded** (`random_seed != None`): Take a `seed` parameter. Produce different random inputs per seed. Variants come from `build_all_problems()` iterating over 15 programmatic seeds.

2. **Deterministic** (`random_seed = None`): Take size/shape parameters. Variants come from `_log_spaced_ints()` producing ~5 different sizes.

---

## Problem Inventory (72 factories)

### Seeded factories (41) — 15 seeds each = 615 problems

| Category | Factories |
|---|---|
| **Monte Carlo** | `monte_carlo_pi`, `monte_carlo_integration` |
| **Simulation** | `random_walk`, `nbody_simulation`, `game_of_life`, `spring_network` |
| **Matrix / LA** | `matrix_multiplication`, `gaussian_elimination`, `power_iteration`, `lu_decomposition` |
| **Graph** | `floyd_warshall`, `dijkstra`, `bfs_shortest_paths`, `connected_components`, `topological_sort_count`, `pagerank`, `minimum_spanning_tree` |
| **Geometry** | `convex_hull`, `closest_pair`, `point_in_polygon`, `delaunay_check` |
| **Physics** | `heat_equation` |
| **ML / Stats** | `kmeans`, `knn_classify`, `linear_regression`, `correlation_matrix`, `bootstrap_mean`, `naive_bayes` |
| **String** | `levenshtein_matrix`, `longest_common_subsequence`, `suffix_array` |
| **DP** | `knapsack`, `edit_distance_batch`, `longest_increasing_subsequence`, `matrix_chain` |
| **Image / Grid** | `image_convolution`, `flood_fill`, `distance_transform`, `edge_detection` |
| **Crypto** | `password_hash` |
| **Sorting** | `large_sort` |

### Deterministic factories (31) — ~5 sizes each = ~155 problems

| Category | Factories |
|---|---|
| **Arithmetic** | `sum_of_squares`, `digit_sum`, `harmonic_sum`, `dot_product`, `prefix_sum`, `triangular_numbers` |
| **Number Theory** | `fibonacci`, `collatz`, `count_divisors`, `factorial_digits`, `gcd_pairs`, `perfect_numbers`, `prime_sum`, `palindrome_count`, `power_mod`, `sieve_count`, `prime_search`, `partition_count`, `euler_totient_sum`, `catalan_numbers` |
| **Fractals** | `mandelbrot`, `julia_set` |
| **String** | `string_hash`, `pattern_matching_count`, `run_length_encode` |
| **Matrix** | `matrix_trace` |
| **Search / Sort** | `binary_search_count`, `bubble_sort_swaps` |
| **Numerical** | `numerical_integration`, `newtons_method` |
| **Physics** | `wave_equation` |

---

## Variant Generation System

### Seeds

```python
def make_seeds(n: int = 10) -> list[int]:
    rng = _random.Random(0)           # fixed meta-seed
    return [rng.randint(0, 2**31 - 1) for _ in range(n)]

DEFAULT_NUM_SEEDS = 15
```

All seeded factories share the **same 15 seeds**. This is intentional: it means two different problem types with the same seed have independent random streams (each factory calls `random.seed(X)` inside its own starter code).

### Size variants

```python
def _log_spaced_ints(low: int, high: int, n: int) -> list[int]:
```

Log-spacing avoids bunching at the low end. For example, `_log_spaced_ints(200, 1000, 5)` might yield `[200, 299, 447, 669, 1000]`.

---

## Starter Code Conventions

1. **Naive implementations only.** Starter code should be first-pass, textbook, or brute-force. O(n^2) when O(n log n) exists. Left Riemann sum, not Simpson's. No numpy in the starter if the naive approach uses stdlib. The point is to leave headroom for the model to optimize.

2. **Self-contained.** Every `starter_code` is a complete Python file. All imports at the top, all data at module level, `run()` defined, `if __name__ == "__main__"` block at the bottom.

3. **Deterministic output.** Given the same starter code, `run()` must return the same result every time. This is how the harness validates correctness (within 10% tolerance for floating point).

4. **Single-core friendly.** Problems should take 1-30 seconds on a single core with the naive implementation. Too fast = no signal. Too slow = timeout.

5. **`UPPER_CASE` for fixed data.** Module-level generated data uses `UPPER_CASE` names (`POINTS`, `GRAPH`, `INIT_GRID`, etc.). Underscore-prefixed temporaries (`_n`, `_i`, `_rng`) for setup computation that shouldn't leak.

6. **f-string escaping.** Since starter code lives inside f-strings, literal braces must be doubled: `{{` and `}}`. This matters for format strings in `if __name__ == "__main__"` blocks and dict literals.

---

## How to Add a New Problem

### 1. Write the factory

```python
def make_my_new_problem(
    seed: int = 42,              # or size params if deterministic
    some_param: int = 1000,
) -> ResourceConstraintDatapoint:
    return ResourceConstraintDatapoint(
        problem_id=f"my_problem_s{seed}",    # unique ID
        problem_description="""# My Problem
...markdown description...

## Interface
```python
def run():
    return result
```
""",
        starter_code=f'''"""My new problem."""
import random

SOME_PARAM = {some_param}

random.seed({seed})
DATA = [random.randint(0, 100) for _ in range(SOME_PARAM)]


def run():
    """Solve the problem."""
    data = DATA
    # ... naive algorithm ...
    return result


if __name__ == "__main__":
    print(f"Result: {{run()}}")
''',
        random_seed=seed,
        input_params={"some_param": some_param},
        problem_type="my_category",
    )
```

### 2. Register in `PROBLEM_FACTORIES`

```python
PROBLEM_FACTORIES = {
    ...
    "my_new_problem": make_my_new_problem,
}
```

### 3. Add to `build_all_problems()`

For seeded:
```python
for seed in seeds:
    problems.extend([
        ...
        make_my_new_problem(seed=seed),
    ])
```

For deterministic:
```python
for n in _log_spaced_ints(100, 10000, _sv):
    problems.append(make_my_new_problem(some_param=n))
```

### Checklist

- [ ] `run()` contains ONLY the algorithm — all data generation is at module level
- [ ] Starter code is naive/first-pass, not optimized
- [ ] Output is deterministic (no unseeded randomness inside `run()`)
- [ ] Takes 1-30 seconds on a single core
- [ ] Self-contained (all imports, no external files)
- [ ] Factory registered in `PROBLEM_FACTORIES`
- [ ] Added to the appropriate section of `build_all_problems()`
- [ ] `problem_id` is unique (include seed or size in the ID)

---

## Where to Bulk Up

The dataset currently has ~770 problems from 72 factories. Here are the best directions for expansion, roughly ordered by value:

### High value — new algorithmic domains

These are domains with no or thin coverage today:

| Domain | Why valuable | Example problems |
|---|---|---|
| **Sparse matrix ops** | Very different optimization profile from dense | SpMV, sparse Cholesky, CSR/CSC conversion |
| **Interval / range queries** | Classic competitive programming, many optimization levels | Segment tree, Fenwick tree, range-min query |
| **Compression** | Real-world, many algorithmic tiers | Huffman coding, LZ77, dictionary coding |
| **Hashing / probabilistic** | Bloom filters, count-min sketch, HyperLogLog | Membership testing, cardinality estimation |
| **Constraint satisfaction** | Backtracking, pruning, arc consistency | Sudoku solver, graph coloring, SAT |
| **Geometric search** | Spatial indexing is a big optimization lever | k-d tree queries, R-tree, sweep line |

### Medium value — more depth in existing domains

| Domain | Current coverage | What's missing |
|---|---|---|
| **Graph** | 7 factories | Bipartite matching, max flow, strongly connected components, graph isomorphism |
| **String** | 4 factories | Aho-Corasick, regex matching, Burrows-Wheeler, string alignment (bioinformatics) |
| **DP** | 4 factories | Traveling salesman (small N), optimal BST, coin change variants, sequence alignment |
| **Numerical** | 5 factories | FFT, SVD, conjugate gradient, sparse solvers, polynomial evaluation |
| **Image / Grid** | 3 factories | Connected component labeling, histogram equalization, dilation/erosion |

### Lower value but easy wins

- **More size variants** for existing factories (increase `num_size_variants`)
- **More seeds** (increase `DEFAULT_NUM_SEEDS`) — diminishing returns past ~20
- **Parameter sweeps** for existing factories (e.g., vary `edge_probability` for graph problems)

---

## Known Limitations and Gotchas

### Monte Carlo problems

For `monte_carlo_pi`, `monte_carlo_integration`, `random_walk`, and `bootstrap_mean`, the random sampling IS the algorithm. Moving `random.seed()` out but keeping the `random.random()` calls in `run()` means:
- The model might switch from `random` to `numpy.random`, giving different random numbers
- Results are validated within 10% tolerance, which is enough for MC convergence
- These problems are arguably the weakest in the dataset because the "optimization" is mostly "use numpy" — consider whether they belong long-term

### Starter code isn't truly "baseline"

The starter code doubles as both the starting point for the model AND the performance baseline. If the starter code is accidentally well-optimized, there's no headroom. Always write genuinely naive implementations.

### F-string nesting

All `starter_code` is inside Python f-strings. This means:
- `{` and `}` must be doubled to `{{` and `}}`
- Backslashes need care (use `\\n` not `\n` inside the f-string, or use raw strings)
- Nested f-strings inside starter code are painful — avoid them

### Module-level execution cost

Moving data generation to module level means it runs at import time. For the evaluation harness this is fine (it imports both baseline and solution). But if someone imports `dataset.py` to just list factories, all `ALL_PROBLEMS` get built at import time. This is a ~2 second hit. If this becomes a problem, make `ALL_PROBLEMS` lazy.

### Single-file constraint

Each problem must be a single `.py` file. No reading from external data files, no multi-file projects. This limits the complexity of input data — everything must be generated inline.
