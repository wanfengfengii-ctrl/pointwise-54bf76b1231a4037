"""Self-check driver for ``branching.plan``.

Runs, in order:

1. the curated unit suite in ``test_branching.py``;
2. a randomized differential sweep against an independent exhaustive
   oracle (all incoming-edge combinations) on tiny graphs;
3. a structural validation sweep on larger graphs (one incoming edge per
   non-source node, source has none, all nodes reachable, cost matches the
   selected original edges);
4. a medium-size sweep against a second, separately written cost-only oracle
   (recursive textbook Edmonds) on graphs too large to enumerate;
5. a traceability check: every node's selected-edge path is followed back to
   the source using only *original* input edges;
6. scale measurements at the documented maximum (n = 1000, m = 20000).

Usage::

    python3 verify.py             # human-readable report, exit code 0/1
    python3 verify.py --json      # machine-readable summary on stdout

The JSON form is what ``.pointwise-runtime.json`` refers to.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time
import unittest

import branching
import demo


# --------------------------------------------------------------------------- #
# Independent oracle
# --------------------------------------------------------------------------- #

def brute_force(n, root, edges):
    """Exhaustive optimum; only used for tiny graphs (all incoming choices)."""
    incoming = [[] for _ in range(n)]
    for i, (_, v, _) in enumerate(edges):
        incoming[v].append(i)
    targets = [v for v in range(n) if v != root]
    if any(not incoming[v] for v in targets):
        return None
    best = None
    for combo in itertools.product(*(incoming[v] for v in targets)):
        adjacency = [[] for _ in range(n)]
        total = 0
        for i in combo:
            u, v, c = edges[i]
            adjacency[u].append(v)
            total += c
        seen = {root}
        stack = [root]
        while stack:
            x = stack.pop()
            for y in adjacency[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        if len(seen) == n and (best is None or total < best):
            best = total
    return best


def reference_cost(n, root, edges):
    """Independent cost-only oracle: recursive textbook Edmonds contraction.

    Written separately from ``branching.py`` (dict-based, fresh node ids each
    round, cost only) so that it shares no bookkeeping with the delivered
    implementation.  Used on graphs too large for exhaustive enumeration.
    """
    if n == 1:
        return 0
    adjacency = [(u, v, w) for (u, v, w) in edges if u != v]
    nodes = set(range(n))
    offset = 0
    while True:
        in_edge = {}
        for u, v, w in adjacency:
            if v == root:
                continue
            old = in_edge.get(v)
            if old is None or w < old[0]:
                in_edge[v] = (w, u)
        for v in nodes:
            if v != root and v not in in_edge:
                return None
        state = {}
        cycle = None
        for start in nodes:
            if start == root or state.get(start):
                continue
            path = []
            position = {}
            x = start
            while x != root and x not in state:
                state[x] = 1
                position[x] = len(path)
                path.append(x)
                x = in_edge[x][1]
            if x != root and x in position:
                cycle = path[position[x]:]
                break
            for y in path:
                state[y] = 2
        if cycle is None:
            return offset + sum(in_edge[v][0] for v in nodes if v != root)
        members = set(cycle)
        offset += sum(in_edge[v][0] for v in members)
        new_id = max(nodes) + 1
        contracted = []
        for u, v, w in adjacency:
            uu = new_id if u in members else u
            vv = new_id if v in members else v
            if uu == vv:
                continue
            if v in members:
                w -= in_edge[v][0]
            contracted.append((uu, vv, w))
        nodes = (nodes - members) | {new_id}
        adjacency = contracted


# --------------------------------------------------------------------------- #
# Structural checking (shared with the test suite semantics)
# --------------------------------------------------------------------------- #

def structural_problems(n, root, edges, result):
    """Return a list of violations of the delivery contract (empty if valid)."""
    if result is None:
        return []  # existence is judged by the oracle in the sweep
    total, indices = result
    problems = []
    if list(indices) != sorted(indices):
        problems.append("indices are not ascending")
    if len(set(indices)) != len(indices):
        problems.append("duplicate indices")
    if n == 1:
        if not (total == 0 and indices == []):
            problems.append("n == 1 must return (0, [])")
        return problems
    if len(indices) != n - 1:
        problems.append(f"expected {n - 1} selected edges, got {len(indices)}")
    incoming = [0] * n
    adjacency = [[] for _ in range(n)]
    summed = 0
    for i in indices:
        if not 0 <= i < len(edges):
            problems.append(f"index {i} out of range")
            continue
        u, v, c = edges[i]
        adjacency[u].append(v)
        incoming[v] += 1
        summed += c
    if summed != total:
        problems.append(f"reported total {total} != sum of selected costs {summed}")
    if incoming[root] != 0:
        problems.append("source has an incoming edge")
    for v in range(n):
        if v != root and incoming[v] != 1:
            problems.append(f"node {v} has {incoming[v]} incoming edges")
    seen = {root}
    stack = [root]
    while stack:
        x = stack.pop()
        for y in adjacency[x]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    if len(seen) != n:
        problems.append(f"only {len(seen)}/{n} nodes reachable from the source")
    return problems


def traceability_ok(n, root, edges, result):
    """Follow only original edges from every node back to the source."""
    if result is None:
        return False
    _, indices = result
    edge_of = {}
    for i in indices:
        if not 0 <= i < len(edges):
            return False
        u, v, _ = edges[i]
        if v in edge_of:
            return False
        edge_of[v] = i
    for v in range(n):
        if v == root:
            continue
        seen = {v}
        x = v
        while x != root:
            e = edge_of.get(x)
            if e is None:
                return False
            x = edges[e][0]
            if x in seen:
                return False
            seen.add(x)
    return True


# --------------------------------------------------------------------------- #
# Suites
# --------------------------------------------------------------------------- #

def run_unit_tests():
    here = os.path.dirname(os.path.abspath(__file__))
    loader = unittest.TestLoader()
    suite = loader.discover(here, pattern="test_branching.py", top_level_dir=here)
    stream = open(os.devnull, "w")
    runner = unittest.TextTestRunner(stream=stream, verbosity=0)
    started = time.perf_counter()
    outcome = runner.run(suite)
    elapsed = time.perf_counter() - started
    stream.close()
    return {
        "tests_run": outcome.testsRun,
        "failures": len(outcome.failures),
        "errors": len(outcome.errors),
        "seconds": round(elapsed, 3),
        "passed": outcome.wasSuccessful(),
    }


def differential_sweep(trials=12000, seed=20240517):
    """Random tiny graphs, cost compared against exhaustive enumeration."""
    rng = random.Random(seed)
    compared = 0
    mismatches = 0
    invalid = 0
    started = time.perf_counter()
    families = [(-10**12, 10**12), (-50, 50), (0, 0), (-1, -1), (0, 1)]
    for _ in range(trials):
        n = rng.randint(1, 6)
        root = rng.randrange(n)
        m = rng.randint(0, 8)
        lo, hi = families[rng.randrange(len(families))]
        edges = [(rng.randrange(n), rng.randrange(n), rng.randint(lo, hi))
                 for _ in range(m)]
        result = branching.plan(n, root, edges)
        if structural_problems(n, root, edges, result):
            invalid += 1
        expected = brute_force(n, root, edges)
        actual = None if result is None else result[0]
        compared += 1
        if expected != actual:
            mismatches += 1
    return {
        "graphs": trials,
        "brute_force_comparisons": compared,
        "mismatches": mismatches,
        "structurally_invalid": invalid,
        "seconds": round(time.perf_counter() - started, 3),
        "passed": mismatches == 0 and invalid == 0,
    }


def differential_sweep_larger(trials=6000, seed=770):
    """Slightly larger random graphs; checks structure, feasibility and cost
    consistency, plus brute force whenever the search space stays tiny."""
    rng = random.Random(seed)
    checked = 0
    mismatches = 0
    invalid = 0
    started = time.perf_counter()
    for _ in range(trials):
        n = rng.randint(1, 9)
        root = rng.randrange(n)
        m = rng.randint(0, 16)
        edges = [(rng.randrange(n), rng.randrange(n),
                  rng.randint(-10**12, 10**12)) for _ in range(m)]
        result = branching.plan(n, root, edges)
        problems = structural_problems(n, root, edges, result)
        if problems:
            invalid += 1
        if result is not None and not traceability_ok(n, root, edges, result):
            invalid += 1
        if n <= 5 and m <= 7:
            expected = brute_force(n, root, edges)
            actual = None if result is None else result[0]
            checked += 1
            if expected != actual:
                mismatches += 1
    return {
        "graphs": trials,
        "brute_force_comparisons": checked,
        "mismatches": mismatches,
        "structurally_invalid": invalid,
        "seconds": round(time.perf_counter() - started, 3),
        "passed": mismatches == 0 and invalid == 0,
    }


def reference_sweep(trials=1000, seed=4242):
    """Medium graphs (beyond brute force) vs the independent recursive oracle."""
    rng = random.Random(seed)
    mismatches = 0
    invalid = 0
    started = time.perf_counter()
    for _ in range(trials):
        n = rng.randint(2, 60)
        root = rng.randrange(n)
        m = rng.randint(n // 2, 250)
        edges = [(rng.randrange(n), rng.randrange(n),
                  rng.randint(-10**12, 10**12)) for _ in range(m)]
        result = branching.plan(n, root, edges)
        if structural_problems(n, root, edges, result):
            invalid += 1
        expected = reference_cost(n, root, edges)
        actual = None if result is None else result[0]
        if expected != actual:
            mismatches += 1
    return {
        "graphs": trials,
        "reference_comparisons": trials,
        "mismatches": mismatches,
        "structurally_invalid": invalid,
        "seconds": round(time.perf_counter() - started, 3),
        "passed": mismatches == 0 and invalid == 0,
    }


def traceability_demo():
    """Every shipped demo scenario must be traceable through original edges."""
    cases = demo.scenario_cases()
    failed = []
    for name, n, root, edges, result in cases:
        if result is not None and not traceability_ok(n, root, edges, result):
            failed.append(name)
        if result is not None and structural_problems(n, root, edges, result):
            failed.append(name + " (structural)")
    return {"scenarios": [c[0] for c in cases], "failed": failed,
            "passed": not failed}


def scale_suite():
    """Documented maximum size: n = 1000, up to 20000 edges."""
    rng = random.Random(2026)
    n = 1000
    out = {}

    edges = [(rng.randrange(v), v, rng.randint(-10**12, 10**12))
             for v in range(1, n)]
    while len(edges) < 20000:
        edges.append((rng.randrange(n), rng.randrange(n),
                      rng.randint(-10**12, 10**12)))
    out["dense_random"] = timed(n, edges, "dense random costs")

    edges = []
    for i in range(1, n - 1, 2):
        edges.append((i, i + 1, -100))
        edges.append((i + 1, i, -100))
    for i in range(1, n - 1):
        edges.append((i, i + 1, 0))
    edges.append((0, 1, 0))
    while len(edges) < 20000:
        edges.append((rng.randrange(n), rng.randrange(n),
                      rng.randint(-10**12, 10**12)))
    out["many_negative_cycles"] = timed(n, edges, "many negative 2-cycles")

    edges = []
    for i in range(1, n - 1):
        edges.append((i + 1, i, 0))
    for i in range(1, n - 1):
        edges.append((i, i + 1, 0))
    edges.append((0, 1, 0))
    j = 0
    while len(edges) < 20000:
        edges.append(((j * 37) % n, (j * 91 + 7) % n, 5))
        j += 1
    out["contraction_cascade"] = timed(n, edges, "sequential contraction cascade")

    return out


def timed(n, edges, label):
    started = time.perf_counter()
    result, stats = branching._solve(n, 0, edges)
    elapsed = time.perf_counter() - started
    problems = structural_problems(n, 0, edges, result)
    return {
        "label": label,
        "n": n,
        "m": len(edges),
        "seconds": round(elapsed, 3),
        "rounds": stats["rounds"],
        "cycles": stats["cycles"],
        "expansion_depth": stats["levels"],
        "cost": None if result is None else result[0],
        "selected_edges": 0 if result is None else len(result[1]),
        "valid": not problems,
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true",
                        help="emit a machine-readable summary")
    parser.add_argument("--no-scale", action="store_true",
                        help="skip the n=1000 / m=20000 measurements")
    args = parser.parse_args()

    report = {}
    report["unit_tests"] = run_unit_tests()
    report["differential_small"] = differential_sweep()
    report["differential_larger"] = differential_sweep_larger()
    report["reference_medium"] = reference_sweep()
    report["traceability"] = traceability_demo()
    if not args.no_scale:
        report["scale"] = scale_suite()
    checks = [report["unit_tests"]["passed"],
              report["differential_small"]["passed"],
              report["differential_larger"]["passed"],
              report["reference_medium"]["passed"],
              report["traceability"]["passed"]]
    if "scale" in report:
        checks.extend(case["valid"] for case in report["scale"].values())
    report["all_passed"] = all(checks)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for name, value in report.items():
            print(f"{name}: {value}")
        print("\nALL CHECKS PASSED" if report["all_passed"] else "\nFAILURES")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
