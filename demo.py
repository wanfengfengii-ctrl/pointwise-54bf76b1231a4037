"""Human-readable walkthrough for ``branching.plan``.

Run::

    python3 demo.py            # all sections
    python3 demo.py --quick    # skip the large scale measurements

Every scenario prints the *original* selected edges (edge index, u -> v, cost)
and, for every node, the chain of selected original edges that leads back to
the requested source, so the result can be audited by hand.  The
negative-cost, cycle-entry and nested-contraction scenarios are also checked
against an independent exhaustive search over small graphs.
"""

from __future__ import annotations

import argparse
import itertools
import random
import time

import branching


# --------------------------------------------------------------------------- #
# Auditing helpers
# --------------------------------------------------------------------------- #

def brute_force(n, root, edges):
    """Exhaustive optimum over small graphs, used as an independent oracle."""
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


def audit(n, root, edges, result):
    """Validate a result and build node -> (incoming edge, path to root)."""
    if result is None:
        return {"valid": brute_force(n, root, edges) is None, "paths": {}}
    total, indices = result
    incoming = {}
    summed = 0
    for i in indices:
        u, v, c = edges[i]
        incoming[v] = i
        summed += c
    problems = []
    if indices != sorted(set(indices)):
        problems.append("indices not strictly ascending")
    if n > 1 and len(indices) != n - 1:
        problems.append("wrong number of selected edges")
    if sum(c for _, _, c in (edges[i] for i in indices)) != total:
        problems.append("reported cost does not match the selected edges")
    if incoming.get(root) is not None:
        problems.append("source has an incoming edge")
    if len(incoming) != n - 1:
        problems.append("not every non-source node has exactly one incoming edge")
    paths = {}
    for v in range(n):
        if v == root:
            paths[v] = [v]
            continue
        chain = [v]
        seen = set(chain)
        while chain[-1] != root:
            e = incoming.get(chain[-1])
            if e is None:
                problems.append(f"node {v} cannot reach the source")
                chain = []
                break
            u = edges[e][0]
            if u in seen:
                problems.append(f"cycle reached while tracing node {v}")
                chain = []
                break
            chain.append(u)
            seen.add(u)
        if chain:
            paths[v] = chain[::-1]
    return {
        "valid": not problems,
        "problems": problems,
        "total": total,
        "paths": paths,
        "edges": [(i, edges[i]) for i in indices],
        "summed": summed,
    }


def show(title, n, root, edges, result, *, show_paths=True):
    report = audit(n, root, edges, result)
    print(f"\n=== {title} ===")
    print(f"n={n} root={root} m={len(edges)}")
    print("edges (original index: u -> v, cost):")
    for i, (u, v, c) in enumerate(edges):
        print(f"  [{i:2d}] {u} -> {v}  cost={c}")
    if result is None:
        print("result: None (no rooted branching exists)")
        print(f"audit valid: {report['valid']}")
        return report
    total, indices = result
    print(f"result: total cost = {total}, selected original indices = {indices}")
    print("selected original edges:")
    for i, (u, v, c) in report["edges"]:
        print(f"  [{i:2d}] {u} -> {v}  cost={c}")
    print(f"recomputed cost from those edges: {report['summed']}")
    if show_paths:
        print("traceability (each node's path of original edges to the source):")
        for v in range(n):
            if v == root:
                print(f"  node {v}: source")
                continue
            path = report["paths"].get(v)
            if not path:
                print(f"  node {v}: NOT TRACEABLE")
                continue
            hops = []
            for a, b in zip(path, path[1:]):
                e = next(i for i, (u, w, _) in enumerate(edges)
                         if u == a and w == b and i in indices)
                hops.append(f"[{e}] {a}->{b}")
            cost = sum(edges[e][2] for e in
                       (next(i for i, (x, y, _) in enumerate(edges)
                             if x == a and y == b and i in indices)
                        for a, b in zip(path, path[1:])))
            print(f"  node {v}: " + " ; ".join(hops) + f"   (path cost {cost})")
    print(f"audit valid: {report['valid']}"
          + ("" if report["valid"] else f"  problems={report['problems']}"))
    return report


def cross_check(name, n, root, edges, result):
    expected = brute_force(n, root, edges)
    actual = None if result is None else result[0]
    status = "OK" if expected == actual else "MISMATCH"
    print(f"brute-force cross-check ({name}): exhaustive optimum = {expected}, "
          f"plan = {actual} -> {status}")
    assert expected == actual, (name, n, root, edges, expected, actual)


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #

def scenario_statement():
    edges = [(0, 1, 5), (0, 2, 4), (1, 2, 1), (2, 1, 1)]
    result = branching.plan(3, 0, edges)
    report = show("specification example", 3, 0, edges, result)
    cross_check("specification example", 3, 0, edges, result)
    assert result == (5, [1, 3])
    # The tempting per-node greedy picks [0] (0->1, cost 5) and [2] (1->2,
    # cost 1), which forms no cycle but is not reachable... it is reachable,
    # yet costs 6; the optimum enters the 1<->2 cycle at node 2.
    return report


def scenario_negative_entry():
    # 1 <-> 2 is a 2-cycle.  Enter at 1: 50 + (-100) = -50.  Enter at 2:
    # 40 + (-60) = -20.  The cheaper entry is not the better plan.
    edges = [(0, 1, 50), (0, 2, 40), (1, 2, -100), (2, 1, -60)]
    result = branching.plan(3, 0, edges)
    report = show("negative costs: the more expensive entry wins", 3, 0,
                  edges, result)
    cross_check("negative entry", 3, 0, edges, result)
    assert result == (-50, [0, 2])
    return report


def scenario_entries_inside_and_outside():
    # Cycle A = {1, 2}.  Node 3 sits between two cycles: it is entered from
    # inside cycle A and from outside (via the 0 -> 4 -> 3 chain).
    edges = [
        (1, 2, -5),    # [0] cycle A member
        (2, 1, -5),    # [1] cycle A member
        (0, 1, 30),    # [2] external entry into cycle A
        (1, 3, 4),     # [3] entry into 3 from inside cycle A
        (4, 3, 9),     # [4] entry into 3 from outside the cycle
        (0, 4, 2),     # [5] supply for 4 from the source
        (3, 4, -1),    # [6] cheap back edge making {3,4} attractive as cycle
    ]
    result = branching.plan(5, 0, edges)
    report = show("entries from inside and outside a cycle", 5, 0, edges,
                  result)
    cross_check("inside/outside entries", 5, 0, edges, result)
    return report


def scenario_multilevel():
    # Three nested contractions.  Round 1 contracts {1,2}; its incoming edge
    # from 3 pulls {3, ...} into the next cycle; that cycle's incoming edge
    # from 4 pulls in the third level; finally the source enters at 0 -> 1.
    edges = [
        (1, 2, 0),     # [0]
        (2, 1, 0),     # [1]  cycle {1,2}
        (3, 1, 0),     # [2]  edge from 3 into cycle {1,2}
        (2, 3, 0),     # [3]  makes {1,2,3} a cycle after contraction
        (4, 1, 0),     # [4]  edge from 4 into the super-cycle
        (2, 4, 0),     # [5]  makes {1,2,3,4} a cycle after another contraction
        (0, 1, 100),   # [6]  the single external entry, expensive
        (0, 4, 1000),  # [7]  alternative external entry, more expensive
    ]
    # Node 4 is reachable through 2 -> 4 inside the final cycle, so the source
    # only needs the 0 -> 1 edge.
    result, stats = branching._solve(5, 0, edges)
    report = show("multi-level contraction (3 nested rounds)", 5, 0, edges,
                  result)
    print(f"contraction stats: rounds={stats['rounds']} "
          f"cycles={stats['cycles']} deepest expansion level={stats['levels']}")
    cross_check("multi-level contraction", 5, 0, edges, result)
    assert stats["levels"] >= 3, stats
    return report


def scenario_selfloops_parallel():
    edges = [
        (0, 1, 5), (1, 1, -10**9), (0, 2, 4), (2, 2, -10**9),
        (1, 2, 1), (2, 1, 1), (0, 1, 5),
    ]
    result = branching.plan(3, 0, edges)
    report = show("self-loops and parallel edges", 3, 0, edges, result)
    cross_check("self-loops and parallel edges", 3, 0, edges, result)
    return report


def scenario_unreachable():
    edges = [(0, 1, 1)]
    result = branching.plan(3, 0, edges)
    show("unreachable node -> None", 3, 0, edges, result)
    assert result is None
    return None


# --------------------------------------------------------------------------- #
# Scale
# --------------------------------------------------------------------------- #

def scale_case(name, n, edges, root=0):
    start = time.perf_counter()
    result, stats = branching._solve(n, root, edges)
    elapsed = time.perf_counter() - start
    report = audit(n, root, edges, result)
    print(f"\n=== scale: {name} ===")
    print(f"n={n} m={len(edges)} rounds={stats['rounds']} "
          f"cycles={stats['cycles']} expansion depth={stats['levels']}")
    print(f"time = {elapsed:.3f}s  cost = "
          f"{None if result is None else result[0]}  edges selected = "
          f"{0 if result is None else len(result[1])}")
    print(f"audit valid = {report['valid']}")
    return elapsed, result, stats


def scale_benchmarks():
    rng = random.Random(2026)
    n = 1000

    edges = [(rng.randrange(v), v, rng.randint(-10**12, 10**12))
             for v in range(1, n)]
    while len(edges) < 20000:
        edges.append((rng.randrange(n), rng.randrange(n),
                      rng.randint(-10**12, 10**12)))
    dense = scale_case("dense random, max costs", n, edges)

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
    cycles = scale_case("many negative 2-cycles", n, edges)

    # Worst case for the contraction loop: a long cascade where each round
    # merges exactly one pair of nodes into an ever-growing group.
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
    cascade = scale_case("sequential contraction cascade", n, edges)

    return {"dense": dense[0], "cycles": cycles[0], "cascade": cascade[0],
            "cascade_stats": cascade[2]}


SCENARIOS = (
    ("specification example", 3, 0, [(0, 1, 5), (0, 2, 4), (1, 2, 1), (2, 1, 1)]),
    ("negative entry", 3, 0, [(0, 1, 50), (0, 2, 40), (1, 2, -100), (2, 1, -60)]),
    ("inside/outside entries", 5, 0, [
        (1, 2, -5), (2, 1, -5), (0, 1, 30), (1, 3, 4), (4, 3, 9),
        (0, 4, 2), (3, 4, -1)]),
    ("multi-level contraction", 5, 0, [
        (1, 2, 0), (2, 1, 0), (3, 1, 0), (2, 3, 0), (4, 1, 0), (2, 4, 0),
        (0, 1, 100), (0, 4, 1000)]),
    ("self-loops and parallel edges", 3, 0, [
        (0, 1, 5), (1, 1, -10**9), (0, 2, 4), (2, 2, -10**9),
        (1, 2, 1), (2, 1, 1), (0, 1, 5)]),
    ("unreachable -> None", 3, 0, [(0, 1, 1)]),
)


def scenario_cases():
    """(name, n, root, edges, result) for every non-scale demo scenario."""
    cases = []
    for name, n, root, edges in SCENARIOS:
        cases.append((name, n, root, edges, branching.plan(n, root, edges)))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="skip the large scale measurements")
    args = parser.parse_args()

    scenario_statement()
    scenario_negative_entry()
    scenario_entries_inside_and_outside()
    scenario_multilevel()
    scenario_selfloops_parallel()
    scenario_unreachable()
    if not args.quick:
        scale_benchmarks()
    print("\nall demo assertions and brute-force cross-checks passed")


if __name__ == "__main__":
    main()
