"""Tests for branching.plan.

Run with:  python3 -m unittest -v test_branching   (or python3 test_branching.py)
Only the Python standard library is used.
"""

from __future__ import annotations

import itertools
import random
import unittest

import branching


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def brute_force(n, root, edges):
    """Exhaustive optimum over all incoming-edge choices; small graphs only."""
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


def check_selection(testcase, n, root, edges, result):
    """Assert the returned selection is structurally a valid rooted branching."""
    if result is None:
        # No branching exists: the exhaustive check must agree.
        testcase.assertIsNone(brute_force(n, root, edges))
        return
    if n == 1:
        testcase.assertEqual(result, (0, []))
        return
    total, indices = result
    testcase.assertEqual(indices, sorted(indices), "indices must be ascending")
    testcase.assertEqual(len(indices), len(set(indices)), "indices must be unique")
    testcase.assertEqual(len(indices), n - 1, "one edge per non-root node")
    incoming = [0] * n
    adjacency = [[] for _ in range(n)]
    summed = 0
    for i in indices:
        u, v, c = edges[i]
        adjacency[u].append(v)
        incoming[v] += 1
        summed += c
    testcase.assertEqual(summed, total, "reported cost must match selected edges")
    testcase.assertEqual(incoming[root], 0, "the source has no incoming edge")
    for v in range(n):
        if v != root:
            testcase.assertEqual(incoming[v], 1, "every other node has one incoming edge")
    seen = {root}
    stack = [root]
    while stack:
        x = stack.pop()
        for y in adjacency[x]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    testcase.assertEqual(len(seen), n, "all nodes reachable from the source")


# --------------------------------------------------------------------------- #
# Core examples from the specification
# --------------------------------------------------------------------------- #

class SpecExamples(unittest.TestCase):
    def test_example_from_statement(self):
        # 0->1 cost 5, 0->2 cost 4, 1->2 cost 1, 2->1 cost 1
        # Optimum: 0->2 and 2->1, total 5, original indices [1, 3].
        edges = [(0, 1, 5), (0, 2, 4), (1, 2, 1), (2, 1, 1)]
        self.assertEqual(branching.plan(3, 0, edges), (5, [1, 3]))

    def test_single_node(self):
        self.assertEqual(branching.plan(1, 0, []), (0, []))
        self.assertEqual(branching.plan(1, 0, [(0, 0, -5)]), (0, []))

    def test_unreachable_returns_none(self):
        self.assertIsNone(branching.plan(3, 0, [(0, 1, 1)]))
        self.assertIsNone(branching.plan(2, 0, []))

    def test_input_not_modified(self):
        edges = [(0, 1, 5), (0, 2, 4), (1, 2, 1), (2, 1, 1)]
        copy = list(edges)
        branching.plan(3, 0, edges)
        self.assertEqual(edges, copy)

    def test_accepts_tuple_input(self):
        edges = ((0, 1, 5), (0, 2, 4), (1, 2, 1), (2, 1, 1))
        self.assertEqual(branching.plan(3, 0, edges), (5, [1, 3]))


# --------------------------------------------------------------------------- #
# Independent exhaustive cross-check
# --------------------------------------------------------------------------- #

class BruteForceCrossCheck(unittest.TestCase):
    def test_random_small_graphs(self):
        rng = random.Random(20240517)
        comparisons = 0
        for _ in range(6000):
            n = rng.randint(1, 6)
            root = rng.randrange(n)
            m = rng.randint(0, 9)
            edges = [
                (rng.randrange(n), rng.randrange(n),
                 rng.choice([-100, -10, -1, 0, 1, 4, 17, 100]))
                for _ in range(m)
            ]
            result = branching.plan(n, root, edges)
            check_selection(self, n, root, edges, result)
            if not (n <= 5 and m <= 7):
                continue
            expected = brute_force(n, root, edges)
            actual = None if result is None else result[0]
            self.assertEqual(actual, expected, (n, root, edges))
            comparisons += 1
        self.assertGreater(comparisons, 3000)


# --------------------------------------------------------------------------- #
# Targeted scenarios: negative costs, cycle entries, nested contractions
# --------------------------------------------------------------------------- #

class TargetedScenarios(unittest.TestCase):
    def test_negative_costs_make_expensive_entry_worthwhile(self):
        # 1 <-> 2 is a two-cycle.  Entering at 1 costs 50 and the internal
        # 1 -> 2 edge is -100 (total -50); entering at 2 costs 40 and the
        # internal 2 -> 1 edge is -60 (total -20).  The *more expensive*
        # external entry wins because it unlocks the cheaper internal edge,
        # which is exactly what plain "cheapest incoming edge per node"
        # greedy selection gets wrong.
        edges = [
            (0, 1, 50),
            (0, 2, 40),
            (1, 2, -100),
            (2, 1, -60),
        ]
        result = branching.plan(3, 0, edges)
        self.assertEqual(result, (-50, [0, 2]))
        check_selection(self, 3, 0, edges, result)

    def test_negative_cycles_and_parallel_edges(self):
        # A negative parallel edge must win over a positive one.
        edges = [(0, 1, 7), (0, 1, -3), (0, 2, 2), (1, 2, 1), (2, 1, 1)]
        result = branching.plan(3, 0, edges)
        self.assertEqual(result[0], brute_force(3, 0, edges))
        self.assertIn(1, result[1])
        check_selection(self, 3, 0, edges, result)

    def test_self_loops_are_never_selected(self):
        # A hugely negative self-loop must not be part of a branching.
        edges = [(0, 1, 5), (1, 1, -10**9), (0, 2, 4), (2, 2, -10**9),
                 (1, 2, 1), (2, 1, 1)]
        result = branching.plan(3, 0, edges)
        # Enter at 2 with 0 -> 2 (4) and use 2 -> 1 (1); the -1e9 self-loops
        # are cycles on their own and are contracted away, never selected.
        self.assertEqual(result, (5, [2, 5]))
        check_selection(self, 3, 0, edges, result)

    def test_three_level_nested_contraction(self):
        # 1 <-> 2 is a cycle; it feeds 3 <-> 4; both feed the root entry 5.
        # Contracting 3/4 first (or 1/2 first) forces a further contraction
        # of the super-cycle, so expansion must unwind three levels.
        edges = [
            (1, 2, 0), (2, 1, 0),          # inner cycle
            (0, 1, 100),                   # external entry
            (3, 1, 0), (2, 3, 0),          # second cycle 2 <-> 3? see below
            (4, 1, 0), (2, 4, 0),
        ]
        # Provide full reachability for nodes 3 and 4 too.
        edges += [(3, 3, 0)]
        n = 5
        result = branching.plan(n, 0, edges)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], brute_force(n, 0, edges))
        check_selection(self, n, 0, edges, result)

    def test_chain_into_cycle_entry_from_inside(self):
        # Entry into the cycle comes from a node that itself needs an incoming
        # edge from inside the cycle; this must not create a loop.
        n = 4
        edges = [
            (0, 1, 10),
            (1, 2, 1), (2, 1, 1),
            (2, 3, -5), (3, 2, -5),
            (3, 0, 100),
        ]
        result = branching.plan(n, 0, edges)
        self.assertEqual(result[0], brute_force(n, 0, edges))
        check_selection(self, n, 0, edges, result)

    def test_large_negative_cycle_with_external_entry(self):
        n = 6
        edges = [(i, i + 1, -10) for i in range(1, n - 1)]
        edges.append((n - 1, 1, -10))       # big negative cycle 1..5
        edges.append((0, 1, 1000))          # only way in
        edges.append((0, 2, 1))             # tempting but breaks the cycle
        result = branching.plan(n, 0, edges)
        self.assertEqual(result[0], brute_force(n, 0, edges))
        check_selection(self, n, 0, edges, result)

    def test_shared_upstream_node(self):
        # Two nodes both want the same cheapest upstream, but a branching
        # still gives each node one incoming edge (edges may share tails).
        edges = [(0, 1, 1), (0, 2, 1), (1, 2, 9), (2, 1, 9)]
        result = branching.plan(3, 0, edges)
        self.assertEqual(result, (2, [0, 1]))


# --------------------------------------------------------------------------- #
# Scale
# --------------------------------------------------------------------------- #

class Scale(unittest.TestCase):
    def _timed_valid(self, n, root, edges):
        result = branching.plan(n, root, edges)
        check_selection(self, n, root, edges, result)
        return result

    def test_max_size_dense(self):
        rng = random.Random(99)
        n = 1000
        edges = [(rng.randrange(v), v, rng.randint(-10**12, 10**12))
                 for v in range(1, n)]
        while len(edges) < 20000:
            edges.append((rng.randrange(n), rng.randrange(n),
                          rng.randint(-10**12, 10**12)))
        self._timed_valid(n, 0, edges)

    def test_max_size_many_cycles(self):
        rng = random.Random(100)
        n = 1000
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
        self._timed_valid(n, 0, edges)


if __name__ == "__main__":
    unittest.main(verbosity=2)
