# Minimum-cost rooted branching planner

`branching.plan(n, root, edges)` picks the cheapest set of upstream edges that
makes every artifact traceable to the designated source.

* Nodes are `0 .. n-1`; `root` is the source.
* `edges` is a sequence of `(u, v, cost)` triples, at most 20000, costs are
  integers with `|cost| <= 10^12`. Negative costs, self-loops and parallel
  edges are allowed. Inputs are never modified.
* Returns `(minimum_total_cost, ascending original edge indices)`, or `None`
  if no valid selection exists. `n == 1` returns `(0, [])`.

## Why this is not per-node greedy

Counterexample from the specification: `0->1 (5)`, `0->2 (4)`, `1->2 (1)`,
`2->1 (1)`. Giving each node its cheapest incoming edge selects `0->1` and
`1->2` for a cost of 6. The optimum is `0->2` plus `2->1`, cost 5 — it pays
more for the entry so the cycle `1<->2` can be exploited as a unit. The
implementation uses the Chu-Liu/Edmonds minimum arborescence algorithm:
cheapest-incoming selection, cycle contraction with the incoming discount,
and expansion back to original edges. Worst case `O(n * (n + len(edges)))`.

## Files

| File | Purpose |
| --- | --- |
| `branching.py` | Delivered core: `plan`, plus `_solve` returning contraction statistics |
| `test_branching.py` | 15 curated unit tests, including an exhaustive cross-check |
| `demo.py` | Human-readable walkthrough with traceable original edges |
| `verify.py` | Full self-check driver: unit suite, exhaustive differential sweep, independent recursive oracle sweep, traceability, scale runs |
| `.pointwise-runtime.json` | Runtime manifest: entry points, real verification results, limits |

## Usage

```python
import branching

edges = [(0, 1, 5), (0, 2, 4), (1, 2, 1), (2, 1, 1)]
branching.plan(3, 0, edges)     # -> (5, [1, 3])
```

## Verification

```
python3 verify.py          # human-readable, exit 0 on success
python3 verify.py --json   # machine-readable summary
python3 demo.py            # worked examples with per-node traceability
python3 demo.py --quick    # same, without the large scale runs
python3 -m unittest -v test_branching
```

`verify.py` checks optimal cost against two independent oracles (exhaustive
enumeration on tiny graphs, and a separately written recursive Edmonds
implementation on medium graphs), plus structure, reachability, cost
consistency and end-to-end traceability of the selected original edges.
