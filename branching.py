"""Minimum-cost rooted branching (arborescence) over a directed multigraph.

Public API::

    plan(n, root, edges) -> (total_cost, [edge_indices]) | None

Nodes are integers ``0 .. n-1``.  Each edge ``(u, v, cost)`` means ``u`` may
act as an upstream dependency of ``v``.  A valid plan selects one incoming
edge for every node except ``root`` (the source has no incoming edge); the
selected edges must be loop-free and must reach every node from ``root``,
i.e. they form a branching rooted at ``root``.  The function returns the
minimum total cost with the ascending original indices of the selected edges,
or ``None`` when no such selection exists.

Algorithm: Chu-Liu/Edmonds minimum arborescence.

* Give every current group its cheapest incoming edge (the root group gets
  none).
* Walk the selected-edge functional graph: every group has at most one
  outgoing pointer (to the tail of its selected edge), so following pointers
  finds cycles.  A cycle of groups is contracted into one new group.
* Contracting cycle ``C`` subtracts the cheapest incoming weight of each
  member from every edge entering that member.  This discount is exactly the
  cost adjustment needed to later swap the member's selected edge.
* Repeat until the functional graph is acyclic, then expand every contracted
  group back to the original edges.  Expansion is iterative and can unwind up
  to ``n - 1`` nested contractions.

There are at most ``n - 1`` contraction rounds, and each round revisits the
edges incident to the groups it contracts.  In the worst case that is a full
pass over the current edge list, giving ``O(n * (n + len(edges)))`` time.
The algorithm never enumerates subsets of edges and never assumes that each
node can independently keep its globally cheapest incoming edge: cycles are
resolved globally by contraction, which is what makes negative costs,
self-loops, parallel edges and shared upstream nodes come out optimal.

Only the Python standard library is used.  The arguments are never mutated.
"""

from __future__ import annotations


def plan(n, root, edges):
    """Return ``(minimum_total_cost, ascending_edge_indices)`` or ``None``.

    ``edges`` is a sequence of ``(u, v, cost)`` triples with integer costs of
    absolute value at most 10^12.  Self-loops, parallel edges and negative
    costs are allowed.  ``n == 1`` yields ``(0, [])``; if no rooted branching
    exists the function returns ``None``.
    """
    result, _stats = _solve(n, root, edges)
    return result


def _solve(n, root, edges):
    """Implementation of :func:`plan`, plus counters used by the demo tooling.

    Returns ``(result, stats)``.  ``stats`` records contraction rounds, the
    total number of contracted cycles, the deepest nesting level reached
    during expansion, and the number of group/node visits made while detecting
    cycles.  The public ``plan`` wrapper drops it; correctness never depends
    on the counters.
    """
    stats = {"rounds": 0, "cycles": 0, "levels": 0, "lookups": 0}
    if n == 1:
        return (0, []), stats

    m = len(edges)
    # Working copies of the edge fields.  ``tail``/``head``/``weight`` are
    # rewritten by contractions; ``orig_costs``/``orig_heads`` preserve the
    # input values needed to report original costs and to map a selected edge
    # back to the original node it pointed at.
    tail = [0] * m
    head = [0] * m
    weight = [0] * m
    orig_costs = [0] * m
    orig_heads = [0] * m
    # Edges entering (resp. leaving) each current group.  A group is a leaf
    # original node at first, then a contracted cycle with a fresh id >= n.
    in_edges = [set() for _ in range(n)]
    out_edges = [set() for _ in range(n)]
    for i, (u, v, c) in enumerate(edges):
        tail[i] = u
        head[i] = v
        weight[i] = c
        orig_costs[i] = c
        orig_heads[i] = v
        out_edges[u].add(i)
        in_edges[v].add(i)

    # Group metadata.  A leaf group remembers its original node; a compound
    # group remembers its cycle members, the edge that entered each member at
    # contraction time, the original nodes it contains, and, for every
    # original node, the immediate child group it belongs to.
    groups = [{"compound": False, "node": node, "nodes": (node,)}
              for node in range(n)]
    root_gid = root
    active = list(range(n))
    gid_counter = n

    # Cheapest incoming edge of every current non-root group: gid -> (w, e).
    best = {}
    for g in range(n):
        if g == root_gid:
            continue
        chosen_edge = None
        for e in in_edges[g]:
            if (chosen_edge is None or weight[e] < weight[chosen_edge]
                    or (weight[e] == weight[chosen_edge] and e < chosen_edge)):
                chosen_edge = e
        if chosen_edge is not None:
            best[g] = (weight[chosen_edge], chosen_edge)

    for g in active:
        if g != root_gid and g not in best:
            return None, stats  # this node can never receive an incoming edge

    while True:
        # ---- Cycle detection on the selected-incoming-edge graph ----------
        stats["lookups"] += len(active)
        parent_tail = {g: tail[e] for g, (w, e) in best.items()}
        state = {}
        cycles = []
        for start in active:
            if start == root_gid or state.get(start) == 2:
                continue
            path = []
            position = {}
            x = start
            while x != root_gid and state.get(x, 0) == 0:
                state[x] = 1
                position[x] = len(path)
                path.append(x)
                x = parent_tail[x]
            if x != root_gid and state.get(x) == 1:
                cycles.append(path[position[x]:])
            for y in path:
                state[y] = 2

        if not cycles:
            break

        stats["rounds"] += 1
        stats["cycles"] += len(cycles)

        # ---- Contract each cycle into one fresh group ---------------------
        contracted = set()
        for cyc in cycles:
            contracted.update(cyc)
        new_active = [g for g in active if g not in contracted]

        for cyc in cycles:
            new_gid = gid_counter
            gid_counter += 1
            member_set = set(cyc)
            min_in = {g: best[g][0] for g in cyc}

            nodes = []
            child_map = {}
            entry_edges = []
            for child in cyc:
                for original_node in groups[child]["nodes"]:
                    child_map[original_node] = child
                nodes.extend(groups[child]["nodes"])
                entry_edges.append(best[child][1])
            groups.append({
                "compound": True,
                "cycle_members": cyc,
                "cycle_entry_edges": entry_edges,
                "nodes": tuple(nodes),
                "child_map": child_map,
            })

            # Only edges incident to the cycle can change.  Internal edges
            # become self-loops and are deleted; entering edges are
            # discounted by the selected incoming weight of their old head;
            # leaving edges simply point at the new group.
            affected = set()
            for g in cyc:
                affected |= in_edges[g]
                affected |= out_edges[g]
            new_in = set()
            new_out = set()
            for e in affected:
                u = tail[e]
                v = head[e]
                u_inside = u in member_set
                v_inside = v in member_set
                if u_inside and v_inside:
                    out_edges[u].discard(e)
                    in_edges[v].discard(e)
                    continue
                if u_inside:
                    out_edges[u].discard(e)
                    tail[e] = new_gid
                    new_out.add(e)
                if v_inside:
                    in_edges[v].discard(e)
                    weight[e] -= min_in[v]
                    head[e] = new_gid
                    new_in.add(e)
            in_edges.append(new_in)
            out_edges.append(new_out)

            if not new_in:
                return None, stats  # the contracted cycle has no entry at all
            chosen_edge = None
            for e in new_in:
                if (chosen_edge is None or weight[e] < weight[chosen_edge]
                        or (weight[e] == weight[chosen_edge]
                            and e < chosen_edge)):
                    chosen_edge = e
            best[new_gid] = (weight[chosen_edge], chosen_edge)
            new_active.append(new_gid)
            for g in cyc:
                del best[g]

        active = new_active

    # ---- Expand contracted groups to original edges -----------------------
    # The final functional graph is acyclic, so every non-root group has
    # exactly one selected incoming edge and pointers lead back to the root.
    # Expanding a compound group replaces the selected incoming edge of the
    # member that the group's incoming edge points at.
    stack = [(g, best[g][1], 0) for g in active if g != root_gid]
    chosen = []
    while stack:
        g, e, depth = stack.pop()
        if depth > stats["levels"]:
            stats["levels"] = depth
        info = groups[g]
        if not info["compound"]:
            chosen.append(e)
            continue
        # The selected edge of a compound group was originally pointing at
        # one of the original nodes inside that group; ``child_map`` resolves
        # that node to the cycle member at the level where it was contracted.
        child = info["child_map"][orig_heads[e]]
        members = info["cycle_members"]
        entries = info["cycle_entry_edges"]
        for i, member in enumerate(members):
            stack.append((member, e if member == child else entries[i],
                          depth + 1))

    chosen.sort()
    total = 0
    for e in chosen:
        total += orig_costs[e]
    return (total, chosen), stats
