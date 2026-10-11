import random
from collections import defaultdict

import numpy as np

from problem import weighted_value


def received_cost(pi, active, assignments):
    P = pi['interference_matrix']
    cost = np.zeros((pi['n_carriers'], pi['max_subchannels']))
    for j in active:
        col = P[:, j].copy()
        col[j] = 0
        for k in assignments.get(j, set()):
            cost[:, k] += col
    return cost


def headroom(pi, active, assignments):
    P = pi['interference_matrix']
    beam = pi['allocated_beams']
    S = pi['max_subchannels']
    out = {}
    for c in range(pi['n_carriers']):
        if c in active:
            continue
        cost = {k: 0 for k in range(S)}
        for j in active:
            for k in assignments.get(j, set()):
                cost[k] += P[c][j]
        excluded = {k for j in active if beam[j] == beam[c] for k in assignments.get(j, set())}
        available = sorted((k for k in range(S) if k not in excluded), key=lambda k: cost[k])
        if len(available) < pi['sub_channels'][c]:
            out[c] = -np.inf
        else:
            out[c] = pi['threshold'][c] - sum(cost[k] for k in available[:pi['sub_channels'][c]])
    return out


def _state(pi, active, assignments, targets):
    P = pi['interference_matrix']
    beam = pi['allocated_beams']
    occupants = {k: set() for k in range(pi['max_subchannels'])}
    for j in active:
        for k in assignments.get(j, set()):
            occupants[k].add(j)
    same_beam = defaultdict(list)
    for c in targets:
        same_beam[beam[c]].append(c)
    blocked = {}
    for c in targets:
        mask = np.zeros(pi['max_subchannels'], dtype=bool)
        for j in active:
            if beam[j] == beam[c]:
                for k in assignments.get(j, set()):
                    mask[k] = True
        blocked[c] = mask
    load = {}
    for j in active:
        slots = assignments.get(j, set())
        load[j] = sum(P[j][l] * len(slots & assignments.get(l, set())) for l in active if l != j)
    return dict(occupants=occupants, same_beam=same_beam, blocked=blocked, load=load,
                cost=received_cost(pi, active, assignments))


def _construct(pi, order, state, max_retries):
    P = pi['interference_matrix']
    thr = pi['threshold']
    sub = pi['sub_channels']
    beam = pi['allocated_beams']
    occupants = {k: set(v) for k, v in state['occupants'].items()}
    blocked = {c: m.copy() for c, m in state['blocked'].items()}
    load = dict(state['load'])
    cost = state['cost'].copy()
    placed = {}
    total_cost = 0.0
    for c in order:
        excluded = set()
        for attempt in range(max_retries + 1):
            avail = np.flatnonzero(~blocked[c])
            if excluded:
                avail = np.array([k for k in avail if k not in excluded])
            m = sub[c]
            if len(avail) < m:
                break
            costs = cost[c, avail]
            top = np.argpartition(costs, m - 1)[:m] if m < len(costs) else np.arange(len(costs))
            slots = avail[top].tolist()
            own = float(costs[top].sum())
            if own > thr[c]:
                break
            delta = {}
            for k in slots:
                for j in occupants[k]:
                    delta[j] = delta.get(j, 0) + P[j][c]
            violators = [j for j, d in delta.items() if load.get(j, 0) + d > thr[j]]
            if not violators:
                placed[c] = set(slots)
                load[c] = own
                total_cost += own
                for j, d in delta.items():
                    load[j] = load.get(j, 0) + d
                col = P[:, c].copy()
                col[c] = 0
                for k in slots:
                    occupants[k].add(c)
                    cost[:, k] += col
                for t in state['same_beam'][beam[c]]:
                    if t != c:
                        blocked[t][slots] = True
                break
            if attempt == max_retries:
                break
            poisoned = {k for k in slots if occupants[k] & set(violators)}
            if not poisoned:
                break
            excluded |= poisoned
    return placed, total_cost


def activate(pi, active, assignments, targets, n_restarts=8, rcl_alpha=0.5, rng_seed=0,
             max_retries=5):
    h = headroom(pi, active, assignments)
    w = pi['weight']
    order = sorted(targets, key=lambda c: (w[c], h.get(c, -np.inf)), reverse=True)
    state = _state(pi, active, assignments, targets)
    rng = np.random.default_rng(rng_seed)
    best, best_key = None, None
    for r in range(max(1, n_restarts)):
        seq = list(order)
        if r > 0:
            remaining, seq = list(order), []
            while remaining:
                size = max(1, int(np.ceil(len(remaining) * rcl_alpha)))
                seq.append(remaining.pop(rng.integers(0, size)))
        placed, total_cost = _construct(pi, seq, state, max_retries)
        key = (weighted_value(placed, pi), -total_cost)
        if best is None or key > best_key:
            best, best_key = placed, key
    out = dict(assignments)
    out.update(best)
    return list(best), out


def greedy_topup(pi, active, assignments, rng_seed=0):
    active_set = set(active)
    inactive = [c for c in range(pi['n_carriers']) if c not in active_set]
    if not inactive:
        return dict(assignments)
    return activate(pi, list(active), assignments, inactive, rng_seed=rng_seed)[1]


def interval_check(svals, sub, thr, P):
    n = len(svals)
    overlap = np.zeros((n, n))
    intervals = sorted(((i, s, s + sub[i]) for i, s in enumerate(svals)), key=lambda x: x[1])
    for a in range(n - 1):
        for b in range(a + 1, n):
            if intervals[b][1] > intervals[a][2]:
                break
            if intervals[a][1] == 0 or intervals[b][1] == 0:
                continue
            o = max(0, min(intervals[a][2], intervals[b][2]) - max(intervals[a][1], intervals[b][1]))
            if o > 0:
                i, j = intervals[a][0], intervals[b][0]
                overlap[i, j] = overlap[j, i] = o
    slack = np.array([thr[i] - sum(overlap[i, j] * P[i, j] for j in range(n)) for i in range(n)])
    return slack, overlap


def flip(pi, start):
    P = pi['interference_matrix']
    thr = pi['threshold']
    sub = pi['sub_channels']
    n = pi['n_carriers']
    beam = pi['allocated_beams']
    occupied = np.zeros((len(pi['allocated_users']), pi['max_subchannels']))
    svals = np.zeros(n)
    added = [start]
    current = start
    for count in range(n):
        row = occupied[beam[current]]
        m = sub[current]
        if count % 2 == 0:
            s = np.flatnonzero(row == 0)[0]
            occupied[beam[current], s:s + m] += 1
            svals[current] = s
        else:
            e = np.flatnonzero(row == 0)[-1]
            occupied[beam[current], e - m + 1:e + 1] += 1
            svals[current] = e - m + 1
        if count == n - 1:
            break
        best = np.ma.array(P[current], mask=np.isin(range(n), added)).max()
        ties = np.where(P[current] == best)[0]
        for i in ties[np.argsort(sub[ties])]:
            if i not in added:
                current = i
                break
        added.append(current)
    svals += 1
    slack, overlap = interval_check(svals, sub, thr, P)
    drop = np.where(slack <= 0)[0]
    contrib = overlap * P
    kept = []
    for i in drop[np.argsort(slack[drop])[::-1]]:
        kept.append(i)
        rest = drop[np.isin(drop, kept, invert=True)]
        totals = np.sum(contrib[kept, :] * np.isin(range(n), rest, invert=True), axis=1)
        if np.min(thr[kept] - totals) >= 0:
            drop = rest
        else:
            kept.remove(i)
    svals[drop.astype(int)] = 0
    return svals.astype(int)


def flip_random(pi):
    return flip(pi, random.randrange(pi['n_carriers']))


def svals_to_assignments(svals, pi):
    sub = pi['sub_channels']
    return {i: set(range(int(svals[i]) - 1, int(svals[i]) - 1 + int(sub[i])))
            for i in range(pi['n_carriers']) if int(svals[i]) > 0}


def flip_plus(pi):
    svals = flip(pi, int(np.argmax(pi['sub_channels'] / pi['threshold'])))
    assignments = svals_to_assignments(svals, pi)
    combined = greedy_topup(pi, sorted(assignments), assignments)
    active = sorted(c for c in combined if combined[c])
    return active, {c: set(combined[c]) for c in active}
