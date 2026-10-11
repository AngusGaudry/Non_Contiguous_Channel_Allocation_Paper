import random
import time

import numpy as np
from ortools.sat.python import cp_model

from construction import activate, flip_plus, headroom, received_cost
from problem import sanity_check, weighted_value

KICK_STAGNATION = 6
KICK_GIVEUP = 15
KICK_TABU_ROUNDS = 3
KICK_MAX_FRACTION = 0.3
POOL_MAX_DEGENERACY = 6


def sizes(n_active, n_inactive):
    n_s2 = min(n_active, max(1, int(round(0.15 * n_active)))) if n_active > 0 else 0
    n_s3 = min(n_inactive, max(1, int(round(0.5 * n_inactive)))) if n_inactive > 0 else 0
    return min(n_s2, 20), n_s3


def _sample(items, weights, k):
    items, weights = list(items), list(weights)
    chosen = []
    for _ in range(k):
        if not items:
            break
        idx = items.index(random.choices(items, weights=weights, k=1)[0])
        chosen.append(items.pop(idx))
        weights.pop(idx)
    return chosen


def select_neighbourhood(inactive, active, count_s3, count_s2, h, n_s2, n_s3, P):
    eligible = [c for c in inactive if h[c] > -np.inf]
    if not eligible:
        return None, None
    lo, hi = min(h[c] for c in eligible), max(h[c] for c in eligible)
    w3 = [((1 + (h[c] - lo) / (hi - lo)) if hi > lo else 1.0) / (count_s3.get(c, 0) + 1) for c in eligible]
    N_S3 = _sample(eligible, w3, min(n_s3, len(eligible)))
    for c in N_S3:
        count_s3[c] = count_s3.get(c, 0) + 1
    rel = {j: sum(P[j][c] for c in N_S3) for j in active}
    lo, hi = min(rel.values()), max(rel.values())
    w2 = [((1 + (rel[j] - lo) / (hi - lo)) if hi > lo else 1.0) / (count_s2.get(j, 0) + 1) for j in active]
    N_S2 = _sample(active, w2, min(n_s2, len(active)))
    for j in N_S2:
        count_s2[j] = count_s2.get(j, 0) + 1
    return N_S2, N_S3


def shift_slots(pi, active, assignments, N_S2, relief_topk=50):
    P = pi['interference_matrix']
    thr = pi['threshold']
    beam = pi['allocated_beams']
    S = pi['max_subchannels']
    movable = set(N_S2)
    fixed = [c for c in active if c not in movable]
    domain = {}
    for a in N_S2:
        blocked = set()
        for j in fixed:
            if beam[j] == beam[a]:
                blocked.update(assignments.get(j, set()))
        domain[a] = set(range(S)) - blocked
    occupants = {k: set() for k in range(S)}
    for j in active:
        for k in assignments.get(j, set()):
            occupants[k].add(j)
    cost = received_cost(pi, active, assignments)
    load = {}
    for j in active:
        slots = list(assignments.get(j, set()))
        load[j] = float(cost[j, slots].sum()) if slots else 0.0

    def min_slack():
        ranked = sorted(((thr[j] - load[j], j) for j in active), key=lambda x: x[0])
        m = ranked[0][0]
        return m, {j for s, j in ranked if s == m}, ranked

    current, holders, ranked = min_slack()
    improved = True
    while improved:
        improved = False
        blocking = []
        for c in holders:
            for k in assignments.get(c, set()):
                for j in occupants[k]:
                    if j != c and j in movable and P[c][j] > 0:
                        blocking.append((k, j, P[c][j]))
        blocking.sort(key=lambda x: x[2], reverse=True)
        same_beam = {}
        for k_out, holder, _ in blocking:
            if k_out not in assignments.get(holder, set()):
                continue
            if holder not in same_beam:
                same_beam[holder] = {k for j in N_S2 if j != holder and beam[j] == beam[holder]
                                     for k in assignments.get(j, set())}
            cands = [k for k in domain[holder]
                     if k not in same_beam[holder] and k not in assignments.get(holder, set())]
            if not cands:
                continue
            arr = np.array(cands)
            occ = np.fromiter((len(occupants.get(int(k), ())) for k in arr), dtype=np.int64, count=len(arr))
            for k_in in arr[np.lexsort((arr, occ))][:relief_topk].tolist():
                touched = {}
                delta = 0.0
                for j in occupants.get(k_out, set()) - {holder}:
                    touched[j] = load[j] - P[j][holder]
                    delta -= P[holder][j]
                for j in occupants.get(k_in, set()) - {holder}:
                    touched[j] = touched.get(j, load[j]) + P[j][holder]
                    delta += P[holder][j]
                own = load[holder] + delta
                if own > thr[holder] or any(v > thr[j] for j, v in touched.items()):
                    continue
                changed = dict(touched)
                changed[holder] = own
                new_min = min(thr[j] - v for j, v in changed.items())
                for s, j in ranked:
                    if j not in changed:
                        new_min = min(new_min, s)
                        break
                if new_min > current:
                    assignments[holder] = assignments.get(holder, set()) - {k_out} | {k_in}
                    load.update(changed)
                    occupants[k_out].discard(holder)
                    occupants[k_in].add(holder)
                    current, holders, ranked = min_slack()
                    improved = True
                    break
            if improved:
                break
    return assignments


def _cheapest_slots(pi, active, assignments, cost, c):
    beam = pi['allocated_beams']
    blocked = np.zeros(pi['max_subchannels'], dtype=bool)
    for j in active:
        if beam[j] == beam[c]:
            for k in assignments.get(j, set()):
                blocked[k] = True
    avail = np.flatnonzero(~blocked)
    m = pi['sub_channels'][c]
    if len(avail) < m:
        return None, None
    costs = cost[c, avail]
    top = np.argpartition(costs, m - 1)[:m] if m < len(costs) else np.arange(len(costs))
    return avail[top].tolist(), costs[top].sum()


def blocking_scores(pi, active, assignments, candidates):
    P = pi['interference_matrix']
    thr = pi['threshold']
    occupants = {k: set() for k in range(pi['max_subchannels'])}
    for j in active:
        for k in assignments.get(j, set()):
            occupants[k].add(j)
    cost = received_cost(pi, active, assignments)
    load = {j: sum(P[j][l] * len(assignments.get(j, set()) & assignments.get(l, set()))
                   for l in active if l != j) for j in active}
    score, blocked_by = {}, {}
    for c in candidates:
        slots, own = _cheapest_slots(pi, active, assignments, cost, c)
        if slots is None or own > thr[c]:
            continue
        delta = {}
        for k in slots:
            for j in occupants[k]:
                delta[j] = delta.get(j, 0) + P[j][c]
        violators = [j for j, d in delta.items() if load.get(j, 0) + d > thr[j]]
        if violators:
            blocked_by[c] = set(violators)
            for j in violators:
                score[j] = score.get(j, 0) + 1
    return score, blocked_by


def marginal_scores(pi, active, assignments, inactive):
    P = pi['interference_matrix']
    occupants = {k: set() for k in range(pi['max_subchannels'])}
    for j in active:
        for k in assignments.get(j, set()):
            occupants[k].add(j)
    cost = received_cost(pi, active, assignments)
    score = {}
    for c in inactive:
        slots, own = _cheapest_slots(pi, active, assignments, cost, c)
        if slots is None or own <= pi['threshold'][c]:
            continue
        for k in slots:
            for j in occupants[k]:
                score[j] = score.get(j, 0.0) + P[c][j]
    return score


def degeneracy(nodes, adjacency):
    degree = {v: len(adjacency.get(v, set())) for v in nodes}
    remaining = set(nodes)
    worst = 0
    while remaining:
        v = min(remaining, key=lambda u: degree[u])
        worst = max(worst, degree[v])
        remaining.discard(v)
        for u in adjacency.get(v, set()):
            if u in remaining:
                degree[u] -= 1
    return worst


def select_repair_pool(pi, active, assignments, max_degeneracy=POOL_MAX_DEGENERACY,
                       max_blockers=5, sample_size=60):
    P = pi['interference_matrix']
    active_set = set(active)
    inactive = [c for c in range(pi['n_carriers']) if c not in active_set]
    h = headroom(pi, active, assignments)
    sample = sorted(inactive, key=lambda c: h[c], reverse=True)[:sample_size]
    score, blocked_by = blocking_scores(pi, active, assignments, sample)
    pool, adjacency = [], {}

    def try_add(c, limit):
        nonlocal pool, adjacency
        trial = {v: set(adjacency.get(v, set())) for v in pool}
        trial[c] = set()
        for v in pool:
            if P[c][v] > 0 or P[v][c] > 0:
                trial[c].add(v)
                trial[v].add(c)
        if degeneracy(pool + [c], trial) <= limit:
            pool, adjacency = pool + [c], trial

    for j in sorted(score, key=lambda j: -score[j])[:max_blockers]:
        try_add(j, max(1, max_degeneracy // 2))
    blockers = set(pool)
    tier0 = sorted((c for c in sample if h[c] >= 0), key=lambda c: -h[c])
    tier1 = sorted((c for c in sample if h[c] < 0 and c in blocked_by and blocked_by[c] & blockers),
                   key=lambda c: -h[c])
    for c in tier0 + tier1:
        if c not in pool:
            try_add(c, max_degeneracy)
    return set(pool)


def solve_pool(pi, pool, assignments, time_limit):
    n = pi['n_carriers']
    S = pi['max_subchannels']
    sub = pi['sub_channels']
    thr = pi['threshold']
    P = pi['interference_matrix']
    w = pi['weight']
    pool = set(pool)
    model = cp_model.CpModel()
    x, y = {}, {}
    for c in range(n):
        if c in pool:
            y[c] = model.NewBoolVar(f'y_{c}')
            for k in range(S):
                x[c, k] = model.NewBoolVar(f'x_{c}_{k}')
        else:
            assigned = assignments.get(c, set())
            y[c] = 1 if len(assigned) == sub[c] else 0
            for k in range(S):
                x[c, k] = 1 if k in assigned else 0
    for c in pool:
        model.Add(sum(x[c, k] for k in range(S)) == sub[c] * y[c])
    for members in pi['allocated_users'].values():
        for k in range(S):
            model.Add(sum(x[c, k] for c in members) <= 1)
    ordered = sorted(pool)
    z = {}
    for a, i in enumerate(ordered):
        for j in ordered[a + 1:]:
            if P[i][j] == 0 and P[j][i] == 0:
                continue
            for k in range(S):
                z[i, j, k] = model.NewBoolVar(f'z_{i}_{j}_{k}')
                model.Add(z[i, j, k] <= x[i, k])
                model.Add(z[i, j, k] <= x[j, k])
                model.Add(z[i, j, k] >= x[i, k] + x[j, k] - 1)
    for i in range(n):
        if i not in pool:
            continue
        terms = []
        for j in range(n):
            if j == i or int(P[i][j]) == 0:
                continue
            if j in pool:
                a, b = min(i, j), max(i, j)
                if (a, b, 0) in z:
                    terms.append(int(P[i][j]) * sum(z[a, b, k] for k in range(S)))
            else:
                terms.append(sum(int(P[i][j]) * x[i, k] for k in range(S) if x[j, k] == 1))
        if terms:
            model.Add(sum(terms) <= thr[i])
    for j in range(n):
        slots = assignments.get(j)
        if j in pool or not slots:
            continue
        incoming = [int(P[j][i]) * sum(x[i, k] for k in slots) for i in pool if int(P[j][i]) != 0]
        if not incoming:
            continue
        base = sum(int(P[j][l]) * len(slots & assignments.get(l, set()))
                   for l in range(n) if l != j and l not in pool)
        model.Add(base + sum(incoming) <= thr[j])
    model.Maximize(sum(int(w[c]) * y[c] for c in pool))
    for c in pool:
        assigned = assignments.get(c, set())
        model.AddHint(y[c], 1 if len(assigned) == sub[c] else 0)
        for k in range(S):
            model.AddHint(x[c, k], 1 if k in assigned else 0)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    if solver.Solve(model) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return {c: {k for k in range(S) if solver.Value(x[c, k]) == 1}
            for c in pool if solver.Value(y[c]) == 1}


def select_kick(pi, active, assignments, exclude, criterion):
    active_set = set(active)
    inactive = [c for c in range(pi['n_carriers']) if c not in active_set]
    if not inactive or not active:
        return set()
    if criterion == 'marginal':
        score = marginal_scores(pi, active, assignments, inactive)
    else:
        score = blocking_scores(pi, active, assignments, inactive)[0]
    ranked = sorted((j for j in score if j not in exclude), key=lambda j: -score[j])
    return set(ranked[:max(1, int(np.floor(KICK_MAX_FRACTION * len(active))))])


class StallDetector:
    def __init__(self, n_carriers, upper_bound=None, M=4.0, gap_floor=0.5, warmup=2.0,
                 noise=0.01):
        self.n = n_carriers
        self.ub = upper_bound
        self.M = M
        self.gap_floor = gap_floor
        self.warmup = warmup
        self.noise = noise
        self.abs_idle = max(25.0, 0.1 * n_carriers)
        self.gaps = []
        self.last_t = 0.0
        self.ref = None
        self.stopped = False

    def _done(self, value, count):
        return ((count is not None and count >= self.n)
                or (self.ub is not None and value is not None and value >= self.ub))

    def update(self, t, value, count):
        if self.stopped:
            return True
        if self.ref is None:
            self.ref, self.last_t = value, t
            self.stopped = self._done(value, count)
            return self.stopped
        if value > self.ref * (1.0 + self.noise):
            self.gaps.append(t - self.last_t)
            self.last_t, self.ref = t, value
        idle = t - self.last_t
        self.stopped = (self._done(value, count) or idle > self.abs_idle
                        or (self.gaps and t >= self.warmup
                            and idle > self.M * max(float(np.median(self.gaps)), self.gap_floor)))
        return self.stopped


def csa_lns(pi, budget, seed, warm_start=None, stall=None, cp_time_limit=10.0):
    random.seed(seed)
    deadline = time.perf_counter() + budget
    if warm_start is not None:
        active, assignments = warm_start
        active = sorted(int(a) for a in active)
        assignments = {c: set(assignments[c]) for c in sorted(assignments) if assignments[c]}
    else:
        active, assignments = flip_plus(pi)
    P = pi['interference_matrix']
    count_s3, count_s2 = {}, {}
    best_active, best_asg = list(active), {k: set(v) for k, v in assignments.items()}
    best_value, best_count = weighted_value(active, pi), len(active)
    kicks_since_best, tabu = 0, []
    pool_used = False
    iteration = no_improve = 0
    t0 = time.perf_counter()
    while time.perf_counter() < deadline:
        inactive = [c for c in range(pi['n_carriers']) if c not in active]
        if not inactive:
            break
        n_s2, n_s3 = sizes(len(active), len(inactive))
        h = headroom(pi, active, assignments)
        stuck = not any(h[c] >= 0 for c in inactive)
        N_S2, N_S3 = (None, None) if stuck else select_neighbourhood(
            inactive, active, count_s3, count_s2, h, n_s2, n_s3, P)
        if stuck or N_S2 is None:
            no_improve = max(no_improve, KICK_STAGNATION)
        else:
            before = len(active)
            assignments = shift_slots(pi, active, assignments, N_S2)
            post = headroom(pi, active, assignments)
            eligible = [c for c in N_S3 if post[c] >= 0]
            if eligible:
                new, assignments = activate(pi, active, assignments, eligible, rng_seed=seed + iteration)
                active = active + new
            iteration += 1
            no_improve = 0 if len(active) > before else no_improve + 1
        value = weighted_value(active, pi)
        if value > best_value:
            best_value, best_count = value, len(active)
            best_active, best_asg = list(active), {k: set(v) for k, v in assignments.items()}
            kicks_since_best = 0
        if stall is not None and stall.update(time.perf_counter() - t0, best_value, best_count):
            break
        if no_improve < KICK_STAGNATION:
            continue
        if not pool_used:
            pool = select_repair_pool(pi, active, assignments)
            if pool:
                pool_used = True
                sol = solve_pool(pi, pool, assignments,
                                 min(cp_time_limit, max(1.0, deadline - time.perf_counter()))) or {}
                cand = dict(assignments)
                for c in pool:
                    cand[c] = sol.get(c, set())
                cand_value = weighted_value([c for c in cand if cand[c]], pi)
                if cand_value > value and sanity_check(cand, pi):
                    assignments = cand
                    active = [c for c in assignments if assignments[c]]
                    no_improve = 0
                    if cand_value > best_value:
                        best_value, best_count = cand_value, len(active)
                        best_active, best_asg = list(active), {k: set(v) for k, v in assignments.items()}
                        kicks_since_best = 0
                    continue
        if kicks_since_best >= KICK_GIVEUP:
            break
        kick = set()
        for criterion in ('blocking', 'marginal'):
            for exclude in (set(tabu), set()):
                kick = select_kick(pi, best_active, best_asg, exclude, criterion)
                if kick:
                    break
            if kick:
                break
        if not kick:
            break
        active = [c for c in best_active if c not in kick]
        assignments = {c: set(v) for c, v in best_asg.items() if c not in kick}
        for c in kick:
            count_s3[c] = 0
            count_s2.pop(c, None)
        tabu.extend(kick)
        tabu = tabu[-KICK_TABU_ROUNDS * max(1, len(kick)):]
        no_improve = 0
        kicks_since_best += 1
    pool = select_repair_pool(pi, best_active, best_asg)
    if pool:
        sol = solve_pool(pi, pool, best_asg, min(30, max(1.0, deadline - time.perf_counter()))) or {}
        cand = dict(best_asg)
        for c in pool:
            cand[c] = sol.get(c, set())
        cand_active = [c for c in cand if cand[c]]
        if weighted_value(cand_active, pi) > best_value and sanity_check(cand, pi):
            best_active, best_asg = cand_active, cand
    return best_active, best_asg
