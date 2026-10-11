import time

from ortools.sat.python import cp_model

from csa_lns import solve_pool
from problem import sanity_check, weighted_value

ALPHA = 0.3


def _pairs(pi):
    P = pi['interference_matrix']
    thr = pi['threshold']
    beam = pi['allocated_beams']
    keep, forbid = [], []
    n = pi['n_carriers']
    for i in range(n):
        for j in range(i + 1, n):
            if beam[i] == beam[j] or (P[i][j] == 0 and P[j][i] == 0):
                continue
            if P[i][j] > thr[i] or P[j][i] > thr[j]:
                forbid.append((i, j))
            else:
                keep.append((i, j))
    return keep, forbid


def relaxed_model(pi, time_limit, seed, min_value=None):
    n = pi['n_carriers']
    S = pi['max_subchannels']
    sub = pi['sub_channels']
    thr = pi['threshold']
    P = pi['interference_matrix']
    w = pi['weight']
    keep, forbid = _pairs(pi)
    model = cp_model.CpModel()
    y, s, e, iv = {}, {}, {}, {}
    for c in range(n):
        m = int(sub[c])
        y[c] = model.NewBoolVar(f'y_{c}')
        s[c] = model.NewIntVar(0, S - m, f's_{c}')
        e[c] = model.NewIntVar(0, S, f'e_{c}')
        length = model.NewIntVar(0, m, f'l_{c}')
        model.Add(length == m * y[c])
        model.Add(e[c] == s[c] + length)
        iv[c] = model.NewIntervalVar(s[c], length, e[c], f'iv_{c}')
    for members in pi['allocated_users'].values():
        if len(members) > 1:
            model.AddNoOverlap([iv[c] for c in members])
    for i, j in forbid:
        model.AddNoOverlap([iv[i], iv[j]])
    terms = {c: [] for c in range(n)}
    for i, j in keep:
        hi = model.NewIntVar(-S, S, f'v1_{i}_{j}')
        lo = model.NewIntVar(-S, S, f'v2_{i}_{j}')
        ov = model.NewIntVar(0, min(int(sub[i]), int(sub[j])), f'ov_{i}_{j}')
        model.AddMinEquality(hi, [e[i], e[j]])
        model.AddMaxEquality(lo, [s[i], s[j]])
        model.AddMaxEquality(ov, [0, hi - lo])
        if int(P[i][j]) > 0:
            terms[i].append(int(P[i][j]) * ov)
        if int(P[j][i]) > 0:
            terms[j].append(int(P[j][i]) * ov)
    for c in range(n):
        if terms[c]:
            excess = model.NewIntVar(0, int(ALPHA * thr[c]), f'viol_{c}')
            model.Add(sum(terms[c]) <= int(thr[c]) + excess)
    if min_value is not None:
        model.Add(sum(int(w[c]) * y[c] for c in range(n)) >= min_value)
    model.Maximize(sum(int(w[c]) * y[c] for c in range(n)))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 4
    solver.parameters.random_seed = seed
    status = solver.Solve(model)
    assignments = {}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for c in range(n):
            start = solver.Value(s[c])
            assignments[c] = set(range(start, start + int(sub[c]))) if solver.Value(y[c]) == 1 else set()
    active = [c for c in assignments if assignments[c]]
    violations = {}
    for i in active:
        load = sum(P[i][j] * len(assignments[i] & assignments[j]) for j in active if j != i)
        if load > thr[i]:
            violations[i] = load - thr[i]
    return dict(status=solver.StatusName(status), value=weighted_value(active, pi),
                assignments=assignments, violations=violations)


def relax_and_repair(pi, incumbent_active, incumbent_assignments, budget, seed):
    deadline = time.perf_counter() + budget
    inc_value = weighted_value(incumbent_active, pi)
    max_value = weighted_value(range(pi['n_carriers']), pi)
    unchanged = (incumbent_active, incumbent_assignments)

    def window(nominal):
        left = deadline - time.perf_counter()
        return None if left < 1.0 else min(float(nominal), left)

    def solved(r):
        return r['status'] in ('OPTIMAL', 'FEASIBLE')

    t = window(max(2.0, budget * 0.25))
    if t is None:
        return unchanged
    p1 = relaxed_model(pi, t, seed)
    if p1['status'] != 'OPTIMAL':
        floor = max(inc_value + 1, p1['value'] + 1)
        t = window(max(2.0, budget * 0.25))
        if t is not None and floor <= max_value:
            floored = relaxed_model(pi, t, seed, min_value=floor)
            if solved(floored) and floored['value'] > inc_value:
                p1 = floored
    if p1['value'] <= inc_value or not solved(p1):
        return unchanged
    if not p1['violations']:
        merged = {c: v for c, v in p1['assignments'].items() if v}
        return (sorted(merged), p1['assignments']) if sanity_check(merged, pi) else unchanged
    violated = sorted(p1['violations'], key=lambda c: -p1['violations'][c])
    t = window(max(5.0, budget * 0.50))
    if t is None:
        return unchanged
    repaired_pool = solve_pool(pi, violated, p1['assignments'], t)
    if repaired_pool is None:
        return unchanged
    repaired = dict(p1['assignments'])
    for c in violated:
        repaired[c] = repaired_pool.get(c, set())
    merged = {c: v for c, v in repaired.items() if v}
    if weighted_value(list(merged), pi) > inc_value and sanity_check(merged, pi):
        return sorted(merged), repaired
    return unchanged
