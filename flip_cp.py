import random
import time
from itertools import combinations

import numpy as np
from ortools.sat.python import cp_model

from construction import flip_random, svals_to_assignments
from problem import weighted_value


def _overlap(svals, sub):
    n = len(svals)
    M = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        if svals[i] == 0 or svals[j] == 0:
            continue
        o = min(svals[i] + sub[i] - 1, svals[j] + sub[j] - 1) - max(svals[i], svals[j]) + 1
        M[i, j] = M[j, i] = max(0, o)
    return M


def repair(svals, pi, time_limit):
    P = pi['interference_matrix']
    thr = pi['threshold']
    sub = pi['sub_channels']
    S = pi['max_subchannels']
    n = pi['n_carriers']
    w = pi['weight']
    p = P.copy()
    for members in pi['allocated_users'].values():
        for i in members:
            for j in members:
                if i != j:
                    p[i, j] = thr[i] + 1
    free = np.where(svals == 0)[0]
    o = _overlap(svals, sub).astype(int)
    model = cp_model.CpModel()
    start, active = {}, {}
    for i in range(n):
        if i in free:
            start[i] = model.NewIntVar(1, S - sub[i] + 1, f"s{i}")
            active[i] = model.NewBoolVar(f"a{i}")
        else:
            start[i] = int(svals[i])
            active[i] = 1
    product = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            mo = min(sub[i], sub[j])
            if i not in free and j not in free:
                product[(i, j)] = o[i, j] * p[i, j]
                product[(j, i)] = o[i, j] * p[j, i]
            elif (i, j) not in product:
                ov = model.NewIntVar(0, mo, f"o{i}_{j}")
                lo = model.NewIntVar(0, S, f"os{i}_{j}")
                hi = model.NewIntVar(0, S, f"oe{i}_{j}")
                model.AddMinEquality(hi, [start[i] + sub[i] - 1, start[j] + sub[j] - 1])
                model.AddMaxEquality(lo, [start[i], start[j]])
                model.AddMaxEquality(ov, [0, hi - lo + 1])
                for a, b in ((i, j), (j, i)):
                    wt = model.NewIntVar(0, p[a, b] * mo, f"w{a}_{b}")
                    pr = model.NewIntVar(0, p[a, b] * mo, f"p{a}_{b}")
                    model.AddMultiplicationEquality(wt, [int(p[a, b]), ov])
                    model.AddMultiplicationEquality(pr, [wt, active[b]])
                    product[(a, b)] = pr
    for i in range(n):
        model.Add(sum(product[(i, j)] for j in range(n) if (i, j) in product) <= thr[i]).OnlyEnforceIf(active[i])
    model.Maximize(sum(int(w[i]) * active[i] for i in range(n)))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    out = np.zeros(n, dtype=int)
    if solver.Solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i in range(n):
            if isinstance(active[i], int) or solver.Value(active[i]):
                out[i] = start[i] if isinstance(start[i], int) else solver.Value(start[i])
    if not out.any():
        return np.asarray(svals, dtype=int)
    return out


def flip_cp(pi, budget, seed):
    deadline = time.perf_counter() + budget
    restart_budget = min(max(2.0, budget * 0.3), budget)
    t0 = time.perf_counter()
    cp_time = max(1.0, min(budget * 0.7, deadline - t0 - restart_budget))
    random.seed(seed)
    best_val, best = -1, None
    while time.perf_counter() - t0 < restart_budget:
        svals = flip_random(pi)
        val = weighted_value([i for i in range(len(svals)) if svals[i] > 0], pi)
        if val > best_val:
            best_val, best = val, svals
    if best is None:
        best = flip_random(pi)
    asg = svals_to_assignments(repair(best, pi, cp_time), pi)
    return [i for i in asg if asg[i]], asg
