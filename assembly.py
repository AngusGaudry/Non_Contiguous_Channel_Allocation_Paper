import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from patterns import pattern_power, z_upper_bounds, solve_rmp_lp, pattern_selection
from problem import sanity_check

HEURISTIC_FILTER_MIN = 600


def _relaxation(pi, pats, pp, ub):
    n = pi['n_carriers']
    nP = len(pats)
    inv = {i: [] for i in range(n)}
    for j, p in enumerate(pats):
        for i in p:
            inv[i].append(j)
    rows, cols, vals = [], [], []
    b = np.zeros(2 * n + 1)
    for i in range(n):
        rows.append(i); cols.append(nP + i); vals.append(float(pi['sub_channels'][i]))
        for j in inv[i]:
            rows.append(i); cols.append(j); vals.append(-1.0)
            if pp[j][i]:
                rows.append(n + i); cols.append(j); vals.append(float(pp[j][i]))
        b[n + i] = float(int(pi['threshold'][i]))
    for j in range(nP):
        rows.append(2 * n); cols.append(j); vals.append(1.0)
    b[2 * n] = pi['max_subchannels']
    A = sp.csr_matrix((vals, (rows, cols)), shape=(2 * n + 1, nP + n))
    c = np.concatenate([np.zeros(nP), pi['weight'].astype(float)])
    u = np.concatenate([np.array(ub, float), np.ones(n)])
    return A, b, c, u


def safe_filter(pi, patterns, incumbent_value, time_limit, tol=1e-6):
    pats = list(patterns)
    pp = pattern_power(pi, pats)
    ub = z_upper_bounds(pi, pp)
    live = [j for j in range(len(pats)) if ub[j] > 0]
    lp_pats = [pats[j] for j in live]
    try:
        A, b, c, u = _relaxation(pi, lp_pats, [pp[j] for j in live], [ub[j] for j in live])
        r = linprog(-c, A_ub=A, b_ub=b, bounds=list(zip(np.zeros_like(u), u)), method='highs',
                    options={'time_limit': float(time_limit)})
        marg = None if r.ineqlin is None else r.ineqlin.marginals
        if marg is None or not np.all(np.isfinite(marg)):
            return pats, False
        y = np.maximum(-np.asarray(marg, float), 0.0)
        d = c - A.T @ y
        L = float(b @ y + u @ np.maximum(d, 0.0))
    except Exception:
        return pats, False
    target = incumbent_value + 1
    if L < target - tol:
        return [], True
    score = L + np.minimum(d[:len(lp_pats)], 0.0)
    return [p for p, sc in zip(lp_pats, score) if sc >= target - tol], False


def lp_support_filter(pi, patterns, time_limit, eps=1e-6):
    try:
        sol = solve_rmp_lp(pi, patterns, pattern_power(pi, patterns), time_limit=time_limit)
    except Exception:
        return list(patterns)
    P = pi['interference_matrix']
    keep = []
    for p in patterns:
        S = np.fromiter(p, dtype=int)
        rc = sol['pi'][S].sum() - sol['sig'] - float(sol['lam'][S] @ P[np.ix_(S, S)].sum(axis=1))
        if rc >= -eps:
            keep.append(p)
    return keep or list(patterns)


def assemble(pi, bb, time_limit, deadline=None):
    def window():
        if deadline is None:
            return time_limit
        return max(1.0, min(time_limit, deadline - time.perf_counter()))

    ps, ms = bb.incumbent_patterns(pi)
    inc_pats = set(frozenset(p) for p in ps)
    pool = list(set(frozenset(p) for p in bb.pool) | inc_pats)
    pool, certified = safe_filter(pi, pool, bb.incumbent['objective'], 0.25 * window())
    if certified:
        return
    if len(pool) > HEURISTIC_FILTER_MIN:
        pool = lp_support_filter(pi, pool, 0.25 * window())
    pool = list(set(frozenset(p) for p in pool) | inc_pats)
    obj, cand = pattern_selection(pi, pool, window(), hint=dict(zip(ps, ms)))
    if obj is not None and obj > bb.incumbent['objective'] and sanity_check(cand, pi):
        bb.post_incumbent([c for c in cand if cand[c]], cand, obj, 'assembly')
