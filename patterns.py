import time

import numpy as np
import scipy.sparse as sp
from ortools.sat.python import cp_model
from scipy.optimize import linprog, milp, LinearConstraint, Bounds


def pattern_power(pi, patterns):
    P = pi['interference_matrix']
    out = []
    for pat in patterns:
        S = np.fromiter(pat, dtype=int)
        q = P[np.ix_(S, S)].sum(axis=1)
        out.append({int(c): int(np.ceil(v)) for c, v in zip(S, q)})
    return out


def z_upper_bounds(pi, pp):
    G = pi['threshold']
    ubs = []
    for d in pp:
        u = pi['max_subchannels']
        for i, q in d.items():
            if q > 0:
                u = min(u, int(G[i] // q))
        ubs.append(max(0, int(u)))
    return ubs


def patterns_from_assignment(pi, assignments, active):
    occ = [[] for _ in range(pi['max_subchannels'])]
    for c in set(active):
        for k in assignments.get(c, ()):
            occ[k].append(c)
    counts = {}
    for carriers in occ:
        if carriers:
            f = frozenset(carriers)
            counts[f] = counts.get(f, 0) + 1
    return list(counts), list(counts.values())


class LPTimeLimit(RuntimeError):
    pass


def solve_rmp_lp(pi, patterns, pp, time_limit=None):
    n = pi['n_carriers']
    nP = len(patterns)
    inv = {i: [] for i in range(n)}
    for j, pat in enumerate(patterns):
        for i in pat:
            inv[i].append(j)
    rows, cols, vals = [], [], []
    b = np.zeros(2 * n + 1)
    for i in range(n):
        rows.append(i); cols.append(nP + i); vals.append(float(pi['sub_channels'][i]))
        for j in inv[i]:
            rows.append(i); cols.append(j); vals.append(-1.0)
            if pp[j][i]:
                rows.append(n + i); cols.append(j); vals.append(float(pp[j][i]))
        b[n + i] = pi['threshold'][i]
    for j in range(nP):
        rows.append(2 * n); cols.append(j); vals.append(1.0)
    b[2 * n] = pi['max_subchannels']
    A = sp.csr_matrix((vals, (rows, cols)), shape=(2 * n + 1, nP + n))
    c = np.zeros(nP + n)
    c[nP:] = -pi['weight'].astype(float)
    ub = np.concatenate([np.full(nP, pi['max_subchannels'], float), np.ones(n)])
    opts = {} if time_limit is None else {'time_limit': float(time_limit)}
    r = linprog(c, A_ub=A, b_ub=b, bounds=list(zip(np.zeros(nP + n), ub)), method='highs', options=opts)
    if not r.success:
        if 'ime limit' in str(r.message) or r.status == 1:
            raise LPTimeLimit(r.message)
        raise RuntimeError(r.message)
    duals = -r.ineqlin.marginals
    return dict(obj=-r.fun, x=r.x[nP:], pi=duals[:n], lam=duals[n:2 * n], sig=float(duals[2 * n]))


def lp_activation(pi, pool, max_pool=4000):
    if len(pool) < 2:
        return None, None
    try:
        pats = [tuple(sorted(p)) for p in pool][:max_pool]
        sol = solve_rmp_lp(pi, [frozenset(p) for p in pats], pattern_power(pi, pats))
        return np.clip(np.asarray(sol['x'], dtype=float), 0.0, 1.0), float(sol['obj'])
    except Exception:
        return None, None


def _pair_weights(pi, cand, lam):
    C = np.asarray(cand)
    Wd = lam[C][:, None] * pi['interference_matrix'][np.ix_(C, C)]
    return Wd + Wd.T


def price_heuristic(pi, pi_d, lam, sig, rng, n_restarts=12):
    cand = np.flatnonzero(pi_d > 1e-9)
    if cand.size == 0:
        return []
    W = _pair_weights(pi, cand, lam)
    pv = pi_d[cand]
    beam = np.asarray(pi['allocated_beams'])[cand]
    beams = {}
    for a, b in enumerate(beam):
        beams.setdefault(int(b), []).append(a)
    found = {}
    for r in range(n_restarts):
        sel = []
        order = list(beams)
        rng.shuffle(order)
        for b in order:
            best, bestg = None, 1e-12
            for a in beams[b]:
                g = pv[a] - (W[a, sel].sum() if sel else 0.0)
                if g > bestg or (r and g > 1e-12 and rng.random() < 0.15):
                    best, bestg = a, g
            if best is not None:
                sel.append(best)
        improved = True
        while improved:
            improved = False
            for b in list(beams):
                cur = [a for a in sel if beam[a] == b]
                base = [a for a in sel if beam[a] != b]
                bestcand, bestg = None, (pv[cur[0]] - W[cur[0], base].sum()) if cur else 0.0
                for a in beams[b]:
                    g = pv[a] - (W[a, base].sum() if base else 0.0)
                    if g > bestg + 1e-9:
                        bestcand, bestg = a, g
                if bestcand is not None:
                    sel = base + [bestcand]
                    improved = True
                elif cur and bestg < -1e-9:
                    sel = base
                    improved = True
        if not sel:
            continue
        rc = pv[sel].sum() - 0.5 * W[np.ix_(sel, sel)].sum() - sig
        if rc > 1e-6:
            p = frozenset(int(cand[a]) for a in sel)
            if p not in found or rc > found[p]:
                found[p] = float(rc)
    return sorted(found.items(), key=lambda t: -t[1])


def price_exact(pi, pi_d, lam, sig, time_limit):
    cand = np.flatnonzero(pi_d > 1e-9)
    if cand.size == 0:
        return None, -sig, True
    W = _pair_weights(pi, cand, lam)
    pv = pi_d[cand]
    m = cand.size
    pairs = [(a, b) for a in range(m) for b in range(a + 1, m) if W[a, b] > 1e-12]
    nv = m + len(pairs)
    c = np.zeros(nv)
    c[:m] = -pv
    rows, hi = [], []
    for t, (a, b) in enumerate(pairs):
        c[m + t] = W[a, b]
        r = np.zeros(nv)
        r[a] = r[b] = 1
        r[m + t] = -1
        rows.append(r)
        hi.append(1.0)
    beam = np.asarray(pi['allocated_beams'])[cand]
    for b in np.unique(beam):
        r = np.zeros(nv)
        r[np.flatnonzero(beam == b)] = 1
        rows.append(r)
        hi.append(1.0)
    integrality = np.zeros(nv)
    integrality[:m] = 1
    res = milp(c=c, constraints=LinearConstraint(np.array(rows), -np.inf, hi), integrality=integrality,
               bounds=Bounds(np.zeros(nv), np.concatenate([np.ones(m), np.full(len(pairs), np.inf)])),
               options=dict(time_limit=time_limit))
    if res.x is None:
        return None, None, False
    sel = [a for a in range(m) if res.x[a] > 0.5]
    proven = res.status == 0
    if not sel:
        return None, -sig, proven
    return frozenset(int(cand[a]) for a in sel), float(-res.fun - sig), proven


def column_generation(pi, time_limit, max_iter=300):
    pats = [frozenset([i]) for i in range(pi['n_carriers'])]
    pp = pattern_power(pi, pats)
    t0 = time.time()
    rng = np.random.default_rng(0)

    def left():
        return time_limit - (time.time() - t0)

    for _ in range(max_iter):
        if left() <= 0:
            break
        try:
            sol = solve_rmp_lp(pi, pats, pp, time_limit=max(1.0, left()))
        except LPTimeLimit:
            break
        if left() <= 0:
            break
        cols = price_heuristic(pi, sol['pi'], sol['lam'], sol['sig'], rng)
        if not cols:
            p, rc, _ = price_exact(pi, sol['pi'], sol['lam'], sol['sig'], max(1.0, min(60.0, left())))
            if p is None or rc is None or rc <= 1e-6:
                break
            cols = [(p, rc)]
        added = 0
        for p, _ in cols[:20]:
            if p not in set(pats):
                pats.append(p)
                pp.append(pattern_power(pi, [p])[0])
                added += 1
        if added == 0 or left() <= 0:
            break
    return pats


def pattern_selection(pi, patterns, time_limit, hint=None, conditional=False):
    n = pi['n_carriers']
    pp = pattern_power(pi, patterns)
    ub = z_upper_bounds(pi, pp)
    keep = [j for j in range(len(patterns)) if ub[j] > 0]
    patterns = [patterns[j] for j in keep]
    pp = [pp[j] for j in keep]
    ub = [ub[j] for j in keep]
    inv = {i: [] for i in range(n)}
    for j, pat in enumerate(patterns):
        for i in pat:
            inv[i].append(j)
    m = cp_model.CpModel()
    z = [m.NewIntVar(0, ub[j], f"z_{j}") for j in range(len(patterns))]
    x = [m.NewBoolVar(f"x_{i}") for i in range(n)]
    for i in range(n):
        if not inv[i]:
            m.Add(x[i] == 0)
            continue
        m.Add(sum(z[j] for j in inv[i]) >= int(pi['sub_channels'][i]) * x[i])
        terms = [pp[j][i] * z[j] for j in inv[i] if pp[j][i]]
        if terms:
            ct = m.Add(sum(terms) <= int(pi['threshold'][i]))
            if conditional:
                ct.OnlyEnforceIf(x[i])
    m.Add(sum(z) <= pi['max_subchannels'])
    w = pi['weight']
    m.Maximize(sum(int(w[i]) * x[i] for i in range(n)))
    if hint:
        idx = {p: j for j, p in enumerate(patterns)}
        for p, mult in hint.items():
            if p in idx:
                m.AddHint(z[idx[p]], int(mult))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = float(time_limit)
    s.parameters.num_workers = 8
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None
    zv = [s.Value(v) for v in z]
    xv = [s.Value(v) for v in x]
    obj = int(sum(int(w[i]) * xv[i] for i in range(n)))
    return obj, extract_assignment(pi, patterns, zv, xv)


def extract_assignment(pi, patterns, z, x):
    asg = {i: set() for i in range(pi['n_carriers'])}
    k = 0
    for j, pat in enumerate(patterns):
        for _ in range(int(z[j])):
            for i in pat:
                asg[i].add(k)
            k += 1
    for i in asg:
        if not x[i]:
            asg[i] = set()
        elif len(asg[i]) > pi['sub_channels'][i]:
            asg[i] = set(sorted(asg[i])[:int(pi['sub_channels'][i])])
    return asg
