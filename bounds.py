import os
import time
import contextlib

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

from patterns import pattern_power, solve_rmp_lp, price_heuristic

CERT_TOL = 1e-6


def floor(v):
    return None if v is None else float(np.floor(v + 1e-9))


def counting_bound(pi):
    n = pi['n_carriers']
    S = float(pi['max_subchannels'])
    nsub = np.asarray(pi['sub_channels'], dtype=float)
    gam = np.asarray(pi['threshold'], dtype=float)
    P = np.asarray(pi['interference_matrix'], dtype=float)
    ab = np.asarray(pi['allocated_beams'])
    w = np.asarray(pi['weight'], dtype=float)
    Q = P + P.T
    np.fill_diagonal(Q, np.inf)
    Q = np.where(ab[:, None] == ab[None, :], np.inf, Q)
    cap = np.minimum(nsub[:, None], nsub[None, :])
    iu = np.triu_indices(n, k=1)
    q, cp = Q[iu], cap[iu]
    keep = np.isfinite(q)
    q, cp = q[keep], cp[keep]
    if q.size == 0:
        return float(w.sum())
    order = np.argsort(q, kind='stable')
    q, cp = q[order], cp[order]
    cum_units = np.cumsum(cp)
    cum_cost = np.cumsum(q * cp)

    def cheapest_cost(R):
        if R <= 0:
            return 0.0
        if R > cum_units[-1]:
            return np.inf
        j = int(np.searchsorted(cum_units, R))
        pu = cum_units[j - 1] if j else 0.0
        pc = cum_cost[j - 1] if j else 0.0
        return float(pc + (R - pu) * q[j])

    cumN = np.cumsum(np.sort(nsub))
    cumG = np.cumsum(np.sort(gam)[::-1])
    cumW = np.cumsum(np.sort(w)[::-1])
    k_ok = 0
    for k in range(1, n + 1):
        a, r = divmod(int(round(cumN[k - 1])), int(round(S)))
        R = r * (a + 1) * a / 2.0 + (int(round(S)) - r) * a * (a - 1) / 2.0
        if cheapest_cost(R) <= cumG[k - 1] + 1e-9:
            k_ok = k
    return float(cumW[k_ok - 1]) if k_ok > 0 else 0.0


def scale_free_bound(pi):
    return float(min(float(np.asarray(pi['weight']).sum()), counting_bound(pi)))


def psi_segments(P_row, members_c, nsub, n_i):
    caps = np.minimum(n_i, nsub[members_c]).astype(float)
    costs = np.asarray(P_row)[members_c].astype(float)
    order = np.argsort(costs, kind='stable')
    caps, costs = caps[order], costs[order]
    segs = []
    x = y = 0.0
    for cp, cst in zip(caps, costs):
        if x >= n_i:
            break
        take = min(cp, n_i - x)
        segs.append((x, y, cst))
        y += take * cst
        x += take
    return segs, x


class _Curves:
    def __init__(self, P, members, nsub):
        self.P = P
        self.members = members
        self.nsub = nsub
        self._c = {}

    def get(self, i, c):
        k = (int(i), int(c))
        if k not in self._c:
            self._c[k] = psi_segments(self.P[k[0]], self.members[k[1]], self.nsub, float(self.nsub[k[0]]))
        return self._c[k]

    def value(self, i, c, x):
        segs, cap = self.get(i, c)
        if cap <= 0 or x <= 0:
            return 0.0
        return max(ys + sl * (x - xs) for (xs, ys, sl) in segs)


def _aggregation_model(pi, pair_set, curves, integral, time_limit):
    n = pi['n_carriers']
    S = int(pi['max_subchannels'])
    nsub = np.asarray(pi['sub_channels'], dtype=float)
    gam = np.asarray(pi['threshold'], dtype=float)
    w = np.asarray(pi['weight'], dtype=float)
    members = curves.members
    need = set()
    for b, c in pair_set:
        for i in members[b]:
            need.add((int(i), int(c)))
        for j in members[c]:
            need.add((int(j), int(b)))
    O, t, segs_of = {}, {}, {}
    nv = n
    for key in sorted(need):
        segs, cap = curves.get(*key)
        if cap <= 0 or not segs:
            continue
        O[key] = nv
        nv += 1
        segs_of[key] = segs
        if len(segs) > 1:
            t[key] = nv
            nv += 1
    rows, cols, vals, rhs = [], [], [], []

    def add(entries, bound):
        r = len(rhs)
        for cc, vv in entries:
            rows.append(r)
            cols.append(cc)
            vals.append(vv)
        rhs.append(bound)

    for b, c in pair_set:
        for src, dst in ((b, c), (c, b)):
            ent = [(O[(int(i), int(dst))], -1.0) for i in members[src] if (int(i), int(dst)) in O]
            if not ent:
                continue
            ent += [(int(i), float(nsub[i])) for i in members[src]]
            ent += [(int(j), float(nsub[j])) for j in members[dst]]
            add(ent, float(S))
    for (i, c), oi in O.items():
        add([(oi, 1.0), (i, -float(curves.get(i, c)[1]))], 0.0)
    for key, ti in t.items():
        for xs, ys, sl in segs_of[key]:
            add([(ti, -1.0), (O[key], sl)], sl * xs - ys)
    by_recv = {}
    for i, c in O:
        by_recv.setdefault(i, []).append(c)
    for i, cs in by_recv.items():
        ent = [(i, -float(gam[i]))]
        for c in cs:
            ent.append((t[(i, c)], 1.0) if (i, c) in t else (O[(i, c)], float(segs_of[(i, c)][0][2])))
        add(ent, 0.0)
    for b in sorted(members):
        add([(int(i), float(nsub[i])) for i in members[b]], float(S))
    A = sp.csr_matrix((vals, (rows, cols)), shape=(len(rhs), nv))
    c_obj = np.zeros(nv)
    c_obj[:n] = -w
    ub = np.full(nv, np.inf)
    ub[:n] = 1.0
    if not integral:
        res = linprog(c_obj, A_ub=A, b_ub=np.array(rhs), bounds=list(zip(np.zeros(nv), ub)), method='highs')
        if not res.success:
            return None, None
        return float(-res.fun), res.x[:n]
    integrality = np.zeros(nv)
    integrality[:n] = 1
    res = milp(c=c_obj, constraints=LinearConstraint(A, -np.inf, np.array(rhs)), integrality=integrality,
               bounds=Bounds(np.zeros(nv), ub), options=dict(time_limit=max(1.0, time_limit)))
    db = getattr(res, 'mip_dual_bound', None)
    if db is None or not np.isfinite(db):
        return None, None
    return float(-db), (res.x[:n] if res.x is not None else None)


def aggregation_bound(pi, deadline, seed_pairs=200, add_per_round=300, integral=False, pairs=None):
    S = int(pi['max_subchannels'])
    nsub = np.asarray(pi['sub_channels'], dtype=float)
    gam = np.asarray(pi['threshold'], dtype=float)
    ab = np.asarray(pi['allocated_beams'])
    beams = sorted(set(int(b) for b in ab))
    members = {b: np.flatnonzero(ab == b) for b in beams}
    Nfull = {b: float(nsub[members[b]].sum()) for b in beams}
    curves = _Curves(np.asarray(pi['interference_matrix'], dtype=float), members, nsub)

    def score(b, c, Nb, Nc):
        F = Nb + Nc - S
        if F <= 0:
            return 0.0
        cost = 0.0
        for src, dst in ((b, c), (c, b)):
            share = F / max(1, len(members[src]))
            cost += sum(curves.value(int(i), dst, share) for i in members[src])
        return cost / max(gam[members[b]].sum() + gam[members[c]].sum(), 1.0)

    allp = [(b, c) for ii, b in enumerate(beams) for c in beams[ii + 1:] if Nfull[b] + Nfull[c] - S > 0]
    if not allp:
        return None, True, []
    if pairs is not None:
        active, remaining = list(pairs), []
    else:
        scored = sorted(allp, key=lambda p: -score(p[0], p[1], Nfull[p[0]], Nfull[p[1]]))
        active, remaining = scored[:seed_pairs], scored[seed_pairs:]
    best, converged = None, False
    prev_t = prev_sz = None
    while True:
        rem = deadline - time.perf_counter()
        if rem <= 0:
            break
        sz = len(active)
        if prev_t is not None and prev_t * (sz / max(prev_sz, 1)) ** 1.5 > rem:
            break
        t_round = time.perf_counter()
        val, yv = _aggregation_model(pi, active, curves, integral, rem)
        prev_t, prev_sz = time.perf_counter() - t_round, sz
        if val is not None and (best is None or val < best):
            best = val
        if not remaining:
            converged = True
            break
        if yv is None or time.perf_counter() >= deadline:
            break
        Ny = {b: float(nsub[members[b]] @ yv[members[b]]) for b in beams}
        viol = [(score(b, c, Ny[b], Ny[c]), (b, c)) for b, c in remaining]
        viol = sorted((v for v in viol if v[0] > 0), key=lambda t: -t[0])
        if not viol:
            converged = True
            break
        grow = max(1.0, ((deadline - time.perf_counter()) / max(prev_t, 1e-6)) ** (1.0 / 1.5))
        take = [p for _, p in viol[:min(add_per_round, max(0, int(prev_sz * grow) - prev_sz))]]
        if not take:
            break
        active += take
        taken = set(take)
        remaining = [p for p in remaining if p not in taken]
    return best, converged, active


def beam_relaxation_bound(pi, pi_d):
    ab = np.asarray(pi['allocated_beams'])
    pos = np.maximum(np.asarray(pi_d, dtype=float), 0.0)
    if not pos.any():
        return 0.0
    best = np.zeros(int(ab.max()) + 1)
    np.maximum.at(best, ab, pos)
    return float(best.sum())


def pricing_milp_bound(pi, pi_d, lam, time_limit):
    P = np.asarray(pi['interference_matrix'], dtype=float)
    ab = np.asarray(pi['allocated_beams'])
    pi_d = np.asarray(pi_d, dtype=float)
    lam = np.asarray(lam, dtype=float)
    C = np.flatnonzero(pi_d > 1e-9)
    if C.size == 0:
        return 0.0
    m = C.size
    pv = pi_d[C]
    Wd = lam[C][:, None] * P[np.ix_(C, C)]
    W = Wd + Wd.T
    np.fill_diagonal(W, 0.0)
    if m == 1:
        return float(max(0.0, pv[0]))
    ai, bi = np.triu_indices(m, k=1)
    keep = W[ai, bi] > 1e-12
    ai, bi, wv = ai[keep], bi[keep], W[ai, bi][keep]
    npair = ai.size
    nv = m + npair
    c = np.zeros(nv)
    c[:m] = -pv
    c[m:] = wv
    rows, cols, vals = [], [], []
    for t in range(npair):
        rows += [t, t, t]
        cols += [int(ai[t]), int(bi[t]), m + t]
        vals += [1.0, 1.0, -1.0]
    r = npair
    beam = ab[C]
    for b in np.unique(beam):
        idx = np.flatnonzero(beam == b)
        rows += [r] * idx.size
        cols += [int(i) for i in idx]
        vals += [1.0] * idx.size
        r += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, nv))
    integrality = np.zeros(nv)
    integrality[:m] = 1
    res = milp(c=c, constraints=LinearConstraint(A, -np.inf, 1.0), integrality=integrality,
               bounds=Bounds(np.zeros(nv), np.concatenate([np.ones(m), np.full(npair, np.inf)])),
               options=dict(time_limit=time_limit, mip_rel_gap=0.0))
    db = getattr(res, 'mip_dual_bound', None)
    if db is None or not np.isfinite(db):
        return float(np.maximum(pv, 0).sum())
    return float(max(0.0, -db))


def reduced_cost_bound(pi, sol, deadline, current_ub):
    S = pi['max_subchannels']
    lp, sig = sol['obj'], sol['sig']
    rho_ub = beam_relaxation_bound(pi, sol['pi'])
    v = lp + S * max(0.0, rho_ub - sig)
    best = v if v < current_ub else None
    rem = deadline - time.perf_counter()
    if rem > 0.3:
        try:
            with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
                uf = pricing_milp_bound(pi, sol['pi'], sol['lam'], rem)
            rho_ub = min(rho_ub, uf)
            v = lp + S * max(0.0, uf - sig)
            if best is None or v < best:
                best = v
        except Exception:
            pass
    return best, rho_ub <= sig + CERT_TOL


def cg_columns(pi, deadline, init_patterns=None, max_iter=5000):
    pats = [frozenset([i]) for i in range(pi['n_carriers'])]
    for p in init_patterns or ():
        if len(p) > 1:
            pats.append(frozenset(p))
    seen = set(pats)
    pp = pattern_power(pi, pats)
    rng = np.random.default_rng(0)
    sol = None
    exhausted = False
    for _ in range(max_iter):
        if time.perf_counter() >= deadline:
            break
        sol = solve_rmp_lp(pi, pats, pp)
        cols = price_heuristic(pi, sol['pi'], sol['lam'], sol['sig'], rng)
        added = 0
        for p, _ in cols[:20]:
            if p not in seen:
                seen.add(p)
                pats.append(p)
                pp.append(pattern_power(pi, [p])[0])
                added += 1
        if added == 0:
            exhausted = True
            break
    if sol is None and pats:
        sol = solve_rmp_lp(pi, pats, pp)
    return pats, sol, exhausted
