import time

import numpy as np

from construction import greedy_topup
from problem import sanity_check, weighted_value


def _refill(pi, kept, seed):
    out = greedy_topup(pi, sorted(kept), kept, rng_seed=seed)
    return {int(c): set(v) for c, v in out.items() if v}


def _accept(pi, cand, base_value):
    if not cand:
        return None
    if weighted_value(sorted(cand), pi) <= base_value or not sanity_check(cand, pi):
        return None
    return cand


def targeted_eviction(pi, assignments, budget, seed, eject_k, pool_size=12, attempts=4):
    P = np.asarray(pi['interference_matrix'], dtype=float)
    thr = np.asarray(pi['threshold'], dtype=float)
    rng = np.random.default_rng(seed)
    cur = {int(c): set(v) for c, v in assignments.items() if v}
    best = weighted_value(sorted(cur), pi)
    t0 = time.perf_counter()
    got = _accept(pi, _refill(pi, {c: set(v) for c, v in cur.items()}, seed), best)
    if got:
        cur, best = got, weighted_value(sorted(got), pi)
    while time.perf_counter() - t0 < budget:
        act = sorted(cur)
        if len(act) < 2:
            break
        load = {i: sum(P[i][j] * len(cur[i] & cur[j]) for j in act if j != i) for i in act}
        slack = {i: thr[i] - load[i] for i in act}
        eps = max(1.0, float(np.mean([abs(s) for s in slack.values()])) * 1e-6)
        score = {a: sum(P[j][a] * len(cur[a] & cur[j]) / max(slack[j], eps) for j in act if j != a)
                 for a in act}
        ranked = sorted(act, key=lambda a: -score[a])
        candidates = ranked[:max(eject_k, min(pool_size, len(ranked)))]
        improved = False
        for _ in range(attempts):
            if time.perf_counter() - t0 >= budget:
                break
            victims = set(rng.choice(candidates, size=min(eject_k, len(candidates)), replace=False).tolist())
            trial = {c: set(v) for c, v in cur.items() if c not in victims}
            if not trial:
                continue
            got = _accept(pi, _refill(pi, trial, seed), best)
            if got:
                cur, best = got, weighted_value(sorted(got), pi)
                improved = True
                break
        if not improved:
            break
    return cur
