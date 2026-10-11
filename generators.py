import time

from construction import flip_plus, greedy_topup
from csa_lns import StallDetector, csa_lns
from direct_model import solve_direct
from eviction import targeted_eviction
from flip_cp import flip_cp
from patterns import column_generation, pattern_selection, patterns_from_assignment
from problem import sanity_check, weighted_value
from relax_repair import relax_and_repair


def warm_start(pi, incumbent, seed, lp_x):
    asg = {int(c): set(v) for c, v in incumbent['assignments'].items() if v}
    if not asg:
        return None
    keep = asg
    if lp_x is not None:
        sub = {c: v for c, v in asg.items() if float(lp_x[c]) > 0.99}
        if sub and len(sub) < len(asg):
            keep = sub
    comb = greedy_topup(pi, sorted(keep), keep, rng_seed=seed)
    comb = {int(c): set(v) for c, v in comb.items() if v}
    return sorted(comb), comb


def _csa_lns(pi, budget, seed, incumbent, lp_x, upper_bound, cold_start):
    warm = None
    if not cold_start and incumbent is not None and incumbent['assignments']:
        warm = warm_start(pi, incumbent, seed, lp_x)
    return csa_lns(pi, budget, seed, warm_start=warm,
                   stall=StallDetector(pi['n_carriers'], upper_bound=upper_bound),
                   cp_time_limit=min(10.0, budget))


def _eviction(pi, budget, seed, incumbent, *_):
    asg = {int(c): set(v) for c, v in incumbent['assignments'].items() if v}
    k = int(min(max(1, round(0.15 * max(1, len(asg)))), 20))
    out = targeted_eviction(pi, asg, budget, seed, k)
    return sorted(out), out


def _cg(pi, budget, seed, incumbent, *_):
    deadline = time.perf_counter() + budget
    pool = set(column_generation(pi, min(budget * 0.5, max(0.0, deadline - time.perf_counter()))))
    hint = None
    if incumbent is not None:
        ps, ms = patterns_from_assignment(pi, incumbent['assignments'], incumbent['active'])
        pool |= set(ps)
        hint = dict(zip(ps, ms))
    _, cand = pattern_selection(pi, list(pool), max(1.0, min(budget * 0.5, deadline - time.perf_counter())),
                                hint=hint, conditional=True)
    if cand is None:
        return [], {}
    return [c for c in cand if cand[c]], cand


def _relax_repair(pi, budget, seed, incumbent, *_):
    if incumbent is None:
        active, asg = flip_plus(pi)
    else:
        active, asg = incumbent['active'], incumbent['assignments']
    return relax_and_repair(pi, active, asg, budget, seed)


def _flip_cp(pi, budget, seed, *_):
    return flip_cp(pi, budget, seed)


def _flip_plus(pi, *_):
    return flip_plus(pi)


def _direct(pi, budget, *_):
    return solve_direct(pi, budget)


GENERATORS = {
    'csa_lns': _csa_lns,
    'eviction': _eviction,
    'cg': _cg,
    'relax_repair': _relax_repair,
    'flip_cp': _flip_cp,
    'flip_plus': _flip_plus,
    'direct': _direct,
}


def solve(name, pi, budget, seed=0, incumbent=None, lp_x=None, upper_bound=None, cold_start=False):
    active, assignments = GENERATORS[name](pi, budget, seed, incumbent, lp_x, upper_bound, cold_start)
    current = {int(c): set(v) for c, v in assignments.items() if v}
    if current:
        topped = {int(c): set(v) for c, v in greedy_topup(pi, sorted(current), current, rng_seed=seed).items() if v}
        if weighted_value(sorted(topped), pi) > weighted_value(sorted(current), pi):
            active, assignments = sorted(topped), topped
    feasible = sanity_check(assignments, pi)
    return dict(name=name, active=list(active), assignments=assignments, feasible=feasible,
                objective=weighted_value(active, pi) if feasible else None)
