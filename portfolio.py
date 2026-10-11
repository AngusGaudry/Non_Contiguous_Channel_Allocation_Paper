import time

from assembly import assemble
from bounds import (aggregation_bound, cg_columns, floor, reduced_cost_bound,
                    scale_free_bound)
from construction import flip_plus
from generators import solve
from patterns import lp_activation, patterns_from_assignment
from problem import weighted_value

PROBE_SHARE = 0.30
DIAGNOSTIC_SHARE = 0.06
CG_EXTENSION_END = 0.60
ASSEMBLY_RESERVE = 30.0
MIN_CALL = 5.0
SCHEDULER_START = 10.0
CALL_CAP = 45.0
CAPS = {'cg': 20.0, 'flip_cp': 30.0}
H_STALL = 3
N_CUT = 100
SATURATION = 0.995
LP_EPS = 1e-6


class Blackboard:
    def __init__(self):
        self.ub = None
        self.ub_source = None
        self.incumbent = None
        self.pool = set()

    def post_ub(self, value, source):
        if value is not None and (self.ub is None or value < self.ub - 1e-9):
            self.ub, self.ub_source = float(value), source

    def post_incumbent(self, active, assignments, objective, source):
        if self.incumbent is None or int(objective) > self.incumbent['objective']:
            self.incumbent = dict(active=list(active), assignments={k: set(v) for k, v in assignments.items()},
                                  objective=int(objective), source=source)
            return True
        return False

    def post_patterns(self, patterns):
        self.pool.update(frozenset(p) for p in patterns)

    def gap(self):
        return self.ub - self.incumbent['objective']

    def incumbent_patterns(self, pi):
        return patterns_from_assignment(pi, self.incumbent['assignments'], self.incumbent['active'])

    def proven_optimal(self):
        return self.ub is not None and self.ub - self.incumbent['objective'] < 1.0


def harvest(pi, bb, result):
    if not result['feasible'] or result['objective'] is None:
        return False
    raised = bb.post_incumbent(result['active'], result['assignments'], result['objective'], result['name'])
    bb.post_patterns(patterns_from_assignment(pi, result['assignments'], result['active'])[0])
    return raised


def probe(pi, bb, budget, pool_seed_time, min_slice=2.0):
    deadline = time.perf_counter() + budget
    active, asg = flip_plus(pi)
    bb.post_incumbent(active, asg, weighted_value(active, pi), 'FLIP')
    bb.post_patterns(patterns_from_assignment(pi, asg, active)[0])
    bb.post_ub(floor(scale_free_bound(pi)), 'scale_free')
    bound_deadline = deadline - max(0.0, pool_seed_time)
    converged, pairs = False, None
    try:
        val, converged, pairs = aggregation_bound(pi, bound_deadline - min_slice)
        bb.post_ub(floor(val), 'aggregation_lp')
    except Exception:
        pass
    if converged and pairs and bound_deadline - time.perf_counter() > min_slice:
        try:
            val, _, _ = aggregation_bound(pi, bound_deadline - min_slice * 0.5, pairs=pairs, integral=True)
            bb.post_ub(floor(val), 'aggregation_milp')
        except Exception:
            pass
    exhausted = False
    cg_ran = False
    left = bound_deadline - time.perf_counter()
    if left >= min_slice:
        try:
            pats, sol, exhausted = cg_columns(pi, time.perf_counter() + left * 0.55)
            cg_ran = True
            bb.post_patterns(pats)
            if sol is not None:
                val, _ = reduced_cost_bound(pi, sol, bound_deadline, bb.ub)
                bb.post_ub(floor(val), 'reduced_cost')
        except Exception:
            pass
    if not cg_ran and pool_seed_time > 0:
        try:
            pats, _, exhausted = cg_columns(pi, deadline)
            bb.post_patterns(pats)
        except Exception:
            pass
    return len(active) / pi['n_carriers'] >= SATURATION, exhausted


def _small_schedule():
    for r in range(24):
        yield from (('csa_lns', r), ('cg', r), ('eviction', r))


def _large_schedule(last):
    yield 'csa_lns', 0
    first_raised = last['raised']
    yield 'eviction', 0
    if not first_raised:
        yield 'flip_cp', 0
    plateaus = 0
    for r in range(1, 24):
        yield 'csa_lns', r
        if not last['raised']:
            plateaus += 1
            yield ('cg' if plateaus % 2 == 0 else 'eviction'), plateaus


def run_schedule(pi, bb, budget, saturated):
    deadline = time.perf_counter() + budget
    if bb.proven_optimal():
        return
    if not saturated:
        _generate(pi, bb, deadline)
    if not bb.proven_optimal():
        assemble(pi, bb, max(5.0, deadline - time.perf_counter()), deadline)


def _generate(pi, bb, deadline):
    last = {'raised': None}
    schedule = _large_schedule(last) if pi['n_carriers'] > N_CUT else _small_schedule()
    lp_value = lp_activation(pi, bb.pool)[1]
    flat = 0
    cold = False
    excess = {}
    for name, seed in schedule:
        remaining = deadline - time.perf_counter() - ASSEMBLY_RESERVE
        if remaining <= 1.0 or bb.proven_optimal():
            return
        margin = excess[name] if name in excess else max(excess.values(), default=0.0)
        call_budget = min(CAPS.get(name, CALL_CAP), remaining - margin)
        if call_budget < MIN_CALL:
            return
        t_launch = time.perf_counter()
        result = solve(name, pi, call_budget, seed, incumbent=bb.incumbent,
                       lp_x=lp_activation(pi, bb.pool)[0], upper_bound=bb.ub, cold_start=cold)
        raised = harvest(pi, bb, result)
        last['raised'] = raised
        previous, lp_value = lp_value, lp_activation(pi, bb.pool)[1]
        excess[name] = max(excess.get(name, 0.0), time.perf_counter() - t_launch - call_budget, 0.0)
        rose = lp_value is not None and previous is not None and lp_value > previous + LP_EPS
        flat = 0 if (rose or raised) else flat + 1
        if bb.proven_optimal():
            return
        if flat < H_STALL:
            continue
        before = bb.incumbent['objective']
        above_reserve = deadline - time.perf_counter() - ASSEMBLY_RESERVE
        if above_reserve < MIN_CALL:
            return
        assemble(pi, bb, min(CALL_CAP, above_reserve), deadline - ASSEMBLY_RESERVE)
        if bb.incumbent['objective'] > before:
            flat = 0
            lp_value = lp_activation(pi, bb.pool)[1]
            if bb.proven_optimal():
                return
        elif not cold:
            cold = True
            flat = 0
            lp_value = lp_activation(pi, bb.pool)[1]
        else:
            return


def solve_portfolio(pi, budget):
    t0 = time.perf_counter()
    bb = Blackboard()
    probe_budget = max(5.0, PROBE_SHARE * budget)
    saturated, exhausted = probe(pi, bb, probe_budget, max(3.0, 0.25 * probe_budget))
    before = bb.incumbent['objective']
    assemble(pi, bb, max(3.0, DIAGNOSTIC_SHARE * budget))
    lift = bb.incumbent['objective'] - before
    if bb.gap() < 1.0:
        return bb
    if lift > 0 and not exhausted:
        extra = CG_EXTENSION_END * budget - (time.perf_counter() - t0)
        if extra > 2.0:
            pats, _, _ = cg_columns(pi, time.perf_counter() + extra,
                                    init_patterns=[tuple(sorted(p)) for p in bb.pool])
            bb.post_patterns([p for p in pats if len(p) > 1])
    remaining = budget - (time.perf_counter() - t0)
    if remaining > SCHEDULER_START:
        run_schedule(pi, bb, remaining, saturated)
    elif lift > 0 and remaining > MIN_CALL:
        assemble(pi, bb, remaining)
    return bb
