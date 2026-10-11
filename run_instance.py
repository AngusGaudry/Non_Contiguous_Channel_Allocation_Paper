import argparse
import time

from assembly import assemble
from baselines import fixed_schedule
from generators import GENERATORS, solve
from patterns import lp_activation
from portfolio import Blackboard, probe, solve_portfolio
from problem import HARDNESS, load_instance, sanity_check, stable_seed, weighted_value

METHODS = ['portfolio', 'static', 'random'] + list(GENERATORS)


def _order_seed(name, rep):
    a, b = name.split('-')
    if a == 'HTS':
        spec = ('hts', int(b), None)
    elif b == 'R':
        spec = ('legacy', int(a), None)
    else:
        spec = ('synthetic', int(a), HARDNESS[b])
    return stable_seed('schedule_order', *spec, rep)


def run(instance, method, budget=300.0, seed=0, uniform_weights=False, data_dir='data'):
    pi = load_instance(instance, data_dir, uniform_weights)
    t0 = time.perf_counter()
    ub = ub_source = None
    if method == 'portfolio':
        bb = solve_portfolio(pi, budget)
        inc, ub, ub_source = bb.incumbent, bb.ub, bb.ub_source
    elif method in ('static', 'random'):
        inc = fixed_schedule(pi, budget, method, _order_seed(instance, seed)).incumbent
    else:
        incumbent = lp_x = None
        if method != 'direct':
            bb = Blackboard()
            probe(pi, bb, 30.0, 7.5)
            assemble(pi, bb, 15.0)
            incumbent, lp_x = bb.incumbent, lp_activation(pi, bb.pool)[0]
        inc = solve(method, pi, budget, seed, incumbent=incumbent, lp_x=lp_x)
    wall = time.perf_counter() - t0
    asg = {int(c): set(v) for c, v in inc['assignments'].items() if v}
    feasible = bool(asg) and sanity_check(asg, pi)
    return dict(instance=instance, method=method, rep=seed, uniform=uniform_weights,
                objective=weighted_value(asg, pi) if feasible else None, ub=ub, ub_source=ub_source,
                source=inc.get('source', method), n_active=len(asg), wall=round(wall, 1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance', default='HTS-1')
    ap.add_argument('--method', default='portfolio', choices=METHODS)
    ap.add_argument('--budget', type=float, default=300.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--uniform', action='store_true')
    ap.add_argument('--data', default='data')
    args = ap.parse_args()
    print(run(args.instance, args.method, args.budget, args.seed, args.uniform, args.data))
