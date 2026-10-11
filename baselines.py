import random
import time

from assembly import assemble
from construction import flip_plus
from generators import solve
from patterns import lp_activation, patterns_from_assignment
from portfolio import ASSEMBLY_RESERVE, CALL_CAP, Blackboard, harvest
from problem import weighted_value

SCHEDULED = ('csa_lns', 'eviction', 'cg', 'flip_cp')


def fixed_schedule(pi, budget, order, seed=0):
    deadline = time.perf_counter() + budget
    bb = Blackboard()
    active, asg = flip_plus(pi)
    bb.post_incumbent(active, asg, weighted_value(active, pi), 'FLIP')
    bb.post_patterns(patterns_from_assignment(pi, asg, active)[0])
    rng = random.Random(seed)
    k = 0
    while True:
        remaining = deadline - time.perf_counter() - ASSEMBLY_RESERVE
        if remaining <= 1.0:
            break
        name = SCHEDULED[k % len(SCHEDULED)] if order == 'static' else rng.choices(SCHEDULED)[0]
        result = solve(name, pi, min(CALL_CAP, remaining), seed=k, incumbent=bb.incumbent,
                       lp_x=lp_activation(pi, bb.pool)[0], upper_bound=bb.ub)
        harvest(pi, bb, result)
        k += 1
    assemble(pi, bb, max(5.0, min(ASSEMBLY_RESERVE, deadline - time.perf_counter())), deadline)
    return bb
