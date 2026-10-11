import argparse
import csv
import os

from problem import INSTANCES
from run_instance import METHODS, run

FIELDS = ['instance', 'method', 'rep', 'uniform', 'objective', 'ub', 'ub_source', 'source',
          'n_active', 'wall']


def main(out, methods, instances, reps, budget, data_dir):
    done = set()
    if os.path.exists(out):
        with open(out, newline='') as f:
            done = {(r['instance'], r['method'], r['rep'], r['uniform']) for r in csv.DictReader(f)}
    for uniform in (True, False):
        for instance in instances:
            for rep in range(reps):
                for method in methods:
                    if (instance, method, str(rep), str(uniform)) in done:
                        continue
                    row = run(instance, method, budget, rep, uniform, data_dir)
                    print(row, flush=True)
                    new = not os.path.exists(out)
                    with open(out, 'a', newline='') as f:
                        w = csv.DictWriter(f, fieldnames=FIELDS)
                        if new:
                            w.writeheader()
                        w.writerow(row)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results.csv')
    ap.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS)
    ap.add_argument('--instances', nargs='+', default=INSTANCES)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--budget', type=float, default=300.0)
    ap.add_argument('--data', default='data')
    args = ap.parse_args()
    main(args.out, args.methods, args.instances, args.reps, args.budget, args.data)
