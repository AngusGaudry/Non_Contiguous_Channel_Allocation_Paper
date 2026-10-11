from ortools.sat.python import cp_model


def solve_direct(pi, time_limit):
    n = pi['n_carriers']
    S = pi['max_subchannels']
    sub = pi['sub_channels']
    P = pi['interference_matrix']
    w = pi['weight']
    model = cp_model.CpModel()
    x = {(c, k): model.NewBoolVar(f'x_{c}_{k}') for c in range(n) for k in range(S)}
    y = {c: model.NewBoolVar(f'y_{c}') for c in range(n)}
    for c in range(n):
        model.Add(sum(x[c, k] for k in range(S)) == sub[c] * y[c])
    for members in pi['allocated_users'].values():
        for k in range(S):
            model.Add(sum(x[c, k] for c in members) <= 1)
    u = {}
    for i in range(n):
        for j in range(i + 1, n):
            if P[i][j] > 0 or P[j][i] > 0:
                for k in range(S):
                    u[i, j, k] = model.NewBoolVar(f'u_{i}_{j}_{k}')
                    model.Add(u[i, j, k] <= x[i, k])
                    model.Add(u[i, j, k] <= x[j, k])
                    model.Add(u[i, j, k] >= x[i, k] + x[j, k] - 1)
    for i in range(n):
        terms = [int(P[i][j]) * sum(u[min(i, j), max(i, j), k] for k in range(S))
                 for j in range(n) if j != i and int(P[i][j]) != 0]
        if terms:
            model.Add(sum(terms) <= int(pi['threshold'][i]))
    model.Maximize(sum(int(w[c]) * y[c] for c in range(n)))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [], {}
    active = [c for c in range(n) if solver.Value(y[c]) == 1]
    return active, {c: {k for k in range(S) if solver.Value(x[c, k]) == 1} for c in active}
