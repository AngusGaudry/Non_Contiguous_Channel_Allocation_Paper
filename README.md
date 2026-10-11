# Non-contiguous channel allocation in multi-beam satellites

Code for *Non-Contiguous Channel Allocation in Multi-Beam Satellites: Formulations, Upper Bounds and Cooperative Matheuristics*.

## Setup

```
pip install -r requirements.txt
```

Place the instance workbooks in `data/`:

- `{20,40,100,500}_carriers.xlsx` and `{20,40,100,500}_carriers_interference_matrix.xlsx` (realistic, `#-R`)
- `{20,40,100,500}_carriers_{sparse,moderate,dense}.xlsx` and the matching `_interference_matrix.xlsx` files (synthetic, `#-S/M/D`)
- `link_ouput_trudy.xlsx` and `Pij_output_trudy.xlsx` (HTS, `HTS-1` to `HTS-8`)

## Usage

One run:

```
python run_instance.py --instance HTS-1 --method portfolio --budget 300
python run_instance.py --instance 40-D --method csa_lns --budget 300 --uniform
```

All experiments, written to a CSV that is resumed if it already exists:

```
python run_experiments.py --out results.csv --reps 3 --budget 300
```

`--uniform` sets every carrier weight to one. Without it, carriers carry the priority-tiered weights.

## Methods

| `--method` | Paper |
|---|---|
| `portfolio` | Cooperative portfolio (Sections 3 and 4) |
| `csa_lns` | CSA-LNS (Section 5.1) |
| `cg` | Heuristic column generation (Section 5.2) |
| `flip_plus` | FLIP+ (Section 5.3) |
| `flip_cp` | FLIP-CP (Section 5.4) |
| `eviction` | Targeted eviction (Section 5.5) |
| `relax_repair` | Relax-and-repair CP (Section 5.6) |
| `direct` | Direct model, Model 1 |
| `static`, `random` | Fixed schedules of the orchestration comparison |

Each generator run alone first runs the probe with a 30-second allowance and a 15-second assembly, then receives the full budget. The direct model starts cold.

## Modules

| File | Contents |
|---|---|
| `problem.py` | Instance loading, priority tiers, feasibility check |
| `construction.py` | FLIP, FLIP+, greedy top-up and the semi-greedy activation |
| `patterns.py` | Pattern model, its LP relaxation, pricing and column generation |
| `bounds.py` | Minimum-overlap maximum-tolerance, beam-aggregation and reduced-cost bounds |
| `assembly.py` | Safe and heuristic pool filters, pattern-pool recombination |
| `csa_lns.py` | CSA-LNS, its repair pool, kicks and stall detector |
| `flip_cp.py` | FLIP-CP |
| `eviction.py` | Targeted eviction |
| `relax_repair.py` | Relax-and-repair CP |
| `direct_model.py` | Direct model |
| `generators.py` | Common interface for the generators |
| `portfolio.py` | Shared repository, probe, gate and generator schedule |
| `baselines.py` | Fixed schedules |
| `run_instance.py`, `run_experiments.py` | Entry points |
