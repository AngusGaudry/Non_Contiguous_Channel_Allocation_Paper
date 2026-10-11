import hashlib
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

MAX_SUBCHANNELS = 200
TIER_WEIGHTS = {1: 16, 2: 8, 3: 4, 4: 2, 5: 1}
TIER_FRACTIONS = {1: 0.10, 2: 0.325, 3: 0.325, 4: 0.15, 5: 0.10}
HARDNESS = {'S': 'sparse', 'M': 'moderate', 'D': 'dense'}


def stable_seed(*parts):
    s = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(s).digest()[:4], "big")


def _tier_counts(n):
    n1 = int(round(TIER_FRACTIONS[1] * n))
    n4 = int(round(TIER_FRACTIONS[4] * n))
    n5 = int(round(TIER_FRACTIONS[5] * n))
    while n1 + n4 + n5 > n:
        for t in (4, 5, 1):
            if t == 4 and n4 > 0:
                n4 -= 1
            elif t == 5 and n5 > 0:
                n5 -= 1
            elif t == 1 and n1 > 0:
                n1 -= 1
            if n1 + n4 + n5 <= n:
                break
    rest = n - n1 - n4 - n5
    return {1: n1, 2: rest - rest // 2, 3: rest // 2, 4: n4, 5: n5}


def tier_weights(n, seed):
    counts = _tier_counts(n)
    order = np.random.default_rng(seed).permutation(n)
    w = np.ones(n, dtype=np.int64)
    start = 0
    for t in (1, 2, 3, 4, 5):
        end = start + counts[t]
        w[order[start:end]] = TIER_WEIGHTS[t]
        start = end
    return w


def weighted_value(carriers, pi):
    w = pi['weight']
    return int(sum(int(w[c]) for c in carriers))


def sanity_check(assignments, pi):
    sub = pi['sub_channels']
    thr = pi['threshold']
    P = pi['interference_matrix']
    beam = pi['allocated_beams']
    keys = list(assignments)
    for i in keys:
        if not assignments[i]:
            continue
        if len(assignments[i]) != sub[i]:
            return False
        for j in keys:
            if j != i and beam[j] == beam[i] and assignments[i] & assignments[j]:
                return False
        load = sum(P[i][j] * len(assignments[i] & assignments[j]) for j in keys if j != i)
        if load > thr[i]:
            return False
    return True


def _instance(P, threshold, sub_channels, beams, seed):
    beams = np.asarray(beams)
    users = {b: [] for b in range(int(beams.max()) + 1)}
    for c, b in enumerate(beams):
        users[int(b)].append(c)
    n = len(sub_channels)
    return {
        'interference_matrix': P,
        'threshold': threshold,
        'sub_channels': sub_channels,
        'allocated_users': users,
        'allocated_beams': beams,
        'max_subchannels': MAX_SUBCHANNELS,
        'n_carriers': n,
        'weight': tier_weights(n, seed),
    }


def load_synthetic(size, hardness, data_dir):
    d = Path(data_dir)
    df = pd.read_excel(d / f"{size}_carriers_{hardness}.xlsx")
    P = pd.read_excel(d / f"{size}_carriers_{hardness}_interference_matrix.xlsx")
    P = P.drop(columns=["Unnamed: 0"]).to_numpy(dtype=int)
    return _instance(P, df["max_interference"].values, df["Sub_Channels"].values,
                     df["allocated_beam"].values, stable_seed(size, hardness))


def load_realistic(size, data_dir):
    d = Path(data_dir)
    df = pd.read_excel(d / f"{size}_carriers.xlsx")
    P = pd.read_excel(d / f"{size}_carriers_interference_matrix.xlsx").to_numpy(dtype=int)[:, 1:]
    return _instance(P, df["max_interference"].values, df["Sub_Channels"].values,
                     df["allocated_beam"].values, stable_seed("base", size))


def _parse_hts(data_dir):
    link_path = os.path.join(data_dir, "link_ouput_trudy.xlsx")
    pij_path = os.path.join(data_dir, "Pij_output_trudy.xlsx")
    cache = os.path.join(data_dir, "hts_cache.pkl")
    key = (os.path.getmtime(link_path), os.path.getmtime(pij_path))
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            parsed = pickle.load(f)
        if parsed["key"] == key:
            return parsed
    df = pd.read_excel(link_path)
    n = len(df)
    p = np.zeros((n, n))
    beam = np.full(n, -1, dtype=np.int64)
    for data in pd.read_excel(pij_path, sheet_name=None).values():
        pairs = data["Carriers c1, c2"].str.extract(r"\((\d+),\s*(\d+)\)").astype(np.int64)
        beams = data["Beams a, b"].str.extract(r"\((\d+),\s*(\d+)\)").astype(np.int64)
        c1 = pairs[0].to_numpy() - 1
        c2 = pairs[1].to_numpy() - 1
        p[c1, c2] = data["Pij_Watt"].to_numpy()
        beam[c1] = beams[0].to_numpy()
    np.fill_diagonal(p, 0.0)
    parsed = {
        "key": key,
        "p": p,
        "beam": beam,
        "sub_channels": df["sub_channels"].to_numpy().astype(np.int64),
        "threshold": df["max_interference_Watt"].to_numpy(dtype=float),
        "colour": df["beam_color"].to_numpy(dtype=np.int64),
    }
    with open(cache, "wb") as f:
        pickle.dump(parsed, f)
    return parsed


def load_hts(colour, data_dir):
    raw = _parse_hts(data_dir)
    idx = np.where(raw["colour"] == colour)[0]
    p = raw["p"][np.ix_(idx, idx)]
    scale = 1000.0 / p[p > 0].min()
    P = np.ceil(p * scale).astype(np.int64)
    threshold = np.floor(raw["threshold"][idx] * scale).astype(np.int64)
    _, beams = np.unique(raw["beam"][idx], return_inverse=True)
    return _instance(P, threshold, raw["sub_channels"][idx], beams, stable_seed("hts", colour))


def load_instance(name, data_dir='data', uniform_weights=False):
    a, b = name.split('-')
    if a == 'HTS':
        pi = load_hts(int(b), data_dir)
    elif b == 'R':
        pi = load_realistic(int(a), data_dir)
    else:
        pi = load_synthetic(int(a), HARDNESS[b], data_dir)
    if uniform_weights:
        pi['weight'] = np.ones(pi['n_carriers'], dtype=np.int64)
    return pi


INSTANCES = ([f"{n}-{h}" for n in (20, 40, 100, 500) for h in 'SMD']
             + [f"{n}-R" for n in (20, 40, 100, 500)]
             + [f"HTS-{k}" for k in range(1, 9)])
