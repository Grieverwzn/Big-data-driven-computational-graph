"""Step 4: number and choice of the templates.

Paper
-----
Section 7.3.3 (Template Generation and Selection) with the methods of Section 5.1 (SVD of the centred
observations, Eq. 45; N_sigma = argmax_k sigma_k^2 / sigma_{k+1}^2, Eq. 46; at least N_sigma + 1 templates, Eq. 47)
and Section 5.2 (theta-distance d_t^m of an observation, Eq. 48, and its mean D over days and steps, Eq. 49; swap
search). Gives the numbers of Fig. 21 and Table 10 (drawn / tabulated by steps 7 and 6).

What
----
Only the 80 training days are used (test days and withheld sensors are never used for choosing templates).
1. SVD of the observations. The sensor counts of the training days (80 days x 12 slots 7:00-9:45 x 448 sensors) are
   centred by the mean of each slot over the days (only the differences between days remain) and stacked into a
   960 x 448 matrix. The shares of the squared singular values show how many systematic directions the days have
   (two: stadium and POI events -> K + 1 = 3 templates). The same is done per slot (N_sigma,t = argmax_k
   sigma_k^2 / sigma_(k+1)^2).
2. theta-distance of a set of templates. Each day's demand (main version: the day's period OD demand, i.e. the MTCG
   input, split over the slots by the mean slot shares of the training days; second version: the day's per-step
   demand) is assigned by Logit to the path set of every template with ONE external reference theta for all
   templates and slots (0.5 x wage -> 0.377 per minute; sensitivity 0.25 x and 1 x wage), slot by slot from 7:00
   with the link times of the previous slot (free-flow at 7:00), BPR with the hourly flow rate. The distance of an
   observation (day, slot: 448 counts) is its relative distance to the convex hull of the templates' flows of the
   same day and slot (non-negative weights summing to one, solved by NNLS with a heavily weighted sum row).
3. Selection of 3 of 10 candidates (T1-T3 and the 7 decoys of step 2) by the mean theta-distance: greedy from T1,
   swaps from the greedy result and from two starts made only of decoys, and the exhaustive search over all 120 sets.
Same code and numbers as the experiment scripts theta_distance.py, theta_select.py and svd_per_slot.py.

Usage
-----
    python 4-template_selection.py                       all three reference thetas (about 15-30 min)
    python 4-template_selection.py --thetas 0.5          main reference theta only (about 5-10 min)
    python 4-template_selection.py --skip-select         only SVD and theta-distance of T1-T3

Inputs
------
    data/days100_logit_start7_cap4_k2/train/ (step 3), data/data_new_{1,2,3}/agent_new.csv,
    data/candidates/<name>/agent_new.csv (step 2), data/network/link.csv, OD_pair.csv
Outputs (results/template_selection/)
-------
    svd_stacked.csv, svd_per_slot.csv          shares and ratios of the squared singular values
    theta_distance_summary.csv                 mean / median / 95 % / shares within 5 % and 10 % / failure bound
    theta_distance_by_slot.csv                 mean distance by day type and slot (T1 alone, T1+T2+T3; theta 0.377,
                                               period demand; % with one decimal, as used by the paper figure)
    theta_distance_by_slot_full.csv            the same for both demand versions, full precision
    selection_single.csv, selection_search.csv, selection_exhaustive.csv, summary.txt

Authors:
    Xin (Bruce) Wu, Department of Civil and Environmental Engineering, Villanova University, PA, USA
    Feng Shao, School of Mathematics, China University of Mining and Technology, China

Contact: xwu03@villanova.edu (Villanova University), xinwu8592@gmail.com (personal)

MIT License
Copyright (c) 2026 Xin (Bruce) Wu, Feng Shao
"""
import os
import re
import time
import argparse
import itertools
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import nnls
from scipy.stats import beta as betadist
import mtcg_common as C

ap = argparse.ArgumentParser()
ap.add_argument('--thetas', default='0.5,0.25,1', help='reference theta as multiples of the wage / 60 (0.5 = main)')
ap.add_argument('--skip-select', action='store_true', help='skip the selection among the 10 candidates')
args = ap.parse_args()
OUT = os.path.join(C.RES, 'template_selection'); os.makedirs(OUT, exist_ok=True)
t_start = time.time()
C.need(os.path.join(C.DAYS, 'train', 'days.npz'), '3-data_generation.py')

K, SLOTS = C.K, list(range(C.W0, C.W0 + C.T))
link = pd.read_csv(os.path.join(C.NETWORK, 'link.csv'))
lid = {int(l): i for i, l in enumerate(link.link_id)}
fftt = link.VDF_fftt1.values.astype(float); cap = link.capacity.values.astype(float)
od = pd.read_csv(os.path.join(C.NETWORK, 'OD_pair.csv'))
odi = {(int(o), int(d)): i for i, (o, d) in enumerate(zip(od.o_zone_id, od.d_zone_id))}
W, A = len(od), len(link)
tr = np.load(os.path.join(C.DAYS, 'train', 'days.npz'))
sens = np.array([lid[int(s)] for s in tr['sensor_ids']])
OBS = tr['link_flow'][:, SLOTS, :]                                             # (80, 12, 448) observed counts
dtype = pd.read_csv(os.path.join(C.DAYS, 'train', 'days.csv')).type.values
THETA = None                                                                   # reference theta per slot (set below)


def paths(file):
    """Path-link matrix (num_od * K paths x links) of the top-K paths by UE flow, padded; owner = OD index of a path.
    Unlike the model, free-flow times of the connectors are kept here (as in the experiment's selection code)."""
    a = pd.read_csv(file, usecols=['o_zone_id', 'd_zone_id', 'path_id', 'volume', 'link_sequence'])
    a['volume'] = pd.to_numeric(a['volume'], errors='coerce').fillna(0.0)
    a = a.sort_values(['o_zone_id', 'd_zone_id', 'volume', 'path_id'], ascending=[True, True, False, True])
    a['path_id'] = a.groupby(['o_zone_id', 'd_zone_id']).cumcount()
    a = a[a.path_id < K]
    rows, cols, owner = [], [], []
    for (o, d), g in a.groupby(['o_zone_id', 'd_zone_id'], sort=False):
        seqs = list(g.link_sequence.astype(str))
        while len(seqs) < K:
            seqs.append(seqs[0])                                               # repeat the main path
        for s in seqs:
            s = s.split('link path:')[-1]
            p = len(owner); owner.append(odi[(int(o), int(d))])
            for x in re.split(r'[;\s]+', s.strip()):
                if x:
                    rows.append(p); cols.append(lid[int(float(x))])
    owner = np.array(owner)
    assert len(owner) == W * K and np.bincount(owner, minlength=W).min() == K
    P = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(owner), A))
    P.data[:] = 1.0
    return P, owner


def rep_flows(P, owner, D):
    """Template flows v_{s,t}^m(theta) of Eq. 48: (12, 448) Logit flows on the sensors, slot by slot from 7:00; D = demand (num_od x 16)."""
    order = np.argsort(owner, kind='stable'); P = P[order]; owner = owner[order]
    t_link = fftt.copy(); out = []
    for j, t in enumerate(SLOTS):
        tau = (P @ t_link).reshape(W, K)
        e = np.exp(-THETA[j] * (tau - tau.min(1, keepdims=True))); sh = e / e.sum(1, keepdims=True)
        f = (D[:, t][:, None] * sh).reshape(-1)
        v = P.T @ f
        out.append(v[sens]); t_link = fftt * (1 + 0.15 * (4 * v / cap) ** 4)
    return np.array(out)


TH = {'0.25 x wage': 0.25 * C.WAGE / 60, '0.5 x wage': 0.5 * C.WAGE / 60, '1 x wage': C.WAGE / 60}
want = {f'{float(x):g} x wage' for x in args.thetas.split(',')}
TH_RUN = {k: v for k, v in TH.items() if k in want}
dem = tr['demand'][:, SLOTS, :]                                                # (80, 12, W) per-step demand
period = dem.sum(1)                                                            # (80, W) MTCG input
tot = dem.sum((0, 1)); share = np.where(tot > 0, dem.sum(0) / np.maximum(tot, 1e-12), 1 / len(SLOTS))   # (12, W)
DEM = {'period': period[:, None, :] * share[None], 'per-step': dem}
CAND = {'T1': C.template_dir(1), 'T2': C.template_dir(2), 'T3': C.template_dir(3)}
for n in ('lvl080', 'lvl120', 'stad2x', 'stad_early', 'poi_late', 'other_event', 'both'):
    CAND[n] = os.path.join(C.CANDIDATES, n)
PATHS = {}


def get_paths(name):
    if name not in PATHS:
        PATHS[name] = paths(os.path.join(CAND[name], 'agent_new.csv'))
    return PATHS[name]


def flows(name, D, theta):
    """(80, 12, 448) sensor flows of template `name` for every training day."""
    global THETA
    THETA = np.full(len(SLOTS), theta)
    P, owner = get_paths(name); out = []
    for m in range(D.shape[0]):
        Dm = np.zeros((D.shape[2], 16)); Dm[:, SLOTS] = D[m].T
        out.append(rep_flows(P, owner, Dm))
    return np.array(out)


def dist(F, sel, big=1e4):
    """theta-distance d_t^m(S; theta) (Eq. 48) of every observation (day m, slot t) to the convex hull of the flows of the templates in sel."""
    d = np.zeros(OBS.shape[:2])
    for m in range(OBS.shape[0]):
        for j in range(len(SLOTS)):
            B = np.array([F[n][m, j] for n in sel]); M = np.vstack([B.T, big * np.ones(len(sel))])
            y = OBS[m, j]; lam, _ = nnls(M, np.append(y, big)); d[m, j] = np.linalg.norm(y - B.T @ lam) / np.linalg.norm(y)   # NNLS with a heavy sum row: lambda >= 0, sum lambda = 1
    return d


summary = ['# Template selection on the 80 training days (results/template_selection)', '']

# ---------------------------------------------------------------------------------------------------------------
# 1. SVD
# ---------------------------------------------------------------------------------------------------------------
def nsig(s, kmax=10):                                     # N_sigma = argmax_k sigma_k^2 / sigma_{k+1}^2 (Eq. 46)
    r = s[:kmax] ** 2 / s[1:kmax + 1] ** 2
    return int(np.argmax(r)) + 1, r


Xs = OBS - OBS.mean(0, keepdims=True)                                          # centred observations (Eq. 45)
s = np.linalg.svd(Xs.reshape(-1, OBS.shape[2]), compute_uv=False)
e = s ** 2 / np.sum(s ** 2)
n_st, r_st = nsig(s)
pd.DataFrame({'k': np.arange(1, 11), 'share_of_sigma2': e[:10], 'ratio_sigma2_k_to_k+1': r_st}).to_csv(
    os.path.join(OUT, 'svd_stacked.csv'), index=False)
rows = []
for j in range(len(SLOTS)):
    X = OBS[:, j, :] - OBS[:, j, :].mean(0)
    sj = np.linalg.svd(X, compute_uv=False); ej = sj ** 2 / np.sum(sj ** 2); nj, rj = nsig(sj)
    rows.append({'slot': j + 1, 'clock': C.CLOCK[j], **{f'share_{i + 1}': ej[i] for i in range(5)},
                 **{f'ratio_{i + 1}': rj[i] for i in range(5)}, 'N_sigma_t': nj})
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'svd_per_slot.csv'), index=False)
summary += ['## SVD (each slot centred by its mean over the days)',
            'stacked energy shares of the first 6 directions: ' + ', '.join(f'{x:.1%}' for x in e[:6]),
            f'stacked N_sigma = {n_st}; per slot N_sigma,t = {[r["N_sigma_t"] for r in rows]}', '']
C.log('SVD: shares ' + ', '.join(f'{x:.1%}' for x in e[:3]))

# ---------------------------------------------------------------------------------------------------------------
# 2. theta-distance of the scenario templates
# ---------------------------------------------------------------------------------------------------------------
SETS = [('T1',), ('T1', 'T2'), ('T1', 'T3'), ('T2', 'T3'), ('T1', 'T2', 'T3')]
rows, slot_rows = [], []
for dv in ('period', 'per-step'):
    for tl, th in TH_RUN.items():
        F = {n: flows(n, DEM[dv], th) for n in ('T1', 'T2', 'T3')}
        for sel in SETS:
            d = dist(F, sel); f = d.ravel(); kk = int((f > 0.10).sum())
            ub = 1.0 if kk >= f.size else betadist.ppf(0.95, kk + 1, f.size - kk)
            rows.append({'demand': dv, 'theta': th, 'theta_label': tl, 'templates': '+'.join(sel),
                         'mean_%': 100 * f.mean(), 'median_%': 100 * np.median(f), 'q95_%': 100 * np.quantile(f, .95),
                         'share_le_5%': 100 * (f <= .05).mean(), 'share_le_10%': 100 * (f <= .10).mean(),
                         'fail_prob_upper_bound_10%': 100 * ub})
            if tl == '0.5 x wage' and sel in (('T1',), ('T1', 'T2', 'T3')):
                for ty in ('regular', 'poi', 'stadium', 'both'):
                    slot_rows.append({'demand': dv, 'templates': '+'.join(sel), 'day_type': ty,
                                      **dict(zip(C.CLOCK, 100 * d[dtype == ty].mean(0)))})
        C.log(f'theta-distance {dv} demand, {tl} done ({time.time() - t_start:.0f} s)')
tab = pd.DataFrame(rows); tab.to_csv(os.path.join(OUT, 'theta_distance_summary.csv'), index=False)
full = pd.DataFrame(slot_rows); full.to_csv(os.path.join(OUT, 'theta_distance_by_slot_full.csv'), index=False)
if len(full):
    main = full[full.demand == 'period'].drop(columns='demand').copy()
    for c in C.CLOCK:
        main[c] = main[c].map(lambda x: f'{x:.1f}')
    main.to_csv(os.path.join(OUT, 'theta_distance_by_slot.csv'), index=False)
summary += ['## theta-distance (%) of the scenario templates', tab.round(2).to_string(index=False), '']

# ---------------------------------------------------------------------------------------------------------------
# 3. choose 3 of 10 candidates
# ---------------------------------------------------------------------------------------------------------------
if not args.skip_select:
    names = list(CAND)
    single_rows, search_rows, ex_rows = [], [], []
    for tl, theta in TH_RUN.items():
        F = {n: flows(n, DEM['period'], theta) for n in names}
        cache = {}

        def score(sel, big=1e4):
            key = tuple(sorted(sel))
            if key not in cache:
                tot_ = 0.0
                for m in range(OBS.shape[0]):
                    for j in range(len(SLOTS)):
                        B = np.array([F[n][m, j] for n in key]); M = np.vstack([B.T, big * np.ones(len(key))])
                        y = OBS[m, j]; lam, _ = nnls(M, np.append(y, big))
                        tot_ += np.linalg.norm(y - B.T @ lam) / np.linalg.norm(y)
                cache[key] = tot_ / OBS[:, :, 0].size               # D(S; theta), Eq. 49
            return cache[key]
        single = sorted((score([n]), n) for n in names)
        single_rows += [{'theta': theta, 'theta_label': tl, 'candidate': n, 'mean_distance_%': 100 * v} for v, n in single]
        sel, log = ['T1'], []
        for _ in range(2):
            v, n = min((score(sel + [n]), n) for n in names if n not in sel); sel.append(n); log.append(f'+{n} {v:.2%}')
        search_rows.append({'theta': theta, 'theta_label': tl, 'method': 'greedy from T1', 'start': 'T1',
                            'steps': ', '.join(log), 'result': '+'.join(sel), 'mean_distance_%': 100 * score(sel)})
        for start in (list(sel), ['lvl120', 'stad_early', 'poi_late'], ['lvl080', 'other_event', 'stad2x']):
            cur = list(start); steps = []
            while True:
                v, i, n = min((score(cur[:i] + [n] + cur[i + 1:]), i, n) for i in range(3) for n in names if n not in cur)
                if v >= score(cur) - 1e-9:
                    break
                steps.append(f'{cur[i]} -> {n} ({v:.2%})'); cur[i] = n
            search_rows.append({'theta': theta, 'theta_label': tl, 'method': 'swaps', 'start': '+'.join(start),
                                'steps': '; '.join(steps) or 'none', 'result': '+'.join(sorted(cur)),
                                'mean_distance_%': 100 * score(cur)})
        allsets = sorted((score(c), c) for c in itertools.combinations(names, 3))
        ex_rows += [{'theta': theta, 'theta_label': tl, 'rank': i + 1, 'templates': '+'.join(c), 'mean_distance_%': 100 * v}
                    for i, (v, c) in enumerate(allsets)]
        rt = [i for i, (v, c) in enumerate(allsets) if set(c) == {'T1', 'T2', 'T3'}][0] + 1
        summary += [f'## Selection, theta = {theta:.3f} per minute ({tl}, VOT {60 * theta:.1f} AUD/h)',
                    'single candidates: ' + ', '.join(f'{n} {v:.2%}' for v, n in single),
                    'greedy from T1: ' + ', '.join(log)]
        summary += [f'swaps from {r["start"]}: {r["steps"]} => {r["result"]} ({r["mean_distance_%"]:.2f}%)'
                    for r in search_rows if r['theta'] == theta and r['method'] == 'swaps']
        summary += ['exhaustive, best five: ' + '; '.join(f'{"+".join(c)} {v:.2%}' for v, c in allsets[:5]),
                    f'T1+T2+T3: rank {rt} of {len(allsets)} ({score(["T1", "T2", "T3"]):.2%}); worst set {allsets[-1][0]:.2%}', '']
        C.log(f'selection {tl} done ({time.time() - t_start:.0f} s)')
    pd.DataFrame(single_rows).to_csv(os.path.join(OUT, 'selection_single.csv'), index=False)
    pd.DataFrame(search_rows).to_csv(os.path.join(OUT, 'selection_search.csv'), index=False)
    pd.DataFrame(ex_rows).to_csv(os.path.join(OUT, 'selection_exhaustive.csv'), index=False)

open(os.path.join(OUT, 'summary.txt'), 'w', encoding='utf-8').write('\n'.join(summary) + '\n')
print('\n'.join(summary))
C.log(f'wrote {os.path.relpath(OUT, C.ROOT)} ({time.time() - t_start:.0f} s)')
