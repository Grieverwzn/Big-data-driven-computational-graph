"""
Sioux Falls template generation for the Multi-Template Computational Graph (MTCG), paper "Can Contextual Archetypes
Explain Daily Traffic Variation? A Multi-Template Computational Graph with Attention-Based Fusion" (Section 7.2.2,
"Template generation"; the number of templates follows Section 5.1, Fig. 15).

Authors:
    Xin (Bruce) Wu, Department of Civil and Environmental Engineering, Villanova University, PA, USA
    Feng Shao, School of Mathematics, China University of Mining and Technology, China

Contact: xwu03@villanova.edu (Villanova University), xinwu8592@gmail.com (personal)

MIT License
Copyright (c) 2026 Xin (Bruce) Wu, Feng Shao

Usage  : python "2-template_generation.py"
         Needs data/ (from "1-data_generation.py") and sf_common.py next to this script (numpy, pandas, networkx,
         scikit-learn). Writes templates/lp_S1.npz ... templates/lp_S5.npz and templates/templates_info.json
         (about 10 seconds). Deterministic (k-means with random_state 42): identical output on every run.

A template is a route set: K = 5 paths per OD pair (link-path incidence matrix, 76 links x 480 paths).
  Template 1   : the 5 free-flow shortest paths of every OD pair.
  Templates 2..S: the 400 training samples are grouped by their aggregate OD demand (the model input, 96 values = sum
                 of the 8 steps 7:00-8:45) with k-means into S-1 groups. For each group, the group mean of the aggregate
                 demand divided by T = 8 (veh/h per step) is assigned to the free-flow path set (Logit, theta = 0.2,
                 BPR, method of successive averages). Under the resulting congested link times, the template keeps the
                 3 free-flow shortest paths and adds the 2 shortest paths that are not among them (if fewer than 2 new
                 paths exist, the 4th and 5th free-flow paths fill the set).
Only the training-sample aggregate demand and the network are used (no per-step demand, no test sample, and not the
true route sets of the day types).
S = 1 (lp_S1.npz) uses template 1 only, computed as in the original MTCG notebook, where the free-flow link times were
given to the path search in single precision (float32). Several paths have equal free-flow length; with float32 times
the ties are broken differently for 23 of the 96 OD pairs (9 of them with a different set of 5 paths) than with the
double-precision times used for template 1 of lp_S2..lp_S5 and data/paths_template1.npz. Both are valid sets of 5
free-flow shortest paths; lp_S1.npz is kept so that the S = 1 results of the paper are reproduced exactly.

Output: templates/lp_S<S>.npz with keys LP (S x 76 x 480, float32; LP[0] = template 1) and, for S >= 2, labels (k-means
group of each training sample); templates/templates_info.json (samples per group, share of OD pairs whose route set equals the
template-1 set, and, as a diagnostic only, the true day types in each group).
"""
import os, json, time
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sf_common import DATA, TEMPLATES, K, T, NSTEP, NDAY, NTRAIN, num_od, fft, paths, to_LP, msa_times

N_FF, THETA = 3, 0.2          # free-flow paths kept in every template; Logit parameter for the congested link times
S_MAX = 5
t0 = time.time()
os.makedirs(TEMPLATES, exist_ok=True)

# aggregate OD demand of the training samples (400 x 96): the model input
agg = pd.read_csv(os.path.join(DATA, 'demand.csv')).values.reshape(NSTEP, NDAY, num_od)[:T, :NTRAIN].sum(0)


def mixed(P_ff, P_cong):
    """the N_FF free-flow shortest paths + the congested shortest paths not among them (filled with the other free-flow paths)"""
    out = []
    for a, b in zip(P_ff, P_cong):
        sel = []
        for p in a[:N_FF] + [p for p in b if p not in a[:N_FF]] + a[N_FF:]:
            if p not in sel:
                sel.append(p)
        out.append(sel[:K])
    return out


P_ff = paths(fft)                       # template 1: the 5 free-flow shortest paths
LP1 = to_LP(P_ff)
np.savez(os.path.join(TEMPLATES, 'lp_S1.npz'), LP=to_LP(paths(fft.astype(np.float32)))[None])   # S = 1 (see above)
typ = pd.read_csv(os.path.join(DATA, 'day_info.csv'))['type'].values[:NTRAIN]   # diagnostic only
info = {}
for S in range(2, S_MAX + 1):
    km = KMeans(n_clusters=S - 1, n_init=10, random_state=42).fit(agg)
    LPs = [LP1] + [to_LP(mixed(P_ff, paths(msa_times(agg[km.labels_ == g].mean(0) / T, LP1, THETA)))) for g in range(S - 1)]
    np.savez(os.path.join(TEMPLATES, f'lp_S{S}.npz'), LP=np.stack(LPs), labels=km.labels_)
    info[f'S={S}'] = {
        'samples per group': np.bincount(km.labels_).tolist(),
        'share of OD pairs with the template-1 route set, templates 2..S':
            [round(float(np.mean([(L[:, i*K:(i+1)*K] == LP1[:, i*K:(i+1)*K]).all() for i in range(num_od)])), 3) for L in LPs[1:]],
        'true day types 1/2/3 per group (diagnostic, not used)':
            [np.bincount(typ[km.labels_ == g], minlength=4)[1:].tolist() for g in range(S - 1)]}
    print(f'S={S}: {info[f"S={S}"]}')
json.dump(info, open(os.path.join(TEMPLATES, 'templates_info.json'), 'w'), indent=1)
print(f'done ({time.time() - t0:.0f} s)')
