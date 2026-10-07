"""
Sioux Falls data generation for the Multi-Template Computational Graph (MTCG), Section 7.2.1 of the paper
"Can Contextual Archetypes Explain Daily Traffic Variation? A Multi-Template Computational Graph with
Attention-Based Fusion".

Authors:
    Xin (Bruce) Wu, Department of Civil and Environmental Engineering, Villanova University, PA, USA
    Feng Shao, School of Mathematics, China University of Mining and Technology, China

Contact: xwu03@villanova.edu (Villanova University), xinwu8592@gmail.com (personal)

MIT License
Copyright (c) 2026 Xin (Bruce) Wu, Feng Shao

Usage  : python "1-data_generation.py"
         Needs only data/link_attributes.csv and data/od_pair.csv next to this script (numpy, pandas, networkx)
         and writes the generated data into the same data/ folder. Fixed random seed 42: identical output on
         every run (about 2 seconds).

Overview (paper Section 7.2.1). We generate 500 synthetic daily samples m, 400 for training and 100 for
testing. Each sample is generated in three steps:
  Step 1. The sample draws, with equal probability, one of three day types c in {1, 2, 3}, i.e., three spatial
          distributions of the OD trips.
  Step 2. Its OD demand is generated as (paper Eq. (50))
              q^m_{w,t} = kappa_{c,w} * qbar_{t - omega^m} + xi^m_w,
          where kappa_{c,w} gives the spatial distribution of the trips, the shifted curve qbar_{t - omega^m}
          gives their temporal distribution, and xi^m_w ~ N(0, sigma^2), sigma ~ 29 veh/h, is a random deviation
          drawn once for each OD pair and sample and the same at all time steps of the sample.
  Step 3. With the demand q^m_{w,t} and the behavioral parameters of the sample (its route set and Logit
          parameter theta^m) given, the observed link flows are obtained by employing one forward pass of the
          three-layer CG (paper Section 3.2.1) at each time step, where the link travel times are calculated by
          the BPR function (with parameters 0.15 and 4) from the link flows generated in the previous time step.
The four components:
  1. Spatial distribution of OD trips: kappa_{c,w} = exp(0.15 chi_{c,w}) / ( (1/|W|) sum_w' exp(0.15 chi_{c,w'}) ),
     chi_{c,w} ~ N(0, 1)                                                   -> variables chi, share
  2. Temporal distribution of OD trips: all samples are derived from one designed mean demand per OD pair, qbar_t:
     250 veh/h at 7:00 AM (t = 1), a peak of 340 veh/h at 8:15 AM (t = 6) and 313 veh/h at 8:45 AM (t = 8).
     Each sample shifts this piecewise-linear curve by omega^m ~ U(-0.5, 0.5) time steps (up to 7.5 min earlier
     or later), with qbar_{t - omega^m} linearly interpolated between neighbouring steps.  -> QBAR, omega
  3. Distribution of behavioral parameters: (i) each sample draws theta^m ~ U(0.18, 0.22); (ii) each day type has
     its own route set of five paths per OD pair: the three free-flow shortest paths plus the two shortest new
     paths under that type's congestion. The congested link costs come from assigning the type's mean peak demand
     (8:15 AM) to the free-flow paths (Logit, theta = 0.2, method of successive averages) with the VDF; they only
     serve to build the route sets and are not given to the model.        -> theta, LP_type
  4. Congestion and VDF: BPR t_a(v_a) = t_a^0 [1 + 0.15 (v_a / c_a)^4], c_a = 2000 veh/h. Path choice at each
     time step uses the travel times of the previous step (free-flow times at t = 1). No measurement error is
     added to the link flows.                                              -> bpr, load
Implementation note: demand values are generated before a factor 0.3 (qbar = 832 + ..., sigma = 95), rounded,
multiplied by 0.3 and kept at least 0.3 * 50 veh/h; the values quoted above and in the paper are after the factor.
The model later uses only the aggregate OD demand of each sample over 7:00-9:00 (8 steps) and the link flows of
the 66 links with sensors; the per-step demand q^m_{w,t} is used only to evaluate the estimates.

Output (data/): demand.csv (6000 x 96 = 12 steps x 500 samples, step-major), link_flow.csv (6000 x 76),
day_info.csv (day type, time shift omega^m, theta^m of each sample), paths_template1.npz (five free-flow shortest
paths, link x path), paths_type.npz (route sets and OD shares of the three day types), checks.json (basic
statistics reported in Section 7.2.1: mean demand by step, CV, v/c, travel-time ratio, N_sigma).
"""
import os, json, time
import copy as cp
import numpy as np, pandas as pd, networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
SEED, SCALE, K, NSTEP, NDAY = 42, 0.3, 5, 12, 500
QBAR = 832.0 + np.array([0, 60, 120, 180, 240, 300, 270, 210, 150, 90, 30, -30], float)   # qbar_t before the factor 0.3 (12 steps from 7:00)
SPATIAL_SD, SHIFT, EPS_SD = 0.15, 0.5, 95.0      # 0.15 in kappa; omega^m ~ U(-0.5, 0.5); sigma of xi before the factor 0.3
N_FF, THETA0, THETA_RANGE = 3, 0.2, (0.18, 0.22)   # free-flow paths kept in a route set; theta for the congested costs; theta^m range
t0 = time.time()

link_attributes = pd.read_csv(os.path.join(DATA, 'link_attributes.csv'))
od_pair = pd.read_csv(os.path.join(DATA, 'od_pair.csv')).values
num_link, num_od = len(link_attributes), len(od_pair)
fft = link_attributes['free flow travel time'].values.astype(float)
cap = link_attributes['link capacity'].values.astype(float)

# ---- Steps 1-2: day types and OD demand q^m_{w,t} = kappa_{c,w} qbar_{t-omega^m} + xi^m_w (components 1, 2) ----
rng = np.random.default_rng(SEED)
c = rng.integers(0, 3, size=NDAY)                                          # Step 1: day type c of each sample (0, 1, 2 = types 1, 2, 3)
chi = rng.standard_normal((3, num_od))                                     # chi_{c,w} ~ N(0, 1), once per type and OD pair
share = np.exp(SPATIAL_SD * chi); share /= share.mean(1, keepdims=True)    # component 1: OD shares kappa_{c,w}
omega = rng.uniform(-SHIFT, SHIFT, NDAY)                                   # component 2: time shift omega^m (steps)
xi = rng.normal(0.0, EPS_SD, (NDAY, num_od))                               # deviation xi^m_w, same at all steps
theta = rng.uniform(*THETA_RANGE, NDAY)                                    # component 3 (i): theta^m
qbar = np.stack([np.interp(np.arange(NSTEP) - d, np.arange(NSTEP), QBAR) for d in omega], 1)      # (12, 500)
raw = qbar[:, :, None] * share[c][None] + xi[None]
n_low = int((raw < 50).sum())
q = SCALE * np.round(np.maximum(raw, 50.0))                                # (12, 500, 96) veh/h

# ---- network and k shortest paths (the functions of the MTCG notebook) ----
edge_list = link_attributes[['start', 'end']].values.tolist()
edges = pd.DataFrame({'sources': np.array(edge_list)[:, 0], 'targets': np.array(edge_list)[:, 1]})

def k_shortest_paths(G, source, target, k, weight='weights'):
    """Yen's k shortest paths (source: https://github.com/Mokerpoker/k_shortest_paths); returns paths as link numbers"""
    A = [nx.dijkstra_path(G, source, target, weight='weights')]
    A_len = [sum([G[A[0][l]][A[0][l + 1]]['weights'] for l in range(len(A[0]) - 1)])]
    B = []
    for i in range(1, k):
        for j in range(0, len(A[-1]) - 1):
            Gcopy = cp.deepcopy(G)
            spurnode = A[-1][j]
            rootpath = A[-1][:j + 1]
            for path in A:
                if rootpath == path[0:j + 1]:
                    if Gcopy.has_edge(path[j], path[j + 1]):
                        Gcopy.remove_edge(path[j], path[j + 1])
                    if Gcopy.has_edge(path[j + 1], path[j]):
                        Gcopy.remove_edge(path[j + 1], path[j])
            for n in rootpath:
                if n != spurnode:
                    Gcopy.remove_node(n)
            try:
                spurpath = nx.dijkstra_path(Gcopy, spurnode, target, weight='weights')
                totalpath = rootpath + spurpath[1:]
                if totalpath not in B:
                    B += [totalpath]
            except nx.NetworkXNoPath:
                continue
        if len(B) == 0:
            break
        lenB = [sum([G[path[l]][path[l + 1]]['weights'] for l in range(len(path) - 1)]) for path in B]
        B = [p for _, p in sorted(zip(lenB, B))]
        A.append(B[0])
        A_len.append(sorted(lenB)[0])
        B.remove(B[0])
    A_link = []
    for path in A:
        A_link.append([edge_list.index([path[j], path[j + 1]]) + 1 for j in range(len(path) - 1)])
    return A_link, A_len

def paths(weights):
    edges['weights'] = weights
    G = nx.from_pandas_edgelist(edges, source='sources', target='targets', edge_attr='weights', create_using=nx.DiGraph())
    return [[list(p) for p in k_shortest_paths(G, od_pair[i, 0], od_pair[i, 1], K, weight='weights')[0]] for i in range(num_od)]

def to_LP(P):
    LP = np.zeros([num_link, num_od * K])
    for i in range(num_od):
        for j, p in enumerate(P[i]):
            LP[np.array(p) - 1, i * K + j] = 1
    return LP

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

def bpr(v): return fft * (1 + 0.15 * (v / cap) ** 4)                    # component 4: BPR VDF, c_a = 2000 veh/h

def msa_times(qw, LP, th=THETA0, it_max=200):
    v = np.zeros(num_link)
    for it in range(1, it_max + 1):
        cost = (bpr(v) @ LP).reshape(num_od, K)
        e = np.exp(-th * (cost - cost.min(1, keepdims=True))); P = e / e.sum(1, keepdims=True)
        v = v + ((qw[:, None] * P).reshape(-1) @ LP.T - v) / it
    return bpr(v)

# ---- Component 3 (ii): route sets (template-1 set = five free-flow shortest paths; one route set per day type) ----
P_ff = paths(fft); LP1 = to_LP(P_ff)
LP_type = np.stack([to_LP(mixed(P_ff, paths(msa_times(SCALE * QBAR[5] * share[k], LP1)))) for k in range(3)])
same = np.array([[(LP_type[k][:, i*K:(i+1)*K] == LP1[:, i*K:(i+1)*K]).all() for i in range(num_od)] for k in range(3)])

# ---- Step 3 + component 4: observed link flows by one forward pass of the three-layer CG at each time step ----
def load(days):
    v = np.zeros([NSTEP, len(days), num_link]); tt = np.repeat(fft[None], len(days), 0)
    LPd = LP_type[c[days]]
    for s in range(NSTEP):
        cost = np.einsum('na,nap->np', tt, LPd).reshape(len(days), num_od, K)
        e = np.maximum(np.exp(-theta[days][:, None, None] * cost), 1e-10); P = e / e.sum(2, keepdims=True)
        v[s] = np.einsum('nap,np->na', LPd, (q[s][days][:, :, None] * P).reshape(len(days), -1))
        tt = bpr(v[s])
    return v
v = np.zeros([NSTEP, NDAY, num_link])
for a in range(0, NDAY, 50):
    v[:, a:a + 50] = load(np.arange(a, min(a + 50, NDAY)))

# ---- write ----
cols = [str(w) for w in range(num_od)]
pd.DataFrame(q.reshape(-1, num_od), columns=cols).to_csv(os.path.join(DATA, 'demand.csv'), index=False)
pd.DataFrame(v.reshape(-1, num_link), columns=[str(a) for a in range(num_link)]).to_csv(os.path.join(DATA, 'link_flow.csv'), index=False)
pd.DataFrame({'sample': np.arange(NDAY), 'type': c + 1, 'time shift (steps)': omega, 'theta': theta}).to_csv(os.path.join(DATA, 'day_info.csv'), index=False)
np.savez(os.path.join(DATA, 'paths_template1.npz'), LP=LP1)
np.savez(os.path.join(DATA, 'paths_type.npz'), LP=LP_type, share=share)

# ---- basic statistics (Section 7.2.1: Table of designed vs generated mean demand, v/c, travel-time ratio) ----
v8, q8 = v[:8], q[:8]
cv = q8.std(1) / q8.mean(1); tr = np.stack([bpr(v8[s]) for s in range(8)]) / fft
d = np.abs(np.diff(v8, axis=0)) / v8.mean((0, 1))
UNOBS = np.array([47, 41, 67, 21, 25, 7, 61, 13, 37, 4]) - 1          # links without sensors (paper Fig. 13)
x = v8[:, :400][:, :, np.setdiff1d(np.arange(num_link), UNOBS)]; x = x - x.mean(1, keepdims=True)
s2 = np.linalg.svd(x.reshape(-1, x.shape[2]), compute_uv=False) ** 2; r = s2[:-1] / s2[1:]
chk = {'samples per type: all / training / test': [np.bincount(c, minlength=3).tolist(), np.bincount(c[:400], minlength=3).tolist(), np.bincount(c[400:], minlength=3).tolist()],
       'demand cells raised to the floor (of 576,000)': n_low,
       'designed mean demand qbar (veh/h), 7:00-8:45': (SCALE * QBAR[:8]).round(1).tolist(),
       'generated mean demand (veh/h), 7:00-8:45': q8.mean((1, 2)).round(1).tolist(),
       'demand CV over samples: min / mean / median / max': [round(float(cv.min()), 3), round(float(cv.mean()), 3), round(float(np.median(cv)), 3), round(float(cv.max()), 3)],
       'v/c mean, by step': [round(float((v8 / cap).mean()), 3), (v8 / cap).mean((1, 2)).round(2).tolist()],
       'v/c max': round(float((v8 / cap).max()), 2),
       'travel time / free flow: mean, by step': [round(float(tr.mean()), 3), tr.mean((1, 2)).round(2).tolist()],
       'lowest flow / link mean (min over links)': round(float((v8.min((0, 1)) / v8.mean((0, 1))).min()), 2),
       'link-step changes > 30% of link mean': int((d > 0.3).sum()),
       'share of OD pairs whose type route set = free-flow set, by type': same.mean(1).round(3).tolist(),
       'stacked centred SVD (training samples, 66 sensor links): N_sigma': int(r.argmax()) + 1,
       '  sigma_k^2 / sigma_k+1^2, k = 1..6': np.round(r[:6], 2).tolist()}
json.dump(chk, open(os.path.join(DATA, 'checks.json'), 'w'), indent=1)
for k_, v_ in chk.items():
    print(f'  {k_}: {v_}')
print(f'done ({time.time() - t0:.0f} s)')
