"""Generate the link flows data/link_flow.csv from the OD demand data/demand.csv (Section 5.2.1).
- demand: data/demand.csv, hourly rates (veh/h), one value per 15-min step; 12 steps x 500 samples x 96 OD pairs
  (the first-submission demand x 0.3)
- link flows: recursive Logit loading from 7:00, the same structure as the model: at each step the demand of each OD pair
  is split over its 5 free-flow shortest paths (template-1 path set of the notebook) with theta = 0.2; step 1 uses the
  free-flow times, each later step the BPR times of the previous step's flows; no sensor error
- the graph, k-shortest-path, BPR and assignment functions are the cells of 2-MTCG.ipynb, executed as they are;
  the flows are computed for all samples at once with numpy and checked against the notebook's own assignment function
usage: python "1-Data generation.py"
"""
import os, json, math, time
import copy as cp
import numpy as np, pandas as pd, networkx as nx, torch

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, 'data')
THETA, K, NSTEP, NDAY = 0.2, 5, 12, 500
t_start = time.time()

# ---- notebook cells, executed as they are ----
nb = json.load(open(os.path.join(HERE, '2-MTCG.ipynb'), encoding='utf-8'))
cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
def cell(marker):
    hit = [c for c in cells if marker in c]
    assert len(hit) == 1, marker
    return hit[0]
link_attributes = pd.read_csv(os.path.join(DATA, 'link_attributes.csv'))
od_pair = pd.read_csv(os.path.join(DATA, 'od_pair.csv')).values
g = {'np': np, 'pd': pd, 'nx': nx, 'cp': cp, 'math': math, 'torch': torch, 'link_attributes': link_attributes, 'k': K}
exec(cell('edge_list = link_attributes[['), g)                          # graph
exec(cell('def k_shortest_paths('), g)                                  # Yen k shortest paths
exec(cell('def BPR('), g)                                               # BPR
exec(cell('def assignment('), g)                                        # Logit assignment
free_flow_tt = link_attributes['free flow travel time'].values
link_capacity = link_attributes['link capacity'].values
num_link, num_od = len(link_attributes), len(od_pair)

# ---- template-1 path set, as in the notebook for s = 0 (free-flow weights) ----
edges = g['edges']; edges['weights'] = free_flow_tt
G = nx.from_pandas_edgelist(edges, source='sources', target='targets', edge_attr='weights', create_using=nx.DiGraph())
LP = np.zeros([num_link, num_od * K])
for i in range(num_od):
    path, _ = g['k_shortest_paths'](G, od_pair[i, 0], od_pair[i, 1], K, weight='weights')
    assert len(path) == K, (i, len(path))
    for j, p in enumerate(path):
        LP[np.array(p) - 1, i * K + j] = 1

# ---- recursive Logit loading (numpy, all samples at once) ----
dem = pd.read_csv(os.path.join(DATA, 'demand.csv'))
q = np.reshape(dem.values, [NSTEP, NDAY, num_od])                       # (12, 500, 96), veh/h
def bpr(v): return free_flow_tt * (1 + 0.15 * (v / link_capacity) ** 4)
v_all = np.zeros([NSTEP, NDAY, num_link]); tt = np.repeat(free_flow_tt[None], NDAY, 0)
for t in range(NSTEP):
    c = (tt @ LP).reshape(NDAY, num_od, K)
    e = np.maximum(np.exp(-THETA * c), 1e-10); P = e / e.sum(2, keepdims=True)   # as the notebook (lower bound 1e-10)
    v_all[t] = (q[t][:, :, None] * P).reshape(NDAY, -1) @ LP.T
    tt = bpr(v_all[t])
pd.DataFrame(v_all.reshape(-1, num_link), columns=[str(a) for a in range(num_link)]).to_csv(os.path.join(DATA, 'link_flow.csv'), index=False)
print(f'wrote data/link_flow.csv ({time.time() - t_start:.1f} s)')

# ---- checks ----
g.update({'free_flow_tt': torch.from_numpy(free_flow_tt).float(), 'link_capacity': torch.from_numpy(link_capacity).float()})
LP_t = torch.from_numpy(LP).float(); qt = torch.from_numpy(q).float(); theta = torch.full((num_od,), THETA)
link_time = g['free_flow_tt'].repeat(NDAY, 1); path_time = torch.mm(link_time, LP_t); maxdiff = 0.0
for t in range(NSTEP):
    lf, link_time = g['assignment'](qt[t], theta, path_time, LP_t)
    path_time = torch.mm(link_time, LP_t)
    maxdiff = max(maxdiff, float(np.abs(lf.double().numpy() - v_all[t]).max() / v_all[t].max()))
vc = v_all[:8] / link_capacity; tr = np.stack([bpr(v_all[t]) for t in range(8)]) / free_flow_tt
print(f'  notebook assignment vs numpy: max relative difference {maxdiff:.1e}')
print(f'  7:00-8:45: mean v/c {vc.mean():.2f} (by step: {" ".join(f"{x:.2f}" for x in vc.mean((1, 2)))}), '
      f'travel time / free flow {tr.mean():.2f} (max {tr.max():.2f})')
