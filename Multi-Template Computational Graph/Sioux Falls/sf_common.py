"""
Shared helper functions for the Sioux Falls scripts of the Multi-Template Computational Graph (MTCG), paper
"Can Contextual Archetypes Explain Daily Traffic Variation? A Multi-Template Computational Graph with
Attention-Based Fusion" (Section 7.2).

Authors:
    Xin (Bruce) Wu, Department of Civil and Environmental Engineering, Villanova University, PA, USA
    Feng Shao, School of Mathematics, China University of Mining and Technology, China

Contact: xwu03@villanova.edu (Villanova University), xinwu8592@gmail.com (personal)

MIT License
Copyright (c) 2026 Xin (Bruce) Wu, Feng Shao

Usage  : imported by "2-template_generation.py", "3-mtcg.py", "4-baselines.py", "5-tables_and_figures.py" and
         "run_all.py"; not run on its own.

Contents: the folders of the package, the network (links, OD pairs, free-flow times, capacities), the links with and
without sensors (paper Fig. 13), the data loader, Yen's k shortest paths (the same function as in
"1-data_generation.py"), the link-path incidence matrix of a path set, the BPR function and a Logit assignment by the
method of successive averages (MSA).
"""
import os
import copy as cp
import numpy as np, pandas as pd, networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
TEMPLATES = os.path.join(HERE, 'templates')
RESULTS = os.path.join(HERE, 'results')

K = 5          # paths per OD pair in every template
T = 8          # time steps used by the model (7:00-8:45, 15 min each)
NSTEP, NDAY, NTRAIN = 12, 500, 400   # steps stored in data/, samples (days), training samples (the first 400)
SEEDS = (42, 43, 44)
UNOBSERVED = np.array([47, 41, 67, 21, 25, 7, 61, 13, 37, 4])   # links without sensors (1-based, paper Fig. 13)

link_attributes = pd.read_csv(os.path.join(DATA, 'link_attributes.csv'))
od_pair = pd.read_csv(os.path.join(DATA, 'od_pair.csv')).values
num_link, num_od = len(link_attributes), len(od_pair)
fft = link_attributes['free flow travel time'].values.astype(float)
cap = link_attributes['link capacity'].values.astype(float)
OBSERVED = np.setdiff1d(np.arange(1, num_link + 1), UNOBSERVED)  # 66 links with sensors (1-based)


def load_data():
    """OD demand (12, 500, 96) and link flows (12, 500, 76), veh/h, from data/ (written by 1-data_generation.py)"""
    q = pd.read_csv(os.path.join(DATA, 'demand.csv')).values.reshape(NSTEP, NDAY, num_od)
    v = pd.read_csv(os.path.join(DATA, 'link_flow.csv')).values.reshape(NSTEP, NDAY, num_link)
    return q, v


# ---- network and k shortest paths (as in 1-data_generation.py) ----
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
    """the K shortest paths of every OD pair under the given link costs (list over OD pairs of lists of link numbers)"""
    edges['weights'] = weights
    G = nx.from_pandas_edgelist(edges, source='sources', target='targets', edge_attr='weights', create_using=nx.DiGraph())
    return [[list(p) for p in k_shortest_paths(G, od_pair[i, 0], od_pair[i, 1], K, weight='weights')[0]] for i in range(num_od)]


def to_LP(P, dtype=np.float32):
    """link-path incidence matrix (num_link x num_od*K) of a path set; path j of OD pair i is column i*K + j"""
    LP = np.zeros([num_link, num_od * K], dtype=dtype)
    for i in range(num_od):
        for j, p in enumerate(P[i]):
            LP[np.array(p) - 1, i * K + j] = 1
    return LP


def bpr(v):
    """BPR link travel time t_a(v_a) = t_a^0 [1 + 0.15 (v_a / c_a)^4]"""
    return fft * (1 + 0.15 * (v / cap) ** 4)


def msa_times(qw, LP, theta=0.2, it_max=200):
    """Logit assignment (parameter theta) of the OD demand qw (veh/h, one value per OD pair) on the path set LP with BPR
    times, solved by the method of successive averages; returns the congested link travel times"""
    v = np.zeros(num_link)
    for it in range(1, it_max + 1):
        cost = (bpr(v) @ LP).reshape(num_od, K)
        e = np.exp(-theta * (cost - cost.min(1, keepdims=True))); P = e / e.sum(1, keepdims=True)
        v = v + ((qw[:, None] * P).reshape(-1) @ LP.T - v) / it
    return bpr(v)


def run_dir(S, seed, od_weighted=False, layout=None):
    """output folder of one MTCG training run: results/S<S>_seed<seed>[_odw]/, or for a sensor-coverage run (Section
    7.2.5) results/coverage/<layout>_s<seed>_S<S>/"""
    if layout:
        return os.path.join(RESULTS, 'coverage', f'{layout}_s{seed}_S{S}')
    return os.path.join(RESULTS, f'S{S}_seed{seed}' + ('_odw' if od_weighted else ''))
