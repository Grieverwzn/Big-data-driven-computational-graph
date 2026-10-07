"""Shared settings and helpers of the Melbourne MTCG case study.

Paper
-----
Section 7.3 (Case 3: Melbourne); used by all numbered scripts.

What
----
Folder layout, time slots, the reading of the path sets (top-k paths by UE flow, padded to k), and small utilities
(memory wait, logging) used by the numbered scripts 1-7. Importing this module has no side effects except
defining constants.

Usage
-----
    import mtcg_common as C
    C.DATA, C.RES, C.template_dir(2), ...

Inputs / outputs
----------------
None by itself; see the numbered scripts.

Authors:
    Xin (Bruce) Wu, Department of Civil and Environmental Engineering, Villanova University, PA, USA
    Feng Shao, School of Mathematics, China University of Mining and Technology, China

Contact: xwu03@villanova.edu (Villanova University), xinwu8592@gmail.com (personal)

MIT License
Copyright (c) 2026 Xin (Bruce) Wu, Feng Shao
"""
import os
import sys
import time
import glob

# ---------------------------------------------------------------------------------------------------------------
# Folders (all relative to this file, so the package runs from any working directory)
# ---------------------------------------------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')
OPEN_DATA = os.path.join(DATA, 'open_data')          # open GMNS Melbourne data (OD matrices, nodes, counts)
NETWORK = os.path.join(DATA, 'network')              # renumbered network with centroid connectors (7,615 links)
CANDIDATES = os.path.join(DATA, 'candidates')        # 7 extra candidate templates (Section 7.3.3)
DAYS_NAME = 'days100_logit_start7_cap4_k2'           # name kept from the experiment log (Logit data from 7:00, hourly
DAYS = os.path.join(DATA, DAYS_NAME)                 #   capacity x4, k = 2 paths per OD pair)
RES = os.path.join(ROOT, 'results')
EVENT_FILE = os.path.join(DATA, 'event_extras.npz')


def template_dir(s):
    """Folder of template s (1 = regular day, 2 = stadium event, 3 = POI event)."""
    return os.path.join(DATA, f'data_new_{s}')


# ---------------------------------------------------------------------------------------------------------------
# Time: 16 slots of 15 min from 6:00 (slot 0) to 9:45 (slot 15); the model window is 7:00-9:45 (slots 4..15, T = 12)
# ---------------------------------------------------------------------------------------------------------------
N_SLOT = 16
W0, T = 4, 12
CLOCK16 = [f'{6 + (15 * t) // 60}:{(15 * t) % 60:02d}' for t in range(N_SLOT)]
CLOCK = [f'{7 + (15 * t) // 60}:{(15 * t) % 60:02d}' for t in range(T)]
K = 2                                                # paths per OD pair kept in every template
STADIUM_ZONES, POI_ZONES = range(2341, 2347), range(2347, 2349)   # venue zones after renumbering
WAGE = 45.19                                         # average hourly wage (ABS), AUD/h

# slots on which each template's representative demand gets weight 2.5 in the UE path generation (Section 7.3.1)
PATH_SLOTS = {1: '4,5,6,7', 2: '11,12,13,14', 3: '4,5,6'}


def mtcg_run_dir(S, root=None):
    """Results folder of an MTCG run with S templates. Accepts the package name (mtcg_S<S>) and the long names of
    the original experiment folders (..._S<S>_..._alt), so the tables and figures can be made from either."""
    root = root or RES
    p = os.path.join(root, f'mtcg_S{S}')
    if os.path.exists(os.path.join(p, 'run_info.json')):
        return p
    hits = sorted(glob.glob(os.path.join(root, f'*_S{S}_*_alt')))
    hits = [h for h in hits if os.path.exists(os.path.join(h, 'run_info.json'))]
    return hits[0] if hits else None


# ---------------------------------------------------------------------------------------------------------------
# Path sets. Each template keeps, for every OD pair, the k paths with the largest UE flow (path4gmns 'volume');
# OD pairs with fewer than k paths repeat their main path. Same code as the model notebook of the experiment.
# ---------------------------------------------------------------------------------------------------------------
def top_k_paths(file, k):
    import pandas as pd
    a = pd.read_csv(file)
    a['volume'] = pd.to_numeric(a['volume'], errors='coerce').fillna(0.0)
    a = a.sort_values(['o_zone_id', 'd_zone_id', 'volume', 'path_id'], ascending=[True, True, False, True])
    a['path_id'] = a.groupby(['o_zone_id', 'd_zone_id']).cumcount()
    return a.loc[a['path_id'] < k, ['o_zone_id', 'd_zone_id', 'path_id', 'link_sequence']]


def pad_to_k(p, k):
    import pandas as pd
    p = p.copy()
    c = p.groupby(['o_zone_id', 'd_zone_id'])['path_id'].transform('size')
    main = p[p['path_id'] == 0]
    need = (k - c[main.index]).clip(lower=0)
    extra = main.loc[main.index.repeat(need.values)]
    out = pd.concat([p, extra]).sort_values(['o_zone_id', 'd_zone_id', 'path_id'], kind='mergesort')
    out = out.reset_index(drop=True)
    assert (out.groupby(['o_zone_id', 'd_zone_id']).size() == k).all(), 'pad_to_k: not k paths per OD'
    return out


def load_model_network(k=K, templates=(1, 2, 3)):
    """Network and path-link matrices exactly as the model reads them (cells 3-9 of the experiment notebook).

    Returns a dict with
      od_pair (num_od x 2), num_od, num_link, free_flow_tt and link_capacity (torch float32; connectors get a
      free-flow time of 0), path_last (list of path tables, k rows per OD pair, OD order of OD_pair.csv),
      LP0 / LPT0 (lists of sparse CSR matrices, link x path and path x link, one per template) and spmm(A, x) = A @ x
      with a hand-written backward (gradient = A^T g), which is much faster than PyTorch's own CSR backward."""
    import re
    import numpy as np
    import pandas as pd
    import torch
    od_pair = pd.read_csv(os.path.join(template_dir(1), 'OD_pair.csv')).values
    num_od = len(od_pair)
    la = pd.read_csv(os.path.join(template_dir(1), 'link.csv'))
    la = la[['link_id', 'from_node_id', 'to_node_id', 'capacity', 'VDF_fftt1', 'is_connector']]
    la.loc[la['is_connector'] == 1, 'VDF_fftt1'] = 0       # centroid connectors carry no travel time in the model
    num_link = len(la)
    free_flow_tt = torch.from_numpy(la['VDF_fftt1'].values).to(torch.float32)
    link_capacity = torch.from_numpy(la['capacity'].values).to(torch.float32)

    def sparse_lp(rows, cols, n_path, n_link):
        idx = torch.tensor([cols, rows], dtype=torch.long)
        return torch.sparse_coo_tensor(idx, torch.ones(len(rows)), (n_link, n_path)).coalesce()

    path_last, LP0 = [], []
    for s in templates:
        p = top_k_paths(os.path.join(template_dir(s), 'agent_new.csv'), k).reset_index(drop=True)
        p = pad_to_k(p, k).sort_values(['o_zone_id', 'd_zone_id', 'path_id']).reset_index(drop=True)
        num_path = len(p)
        rows, cols = [], []
        for i in range(num_path):
            seq = str(p.loc[i, 'link_sequence'])
            m = re.search(r'link path:\s*(.*)$', seq, flags=re.IGNORECASE)   # rows added by find_shortest_path
            if m:
                seq = m.group(1)
            parts = [x.strip() for x in seq.split(';') if x.strip().isdigit()]
            u = np.unique(np.array(list(map(int, parts))) - 1)                  # a link counts once per path
            rows.extend([i] * len(u)); cols.extend(u.tolist())
        LP0.append(sparse_lp(rows, cols, num_path, num_link))
        path_last.append(p)
    LPT0 = [x.t().coalesce() for x in LP0]
    LP0 = [x.to_sparse_csr() for x in LP0]
    LPT0 = [x.to_sparse_csr() for x in LPT0]
    transpose = {}
    for a, b in zip(LP0, LPT0):
        transpose[id(a)] = b; transpose[id(b)] = a

    class _SpMM(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A, x):
            ctx.AT = transpose[id(A)]
            return A @ x

        @staticmethod
        def backward(ctx, g):
            return None, ctx.AT @ g

    return dict(od_pair=od_pair, num_od=num_od, num_link=num_link, free_flow_tt=free_flow_tt,
                link_capacity=link_capacity, path_last=path_last, LP0=LP0, LPT0=LPT0, spmm=_SpMM.apply, k=k)


# ---------------------------------------------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------------------------------------------
def log(msg, file=None):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    if file:
        try:
            with open(file, 'a', encoding='utf-8') as f:
                f.write(line + '\n')
        except OSError:
            pass


def wait_for_memory(min_gb, poll_s=60, max_h=3.0, also_file=None):
    """Wait until psutil reports more than min_gb GB available (and, optionally, until also_file exists).
    Returns True when the condition holds, False after max_h hours."""
    import psutil
    t0 = time.time()
    while True:
        avail = psutil.virtual_memory().available / 1e9
        ok_file = also_file is None or os.path.exists(also_file)
        if avail > min_gb and ok_file:
            return True
        if time.time() - t0 > max_h * 3600:
            return False
        log(f'waiting: available memory {avail:.1f} GB (need > {min_gb} GB)'
            + ('' if ok_file else f', waiting for {also_file}'))
        time.sleep(poll_s)


def need(path, step):
    """Stop with a clear message when an input of an earlier step is missing."""
    if not os.path.exists(path):
        sys.exit(f'missing {os.path.relpath(path, ROOT)} -- run {step} first')
