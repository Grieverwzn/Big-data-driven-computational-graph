"""Step 2: path sets of the templates by user-equilibrium column generation.

Paper
-----
Section 7.3.1 (route sets of the three trip classes: K = 2 routes per OD pair with the largest equilibrium
flow, column generation with Path4GMNS) and Section 7.3.3 (path sets of the ten candidate templates T1-T10, Table 10).

What
----
For every template (T1, T2, T3) and for the 7 extra candidate templates of the template-selection test, a path set
is generated with path4gmns 0.9.9 (column generation, 40 iterations, no column update):
  - representative demand of the template = mean over the 16 slots of its OD demand, with weight 2.5 on its own
    event slots (T1: 7:00-7:45, T2: 8:45-9:30, T3: 7:00-7:30) and 1 elsewhere, times 4 (15-min demand -> hourly
    flow rate, because the link capacities are hourly);
  - BPR link costs (alpha 0.15, beta 4) with the hourly capacity;
  - OD pairs without a generated route get their shortest path.
The model (step 5), the Logit benchmark (step 3) and the selection (step 4) keep, per OD pair, the k = 2 paths with
the largest UE flow ('volume'); OD pairs with one path repeat it.
Candidates (contexts that are not in the data; built from the same components d1, e2, e3 of step 1):
  lvl080 / lvl120  d1 x 0.8 / x 1.2              stad2x      d1 + 2 e2 (stadium event twice as large)
  stad_early       d1 + e2 one hour earlier      poi_late    d1 + e3 two hours later
  other_event      d1 + a POI-size event (7,975 cars, 10/30/60 % at 7:00-7:30) at the two non-venue zones with the
                   largest morning inflow, spread over their OD pairs in proportion to d1
  both             d1 + e2 + e3 (stadium and POI events on the same day)
path4gmns needs the files in its working directory, so every run happens in its own folder data/pathgen/<name>/ in
a separate process; a finished folder (agent_new.csv present) is skipped, so the step can be restarted.

Usage
-----
    python 2-path_generation.py                   all 3 templates + 7 candidates (about 1-2 min each, ~20 min)
    python 2-path_generation.py --only T1,T2,T3   templates only
    python 2-path_generation.py --force           regenerate even if the output exists

Inputs
------
    data/network/link.csv, node.csv, OD_pair.csv; data/data_new_{1,2,3}/demand_6-10.csv (step 1)
Outputs
-------
    data/data_new_{1,2,3}/agent_new.csv            path sets of the templates (about 35 MB each)
    data/candidates/<name>/demand_6-10.csv, agent_new.csv
    data/pathgen/<name>/                           path4gmns working folders (log.txt, link_performance_rep.csv, ...)

Authors:
    Xin (Bruce) Wu, Department of Civil and Environmental Engineering, Villanova University, PA, USA
    Feng Shao, School of Mathematics, China University of Mining and Technology, China

Contact: xwu03@villanova.edu (Villanova University), xinwu8592@gmail.com (personal)

MIT License
Copyright (c) 2026 Xin (Bruce) Wu, Feng Shao
"""
import os
import sys
import shutil
import argparse
import subprocess
import time
import numpy as np
import pandas as pd
import mtcg_common as C

SCALE_K = 4.0            # 15-min demand x 4 = hourly flow rate, matching the hourly link capacity
COLUMN_GEN = 40          # column-generation iterations
COLUMN_UPDATE = 0        # no column update (the route shares are not used; only the path sets)
SEED_VALUE = 1e-4        # tiny demand for OD pairs without demand, so that they also get a path


def worker(folder, slots):
    """One path4gmns run in `folder` (same steps as the experiment's pathgen.py)."""
    import path4gmns as pg
    os.chdir(folder)
    print('path4gmns version:', getattr(pg, '__version__', 'unknown'), flush=True)
    link = pd.read_csv('link.csv'); node = pd.read_csv('node.csv'); od = pd.read_csv('OD_pair.csv')
    for c in ['from_node_id', 'to_node_id', 'link_id']:
        link[c] = pd.to_numeric(link[c], errors='coerce').fillna(0).astype(int)
    for c in ['node_id', 'zone_id']:
        node[c] = pd.to_numeric(node[c], errors='coerce').fillna(0).astype(int)
    for c in ['o_zone_id', 'd_zone_id']:
        od[c] = pd.to_numeric(od[c], errors='coerce').fillna(0).astype(int)
    link['VDF_theta1'] = 1.0
    need = link['VDF_fftt1'].isna() | (pd.to_numeric(link['VDF_fftt1'], errors='coerce') <= 0)
    if need.any():                                  # length [km] / free speed [km/h] x 60 -> minutes
        tt = pd.to_numeric(link['length'], errors='coerce') / pd.to_numeric(link['free_speed'], errors='coerce') * 60.0
        link.loc[need, 'VDF_fftt1'] = tt.clip(lower=0.1).fillna(1.0)
    link['VDF_fftt1'] = pd.to_numeric(link['VDF_fftt1'], errors='coerce')
    link['VDF_cap1'] = pd.to_numeric(link['capacity'], errors='coerce').fillna(0.0).astype(float)
    link.to_csv('link.csv', index=False); node.to_csv('node.csv', index=False); od.to_csv('OD_pair.csv', index=False)

    D_df = pd.read_csv('demand_6-10.csv', header=None).dropna(how='all', axis=0).dropna(how='all', axis=1)
    if D_df.shape[0] == len(od) + 1:                # drop the header row 0..15
        first = pd.to_numeric(D_df.iloc[0], errors='coerce')
        if np.allclose(first.to_numpy(), np.arange(D_df.shape[1], dtype=float)):
            D_df = D_df.iloc[1:].reset_index(drop=True)
    assert D_df.shape[0] == len(od)
    D = D_df.to_numpy(dtype=float)
    weights = np.ones(16); weights[[int(x) for x in slots.split(',')]] = 2.5
    weights = weights / weights.mean()
    vol = (D * weights).mean(axis=1) * SCALE_K       # representative hourly demand
    vol[(D <= 0.0).all(axis=1)] = SEED_VALUE
    dem = od.copy(); dem['volume'] = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0).astype(float)
    dem.to_csv('demand.csv', index=False)

    network = pg.read_network()
    pg.load_demand(network)
    pg.perform_column_generation(COLUMN_GEN, COLUMN_UPDATE, network)
    pg.output_columns(network)
    pg.output_link_performance(network)
    for src, dst in (('route_assignment.csv', 'agent_new.csv'), ('agent.csv', 'agent_new.csv'),
                     ('link_performance.csv', 'link_performance_rep.csv')):
        if os.path.exists(src):
            if os.path.exists(dst):
                os.remove(dst)
            os.rename(src, dst)

    # OD pairs without a route: add the shortest path (rows get path_id 0 and no volume)
    agent = pd.read_csv('agent_new.csv')
    has = agent[['o_zone_id', 'd_zone_id']].drop_duplicates()
    miss = od[['o_zone_id', 'd_zone_id']].merge(has, how='left', indicator=True)
    miss = miss[miss['_merge'] == 'left_only']
    if len(miss):
        z2n = node[['node_id', 'zone_id']].drop_duplicates('zone_id').set_index('zone_id')['node_id'].astype(int).to_dict()
        new = []
        for o, d in zip(miss.o_zone_id, miss.d_zone_id):
            seq = network.find_shortest_path(int(z2n[o]), int(z2n[d]), seq_type='link')
            if isinstance(seq, str) and seq.strip():
                new.append({'o_zone_id': o, 'd_zone_id': d, 'path_id': 0, 'link_sequence': seq})
        if new:
            new = pd.DataFrame(new)
            for c in agent.columns:
                if c not in new.columns:
                    new[c] = np.nan
            agent = pd.concat([agent, new[agent.columns]], ignore_index=True)
            agent.to_csv('agent_new.csv', index=False)
        print(f'added a shortest path for {len(new)} OD pairs without a route', flush=True)
    else:
        print('every OD pair has a route', flush=True)


def candidate_demand():
    """Demand (num_od x 16) and weighted slots of the 7 extra candidates (as Melbourne_greedy/make_candidates.py)."""
    dem = lambda s: pd.read_csv(os.path.join(C.template_dir(s), 'demand_6-10.csv')).values.astype(float)
    d1, d2, d3 = dem(1), dem(2), dem(3)
    e2, e3 = d2 - d1, d3 - d1
    od = pd.read_csv(os.path.join(C.NETWORK, 'OD_pair.csv'))

    def shift(e, k):
        out = np.zeros_like(e)
        if k > 0:
            out[:, k:] = e[:, :-k]
        else:
            out[:, :k] = e[:, -k:]
        return out
    venues = set(range(2341, 2349))
    inflow = pd.Series(d1[:, 4:7].sum(1)).groupby(od.d_zone_id.values).sum()
    inflow = inflow[[z not in venues for z in inflow.index]].sort_values(ascending=False)
    zones = list(inflow.index[:2])
    mask = od.d_zone_id.isin(zones).values
    share = np.where(mask, d1[:, 4:7].sum(1), 0.0); share /= share.sum()
    e_other = np.zeros_like(d1)
    for j, w in enumerate([0.10, 0.30, 0.60]):
        e_other[:, 4 + j] = 7975 * w * share
    return {'lvl080': (0.8 * d1, '6,7,8,9'), 'lvl120': (1.2 * d1, '6,7,8,9'),
            'stad2x': (d1 + 2 * e2, '11,12,13,14'), 'stad_early': (d1 + shift(e2, -4), '7,8,9,10'),
            'poi_late': (d1 + shift(e3, 8), '12,13,14'), 'other_event': (d1 + e_other, '4,5,6'),
            'both': (d1 + e2 + e3, '4,5,6,11,12,13,14')}


def run_one(name, demand_file, slots, dest, force):
    out = os.path.join(dest, 'agent_new.csv')
    if os.path.exists(out) and not force:
        C.log(f'{name}: {os.path.relpath(out, C.ROOT)} exists, skipped'); return
    work = os.path.join(C.DATA, 'pathgen', name)
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(work)
    for f in ('link.csv', 'OD_pair.csv'):
        shutil.copy(os.path.join(C.NETWORK, f), os.path.join(work, f))
    node = pd.read_csv(os.path.join(C.NETWORK, 'node.csv'))
    node[['x_coord', 'y_coord']] = node[['x_coord', 'y_coord']].fillna(0.0)     # empty coordinates -> 0 (geometry only)
    node.to_csv(os.path.join(work, 'node.csv'), index=False)
    shutil.copy(demand_file, os.path.join(work, 'demand_6-10.csv'))
    t0 = time.time()
    with open(os.path.join(work, 'log.txt'), 'w', encoding='utf-8') as lf:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), '--worker', work, slots], stdout=lf,
                           stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    if r.returncode != 0:
        sys.exit(f'{name}: path4gmns failed, see {os.path.join(work, "log.txt")}')
    os.makedirs(dest, exist_ok=True)
    shutil.move(os.path.join(work, 'agent_new.csv'), out)          # the working folder keeps the logs only
    C.log(f'{name}: paths done in {time.time() - t0:.0f} s -> {os.path.relpath(out, C.ROOT)}')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(sys.argv[2], sys.argv[3]); sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='', help='comma list of names (T1,T2,T3,lvl080,...); default all')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    only = set(x for x in a.only.split(',') if x)
    for s in (1, 2, 3):
        C.need(os.path.join(C.template_dir(s), 'demand_6-10.csv'), '1-data_templates.py')
    jobs = [(f'T{s}', os.path.join(C.template_dir(s), 'demand_6-10.csv'), C.PATH_SLOTS[s], C.template_dir(s)) for s in (1, 2, 3)]
    cols = pd.read_csv(os.path.join(C.template_dir(1), 'demand_6-10.csv'), nrows=0).columns
    for name, (D, slots) in candidate_demand().items():
        folder = os.path.join(C.CANDIDATES, name); os.makedirs(folder, exist_ok=True)
        f = os.path.join(folder, 'demand_6-10.csv')
        pd.DataFrame(D, columns=cols).to_csv(f, index=False)
        jobs.append((name, f, slots, folder))
    for name, f, slots, dest in jobs:
        if not only or name in only:
            run_one(name, f, slots, dest, a.force)
