"""Step 3: the 100 semi-synthetic days of the Melbourne case: demand, Logit benchmark flows and sensor counts.

Paper
-----
Section 7.3.1 (Data Generation): synthesized OD trips (Eq. 51), synthesized VOT and route choices (Eq. 52,
theta_{w,t} = 0.377 x peak x trip x OD factor), link counts from sensors (Table 9 is made from these days by step 6).
Code -> paper (Eq. 51): q_m = q_t^m, d1 = q_t^0, eps_m = epsilon_t^m, b2_m a2_m = z_1^m (stadium event size, 0 if
no event), b3_m a3_m = z_2^m (POI), e2 / e3 = a_{j,t} Delta q_j; theta_od = theta_{w,t}, eta = OD factor (Eq. 52).

What
----
1. Days and demand (80 training and 20 test days; seed 2026). Day m has the OD demand
       q_m = d1 (1 + eps_m) + b2_m a2_m e2 + b3_m a3_m e3
   d1 = Template 1 (real day), e2 / e3 = stadium / POI event trips (step 1); b2, b3 = whether the event takes place
   (day types regular / POI / stadium / both: 40/20/12/8 training and 10/5/3/2 test days); a2, a3 ~ U(0.5, 1) =
   event size; eps ~ N(0, 5 %) per OD pair and slot, clipped to +-20 %.
2. Link counts from a Logit route-choice model with KNOWN parameters (the benchmark, so that the estimated VOT can be
   checked). Every OD pair belongs to one class by its destination: stadium trips (zones 2341-2346) use the k = 2
   paths of Template 2, POI trips (2347-2348) those of Template 3, all other trips those of Template 1.
       theta_{w,t} = theta0 * peak(t) * class(w) * eta_w,   theta0 = 0.5 x 45.19 / 60 = 0.377 per minute
       peak = 1.2 at 7:30-8:15, class = 1.3 (stadium) / 1.1 (POI) / 1 (other), eta_w = exp(N(0, 0.15^2)) (seed 2027)
   Assignment slot by slot from 7:00 on an empty network (free-flow times at 7:00): route times from the link times of
   the previous slot, Logit shares exp(-theta tau_r) / sum exp(-theta tau_r'), link flows = sum of path flows,
   link times by BPR t0 (1 + 0.15 (4 v / C)^4) (4 v = hourly flow rate). This is the computation of the model itself.
3. Sensor counts = noise-free flows on the 448 sensor links x (1 + nu), nu ~ N(0, 5 %) clipped to +-20 % (seed 2028).
Same code and random streams as the experiment (build_days.py 100 and build_days_logit.py --start 4 --cap 4); the
event counts of the earlier UE-type data are not needed and not computed (their random draws are kept in sequence).

Usage
-----
    python 3-data_generation.py            (about 1 min, about 3 GB memory)

Inputs
------
    data/data_new_{1,2,3}/ (steps 1-2), data/event_extras.npz (step 1)
Outputs
-------
    data/days100_logit_start7_cap4_k2/train|test/days.npz   demand (days x 16 x 30040), link_flow (days x 16 x 448,
                 with sensor error), link_flow_all (days x 16 x 7615, noise-free), theta_expected (days x 16,
                 demand-weighted mean theta), theta_od (16 x 30040), eta, sensor_ids, od_o, od_d
    data/days100_logit_start7_cap4_k2/train|test/days.csv   day, type, a2, a3, split, source_day
    data/days100_logit_start7_cap4_k2/summary.txt

Authors:
    Xin (Bruce) Wu, Department of Civil and Environmental Engineering, Villanova University, PA, USA
    Feng Shao, School of Mathematics, China University of Mining and Technology, China

Contact: xwu03@villanova.edu (Villanova University), xinwu8592@gmail.com (personal)

MIT License
Copyright (c) 2026 Xin (Bruce) Wu, Feng Shao
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import mtcg_common as C

# days and demand (build_days.py)
COMPOSITION = ((40, 20, 12, 8), (10, 5, 3, 2))      # (regular, POI only, stadium only, both) for (train, test)
SEED, OD_SD, SENSOR_SD_UE, SIZE_LOW = 2026, 0.05, 0.05, 0.5
# Logit benchmark (build_days_logit.py)
SEED_LOGIT = 2027
THETA0 = 0.5 * C.WAGE / 60                          # half of the average hourly wage, per minute
PEAK_SLOTS = (6, 7, 8, 9)                           # 7:30, 7:45, 8:00, 8:15
PEAK_MULT, STADIUM_MULT, POI_MULT = 1.2, 1.3, 1.1
ETA_SD, SENSOR_SD = 0.15, 0.05
START_SLOT = 4                                      # recursion starts at 7:00 with free-flow times (as the model)
FLOW_TO_HOURLY = 4                                  # v / C with the hourly flow rate 4 v

t_start = time.time()
for s in (1, 2, 3):
    C.need(os.path.join(C.template_dir(s), 'agent_new.csv'), '2-path_generation.py')
C.need(C.EVENT_FILE, '1-data_templates.py')
torch.set_num_threads(8)

# ---------------------------------------------------------------------------------------------------------------
# 1. days and demand
# ---------------------------------------------------------------------------------------------------------------
od = pd.read_csv(os.path.join(C.template_dir(1), 'OD_pair.csv'))[['o_zone_id', 'd_zone_id']]
flow = pd.read_csv(os.path.join(C.template_dir(1), 'link_flow.csv'))      # real counts: only the sensor link ids
sensor_ids = flow.columns.values.astype(float).astype(int)
d1 = pd.read_csv(os.path.join(C.template_dir(1), 'demand_6-10.csv')).values.astype(float)   # (num_od, 16)
ex = np.load(C.EVENT_FILE); e2, e3 = ex['e2'], ex['e3']

rng = np.random.default_rng(SEED)
types = ['regular', 'poi', 'stadium', 'both']
rows = []
for split, counts in zip(('train', 'test'), COMPOSITION):
    for typ, cnt in zip(types, counts):
        for _ in range(cnt):
            a2 = rng.uniform(SIZE_LOW, 1) if typ in ('stadium', 'both') else 0.0
            a3 = rng.uniform(SIZE_LOW, 1) if typ in ('poi', 'both') else 0.0
            rows.append({'type': typ, 'a2': a2, 'a3': a3, 'split': split})
days = pd.DataFrame(rows)
days = pd.concat([days[days.split == 'train'].sample(frac=1, random_state=SEED),        # shuffle the training days
                  days[days.split == 'test']]).reset_index(drop=True)
days.insert(0, 'day', np.arange(len(days)))
n, (num_od, n_slot) = len(days), d1.shape
demand = np.zeros((n, n_slot, num_od), np.float32)
for m, r in days.iterrows():
    eps = np.clip(rng.normal(0, OD_SD, d1.shape), -0.2, 0.2)
    demand[m] = (d1 + eps * d1 + r.a2 * e2 + r.a3 * e3).T   # Eq. 51 (d1 = q^0, a2 / a3 = z_1 / z_2)
    rng.normal(0, SENSOR_SD_UE, (n_slot, len(sensor_ids)))   # sensor error of the earlier UE-type counts (not used;
                                                             # drawn to keep the random stream of the experiment)
C.log(f'days and demand: {n} days, {num_od} OD pairs ({time.time() - t_start:.0f} s)')

# ---------------------------------------------------------------------------------------------------------------
# 2. Logit benchmark assignment with the model's own path sets and link data
# ---------------------------------------------------------------------------------------------------------------
net = C.load_model_network()
LP, LPT, spmm = net['LP0'], net['LPT0'], net['spmm']
free_flow_tt, link_capacity, k, num_link = net['free_flow_tt'], net['link_capacity'], net['k'], net['num_link']
assert net['num_od'] == num_od
for s, pl in enumerate(net['path_last']):
    o = pl['o_zone_id'].values.reshape(num_od, k)[:, 0]; d = pl['d_zone_id'].values.reshape(num_od, k)[:, 0]
    assert (o == od.o_zone_id.values).all() and (d == od.d_zone_id.values).all(), f'path order of template {s + 1}'
C.log(f'route sets loaded: {num_od} OD pairs x k={k}, {num_link} links ({time.time() - t_start:.0f} s)')


def bpr(v):
    return free_flow_tt * (1 + 0.15 * (FLOW_TO_HOURLY * v / link_capacity) ** 4)


dz = od.d_zone_id.values
cls = np.where(np.isin(dz, list(C.STADIUM_ZONES)), 1, np.where(np.isin(dz, list(C.POI_ZONES)), 2, 0))  # 0 base, 1 stadium, 2 POI
tmpl_of_class = {0: 0, 1: 1, 2: 2}                    # base -> T1, stadium -> T2, POI -> T3
rng_l = np.random.default_rng(SEED_LOGIT)
eta = np.exp(rng_l.normal(0, ETA_SD, num_od))                     # OD factor exp(N(0, 0.15^2)) of Eq. 52
peak = np.ones(16); peak[list(PEAK_SLOTS)] = PEAK_MULT
cmult = np.select([cls == 1, cls == 2], [STADIUM_MULT, POI_MULT], 1.0)
theta_od = THETA0 * peak[:, None] * (cmult * eta)[None, :]        # theta_{w,t} of Eq. 52, (16, num_od)
assert np.allclose(demand[days.type.isin(['regular', 'stadium']).values][:, :, cls == 2], 0)
assert np.allclose(demand[days.type.isin(['regular', 'poi']).values][:, :, cls == 1], 0)


def assign(theta_tw):
    """Noise-free link flows (days, 16, num_link) of all days; recursion over the slots from START_SLOT."""
    q = torch.from_numpy(demand.astype(np.float32))
    B = q.shape[0]
    out = torch.zeros(B, 16, num_link)
    link_time = free_flow_tt.repeat(B, 1)
    for t in range(START_SLOT, 16):
        th = torch.from_numpy(theta_tw[t].astype(np.float32))
        v = torch.zeros(B, num_link)
        for c in (0, 1, 2):
            s = tmpl_of_class[c]
            qc = q[:, t, :] * torch.from_numpy(cls == c)          # demand of this class only
            if float(qc.sum()) == 0:
                continue
            pt = spmm(LPT[s], link_time.T).T.reshape(B, num_od, k)   # route times on the paths of template s
            p = torch.softmax(-th[None, :, None] * pt, dim=2)        # Logit shares
            pf = (qc[:, :, None] * p).reshape(B, -1)
            v = v + spmm(LP[s], pf.T).T
        out[:, t] = v
        link_time = bpr(v)                                           # link times for the next slot
    return out.numpy()


with torch.no_grad():
    v_all = assign(theta_od)
sensor_idx = sensor_ids.astype(int) - 1
nu = np.clip(np.random.default_rng(SEED_LOGIT + 1).normal(0, SENSOR_SD, (n, 16, len(sensor_idx))), -0.2, 0.2)
obs = np.maximum(v_all[:, :, sensor_idx] * (1 + nu), 0).astype(np.float32)   # link counts: 5 % sensor error, clipped to +-20 %
qsum = demand.sum(2)
theta_exp = (demand * theta_od[None]).sum(2) / np.where(qsum > 0, qsum, 1)        # (days, 16)
C.log(f'Logit assignment done ({time.time() - t_start:.0f} s)')

# ---------------------------------------------------------------------------------------------------------------
# 3. write
# ---------------------------------------------------------------------------------------------------------------
for split in ('train', 'test'):
    ix = days.index[days.split == split].values
    out = os.path.join(C.DAYS, split); os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, 'days.npz'), demand=demand[ix], link_flow=obs[ix], sensor_ids=sensor_ids,
                        od_o=od.o_zone_id.values, od_d=od.d_zone_id.values,
                        link_flow_all=v_all[ix].astype(np.float32), theta_expected=theta_exp[ix].astype(np.float32),
                        theta_od=theta_od.astype(np.float32), eta=eta.astype(np.float32))
    dd = days.loc[ix].copy(); dd['source_day'] = dd['day']; dd['split'] = split
    dd.to_csv(os.path.join(out, 'days.csv'), index=False)

pt0 = spmm(LPT[0], free_flow_tt.reshape(-1, 1)).reshape(num_od, k).numpy()
mm = (pt0.max(1) - pt0.min(1)) > 1e-6
u = -theta_od[8][mm, None] * pt0[mm]; u -= u.max(1, keepdims=True); pp = np.exp(u); pp /= pp.sum(1, keepdims=True)
tab = pd.DataFrame({'clock': C.CLOCK16})
for typ in types:
    sel = days.index[days.type == typ].values
    tab[f'{typ} sensors (k)'] = (obs[sel].sum(2).mean(0) / 1000).round(1)
    tab[f'{typ} theta_exp'] = theta_exp[sel].mean(0).round(3)
tab['real counts (k)'] = (flow.values.sum(1) / 1000).round(1)
lines = [f'{C.DAYS_NAME}: theta0 {THETA0:.4f}, peak x{PEAK_MULT} at slots {PEAK_SLOTS}, stadium x{STADIUM_MULT}, '
         f'POI x{POI_MULT}, eta sd {ETA_SD}, sensor sd {SENSOR_SD}, recursion from slot {START_SLOT}, seed {SEED_LOGIT}',
         f'fastest-route share at 8:00 (free-flow, T1): {float(np.median(pp.max(1))):.3f} (even split 0.5)',
         pd.crosstab(days.type, days.split).to_string(), '', tab.to_string(index=False)]
open(os.path.join(C.DAYS, 'summary.txt'), 'w').write('\n'.join(lines))
print('\n'.join(lines))
C.log(f'wrote {os.path.relpath(C.DAYS, C.ROOT)} ({time.time() - t_start:.0f} s)')
