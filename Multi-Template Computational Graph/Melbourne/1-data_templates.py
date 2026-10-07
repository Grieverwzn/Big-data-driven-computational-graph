"""Step 1: demand of the three scenario templates from the open GMNS Melbourne data.

Paper
-----
Section 7.3.1 (Data Generation): observed demand q_t^0, event trips Delta q_j at full size and their time
shares a_{j,t} of Eq. 51 (j = 1 stadium, j = 2 POI); the venues and cars of Table 8 (Table 8 itself: step 6).
Code -> paper: d1 = q_t^0 (16 slots 6:00-9:45), e2 = a_{1,t} Delta q_1, e3 = a_{2,t} Delta q_2.

What
----
Template 1 (regular day) is the open-data OD demand d1 of the real morning (16 slots of 15 min, 6:00-10:00).
Templates 2 and 3 add the extra trips of an event to d1:
  Template 2 = d1 + e2, stadium events at six venues (MCG, AAMI Park, Rod Laver Arena, Margaret Court Arena,
               John Cain Arena, Marvel Stadium), event start 9:45, arrivals at 8:45 / 9:00 / 9:15 / 9:30 with
               10 / 20 / 30 / 40 %. Size per venue = seats x attendance 50 % x car share 25 % / 2.2 persons per car
               (12,437 cars in total).
  Template 3 = d1 + e3, points of interest (Queen Victoria Market, St Kilda Beach), event start 7:45, arrivals at
               7:00 / 7:15 / 7:30 with 10 / 30 / 60 % (7,975 cars: 10,300 visitors x 55 % + 3,300 x 70 %).
Origins of the event trips: gravity distribution p_o ~ exp(-3 d_o / d_max) over all zones o (d = straight-line
distance to the venue). The computation follows, step by step and with the same floating-point operations, the
original preprocessing of the paper (scripts 1-Data-template{1,2,3} preprocessing.py, which first built full-size
events, and build_templates.py, which rescaled the stadium events and moved the arrival times), so the output is
identical to the files used in the paper.
The real link counts (448 sensors, 16 slots) are also rearranged to the renumbered links (only their sensor link ids
are used by the model; the Logit benchmark in step 3 replaces the counts).

Usage
-----
    python 1-data_templates.py                 (about 10 s)

Inputs
------
    data/open_data/OD_matrix_*.csv, node.csv, observed_traffic_volume.csv   open GMNS Melbourne data
    data/network/OD_pair.csv, node_unique.csv, link.csv, link_last.csv       renumbered network (centroid connectors
                                                                              added by the original preprocessing)
Outputs
-------
    data/data_new_1/demand_6-10.csv, link.csv, OD_pair.csv, link_flow.csv   Template 1 (30,040 OD pairs x 16 slots)
    data/data_new_2/demand_6-10.csv                                          Template 2
    data/data_new_3/demand_6-10.csv                                          Template 3
    data/event_extras.npz                                                    e2, e3, stadium_arrivals, poi_total

Authors:
    Xin (Bruce) Wu, Department of Civil and Environmental Engineering, Villanova University, PA, USA
    Feng Shao, School of Mathematics, China University of Mining and Technology, China

Contact: xwu03@villanova.edu (Villanova University), xinwu8592@gmail.com (personal)

MIT License
Copyright (c) 2026 Xin (Bruce) Wu, Feng Shao
"""
import io
import os
import glob
import shutil
import numpy as np
import pandas as pd
from pyproj import Transformer
import mtcg_common as C

# ---------------------------------------------------------------------------------------------------------------
# Venues (coordinates in WGS84, projected to UTM zone 55S as the GMNS node coordinates)
# ---------------------------------------------------------------------------------------------------------------
STADIUMS = [("Melbourne Cricket Ground", 144.98312373885017, -37.81863186124403, 100024),
            ("AAMI Park", 144.98466869128995, -37.823106686435935, 30050),
            ("Rod Laver Arena", 144.97866054291288, -37.819920096304315, 14820),
            ("Margaret Court Arena", 144.97797389738406, -37.81971669226319, 7500),
            ("John Cain Arena", 144.98235126263023, -37.82114050878017, 10500),
            ("Marvel Stadium", 144.94748404232723, -37.81633586001247, 56000)]
POIS = [("Queen Victoria Market", 144.9569, -37.8060, 10300, 0.55),      # (name, lon, lat, morning visitors, car share)
        ("St Kilda Beach", 144.9739, -37.8676, 3300, 0.70)]
GRAVITY = 3.0                       # p_o ~ exp(-GRAVITY * d_o / d_max)
FULL_CAR_SHARE = 0.5                # the original full-size events: every seat, half of the spectators drive
STADIUM_SCALE = (0.50 * 0.25 / 2.2) / 0.5   # realistic size relative to the full-size event (attendance 50 %,
                                            # car share 25 %, 2.2 persons per car)
W_ARR_STADIUM = np.array([0.10, 0.20, 0.30, 0.40])   # 8:45, 9:00, 9:15, 9:30 (slots 11-14)
W_ARR_POI = np.array([0.10, 0.30, 0.60])             # 7:00, 7:15, 7:30 (slots 4-6)

# ---------------------------------------------------------------------------------------------------------------
# Zones and coordinates (as in the original preprocessing)
# ---------------------------------------------------------------------------------------------------------------
node = pd.read_csv(os.path.join(C.OPEN_DATA, 'node.csv'))
node = node.iloc[:, :-3]
tr = Transformer.from_crs("EPSG:4326", "EPSG:32755", always_xy=True)
nid0 = int(node['node_id'].max())
venues = []                                           # (node id, x, y)
for j, (name, lon, lat, *_) in enumerate(STADIUMS + POIS):
    x, y = tr.transform(lon, lat)
    venues.append((nid0 + 1 + j, x, y))
stadium_ids = [v[0] for v in venues[:6]]
poi_ids = [v[0] for v in venues[6:]]
xy = node.set_index('node_id')[['x_coord', 'y_coord']].apply(pd.to_numeric, errors='coerce').fillna(0.0)
xy = {int(k): (float(r.x_coord), float(r.y_coord)) for k, r in xy.iterrows()}
xy.update({v[0]: (float(v[1]), float(v[2])) for v in venues})

files = sorted(glob.glob(os.path.join(C.OPEN_DATA, 'OD_matrix*.csv')))
assert len(files) == C.N_SLOT
first = pd.read_csv(files[0], index_col=0).iloc[:-1, :-1]          # last row / column = totals
rows = list(first.index) + [str(i) for i in stadium_ids + poi_ids]
cols = list(first.columns) + [str(i) for i in stadium_ids + poi_ids]
assert rows == cols, 'OD matrix rows and columns are not in the same order'
pos = {int(z): i for i, z in enumerate(rows)}
n = len(rows)

# real-day OD demand of every slot on the extended zone set (venue zones carry no regular demand)
D1 = np.zeros((C.N_SLOT, n, n))
for t, f in enumerate(files):
    M = pd.read_csv(f, index_col=0).iloc[:-1, :-1]
    assert list(M.index) == rows[:len(M)] and list(M.columns) == cols[:len(M)]
    D1[t, :len(M), :len(M)] = M.values


def csv_roundtrip(x):
    """The original event trips were stored in a csv file and read back with pandas' default (fast, not always
    round-trip exact) float parser; repeating this keeps the numbers bit-identical to the paper's files."""
    txt = 'v\n' + '\n'.join(repr(float(v)) for v in x)
    return pd.read_csv(io.StringIO(txt))['v'].values


def gravity(sid):
    """Share of each origin zone in the trips to venue sid: exp(-3 d / d_max), normalized (original code)."""
    O = [int(o) for o in rows if int(o) != sid]
    d = np.array([np.hypot(xy[o][0] - xy[sid][0], xy[o][1] - xy[sid][1]) / 1000.0 for o in O], dtype=float)
    p = np.exp(-GRAVITY * (d / np.max(d)))
    p /= p.sum()
    return O, p


# full-size stadium arrivals in the 60 % slot of the original template, divided by 0.6 (= total arrivals); the arrival
# weights are normalised twice in the original code, which is repeated here so that the numbers are bit-identical
arr = np.array([0.10, 0.30, 0.60]); arr /= arr.sum()
w_row = np.zeros(C.N_SLOT); w_row[3:6] = arr; w_row /= w_row.sum()
wA = w_row[5]
E2 = np.zeros((n, n))                                 # total arrivals of the six stadium events, per OD pair
for (name, lon, lat, cap), sid in zip(STADIUMS, stadium_ids):
    O, p = gravity(sid)
    add = cap * FULL_CAR_SHARE * wA * p
    E2[[pos[o] for o in O], pos[sid]] = (csv_roundtrip(add) - 0.0) / 0.6
E3 = np.zeros((n, n))                                 # total morning POI trips, per OD pair
for (name, lon, lat, vis, car), sid in zip(POIS, poi_ids):
    O, p = gravity(sid)
    add = vis * car / max(1, C.N_SLOT) * p            # original: spread evenly over the 16 slots ...
    E3[[pos[o] for o in O], pos[sid]] = csv_roundtrip(add) * 16   # ... morning total = 16 x the slot value

# ---------------------------------------------------------------------------------------------------------------
# Renumbered OD pairs (30,040 pairs; the order of OD_pair.csv is the order of every demand vector in the package)
# ---------------------------------------------------------------------------------------------------------------
od = pd.read_csv(os.path.join(C.NETWORK, 'OD_pair.csv'))
nu = pd.read_csv(os.path.join(C.NETWORK, 'node_unique.csv')).iloc[:, 0].values.astype(np.int64)  # new id i+1 -> old id
oi = np.array([pos[int(nu[o - 1])] for o in od.o_zone_id]); di = np.array([pos[int(nu[d - 1])] for d in od.d_zone_id])
d1 = D1[:, oi, di].T                                  # (num_od, 16)
stadium_arrivals = E2[oi, di].clip(min=0) * STADIUM_SCALE
poi_total = E3[oi, di].clip(min=0)
assert np.isin(od.d_zone_id.values[stadium_arrivals > 0], list(C.STADIUM_ZONES)).all()
assert np.isin(od.d_zone_id.values[poi_total > 0], list(C.POI_ZONES)).all()

e2 = np.zeros_like(d1); e2[:, 11:15] = stadium_arrivals[:, None] * W_ARR_STADIUM[None, :]
e3 = np.zeros_like(d1)
for j, w in enumerate(W_ARR_POI):
    e3[:, 4 + j] = poi_total * w
d2, d3 = d1 + e2, d1 + e3

# ---------------------------------------------------------------------------------------------------------------
# Real link counts on the renumbered links (448 sensors x 16 slots), as in the original preprocessing
# ---------------------------------------------------------------------------------------------------------------
link_last = pd.read_csv(os.path.join(C.NETWORK, 'link_last.csv'))
new_id = dict(zip(link_last.link_id, link_last.link_id_new))
f0 = pd.read_csv(os.path.join(C.OPEN_DATA, 'observed_traffic_volume.csv'))
f0['link_id_new'] = [float(new_id[i]) for i in f0['link_ID']]
flow = pd.DataFrame(np.reshape(f0['observed_volume'].values, [-1, 16])).T
flow.columns = f0['link_id_new'].unique()

# ---------------------------------------------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------------------------------------------
for s, d in ((1, d1), (2, d2), (3, d3)):
    out = C.template_dir(s); os.makedirs(out, exist_ok=True)
    pd.DataFrame(d, columns=[str(t) for t in range(C.N_SLOT)]).to_csv(os.path.join(out, 'demand_6-10.csv'), index=False)
for f in ('link.csv', 'OD_pair.csv'):
    shutil.copy(os.path.join(C.NETWORK, f), os.path.join(C.template_dir(1), f))
flow.to_csv(os.path.join(C.template_dir(1), 'link_flow.csv'), index=False)
np.savez_compressed(C.EVENT_FILE, e2=e2, e3=e3, stadium_arrivals=stadium_arrivals, poi_total=poi_total)

tab = pd.DataFrame({'clock': C.CLOCK16, 'T1 regular': d1.sum(0).round(), 'T2 stadium': d2.sum(0).round(),
                    'T3 POI': d3.sum(0).round(), 'sensor counts': flow.values.sum(1)})
print('total demand per slot (veh / 15 min):'); print(tab.to_string(index=False))
print(f'stadium events: {stadium_arrivals.sum():,.0f} cars on {(stadium_arrivals > 0).sum():,} OD pairs; '
      f'POI events: {poi_total.sum():,.0f} cars on {(poi_total > 0).sum():,} OD pairs')
print('wrote data/data_new_{1,2,3}/demand_6-10.csv, data/data_new_1/link_flow.csv, data/event_extras.npz')
