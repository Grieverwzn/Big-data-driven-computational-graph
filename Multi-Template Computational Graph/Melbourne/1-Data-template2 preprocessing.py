"""
MTCG: Multi-Template Computational Graph for Traffic Demand Flow Estimation
Melbourne Network Case Study

Authors:
    Xin (Bruce) Wu, Department of Civil and Environmental Engineering, Villanova University, USA
    Feng Shao, School of Mathematics, China University of Mining and Technology, China

Contact: xwu03@villanova.edu

MIT License
Copyright (c) 2026 Xin (Bruce) Wu, Feng Shao
"""
# Description: Template 2 data preprocessing: attraction-based demand generation

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pandas import DataFrame
import scipy.sparse as sp
import math
import time
import glob
from tqdm.notebook import trange

node = pd.read_csv('data/node.csv')
node = node.iloc[:,:-3]

link = pd.read_csv('data/link.csv')
link = link[['link_id', 'from_node_id', 'to_node_id','capacity', 'free_speed','lanes','length']]


def length(node_1, node_2):
    start = node[node['node_id'] == node_1][['x_coord', 'y_coord']].values
    end = node[node['node_id'] == node_2][['x_coord', 'y_coord']].values
    if (len(start) > 0) & (len(end) > 0):
        length = (np.sum((start - end) ** 2) ** 0.5) / 1000  # km
    else:
        length = -1

    return length

for i in range(len(link)):
    link.loc[i, 'length'] = length(link.loc[i, 'from_node_id'], link.loc[i, 'to_node_id'])
#link = link[link['length']>0]
link.loc[link['length']==-1, 'length'] = link.loc[link['length']>0, 'length'].mean()

link['free_flow_travel_time'] = link['length']/link['free_speed']*60 # minute


from pyproj import Transformer


TARGET_EPSG = "EPSG:32755"
transformer = Transformer.from_crs("EPSG:4326", TARGET_EPSG, always_xy=True)


stadiums_ll = pd.DataFrame([
    {"name": "Melbourne Cricket Ground", "lon":144.98312373885017, "lat":-37.81863186124403,  "capacity":100024, "car_ratio":0.5},
    {"name": "AAMI Park",                 "lon":144.98466869128995, "lat":-37.823106686435935, "capacity":30050,  "car_ratio":0.5},
    {"name": "Rod Laver Arena",           "lon":144.97866054291288, "lat":-37.819920096304315, "capacity":14820,  "car_ratio":0.5},
    {"name": "Margaret Court Arena",      "lon":144.97797389738406, "lat":-37.81971669226319,  "capacity":7500,   "car_ratio":0.5},
    {"name": "John Cain Arena",           "lon":144.98235126263023, "lat":-37.82114050878017,  "capacity":10500,  "car_ratio":0.5},
    {"name": "Marvel Stadium",            "lon":144.94748404232723, "lat":-37.81633586001247,  "capacity":56000,  "car_ratio":0.5},
])


stadiums_ll[["x_coord","y_coord"]] = stadiums_ll.apply(
    lambda r: transformer.transform(r["lon"], r["lat"]), axis=1, result_type="expand"
)

max_node_id_now = int(node["node_id"].max())
stadiums_ll["node_id"] = np.arange(max_node_id_now+1, max_node_id_now+1+len(stadiums_ll))


for col in ["x_coord","y_coord","name"]:
    if col not in node.columns: node[col] = np.nan


node = pd.concat([node, stadiums_ll[["node_id","x_coord","y_coord","name"]]], ignore_index=True)


pois_ll = pd.DataFrame([
    {"name": "Queen Victoria Market", "lon": 144.9569, "lat": -37.8060, "car_ratio": 0.35},
    {"name": "St Kilda Beach",       "lon": 144.9739, "lat": -37.8676, "car_ratio": 0.50},
])


pois_ll[["x_coord","y_coord"]] = pois_ll.apply(
    lambda r: transformer.transform(r["lon"], r["lat"]), axis=1, result_type="expand"
)


max_node_id_now = int(node["node_id"].max())
pois_ll["node_id"] = np.arange(max_node_id_now+1, max_node_id_now+1+len(pois_ll))


node = pd.concat([node, pois_ll[["node_id","x_coord","y_coord","name"]]], ignore_index=True)


POI_NODE_IDS = pois_ll["node_id"].astype(int).tolist()
POI_META = pois_ll.set_index("node_id")[["car_ratio"]].to_dict("index")


STADIUM_NODE_IDS = stadiums_ll["node_id"].astype(int).tolist()
STADIUM_META = stadiums_ll.set_index("node_id")[["capacity","car_ratio"]].to_dict("index")

file_path = glob.glob('data/OD_matrix*.csv')
file_path

demand0 = pd.read_csv(file_path[0],index_col=0).iloc[:-1,:-1]
demand0.shape

# Calculate the distances between centroids and other nodes in the network
def distance(node_1, node_2):
    start = node[node['node_id'] == node_1][['x_coord', 'y_coord']].values
    end = node[node['node_id'] == node_2][['x_coord', 'y_coord']].values
    if (len(start) > 0) & (len(end) > 0):
        distance = (np.sum((start - end) ** 2) ** 0.5) / 1000  # km
    else:
        distance = 10000
    return distance


rows = list(demand0.index)
cols = list(demand0.columns)


EXTRA_CENTROIDS = list(map(int, STADIUM_NODE_IDS + POI_NODE_IDS))
for sid in EXTRA_CENTROIDS:
    sid_str = str(sid)
    if sid_str not in rows: rows.append(sid_str)
    if sid_str not in cols: cols.append(sid_str)


demand0_ext = pd.DataFrame(0, index=rows, columns=cols, dtype=float)
demand0_ext.loc[demand0.index, demand0.columns] = demand0.values
demand0 = demand0_ext.copy()



Centroid = demand0.columns.values.astype(int)

# starting nodes of all links
link_from_node_id = link['from_node_id'].unique()
# Ending nodes of all links
link_to_node_id = link['to_node_id'].unique()


cent_set = set(Centroid.astype(int).tolist())

link_from_node_id = np.array([int(x) for x in link_from_node_id if int(x) not in cent_set], dtype=int)
link_to_node_id   = np.array([int(x) for x in link_to_node_id   if int(x) not in cent_set], dtype=int)
# =============================================================

link_add = np.zeros([1, 2])
for i in trange(len(Centroid)):
    cen = Centroid[i]

    # Ensure that all centroids can output traffic flow as starting points
    if cen not in link_from_node_id:
        node_distance = np.zeros(link_from_node_id.shape)
        for j in range(len(node_distance)):
            node_distance[j] = distance(link_from_node_id[j], cen)
        link_add_i_1 = np.stack([cen * np.ones(4), link_from_node_id[np.argsort(node_distance)[:4]]]).T
        link_add = np.vstack([link_add, link_add_i_1])

    # Ensure that all centroids are reachable as destinations.
    if cen not in link_to_node_id:
        node_distance = np.zeros(link_to_node_id.shape)
        for j in range(len(node_distance)):
            node_distance[j] = distance(link_to_node_id[j], cen)
        link_add_i_2 = np.stack([link_to_node_id[np.argsort(node_distance)[:4]], cen * np.ones(4)]).T
        link_add = np.vstack([link_add, link_add_i_2])

link_add = link_add[1:]
link_add = DataFrame(link_add)
link_add.columns = ['from_node_id', 'to_node_id']
link_add


link_add['is_connector'] = 1


link_mean_len = link[link['length'] > 0]['length'].mean() if (link['length'] > 0).any() else 1.0
link_add['length'] = link_add.apply(
    lambda r: max(1e-6, length(int(r['from_node_id']), int(r['to_node_id']))), axis=1
)
link_add.loc[link_add['length'] < 0, 'length'] = link_mean_len


CONNECTOR_FF_TT = 1000.0  # minute

link_add['free_flow_travel_time'] = CONNECTOR_FF_TT


link_add['free_speed'] = (link_add['length'] / (CONNECTOR_FF_TT / 60.0)).replace([np.inf, 0], np.nan)
if link_add['free_speed'].isna().any():
    link_add['free_speed'] = link_add['free_speed'].fillna(link['free_speed'].median())


link_add['capacity'] = link['capacity'].median() * 10.0
link_add['lanes'] = link['lanes'].median()
# ================================================================================

link_last = pd.concat([link, link_add])
link_last.reset_index(drop=True, inplace=True)
link_last['link_id_new'] = link_last.index+1


if 'is_connector' not in link_last.columns:
    link_last['is_connector'] = 0
link_last['is_connector'] = link_last['is_connector'].fillna(0).astype(int)
# ===============================================================================

link_last.to_csv('data_2/link_last.csv',index=False)

link_last = pd.read_csv('data_2/link_last.csv')
link_last

flow0 = pd.read_csv('data/observed_traffic_volume.csv')
flow0

# re-number the links
ob_unique = flow0['link_ID'].unique()
for i in ob_unique:
    flow0.loc[flow0['link_ID']==i, 'link_id_new'] = int(link_last.loc[link_last['link_id']==i, 'link_id_new'])

flow = DataFrame(np.reshape(flow0['observed_volume'].values, [-1,16])).T
flow.columns = flow0['link_id_new'].unique()
flow.index = flow0['time'].unique()
print(flow.shape)
flow


# re-number the nodes
node_unique = np.unique(np.hstack([link_last['from_node_id'].values, link_last['to_node_id'].values]))
print(node_unique.shape)
node_unique

DataFrame(node_unique).to_csv('data_new_2/node_unique.csv',index=False)



for i in range(len(node_unique)):
    link_last.loc[link_last['from_node_id']==node_unique[i],'from_node_id'] = i+1
    link_last.loc[link_last['to_node_id']==node_unique[i],'to_node_id'] = i+1
    Centroid[Centroid==node_unique[i]] = i+1

link_last





id_map = {int(old): int(i+1) for i, old in enumerate(node_unique)}  # old -> new

stad_chk = stadiums_ll[['name','node_id']].copy()
stad_chk['node_id'] = stad_chk['node_id'].astype(int)
stad_chk['new_id'] = stad_chk['node_id'].map(id_map)

poi_chk = pois_ll[['name','node_id']].copy()
poi_chk['node_id'] = poi_chk['node_id'].astype(int)
poi_chk['new_id'] = poi_chk['node_id'].map(id_map)

print("\n=== Stadium old->new ===")
print(stad_chk)
print("\n=== POI old->new ===")
print(poi_chk)
# ============================================



num_od_node = len(Centroid)
print(num_od_node)

OD_pair = np.zeros([num_od_node * num_od_node, 2])
for i in range(num_od_node):
    for j in range(num_od_node):
        OD_pair[num_od_node * i+j, 0] = Centroid[i]
        OD_pair[num_od_node * i+j, 1] = Centroid[j]
print(OD_pair.shape)




old_centroids = demand0.columns.astype(int).tolist()
new_centroids_expected = [id_map[x] for x in old_centroids]



num_od = len(old_centroids)
OD_old = np.zeros((num_od*num_od, 2), dtype=int)
idx_tmp = 0
for o in old_centroids:
    for d in old_centroids:
        OD_old[idx_tmp, 0] = o
        OD_old[idx_tmp, 1] = d
        idx_tmp += 1

OD_old_mapped = np.column_stack([
    np.vectorize(id_map.get)(OD_old[:,0]),
    np.vectorize(id_map.get)(OD_old[:,1]),
]).astype(int)



file_path = sorted(file_path)
T = len(file_path)

SLOT_MINUTES = 15

ARRIVE_WEIGHTS = np.array([0.10, 0.30, 0.60])
DEPART_WEIGHTS = np.array([0.70, 0.30])
ARRIVE_WEIGHTS /= ARRIVE_WEIGHTS.sum()
DEPART_WEIGHTS /= DEPART_WEIGHTS.sum()


EVENTS = [
    {"sid": STADIUM_NODE_IDS[0], "t_event": 6, "duration_min": 120, "attend": 100024, "car_ratio": 0.5},  # MCG
    {"sid": STADIUM_NODE_IDS[1], "t_event": 6, "duration_min": 120, "attend": 30050,  "car_ratio": 0.5},  # AAMI Park
    {"sid": STADIUM_NODE_IDS[2], "t_event": 6, "duration_min": 120, "attend": 14820,  "car_ratio": 0.5},  # Rod Laver Arena
    {"sid": STADIUM_NODE_IDS[3], "t_event": 6, "duration_min": 120, "attend": 7500,   "car_ratio": 0.5},  # Margaret Court Arena
    {"sid": STADIUM_NODE_IDS[4], "t_event": 6, "duration_min": 120, "attend": 10500,  "car_ratio": 0.5},  # John Cain Arena
    {"sid": STADIUM_NODE_IDS[5], "t_event": 6, "duration_min": 120, "attend": 56000,  "car_ratio": 0.5},  # Marvel Stadium
]

ATTRACTIONS = [
    {"sid": POI_NODE_IDS[0], "visitors_am": 10300, "car_ratio": 0.35, "col_start": 0, "col_end": T-1},
    {"sid": POI_NODE_IDS[1], "visitors_am":  3300,  "car_ratio": 0.50, "col_start": 0, "col_end": T-1},
]


W_ARR = np.zeros((len(EVENTS), T), dtype=float)
W_DEP = np.zeros((len(EVENTS), T), dtype=float)

for k, ev in enumerate(EVENTS):
    t0 = int(ev["t_event"])
    if "duration_cols" in ev and pd.notna(ev["duration_cols"]):
        dur_cols = int(ev["duration_cols"])
    elif "duration_min" in ev and pd.notna(ev["duration_min"]):
        dur_cols = int(np.ceil(float(ev["duration_min"]) / max(1, SLOT_MINUTES)))
    else:
        dur_cols = 8
    t_end = t0 + dur_cols

    for j, w in enumerate(ARRIVE_WEIGHTS, start=1):
        tt = t0 - (len(ARRIVE_WEIGHTS) - j + 1)
        if 0 <= tt < T:
            W_ARR[k, tt] = w

    for j, w in enumerate(DEPART_WEIGHTS):
        tt = t_end + j
        if 0 <= tt < T:
            W_DEP[k, tt] = w

    sa = W_ARR[k].sum()
    if sa > 0: W_ARR[k] /= sa
    sd = W_DEP[k].sum()
    if sd > 0: W_DEP[k] /= sd


_node_xy = node.set_index("node_id")[["x_coord","y_coord"]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_dict("index")
def _xy(nid):
    rec = _node_xy.get(int(nid), {"x_coord": 0.0, "y_coord": 0.0})
    return float(rec.get("x_coord", 0.0)), float(rec.get("y_coord", 0.0))
def _dist_km(a, b):
    ax, ay = _xy(a); bx, by = _xy(b)
    return np.hypot(ax - bx, ay - by) / 1000.0

THETA = 3.0
rows = list(demand0.index)
cols = list(demand0.columns)


demand_cols = []
for t_idx, path in enumerate(file_path):
    M = pd.read_csv(path, index_col=0).iloc[:-1, :-1]   # 背景OD
    M_ext = pd.DataFrame(0.0, index=rows, columns=cols)
    M_ext.loc[M.index, M.columns] = M.values


    for k, ev in enumerate(EVENTS):
        sid = int(ev["sid"])
        cap = float(STADIUM_META[sid]["capacity"])
        default_car = float(STADIUM_META[sid]["car_ratio"])

        att_raw = ev.get("attend", None)
        if att_raw is None or (isinstance(att_raw, float) and np.isnan(att_raw)):
            att = cap
        else:
            att = float(att_raw)

        car_raw = ev.get("car_ratio", None)
        if car_raw is None or (isinstance(car_raw, float) and np.isnan(car_raw)):
            car = default_car
        else:
            car = float(car_raw)
            if car > 1.0:
                car /= 100.0

        Qtot = att * car
        wA = W_ARR[k, t_idx]
        wD = W_DEP[k, t_idx]

        if wA > 0 and Qtot > 0:
            O = [int(o) for o in rows if int(o) != sid]
            d = np.array([_dist_km(o, sid) for o in O], dtype=float)
            if (not np.isfinite(d).all()) or (np.max(d) <= 0):
                p = np.ones_like(d) / len(d)
            else:
                p = np.exp(-THETA * (d / np.max(d))); p /= p.sum()
            for o, add in zip(O, Qtot * wA * p):
                M_ext.loc[str(o), str(sid)] += add

        if wD > 0 and Qtot > 0:
            Dst = [int(dz) for dz in cols if int(dz) != sid]
            d2 = np.array([_dist_km(sid, dz) for dz in Dst], dtype=float)
            if (not np.isfinite(d2).all()) or (np.max(d2) <= 0):
                p2 = np.ones_like(d2) / len(d2)
            else:
                p2 = np.exp(-THETA * (d2 / np.max(d2))); p2 /= p2.sum()
            for dz, add in zip(Dst, Qtot * wD * p2):
                M_ext.loc[str(sid), str(dz)] += add

    for a in ATTRACTIONS:
        sid  = int(a["sid"])
        vis  = float(a.get("visitors_am", 0) or 0)
        carr = float(a.get("car_ratio", 0.4) or 0.4)
        if carr > 1.0: carr /= 100.0
        cs   = max(0, int(a.get("col_start", 0)))
        ce   = min(T-1, int(a.get("col_end", T-1)))

        if cs <= t_idx <= ce and vis > 0:
            Ncols = ce - cs + 1
            Qtot  = vis * carr / max(1, Ncols)
            O = [int(o) for o in rows if int(o) != sid]
            d = np.array([_dist_km(o, sid) for o in O], dtype=float)
            if (not np.isfinite(d).all()) or (np.max(d) <= 0):
                p = np.ones_like(d) / len(d)
            else:
                p = np.exp(-THETA * (d / np.max(d))); p /= p.sum()
            for o, add in zip(O, Qtot * p):
                M_ext.loc[str(o), str(sid)] += add

    demand_cols.append(M_ext.values.reshape(-1, 1))

demand = np.hstack(demand_cols)
print("demand.shape =", demand.shape)


import networkx as nx
G = nx.DiGraph()
G.add_edges_from(zip(link_last['from_node_id'].astype(int),
                     link_last['to_node_id'].astype(int)))

Z = np.array(Centroid, dtype=int)
reachable_pairs = set()
for o in Z:
    reach = nx.descendants(G, o) | {o}
    reach_in_Z = set(reach).intersection(Z)
    for d in reach_in_Z:
        reachable_pairs.add((o, d))

reachable_mask = np.array([(int(OD_pair[i,0]), int(OD_pair[i,1])) in reachable_pairs
                           for i in range(OD_pair.shape[0])])



idx = demand.sum(axis=1)
demand = demand[idx>0,:]
OD_pair = OD_pair[idx>0,:]



link_last['link_id'] = link_last['link_id_new']
link_last['facility_type'] = 'Highway'
link_last['dir_flag'] = 1
link_last['length'] = link_last['length'].fillna(100)
link_last['lanes'] = link_last['lanes'].fillna(0)
link_last['capacity'] = link_last['capacity'].fillna(100)
link_last['free_speed'] = link_last['free_speed'].fillna(60)
link_last['link_type'] = 1
link_last['cost'] = 0
link_last['VDF_fftt1'] = link_last['free_flow_travel_time'].fillna(100)
link_last['VDF_cap1'] = link_last['capacity']
link_last['VDF_alpha1'] = 0.15
link_last['VDF_beta1'] = 4
link_last['VDF_theta1'] = 1
link_last['VDF_gamma1'] = 1
link_last['VDF_mu1'] = 100
link_last['RUC_resource1'] = 0
link_last['RUC_rho1'] = 10
link_last['RUC_type'] = 1
link_last['name'] = np.NaN


link = link_last[['name','link_id','from_node_id','to_node_id','facility_type','dir_flag','length',
                  'lanes','capacity','free_speed','link_type','cost','VDF_fftt1','VDF_cap1','VDF_alpha1',
                  'VDF_beta1','VDF_theta1','VDF_gamma1','VDF_mu1','RUC_rho1','RUC_resource1','RUC_type',
                  'is_connector']]

link.to_csv('data_2/link.csv',index=False)

node = DataFrame(np.arange(1,len(node_unique)+1))
node.columns = ['node_id']
node['zone_id'] = node['node_id']
node[['name','bin_index','node_type','control_type','x_coord','y_coord','geometry']] = np.NaN
node[['production', 'attraction']] = 1
node.to_csv('data_2/node.csv', index = False)



OD_pair = DataFrame(OD_pair)
OD_pair.columns = ['o_zone_id', 'd_zone_id']
OD_pair.to_csv('data_2/OD_pair.csv', index = False)

def round_preserve_total(arr):
    arr_int = np.floor(arr).astype(int)
    diff = int(round(arr.sum() - arr_int.sum()))
    if diff > 0:
        idx = np.random.choice(len(arr), diff, replace=False)
        arr_int[idx] += 1
    return arr_int


DataFrame(demand).to_csv('data_2/demand_6-10.csv', index=False)

flow.to_csv('data_2/link_flow.csv', index = False)
