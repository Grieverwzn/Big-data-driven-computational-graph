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
# Description: K-shortest path generation for all templates

import os
import numpy as np
import pandas as pd
import path4gmns as pg


BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data_3")
os.chdir(DATA_DIR)
print("CWD =", os.getcwd())
print("path4gmns version:", getattr(pg, "__version__", "unknown"))


link = pd.read_csv("link.csv")
node = pd.read_csv("node.csv")
od   = pd.read_csv("OD_pair.csv")

for c in ["from_node_id", "to_node_id", "link_id"]:
    if c in link.columns:
        link[c] = pd.to_numeric(link[c], errors="coerce").fillna(0).astype(int)
for c in ["node_id", "zone_id"]:
    if c in node.columns:
        node[c] = pd.to_numeric(node[c], errors="coerce").fillna(0).astype(int)
for c in ["o_zone_id", "d_zone_id"]:
    if c in od.columns:
        od[c] = pd.to_numeric(od[c], errors="coerce").fillna(0).astype(int)


CONST_THETA = 1

link["VDF_theta1"] = float(CONST_THETA)


if "VDF_fftt1" not in link.columns:
    link["VDF_fftt1"] = np.nan
need = link["VDF_fftt1"].isna() | (pd.to_numeric(link["VDF_fftt1"], errors="coerce") <= 0)
if need.any():
    # length[km] / free_speed[km/h] * 60 -> min
    tt_backup = (pd.to_numeric(link.get("length", pd.Series(np.nan)), errors="coerce") /
                 pd.to_numeric(link.get("free_speed", pd.Series(np.nan)), errors="coerce") * 60.0)
    link.loc[need, "VDF_fftt1"] = tt_backup.clip(lower=0.1).fillna(1.0)  # 最少 0.1 分钟，兜底 1.0


TIME_SCALE = 1


link["VDF_fftt1"] = pd.to_numeric(link["VDF_fftt1"], errors="coerce")
link["VDF_fftt1"] *= TIME_SCALE


if "VDF_alpha1" not in link.columns: link["VDF_alpha1"] = 0.15
if "VDF_beta1"  not in link.columns: link["VDF_beta1"]  = 4

base_cap = pd.to_numeric(link["capacity"], errors="coerce").fillna(0.0).astype(float)
link["VDF_cap1"] = base_cap * CONST_THETA

if "free_speed" not in link.columns: link["free_speed"] = 60
if "lanes" not in link.columns:      link["lanes"] = 1

link.to_csv("link.csv", index=False)
node.to_csv("node.csv", index=False)
od.to_csv("OD_pair.csv", index=False)

demand_file = (
    "demand_6-10.csv" if os.path.exists("demand_6-10.csv")
    else ("demand610.csv" if os.path.exists("demand610.csv") else "demand_6_10.csv")
)
D_df = pd.read_csv(demand_file, header=None)
D_df = D_df.dropna(how="all", axis=0).dropna(how="all", axis=1)


od_len = len(od)
if D_df.shape[0] == od_len + 1:
    first = pd.to_numeric(D_df.iloc[0], errors="coerce")
    if np.allclose(first.to_numpy(), np.arange(D_df.shape[1], dtype=float)):
        D_df = D_df.iloc[1:].reset_index(drop=True)


if D_df.shape[0] > od_len:
    D_df = D_df.iloc[:od_len, :].reset_index(drop=True)

assert D_df.shape[0] == od_len, f"需求矩阵行数({D_df.shape[0]}) != OD_pair({od_len})"
D = D_df.to_numpy(dtype=float)
T = D.shape[1]

SCALE_K    = 12.0
SEED_VALUE = 1e-4
EPS = 0.0


weights = np.array([
    0.5, 0.5, 1.0, 1.0,   # t0..t3
    2.5, 2.5, 2.5, 2.5,   # t4..t7
    1.0, 1.0, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5    # t12..t15
], dtype=float)


weights = weights / weights.mean()


vol_rep = (D * weights).mean(axis=1) * SCALE_K


is_all_zero = (D <= EPS).all(axis=1)

vol_rep[is_all_zero] = SEED_VALUE




demand = od.copy()
demand["volume"] = np.nan_to_num(vol_rep, nan=0.0, posinf=0.0, neginf=0.0).astype(float)
demand.to_csv("demand.csv", index=False)


network = pg.read_network()
pg.load_demand(network)


column_gen_num    = 40
column_update_num = 800
pg.perform_column_generation(column_gen_num, column_update_num, network)


pg.output_columns(network)               # -> agent.csv
pg.output_link_performance(network)      # -> link_performance.csv


if os.path.exists("agent.csv"):
    dst = "agent_new.csv"
    if os.path.exists(dst):
        os.remove(dst)
    os.rename("agent.csv", dst)

if os.path.exists("link_performance.csv"):
    if os.path.exists("link_performance_rep.csv"):
        os.remove("link_performance_rep.csv")
    os.rename("link_performance.csv", "link_performance_rep.csv")



od_pair = pd.read_csv("OD_pair.csv")[["o_zone_id", "d_zone_id"]]



agent_df = pd.read_csv("agent_new.csv")


ods_has = agent_df[["o_zone_id", "d_zone_id"]].drop_duplicates()


missing = pd.merge(
    od_pair, ods_has,
    on=["o_zone_id", "d_zone_id"],
    how="left",
    indicator=True
)
missing = missing[missing["_merge"] == "left_only"][["o_zone_id", "d_zone_id"]]



if len(missing) > 0:

    node_df = pd.read_csv("node.csv")[["node_id", "zone_id"]]
    node_df = node_df.dropna(subset=["zone_id"])
    zone2node = (
        node_df.drop_duplicates(subset=["zone_id"])
               .set_index("zone_id")["node_id"]
               .astype(int)
               .to_dict()
    )

    new_rows = []


    for _, row in missing.iterrows():
        oz = int(row["o_zone_id"])
        dz = int(row["d_zone_id"])



        on = int(zone2node[oz])
        dn = int(zone2node[dz])

        try:

            link_seq = network.find_shortest_path(on, dn, seq_type="link")
        except Exception as e:
            print(f"  [警告] OD ({oz}, {dz}) 最短路计算失败：{e}")
            continue

        if not isinstance(link_seq, str) or len(link_seq.strip()) == 0:
            print(f"  [警告] OD ({oz}, {dz}) 未找到有效路径，跳过。")
            continue

        new_rows.append({
            "o_zone_id": oz,
            "d_zone_id": dz,
            "path_id": 0,
            "link_sequence": link_seq
        })

    if new_rows:
        new_rows_df = pd.DataFrame(new_rows)


        for col in agent_df.columns:
            if col not in new_rows_df.columns:
                new_rows_df[col] = np.nan

        agent_df = pd.concat(
            [agent_df, new_rows_df[agent_df.columns]],
            ignore_index=True
        )

        agent_df.to_csv("agent_new.csv", index=False)
        print(f"已为 {len(new_rows)} 个缺失 OD 补充最短路径。")
    else:
        print("虽然检测到缺失 OD，但未成功补充任何路径，请检查日志。")
else:
    print("没有缺失 OD，agent_new.csv 已覆盖全部 OD。")


